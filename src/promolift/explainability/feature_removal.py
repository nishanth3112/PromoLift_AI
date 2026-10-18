"""What does a feature add? Refit the model without it and compare, out of fold.

SHAP says how much the champion *uses* a feature, not how much worse it would
be without it: correlated features pick up the slack. This answers the second
question directly. The same model and hyperparameters are fitted twice on the
same stratified folds -- with every feature, and without the dropped ones --
and each client is scored only by a fold model that never saw it. The two
rankings are then compared on the same bootstrap resamples, so client-level
noise shared by both cancels out of the difference.

For an age check it also reports who gets texted: the age mix of the top
``depth`` of clients under each ranking against the whole population, and how
strongly each ranking still follows age. Removing ``age`` does not remove age
information -- tenure, spend, and points correlate with it -- so the second
number says how much is left.

Everything stays inside train + val; the test split is never loaded. The
hyperparameters were tuned with every feature present, which if anything
favours the "with" arm.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import polars as pl

from promolift.evaluation.ranking_metrics import ranking_metrics, uplift_at_k_name
from promolift.evaluation.uncertainty import (
    DEFAULT_CONFIDENCE,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_SEED,
    PairedComparison,
    paired_comparison,
)
from promolift.explainability.segments import AGE_LABELS, AGE_NOT_VALID, age_band
from promolift.models.ablation import out_of_fold_scores
from promolift.models.dataset import ModelFrame
from promolift.models.training import ModelSpec
from promolift.models.tuning import DEFAULT_N_FOLDS, cv_folds
from promolift.tracking.mlflow_tracking import Experiment, TrackingConfig, start_run

DEFAULT_DROPPED = ("age", "age_is_valid")
WITH_ARM = "with"
WITHOUT_ARM = "without"
_AGE_BANDS = (*AGE_LABELS, AGE_NOT_VALID)


@dataclass(frozen=True)
class ArmSummary:
    """One ranking's point estimates on the pooled out-of-fold scores."""

    qini_auc: float
    uplift_at_depth: float
    age_correlation: float | None


@dataclass
class FeatureRemoval:
    """The model with and without ``dropped``, compared on the same clients."""

    model_label: str
    dropped: tuple[str, ...]
    depth: float
    n_clients: int
    n_folds: int
    n_bootstrap: int
    seed: int
    with_features: ArmSummary
    without_features: ArmSummary
    comparison: PairedComparison
    age_mix: pl.DataFrame | None

    @property
    def verdict(self) -> str:
        """What the paired Qini difference (with minus without) says about the dropped features."""
        qini = self.comparison.qini_auc
        if qini.lower > 0:
            return "removing them lowers ranking quality"
        if qini.upper < 0:
            return "removing them improves ranking quality"
        return "no measurable difference in ranking quality"


def without_columns(frame: ModelFrame, columns: Sequence[str]) -> ModelFrame:
    """``frame`` minus feature ``columns`` (labels and clients untouched).

    Raises:
        ValueError: If a column isn't a feature of the frame.
    """
    missing = [column for column in columns if column not in frame.features.columns]
    if missing:
        raise ValueError(f"Cannot drop {missing}: not features of the frame")
    return ModelFrame(
        split=frame.split,
        client_ids=frame.client_ids,
        features=frame.features.drop(columns=list(columns)),
        treatment=frame.treatment,
        outcome=frame.outcome,
    )


def targeted_age_mix(
    scores: np.ndarray, age: pd.Series, is_valid: pd.Series | None, depth: float
) -> pl.DataFrame:
    """Age mix of the top ``depth`` of clients by score, against everyone.

    One row per age band present: ``population_share``, ``targeted_share``, and
    their ratio (above 1 = texted more often than the band's weight in the
    population). Ties in score are broken by position, so the result is
    deterministic.
    """
    bands = age_band(
        age.reset_index(drop=True), None if is_valid is None else is_valid.reset_index(drop=True)
    )
    n_targeted = round(depth * len(scores))
    top = np.argsort(-np.asarray(scores), kind="stable")[:n_targeted]
    targeted = bands.iloc[top]
    rows = []
    for band in _AGE_BANDS:
        population_share = float((bands == band).mean())
        if population_share == 0:
            continue
        targeted_share = float((targeted == band).mean())
        rows.append(
            {
                "age_band": band,
                "population_share": population_share,
                "targeted_share": targeted_share,
                "ratio": targeted_share / population_share,
            }
        )
    return pl.DataFrame(rows)


def _age_correlation(
    scores: np.ndarray, age: pd.Series, is_valid: pd.Series | None
) -> float | None:
    trusted = age.notna() if is_valid is None else (is_valid == 1) & age.notna()
    pair = pd.DataFrame({"score": scores, "age": age.to_numpy()})[trusted.to_numpy()]
    value = pair["score"].corr(pair["age"], method="spearman")
    return None if pd.isna(value) else float(value)


def run_feature_removal(
    frame: ModelFrame,
    spec: ModelSpec,
    *,
    depth: float,
    dropped: Sequence[str] = DEFAULT_DROPPED,
    n_folds: int = DEFAULT_N_FOLDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    progress: Callable[[str], None] = lambda _message: None,
) -> FeatureRemoval:
    """Compare ``spec`` with and without the ``dropped`` features, by cross-validation.

    Args:
        frame: Train + val clients (never test).
        spec: The model and its (tuned) hyperparameters, held fixed in both arms.
        depth: The targeting depth for the business view (uplift at that share).

    Raises:
        ValueError: If ``depth`` is outside (0, 1) or a dropped column isn't a feature.
    """
    if not 0 < depth < 1:
        raise ValueError("depth must be strictly between 0 and 1")
    reduced = without_columns(frame, dropped)
    folds = cv_folds(frame.treatment, frame.outcome, n_folds=n_folds, seed=seed)
    progress(f"Scoring {len(frame.outcome):,} clients out of fold, all features...")
    full = out_of_fold_scores(
        spec.model_name, frame, folds, config=WITH_ARM, overrides=spec.overrides, seed=seed
    )
    progress(f"Scoring again without {', '.join(dropped)}...")
    cut = out_of_fold_scores(
        spec.model_name, reduced, folds, config=WITHOUT_ARM, overrides=spec.overrides, seed=seed
    )
    progress(f"Comparing the rankings ({n_bootstrap} paired resamples)...")
    comparison = paired_comparison(
        full.oof_scores,
        cut.oof_scores,
        frame.treatment,
        frame.outcome,
        k_grid=(depth,),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )

    has_age = "age" in frame.features.columns
    age = frame.features["age"] if has_age else None
    valid = frame.features["age_is_valid"] if "age_is_valid" in frame.features.columns else None

    def summarize(scores: np.ndarray) -> ArmSummary:
        metrics = ranking_metrics(scores, frame.treatment, frame.outcome, k_grid=(depth,))
        return ArmSummary(
            qini_auc=metrics.qini_auc,
            uplift_at_depth=metrics.uplift_at_k[depth],
            age_correlation=_age_correlation(scores, age, valid) if has_age else None,
        )

    mix = None
    if has_age:
        with_mix = targeted_age_mix(full.oof_scores, age, valid, depth)
        without_mix = targeted_age_mix(cut.oof_scores, age, valid, depth)
        mix = with_mix.join(
            without_mix.select(
                "age_band",
                pl.col("targeted_share").alias("targeted_share_without"),
                pl.col("ratio").alias("ratio_without"),
            ),
            on="age_band",
        ).rename({"targeted_share": "targeted_share_with", "ratio": "ratio_with"})
    return FeatureRemoval(
        model_label=spec.label,
        dropped=tuple(dropped),
        depth=depth,
        n_clients=len(frame.outcome),
        n_folds=n_folds,
        n_bootstrap=n_bootstrap,
        seed=seed,
        with_features=summarize(full.oof_scores),
        without_features=summarize(cut.oof_scores),
        comparison=comparison,
        age_mix=mix,
    )


def _pp(value: float) -> str:
    return f"{value * 100:+.2f}pp"


def summary(result: FeatureRemoval) -> str:
    """The comparison in a screenful: ranking quality, business view, and age mix."""
    qini, at_depth = result.comparison.qini_auc, result.comparison.uplift_at_k[result.depth]
    level = round(result.comparison.confidence * 100)
    w, wo = result.with_features, result.without_features
    lines = [
        f"{result.model_label} with and without {', '.join(result.dropped)} "
        f"({result.n_clients:,} train+val clients, {result.n_folds}-fold out of fold)",
        "",
        f"  Qini AUC: {w.qini_auc:+.4f} with, {wo.qini_auc:+.4f} without; with minus without "
        f"{qini.difference:+.4f} ({level}% interval {qini.lower:+.4f} to {qini.upper:+.4f})",
        f"  uplift in the top {result.depth:.0%}: {_pp(w.uplift_at_depth)} with, "
        f"{_pp(wo.uplift_at_depth)} without; difference {_pp(at_depth.difference)} "
        f"({_pp(at_depth.lower)} to {_pp(at_depth.upper)})",
        f"  verdict on ranking quality: {result.verdict}",
    ]
    if result.with_features.age_correlation is not None:
        lines += [
            "",
            f"  rank correlation of the score with age: {w.age_correlation:+.2f} with, "
            f"{wo.age_correlation:+.2f} without",
            f"  age mix of the top {result.depth:.0%} (share of texted / share of population):",
        ]
        for row in result.age_mix.iter_rows(named=True):
            lines.append(
                f"    {row['age_band']:14s} population {row['population_share']:5.1%}   "
                f"with {row['targeted_share_with']:5.1%} (x{row['ratio_with']:.2f})   "
                f"without {row['targeted_share_without']:5.1%} (x{row['ratio_without']:.2f})"
            )
    return "\n".join(lines)


def log_feature_removal(
    result: FeatureRemoval,
    *,
    config: TrackingConfig | None = None,
    lineage: dict[str, Path] | None = None,
) -> str:
    """Record the comparison as a run in ``promolift-feature-ablation``; returns its run id."""
    qini, at_depth = result.comparison.qini_auc, result.comparison.uplift_at_k[result.depth]
    name = uplift_at_k_name(result.depth)
    with start_run(
        Experiment.FEATURE_ABLATION,
        run_name=f"remove-{'-'.join(result.dropped)}",
        tags={
            "run_type": "feature_removal",
            "model": result.model_label,
            "dropped": ",".join(result.dropped),
            "verdict": result.verdict,
        },
        config=config,
        **(lineage or {}),
    ) as run:
        mlflow.log_params(
            {
                "n_folds": result.n_folds,
                "n_bootstrap": result.n_bootstrap,
                "seed": result.seed,
                "depth": result.depth,
                "n_clients": result.n_clients,
            }
        )
        metrics = {
            "qini_with": result.with_features.qini_auc,
            "qini_without": result.without_features.qini_auc,
            "qini_difference": qini.difference,
            "qini_difference_lower": qini.lower,
            "qini_difference_upper": qini.upper,
            "qini_difference_win_rate": qini.win_rate,
            f"{name}_with": result.with_features.uplift_at_depth,
            f"{name}_without": result.without_features.uplift_at_depth,
            f"{name}_difference": at_depth.difference,
            f"{name}_difference_lower": at_depth.lower,
            f"{name}_difference_upper": at_depth.upper,
        }
        for arm, summary_ in (("with", result.with_features), ("without", result.without_features)):
            if summary_.age_correlation is not None:
                metrics[f"age_correlation_{arm}"] = summary_.age_correlation
        mlflow.log_metrics(metrics)
        mlflow.log_text(summary(result), "feature_removal/summary.txt")
        mlflow.log_dict(
            json.loads(json.dumps(asdict(result.comparison))), "feature_removal/comparison.json"
        )
        if result.age_mix is not None:
            mlflow.log_text(result.age_mix.write_csv(), "feature_removal/age_mix.csv")
        return run.info.run_id
