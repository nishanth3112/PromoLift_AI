"""Train, evaluate, and log models on the canonical split, then compare them on a leaderboard.

Every model variant -- a registered model with default or tuned
hyperparameters -- is fitted on train and judged on val by
``evaluate_ranking`` against one shared noise floor. The leaderboard adds
what single-model reports can't: paired comparisons, which decide whether one
model *actually* beats another, and whether tuning actually helped.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

from promolift.evaluation.plots import qini_curve_figure
from promolift.evaluation.ranking_metrics import uplift_at_k_name
from promolift.evaluation.report import EvaluationReport, evaluate_ranking, log_evaluation
from promolift.evaluation.uncertainty import (
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_N_RANKINGS,
    DEFAULT_SEED,
    PairedComparison,
    paired_comparison,
    random_ranking_noise_floor,
)
from promolift.models.base import Params
from promolift.models.dataset import ModelFrame
from promolift.models.registry import build_model
from promolift.tracking.mlflow_tracking import Experiment, TrackingConfig, start_run

# Reference rankings every Qini plot keeps: the floor and the incumbent.
_REFERENCES = ("random", "response")
_MAX_PLOTTED = 8
_TUNED_SUFFIX = "_tuned"


@dataclass(frozen=True)
class ModelSpec:
    """One leaderboard entry: a registered model, optionally with overridden hyperparameters."""

    label: str
    model_name: str
    overrides: Params | None = None
    variant: str = "default"


def tuned_specs(tuned_params: dict[str, Params]) -> list[ModelSpec]:
    """``<model>_tuned`` variants from ``tuning.load_tuned_params()``, in file order."""
    return [
        ModelSpec(f"{name}{_TUNED_SUFFIX}", name, params, variant="tuned")
        for name, params in tuned_params.items()
    ]


def _as_spec(spec: str | ModelSpec) -> ModelSpec:
    return spec if isinstance(spec, ModelSpec) else ModelSpec(spec, spec)


@dataclass
class TrainedModel:
    """One variant's validation scores and evaluation, and the run they're logged in."""

    name: str
    model_name: str
    variant: str
    run_id: str
    fit_seconds: float
    scores: np.ndarray
    val: ModelFrame
    report: EvaluationReport


def train_and_evaluate(
    specs: Sequence[str | ModelSpec],
    train: ModelFrame,
    val: ModelFrame,
    *,
    seed: int = DEFAULT_SEED,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    n_rankings: int = DEFAULT_N_RANKINGS,
    config: TrackingConfig | None = None,
    lineage: dict[str, Path] | None = None,
) -> list[TrainedModel]:
    """Fit each variant on ``train``, evaluate on ``val``, one MLflow run per variant.

    Args:
        specs: Variants to train, in order; a bare name means that model's defaults.
        train, val: Model frames from ``model_frame``.
        config: Tracking destination; defaults to ``default_tracking_config()``.
        lineage: ``repo_dir`` / ``data_dir`` / ``processed_dir`` overrides (for tests).
    """
    noise_floor = random_ranking_noise_floor(
        val.treatment, val.outcome, n_rankings=n_rankings, seed=seed
    )
    results = []
    for spec in map(_as_spec, specs):
        name = spec.label
        model = build_model(spec.model_name, seed=seed, overrides=spec.overrides)
        with start_run(
            Experiment.UPLIFT_MODELS,
            run_name=name,
            tags={
                "run_type": "model",
                "model_name": name,
                "base_model": spec.model_name,
                "variant": spec.variant,
            },
            config=config,
            **(lineage or {}),
        ) as run:
            mlflow.log_params(model.params())
            started = time.perf_counter()
            model.fit(train.features, train.treatment, train.outcome)
            fit_seconds = time.perf_counter() - started
            scores = model.predict_uplift(val.features)
            report = evaluate_ranking(
                name,
                scores,
                val.treatment,
                val.outcome,
                split=val.split,
                noise_floor=noise_floor,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            mlflow.log_metrics({"fit_seconds": fit_seconds, "n_train_clients": len(train.features)})
            log_evaluation(report)
        results.append(
            TrainedModel(
                name,
                spec.model_name,
                spec.variant,
                run.info.run_id,
                fit_seconds,
                scores,
                val,
                report,
            )
        )
    return results


def _tuning_pairs(results: Sequence[TrainedModel]) -> list[tuple[str, str]]:
    """(tuned, default) label pairs for every model trained in both variants."""
    defaults = {r.model_name: r.name for r in results if r.variant == "default"}
    return [
        (r.name, defaults[r.model_name])
        for r in results
        if r.variant == "tuned" and r.model_name in defaults
    ]


def _comparison_pairs(results: Sequence[TrainedModel], best: str) -> list[tuple[str, str]]:
    """Tuned vs default per model (oriented tuned - default), then best vs every other.

    Each unordered pair is compared once.
    """
    pairs = _tuning_pairs(results)
    seen = {frozenset(pair) for pair in pairs}
    for other in (r.name for r in results if r.name != best):
        if frozenset((best, other)) not in seen:
            pairs.append((best, other))
            seen.add(frozenset((best, other)))
    return pairs


def _plotted(ranked: Sequence[TrainedModel]) -> list[TrainedModel]:
    """Top-ranked variants plus the random and response references, at most 8, in rank order."""
    references = [r for r in ranked if r.name in _REFERENCES]
    others = [r for r in ranked if r.name not in _REFERENCES][: _MAX_PLOTTED - len(references)]
    chosen = {r.name for r in (*references, *others)}
    return [r for r in ranked if r.name in chosen]


def _leaderboard_row(result: TrainedModel) -> dict:
    report = result.report
    row: dict = {"model": result.name, "base_model": result.model_name, "variant": result.variant}
    intervals = {
        "qini_auc": report.bootstrap.qini_auc,
        "uplift_auc": report.bootstrap.uplift_auc,
        **{uplift_at_k_name(k): v for k, v in report.bootstrap.uplift_at_k.items()},
    }
    floor_lower = {
        "qini_auc": report.noise_floor.qini_auc.lower,
        "uplift_auc": report.noise_floor.uplift_auc.lower,
        **{uplift_at_k_name(k): v.lower for k, v in report.noise_floor.uplift_at_k.items()},
    }
    for name, interval in intervals.items():
        row[name] = interval.estimate
        row[f"{name}_ci_lower"] = interval.lower
        row[f"{name}_ci_upper"] = interval.upper
        row[f"beats_noise_floor_{name}"] = report.beats_noise_floor[name]
        # Below the band a random ranking stays inside 95% of the time: worse than random.
        row[f"below_noise_floor_{name}"] = interval.estimate < floor_lower[name]
    row["fit_seconds"] = result.fit_seconds
    row["run_id"] = result.run_id
    return row


def _comparison_record(a: str, b: str, comparison: PairedComparison) -> dict:
    return {
        "model_a": a,
        "model_b": b,
        "n_bootstrap": comparison.n_bootstrap,
        "confidence": comparison.confidence,
        "qini_auc": asdict(comparison.qini_auc),
        "uplift_auc": asdict(comparison.uplift_auc),
        **{uplift_at_k_name(k): asdict(v) for k, v in comparison.uplift_at_k.items()},
    }


def tuning_effect_table(
    records: Sequence[dict], by_name: dict[str, TrainedModel]
) -> pl.DataFrame | None:
    """Tuned minus default Qini per model, from the paired-comparison records.

    Only same-model (tuned, default) comparisons count: the best variant is
    also compared with other models' defaults, and those aren't tuning effects.
    """

    def is_tuning_pair(record: dict) -> bool:
        a, b = by_name[record["model_a"]], by_name[record["model_b"]]
        return a.variant == "tuned" and b.variant == "default" and a.model_name == b.model_name

    rows = [
        {
            "base_model": by_name[r["model_a"]].model_name,
            "default_qini_auc": by_name[r["model_b"]].report.metrics.qini_auc,
            "tuned_qini_auc": by_name[r["model_a"]].report.metrics.qini_auc,
            "qini_diff": r["qini_auc"]["difference"],
            "qini_diff_ci_lower": r["qini_auc"]["lower"],
            "qini_diff_ci_upper": r["qini_auc"]["upper"],
            "win_rate": r["qini_auc"]["win_rate"],
        }
        for r in records
        if is_tuning_pair(r)
    ]
    return pl.DataFrame(rows) if rows else None


def log_leaderboard(
    results: Sequence[TrainedModel],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    config: TrackingConfig | None = None,
    lineage: dict[str, Path] | None = None,
) -> str:
    """Log a leaderboard run comparing already-evaluated variants; return its run id.

    Variants are ranked by val Qini AUC. Paired comparisons (identical
    bootstrap resamples) cover tuned vs default for every model trained both
    ways -- summarized in ``tuning_effect.csv`` -- and the best variant against
    every other.
    """
    ranked = sorted(results, key=lambda r: r.report.metrics.qini_auc, reverse=True)
    best = ranked[0]
    val = best.val
    by_name = {r.name: r for r in results}

    records = []
    comparison_metrics = {}
    for a, b in _comparison_pairs(ranked, best.name):
        comparison = paired_comparison(
            by_name[a].scores,
            by_name[b].scores,
            val.treatment,
            val.outcome,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        records.append(_comparison_record(a, b, comparison))
        comparison_metrics[f"qini_diff_{a}_vs_{b}"] = comparison.qini_auc.difference
        comparison_metrics[f"qini_win_rate_{a}_vs_{b}"] = comparison.qini_auc.win_rate

    plotted = _plotted(ranked)
    with start_run(
        Experiment.UPLIFT_MODELS,
        run_name="leaderboard",
        tags={
            "run_type": "leaderboard",
            "best_model": best.name,
            "models": ",".join(r.name for r in ranked),
            "plotted_models": ",".join(r.name for r in plotted),
            "evaluated_split": val.split,
        },
        config=config,
        **(lineage or {}),
    ) as run:
        mlflow.log_params({"n_bootstrap": n_bootstrap, "seed": seed})
        mlflow.log_metrics(
            {f"best_{val.split}_qini_auc": best.report.metrics.qini_auc, **comparison_metrics}
        )
        table = pl.DataFrame([_leaderboard_row(r) for r in ranked])
        mlflow.log_text(table.write_csv(), "leaderboard/leaderboard.csv")
        mlflow.log_dict(records, "leaderboard/paired_comparisons.json")
        effect = tuning_effect_table(records, by_name)
        if effect is not None:
            mlflow.log_text(effect.write_csv(), "leaderboard/tuning_effect.csv")
        mlflow.log_figure(
            qini_curve_figure([r.report for r in plotted]), "leaderboard/qini_curves.png"
        )
        return run.info.run_id
