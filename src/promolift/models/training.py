"""Train, evaluate, and log models on the canonical split, then compare them on a leaderboard.

Every model is fitted on train and judged on val by ``evaluate_ranking``
against one shared noise floor. The leaderboard adds what single-model
reports can't: paired comparisons, which decide whether one model *actually*
beats another or they are statistically tied.
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
from promolift.models.dataset import ModelFrame
from promolift.models.registry import build_model
from promolift.tracking.mlflow_tracking import Experiment, TrackingConfig, start_run

# Not uplift models: the floor to clear and the incumbent to beat.
_BASELINES = ("random", "response")
_INCUMBENT = "response"


@dataclass
class TrainedModel:
    """One model's validation scores and evaluation, and the run they're logged in."""

    name: str
    run_id: str
    fit_seconds: float
    scores: np.ndarray
    val: ModelFrame
    report: EvaluationReport


def train_and_evaluate(
    model_names: Sequence[str],
    train: ModelFrame,
    val: ModelFrame,
    *,
    seed: int = DEFAULT_SEED,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    n_rankings: int = DEFAULT_N_RANKINGS,
    config: TrackingConfig | None = None,
    lineage: dict[str, Path] | None = None,
) -> list[TrainedModel]:
    """Fit each model on ``train``, evaluate on ``val``, one MLflow run per model.

    Args:
        model_names: Registered model names, in the order to train them.
        train, val: Model frames from ``model_frame``.
        config: Tracking destination; defaults to ``default_tracking_config()``.
        lineage: ``repo_dir`` / ``data_dir`` / ``processed_dir`` overrides (for tests).
    """
    noise_floor = random_ranking_noise_floor(
        val.treatment, val.outcome, n_rankings=n_rankings, seed=seed
    )
    results = []
    for name in model_names:
        model = build_model(name, seed=seed)
        with start_run(
            Experiment.UPLIFT_MODELS,
            run_name=name,
            tags={"run_type": "model", "model_name": name},
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
        results.append(TrainedModel(name, run.info.run_id, fit_seconds, scores, val, report))
    return results


def _comparison_pairs(results: Sequence[TrainedModel], best: str) -> list[tuple[str, str]]:
    names = [r.name for r in results]
    pairs = [(best, other) for other in names if other != best]
    if _INCUMBENT in names:
        pairs += [(name, _INCUMBENT) for name in names if name not in _BASELINES and name != best]
    return pairs


def _leaderboard_row(result: TrainedModel) -> dict:
    report = result.report
    row: dict = {"model": result.name}
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


def log_leaderboard(
    results: Sequence[TrainedModel],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    config: TrackingConfig | None = None,
    lineage: dict[str, Path] | None = None,
) -> str:
    """Log a leaderboard run comparing already-evaluated models; return its run id.

    Models are ranked by val Qini AUC. Paired comparisons cover the best model
    against every other, and every uplift model against the response model
    (the incumbent), each on identical bootstrap resamples.
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

    with start_run(
        Experiment.UPLIFT_MODELS,
        run_name="leaderboard",
        tags={
            "run_type": "leaderboard",
            "best_model": best.name,
            "models": ",".join(r.name for r in ranked),
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
        mlflow.log_figure(
            qini_curve_figure([r.report for r in ranked]), "leaderboard/qini_curves.png"
        )
        return run.info.run_id
