"""One evaluation report per model and split, and its MLflow logging.

``evaluate_ranking`` is the single sanctioned way to judge a model: it
enforces the test-split lock, and bundles point estimates, bootstrap CIs, the
noise-floor verdict, deciles, and the Qini curve so no result is reported
without its uncertainty.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import mlflow
import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from promolift.data.split import Split, assert_split_access
from promolift.evaluation.plots import decile_uplift_figure, qini_curve_figure
from promolift.evaluation.ranking_metrics import (
    DEFAULT_K_GRID,
    QiniCurve,
    RankingMetrics,
    qini_curve_points,
    ranking_metrics,
    uplift_at_k_name,
    uplift_by_decile,
)
from promolift.evaluation.uncertainty import (
    DEFAULT_CONFIDENCE,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_SEED,
    BootstrapMetrics,
    Interval,
    NoiseFloor,
    beats_noise_floor,
    bootstrap_metrics,
    random_ranking_noise_floor,
)


@dataclass
class EvaluationReport:
    """Everything known about one model's ranking quality on one split."""

    model_name: str
    split: str
    final_evaluation: bool
    n_clients: int
    ate: float
    metrics: RankingMetrics
    bootstrap: BootstrapMetrics
    noise_floor: NoiseFloor
    beats_noise_floor: dict[str, bool]
    deciles: pl.DataFrame
    qini_curve: QiniCurve

    def summary_dict(self) -> dict:
        """JSON-serializable summary; deciles and the curve are logged separately."""
        return {
            "model_name": self.model_name,
            "split": self.split,
            "final_evaluation": self.final_evaluation,
            "n_clients": self.n_clients,
            "ate": self.ate,
            "metrics": self.metrics.as_flat_dict(),
            "bootstrap": {
                "n_bootstrap": self.bootstrap.n_bootstrap,
                "confidence": self.bootstrap.confidence,
                "intervals": {k: asdict(v) for k, v in _flat(self.bootstrap).items()},
            },
            "noise_floor": {
                "n_rankings": self.noise_floor.n_rankings,
                "confidence": self.noise_floor.confidence,
                "intervals": {k: asdict(v) for k, v in _flat(self.noise_floor).items()},
            },
            "beats_noise_floor": self.beats_noise_floor,
        }


def _flat(result: BootstrapMetrics | NoiseFloor) -> dict[str, Interval]:
    return {
        "qini_auc": result.qini_auc,
        "uplift_auc": result.uplift_auc,
        **{uplift_at_k_name(k): v for k, v in result.uplift_at_k.items()},
    }


def evaluate_ranking(
    model_name: str,
    scores: ArrayLike,
    treatment: ArrayLike,
    outcome: ArrayLike,
    *,
    split: Split | str,
    final_evaluation: bool = False,
    noise_floor: NoiseFloor | None = None,
    k_grid: Sequence[float] = DEFAULT_K_GRID,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> EvaluationReport:
    """Evaluate one model's scores on one split.

    Args:
        model_name: Label used in plots and the report.
        scores, treatment, outcome: One entry per client of ``split``.
        split: Which split the clients come from; ``test`` requires
            ``final_evaluation=True``.
        noise_floor: A floor already computed for these exact clients. The
            floor depends only on the split, so compute it once with
            ``random_ranking_noise_floor`` and pass it to every model.
    """
    assert_split_access(split, final_evaluation=final_evaluation)
    s, t, y = np.asarray(scores, dtype=float), np.asarray(treatment), np.asarray(outcome)
    metrics = ranking_metrics(s, t, y, k_grid=k_grid)

    if noise_floor is None:
        noise_floor = random_ranking_noise_floor(
            t, y, k_grid=k_grid, confidence=confidence, seed=seed
        )
    elif noise_floor.n_clients != len(s) or set(noise_floor.uplift_at_k) != set(k_grid):
        msg = (
            f"The noise floor was computed for {noise_floor.n_clients} clients and "
            f"k_grid {sorted(noise_floor.uplift_at_k)}, but these scores have {len(s)} "
            f"clients and k_grid {sorted(k_grid)}"
        )
        raise ValueError(msg)

    return EvaluationReport(
        model_name=model_name,
        split=Split(split).value,
        final_evaluation=final_evaluation,
        n_clients=len(s),
        ate=float(y[t == 1].mean() - y[t == 0].mean()),
        metrics=metrics,
        bootstrap=bootstrap_metrics(
            s, t, y, k_grid=k_grid, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed
        ),
        noise_floor=noise_floor,
        beats_noise_floor=beats_noise_floor(metrics, noise_floor),
        deciles=uplift_by_decile(s, t, y),
        qini_curve=qini_curve_points(s, t, y),
    )


def log_evaluation(report: EvaluationReport) -> None:
    """Log a report to the active MLflow run, with every name prefixed by its split.

    Split-prefixed names (``val_qini_auc``) let one run carry train and val
    evaluations side by side, and make a test-split number impossible to
    mistake for a validation one.

    Raises:
        RuntimeError: If no run is active.
    """
    if mlflow.active_run() is None:
        msg = "log_evaluation must be called inside tracking.mlflow_tracking.start_run(...)"
        raise RuntimeError(msg)

    prefix = f"{report.split}_"
    metrics = {f"{prefix}{name}": value for name, value in report.metrics.as_flat_dict().items()}
    for name, interval in _flat(report.bootstrap).items():
        metrics[f"{prefix}{name}_ci_lower"] = interval.lower
        metrics[f"{prefix}{name}_ci_upper"] = interval.upper
    for name, band in _flat(report.noise_floor).items():
        metrics[f"{prefix}noise_floor_{name}_lower"] = band.lower
        metrics[f"{prefix}noise_floor_{name}_upper"] = band.upper
    metrics[f"{prefix}ate"] = report.ate
    metrics[f"{prefix}n_clients"] = report.n_clients
    mlflow.log_metrics(metrics)

    mlflow.log_params(
        {
            f"{prefix}n_bootstrap": report.bootstrap.n_bootstrap,
            f"{prefix}confidence": report.bootstrap.confidence,
            f"{prefix}noise_floor_n_rankings": report.noise_floor.n_rankings,
        }
    )
    mlflow.set_tags(
        {
            "model_name": report.model_name,
            "evaluated_split": report.split,
            "final_evaluation": str(report.final_evaluation).lower(),
            **{
                f"{prefix}beats_noise_floor_{name}": str(beats).lower()
                for name, beats in report.beats_noise_floor.items()
            },
        }
    )

    artifact_dir = f"evaluation/{report.split}"
    mlflow.log_dict(report.summary_dict(), f"{artifact_dir}/report.json")
    mlflow.log_text(report.deciles.write_csv(), f"{artifact_dir}/deciles.csv")
    mlflow.log_figure(qini_curve_figure([report]), f"{artifact_dir}/qini_curve.png")
    mlflow.log_figure(decile_uplift_figure(report), f"{artifact_dir}/decile_uplift.png")
