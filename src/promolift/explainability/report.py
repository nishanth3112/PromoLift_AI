"""Assemble and record an explanation: tables, figures, a plain-text summary, one MLflow run.

The run holds the importance, segment, and dependence tables (CSV), the two
figures, and the summary -- not per-client contributions, which would be a
client-level file. Metrics carry each feature's importance, so versions of
the model can be compared in the MLflow UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import polars as pl

from promolift.evaluation.plots import dependence_figure, importance_figure
from promolift.explainability.importance import dependence_table, global_importance
from promolift.explainability.segments import default_segments, segment_importance, top_features
from promolift.explainability.shap_values import Explanation
from promolift.tracking.mlflow_tracking import (
    Experiment,
    TrackingConfig,
    default_tracking_config,
    start_run,
)

DEPENDENCE_FEATURES = 6
TOP_FEATURES_SHOWN = 8
SEGMENT_TOP_FEATURES = 3
FIT_POPULATION = "train+val"
CAVEAT = (
    "SHAP describes what the model does, not what the features cause: correlated features "
    "share credit, and a feature the model leans on may not move buying."
)


@dataclass
class ExplanationReport:
    """The tables an explanation is reported from."""

    importance: pl.DataFrame
    segments: pl.DataFrame
    segment_top: pl.DataFrame
    dependence: dict[str, pl.DataFrame]


def build_report(
    explanation: Explanation, *, dependence_features: int = DEPENDENCE_FEATURES
) -> ExplanationReport:
    """Importance, per-segment importance, and dependence for the most important features."""
    importance = global_importance(explanation)
    segments = segment_importance(explanation, default_segments(explanation))
    leading = importance["feature"].head(dependence_features).to_list()
    return ExplanationReport(
        importance=importance,
        segments=segments,
        segment_top=top_features(segments, SEGMENT_TOP_FEATURES),
        dependence={feature: dependence_table(explanation, feature) for feature in leading},
    )


def _pp(value: float) -> str:
    return f"{value * 100:+.2f}pp"


def summary(
    explanation: Explanation, report: ExplanationReport, *, model_name: str, model_version: str
) -> str:
    """The explanation in a screenful: leading features with direction, tiers, and the caveat."""
    lines = [
        f"SHAP explanation of {model_name} version {model_version} on "
        f"{len(explanation.scores):,} {FIT_POPULATION} clients",
        f"  an average client's predicted uplift is {_pp(explanation.base_value)}; each client's "
        f"contributions add up to their score (largest gap {explanation.additivity_error:.1e})",
        "",
        "  Most important features (mean absolute contribution, share of the total):",
    ]
    for row in report.importance.head(TOP_FEATURES_SHOWN).iter_rows(named=True):
        size = row["mean_abs_contribution"] * 100
        lines.append(
            f"    {row['rank']:2d}. {row['feature']:24s} {size:5.2f}pp "
            f"({row['share']:4.0%})  {row['direction']}"
        )
    lines += ["", "  Leading features by predicted-uplift tier and other segments:"]
    for row in report.segment_top.iter_rows(named=True):
        lines.append(
            f"    {row['segment_type']}={row['segment']} (n={row['n']:,}): {row['top_features']}"
        )
    return "\n".join([*lines, "", f"  {CAVEAT}"])


def log_explanation(
    explanation: Explanation,
    report: ExplanationReport,
    *,
    model_name: str,
    model_version: str,
    seed: int,
    config: TrackingConfig | None = None,
    lineage: dict | None = None,
) -> str:
    """Record the explanation as a run in ``promolift-explainability``; returns its run id."""
    config = config if config is not None else default_tracking_config()
    leader = report.importance.row(0, named=True)
    with start_run(
        Experiment.EXPLAINABILITY,
        run_name="shap_explanation",
        tags={
            "run_type": "shap_explanation",
            "model_name": model_name,
            "model_version": model_version,
            "fit_population": FIT_POPULATION,
            "top_feature": leader["feature"],
        },
        config=config,
        **(lineage or {}),
    ) as run:
        mlflow.log_params(
            {
                "model_name": model_name,
                "model_version": model_version,
                "n_clients": len(explanation.scores),
                "seed": seed,
            }
        )
        mlflow.log_metrics(
            {
                "base_uplift": explanation.base_value,
                "mean_predicted_uplift": float(explanation.scores.mean()),
                "additivity_error": explanation.additivity_error,
                "top_feature_share": leader["share"],
            }
            | {
                f"importance_{row['feature']}": row["mean_abs_contribution"]
                for row in report.importance.iter_rows(named=True)
            }
        )
        mlflow.log_text(
            summary(explanation, report, model_name=model_name, model_version=model_version),
            "explainability/summary.txt",
        )
        mlflow.log_text(report.importance.write_csv(), "explainability/importance.csv")
        mlflow.log_text(report.segments.write_csv(), "explainability/segments.csv")
        mlflow.log_text(report.segment_top.write_csv(), "explainability/segment_top_features.csv")
        for feature, table in report.dependence.items():
            mlflow.log_text(table.write_csv(), f"explainability/dependence/{feature}.csv")
        mlflow.log_figure(importance_figure(report.importance), "explainability/importance.png")
        mlflow.log_figure(dependence_figure(report.dependence), "explainability/dependence.png")
        return run.info.run_id
