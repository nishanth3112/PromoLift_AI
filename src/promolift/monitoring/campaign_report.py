"""Present a campaign evaluation: a plain-text summary, and the MLflow run that records it.

The run holds the numbers (metrics), the per-decile table and the summary
(artifacts), and a calibration plot -- not the client results, which stay in
the results file.
"""

from __future__ import annotations

import mlflow

from promolift.evaluation.plots import decile_calibration_figure
from promolift.evaluation.ranking_metrics import uplift_at_k_name
from promolift.monitoring.campaign import CampaignEvaluation
from promolift.tracking.mlflow_tracking import (
    Experiment,
    TrackingConfig,
    default_tracking_config,
    start_run,
)


def _pp(value: float) -> str:
    return f"{value * 100:+.2f}pp"


def summary(evaluation: CampaignEvaluation) -> str:
    """The campaign's result in a few lines: realized vs predicted, ranking, and caveats."""
    e, level = evaluation, round(evaluation.confidence * 100)
    verdict = (
        "the prediction is inside the realized interval"
        if e.prediction_within_interval
        else "the prediction is OUTSIDE the realized interval"
    )
    lines = [
        f"Campaign from scoring run {e.scoring_run_id}",
        f"  texted {e.n_treated:,} clients, held back {e.n_control:,}",
        f"  purchase rate: texted {e.treated_rate:.2%}, held back {e.control_rate:.2%}",
        f"  realized uplift {_pp(e.realized_uplift.estimate)} "
        f"({level}% interval {_pp(e.realized_uplift.lower)} to {_pp(e.realized_uplift.upper)}); "
        f"predicted {_pp(e.predicted_uplift)}: {verdict}",
        f"  realized minus predicted {_pp(e.deviation.estimate)} "
        f"({_pp(e.deviation.lower)} to {_pp(e.deviation.upper)})",
        f"  about {e.incremental_purchases:,.0f} purchases caused by the texts",
        f"  Qini AUC {e.ranking.qini_auc.estimate:+.4f} "
        f"({e.ranking.qini_auc.lower:+.4f} to {e.ranking.qini_auc.upper:+.4f})",
    ]
    for k, interval in e.ranking.uplift_at_k.items():
        lines.append(
            f"  uplift in the top {k:.0%} of targeted: realized {_pp(interval.estimate)} "
            f"({_pp(interval.lower)} to {_pp(interval.upper)}), "
            f"predicted {_pp(e.predicted_uplift_at_k[k])}"
        )
    if e.n_missing_treated or e.n_missing_control:
        lines.append(
            f"  CAUTION: no result for {e.n_missing_treated:,} texted and "
            f"{e.n_missing_control:,} held-back clients; unequal gaps bias the comparison"
        )
    return "\n".join(lines)


def log_campaign_evaluation(
    evaluation: CampaignEvaluation,
    *,
    config: TrackingConfig | None = None,
    results_name: str | None = None,
    lineage: dict | None = None,
) -> str:
    """Record the evaluation as a run in ``promolift-campaigns`` and return its run id.

    Args:
        evaluation: Output of ``campaign.evaluate_campaign``.
        config: Tracking destination; defaults to ``default_tracking_config()``.
        results_name: The results file's name, kept as a tag.
        lineage: ``repo_dir`` / ``data_dir`` / ``processed_dir`` overrides (for tests).
    """
    e = evaluation
    config = config if config is not None else default_tracking_config()
    tags = {
        "run_type": "campaign_evaluation",
        "scoring_run_id": e.scoring_run_id,
        "prediction_within_interval": str(e.prediction_within_interval).lower(),
    }
    if results_name is not None:
        tags["results_file"] = results_name
    with start_run(
        Experiment.CAMPAIGNS,
        run_name="campaign_evaluation",
        tags=tags,
        config=config,
        **(lineage or {}),
    ) as run:
        mlflow.log_params(
            {
                "scoring_run_id": e.scoring_run_id,
                "confidence": e.confidence,
                "n_bootstrap": e.ranking.n_bootstrap,
            }
        )
        mlflow.log_metrics(
            {
                "n_treated": e.n_treated,
                "n_control": e.n_control,
                "n_missing_treated": e.n_missing_treated,
                "n_missing_control": e.n_missing_control,
                "treated_rate": e.treated_rate,
                "control_rate": e.control_rate,
                "realized_uplift": e.realized_uplift.estimate,
                "realized_uplift_lower": e.realized_uplift.lower,
                "realized_uplift_upper": e.realized_uplift.upper,
                "predicted_uplift": e.predicted_uplift,
                "uplift_deviation": e.deviation.estimate,
                "uplift_deviation_lower": e.deviation.lower,
                "uplift_deviation_upper": e.deviation.upper,
                "incremental_purchases": e.incremental_purchases,
                "qini_auc": e.ranking.qini_auc.estimate,
                "qini_auc_lower": e.ranking.qini_auc.lower,
                "qini_auc_upper": e.ranking.qini_auc.upper,
            }
            | {
                f"{uplift_at_k_name(k)}_{part}": value
                for k, interval in e.ranking.uplift_at_k.items()
                for part, value in (
                    ("realized", interval.estimate),
                    ("lower", interval.lower),
                    ("upper", interval.upper),
                    ("predicted", e.predicted_uplift_at_k[k]),
                )
            }
        )
        mlflow.log_text(summary(e), "campaign/summary.txt")
        mlflow.log_text(e.deciles.write_csv(), "campaign/deciles.csv")
        mlflow.log_figure(decile_calibration_figure(e.deciles), "campaign/decile_calibration.png")
        return run.info.run_id
