"""Unit tests for the campaign summary and its MLflow run."""

import mlflow
import polars as pl
import pytest
from mlflow import MlflowClient

from promolift.evaluation.plots import decile_calibration_figure
from promolift.monitoring.campaign import evaluate_campaign
from promolift.monitoring.campaign_data import match_results, validate_results
from promolift.monitoring.campaign_report import log_campaign_evaluation, summary
from promolift.monitoring.campaign_simulation import simulate_campaign
from promolift.tracking.mlflow_tracking import Experiment, local_tracking_config


@pytest.fixture
def evaluation(make_campaign_batch):
    batch = make_campaign_batch(6_000, send_share=0.5, holdout_share=0.2)
    data = match_results(
        batch, validate_results(simulate_campaign(batch, effect_scale=0.5, seed=2))
    )
    return evaluate_campaign(data, n_bootstrap=200, ranking_bootstrap=5)


def test_the_summary_states_realized_against_predicted_and_the_verdict(evaluation) -> None:
    text = summary(evaluation)

    assert "scoring run run-a" in text
    assert "texted 2,400 clients, held back 600" in text
    assert "realized uplift" in text
    assert "95% interval" in text
    assert "predicted" in text
    assert ("OUTSIDE" in text) == (not evaluation.prediction_within_interval)
    assert "Qini AUC" in text
    assert "uplift in the top 10% of targeted" in text
    assert "CAUTION" not in text


def test_the_summary_warns_about_clients_without_results(make_campaign_batch) -> None:
    batch = make_campaign_batch(6_000, send_share=0.5, holdout_share=0.2)
    results = simulate_campaign(batch, seed=2).filter(pl.int_range(pl.len()) % 20 != 0)
    data = match_results(batch, validate_results(results))

    text = summary(evaluate_campaign(data, n_bootstrap=100, ranking_bootstrap=5))

    assert data.n_missing_treated + data.n_missing_control == 150
    assert (
        f"CAUTION: no result for {data.n_missing_treated:,} texted and "
        f"{data.n_missing_control:,} held-back clients"
    ) in text


def test_the_run_records_metrics_tags_and_artifacts(evaluation, lineage_dirs, tmp_path) -> None:
    config = local_tracking_config(tmp_path / "mlruns")

    run_id = log_campaign_evaluation(
        evaluation, config=config, results_name="results.csv", lineage=lineage_dirs
    )

    client = MlflowClient(tracking_uri=config.tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["run_type"] == "campaign_evaluation"
    assert run.data.tags["scoring_run_id"] == "run-a"
    assert run.data.tags["results_file"] == "results.csv"
    assert run.data.tags["prediction_within_interval"] in {"true", "false"}
    metrics = run.data.metrics
    assert metrics["realized_uplift"] == pytest.approx(evaluation.realized_uplift.estimate)
    assert metrics["predicted_uplift"] == pytest.approx(evaluation.predicted_uplift)
    assert metrics["uplift_deviation_upper"] == pytest.approx(evaluation.deviation.upper)
    assert metrics["n_control"] == 600
    assert metrics["uplift_at_10pct_predicted"] == pytest.approx(
        evaluation.predicted_uplift_at_k[0.1]
    )
    assert "uplift_at_10pct_realized" in metrics
    artifacts = {a.path for a in client.list_artifacts(run_id, "campaign")}
    assert artifacts == {
        "campaign/summary.txt",
        "campaign/deciles.csv",
        "campaign/decile_calibration.png",
    }
    experiment = mlflow.get_experiment(run.info.experiment_id)
    assert experiment.name.endswith(Experiment.CAMPAIGNS)


def test_the_calibration_figure_skips_deciles_without_a_realized_uplift(evaluation) -> None:
    deciles = pl.concat(
        [
            evaluation.deciles,
            evaluation.deciles.head(1).with_columns(
                pl.lit(6, dtype=pl.Int8).alias("decile"),
                pl.lit(None, dtype=pl.Float64).alias("realized_uplift"),
                pl.lit(None, dtype=pl.Float64).alias("realized_lower"),
                pl.lit(None, dtype=pl.Float64).alias("realized_upper"),
            ),
        ],
        how="vertical_relaxed",
    )

    figure = decile_calibration_figure(deciles)

    axes = figure.axes[0]
    assert list(axes.get_xticks()) == [1, 2, 3, 4, 5]
    assert {text.get_text() for text in axes.get_legend().get_texts()} == {
        "realized (95% interval)",
        "predicted",
    }
