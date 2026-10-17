"""Unit tests for assembling, summarizing, and logging an explanation."""

import mlflow
import pytest
from mlflow import MlflowClient

from promolift.explainability.report import (
    CAVEAT,
    DEPENDENCE_FEATURES,
    build_report,
    log_explanation,
    summary,
)
from promolift.explainability.shap_values import explain
from promolift.tracking.mlflow_tracking import Experiment, local_tracking_config


@pytest.fixture(scope="module")
def explanation(explained_model):
    return explain(explained_model.model, explained_model.features)


@pytest.fixture(scope="module")
def report(explanation):
    return build_report(explanation)


# --- the report -----------------------------------------------------------------------------


def test_the_report_covers_every_feature_and_segment(explanation, report) -> None:
    assert report.importance["feature"].to_list()[0] == "b"
    assert set(report.importance["feature"]) == set(explanation.contributions.columns)
    assert set(report.segments["segment_type"]) >= {"predicted_uplift_tier"}
    assert report.segment_top["segment"].n_unique() == report.segment_top.height


def test_dependence_covers_the_leading_features_in_order(explanation) -> None:
    report = build_report(explanation, dependence_features=2)

    assert list(report.dependence) == report.importance["feature"].head(2).to_list()
    assert len(build_report(explanation).dependence) == min(
        DEPENDENCE_FEATURES, len(explanation.contributions.columns)
    )


# --- the summary ----------------------------------------------------------------------------


def test_the_summary_names_the_model_the_leading_feature_and_the_caveat(
    explanation, report
) -> None:
    text = summary(explanation, report, model_name="promolift_ai.models.m", model_version="7")

    assert "promolift_ai.models.m version 7" in text
    assert f"{len(explanation.scores):,} train+val clients" in text
    assert " 1. b " in text
    assert "predicted_uplift_tier=top 10%" in text
    assert CAVEAT in text


def test_the_summary_reports_the_additivity_gap(explanation, report) -> None:
    text = summary(explanation, report, model_name="m", model_version="1")

    assert "add up to their score (largest gap" in text


# --- the run --------------------------------------------------------------------------------


def test_the_run_records_metrics_tags_and_artifacts(
    explanation, report, lineage_dirs, tmp_path
) -> None:
    config = local_tracking_config(tmp_path / "mlruns")

    run_id = log_explanation(
        explanation, report, model_name="m", model_version="3", seed=5, config=config,
        lineage=lineage_dirs,
    )  # fmt: skip

    client = MlflowClient(tracking_uri=config.tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["run_type"] == "shap_explanation"
    assert run.data.tags["model_version"] == "3"
    assert run.data.tags["top_feature"] == "b"
    assert run.data.params["seed"] == "5"
    assert run.data.params["n_clients"] == str(len(explanation.scores))
    metrics = run.data.metrics
    assert metrics["additivity_error"] < 1e-9
    assert metrics["top_feature_share"] == pytest.approx(report.importance["share"][0])
    assert metrics["importance_b"] == pytest.approx(report.importance["mean_abs_contribution"][0])
    assert {"importance_noise", "importance_segment"} <= set(metrics)
    files = {a.path for a in client.list_artifacts(run_id, "explainability")}
    assert files == {
        "explainability/summary.txt",
        "explainability/importance.csv",
        "explainability/segments.csv",
        "explainability/segment_top_features.csv",
        "explainability/dependence",
        "explainability/importance.png",
        "explainability/dependence.png",
    }
    dependence = {a.path for a in client.list_artifacts(run_id, "explainability/dependence")}
    assert dependence == {f"explainability/dependence/{f}.csv" for f in report.dependence}
    assert mlflow.get_experiment(run.info.experiment_id).name.endswith(Experiment.EXPLAINABILITY)


def test_no_client_level_file_is_logged(explanation, report, lineage_dirs, tmp_path) -> None:
    config = local_tracking_config(tmp_path / "mlruns")

    run_id = log_explanation(
        explanation, report, model_name="m", model_version="1", seed=1, config=config,
        lineage=lineage_dirs,
    )  # fmt: skip

    client = MlflowClient(tracking_uri=config.tracking_uri)
    names = {a.path for a in client.list_artifacts(run_id, "explainability")}
    assert not any("contribution" in name or "client" in name for name in names)
