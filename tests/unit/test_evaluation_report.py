"""Unit tests for the evaluation report, its plots, and MLflow logging."""

from pathlib import Path

import numpy as np
import pytest
from mlflow import MlflowClient

from promolift.data.split import LockedSplitError
from promolift.evaluation.plots import decile_uplift_figure, qini_curve_figure
from promolift.evaluation.report import evaluate_ranking, log_evaluation
from promolift.evaluation.uncertainty import random_ranking_noise_floor
from promolift.tracking.mlflow_tracking import TrackingConfig, local_tracking_config, start_run

_FAST = {"n_bootstrap": 30}


@pytest.fixture
def experiment() -> dict[str, np.ndarray]:
    # Only clients with x > 0.5 respond to treatment (+20pp): x is the oracle ranking.
    rng = np.random.default_rng(0)
    n = 2_000
    x = rng.random(n)
    treatment = rng.integers(0, 2, n)
    outcome = (rng.random(n) < 0.3 + 0.2 * (x > 0.5) * treatment).astype(int)
    return {"x": x, "t": treatment, "y": outcome}


@pytest.fixture
def floor(experiment: dict):
    return random_ranking_noise_floor(experiment["t"], experiment["y"], n_rankings=50)


def test_report_combines_estimates_intervals_and_noise_floor(experiment: dict, floor) -> None:
    report = evaluate_ranking(
        "oracle",
        experiment["x"],
        experiment["t"],
        experiment["y"],
        split="val",
        noise_floor=floor,
        **_FAST,
    )

    assert report.model_name == "oracle"
    assert report.split == "val"
    assert report.final_evaluation is False
    assert report.bootstrap.qini_auc.estimate == pytest.approx(report.metrics.qini_auc)
    assert report.beats_noise_floor["qini_auc"] is True
    assert report.deciles.height == 10


def test_report_reuses_a_supplied_noise_floor(experiment: dict, floor) -> None:
    report = evaluate_ranking(
        "oracle",
        experiment["x"],
        experiment["t"],
        experiment["y"],
        split="val",
        noise_floor=floor,
        **_FAST,
    )

    assert report.noise_floor is floor


def test_report_rejects_a_noise_floor_from_different_clients(experiment: dict, floor) -> None:
    with pytest.raises(ValueError, match="noise floor"):
        evaluate_ranking(
            "oracle",
            experiment["x"][:-1],
            experiment["t"][:-1],
            experiment["y"][:-1],
            split="val",
            noise_floor=floor,
            **_FAST,
        )


def test_report_rejects_a_noise_floor_with_different_k_grid(experiment: dict, floor) -> None:
    with pytest.raises(ValueError, match="noise floor"):
        evaluate_ranking(
            "oracle",
            experiment["x"],
            experiment["t"],
            experiment["y"],
            split="val",
            noise_floor=floor,
            k_grid=(0.25,),
            **_FAST,
        )


def test_evaluating_on_test_is_locked(experiment: dict) -> None:
    with pytest.raises(LockedSplitError):
        evaluate_ranking("oracle", experiment["x"], experiment["t"], experiment["y"], split="test")


def test_final_evaluation_on_test_is_allowed_and_flagged(experiment: dict, floor) -> None:
    report = evaluate_ranking(
        "oracle",
        experiment["x"],
        experiment["t"],
        experiment["y"],
        split="test",
        final_evaluation=True,
        noise_floor=floor,
        **_FAST,
    )

    assert report.final_evaluation is True


def _report(experiment: dict, floor, scores: np.ndarray, name: str = "oracle"):
    return evaluate_ranking(
        name, scores, experiment["t"], experiment["y"], split="val", noise_floor=floor, **_FAST
    )


def test_report_stores_a_downsampled_qini_curve_from_zero(experiment: dict, floor) -> None:
    curve = _report(experiment, floor, experiment["x"]).qini_curve

    assert curve.fraction_targeted[0] == 0.0
    assert curve.fraction_targeted[-1] == pytest.approx(1.0)
    assert len(curve.fraction_targeted) <= 201
    assert len(curve.incremental_conversions) == len(curve.fraction_targeted)


def test_qini_figure_draws_one_line_per_model_plus_random(experiment: dict, floor) -> None:
    reports = [
        _report(experiment, floor, experiment["x"], "oracle"),
        _report(experiment, floor, -experiment["x"], "inverted"),
    ]

    axes = qini_curve_figure(reports).axes[0]

    assert [line.get_label() for line in axes.get_lines()] == [
        "oracle",
        "inverted",
        "Random targeting",
    ]
    assert axes.get_legend() is not None


def test_qini_figure_refuses_to_overlay_different_splits(experiment: dict, floor) -> None:
    val = _report(experiment, floor, experiment["x"])
    other = evaluate_ranking(
        "oracle",
        experiment["x"],
        experiment["t"],
        experiment["y"],
        split="train",
        noise_floor=floor,
        **_FAST,
    )

    with pytest.raises(ValueError, match="same split"):
        qini_curve_figure([val, other])


def test_decile_figure_draws_one_bar_per_bucket(experiment: dict, floor) -> None:
    report = evaluate_ranking(
        "oracle",
        experiment["x"],
        experiment["t"],
        experiment["y"],
        split="val",
        noise_floor=floor,
        **_FAST,
    )

    figure = decile_uplift_figure(report)

    assert len(figure.axes[0].patches) == 10


@pytest.fixture
def config(tmp_path: Path) -> TrackingConfig:
    return local_tracking_config(tmp_path / "mlruns")


def test_log_evaluation_records_split_prefixed_metrics_tags_and_artifacts(
    experiment: dict, floor, config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    report = evaluate_ranking(
        "oracle",
        experiment["x"],
        experiment["t"],
        experiment["y"],
        split="val",
        noise_floor=floor,
        **_FAST,
    )

    with start_run("test-evaluation", config=config, **lineage_dirs) as run:
        log_evaluation(report)
        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=config.tracking_uri)
    data = client.get_run(run_id).data
    assert data.metrics["val_qini_auc"] == pytest.approx(report.metrics.qini_auc)
    assert data.metrics["val_qini_auc_ci_lower"] == pytest.approx(report.bootstrap.qini_auc.lower)
    assert data.metrics["val_noise_floor_qini_auc_upper"] == pytest.approx(floor.qini_auc.upper)
    assert "val_uplift_at_30pct" in data.metrics
    assert data.tags["evaluated_split"] == "val"
    assert data.tags["final_evaluation"] == "false"
    assert data.tags["val_beats_noise_floor_qini_auc"] == "true"
    artifacts = {a.path for a in client.list_artifacts(run_id, "evaluation/val")}
    assert artifacts == {
        "evaluation/val/report.json",
        "evaluation/val/deciles.csv",
        "evaluation/val/qini_curve.png",
        "evaluation/val/decile_uplift.png",
    }


def test_log_evaluation_requires_an_active_run(experiment: dict, floor) -> None:
    report = evaluate_ranking(
        "oracle",
        experiment["x"],
        experiment["t"],
        experiment["y"],
        split="val",
        noise_floor=floor,
        **_FAST,
    )

    with pytest.raises(RuntimeError, match="start_run"):
        log_evaluation(report)
