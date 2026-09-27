"""Unit tests for the train-evaluate-log harness and the leaderboard run."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from mlflow import MlflowClient

from promolift.models.dataset import ModelFrame
from promolift.models.training import log_leaderboard, train_and_evaluate
from promolift.tracking.mlflow_tracking import (
    Experiment,
    TrackingConfig,
    local_tracking_config,
)

_FAST = {"n_bootstrap": 20, "n_rankings": 30}
_MODELS = ["random", "response", "t_learner"]


def _frame(split: str, n: int, seed: int) -> ModelFrame:
    # Sure things (b >= 0.5) buy 80% regardless; persuadables 10% -> 40% if treated.
    rng = np.random.default_rng(seed)
    b = rng.random(n)
    treatment = rng.integers(0, 2, n)
    outcome = (rng.random(n) < np.where(b >= 0.5, 0.8, 0.1 + 0.3 * treatment)).astype(int)
    return ModelFrame(
        split=split,
        client_ids=pl.Series([f"{split}{i}" for i in range(n)]),
        features=pd.DataFrame({"b": b, "noise": rng.random(n)}),
        treatment=treatment,
        outcome=outcome,
    )


@pytest.fixture
def config(tmp_path: Path) -> TrackingConfig:
    return local_tracking_config(tmp_path / "mlruns")


@pytest.fixture
def results(config: TrackingConfig, lineage_dirs: dict[str, Path]) -> list:
    return train_and_evaluate(
        _MODELS,
        _frame("train", 4_000, seed=0),
        _frame("val", 3_000, seed=1),
        config=config,
        lineage=lineage_dirs,
        **_FAST,
    )


def _runs(config: TrackingConfig, run_type: str) -> list:
    client = MlflowClient(tracking_uri=config.tracking_uri)
    experiment = client.get_experiment_by_name(config.experiment_name(Experiment.UPLIFT_MODELS))
    return client.search_runs(
        [experiment.experiment_id], filter_string=f"tags.run_type = '{run_type}'"
    )


def test_logs_one_evaluated_run_per_model(results: list, config: TrackingConfig) -> None:
    runs = {r.data.tags["model_name"]: r for r in _runs(config, "model")}

    assert set(runs) == set(_MODELS)
    for name, run in runs.items():
        assert run.data.params["model"] == name
        assert "val_qini_auc" in run.data.metrics
        assert "fit_seconds" in run.data.metrics
    assert [r.name for r in results] == _MODELS


def test_all_models_share_one_noise_floor(results: list) -> None:
    floors = {id(r.report.noise_floor) for r in results}

    assert len(floors) == 1


def test_leaderboard_ranks_by_qini_and_compares_against_best_and_response(
    results: list, config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    run_id = log_leaderboard(results, config=config, lineage=lineage_dirs, n_bootstrap=20)

    client = MlflowClient(tracking_uri=config.tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["run_type"] == "leaderboard"
    assert run.data.tags["best_model"] == "t_learner"
    artifacts = {a.path for a in client.list_artifacts(run_id, "leaderboard")}
    assert artifacts == {
        "leaderboard/leaderboard.csv",
        "leaderboard/paired_comparisons.json",
        "leaderboard/qini_curves.png",
    }

    local = client.download_artifacts(run_id, "leaderboard/paired_comparisons.json")
    comparisons = json.loads(Path(local).read_text())
    pairs = {(c["model_a"], c["model_b"]) for c in comparisons}
    assert pairs == {("t_learner", "random"), ("t_learner", "response")}
    t_vs_response = next(c for c in comparisons if c["model_b"] == "response")
    assert t_vs_response["qini_auc"]["win_rate"] > 0.9


def test_leaderboard_table_is_sorted_best_first(
    results: list, config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    run_id = log_leaderboard(results, config=config, lineage=lineage_dirs, n_bootstrap=20)

    local = MlflowClient(tracking_uri=config.tracking_uri).download_artifacts(
        run_id, "leaderboard/leaderboard.csv"
    )
    table = pl.read_csv(local)
    assert table["model"][0] == "t_learner"
    assert table["qini_auc"].is_sorted(descending=True)
    # On sure-thing data the response model ranks *below* random, not just within noise.
    response = table.filter(pl.col("model") == "response")
    assert response["below_noise_floor_qini_auc"][0] is True
    assert response["beats_noise_floor_qini_auc"][0] is False
    assert {"qini_auc_ci_lower", "uplift_at_30pct", "beats_noise_floor_qini_auc", "run_id"} <= set(
        table.columns
    )
