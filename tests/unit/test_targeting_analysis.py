"""Unit tests for the out-of-fold profit analysis and its MLflow run."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from mlflow import MlflowClient

from promolift.models.dataset import ModelFrame
from promolift.models.training import ModelSpec
from promolift.optimization.analysis import out_of_fold_rankings, run_targeting_analysis
from promolift.optimization.business import BusinessConfig
from promolift.tracking.mlflow_tracking import TrackingConfig, local_tracking_config

_GRID = (0.02, 0.1, 0.3)


def _frame(n: int, seed: int) -> ModelFrame:
    # Persuadables (b < 0.3) buy 10% -> 50% if texted; everyone else 40% either way.
    rng = np.random.default_rng(seed)
    b = rng.random(n)
    treatment = rng.integers(0, 2, n)
    outcome = (rng.random(n) < np.where(b < 0.3, 0.1 + 0.4 * treatment, 0.4)).astype(int)
    return ModelFrame(
        split="train+val",
        client_ids=pl.Series([f"c{i}" for i in range(n)]),
        features=pd.DataFrame({"b": b, "noise": rng.random(n)}),
        treatment=treatment,
        outcome=outcome,
    )


def _config(**overrides) -> BusinessConfig:
    values = {
        "currency": "RUB",
        "sms_cost": 3.0,
        "gross_margin": 0.25,
        "margin_per_purchase": None,
        "budget": None,
        "campaign_clients": None,
        "break_even_grid": _GRID,
    } | overrides
    return BusinessConfig(**values)


@pytest.fixture(scope="module")
def frame() -> ModelFrame:
    return _frame(6_000, seed=0)


@pytest.fixture(scope="module")
def rankings(frame: ModelFrame) -> dict:
    rng = np.random.default_rng(1)
    # An oracle that ranks persuadables first, and a random ranking.
    return {"oracle": -frame.features["b"].to_numpy(), "random": rng.random(len(frame.outcome))}


@pytest.fixture
def tracking(tmp_path: Path) -> TrackingConfig:
    return local_tracking_config(tmp_path / "mlruns")


def _analyze(rankings, frame, tracking, lineage, **config_overrides):
    return run_targeting_analysis(
        rankings,
        frame,
        chosen="oracle",
        config=_config(**config_overrides),
        average_transaction=100.0,  # margin 25, break-even 3 / 25 = 12%
        leaderboard_run_id="leaderboard123",
        n_folds=3,
        n_bootstrap=50,
        tracking=tracking,
        lineage=lineage,
    )


def test_out_of_fold_rankings_score_every_client_with_within_fold_ranks(
    frame: ModelFrame,
) -> None:
    specs = [ModelSpec("t_learner", "t_learner"), ModelSpec("random", "random")]

    rankings = out_of_fold_rankings(specs, frame, n_folds=3, seed=0)

    assert set(rankings) == {"t_learner", "random"}
    for scores in rankings.values():
        assert scores.shape == frame.outcome.shape
        assert scores.min() > 0
        assert scores.max() == pytest.approx(1.0)


def test_the_oracle_texts_the_persuadables_and_beats_random(
    rankings: dict, frame: ModelFrame, tracking: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    analysis = _analyze(rankings, frame, tracking, lineage_dirs)

    # Persuadables are the top ~30% (uplift 0.4 > break-even 0.12); the rest have none.
    assert 0.15 <= analysis.choice.depth <= 0.4
    assert analysis.choice.profit_per_1000 > 0
    assert analysis.curves["oracle"].profit_per_1000.max() > (
        analysis.curves["random"].profit_per_1000.max()
    )
    assert analysis.economics.break_even_uplift == pytest.approx(0.12)


def test_logs_the_decision_curves_and_sensitivity(
    rankings: dict, frame: ModelFrame, tracking: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    analysis = _analyze(rankings, frame, tracking, lineage_dirs)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    run = client.get_run(analysis.run_id)
    assert run.data.tags["run_type"] == "targeting"
    assert run.data.tags["chosen_model"] == "oracle"
    assert run.data.tags["reference_models"] == "random"
    assert run.data.tags["depth_rule"] == "max_expected_profit"
    assert run.data.metrics["depth"] == analysis.choice.depth
    assert run.data.metrics["shallowest_tied_depth"] == analysis.choice.shallowest_tied_depth
    assert run.data.metrics["margin_per_purchase"] == pytest.approx(25.0)
    assert "random_best_profit_per_1000" in run.data.metrics
    artifacts = {a.path for a in client.list_artifacts(analysis.run_id, "targeting")}
    assert artifacts == {
        "targeting/decision.json",
        "targeting/profit_curves.csv",
        "targeting/sensitivity.csv",
        "targeting/profit_curves.png",
    }
    decision = json.loads(
        Path(client.download_artifacts(analysis.run_id, "targeting/decision.json")).read_text()
    )
    assert decision["chosen_model"] == "oracle"
    curves = pl.read_csv(client.download_artifacts(analysis.run_id, "targeting/profit_curves.csv"))
    assert curves.height == 2 * 101
    assert set(curves["model"]) == {"oracle", "random"}


def test_sensitivity_has_one_row_per_break_even_and_never_deepens(
    rankings: dict, frame: ModelFrame, tracking: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    table = _analyze(rankings, frame, tracking, lineage_dirs).sensitivity

    assert table["break_even_uplift"].to_list() == list(_GRID)
    assert table["sms_cost"].to_list() == pytest.approx([r * 25.0 for r in _GRID])
    assert table["depth"].is_sorted(descending=True)
    assert (table["shallowest_tied_depth"] <= table["depth"]).all()
    assert "random_best_profit_per_1000" in table.columns


def test_a_budget_caps_the_depth(
    rankings: dict, frame: ModelFrame, tracking: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    # 3 RUB x 10,000 clients x 10% = 3,000 RUB.
    analysis = _analyze(
        rankings, frame, tracking, lineage_dirs, budget=3_000.0, campaign_clients=10_000
    )

    assert analysis.choice.max_depth == pytest.approx(0.1)
    assert analysis.choice.depth <= 0.1


def test_the_chosen_model_needs_a_ranking(
    rankings: dict, frame: ModelFrame, tracking: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    with pytest.raises(ValueError, match="No ranking"):
        _analyze({"random": rankings["random"]}, frame, tracking, lineage_dirs)
