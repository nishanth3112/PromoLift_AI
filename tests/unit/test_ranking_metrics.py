"""Unit tests for uplift ranking metrics, using synthetic experiments with a known effect."""

import numpy as np
import pytest

from promolift.evaluation.ranking_metrics import (
    DEFAULT_K_GRID,
    ranking_metrics,
    uplift_by_decile,
)


@pytest.fixture
def experiment() -> dict[str, np.ndarray]:
    # Randomized experiment where only clients with x > 0.5 respond to treatment
    # (+20pp); a ranking by x is the oracle, by -x the worst possible.
    rng = np.random.default_rng(0)
    n = 4_000
    x = rng.random(n)
    treatment = rng.integers(0, 2, n)
    outcome = (rng.random(n) < 0.3 + 0.2 * (x > 0.5) * treatment).astype(int)
    return {"x": x, "treatment": treatment, "outcome": outcome, "noise": rng.random(n)}


def test_oracle_ranking_beats_random_which_beats_inverted(experiment: dict) -> None:
    t, y = experiment["treatment"], experiment["outcome"]

    oracle = ranking_metrics(experiment["x"], t, y)
    random = ranking_metrics(experiment["noise"], t, y)
    inverted = ranking_metrics(-experiment["x"], t, y)

    assert oracle.qini_auc > 0.05
    assert inverted.qini_auc < -0.05
    assert oracle.qini_auc > random.qini_auc > inverted.qini_auc
    assert oracle.uplift_auc > random.uplift_auc > inverted.uplift_auc


def test_reports_uplift_at_every_k_in_the_grid(experiment: dict) -> None:
    metrics = ranking_metrics(experiment["x"], experiment["treatment"], experiment["outcome"])

    assert set(metrics.uplift_at_k) == set(DEFAULT_K_GRID)
    assert metrics.n_clients == 4_000
    # Oracle's top 30% are all responders (x > 0.7), so uplift there is ~+20pp.
    assert metrics.uplift_at_k[0.3] == pytest.approx(0.2, abs=0.05)


def test_uplift_at_k_is_treated_minus_control_rate_in_the_top_k() -> None:
    scores = np.arange(10, 0, -1)  # already ranked, highest first
    treatment = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 0])
    outcome = np.array([1, 1, 0, 0, 0, 0, 1, 0, 1, 1])

    metrics = ranking_metrics(scores, treatment, outcome, k_grid=(0.5,))

    # Top 5: treated conversions 2/3, control conversions 0/2.
    assert metrics.uplift_at_k[0.5] == pytest.approx(2 / 3)


def test_uplift_by_decile_orders_clients_by_score(experiment: dict) -> None:
    table = uplift_by_decile(experiment["x"], experiment["treatment"], experiment["outcome"])

    assert table.height == 10
    assert set(table.columns) >= {"bucket", "n_treatment", "n_control", "uplift"}
    assert table["n_treatment"].sum() + table["n_control"].sum() == 4_000
    top, bottom = table["uplift"][0], table["uplift"][-1]
    assert top > bottom


@pytest.mark.parametrize(
    ("scores", "treatment", "outcome", "message"),
    [
        ([0.1, 0.2], [0, 1, 1], [0, 1, 0], "same length"),
        ([0.1, 0.2, 0.3], [0, 1, 2], [0, 1, 0], "treatment"),
        ([0.1, 0.2, 0.3], [0, 1, 1], [0, 5, 0], "outcome"),
        ([0.1, np.nan, 0.3], [0, 1, 1], [0, 1, 0], "NaN"),
        ([0.1, 0.2, 0.3], [1, 1, 1], [0, 1, 0], "both treated and control"),
    ],
)
def test_rejects_invalid_inputs(scores, treatment, outcome, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ranking_metrics(np.array(scores), np.array(treatment), np.array(outcome))
