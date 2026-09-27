"""Unit tests for bootstrap CIs, the random-ranking noise floor, and paired comparison."""

import numpy as np
import pytest

from promolift.evaluation.ranking_metrics import ranking_metrics
from promolift.evaluation.uncertainty import (
    beats_noise_floor,
    bootstrap_metrics,
    paired_comparison,
    random_ranking_noise_floor,
    stratified_resample_indices,
)

_FAST = {"n_bootstrap": 60}


@pytest.fixture
def experiment() -> dict[str, np.ndarray]:
    # Only clients with x > 0.5 respond to treatment (+20pp): x is the oracle ranking.
    rng = np.random.default_rng(0)
    n = 2_000
    x = rng.random(n)
    treatment = rng.integers(0, 2, n)
    outcome = (rng.random(n) < 0.3 + 0.2 * (x > 0.5) * treatment).astype(int)
    return {"x": x, "t": treatment, "y": outcome, "noise": rng.random(n)}


def test_stratified_resampling_keeps_each_arm_the_same_size() -> None:
    treatment = np.array([1] * 30 + [0] * 70)
    rng = np.random.default_rng(0)

    indices = stratified_resample_indices(treatment, rng)

    assert len(indices) == 100
    assert treatment[indices].sum() == 30


def test_bootstrap_interval_brackets_the_full_data_estimate(experiment: dict) -> None:
    point = ranking_metrics(experiment["x"], experiment["t"], experiment["y"])

    result = bootstrap_metrics(experiment["x"], experiment["t"], experiment["y"], **_FAST)

    assert result.qini_auc.estimate == pytest.approx(point.qini_auc)
    assert result.qini_auc.lower < result.qini_auc.estimate < result.qini_auc.upper
    assert result.uplift_at_k[0.3].estimate == pytest.approx(point.uplift_at_k[0.3])
    assert result.n_bootstrap == 60


def test_bootstrap_is_reproducible_for_a_seed(experiment: dict) -> None:
    first = bootstrap_metrics(experiment["x"], experiment["t"], experiment["y"], seed=5, **_FAST)
    second = bootstrap_metrics(experiment["x"], experiment["t"], experiment["y"], seed=5, **_FAST)

    assert first == second


def test_noise_floor_is_centred_on_zero_qini_and_on_the_ate_for_uplift_at_k(
    experiment: dict,
) -> None:
    t, y = experiment["t"], experiment["y"]
    ate = y[t == 1].mean() - y[t == 0].mean()

    floor = random_ranking_noise_floor(t, y, n_rankings=100)

    assert floor.qini_auc.lower < 0 < floor.qini_auc.upper
    # A random top-k% is just a random sample, so its uplift is the ATE.
    assert floor.uplift_at_k[0.3].lower < ate < floor.uplift_at_k[0.3].upper
    assert floor.n_rankings == 100


def test_oracle_beats_the_noise_floor_but_a_random_ranking_does_not(experiment: dict) -> None:
    t, y = experiment["t"], experiment["y"]
    floor = random_ranking_noise_floor(t, y, n_rankings=100)

    oracle = beats_noise_floor(ranking_metrics(experiment["x"], t, y), floor)
    random = beats_noise_floor(ranking_metrics(experiment["noise"], t, y), floor)

    assert oracle["qini_auc"] is True
    assert random["qini_auc"] is False


def test_paired_comparison_prefers_oracle_over_random(experiment: dict) -> None:
    comparison = paired_comparison(
        experiment["x"], experiment["noise"], experiment["t"], experiment["y"], **_FAST
    )

    assert comparison.qini_auc.win_rate == pytest.approx(1.0)
    assert comparison.qini_auc.lower > 0


def test_paired_comparison_of_a_model_with_itself_is_a_tie(experiment: dict) -> None:
    comparison = paired_comparison(
        experiment["x"], experiment["x"], experiment["t"], experiment["y"], **_FAST
    )

    assert comparison.qini_auc.difference == pytest.approx(0.0)
    assert comparison.qini_auc.win_rate == pytest.approx(0.0)
    assert comparison.uplift_at_k[0.3].difference == pytest.approx(0.0)


def test_paired_comparison_rejects_score_arrays_of_different_length(experiment: dict) -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_comparison(
            experiment["x"], experiment["x"][:-1], experiment["t"], experiment["y"], **_FAST
        )
