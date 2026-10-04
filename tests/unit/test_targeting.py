"""Unit tests for the targeting math: uplift by depth, profit curves, and the depth rule."""

import numpy as np
import pytest
from sklift.metrics import uplift_at_k

from promolift.optimization.targeting import (
    Economics,
    budget_depth_cap,
    choose_depth,
    profit_curve,
    random_targeting_profit,
    uplift_by_depth,
)

_DEPTHS = tuple(round(i / 50, 2) for i in range(51))  # 0%, 2%, ..., 100%


def _known_optimum(n: int, seed: int, tail_uplift: float = 0.0):
    # Clients ranked best-first. The top 20% are persuadable (10% -> 40% if
    # texted, uplift 0.30); the rest convert at 30% + tail_uplift if texted.
    rng = np.random.default_rng(seed)
    scores = -np.arange(n, dtype=float)
    top = np.arange(n) < 0.2 * n
    treatment = rng.integers(0, 2, n)
    rate = np.where(top, 0.1 + 0.3 * treatment, 0.3 + tail_uplift * treatment)
    outcome = (rng.random(n) < rate).astype(int)
    return scores, treatment, outcome


@pytest.fixture(scope="module")
def known() -> object:
    scores, treatment, outcome = _known_optimum(40_000, seed=0)
    return uplift_by_depth(scores, treatment, outcome, depths=_DEPTHS, n_bootstrap=200, seed=1)


def test_profit_peaks_where_uplift_drops_below_break_even(known) -> None:
    # Break-even 10%: texting the top 20% earns 0.30 - 0.10 per client, every
    # later client loses 0.10, so profit peaks at 20%: 1,000 x 0.2 x 0.2 = 40.
    curve = profit_curve(known, Economics(sms_cost=0.1, margin_per_purchase=1.0))

    choice = choose_depth(curve)

    assert 0.16 <= choice.best_depth <= 0.24
    assert curve.profit_per_1000.max() == pytest.approx(40, abs=6)
    assert choice.chosen_depth <= choice.best_depth
    assert choice.chosen_depth in choice.tied_depths


def test_profit_is_margin_times_extra_purchases_minus_sms_cost(known) -> None:
    economics = Economics(sms_cost=0.05, margin_per_purchase=2.0)

    curve = profit_curve(known, economics)

    expected = 2.0 * curve.extra_purchases_per_1000 - 0.05 * curve.sms_per_1000
    np.testing.assert_allclose(curve.profit_per_1000, expected)
    assert curve.profit_per_1000[0] == 0
    assert np.isnan(curve.roi[0])
    assert (curve.profit_lower <= curve.profit_per_1000 + 1e-9).all()


def test_uplift_matches_scikit_uplift_at_each_depth() -> None:
    rng = np.random.default_rng(3)
    n = 2_000
    scores, treatment = rng.random(n), rng.integers(0, 2, n)
    outcome = (rng.random(n) < 0.2 + 0.2 * scores * treatment).astype(int)

    by_depth = uplift_by_depth(scores, treatment, outcome, depths=(0.1, 0.3, 0.5), n_bootstrap=5)

    for k, ours in zip((0.1, 0.3, 0.5), by_depth.uplift, strict=True):
        assert ours == pytest.approx(uplift_at_k(outcome, scores, treatment, "overall", k=k))
    np.testing.assert_allclose(by_depth.share, [0.1, 0.3, 0.5])


def test_text_nobody_when_no_depth_pays() -> None:
    scores, treatment, outcome = _known_optimum(20_000, seed=2)
    by_depth = uplift_by_depth(scores, treatment, outcome, depths=_DEPTHS, n_bootstrap=100)

    # Break-even 100%: no client's uplift can pay for the SMS.
    choice = choose_depth(profit_curve(by_depth, Economics(1.0, 1.0)))

    assert choice.best_depth == 0
    assert choice.chosen_depth == 0
    assert choice.profit_per_1000 == 0


def test_free_sms_with_uplift_everywhere_texts_deep() -> None:
    scores, treatment, outcome = _known_optimum(40_000, seed=4, tail_uplift=0.1)
    by_depth = uplift_by_depth(scores, treatment, outcome, depths=_DEPTHS, n_bootstrap=100)

    choice = choose_depth(profit_curve(by_depth, Economics(0.0, 1.0)))

    assert choice.best_depth >= 0.9


def test_shallowest_tied_depth_on_a_profit_plateau() -> None:
    # The tail's uplift equals break-even exactly, so profit is flat after 20%:
    # every depth from ~20% on is tied, and the rule picks the shallow end.
    scores, treatment, outcome = _known_optimum(40_000, seed=5, tail_uplift=0.1)
    by_depth = uplift_by_depth(scores, treatment, outcome, depths=_DEPTHS, n_bootstrap=200)

    choice = choose_depth(profit_curve(by_depth, Economics(0.1, 1.0)))

    assert 0.1 <= choice.chosen_depth <= 0.3
    assert choice.chosen_depth <= choice.best_depth
    assert len(choice.tied_depths) > 1


def test_budget_cap_limits_the_depth(known) -> None:
    curve = profit_curve(known, Economics(0.1, 1.0))

    choice = choose_depth(curve, max_depth=0.1)

    assert choice.best_depth <= 0.1
    assert choice.max_depth == 0.1
    assert all(depth <= 0.1 for depth in choice.tied_depths)


def test_best_depth_never_deepens_as_sms_gets_dearer(known) -> None:
    best = [
        choose_depth(profit_curve(known, Economics(cost, 1.0))).best_depth
        for cost in (0.0, 0.05, 0.1, 0.2, 0.5)
    ]

    assert best == sorted(best, reverse=True)


def test_bootstrap_is_reproducible_for_a_seed() -> None:
    scores, treatment, outcome = _known_optimum(5_000, seed=6)

    a = uplift_by_depth(scores, treatment, outcome, depths=_DEPTHS, n_bootstrap=20, seed=9)
    b = uplift_by_depth(scores, treatment, outcome, depths=_DEPTHS, n_bootstrap=20, seed=9)

    np.testing.assert_array_equal(a.sample_uplift, b.sample_uplift)


def test_random_targeting_is_all_or_nothing() -> None:
    assert random_targeting_profit(0.03, Economics(1.0, 100.0)) == pytest.approx(2_000)
    assert random_targeting_profit(0.03, Economics(5.0, 100.0)) == 0


def test_budget_depth_cap() -> None:
    assert budget_depth_cap(30_000, sms_cost=3.0, campaign_clients=100_000) == pytest.approx(0.1)
    assert budget_depth_cap(10**9, sms_cost=3.0, campaign_clients=100_000) == 1.0
    assert budget_depth_cap(100, sms_cost=0.0, campaign_clients=100_000) == 1.0
    with pytest.raises(ValueError):
        budget_depth_cap(-1, sms_cost=3.0, campaign_clients=100)


def test_economics_validation_and_break_even() -> None:
    assert Economics(3.0, 120.0).break_even_uplift == pytest.approx(0.025)
    with pytest.raises(ValueError):
        Economics(-1.0, 100.0)
    with pytest.raises(ValueError):
        Economics(1.0, 0.0)


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        uplift_by_depth([0.1, 0.2], [1], [0, 1])
    with pytest.raises(ValueError, match="depths"):
        uplift_by_depth([0.2, 0.1], [1, 0], [1, 0], depths=(0.5, 1.5))
    # The top half holds only treated clients: no control rate to compare with.
    with pytest.raises(ValueError, match="no treated or no control"):
        uplift_by_depth([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], [1, 0, 0, 1], depths=(0.5,))
