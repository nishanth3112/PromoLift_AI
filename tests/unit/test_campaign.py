"""Unit tests for campaign evaluation: realized against predicted uplift, on known effects."""

import numpy as np
import polars as pl
import pytest

from promolift.monitoring.campaign import (
    decile_comparison,
    evaluate_campaign,
    uplift_interval,
)
from promolift.monitoring.campaign_data import CampaignData, match_results, validate_results
from promolift.monitoring.campaign_simulation import simulate_campaign

_FAST = {"n_bootstrap": 300, "ranking_bootstrap": 10}


@pytest.fixture
def campaign(make_campaign_batch):
    """Build evaluation inputs for a simulated campaign with a chosen true effect."""
    batch = make_campaign_batch(60_000, send_share=0.5, holdout_share=0.2)

    def build(effect_scale: float) -> CampaignData:
        results = simulate_campaign(batch, effect_scale=effect_scale, seed=3)
        return match_results(batch, validate_results(results))

    return build


# --- the realized-uplift interval --------------------------------------------------------


def test_uplift_interval_is_the_rate_difference_with_a_matching_spread() -> None:
    rng = np.random.default_rng(0)

    interval = uplift_interval(600, 1_000, 400, 1_000, rng=rng, n_bootstrap=2_000, confidence=0.95)

    assert interval.estimate == pytest.approx(0.2)
    # Standard error sqrt(0.24 / 1000 * 2) = 0.0219, so +/- 1.96 SE is about +/- 0.043.
    assert interval.lower == pytest.approx(0.2 - 0.043, abs=0.01)
    assert interval.upper == pytest.approx(0.2 + 0.043, abs=0.01)


def test_a_higher_confidence_gives_a_wider_interval() -> None:
    def width(confidence: float) -> float:
        interval = uplift_interval(
            60, 100, 40, 100, rng=np.random.default_rng(0), n_bootstrap=2_000, confidence=confidence
        )
        return interval.upper - interval.lower

    assert width(0.99) > width(0.95) > width(0.8)


# --- overall comparison ---------------------------------------------------------------


def test_a_calibrated_model_is_confirmed_by_its_campaign(campaign) -> None:
    data = campaign(1.0)

    evaluation = evaluate_campaign(data, **_FAST)

    assert evaluation.scoring_run_id == "run-a"
    assert (evaluation.n_treated, evaluation.n_control) == (24_000, 6_000)
    assert evaluation.realized_uplift.estimate == pytest.approx(
        evaluation.treated_rate - evaluation.control_rate
    )
    assert evaluation.realized_uplift.estimate == pytest.approx(
        evaluation.predicted_uplift, abs=0.02
    )
    assert evaluation.prediction_within_interval
    assert evaluation.realized_uplift.lower < evaluation.realized_uplift.estimate
    assert evaluation.realized_uplift.estimate < evaluation.realized_uplift.upper


def test_an_overconfident_model_is_flagged(campaign) -> None:
    evaluation = evaluate_campaign(campaign(0.5), **_FAST)

    assert not evaluation.prediction_within_interval
    # Half the promised effect is real: realized sits well below predicted.
    assert evaluation.deviation.upper < 0
    assert evaluation.deviation.estimate == pytest.approx(
        -0.5 * evaluation.predicted_uplift, abs=0.02
    )


def test_a_campaign_whose_texts_do_nothing_shows_no_realized_uplift(campaign) -> None:
    evaluation = evaluate_campaign(campaign(0.0), **_FAST)

    assert evaluation.realized_uplift.lower < 0 < evaluation.realized_uplift.upper
    assert not evaluation.prediction_within_interval
    assert evaluation.deviation.upper < 0


def test_the_deviation_is_realized_minus_predicted(campaign) -> None:
    evaluation = evaluate_campaign(campaign(0.7), **_FAST)

    shift = evaluation.predicted_uplift
    assert evaluation.deviation.estimate == pytest.approx(
        evaluation.realized_uplift.estimate - shift
    )
    assert evaluation.deviation.lower == pytest.approx(evaluation.realized_uplift.lower - shift)
    assert evaluation.deviation.upper == pytest.approx(evaluation.realized_uplift.upper - shift)


def test_incremental_purchases_scale_the_uplift_to_the_texted_clients(campaign) -> None:
    evaluation = evaluate_campaign(campaign(1.0), **_FAST)

    assert evaluation.incremental_purchases == pytest.approx(
        evaluation.realized_uplift.estimate * evaluation.n_treated
    )


def test_the_same_seed_reproduces_every_interval(campaign) -> None:
    data = campaign(1.0)

    first = evaluate_campaign(data, seed=5, **_FAST)
    again = evaluate_campaign(data, seed=5, **_FAST)
    other = evaluate_campaign(data, seed=6, **_FAST)

    assert first.realized_uplift == again.realized_uplift
    assert first.ranking.qini_auc == again.ranking.qini_auc
    assert first.deciles.equals(again.deciles)
    assert first.realized_uplift != other.realized_uplift


def test_missing_results_are_carried_into_the_evaluation(make_campaign_batch) -> None:
    batch = make_campaign_batch(10_000, send_share=0.5, holdout_share=0.2)
    results = simulate_campaign(batch, seed=1)
    dropped = (pl.col("treated") == 1) & (pl.int_range(pl.len()) % 10 == 0)
    kept = results.filter(~dropped)
    data = match_results(batch, validate_results(kept))

    evaluation = evaluate_campaign(data, **_FAST)

    assert evaluation.n_missing_treated > 0
    assert evaluation.n_missing_control == 0
    assert evaluation.n_missing_treated == data.n_targeted_treated - evaluation.n_treated


# --- ranking ------------------------------------------------------------------------


def test_the_ranking_is_confirmed_when_the_effect_follows_the_scores(campaign) -> None:
    real = evaluate_campaign(campaign(1.0), **_FAST)
    none = evaluate_campaign(campaign(0.0), **_FAST)

    assert real.ranking.qini_auc.lower > 0
    assert real.ranking.qini_auc.estimate > none.ranking.qini_auc.estimate
    top = real.ranking.uplift_at_k[0.1]
    assert top.estimate > real.realized_uplift.estimate


def test_predicted_uplift_at_k_falls_as_more_clients_are_included(campaign) -> None:
    evaluation = evaluate_campaign(campaign(1.0), k_grid=(0.1, 0.3, 0.5, 0.9), **_FAST)

    predicted = evaluation.predicted_uplift_at_k
    assert list(predicted) == [0.1, 0.3, 0.5, 0.9]
    assert predicted[0.1] > predicted[0.3] > predicted[0.5] > predicted[0.9]
    # Dropping the lowest-scored tenth can only raise the mean.
    assert predicted[0.9] > evaluation.predicted_uplift


# --- by decile ----------------------------------------------------------------------


def test_the_deciles_cover_the_targeted_part_of_the_ranking(campaign) -> None:
    evaluation = evaluate_campaign(campaign(1.0), **_FAST)

    deciles = evaluation.deciles
    # Half the clients were selected, so deciles 1-5 were targeted.
    assert deciles["decile"].to_list() == [1, 2, 3, 4, 5]
    assert deciles["n_treated"].sum() == evaluation.n_treated
    assert deciles["n_control"].sum() == evaluation.n_control
    assert deciles["predicted_uplift"].is_sorted(descending=True)
    assert (deciles["realized_lower"] <= deciles["realized_uplift"]).all()
    assert (deciles["realized_uplift"] <= deciles["realized_upper"]).all()


def test_a_calibrated_model_is_close_to_its_prediction_in_every_decile(campaign) -> None:
    deciles = evaluate_campaign(campaign(1.0), **_FAST).deciles

    # Each decile is ~6,000 clients, so its realized uplift is noisy; require the
    # prediction to sit inside the 95% interval for most deciles, not every one.
    inside = (deciles["realized_lower"] <= deciles["predicted_uplift"]) & (
        deciles["predicted_uplift"] <= deciles["realized_upper"]
    )
    assert inside.sum() >= 4


def test_an_overconfident_model_over_promises_most_in_its_top_decile(campaign) -> None:
    deciles = evaluate_campaign(campaign(0.5), **_FAST).deciles

    # The shortfall is half the prediction, so it is largest where predictions are.
    assert deciles["deviation"][0] < deciles["deviation"][-1] < 0.01
    assert (deciles["deviation"] < 0.02).all()


def test_a_decile_without_both_arms_has_no_realized_uplift() -> None:
    frame = pl.DataFrame(
        {
            "client_id": ["a", "b", "c", "d"],
            "uplift_score": [0.2, 0.1, 0.05, 0.04],
            "decile": [1, 1, 2, 2],
            "treated": [1, 0, 1, 1],
            "purchased": [1, 0, 1, 0],
        }
    )

    table = decile_comparison(frame, rng=np.random.default_rng(0), n_bootstrap=100, confidence=0.95)

    assert table["n_control"].to_list() == [1, 0]
    assert table["realized_uplift"][0] == pytest.approx(1.0)
    assert table["realized_uplift"][1] is None
    assert table["deviation"][1] is None
    assert table["predicted_uplift"][1] == pytest.approx(0.045)
