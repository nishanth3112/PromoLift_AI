"""Unit tests for the simulated campaigns used to check the evaluation."""

import polars as pl
import pytest

from promolift.monitoring.campaign_data import RESULT_COLUMNS, match_results, validate_results
from promolift.monitoring.campaign_simulation import simulate_campaign


def test_simulates_a_result_for_every_targeted_client_in_the_batch_s_arms(
    make_campaign_batch,
) -> None:
    batch = make_campaign_batch(1_000, send_share=0.5, holdout_share=0.2)

    results = simulate_campaign(batch, seed=1)

    assert results.columns == list(RESULT_COLUMNS)
    assert results.height == 500
    assert results["treated"].sum() == 400
    assert set(results["scoring_run_id"]) == {"run-a"}
    # The output is exactly what the evaluation accepts.
    assert match_results(batch, validate_results(results)).frame.height == 500


def test_the_same_seed_gives_the_same_campaign_and_a_new_seed_a_new_one(
    make_campaign_batch,
) -> None:
    batch = make_campaign_batch(1_000)

    assert simulate_campaign(batch, seed=1).equals(simulate_campaign(batch, seed=1))
    assert not simulate_campaign(batch, seed=1).equals(simulate_campaign(batch, seed=2))


def test_texting_lifts_purchases_by_the_scaled_prediction(make_campaign_batch) -> None:
    batch = make_campaign_batch(40_000, send_share=0.5, holdout_share=0.25)
    targeted = batch.filter(pl.col("send") | pl.col("holdout"))
    predicted = targeted["uplift_score"].mean()

    results = simulate_campaign(batch, effect_scale=1.0, baseline_rate=0.5, seed=3)

    rates = results.group_by("treated").agg(pl.col("purchased").mean()).sort("treated")
    control, treated = rates["purchased"]
    assert control == pytest.approx(0.5, abs=0.015)
    assert treated - control == pytest.approx(predicted, abs=0.02)


def test_a_zero_effect_scale_means_texting_changes_nothing(make_campaign_batch) -> None:
    batch = make_campaign_batch(40_000, send_share=0.5, holdout_share=0.25)

    results = simulate_campaign(batch, effect_scale=0.0, baseline_rate=0.5, seed=3)

    rates = results.group_by("treated").agg(pl.col("purchased").mean()).sort("treated")
    control, treated = rates["purchased"]
    assert treated - control == pytest.approx(0.0, abs=0.015)


def test_rejects_a_bad_baseline_and_a_batch_of_several_runs(make_campaign_batch) -> None:
    batch = make_campaign_batch(200)

    with pytest.raises(ValueError, match="baseline_rate"):
        simulate_campaign(batch, baseline_rate=1.5)
    with pytest.raises(ValueError, match="one scoring run"):
        simulate_campaign(pl.concat([batch, make_campaign_batch(200, run_id="run-b")]))
