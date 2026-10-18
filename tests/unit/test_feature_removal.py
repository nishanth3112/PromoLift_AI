"""Unit tests for refitting without features: ranking quality, age mix, and the run."""

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import pytest
from mlflow import MlflowClient

from promolift.explainability.feature_removal import (
    DEFAULT_DROPPED,
    log_feature_removal,
    run_feature_removal,
    summary,
    targeted_age_mix,
    without_columns,
)
from promolift.models.dataset import ModelFrame
from promolift.models.training import ModelSpec
from promolift.tracking.mlflow_tracking import Experiment, local_tracking_config

_SPEC = ModelSpec(
    "ct_test",
    "class_transformation",
    {"n_estimators": 60, "learning_rate": 0.1, "min_child_samples": 50},
)
_FAST = {"n_folds": 3, "n_bootstrap": 80, "depth": 0.4}


def _frame(*, age_matters: bool, n: int = 9_000, seed: int = 0) -> ModelFrame:
    """Clients where either age or ``x`` decides who the SMS persuades (30 points), not both."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(18, 80, n)
    x = rng.random(n)
    treatment = rng.integers(0, 2, n)
    persuadable = (age > 50) if age_matters else (x > 0.5)
    outcome = (rng.random(n) < 0.3 + 0.3 * treatment * persuadable).astype(int)
    features = pd.DataFrame({"age": age, "age_is_valid": 1.0, "x": x, "noise": rng.random(n)})
    return ModelFrame(
        "train+val", pl.Series([f"c{i}" for i in range(n)]), features, treatment, outcome
    )


@pytest.fixture(scope="module")
def age_driven():
    return run_feature_removal(_frame(age_matters=True), _SPEC, **_FAST)


@pytest.fixture(scope="module")
def age_irrelevant():
    return run_feature_removal(_frame(age_matters=False), _SPEC, **_FAST)


# --- dropping columns -----------------------------------------------------------------------


def test_without_columns_drops_features_and_keeps_everything_else() -> None:
    frame = _frame(age_matters=True, n=100)

    reduced = without_columns(frame, ["age", "age_is_valid"])

    assert list(reduced.features.columns) == ["x", "noise"]
    assert reduced.client_ids.equals(frame.client_ids)
    assert reduced.treatment is frame.treatment
    assert list(frame.features.columns) == ["age", "age_is_valid", "x", "noise"]


def test_dropping_a_column_that_is_not_a_feature_is_refused() -> None:
    with pytest.raises(ValueError, match=r"Cannot drop \['nope'\]"):
        without_columns(_frame(age_matters=True, n=50), ["nope"])


def test_the_default_is_age_and_the_flag_derived_from_it() -> None:
    assert DEFAULT_DROPPED == ("age", "age_is_valid")


# --- ranking quality ------------------------------------------------------------------------


def test_removing_the_feature_the_effect_depends_on_hurts_ranking(age_driven) -> None:
    qini = age_driven.comparison.qini_auc

    assert age_driven.with_features.qini_auc > age_driven.without_features.qini_auc
    assert qini.lower > 0
    assert qini.difference == pytest.approx(
        age_driven.with_features.qini_auc - age_driven.without_features.qini_auc, abs=1e-9
    )
    assert age_driven.verdict == "removing them lowers ranking quality"
    assert age_driven.comparison.uplift_at_k[0.4].lower > 0


def test_removing_an_irrelevant_feature_changes_nothing_measurable(age_irrelevant) -> None:
    qini = age_irrelevant.comparison.qini_auc

    assert qini.lower < 0 < qini.upper
    assert age_irrelevant.verdict == "no measurable difference in ranking quality"


def test_the_run_describes_what_was_compared(age_driven) -> None:
    assert age_driven.model_label == "ct_test"
    assert age_driven.dropped == ("age", "age_is_valid")
    assert (age_driven.n_clients, age_driven.n_folds, age_driven.depth) == (9_000, 3, 0.4)


def test_the_comparison_is_reproducible() -> None:
    frame = _frame(age_matters=True, n=3_000)

    first = run_feature_removal(frame, _SPEC, n_folds=3, n_bootstrap=30, depth=0.4)
    again = run_feature_removal(frame, _SPEC, n_folds=3, n_bootstrap=30, depth=0.4)

    assert first.comparison.qini_auc == again.comparison.qini_auc
    assert first.with_features == again.with_features


@pytest.mark.parametrize("depth", [0.0, 1.0, -0.1, 1.5])
def test_depth_must_be_strictly_between_zero_and_one(depth: float) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        run_feature_removal(_frame(age_matters=True, n=100), _SPEC, depth=depth)


# --- who gets texted -------------------------------------------------------------------------


def test_a_ranking_that_follows_age_texts_older_clients_more(age_driven) -> None:
    older = age_driven.age_mix.filter(pl.col("age_band") == "60 and over").row(0, named=True)
    younger = age_driven.age_mix.filter(pl.col("age_band") == "under 30").row(0, named=True)

    assert older["ratio_with"] > 1.3
    assert younger["ratio_with"] < 0.5
    # Without age the ranking has nothing tied to age left, so the mix is the population's.
    assert older["ratio_without"] == pytest.approx(1.0, abs=0.15)
    assert younger["ratio_without"] == pytest.approx(1.0, abs=0.2)


def test_the_ranking_follows_age_with_it_and_not_without(age_driven) -> None:
    assert age_driven.with_features.age_correlation > 0.4
    assert abs(age_driven.without_features.age_correlation) < 0.1


def test_the_age_mix_shares_each_sum_to_one(age_driven) -> None:
    mix = age_driven.age_mix

    for column in ("population_share", "targeted_share_with", "targeted_share_without"):
        assert mix[column].sum() == pytest.approx(1.0)


def test_age_is_reported_only_when_the_model_has_it() -> None:
    frame = without_columns(_frame(age_matters=False, n=2_000), ["age", "age_is_valid"])

    result = run_feature_removal(frame, _SPEC, dropped=["noise"], **_FAST)

    assert result.age_mix is None
    assert result.with_features.age_correlation is None
    assert "age mix" not in summary(result)


def test_targeted_age_mix_counts_the_top_scores_exactly() -> None:
    ages = pd.Series([25.0, 35.0, 50.0, 70.0, 28.0, 33.0, 52.0, 75.0, 40.0, 41.0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    mix = targeted_age_mix(scores, ages, None, depth=0.4).to_dicts()

    by_band = {row["age_band"]: row for row in mix}
    # The top four are ages 25, 35, 50, 70: one in each of four bands.
    assert by_band["under 30"]["targeted_share"] == pytest.approx(0.25)
    assert by_band["60 and over"]["targeted_share"] == pytest.approx(0.25)
    assert by_band["under 30"]["population_share"] == pytest.approx(0.2)
    assert by_band["under 30"]["ratio"] == pytest.approx(1.25)


def test_invalid_ages_form_their_own_band_and_ties_break_by_position() -> None:
    ages = pd.Series([25.0, 99.0, 25.0, 25.0])
    valid = pd.Series([1.0, 0.0, 1.0, 1.0])
    tied = np.zeros(4)

    mix = targeted_age_mix(tied, ages, valid, depth=0.5).to_dicts()

    by_band = {row["age_band"]: row for row in mix}
    # The first two rows win the tie: one valid 25-year-old and the invalid age.
    assert by_band["age not valid"]["targeted_share"] == pytest.approx(0.5)
    assert by_band["under 30"]["targeted_share"] == pytest.approx(0.5)


# --- summary and run -------------------------------------------------------------------------


def test_the_summary_reports_quality_depth_verdict_and_age(age_driven) -> None:
    text = summary(age_driven)

    assert "ct_test with and without age, age_is_valid" in text
    assert "9,000 train+val clients, 3-fold out of fold" in text
    assert "Qini AUC:" in text
    assert "uplift in the top 40%" in text
    assert "verdict on ranking quality: removing them lowers ranking quality" in text
    assert "60 and over" in text


def test_the_run_records_the_comparison(age_driven, lineage_dirs, tmp_path) -> None:
    config = local_tracking_config(tmp_path / "mlruns")

    run_id = log_feature_removal(age_driven, config=config, lineage=lineage_dirs)

    client = MlflowClient(tracking_uri=config.tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["run_type"] == "feature_removal"
    assert run.data.tags["dropped"] == "age,age_is_valid"
    assert run.data.tags["verdict"] == age_driven.verdict
    assert run.data.params["depth"] == "0.4"
    metrics = run.data.metrics
    assert metrics["qini_difference"] == pytest.approx(age_driven.comparison.qini_auc.difference)
    assert metrics["qini_difference_lower"] > 0
    assert {"uplift_at_40pct_with", "uplift_at_40pct_difference", "age_correlation_with"} <= set(
        metrics
    )
    files = {a.path for a in client.list_artifacts(run_id, "feature_removal")}
    assert files == {
        "feature_removal/summary.txt",
        "feature_removal/comparison.json",
        "feature_removal/age_mix.csv",
    }
    assert mlflow.get_experiment(run.info.experiment_id).name.endswith(Experiment.FEATURE_ABLATION)
