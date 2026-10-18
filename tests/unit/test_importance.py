"""Unit tests for global importance and dependence tables."""

import warnings

import numpy as np
import pandas as pd
import pytest

from promolift.explainability.importance import (
    DIRECTION_THRESHOLD,
    MISSING_LABEL,
    UNUSED,
    dependence_table,
    global_importance,
)
from promolift.explainability.shap_values import Explanation, explain


@pytest.fixture(scope="module")
def explanation(explained_model) -> Explanation:
    return explain(explained_model.model, explained_model.features)


def _handmade(features: dict, contributions: dict) -> Explanation:
    """An explanation with contributions we chose, to test the summaries exactly."""
    frame = pd.DataFrame(features)
    contrib = pd.DataFrame(contributions)
    return Explanation(frame, contrib, base_value=0.0, scores=contrib.sum(axis=1).to_numpy())


# --- global importance ------------------------------------------------------------------


def test_features_are_ranked_by_mean_absolute_contribution(explanation) -> None:
    table = global_importance(explanation)

    assert table["feature"][0] == "b"
    assert table["rank"].to_list() == [1, 2, 3]
    assert table["mean_abs_contribution"].is_sorted(descending=True)
    assert table["share"].sum() == pytest.approx(1.0)
    expected = explanation.contributions["b"].abs().mean()
    assert table.filter(table["feature"] == "b")["mean_abs_contribution"][0] == pytest.approx(
        expected
    )


def test_a_feature_that_lowers_uplift_as_it_rises_is_labelled_so() -> None:
    values = np.arange(100, dtype=float)
    explanation = _handmade(
        {"falls": values, "rises": values, "flat": np.zeros(100)},
        {"falls": -values / 100, "rises": values / 100, "flat": np.zeros(100)},
    )

    rows = {r["feature"]: r for r in global_importance(explanation).iter_rows(named=True)}

    assert rows["falls"]["direction"] == "higher value, lower uplift"
    assert rows["falls"]["direction_corr"] == pytest.approx(-1.0)
    assert rows["rises"]["direction"] == "higher value, higher uplift"
    assert rows["flat"]["direction"] == UNUSED
    assert rows["flat"]["direction_corr"] is None


def test_a_feature_the_model_never_uses_is_reported_without_a_warning() -> None:
    explanation = _handmade(
        {"used": np.arange(50, dtype=float), "ignored": np.arange(50, dtype=float)},
        {"used": np.linspace(-1, 1, 50), "ignored": np.zeros(50)},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a constant-input correlation warning would fail here
        rows = {r["feature"]: r for r in global_importance(explanation).iter_rows(named=True)}

    assert rows["ignored"]["direction"] == UNUSED
    assert rows["ignored"]["mean_abs_contribution"] == 0
    assert rows["ignored"]["share"] == 0
    assert rows["used"]["direction"] == "higher value, higher uplift"


def test_a_feature_with_one_value_everywhere_is_constant() -> None:
    explanation = _handmade({"x": np.ones(20)}, {"x": np.linspace(-1, 1, 20)})

    (row,) = global_importance(explanation).iter_rows(named=True)

    assert row["direction"] == "constant"


def test_a_feature_acting_in_both_directions_is_mixed() -> None:
    values = np.arange(100, dtype=float)
    u_shape = ((values - 49.5) / 50) ** 2
    explanation = _handmade({"u": values}, {"u": u_shape})

    (row,) = global_importance(explanation).iter_rows(named=True)

    assert abs(row["direction_corr"]) < DIRECTION_THRESHOLD
    assert row["direction"] == "mixed"


def test_categorical_features_have_no_direction(explanation) -> None:
    rows = {r["feature"]: r for r in global_importance(explanation).iter_rows(named=True)}

    assert rows["segment"]["direction"] == "categorical"
    assert rows["segment"]["direction_corr"] is None


def test_missing_values_are_reported_with_their_own_effect() -> None:
    values = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
    explanation = _handmade({"x": values}, {"x": [0.1, 0.2, -0.3, -0.5, 0.3, 0.4]})

    (row,) = global_importance(explanation).iter_rows(named=True)

    assert row["missing_share"] == pytest.approx(2 / 6)
    assert row["mean_when_missing"] == pytest.approx(-0.4)


def test_no_missing_values_means_no_missing_effect(explanation) -> None:
    table = global_importance(explanation)

    assert (table["missing_share"] == 0).all()
    assert table["mean_when_missing"].is_null().all()


# --- dependence -------------------------------------------------------------------------


def test_dependence_follows_the_feature_from_low_to_high(explanation) -> None:
    table = dependence_table(explanation, "b", bins=5)

    assert table.height == 5
    assert table["n"].sum() == len(explanation.features)
    assert table["mean_value"].is_sorted()
    # Low b = persuadables (positive), high b = sure things (negative).
    assert table["mean_contribution"][0] > 0 > table["mean_contribution"][-1]


def test_categorical_dependence_has_a_row_per_level(explanation) -> None:
    table = dependence_table(explanation, "segment")

    assert sorted(table["label"].to_list()) == ["F", "M", "U"]
    assert table["mean_value"].is_null().all()
    assert table["n"].sum() == len(explanation.features)


def test_few_distinct_values_are_shown_value_by_value() -> None:
    explanation = _handmade(
        {"flag": [0.0, 1.0, 1.0, 0.0, 1.0]}, {"flag": [-0.1, 0.2, 0.2, -0.1, 0.2]}
    )

    table = dependence_table(explanation, "flag")

    assert table["label"].to_list() == ["0", "1"]
    assert table["n"].to_list() == [2, 3]
    assert table["mean_contribution"].to_list() == pytest.approx([-0.1, 0.2])


def test_missing_clients_form_their_own_last_row() -> None:
    values = np.array([1.0, 2.0, np.nan, np.nan, 3.0])
    explanation = _handmade({"x": values}, {"x": [0.1, 0.2, -0.4, -0.2, 0.3]})

    table = dependence_table(explanation, "x")

    assert table["label"][-1] == MISSING_LABEL
    assert table["n"][-1] == 2
    assert table["mean_contribution"][-1] == pytest.approx(-0.3)
    assert table["mean_value"][-1] is None


def test_dependence_rejects_an_unknown_feature(explanation) -> None:
    with pytest.raises(KeyError, match="not an explained feature"):
        dependence_table(explanation, "nope")
