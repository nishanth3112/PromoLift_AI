"""Unit tests for segmenting explained clients and per-segment importance."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from promolift.explainability.segments import (
    AGE_NOT_VALID,
    AGE_SEGMENT,
    GENDER_SEGMENT,
    TIER_SEGMENT,
    default_segments,
    segment_importance,
    top_features,
)
from promolift.explainability.shap_values import Explanation


def _explanation(n: int = 100, *, with_age: bool = True, with_gender: bool = True) -> Explanation:
    """Clients whose first feature drives the top tier and whose second drives the rest."""
    scores = np.linspace(0.2, -0.05, n)
    top = np.arange(n) < n // 10
    contributions = pd.DataFrame(
        {"driver_top": np.where(top, scores, 0.001), "driver_rest": np.where(top, 0.001, scores)}
    )
    features = pd.DataFrame({"x": np.arange(n, dtype=float)})
    if with_gender:
        features["gender"] = pd.Categorical(["F", "M"] * (n // 2), categories=["F", "M", "U"])
    if with_age:
        features["age"] = np.tile([25.0, 35.0, 50.0, 70.0], n // 4)
        features["age_is_valid"] = 1.0
    return Explanation(features, contributions, base_value=0.0, scores=scores)


# --- segments ---------------------------------------------------------------------------


def test_tiers_split_the_clients_by_predicted_uplift() -> None:
    explanation = _explanation(100)

    tiers = default_segments(explanation)[TIER_SEGMENT]

    assert tiers.value_counts().to_dict() == {"top 10%": 10, "next 30%": 30, "rest": 60}
    # Scores fall with the row number, so the first rows are the top tier.
    assert set(tiers.iloc[:10]) == {"top 10%"}
    assert set(tiers.iloc[40:]) == {"rest"}


def test_segments_are_aligned_to_the_explained_clients() -> None:
    explanation = _explanation(40)

    for labels in default_segments(explanation).values():
        assert labels.index.equals(explanation.features.index)


def test_age_bands_and_invalid_ages() -> None:
    explanation = _explanation(8)
    explanation.features.loc[0, "age_is_valid"] = 0.0
    explanation.features.loc[1, "age"] = np.nan

    bands = default_segments(explanation)[AGE_SEGMENT]

    assert bands.iloc[0] == AGE_NOT_VALID
    assert bands.iloc[1] == AGE_NOT_VALID
    assert bands.iloc[2:6].tolist() == ["45-59", "60 and over", "under 30", "30-44"]


def test_gender_segments_use_the_levels_present() -> None:
    segments = default_segments(_explanation(20))

    assert set(segments[GENDER_SEGMENT]) == {"F", "M"}


def test_only_the_applicable_groupings_are_produced() -> None:
    bare = default_segments(_explanation(20, with_age=False, with_gender=False))

    assert list(bare) == [TIER_SEGMENT]


# --- importance within segments ---------------------------------------------------------


def test_the_top_tier_is_driven_by_its_own_feature() -> None:
    explanation = _explanation(100)

    table = segment_importance(explanation, default_segments(explanation))

    top = table.filter((table["segment_type"] == TIER_SEGMENT) & (table["segment"] == "top 10%"))
    rest = table.filter((table["segment_type"] == TIER_SEGMENT) & (table["segment"] == "rest"))
    assert top.sort("mean_abs_contribution", descending=True)["feature"][0] == "driver_top"
    assert rest.sort("mean_abs_contribution", descending=True)["feature"][0] == "driver_rest"
    assert set(top["n"]) == {10}
    assert top["mean_score"][0] > rest["mean_score"][0]


def test_segments_are_listed_in_their_natural_order() -> None:
    explanation = _explanation(100)

    table = segment_importance(explanation, default_segments(explanation))

    def listed(segment_type: str) -> list[str]:
        return (
            table.filter(table["segment_type"] == segment_type)["segment"]
            .unique(maintain_order=True)
            .to_list()
        )

    assert listed(TIER_SEGMENT) == ["top 10%", "next 30%", "rest"]
    assert listed(AGE_SEGMENT) == ["under 30", "30-44", "45-59", "60 and over"]
    assert listed(GENDER_SEGMENT) == ["F", "M"]


def test_shares_sum_to_one_within_every_segment() -> None:
    explanation = _explanation(100)

    table = segment_importance(explanation, default_segments(explanation))

    sums = table.group_by(["segment_type", "segment"]).agg(pl.col("share").sum())["share"]
    assert sums.to_list() == pytest.approx([1.0] * len(sums))


def test_each_segment_has_a_row_per_feature() -> None:
    explanation = _explanation(100)

    table = segment_importance(explanation, default_segments(explanation))

    counts = table.group_by(["segment_type", "segment"]).len()["len"]
    assert (counts == len(explanation.contributions.columns)).all()


def test_misaligned_segments_are_rejected() -> None:
    explanation = _explanation(20)
    misaligned = pd.Series(["a"] * 20, index=range(100, 120))

    with pytest.raises(ValueError, match="not aligned"):
        segment_importance(explanation, {"bad": misaligned})


def test_top_features_names_the_leading_features_with_shares() -> None:
    explanation = _explanation(100)
    table = segment_importance(
        explanation, {TIER_SEGMENT: default_segments(explanation)[TIER_SEGMENT]}
    )

    summary = top_features(table, k=1)

    top = summary.filter(summary["segment"] == "top 10%")
    assert top["top_features"][0].startswith("driver_top (")
    assert top["top_features"][0].endswith("%)")
    assert summary.height == 3
