"""Unit tests for turning the feature table + split into model-ready arrays."""

import numpy as np
import polars as pl
import pytest

from promolift.data.split import LockedSplitError
from promolift.models.dataset import model_frame

_FEATURES = pl.DataFrame(
    {
        "client_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "age": [30.0, None, 50.0, 60.0, 25.0, 40.0],
        "age_is_valid": [True, False, True, True, True, True],
        "gender": ["F", "M", "U", "F", "M", "U"],
    }
)
_LABELS = pl.DataFrame(
    {
        "client_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "treatment_flg": [1, 0, 1, 0, 1, 0],
        "target": [1, 0, 0, 1, 1, 0],
    }
)
# val holds no "U" clients: its gender codes must still match train's.
_ASSIGNMENT = pl.DataFrame(
    {
        "client_id": ["c6", "c1", "c2", "c3", "c4", "c5"],
        "split": ["train", "train", "train", "train", "val", "val"],
    }
)


def _frame(split: str, **kwargs):
    return model_frame(_FEATURES, split, assignment=_ASSIGNMENT, labels=_LABELS, **kwargs)


def test_returns_only_the_split_clients_sorted_with_aligned_labels() -> None:
    frame = _frame("train")

    assert frame.client_ids.to_list() == ["c1", "c2", "c3", "c6"]
    np.testing.assert_array_equal(frame.treatment, [1, 0, 1, 0])
    np.testing.assert_array_equal(frame.outcome, [1, 0, 0, 0])
    assert len(frame.features) == 4


def test_excludes_client_id_from_the_features() -> None:
    assert list(_frame("train").features.columns) == ["age", "age_is_valid", "gender"]


def test_categorical_codes_are_identical_across_splits() -> None:
    train, val = _frame("train").features, _frame("val").features

    assert list(train["gender"].cat.categories) == list(val["gender"].cat.categories)
    # c4 is "F" in val; its code must equal train's code for "F" (c1).
    assert val["gender"].cat.codes.iloc[0] == train["gender"].cat.codes.iloc[0]


def test_booleans_become_numeric_and_nulls_stay_missing() -> None:
    features = _frame("train").features

    assert features["age_is_valid"].dtype.kind in "iuf"
    assert np.isnan(features["age"].iloc[1])


def test_test_split_is_locked() -> None:
    with pytest.raises(LockedSplitError):
        model_frame(
            _FEATURES,
            "test",
            assignment=_ASSIGNMENT.with_columns(pl.lit("test").alias("split")),
            labels=_LABELS,
        )


def test_final_evaluation_opens_the_test_split() -> None:
    frame = model_frame(
        _FEATURES,
        "test",
        assignment=_ASSIGNMENT.with_columns(pl.lit("test").alias("split")),
        labels=_LABELS,
        final_evaluation=True,
    )

    assert len(frame.features) == 6


def test_rejects_split_clients_without_features() -> None:
    with pytest.raises(ValueError, match="no features"):
        model_frame(
            _FEATURES.filter(pl.col("client_id") != "c1"),
            "train",
            assignment=_ASSIGNMENT,
            labels=_LABELS,
        )
