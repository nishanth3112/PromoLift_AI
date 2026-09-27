"""Unit tests for the train-fitted numeric encoder used by models that can't take NaN."""

import numpy as np
import pandas as pd
import pytest

from promolift.models.preprocessing import NumericEncoder

_CATEGORIES = ["F", "M", "U"]


def _features(age: list, redeem: list, gender: list) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": age,
            "days_to_redeem": redeem,
            "has_redeemed": np.array([1, 0, 1, 1][: len(age)], dtype="int8"),
            "gender": pd.Categorical(gender, categories=_CATEGORIES),
        }
    )


@pytest.fixture
def train() -> pd.DataFrame:
    # days_to_redeem has nulls in train (gets a flag); age doesn't (no flag).
    return _features([20.0, 30.0, 40.0, 50.0], [5.0, np.nan, 15.0, 25.0], ["F", "M", "F", "U"])


@pytest.fixture
def val() -> pd.DataFrame:
    # val has an age null train never had, and no "U" clients.
    return _features([np.nan, 90.0], [np.nan, 1.0], ["M", "F"])


def test_imputes_val_with_train_medians_not_its_own(train, val) -> None:
    encoder = NumericEncoder().fit(train)

    encoded = encoder.transform(val)

    names = encoder.feature_names
    assert encoded[0, names.index("age")] == pytest.approx(35.0)  # train median, not val's 90
    assert encoded[0, names.index("days_to_redeem")] == pytest.approx(15.0)


def test_adds_missing_flags_only_for_columns_with_nulls_in_train(train, val) -> None:
    encoder = NumericEncoder().fit(train)
    encoded = encoder.transform(val)
    names = encoder.feature_names

    assert "days_to_redeem_missing" in names
    assert "age_missing" not in names
    np.testing.assert_array_equal(encoded[:, names.index("days_to_redeem_missing")], [1.0, 0.0])


def test_one_hot_columns_cover_every_category_in_every_split(train, val) -> None:
    encoder = NumericEncoder().fit(train)
    encoded = encoder.transform(val)
    names = encoder.feature_names

    assert [n for n in names if n.startswith("gender_")] == ["gender_F", "gender_M", "gender_U"]
    np.testing.assert_array_equal(encoded[:, names.index("gender_U")], [0.0, 0.0])
    np.testing.assert_array_equal(encoded[:, names.index("gender_M")], [1.0, 0.0])


def test_every_split_gets_the_same_finite_columns(train, val) -> None:
    encoder = NumericEncoder().fit(train)

    train_matrix, val_matrix = encoder.transform(train), encoder.transform(val)

    assert train_matrix.shape[1] == val_matrix.shape[1] == len(encoder.feature_names)
    assert np.isfinite(train_matrix).all()
    assert np.isfinite(val_matrix).all()


def test_transform_before_fit_is_an_error(val) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        NumericEncoder().transform(val)


def test_rejects_columns_it_was_not_fitted_on(train, val) -> None:
    encoder = NumericEncoder().fit(train)

    with pytest.raises(ValueError, match="columns"):
        encoder.transform(val.drop(columns=["age"]))
