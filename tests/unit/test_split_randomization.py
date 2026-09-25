"""Unit tests for the per-split randomization check, using small synthetic frames."""

import polars as pl
import pytest

from promolift.validation.experiment_validity import split_randomization_check


def _frame(test_split_ages: tuple[int, int] = (30, 30)) -> pl.DataFrame:
    # Two splits, each with 2 treated + 2 control clients. Ages and genders are
    # identical across arms, except the test split's treated ages can be skewed.
    treated_age, control_age = test_split_ages
    return pl.DataFrame(
        {
            "split": ["train"] * 4 + ["test"] * 4,
            "treatment_flg": [1, 1, 0, 0] * 2,
            "target": [1, 1, 1, 0, 1, 0, 0, 0],
            "age": [30, 40, 30, 40, treated_age, treated_age, control_age, control_age],
            "gender": ["F", "M", "F", "M"] * 2,
        }
    )


def test_reports_one_entry_per_split_with_its_own_ate() -> None:
    reports = {r.split: r for r in split_randomization_check(_frame())}

    assert set(reports) == {"train", "test"}
    # train: treated 2/2 vs control 1/2; test: treated 1/2 vs control 0/2.
    assert reports["train"].ate.ate == pytest.approx(0.5)
    assert reports["test"].ate.ate == pytest.approx(0.5)
    assert reports["train"].ate.treatment_count == 2


def test_balanced_splits_pass() -> None:
    reports = split_randomization_check(_frame())

    assert all(r.is_balanced for r in reports)


def test_detects_imbalance_confined_to_one_split() -> None:
    reports = {r.split: r for r in split_randomization_check(_frame(test_split_ages=(70, 20)))}

    assert reports["train"].is_balanced is True
    assert reports["test"].is_balanced is False
    age_balance = next(b for b in reports["test"].numeric_balance if b.covariate == "age")
    assert age_balance.is_balanced is False


def test_checks_only_the_requested_covariates() -> None:
    reports = split_randomization_check(
        _frame(), numeric_covariates=["age"], categorical_covariates=[]
    )

    assert [b.covariate for b in reports[0].numeric_balance] == ["age"]
    assert reports[0].categorical_balance == []
