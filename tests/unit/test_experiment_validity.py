"""Unit tests for causal experiment validity checks, using synthetic fixtures."""

import math
from pathlib import Path

import pytest

from promolift.validation.experiment_validity import (
    categorical_covariate_balance,
    numeric_covariate_balance,
    outcome_distribution,
    raw_ate_sanity_check,
)


@pytest.fixture
def balanced_experiment(tmp_path: Path) -> Path:
    # 4 clients: c1/c2 treatment, c3/c4 control. Ages identical across arms
    # (perfect balance), gender identical across arms too.
    (tmp_path / "clients.csv").write_text(
        "client_id,age,gender\nc1,30,F\nc2,40,M\nc3,30,F\nc4,40,M\n"
    )
    (tmp_path / "uplift_train.csv").write_text(
        "client_id,treatment_flg,target\nc1,1,1\nc2,1,0\nc3,0,0\nc4,0,0\n"
    )
    return tmp_path


def test_numeric_covariate_balance_is_zero_for_identical_distributions(
    balanced_experiment: Path,
) -> None:
    result = numeric_covariate_balance("age", balanced_experiment)

    assert result.treatment_mean == pytest.approx(35.0)
    assert result.control_mean == pytest.approx(35.0)
    assert result.standardized_mean_diff == pytest.approx(0.0)
    assert result.is_balanced is True


def test_numeric_covariate_balance_detects_imbalance(tmp_path: Path) -> None:
    (tmp_path / "clients.csv").write_text(
        "client_id,age,gender\nc1,60,F\nc2,60,F\nc3,20,F\nc4,20,F\n"
    )
    (tmp_path / "uplift_train.csv").write_text(
        "client_id,treatment_flg,target\nc1,1,1\nc2,1,0\nc3,0,0\nc4,0,0\n"
    )

    result = numeric_covariate_balance("age", tmp_path)

    assert result.treatment_mean == pytest.approx(60.0)
    assert result.control_mean == pytest.approx(20.0)
    assert result.is_balanced is False


def test_categorical_covariate_balance_is_zero_for_identical_distributions(
    balanced_experiment: Path,
) -> None:
    result = categorical_covariate_balance("gender", balanced_experiment)

    assert result.treatment_proportions == pytest.approx({"F": 0.5, "M": 0.5})
    assert result.control_proportions == pytest.approx({"F": 0.5, "M": 0.5})
    assert result.max_proportion_diff == pytest.approx(0.0)
    assert result.is_balanced is True


def test_categorical_covariate_balance_detects_imbalance(tmp_path: Path) -> None:
    (tmp_path / "clients.csv").write_text(
        "client_id,age,gender\nc1,30,F\nc2,30,F\nc3,30,M\nc4,30,M\n"
    )
    (tmp_path / "uplift_train.csv").write_text(
        "client_id,treatment_flg,target\nc1,1,1\nc2,1,0\nc3,0,0\nc4,0,0\n"
    )

    result = categorical_covariate_balance("gender", tmp_path)

    assert result.treatment_proportions == pytest.approx({"F": 1.0})
    assert result.control_proportions == pytest.approx({"M": 1.0})
    assert result.max_proportion_diff == pytest.approx(1.0)
    assert result.is_balanced is False


def test_outcome_distribution_computes_rates_per_arm(balanced_experiment: Path) -> None:
    result = outcome_distribution(balanced_experiment)

    assert result.treatment_count == 2
    assert result.control_count == 2
    assert result.treatment_rate == pytest.approx(0.5)
    assert result.control_rate == pytest.approx(0.0)
    assert result.overall_rate == pytest.approx(0.25)


def test_raw_ate_sanity_check_computes_difference_in_means(balanced_experiment: Path) -> None:
    result = raw_ate_sanity_check(balanced_experiment, confidence=0.95)

    assert result.ate == pytest.approx(0.5)
    assert result.treatment_rate == pytest.approx(0.5)
    assert result.control_rate == pytest.approx(0.0)
    # Hand-computed SE for p1=0.5,n1=2 and p0=0.0,n0=2: sqrt(0.25/2 + 0/2)
    assert result.standard_error == pytest.approx(math.sqrt(0.25 / 2))
    assert result.ci_lower < result.ate < result.ci_upper
