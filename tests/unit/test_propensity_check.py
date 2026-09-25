"""Unit tests for the propensity-AUC and overlap check.

ML-based checks are inherently a bit stochastic even with fixed seeds
(library version differences, etc.), so these use a strong, unambiguous
signal for the "predictable" case and a generous tolerance band for the
"near-chance" case, rather than asserting exact values.
"""

import numpy as np
import polars as pl

from promolift.validation.experiment_validity import propensity_check


def test_detects_treatment_that_is_fully_predictable_from_covariates() -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    # Treatment is a deterministic function of x -- as predictable as it gets.
    treatment = (x > 0).astype(int)

    covariates = pl.DataFrame({"x": x})
    report = propensity_check(covariates, pl.Series(treatment))

    assert report.auc > 0.9


def test_reports_near_chance_auc_when_treatment_is_unrelated_to_covariates() -> None:
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(size=n)
    treatment = rng.integers(0, 2, size=n)  # independent of x

    covariates = pl.DataFrame({"x": x})
    report = propensity_check(covariates, pl.Series(treatment))

    assert 0.3 < report.auc < 0.7


def test_overlap_range_is_none_when_propensity_scores_are_fully_separated() -> None:
    rng = np.random.default_rng(2)
    n = 300
    # Two well-separated clusters, perfectly aligned with treatment -- the
    # classifier should assign near-0 propensity to one arm and near-1 to
    # the other, leaving no overlap.
    x = np.concatenate(
        [rng.normal(loc=-10, scale=0.1, size=n // 2), rng.normal(loc=10, scale=0.1, size=n // 2)]
    )
    treatment = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)

    covariates = pl.DataFrame({"x": x})
    report = propensity_check(covariates, pl.Series(treatment))

    assert report.overlap_range is None


def test_handles_mixed_numeric_and_categorical_covariates() -> None:
    rng = np.random.default_rng(3)
    n = 200
    age = rng.normal(loc=40, scale=10, size=n)
    gender = rng.choice(["F", "M", "U"], size=n)
    treatment = rng.integers(0, 2, size=n)

    covariates = pl.DataFrame({"age": age, "gender": gender})
    report = propensity_check(covariates, pl.Series(treatment))

    assert 0.0 <= report.auc <= 1.0
