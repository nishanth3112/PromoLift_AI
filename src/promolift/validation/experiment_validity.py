"""Causal experiment validity checks for the RetailHero randomized experiment.

Checks whether the treatment/control split in ``uplift_train.csv`` behaves
like a genuine randomization (covariate balance) and whether the observed
outcome/treatment effect is a plausible, statistically real signal, before
any causal model is built on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import copysign, inf, sqrt
from pathlib import Path
from statistics import NormalDist

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from promolift.data.loader import Dataset, load_lazy

_DEFAULT_BALANCE_THRESHOLD = 0.1
_TREATMENT = 1
_CONTROL = 0


@dataclass
class NumericCovariateBalance:
    """Standardized mean difference for a numeric covariate between arms."""

    covariate: str
    treatment_mean: float
    control_mean: float
    standardized_mean_diff: float
    is_balanced: bool


@dataclass
class CategoricalCovariateBalance:
    """Per-category proportion comparison for a categorical covariate."""

    covariate: str
    treatment_proportions: dict[str, float]
    control_proportions: dict[str, float]
    max_proportion_diff: float
    is_balanced: bool


@dataclass
class OutcomeDistributionReport:
    """Conversion rate overall and by treatment arm."""

    overall_rate: float
    treatment_rate: float
    control_rate: float
    treatment_count: int
    control_count: int


@dataclass
class ATESanityCheckReport:
    """Naive difference-in-means ATE estimate with a normal-approximation CI."""

    ate: float
    standard_error: float
    confidence: float
    ci_lower: float
    ci_upper: float
    treatment_rate: float
    control_rate: float
    treatment_count: int
    control_count: int


def _uplift_train_with_covariate(covariate: str, base_dir: Path | None = None) -> pl.LazyFrame:
    return load_lazy(Dataset.UPLIFT_TRAIN, base_dir).join(
        load_lazy(Dataset.CLIENTS, base_dir).select("client_id", covariate),
        on="client_id",
        how="inner",
    )


def numeric_covariate_balance(
    covariate: str,
    base_dir: Path | None = None,
    *,
    balance_threshold: float = _DEFAULT_BALANCE_THRESHOLD,
) -> NumericCovariateBalance:
    """Standardized mean difference (SMD) of a numeric covariate across arms.

    A well-randomized experiment should show |SMD| well under
    ``balance_threshold`` (0.1 is the common rule of thumb).
    """
    stats = (
        _uplift_train_with_covariate(covariate, base_dir)
        .group_by("treatment_flg")
        .agg(pl.col(covariate).mean().alias("mean"), pl.col(covariate).var().alias("var"))
        .collect()
    )
    by_arm = {row["treatment_flg"]: row for row in stats.to_dicts()}
    treatment, control = by_arm[_TREATMENT], by_arm[_CONTROL]

    mean_diff = treatment["mean"] - control["mean"]
    pooled_std = sqrt((treatment["var"] + control["var"]) / 2)
    if pooled_std:
        smd = mean_diff / pooled_std
    else:
        # Zero within-arm variance but a nonzero mean gap is maximal
        # imbalance (perfect separation), not "no imbalance" -- 0.0 would be
        # misleading here.
        smd = 0.0 if mean_diff == 0 else copysign(inf, mean_diff)

    return NumericCovariateBalance(
        covariate=covariate,
        treatment_mean=treatment["mean"],
        control_mean=control["mean"],
        standardized_mean_diff=smd,
        is_balanced=abs(smd) < balance_threshold,
    )


def categorical_covariate_balance(
    covariate: str,
    base_dir: Path | None = None,
    *,
    balance_threshold: float = _DEFAULT_BALANCE_THRESHOLD,
) -> CategoricalCovariateBalance:
    """Per-category proportion difference of a categorical covariate across arms."""
    counts = (
        _uplift_train_with_covariate(covariate, base_dir)
        .group_by(["treatment_flg", covariate])
        .agg(pl.len().alias("count"))
        .collect()
    )
    totals = counts.group_by("treatment_flg").agg(pl.col("count").sum().alias("total"))
    proportions = counts.join(totals, on="treatment_flg").with_columns(
        (pl.col("count") / pl.col("total")).alias("proportion")
    )

    treatment_props = {
        row[covariate]: row["proportion"]
        for row in proportions.filter(pl.col("treatment_flg") == _TREATMENT).to_dicts()
    }
    control_props = {
        row[covariate]: row["proportion"]
        for row in proportions.filter(pl.col("treatment_flg") == _CONTROL).to_dicts()
    }

    categories = set(treatment_props) | set(control_props)
    max_diff = max(abs(treatment_props.get(c, 0.0) - control_props.get(c, 0.0)) for c in categories)

    return CategoricalCovariateBalance(
        covariate=covariate,
        treatment_proportions=treatment_props,
        control_proportions=control_props,
        max_proportion_diff=max_diff,
        is_balanced=max_diff < balance_threshold,
    )


def outcome_distribution(base_dir: Path | None = None) -> OutcomeDistributionReport:
    """Conversion rate (``target`` mean) overall and per treatment arm."""
    lf = load_lazy(Dataset.UPLIFT_TRAIN, base_dir)
    overall_rate = lf.select(pl.col("target").mean()).collect().item()

    by_arm = (
        lf.group_by("treatment_flg")
        .agg(pl.col("target").mean().alias("rate"), pl.len().alias("count"))
        .collect()
    )
    rows = {row["treatment_flg"]: row for row in by_arm.to_dicts()}
    treatment, control = rows[_TREATMENT], rows[_CONTROL]

    return OutcomeDistributionReport(
        overall_rate=overall_rate,
        treatment_rate=treatment["rate"],
        control_rate=control["rate"],
        treatment_count=treatment["count"],
        control_count=control["count"],
    )


def raw_ate_sanity_check(
    base_dir: Path | None = None, *, confidence: float = 0.95
) -> ATESanityCheckReport:
    """Naive difference-in-means ATE with a normal-approximation confidence interval.

    This is a sanity check, not a rigorous causal estimate: it only confirms
    the randomized experiment shows a plausible, non-degenerate treatment
    effect before any causal model is built on top of it.
    """
    stats = (
        load_lazy(Dataset.UPLIFT_TRAIN, base_dir)
        .group_by("treatment_flg")
        .agg(pl.col("target").mean().alias("rate"), pl.len().alias("count"))
        .collect()
    )
    by_arm = {row["treatment_flg"]: row for row in stats.to_dicts()}
    treatment, control = by_arm[_TREATMENT], by_arm[_CONTROL]

    ate = treatment["rate"] - control["rate"]
    standard_error = sqrt(
        treatment["rate"] * (1 - treatment["rate"]) / treatment["count"]
        + control["rate"] * (1 - control["rate"]) / control["count"]
    )
    z = NormalDist().inv_cdf(0.5 + confidence / 2)

    return ATESanityCheckReport(
        ate=ate,
        standard_error=standard_error,
        confidence=confidence,
        ci_lower=ate - z * standard_error,
        ci_upper=ate + z * standard_error,
        treatment_rate=treatment["rate"],
        control_rate=control["rate"],
        treatment_count=treatment["count"],
        control_count=control["count"],
    )


@dataclass
class PropensityCheckReport:
    """Result of checking whether treatment is predictable from covariates."""

    auc: float
    treatment_propensity_range: tuple[float, float]
    control_propensity_range: tuple[float, float]
    overlap_range: tuple[float, float] | None


def propensity_check(
    covariates: pl.DataFrame,
    treatment: pl.Series,
    *,
    n_folds: int = 5,
    tail_percentile: float = 0.01,
    random_state: int = 42,
) -> PropensityCheckReport:
    """Check whether treatment assignment is predictable from covariates.

    Under true randomization, no combination of pre-treatment covariates
    should predict treatment above chance -- fits a LightGBM classifier and
    reports the cross-validated (out-of-fold) AUC, which should be close to
    0.5. Also reports the ``[tail_percentile, 1 - tail_percentile]`` range of
    predicted propensity scores per arm and their overlap: positivity (every
    unit has a realistic chance of either arm) is a core assumption for
    causal identification, and a shrinking or empty overlap range is a red
    flag before any causal model is trusted.

    Args:
        covariates: One row per unit. String columns are treated as
            categorical by the underlying classifier; nulls are allowed
            (handled natively by LightGBM). Caller decides which covariates
            to check -- e.g. just demographics, or a full engineered feature
            table.
        treatment: Binary treatment indicator, same row order as covariates.
        n_folds: Number of cross-validation folds for out-of-fold predictions.
        tail_percentile: Tail percentile for the propensity-score ranges.
        random_state: Seed for reproducibility.

    Returns:
        overlap_range is None if the treatment and control propensity-score
        ranges don't overlap at all.
    """
    pdf = covariates.to_pandas()
    for col in pdf.columns:
        if pdf[col].dtype == object:
            pdf[col] = pdf[col].astype("category")

    y = treatment.to_numpy()
    classifier = lgb.LGBMClassifier(n_estimators=200, random_state=random_state, verbose=-1)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_scores = cross_val_predict(classifier, pdf, y, cv=cv, method="predict_proba")[:, 1]

    auc = roc_auc_score(y, oof_scores)

    lo, hi = tail_percentile, 1 - tail_percentile
    treatment_scores = oof_scores[y == _TREATMENT]
    control_scores = oof_scores[y == _CONTROL]
    treatment_range = (
        float(np.quantile(treatment_scores, lo)),
        float(np.quantile(treatment_scores, hi)),
    )
    control_range = (
        float(np.quantile(control_scores, lo)),
        float(np.quantile(control_scores, hi)),
    )

    overlap_lo = max(treatment_range[0], control_range[0])
    overlap_hi = min(treatment_range[1], control_range[1])
    overlap_range = (overlap_lo, overlap_hi) if overlap_lo < overlap_hi else None

    return PropensityCheckReport(
        auc=auc,
        treatment_propensity_range=treatment_range,
        control_propensity_range=control_range,
        overlap_range=overlap_range,
    )
