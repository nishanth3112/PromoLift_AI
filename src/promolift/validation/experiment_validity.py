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

import polars as pl

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
