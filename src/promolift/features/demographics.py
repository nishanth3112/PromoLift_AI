"""Client demographic features, built from clients.csv only.

Two known data-quality issues, both confirmed against the real data (cross-
validated independently against a teammate's separate DuckDB-based pipeline,
exact counts matched):

- Age: 1,404 of 400,162 clients have an implausible age (e.g. -7491, 1901,
  or an implausible-for-a-shopper value like 5 or 110). Rather than clip or
  impute a fake value, implausible ages are nulled out and flagged.
- Redemption date leakage: ``first_redeem_date`` extends to 2019-11-20, but
  ``purchases.csv`` (and therefore the campaign/treatment window) ends
  2019-03-18. 44,953 clients redeemed *after* the treatment window --
  treating that as "has redeemed" would leak post-treatment information into
  a feature. Redemptions on/after ``reference_date`` are therefore treated
  as not-yet-redeemed. Separately, 536 clients have a redeem date before
  their issue date (a data-entry inconsistency); ``days_to_redeem`` is
  nulled for those rather than reporting a negative duration.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from promolift.data.loader import Dataset, load_lazy

# Tighter than data_contracts.x5_raw_schemas.ClientsSchema's [0, 120] bound --
# a retail loyalty program shopper outside [10, 100] is implausible either way.
_MIN_PLAUSIBLE_AGE = 10
_MAX_PLAUSIBLE_AGE = 100
_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def build_demographic_features(
    reference_date: datetime, base_dir: Path | None = None
) -> pl.LazyFrame:
    """Build client-level demographic features from clients.csv.

    Args:
        reference_date: Fixed point in time used to compute tenure and to
            decide whether a redemption counts as pre-treatment. Should be
            the same reference date used across all feature builders (e.g.
            the max transaction date in purchases.csv) so features are
            mutually consistent and free of post-treatment leakage.
        base_dir: Optional override of the raw data directory (for tests).

    Returns one row per client with:
        - ``age``: null if the raw value is outside [10, 100] (see ``age_is_valid``)
        - ``age_is_valid``: whether the raw age was plausible
        - ``gender``: raw category (F/M/U)
        - ``tenure_days``: days between ``first_issue_date`` and ``reference_date``
        - ``has_redeemed``: whether the client redeemed *before* ``reference_date``
          (a redemption on/after it is treated as not-yet-redeemed, to avoid leakage)
        - ``days_to_redeem``: days between issue and first redemption; null if never
          redeemed before ``reference_date``, or if the redemption predates issuance
    """
    lf = load_lazy(Dataset.CLIENTS, base_dir)
    parsed = lf.with_columns(
        pl.col("first_issue_date").str.strptime(pl.Datetime, _DATETIME_FORMAT).alias("_issue_dt"),
        pl.col("first_redeem_date")
        .str.strptime(pl.Datetime, _DATETIME_FORMAT, strict=False)
        .alias("_redeem_dt"),
    )
    age_is_valid = pl.col("age").is_between(_MIN_PLAUSIBLE_AGE, _MAX_PLAUSIBLE_AGE)
    redeemed_before_reference = pl.col("_redeem_dt").is_not_null() & (
        pl.col("_redeem_dt") < pl.lit(reference_date)
    )
    days_to_redeem = (pl.col("_redeem_dt") - pl.col("_issue_dt")).dt.total_days()

    return parsed.select(
        "client_id",
        pl.when(age_is_valid).then(pl.col("age")).otherwise(None).alias("age"),
        age_is_valid.alias("age_is_valid"),
        pl.col("gender"),
        (pl.lit(reference_date) - pl.col("_issue_dt")).dt.total_days().alias("tenure_days"),
        redeemed_before_reference.alias("has_redeemed"),
        pl.when(redeemed_before_reference & (days_to_redeem >= 0))
        .then(days_to_redeem)
        .otherwise(None)
        .alias("days_to_redeem"),
    )
