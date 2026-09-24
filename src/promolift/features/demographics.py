"""Client demographic features, built from clients.csv only.

Age has a known data-quality issue (see Phase 3's schema validation): 313 of
400,162 clients have an implausible age (e.g. -7491, 1901). Rather than clip
or impute a fake value, implausible ages are nulled out and flagged, so
downstream consumers know what's actually known vs. missing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from promolift.data.loader import Dataset, load_lazy

# Matches the bounds already encoded in data_contracts.x5_raw_schemas.ClientsSchema.
_MIN_PLAUSIBLE_AGE = 0
_MAX_PLAUSIBLE_AGE = 120
_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def build_demographic_features(
    reference_date: datetime, base_dir: Path | None = None
) -> pl.LazyFrame:
    """Build client-level demographic features from clients.csv.

    Args:
        reference_date: Fixed point in time used to compute tenure. Should be
            the same reference date used across all feature builders (e.g.
            the max transaction date in purchases.csv) so features are
            mutually consistent.
        base_dir: Optional override of the raw data directory (for tests).

    Returns one row per client with:
        - ``age``: null if the raw value is outside [0, 120] (see ``age_is_valid``)
        - ``age_is_valid``: whether the raw age was plausible
        - ``gender``: raw category (F/M/U)
        - ``tenure_days``: days between ``first_issue_date`` and ``reference_date``
        - ``has_redeemed``: whether ``first_redeem_date`` is present
        - ``days_to_redeem``: days between issue and first redemption, null if never redeemed
    """
    lf = load_lazy(Dataset.CLIENTS, base_dir)
    parsed = lf.with_columns(
        pl.col("first_issue_date").str.strptime(pl.Datetime, _DATETIME_FORMAT).alias("_issue_dt"),
        pl.col("first_redeem_date")
        .str.strptime(pl.Datetime, _DATETIME_FORMAT, strict=False)
        .alias("_redeem_dt"),
    )
    age_is_valid = pl.col("age").is_between(_MIN_PLAUSIBLE_AGE, _MAX_PLAUSIBLE_AGE)
    return parsed.select(
        "client_id",
        pl.when(age_is_valid).then(pl.col("age")).otherwise(None).alias("age"),
        age_is_valid.alias("age_is_valid"),
        pl.col("gender"),
        (pl.lit(reference_date) - pl.col("_issue_dt")).dt.total_days().alias("tenure_days"),
        pl.col("_redeem_dt").is_not_null().alias("has_redeemed"),
        (pl.col("_redeem_dt") - pl.col("_issue_dt")).dt.total_days().alias("days_to_redeem"),
    )
