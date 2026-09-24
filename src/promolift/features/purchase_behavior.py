"""Client-level purchase behavior features from purchases.csv.

All transaction-level fields (points, purchase_sum, transaction_datetime)
repeat identically across every line item of the same transaction --
deduplicated to transaction level before aggregating, or client-level
aggregates would be inflated by however many products were in the basket.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from promolift.data.loader import Dataset, load_lazy

_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def build_purchase_behavior_features(
    reference_date: datetime, base_dir: Path | None = None
) -> pl.LazyFrame:
    """Build client-level RFM, monetary volatility, and loyalty points features.

    Only covers clients with at least one transaction in purchases.csv --
    join against the full client universe with a left join to fill in
    clients with no purchase history (see ``build.build_feature_table``).

    Args:
        reference_date: Fixed point in time used to compute recency. Should
            match the reference date used by other feature builders.
        base_dir: Optional override of the raw data directory (for tests).

    Returns one row per client_id with:
        - ``frequency``: number of distinct transactions
        - ``recency_days``: days between the last transaction and ``reference_date``
        - ``monetary_total``: sum of purchase_sum across transactions
        - ``monetary_avg``: average purchase_sum per transaction
        - ``monetary_std``: standard deviation of purchase_sum (null if frequency < 2)
        - ``points_received_total``: sum of regular + express points received
        - ``points_spent_total``: sum of |regular| + |express| points spent
    """
    purchases = load_lazy(Dataset.PURCHASES, base_dir)

    transactions = (
        purchases.select(
            "client_id",
            "transaction_id",
            "transaction_datetime",
            "regular_points_received",
            "express_points_received",
            "regular_points_spent",
            "express_points_spent",
            "purchase_sum",
        )
        .unique(subset=["client_id", "transaction_id"])
        .with_columns(
            pl.col("transaction_datetime").str.strptime(pl.Datetime, _DATETIME_FORMAT).alias("_dt")
        )
    )

    return transactions.group_by("client_id").agg(
        pl.len().alias("frequency"),
        (pl.lit(reference_date) - pl.col("_dt").max()).dt.total_days().alias("recency_days"),
        pl.col("purchase_sum").sum().alias("monetary_total"),
        pl.col("purchase_sum").mean().alias("monetary_avg"),
        pl.col("purchase_sum").std().alias("monetary_std"),
        (pl.col("regular_points_received") + pl.col("express_points_received"))
        .sum()
        .alias("points_received_total"),
        (pl.col("regular_points_spent").abs() + pl.col("express_points_spent").abs())
        .sum()
        .alias("points_spent_total"),
    )
