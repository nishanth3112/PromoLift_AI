"""Client-level promotion responsiveness features from the loyalty points in purchases.csv.

Express points are the promotional kind (campaign-issued, short-lived), so
how a client earns and spends them is the closest pre-treatment signal of how
they react to a promotion. ``trn_sum_from_red`` / ``trn_sum_from_iss`` are
not used: ``red`` is a line's full price, recorded only on transactions that
spent points, and ``iss`` the part paid in money, so at client level they
carry only redemption information that ``redeem_tx_rate`` and
``points_discount_share`` already capture.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from promolift.features.transactions import load_transactions

_POINTS = (
    "regular_points_received",
    "express_points_received",
    "regular_points_spent",
    "express_points_spent",
    "purchase_sum",
)


def build_promo_responsiveness_features(
    reference_date: datetime, base_dir: Path | None = None
) -> pl.LazyFrame:
    """Build client-level promotion responsiveness features.

    Points spent are stored as negative numbers; totals here are magnitudes.

    Returns one row per client_id with:
        - ``express_points_received_total`` / ``express_points_spent_total``
        - ``express_received_tx_rate`` / ``express_spent_tx_rate``: share of
          transactions that earned / spent express points
        - ``redeem_tx_rate``: share of transactions that spent any points
        - ``points_discount_share``: points spent over total spend -- how much
          of the basket the client pays with points (null with no spend)
        - ``days_since_last_redeem``: days from the last redeeming transaction
          to ``reference_date`` (null if the client never redeemed)
    """
    spent = pl.col("regular_points_spent").abs() + pl.col("express_points_spent").abs()
    redeemed = spent > 0

    return (
        load_transactions(reference_date, _POINTS, base_dir)
        .group_by("client_id")
        .agg(
            pl.col("express_points_received").sum().alias("express_points_received_total"),
            pl.col("express_points_spent").abs().sum().alias("express_points_spent_total"),
            (pl.col("express_points_received") > 0).mean().alias("express_received_tx_rate"),
            (pl.col("express_points_spent") != 0).mean().alias("express_spent_tx_rate"),
            redeemed.mean().alias("redeem_tx_rate"),
            spent.sum().alias("_spent"),
            pl.col("purchase_sum").sum().alias("_spend"),
            (pl.lit(reference_date) - pl.col("_dt").filter(redeemed).max())
            .dt.total_days()
            .alias("days_since_last_redeem"),
        )
        .with_columns(
            pl.when(pl.col("_spend") > 0)
            .then(pl.col("_spent") / pl.col("_spend"))
            .otherwise(None)
            .alias("points_discount_share")
        )
        .drop("_spent", "_spend")
    )
