"""Client-level basket composition and store / time-of-week shopping habits.

Basket size and product variety describe *how* a client shops (big planned
trips vs top-ups, habitual vs exploratory); store loyalty and timing describe
*where and when*. All habit rates are shares of transactions, not line items,
so one large basket doesn't count as many visits.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from promolift.features.transactions import load_line_items

# The timestamps are not store-local time: activity peaks at 09:00 and 14:00
# and is near zero after 20:00, consistent with UTC for stores at UTC+3 and
# beyond. Wall-clock bands ("evening") would mean different things by region,
# so time of day is summarized by the hour's mean and spread instead; the
# spread is unaffected by a timezone shift. Near-midnight local hours hold
# <0.1% of transactions, so the weekday is effectively unaffected.
_SATURDAY = 6  # Polars weekday(): Monday = 1 ... Sunday = 7


def build_basket_store_features(
    reference_date: datetime, base_dir: Path | None = None
) -> pl.LazyFrame:
    """Build client-level basket, store loyalty, and timing features.

    Returns one row per client_id with:
        - ``basket_lines_mean``: mean line items (product rows) per transaction
        - ``basket_quantity_mean``: mean total ``product_quantity`` per transaction
        - ``distinct_products``: distinct products ever bought
        - ``product_variety_ratio``: distinct products over line items -- near 1
          for exploratory shoppers, low for habitual repeat buyers
        - ``distinct_stores``: distinct stores visited
        - ``top_store_tx_share``: share of transactions at the client's most
          visited store
        - ``weekend_tx_rate``: share of transactions on Saturday/Sunday
        - ``tx_hour_mean`` / ``tx_hour_std``: mean and spread of the transaction
          hour in the file's clock -- a client who always shops at the same time
          has a low spread (std null with a single transaction)
    """
    lines = load_line_items(
        reference_date, ["store_id", "product_id", "product_quantity"], base_dir
    )

    per_transaction = lines.group_by("client_id", "transaction_id").agg(
        pl.len().alias("_lines"),
        pl.col("product_quantity").sum().alias("_quantity"),
        pl.col("store_id").first(),
        pl.col("_dt").first(),
    )
    habits = per_transaction.group_by("client_id").agg(
        pl.col("_lines").mean().alias("basket_lines_mean"),
        pl.col("_quantity").mean().alias("basket_quantity_mean"),
        pl.col("store_id").n_unique().alias("distinct_stores"),
        # Tied modes are equally frequent, so any of them gives the same share.
        (pl.col("store_id") == pl.col("store_id").mode().first())
        .mean()
        .alias("top_store_tx_share"),
        (pl.col("_dt").dt.weekday() >= _SATURDAY).mean().alias("weekend_tx_rate"),
        pl.col("_dt").dt.hour().cast(pl.Float64).mean().alias("tx_hour_mean"),
        pl.col("_dt").dt.hour().cast(pl.Float64).std().alias("tx_hour_std"),
    )
    variety = lines.group_by("client_id").agg(
        pl.col("product_id").n_unique().alias("distinct_products"),
        (pl.col("product_id").n_unique() / pl.len()).alias("product_variety_ratio"),
    )
    return habits.join(variety, on="client_id", how="left")
