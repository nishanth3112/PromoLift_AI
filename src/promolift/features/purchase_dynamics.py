"""Client-level spend and visit dynamics: regularity, recency windows, and trend.

Complements purchase_behavior.py, which summarizes the whole purchase window
(RFM plus a 30-day window), with how a client's shopping is *changing* and how
regular it is -- a lapsing or erratic client may react to an SMS differently
from a steady one.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from promolift.features.transactions import load_transactions

RECENT_WINDOWS_DAYS = (7, 14, 60, 90)
# Four 4-week bins fit the ~117-day purchase window, and each holds every
# weekday equally often, so weekly shopping rhythm doesn't bias the slope.
TREND_BIN_DAYS = 28
TREND_BINS = 4


def _trend_slope(value: pl.Expr, bin_index: pl.Expr) -> pl.Expr:
    """OLS slope of per-bin totals over time, empty bins counting as zero.

    With x fixed at 0..B-1, the slope is sum((x - mean_x) * y) / Sxx; a
    zero-valued bin adds nothing to the sum, so it can be taken over
    transactions directly instead of over a pivoted bin table.
    """
    x = (TREND_BINS - 1) - bin_index  # bin 0 is the most recent: time runs forward
    x_mean = (TREND_BINS - 1) / 2
    sxx = TREND_BINS * (TREND_BINS**2 - 1) / 12
    in_window = bin_index < TREND_BINS
    return ((x - x_mean) * value).filter(in_window).sum() / sxx


def build_purchase_dynamics_features(
    reference_date: datetime, base_dir: Path | None = None
) -> pl.LazyFrame:
    """Build client-level visit regularity, recency window, and trend features.

    Returns one row per client_id with:
        - ``active_days``: distinct calendar days with a purchase
        - ``interpurchase_days_mean`` / ``_std`` / ``_cv``: gaps in days between
          consecutive purchase days (same-day baskets are one visit). Null with
          fewer than 2 gaps for std and cv, fewer than 1 for the mean.
        - ``recent_frequency_{7,14,60,90}d`` / ``recent_monetary_{7,14,60,90}d``:
          transactions and spend in each window before ``reference_date`` (0, not
          null, without activity; the 30-day window is in purchase_behavior)
        - ``spend_trend_slope`` / ``visit_trend_slope``: OLS slope of spend and
          transaction count across the last four 28-day bins, per bin
    """
    days_before = (pl.lit(reference_date) - pl.col("_dt")).dt.total_days()
    bin_index = days_before // TREND_BIN_DAYS
    gaps = pl.col("_dt").dt.date().unique().sort().diff().dt.total_days().drop_nulls()

    windows = []
    for days in RECENT_WINDOWS_DAYS:
        is_recent = pl.col("_dt") >= pl.lit(reference_date - timedelta(days=days))
        windows += [
            is_recent.sum().alias(f"recent_frequency_{days}d"),
            pl.col("purchase_sum").filter(is_recent).sum().alias(f"recent_monetary_{days}d"),
        ]

    return (
        load_transactions(reference_date, ["purchase_sum"], base_dir)
        .group_by("client_id")
        .agg(
            pl.col("_dt").dt.date().n_unique().alias("active_days"),
            gaps.mean().alias("interpurchase_days_mean"),
            gaps.std().alias("interpurchase_days_std"),
            *windows,
            _trend_slope(pl.col("purchase_sum"), bin_index).alias("spend_trend_slope"),
            _trend_slope(pl.lit(1.0), bin_index).alias("visit_trend_slope"),
        )
        .with_columns(
            pl.when(pl.col("interpurchase_days_mean") > 0)
            .then(pl.col("interpurchase_days_std") / pl.col("interpurchase_days_mean"))
            .otherwise(None)
            .alias("interpurchase_days_cv")
        )
    )
