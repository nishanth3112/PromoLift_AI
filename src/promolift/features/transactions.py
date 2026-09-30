"""Line-item and transaction-level views of purchases.csv for the feature groups.

Transaction fields (points, purchase_sum, transaction_datetime, store_id)
repeat on every line item of a transaction, so transaction-level aggregates
use ``load_transactions``, which de-duplicates to one row per transaction.
Both views drop rows after the reference date: anything later is
post-treatment information.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import polars as pl

from promolift.data.loader import Dataset, load_lazy

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def load_line_items(
    reference_date: datetime, columns: Sequence[str], base_dir: Path | None = None
) -> pl.LazyFrame:
    """Every line item at or before ``reference_date``.

    Returns ``client_id``, ``transaction_id``, ``columns``, and ``_dt`` (the
    parsed transaction time).
    """
    selected = dict.fromkeys(["client_id", "transaction_id", "transaction_datetime", *columns])
    return (
        load_lazy(Dataset.PURCHASES, base_dir)
        .select(list(selected))
        .with_columns(
            pl.col("transaction_datetime").str.strptime(pl.Datetime, DATETIME_FORMAT).alias("_dt")
        )
        .filter(pl.col("_dt") <= pl.lit(reference_date))
    )


def load_transactions(
    reference_date: datetime, columns: Sequence[str], base_dir: Path | None = None
) -> pl.LazyFrame:
    """One row per (client, transaction) at or before ``reference_date``.

    ``columns`` must be transaction-level fields; returns the same columns as
    ``load_line_items``.
    """
    return load_line_items(reference_date, columns, base_dir).unique(
        subset=["client_id", "transaction_id"]
    )
