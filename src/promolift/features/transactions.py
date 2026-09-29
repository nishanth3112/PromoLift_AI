"""Transaction-level view of purchases.csv shared by the transaction-based feature groups.

Transaction fields (points, purchase_sum, transaction_datetime, store_id)
repeat on every line item of a transaction, so they are de-duplicated to one
row per transaction before any client-level aggregate.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import polars as pl

from promolift.data.loader import Dataset, load_lazy

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def load_transactions(
    reference_date: datetime, columns: Sequence[str], base_dir: Path | None = None
) -> pl.LazyFrame:
    """One row per (client, transaction) at or before ``reference_date``.

    Returns ``client_id``, ``transaction_id``, ``columns``, and ``_dt`` (the
    parsed transaction time). Transactions after ``reference_date`` are
    dropped: anything later is post-treatment information.
    """
    selected = dict.fromkeys(["client_id", "transaction_id", "transaction_datetime", *columns])
    return (
        load_lazy(Dataset.PURCHASES, base_dir)
        .select(list(selected))
        .unique(subset=["client_id", "transaction_id"])
        .with_columns(
            pl.col("transaction_datetime").str.strptime(pl.Datetime, DATETIME_FORMAT).alias("_dt")
        )
        .filter(pl.col("_dt") <= pl.lit(reference_date))
    )
