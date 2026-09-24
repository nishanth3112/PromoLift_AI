"""Referential integrity and date-sanity checks for the raw X5 files.

Complements the structural audit in ``data_audit.py`` and the schema
contracts in ``data_contracts`` with checks that span more than one file
(foreign keys) or need domain-aware date parsing, rather than just
null/range stats on a raw string column.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from promolift.data.loader import Dataset, load_lazy

_TRANSACTION_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class ForeignKeyIntegrityReport:
    """Result of checking that a child dataset's key exists in a parent dataset."""

    child_dataset: str
    child_key: str
    parent_dataset: str
    parent_key: str
    total_rows: int
    orphan_count: int
    orphan_rate: float


@dataclass
class DateIntegrityReport:
    """Result of parsing and sanity-checking a datetime column."""

    column: str
    total_rows: int
    unparseable_count: int
    min_date: datetime | None
    max_date: datetime | None
    future_date_count: int


def _foreign_key_integrity(
    child_dataset: Dataset,
    child_key: str,
    parent_dataset: Dataset,
    parent_key: str,
    base_dir: Path | None = None,
) -> ForeignKeyIntegrityReport:
    child_lf = load_lazy(child_dataset, base_dir).select(child_key)
    parent_lf = (
        load_lazy(parent_dataset, base_dir)
        .select(pl.col(parent_key).alias(child_key))
        .unique()
        .with_columns(pl.lit(value=True).alias("_found"))
    )

    # A marker column (rather than the join key itself) survives the join
    # intact for matched rows and comes back null for orphans -- Polars'
    # optimizer coalesces same-named join-key columns, which silently drops
    # any null signal if you try to inspect the key column post-join.
    joined = child_lf.join(parent_lf, on=child_key, how="left")
    stats = (
        joined.select(
            pl.len().alias("total_rows"),
            pl.col("_found").is_null().sum().alias("orphan_count"),
        )
        .collect()
        .row(0, named=True)
    )

    total_rows = stats["total_rows"]
    orphan_count = stats["orphan_count"]
    return ForeignKeyIntegrityReport(
        child_dataset=child_dataset.value,
        child_key=child_key,
        parent_dataset=parent_dataset.value,
        parent_key=parent_key,
        total_rows=total_rows,
        orphan_count=orphan_count,
        orphan_rate=(orphan_count / total_rows) if total_rows else 0.0,
    )


def client_foreign_key_integrity(base_dir: Path | None = None) -> ForeignKeyIntegrityReport:
    """Check that every client_id in purchases.csv exists in clients.csv."""
    return _foreign_key_integrity(
        Dataset.PURCHASES, "client_id", Dataset.CLIENTS, "client_id", base_dir
    )


def product_foreign_key_integrity(base_dir: Path | None = None) -> ForeignKeyIntegrityReport:
    """Check that every product_id in purchases.csv exists in products.csv."""
    return _foreign_key_integrity(
        Dataset.PURCHASES, "product_id", Dataset.PRODUCTS, "product_id", base_dir
    )


def transaction_date_integrity(
    base_dir: Path | None = None,
    *,
    as_of: datetime | None = None,
) -> DateIntegrityReport:
    """Check that purchases.csv's transaction_datetime is parseable and sane.

    ``transaction_datetime`` is stored as a raw string (see
    ``data_contracts.x5_raw_schemas.PurchasesSchema``). This parses it and
    reports how many rows fail to parse, the observed date range, and how
    many transactions are dated after ``as_of`` (defaults to now, UTC).
    """
    as_of = as_of if as_of is not None else datetime.now(UTC).replace(tzinfo=None)
    lf = load_lazy(Dataset.PURCHASES, base_dir).select(
        pl.col("transaction_datetime")
        .str.strptime(pl.Datetime, _TRANSACTION_DATETIME_FORMAT, strict=False)
        .alias("_parsed")
    )
    stats = (
        lf.select(
            pl.len().alias("total_rows"),
            pl.col("_parsed").is_null().sum().alias("unparseable_count"),
            pl.col("_parsed").min().alias("min_date"),
            pl.col("_parsed").max().alias("max_date"),
            (pl.col("_parsed") > pl.lit(as_of)).sum().alias("future_date_count"),
        )
        .collect()
        .row(0, named=True)
    )

    return DateIntegrityReport(
        column="transaction_datetime",
        total_rows=stats["total_rows"],
        unparseable_count=stats["unparseable_count"],
        min_date=stats["min_date"],
        max_date=stats["max_date"],
        future_date_count=stats["future_date_count"],
    )
