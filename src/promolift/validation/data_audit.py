"""Data-audit utilities for the raw X5 RetailHero dataset files.

Provides dataset inventory, schema discovery, null-rate summaries, and
numeric/date range summaries, computed lazily so large files (notably
``purchases.csv`` at ~4.2 GB) are never loaded into memory.

This module intentionally avoids domain-specific expectations -- expected
schemas, client/product foreign-key integrity, treatment/control balance,
outcome distribution, and ATE sanity checks belong to a later, more targeted
audit once the real schemas have been discovered and confirmed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from promolift.data.loader import Dataset, load_lazy, path_for


@dataclass
class FileInventoryEntry:
    """Existence and size of a registered raw dataset file."""

    name: str
    path: Path
    exists: bool
    size_bytes: int | None


@dataclass
class ColumnAudit:
    """Per-column null-rate and range summary."""

    name: str
    dtype: str
    null_count: int
    null_rate: float
    min_value: object | None = None
    max_value: object | None = None


@dataclass
class DatasetAudit:
    """Structural summary of a single dataset file."""

    name: str
    path: Path
    file_size_bytes: int
    row_count: int
    column_count: int
    columns: list[ColumnAudit] = field(default_factory=list)
    duplicate_row_count: int | None = None


def inventory(base_dir: Path | None = None) -> list[FileInventoryEntry]:
    """Check existence and size of every registered raw dataset file."""
    entries = []
    for dataset in Dataset:
        path = path_for(dataset, base_dir)
        exists = path.exists()
        size = path.stat().st_size if exists else None
        entries.append(FileInventoryEntry(name=dataset.value, path=path, exists=exists, size_bytes=size))
    return entries


def audit_dataset(
    dataset: Dataset,
    base_dir: Path | None = None,
    *,
    key_columns: list[str] | None = None,
) -> DatasetAudit:
    """Compute a structural audit for a single dataset.

    Row count, null counts, and numeric/date ranges are computed in a single
    lazy aggregation pass to avoid rescanning large files. Duplicate-key
    checking is a separate, opt-in pass: pass ``key_columns`` to check
    whether those columns uniquely identify rows. Left unset, no duplicate
    check is run, since a full-row duplicate scan is not computationally
    reasonable for a file the size of ``purchases.csv``.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    path = path_for(dataset, base_dir)
    if not path.exists():
        msg = f"Missing dataset file for {dataset.value!r}: expected at {path}"
        raise FileNotFoundError(msg)

    lf = load_lazy(dataset, base_dir)
    schema = lf.collect_schema()

    agg_exprs: list[pl.Expr] = [pl.len().alias("__row_count__")]
    for col, dtype in schema.items():
        agg_exprs.append(pl.col(col).null_count().alias(f"__null__{col}"))
        if dtype.is_numeric() or dtype.is_temporal():
            agg_exprs.append(pl.col(col).min().alias(f"__min__{col}"))
            agg_exprs.append(pl.col(col).max().alias(f"__max__{col}"))

    stats = lf.select(agg_exprs).collect().row(0, named=True)
    row_count = stats["__row_count__"]

    column_audits = [
        ColumnAudit(
            name=col,
            dtype=str(dtype),
            null_count=stats[f"__null__{col}"],
            null_rate=(stats[f"__null__{col}"] / row_count) if row_count else 0.0,
            min_value=stats.get(f"__min__{col}"),
            max_value=stats.get(f"__max__{col}"),
        )
        for col, dtype in schema.items()
    ]

    duplicate_row_count = None
    if key_columns:
        duplicate_row_count = (
            lf.group_by(key_columns).len().filter(pl.col("len") > 1).select(pl.len()).collect().item()
        )

    return DatasetAudit(
        name=dataset.value,
        path=path,
        file_size_bytes=path.stat().st_size,
        row_count=row_count,
        column_count=len(schema),
        columns=column_audits,
        duplicate_row_count=duplicate_row_count,
    )


def column_cardinality(dataset: Dataset, columns: list[str], base_dir: Path | None = None) -> dict[str, int]:
    """Return the number of unique values per column, for key/uniqueness checks."""
    lf = load_lazy(dataset, base_dir)
    stats = lf.select([pl.col(c).n_unique().alias(c) for c in columns]).collect().row(0, named=True)
    return dict(stats)
