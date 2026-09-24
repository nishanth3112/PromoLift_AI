"""Ties the raw dataset loader to its Pandera schema contract.

Kept separate from ``data_contracts`` so the schemas themselves have no
dependency on how a dataset is loaded from disk.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from data_contracts.x5_raw_schemas import SCHEMA_REGISTRY

from promolift.data.loader import Dataset, load_lazy


def validate_dataset(dataset: Dataset, base_dir: Path | None = None) -> pl.DataFrame | pl.LazyFrame:
    """Validate a registered dataset against its schema contract.

    For every dataset except ``Dataset.PURCHASES``, this collects eagerly and
    returns a validated ``pl.DataFrame`` -- these files are small (<= ~22 MB)
    and safe to load fully. Both dtype and value-level checks (``ge``,
    ``le``, ``isin``, ...) are enforced.

    For ``Dataset.PURCHASES`` (~4.2 GB), this returns an unvalidated-until-
    collected ``pl.LazyFrame`` instead. Confirmed empirically: Pandera's
    Polars backend does *not* enforce value-level ``Field`` checks when given
    a ``LazyFrame`` -- known-invalid rows pass silently on ``.collect()``.
    Only dtype/column-structure mismatches are actually caught in that path.
    For full numeric-range verification of ``purchases.csv``, use
    ``promolift.validation.data_audit.audit_dataset`` instead, which computes
    exact ranges via a genuine lazy aggregation.

    Raises:
        pandera.errors.SchemaError | pandera.errors.SchemaErrors: If the
            (eagerly validated) data violates the schema contract.
    """
    schema = SCHEMA_REGISTRY[dataset]
    lf = load_lazy(dataset, base_dir)
    if dataset is Dataset.PURCHASES:
        return schema.validate(lf, lazy=True)
    return schema.validate(lf.collect())
