"""Reusable, lazy loading layer for the X5 RetailHero raw dataset files."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import polars as pl


class Dataset(StrEnum):
    """Registry of the six raw X5 RetailHero files."""

    CLIENTS = "clients"
    PRODUCTS = "products"
    PURCHASES = "purchases"
    UPLIFT_TRAIN = "uplift_train"
    UPLIFT_TEST = "uplift_test"
    UPLIFT_SAMPLE_SUBMISSION = "uplift_sample_submission"


_FILENAMES: dict[Dataset, str] = {
    Dataset.CLIENTS: "clients.csv",
    Dataset.PRODUCTS: "products.csv",
    Dataset.PURCHASES: "purchases.csv",
    Dataset.UPLIFT_TRAIN: "uplift_train.csv",
    Dataset.UPLIFT_TEST: "uplift_test.csv",
    Dataset.UPLIFT_SAMPLE_SUBMISSION: "uplift_sample_submission.csv",
}


def project_root() -> Path:
    """Return the repository root, resolved relative to this file."""
    return Path(__file__).resolve().parents[3]


def raw_data_dir(base_dir: Path | None = None) -> Path:
    """Return the directory containing raw X5 files.

    Args:
        base_dir: Optional override of the raw data directory, mainly for tests.
    """
    return base_dir if base_dir is not None else project_root() / "data" / "raw"


def path_for(dataset: Dataset, base_dir: Path | None = None) -> Path:
    """Return the expected path for a dataset file, without checking existence."""
    return raw_data_dir(base_dir) / _FILENAMES[dataset]


def resolve_path(dataset: Dataset, base_dir: Path | None = None) -> Path:
    """Resolve the full path to a registered dataset file.

    Raises:
        FileNotFoundError: If the file does not exist at the expected location.
    """
    path = path_for(dataset, base_dir)
    if not path.exists():
        msg = f"Missing dataset file for {dataset.value!r}: expected at {path}"
        raise FileNotFoundError(msg)
    return path


def load_lazy(dataset: Dataset, base_dir: Path | None = None) -> pl.LazyFrame:
    """Lazily load a registered dataset as a Polars LazyFrame.

    Uses ``scan_csv`` so no data is read into memory until the query is
    collected. This is required for ``purchases.csv`` (~4.2 GB).
    """
    path = resolve_path(dataset, base_dir)
    return pl.scan_csv(path)
