"""Canonical train/validation/test split of the labelled ``uplift_train`` clients.

The split is a persisted ``client_id -> split`` assignment rather than
something recomputed on the fly, so every model run on every machine is
evaluated on exactly the same clients. The test split is locked: it is not
used for model selection or tuning until final evaluation.

``data/processed/`` is gitignored; the canonical copy lives as an MLflow
artifact on the shared tracking server (see ``scripts/fetch_split.py``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import polars as pl

from promolift.data.loader import project_root

SPLIT_FILENAME = "split_assignment.parquet"
_STRATA = ("treatment_flg", "target")
_DEFAULT_FRACTION_TOLERANCE = 0.01
# Simulated on the real 200k uplift_train rows (20 seeds): stratified 60/20/20
# splits deviate by ~0.00002, unstratified random ones by 0.0029 median
# (0.0072 max) -- 0.001 separates them with wide margin on both sides.
_DEFAULT_STRATUM_TOLERANCE = 0.001


class Split(StrEnum):
    """Split labels as stored in the assignment file."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


EXPECTED_FRACTIONS: dict[Split, float] = {Split.TRAIN: 0.6, Split.VAL: 0.2, Split.TEST: 0.2}


class SplitIntegrityError(ValueError):
    """The assignment is structurally unusable (wrong clients, duplicates, bad labels)."""


@dataclass
class SplitValidationReport:
    """Size and stratification of a structurally valid split."""

    n_clients: int
    fractions: dict[str, float]
    treatment_rates: dict[str, float]
    target_rates: dict[str, float]
    max_fraction_deviation: float
    max_stratum_deviation: float
    matches_expected_fractions: bool
    is_stratified: bool


def split_assignment_path(processed_dir: Path | None = None) -> Path:
    """Expected location of the split file, without checking existence."""
    base = processed_dir if processed_dir is not None else project_root() / "data" / "processed"
    return base / SPLIT_FILENAME


def _check_integrity(assignment: pl.DataFrame, labels: pl.DataFrame) -> None:
    if set(assignment.columns) != {"client_id", "split"}:
        msg = f"Expected columns ['client_id', 'split'], got {assignment.columns}"
        raise SplitIntegrityError(msg)
    if assignment.null_count().sum_horizontal().item():
        raise SplitIntegrityError("Split assignment contains nulls")

    unknown_labels = set(assignment["split"].unique()) - {s.value for s in Split}
    if unknown_labels:
        msg = f"Unknown split labels {sorted(unknown_labels)}; expected {[s.value for s in Split]}"
        raise SplitIntegrityError(msg)

    n_duplicated = assignment.height - assignment["client_id"].n_unique()
    if n_duplicated:
        raise SplitIntegrityError(f"{n_duplicated} client_id(s) assigned more than once")

    assigned = set(assignment["client_id"])
    expected = set(labels["client_id"])
    if missing := expected - assigned:
        raise SplitIntegrityError(f"{len(missing)} uplift_train client(s) missing from the split")
    if unknown := assigned - expected:
        raise SplitIntegrityError(f"{len(unknown)} split client(s) not in uplift_train")


def validate_split(
    assignment: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    fraction_tolerance: float = _DEFAULT_FRACTION_TOLERANCE,
    stratum_tolerance: float = _DEFAULT_STRATUM_TOLERANCE,
) -> SplitValidationReport:
    """Check a split covers exactly the labelled clients, then measure its stratification.

    Args:
        assignment: One row per client with ``client_id`` and ``split`` columns.
        labels: ``uplift_train`` rows (``client_id``, ``treatment_flg``, ``target``).
        fraction_tolerance: Allowed absolute deviation from the 60/20/20 fractions.
        stratum_tolerance: Allowed absolute deviation of any split's
            ``treatment_flg x target`` stratum share from the population share.

    Raises:
        SplitIntegrityError: For structural problems. Size or stratification
            drift is reported, not raised -- the caller decides whether it's
            acceptable.
    """
    _check_integrity(assignment, labels)

    joined = assignment.join(labels.select("client_id", *_STRATA), on="client_id")
    n = joined.height

    per_split = joined.group_by("split").agg(
        pl.len().alias("count"),
        pl.col("treatment_flg").mean().alias("treatment_rate"),
        pl.col("target").mean().alias("target_rate"),
    )
    by_split = {row["split"]: row for row in per_split.to_dicts()}
    fractions = {s.value: by_split.get(s.value, {"count": 0})["count"] / n for s in Split}

    overall_shares = {
        (row["treatment_flg"], row["target"]): row["len"] / n
        for row in joined.group_by(*_STRATA).len().to_dicts()
    }
    split_shares = {
        (row["split"], row["treatment_flg"], row["target"]): row["len"]
        / by_split[row["split"]]["count"]
        for row in joined.group_by("split", *_STRATA).len().to_dicts()
    }
    max_stratum_deviation = max(
        abs(split_shares.get((split, *stratum), 0.0) - share)
        for split in by_split
        for stratum, share in overall_shares.items()
    )
    max_fraction_deviation = max(abs(fractions[s.value] - f) for s, f in EXPECTED_FRACTIONS.items())

    return SplitValidationReport(
        n_clients=n,
        fractions=fractions,
        treatment_rates={s: row["treatment_rate"] for s, row in by_split.items()},
        target_rates={s: row["target_rate"] for s, row in by_split.items()},
        max_fraction_deviation=max_fraction_deviation,
        max_stratum_deviation=max_stratum_deviation,
        matches_expected_fractions=max_fraction_deviation <= fraction_tolerance,
        is_stratified=max_stratum_deviation <= stratum_tolerance,
    )


def split_content_sha256(assignment: pl.DataFrame) -> str:
    """SHA-256 of the assignment's content, independent of row order and file format.

    Hashes sorted ``client_id,split`` lines rather than parquet bytes, so the
    same split rewritten by a different Polars/Arrow version keeps its hash.
    """
    ordered = assignment.select("client_id", "split").sort("client_id")
    lines = (f"{c},{s}\n" for c, s in ordered.iter_rows())
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode())
    return digest.hexdigest()


def write_split(assignment: pl.DataFrame, path: Path, *, force: bool = False) -> Path:
    """Persist the assignment, refusing to replace an existing split unless ``force``.

    Re-splitting silently invalidates every result evaluated on the old split,
    so overwriting has to be a deliberate act.
    """
    if path.exists() and not force:
        msg = f"{path} already exists; pass --force to replace the canonical split"
        raise FileExistsError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    assignment.select("client_id", "split").sort("client_id").write_parquet(path)
    return path


def load_split(path: Path | None = None) -> pl.DataFrame:
    """Load the persisted ``client_id -> split`` assignment."""
    path = path if path is not None else split_assignment_path()
    if not path.exists():
        msg = (
            f"No split assignment at {path}. Download the canonical split with "
            "`uv run --env-file .env python scripts/fetch_split.py`."
        )
        raise FileNotFoundError(msg)
    return pl.read_parquet(path)
