"""Feature drift: does a scoring batch still look like the clients the model was trained on?

``build_reference`` bins each training feature and stores the expected share
of every bin; ``drift_report`` bins a scoring batch the same way and scores
each feature with the population stability index (PSI). The same binning
code serves both, so a batch and its reference can never be binned
differently.

Bins per feature:

- **numeric, at most 10 distinct values** (flags, small counts): one bin per
  training value, plus "other" for values never seen. Deciles would collapse
  a 0/1 flag into one bin and hide a shift in its share.
- **numeric, more values**: decile bins (duplicate edges merged).
- **categorical**: one bin per contract level, plus "other".

Every feature also has a missing-value bin, so a batch where a feature starts
coming back empty registers as drift.

PSI = sum over bins of (actual - expected) * ln(actual / expected), with
shares floored at 1e-4. Common reading: below 0.1 stable, 0.1-0.25 a shift
worth a look, above 0.25 a significant shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

from promolift.data.loader import project_root

REFERENCE_VERSION = 2
MAX_DISCRETE_VALUES = 10
DECILES = tuple(round(q / 10, 1) for q in range(1, 10))
MONITORING_CONFIG_PATH = Path("configs") / "monitoring.yaml"
_SHARE_FLOOR = 1e-4


class DriftStatus(StrEnum):
    """Severity of a feature's (or a batch's) drift, in increasing order."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


class DriftError(RuntimeError):
    """A scoring batch drifted past the failure threshold; no send list is written."""


@dataclass(frozen=True)
class DriftThresholds:
    """PSI above ``warn_psi`` warns; above ``fail_psi`` blocks the batch."""

    warn_psi: float = 0.1
    fail_psi: float = 0.25

    def __post_init__(self) -> None:
        if not 0 <= self.warn_psi <= self.fail_psi:
            raise ValueError("Drift thresholds need 0 <= warn_psi <= fail_psi")

    def status(self, psi: float) -> DriftStatus:
        if psi > self.fail_psi:
            return DriftStatus.FAIL
        if psi > self.warn_psi:
            return DriftStatus.WARN
        return DriftStatus.OK


def load_drift_thresholds(path: Path | None = None) -> DriftThresholds:
    """The ``drift`` section of ``configs/monitoring.yaml``."""
    path = path if path is not None else project_root() / MONITORING_CONFIG_PATH
    drift = (yaml.safe_load(path.read_text()) or {})["drift"]
    return DriftThresholds(float(drift["warn_psi"]), float(drift["fail_psi"]))


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="raise").astype("float64")


def _text(values: pd.Series) -> pd.Series:
    text = values.astype(object).where(values.notna(), None)
    return text.map(lambda v: None if v is None else str(v))


def _bin_counts(values: pd.Series, spec: dict) -> np.ndarray:
    """Counts per bin, missing values last; layout fixed by the reference ``spec``."""
    missing = int(values.isna().sum())
    if spec["kind"] == "categorical":
        present = _text(values).dropna()
        levels = spec["levels"]
        counts = [int((present == level).sum()) for level in levels]
        counts.append(len(present) - sum(counts))  # other
    elif spec["binning"] == "values":
        present = _numeric(values).dropna()
        counts = [int(np.isclose(present, value).sum()) for value in spec["values"]]
        counts.append(len(present) - sum(counts))  # other
    else:
        present = _numeric(values).dropna().to_numpy()
        index = np.searchsorted(np.asarray(spec["edges"], dtype=float), present, side="left")
        counts = list(np.bincount(index, minlength=len(spec["edges"]) + 1))
    return np.asarray([*counts, missing], dtype=float)


def bin_shares(values: pd.Series, spec: dict) -> np.ndarray:
    """Share of ``values`` in each bin of ``spec`` (missing last); sums to 1."""
    counts = _bin_counts(values, spec)
    total = counts.sum()
    return counts / total if total else counts


def _feature_spec(values: pd.Series, categories: tuple[str, ...] | None) -> dict:
    if categories is not None:
        spec: dict = {"kind": "categorical", "levels": list(categories)}
    else:
        present = _numeric(values).dropna()
        distinct = np.unique(present)
        if len(distinct) <= MAX_DISCRETE_VALUES:
            spec = {"kind": "numeric", "binning": "values", "values": distinct.tolist()}
        else:
            edges = np.unique(np.quantile(present, DECILES))
            spec = {"kind": "numeric", "binning": "quantile", "edges": edges.tolist()}
        spec["mean"] = float(present.mean()) if len(present) else None
    spec["expected_shares"] = bin_shares(values, spec).tolist()
    spec["null_share"] = float(values.isna().mean())
    return spec


def build_reference(features: pd.DataFrame, categories: dict[str, tuple[str, ...]]) -> dict:
    """Bins and expected bin shares of every training feature.

    Args:
        features: The training features (canonical input or training layout).
        categories: Categorical column -> its contract levels.
    """
    return {
        "version": REFERENCE_VERSION,
        "n_clients": len(features),
        "features": {
            column: _feature_spec(features[column], categories.get(column))
            for column in features.columns
        },
    }


def check_reference(reference: dict | None) -> dict:
    """``reference`` if it carries bin shares; otherwise a clear error.

    Raises:
        ValueError: If there is no reference, or it predates bin shares
            (model versions registered before reference version 2).
    """
    if reference is None or reference.get("version") != REFERENCE_VERSION:
        msg = (
            "The model has no drift reference with bin shares (reference version "
            f"{REFERENCE_VERSION}); re-register it with scripts/register_model.py"
        )
        raise ValueError(msg)
    return reference


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population stability index of two share vectors over the same bins."""
    e = np.maximum(np.asarray(expected, dtype=float), _SHARE_FLOOR)
    a = np.maximum(np.asarray(actual, dtype=float), _SHARE_FLOOR)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(
    features: pd.DataFrame, reference: dict, thresholds: DriftThresholds
) -> pl.DataFrame:
    """PSI and status per reference feature for a scoring batch, worst first.

    Raises:
        ValueError: If the reference lacks bin shares or a feature is missing.
    """
    specs = check_reference(reference)["features"]
    missing = [column for column in specs if column not in features.columns]
    if missing:
        raise ValueError(f"The batch lacks reference features {missing}")
    rows = []
    for column, spec in specs.items():
        actual = bin_shares(features[column], spec)
        score = psi(np.asarray(spec["expected_shares"]), actual)
        rows.append(
            {
                "feature": column,
                "kind": spec.get("binning", spec["kind"]),
                "psi": score,
                "status": thresholds.status(score).value,
                "null_share": float(actual[-1]),
                "reference_null_share": spec["null_share"],
            }
        )
    return pl.DataFrame(rows).sort("psi", descending=True)


def overall_status(report: pl.DataFrame) -> DriftStatus:
    """The worst feature status in a report."""
    statuses = set(report["status"].to_list())
    for status in (DriftStatus.FAIL, DriftStatus.WARN):
        if status.value in statuses:
            return status
    return DriftStatus.OK
