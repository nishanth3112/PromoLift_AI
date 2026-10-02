"""A parquet cache of built feature tables, keyed by what the table is a function of.

A table is a pure function of the feature code, the raw data, the Polars
version, and the groups chosen, so those four form the cache key: editing a
feature builder or replacing the raw files silently misses the cache rather
than serving stale features. Each parquet file has a JSON manifest beside it
holding the group -> columns map and a content hash, re-checked on every load.

Runs are tagged with both. ``feature_cache_key`` is the reproducible identity:
the same code, data, Polars version, and groups always give the same key.
``feature_table_sha256`` identifies the exact values a run used, but is *not*
reproducible across rebuilds: Polars sums floats in parallel, in an order
that varies run to run, so rebuilt float aggregates differ at ~1e-13
relative (measured on the real data; the Phase 10 features jittered the same
way on every run). Reading one cached file keeps every run comparing models
on identical bytes.

The cache lives in gitignored ``data/interim/features/``, so it never dirties
the working tree mid-run.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from promolift.data.loader import project_root
from promolift.features.build import FeatureGroup, FeatureTable, build_feature_table
from promolift.tracking.lineage import fingerprint_digest, raw_data_fingerprint
from promolift.validation.referential_integrity import transaction_date_integrity

logger = logging.getLogger(__name__)

FEATURE_CACHE_DIR = Path("data") / "interim" / "features"
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# Everything a feature table's values depend on: the builders, the raw
# loader, and the reference-date logic.
_EXTRA_SOURCES = ("data/loader.py", "validation/referential_integrity.py")


def _default_sources() -> list[Path]:
    features = sorted((_PACKAGE_ROOT / "features").glob("*.py"))
    return [*features, *(_PACKAGE_ROOT / extra for extra in _EXTRA_SOURCES)]


def feature_code_sha256(sources: Sequence[Path] | None = None, root: Path | None = None) -> str:
    """SHA-256 over the source files feature values depend on.

    Paths are hashed relative to ``root`` and line endings normalized, so a
    Windows checkout of the same code hashes identically.
    """
    root = root if root is not None else _PACKAGE_ROOT
    digest = hashlib.sha256()
    for path in sorted(sources if sources is not None else _default_sources()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def feature_table_sha256(table: FeatureTable) -> str:
    """SHA-256 of a table's groups, schema, and values, independent of row order.

    Hashes sorted CSV text rather than parquet bytes, so the same table
    rewritten by another Polars/Arrow version keeps its hash.
    """
    digest = hashlib.sha256()
    groups = {group.value: list(columns) for group, columns in table.groups.items()}
    schema = [(name, str(dtype)) for name, dtype in table.frame.schema.items()]
    digest.update(json.dumps({"groups": groups, "schema": schema}).encode())
    digest.update(table.frame.sort("client_id").write_csv().encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class StoredFeatureTable:
    """A feature table, its content hash, and where it is cached."""

    table: FeatureTable
    sha256: str
    cache_key: str
    path: Path
    from_cache: bool

    @property
    def manifest_path(self) -> Path:
        return self.path.with_suffix(".json")

    def lineage_tags(self, groups: Sequence[FeatureGroup | str]) -> dict[str, str]:
        """Run tags identifying the table and the groups a model was trained on."""
        return {
            "feature_cache_key": self.cache_key,
            "feature_table_sha256": self.sha256,
            "feature_groups": ",".join(FeatureGroup(group).value for group in groups),
        }


def _key_inputs(groups: Sequence[FeatureGroup], base_dir: Path | None) -> dict:
    return {
        "feature_code_sha256": feature_code_sha256(),
        "raw_data_digest": fingerprint_digest(raw_data_fingerprint(base_dir)),
        "polars_version": pl.__version__,
        "groups": [group.value for group in groups],
    }


def _cache_key(key_inputs: dict) -> str:
    return hashlib.sha256(json.dumps(key_inputs).encode()).hexdigest()[:16]


def _read(path: Path) -> tuple[FeatureTable, str] | None:
    """The cached table if it exists and matches its manifest's hash, else None."""
    manifest_path = path.with_suffix(".json")
    if not (path.exists() and manifest_path.exists()):
        return None
    manifest = json.loads(manifest_path.read_text())
    groups = {FeatureGroup(g): tuple(cols) for g, cols in manifest["groups"].items()}
    table = FeatureTable(frame=pl.read_parquet(path), groups=groups)
    sha256 = feature_table_sha256(table)
    if sha256 != manifest["feature_table_sha256"]:
        logger.warning("Cached feature table %s does not match its manifest; rebuilding.", path)
        return None
    return table, sha256


def _write(table: FeatureTable, path: Path, *, key_inputs: dict, reference_date: datetime) -> str:
    sha256 = feature_table_sha256(table)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The manifest is written last, so an interrupted write is never read as a hit.
    staged = path.with_suffix(".parquet.partial")
    table.frame.write_parquet(staged)
    staged.replace(path)
    manifest = {
        "feature_cache_key": _cache_key(key_inputs),
        "feature_table_sha256": sha256,
        **{name: value for name, value in key_inputs.items() if name != "groups"},
        "reference_date": reference_date.isoformat(),
        "n_rows": table.frame.height,
        "groups": {group.value: list(columns) for group, columns in table.groups.items()},
        "created_at": datetime.now(tz=UTC).isoformat(),
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return sha256


def load_feature_table(
    groups: Sequence[FeatureGroup | str],
    *,
    base_dir: Path | None = None,
    cache_dir: Path | None = None,
    rebuild: bool = False,
) -> StoredFeatureTable:
    """The feature table for ``groups``, from the cache when it is current.

    On a miss the table is built at the end of the purchase window (the last
    transaction in purchases.csv), cached, and returned.

    Args:
        groups: Feature groups to include (see ``build_feature_table``).
        base_dir: Raw data directory override (for tests).
        cache_dir: Cache directory; defaults to ``data/interim/features``.
        rebuild: Build and overwrite even if a current cache entry exists.
    """
    chosen = [FeatureGroup(group) for group in groups]
    cache_dir = cache_dir if cache_dir is not None else project_root() / FEATURE_CACHE_DIR
    key_inputs = _key_inputs(chosen, base_dir)
    cache_key = _cache_key(key_inputs)
    path = cache_dir / f"features_{cache_key}.parquet"

    cached = None if rebuild else _read(path)
    if cached is not None:
        table, sha256 = cached
        return StoredFeatureTable(table, sha256, cache_key, path, from_cache=True)

    reference_date = transaction_date_integrity(base_dir).max_date
    table = build_feature_table(reference_date, base_dir, groups=chosen)
    sha256 = _write(table, path, key_inputs=key_inputs, reference_date=reference_date)
    return StoredFeatureTable(table, sha256, cache_key, path, from_cache=False)
