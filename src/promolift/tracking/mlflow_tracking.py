"""Thin MLflow wrapper: where runs are stored, and what lineage every run records.

By default runs go to a local SQLite store under ``mlruns/`` at the repository
root (gitignored); no MLflow server or account is required. Set
``MLFLOW_TRACKING_URI`` to point at a shared tracking server instead, and
``PROMOLIFT_EXPERIMENT_ROOT`` when that server needs experiment names to be
workspace paths (Databricks, e.g. ``/Shared/promolift``).

Callers log params/metrics/artifacts with the regular ``mlflow`` API inside
``start_run``; this module only owns configuration and lineage.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import mlflow
from mlflow import ActiveRun

from promolift.data.loader import project_root
from promolift.tracking.lineage import (
    UNKNOWN,
    fingerprint_digest,
    fingerprints_as_dicts,
    git_state,
    lockfile_sha256,
    raw_data_fingerprint,
    split_sha256,
)

logger = logging.getLogger(__name__)

TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
EXPERIMENT_ROOT_ENV = "PROMOLIFT_EXPERIMENT_ROOT"
FINGERPRINT_ARTIFACT = "lineage/raw_data_fingerprint.json"


class Experiment(StrEnum):
    """One MLflow experiment per pipeline stage, so UI comparisons stay like-for-like."""

    DATA_VALIDATION = "promolift-data-validation"
    UPLIFT_MODELS = "promolift-uplift-models"


@dataclass(frozen=True)
class TrackingConfig:
    """Where runs are recorded.

    ``artifact_root`` is None for remote servers, which own their artifact store.
    ``experiment_root`` prefixes experiment names, for servers that require
    them to be workspace paths (Databricks rejects bare names).
    """

    tracking_uri: str
    artifact_root: Path | None = None
    experiment_root: str | None = None

    def experiment_name(self, name: str) -> str:
        """Full experiment name on this tracking server."""
        if self.experiment_root is None:
            return name
        return f"{self.experiment_root.rstrip('/')}/{name}"


def local_tracking_config(store_dir: Path) -> TrackingConfig:
    """SQLite-backed store in ``store_dir``, with artifacts pinned beside it.

    MLflow 3.x rejects a plain-folder store, and with SQLite its default
    artifact location is ``./mlruns`` relative to the process cwd -- so a
    notebook run would scatter artifacts into ``notebooks/mlruns``. Pinning
    the artifact root avoids that.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    return TrackingConfig(
        tracking_uri=f"sqlite:///{(store_dir / 'mlflow.db').as_posix()}",
        artifact_root=store_dir / "artifacts",
    )


def default_tracking_config() -> TrackingConfig:
    """``MLFLOW_TRACKING_URI`` if set, else the repository-local SQLite store.

    ``PROMOLIFT_EXPERIMENT_ROOT``, if set, applies to either destination.
    """
    experiment_root = os.environ.get(EXPERIMENT_ROOT_ENV) or None
    uri = os.environ.get(TRACKING_URI_ENV)
    if uri:
        return TrackingConfig(tracking_uri=uri, experiment_root=experiment_root)
    return replace(
        local_tracking_config(project_root() / "mlruns"), experiment_root=experiment_root
    )


def _get_or_create_experiment(name: str, config: TrackingConfig) -> str:
    full_name = config.experiment_name(name)
    existing = mlflow.get_experiment_by_name(full_name)
    if existing is not None:
        return existing.experiment_id
    # Local artifact folders use the bare name: an absolute experiment root
    # joined onto a Path would escape the store (e.g. to /Shared on disk).
    artifact_location = (
        (config.artifact_root / name).as_uri() if config.artifact_root is not None else None
    )
    return mlflow.create_experiment(full_name, artifact_location=artifact_location)


@contextmanager
def start_run(
    experiment: Experiment | str,
    *,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    config: TrackingConfig | None = None,
    repo_dir: Path | None = None,
    data_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> Iterator[ActiveRun]:
    """Start an MLflow run tagged with code, environment, and raw-data lineage.

    A dirty working tree doesn't block the run (notebook iteration stays
    easy); it is tagged ``git_dirty=true`` and warned about, so such runs can
    be filtered out before choosing a final model. Lineage tags take
    precedence over same-named user ``tags``.

    Args:
        experiment: Experiment name, normally an ``Experiment`` member.
        run_name: Optional human-readable run name.
        tags: Extra run tags.
        config: Tracking destination; defaults to ``default_tracking_config()``.
        repo_dir: Repository to read git/lockfile state from (for tests).
        data_dir: Raw data directory to fingerprint (for tests).
        processed_dir: Directory holding the split file to hash (for tests).
    """
    config = config if config is not None else default_tracking_config()
    mlflow.set_tracking_uri(config.tracking_uri)
    experiment_id = _get_or_create_experiment(str(experiment), config)

    git = git_state(repo_dir)
    if git.is_dirty:
        logger.warning(
            "Working tree has uncommitted changes; run will be tagged git_dirty=true "
            "and is not reproducible from commit %s.",
            git.commit,
        )
    fingerprints = raw_data_fingerprint(data_dir)
    lineage = {
        "git_commit": git.commit,
        "git_dirty": UNKNOWN if git.is_dirty is None else str(git.is_dirty).lower(),
        "uv_lock_sha256": lockfile_sha256(repo_dir) or UNKNOWN,
        "raw_data_digest": fingerprint_digest(fingerprints),
        "split_sha256": split_sha256(processed_dir) or "missing",
        "python_version": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
    }

    with mlflow.start_run(
        experiment_id=experiment_id, run_name=run_name, tags={**(tags or {}), **lineage}
    ) as run:
        mlflow.log_dict(fingerprints_as_dicts(fingerprints), FINGERPRINT_ARTIFACT)
        yield run
