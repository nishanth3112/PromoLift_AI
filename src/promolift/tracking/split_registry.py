"""Canonical train/val/test split, published as an MLflow artifact on the tracking server.

``data/processed/`` is gitignored, so the tracking server is the single source
of truth for which clients are in which split: one ``canonical=true`` run in
the ``promolift-data-split`` experiment holds the parquet, and its
``split_sha256`` lineage tag lets every download be verified.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import polars as pl
from mlflow import MlflowClient

from promolift.data.split import split_assignment_path, split_content_sha256, write_split
from promolift.tracking.lineage import split_sha256
from promolift.tracking.mlflow_tracking import (
    Experiment,
    TrackingConfig,
    default_tracking_config,
    start_run,
)

CANONICAL_TAG = "canonical"
SPLIT_ARTIFACT_PATH = "split/split_assignment.parquet"
_REPORT_ARTIFACT = "reports/split_validation.json"
_CANONICAL_FILTER = f"tags.{CANONICAL_TAG} = 'true' and attributes.status = 'FINISHED'"


class SplitRegistryError(RuntimeError):
    """No usable canonical split on the tracking server."""


@dataclass(frozen=True)
class FetchResult:
    """Outcome of syncing the local split with the canonical one."""

    path: Path
    run_id: str
    split_sha256: str
    changed: bool


def _canonical_runs(client: MlflowClient, config: TrackingConfig) -> list:
    experiment = client.get_experiment_by_name(config.experiment_name(Experiment.DATA_SPLIT))
    if experiment is None:
        return []
    return client.search_runs(
        [experiment.experiment_id],
        filter_string=_CANONICAL_FILTER,
        order_by=["attributes.start_time DESC"],
    )


def publish_split(
    report: dict,
    *,
    params: dict | None = None,
    metrics: dict[str, float] | None = None,
    config: TrackingConfig | None = None,
    repo_dir: Path | None = None,
    data_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> str:
    """Publish the local split file as the new canonical split; return the run id.

    Earlier canonical runs are re-tagged ``superseded`` only after the new run
    finishes, so a failed publish never leaves the server without a canonical split.

    Args:
        report: Validation evidence to store alongside the split (JSON-serializable).
        params: Parameters that produced the split, e.g. the seed.
        metrics: Headline numbers to make comparable in the MLflow UI.
        config: Tracking destination; defaults to ``default_tracking_config()``.
        repo_dir, data_dir, processed_dir: Lineage/split locations (for tests).

    Raises:
        FileNotFoundError: If there is no local split file to publish.
    """
    config = config if config is not None else default_tracking_config()
    path = split_assignment_path(processed_dir)
    if not path.exists():
        raise FileNotFoundError(f"No split to publish at {path}")

    with start_run(
        Experiment.DATA_SPLIT,
        run_name="canonical-split",
        tags={CANONICAL_TAG: "true"},
        config=config,
        repo_dir=repo_dir,
        data_dir=data_dir,
        processed_dir=processed_dir,
    ) as run:
        mlflow.log_params(params or {})
        mlflow.log_metrics(metrics or {})
        mlflow.log_artifact(str(path), artifact_path=str(Path(SPLIT_ARTIFACT_PATH).parent))
        mlflow.log_dict(report, _REPORT_ARTIFACT)
        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=config.tracking_uri)
    for previous in _canonical_runs(client, config):
        if previous.info.run_id != run_id:
            client.set_tag(previous.info.run_id, CANONICAL_TAG, "superseded")
    return run_id


def fetch_canonical_split(
    *,
    config: TrackingConfig | None = None,
    processed_dir: Path | None = None,
    force: bool = False,
) -> FetchResult:
    """Download the canonical split, verify its hash, and write it locally.

    A local split that already matches is left untouched. A *different* local
    split is only replaced with ``force``, since results evaluated on it would
    no longer be comparable.

    Raises:
        SplitRegistryError: If nothing is published, or the artifact's content
            doesn't match the hash recorded on its run.
        FileExistsError: If a different local split exists and ``force`` is False.
    """
    config = config if config is not None else default_tracking_config()
    client = MlflowClient(tracking_uri=config.tracking_uri)
    runs = _canonical_runs(client, config)
    if not runs:
        msg = "No canonical split published yet; run `scripts/make_split.py` first."
        raise SplitRegistryError(msg)
    run = runs[0]
    expected = run.data.tags["split_sha256"]

    destination = split_assignment_path(processed_dir)
    if destination.exists() and split_sha256(processed_dir) == expected:
        return FetchResult(destination, run.info.run_id, expected, changed=False)

    with tempfile.TemporaryDirectory() as tmp:
        downloaded = client.download_artifacts(run.info.run_id, SPLIT_ARTIFACT_PATH, tmp)
        assignment = pl.read_parquet(downloaded)
    actual = split_content_sha256(assignment)
    if actual != expected:
        msg = (
            f"Canonical split artifact of run {run.info.run_id} has content hash "
            f"{actual}, but the run recorded {expected}; refusing to use it."
        )
        raise SplitRegistryError(msg)

    write_split(assignment, destination, force=force)
    return FetchResult(destination, run.info.run_id, expected, changed=True)
