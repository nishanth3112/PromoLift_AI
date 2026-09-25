"""Unit tests for publishing and fetching the canonical split via a local SQLite MLflow store."""

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import polars as pl
import pytest
from mlflow import MlflowClient

from promolift.data.split import (
    load_split,
    split_assignment_path,
    split_content_sha256,
    write_split,
)
from promolift.tracking.mlflow_tracking import TrackingConfig, local_tracking_config
from promolift.tracking.split_registry import (
    CANONICAL_TAG,
    SPLIT_ARTIFACT_PATH,
    SplitRegistryError,
    fetch_canonical_split,
    publish_split,
)

_ASSIGNMENT = pl.DataFrame(
    {"client_id": ["c1", "c2", "c3", "c4", "c5"], "split": ["train"] * 3 + ["val", "test"]}
)
_OTHER_ASSIGNMENT = _ASSIGNMENT.with_columns(pl.lit("train").alias("split"))


@pytest.fixture
def config(tmp_path: Path) -> TrackingConfig:
    # Fresh store per test: "latest canonical run" must not leak across tests.
    return local_tracking_config(tmp_path / "mlruns")


def _publish(
    assignment: pl.DataFrame, config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> str:
    write_split(assignment, split_assignment_path(lineage_dirs["processed_dir"]), force=True)
    return publish_split({"note": "unit test"}, config=config, **lineage_dirs)


def test_fetch_returns_exactly_what_was_published(
    config: TrackingConfig, lineage_dirs: dict[str, Path], tmp_path: Path
) -> None:
    run_id = _publish(_ASSIGNMENT, config, lineage_dirs)
    other_machine = tmp_path / "other_machine"

    result = fetch_canonical_split(config=config, processed_dir=other_machine)

    assert result.run_id == run_id
    assert result.changed is True
    assert result.split_sha256 == split_content_sha256(_ASSIGNMENT)
    assert split_content_sha256(load_split(result.path)) == split_content_sha256(_ASSIGNMENT)


def test_publish_records_the_split_hash_and_canonical_tag(
    config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    run_id = _publish(_ASSIGNMENT, config, lineage_dirs)

    client = MlflowClient(tracking_uri=config.tracking_uri)
    tags = client.get_run(run_id).data.tags
    assert tags[CANONICAL_TAG] == "true"
    assert tags["split_sha256"] == split_content_sha256(_ASSIGNMENT)
    assert [a.path for a in client.list_artifacts(run_id, "split")] == [SPLIT_ARTIFACT_PATH]


def test_publishing_again_supersedes_the_previous_canonical_run(
    config: TrackingConfig, lineage_dirs: dict[str, Path], tmp_path: Path
) -> None:
    first = _publish(_ASSIGNMENT, config, lineage_dirs)
    second = _publish(_OTHER_ASSIGNMENT, config, lineage_dirs)

    client = MlflowClient(tracking_uri=config.tracking_uri)
    assert client.get_run(first).data.tags[CANONICAL_TAG] == "superseded"
    result = fetch_canonical_split(config=config, processed_dir=tmp_path / "fresh")
    assert result.run_id == second


def test_fetch_rejects_an_artifact_that_does_not_match_its_recorded_hash(
    config: TrackingConfig, lineage_dirs: dict[str, Path], tmp_path: Path
) -> None:
    run_id = _publish(_ASSIGNMENT, config, lineage_dirs)
    artifact_uri = MlflowClient(tracking_uri=config.tracking_uri).get_run(run_id).info.artifact_uri
    stored = Path(url2pathname(urlparse(artifact_uri).path)) / SPLIT_ARTIFACT_PATH
    _OTHER_ASSIGNMENT.write_parquet(stored)

    with pytest.raises(SplitRegistryError, match="hash"):
        fetch_canonical_split(config=config, processed_dir=tmp_path / "fresh")


def test_fetch_is_a_no_op_when_local_split_already_matches(
    config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    _publish(_ASSIGNMENT, config, lineage_dirs)

    result = fetch_canonical_split(config=config, processed_dir=lineage_dirs["processed_dir"])

    assert result.changed is False


def test_fetch_refuses_to_replace_a_different_local_split_without_force(
    config: TrackingConfig, lineage_dirs: dict[str, Path], tmp_path: Path
) -> None:
    _publish(_ASSIGNMENT, config, lineage_dirs)
    local = tmp_path / "diverged"
    write_split(_OTHER_ASSIGNMENT, split_assignment_path(local))

    with pytest.raises(FileExistsError, match="--force"):
        fetch_canonical_split(config=config, processed_dir=local)

    result = fetch_canonical_split(config=config, processed_dir=local, force=True)
    assert result.changed is True
    assert split_content_sha256(load_split(result.path)) == split_content_sha256(_ASSIGNMENT)


def test_fetch_explains_when_nothing_has_been_published(
    config: TrackingConfig, tmp_path: Path
) -> None:
    with pytest.raises(SplitRegistryError, match="make_split"):
        fetch_canonical_split(config=config, processed_dir=tmp_path / "fresh")


def test_publish_requires_the_split_file_to_exist(
    config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    with pytest.raises(FileNotFoundError):
        publish_split({}, config=config, **lineage_dirs)
