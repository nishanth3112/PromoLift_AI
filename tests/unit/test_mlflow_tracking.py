"""Unit tests for the MLflow tracking wrapper, using a throwaway local SQLite store."""

import logging
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from mlflow import MlflowClient
from mlflow.entities import RunStatus

from promolift.tracking.mlflow_tracking import (
    EXPERIMENT_ROOT_ENV,
    TRACKING_URI_ENV,
    TrackingConfig,
    default_tracking_config,
    local_tracking_config,
    start_run,
)


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory) -> TrackingConfig:
    # One store per module: creating the SQLite schema costs ~1s, so tests
    # share it and isolate themselves via unique experiment names instead.
    return local_tracking_config(tmp_path_factory.mktemp("mlruns"))


@pytest.fixture
def clean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com"]
    git += ["-c", "commit.gpgsign=false"]
    (repo / "uv.lock").write_text("version = 1\n")
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    subprocess.run([*git, "add", "uv.lock"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "initial"], cwd=repo, check=True)
    return repo


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "uplift_train.csv").write_text("client_id,treatment_flg,target\nc1,1,0\n")
    return raw


def test_local_tracking_config_uses_sqlite_inside_store_dir(tmp_path: Path) -> None:
    store = tmp_path / "store"

    config = local_tracking_config(store)

    assert config.tracking_uri == f"sqlite:///{(store / 'mlflow.db').as_posix()}"
    assert config.artifact_root == store / "artifacts"
    assert store.is_dir()


def test_default_tracking_config_honours_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TRACKING_URI_ENV, "http://tracking.example.com")
    monkeypatch.delenv(EXPERIMENT_ROOT_ENV, raising=False)

    config = default_tracking_config()

    assert config.tracking_uri == "http://tracking.example.com"
    # A remote server owns its artifact store; we must not impose a local path.
    assert config.artifact_root is None
    assert config.experiment_root is None


def test_default_tracking_config_reads_experiment_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TRACKING_URI_ENV, "databricks://promolift")
    monkeypatch.setenv(EXPERIMENT_ROOT_ENV, "/Shared/promolift")

    assert default_tracking_config().experiment_root == "/Shared/promolift"


@pytest.mark.parametrize("root", ["/Shared/promolift", "/Shared/promolift/"])
def test_experiment_name_is_prefixed_with_experiment_root(root: str) -> None:
    config = TrackingConfig(tracking_uri="databricks://promolift", experiment_root=root)

    assert config.experiment_name("promolift-data-validation") == (
        "/Shared/promolift/promolift-data-validation"
    )


def test_experiment_name_is_unchanged_without_experiment_root() -> None:
    config = TrackingConfig(tracking_uri="sqlite:///mlflow.db")

    assert config.experiment_name("promolift-data-validation") == "promolift-data-validation"


def test_start_run_records_lineage_and_user_tags(
    config: TrackingConfig, clean_repo: Path, data_dir: Path
) -> None:
    with start_run(
        "test-lineage",
        run_name="smoke",
        tags={"stage": "unit-test"},
        config=config,
        repo_dir=clean_repo,
        data_dir=data_dir,
    ) as run:
        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=config.tracking_uri)
    tags = client.get_run(run_id).data.tags

    assert tags["stage"] == "unit-test"
    assert len(tags["git_commit"]) == 40
    assert tags["git_dirty"] == "false"
    assert len(tags["uv_lock_sha256"]) == 64
    assert tags["raw_data_digest"]
    assert tags["python_version"]
    artifacts = [a.path for a in client.list_artifacts(run_id, "lineage")]
    assert artifacts == ["lineage/raw_data_fingerprint.json"]


def test_user_tags_cannot_overwrite_lineage(
    config: TrackingConfig, clean_repo: Path, data_dir: Path
) -> None:
    with start_run(
        "test-lineage-precedence",
        tags={"git_commit": "forged"},
        config=config,
        repo_dir=clean_repo,
        data_dir=data_dir,
    ) as run:
        run_id = run.info.run_id

    tags = MlflowClient(tracking_uri=config.tracking_uri).get_run(run_id).data.tags
    assert tags["git_commit"] != "forged"


def test_start_run_writes_artifacts_under_store_not_working_directory(
    config: TrackingConfig,
    clean_repo: Path,
    data_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # MLflow's SQLite default drops artifacts in ./mlruns relative to the
    # process cwd (e.g. notebooks/mlruns); the wrapper must pin them.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    with start_run(
        "test-artifact-location", config=config, repo_dir=clean_repo, data_dir=data_dir
    ) as run:
        artifact_uri = run.info.artifact_uri

    assert not (elsewhere / "mlruns").exists()
    assert config.artifact_root is not None
    assert Path(config.artifact_root).as_uri() in artifact_uri


def test_start_run_keeps_local_artifacts_under_store_when_experiment_root_is_set(
    config: TrackingConfig, clean_repo: Path, data_dir: Path
) -> None:
    # An absolute root like /Shared/promolift must only prefix the experiment
    # name; joined onto a Path it would escape the store to /Shared on disk.
    rooted = replace(config, experiment_root="/Shared/promolift")

    with start_run("test-rooted", config=rooted, repo_dir=clean_repo, data_dir=data_dir) as run:
        experiment_id = run.info.experiment_id
        artifact_uri = run.info.artifact_uri

    experiment = MlflowClient(tracking_uri=config.tracking_uri).get_experiment(experiment_id)
    assert experiment.name == "/Shared/promolift/test-rooted"
    assert config.artifact_root is not None
    assert (config.artifact_root / "test-rooted").as_uri() in artifact_uri


def test_start_run_warns_and_tags_when_tree_is_dirty(
    config: TrackingConfig,
    clean_repo: Path,
    data_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (clean_repo / "uv.lock").write_text("version = 2\n")

    with (
        caplog.at_level(logging.WARNING, logger="promolift.tracking.mlflow_tracking"),
        start_run("test-dirty", config=config, repo_dir=clean_repo, data_dir=data_dir) as run,
    ):
        run_id = run.info.run_id

    tags = MlflowClient(tracking_uri=config.tracking_uri).get_run(run_id).data.tags
    assert tags["git_dirty"] == "true"
    assert "uncommitted changes" in caplog.text


def test_start_run_marks_run_failed_on_exception(
    config: TrackingConfig, clean_repo: Path, data_dir: Path
) -> None:
    run_id = None
    with (
        pytest.raises(RuntimeError),
        start_run("test-failure", config=config, repo_dir=clean_repo, data_dir=data_dir) as run,
    ):
        run_id = run.info.run_id
        raise RuntimeError("boom")

    run_info = MlflowClient(tracking_uri=config.tracking_uri).get_run(run_id).info
    assert run_info.status == RunStatus.to_string(RunStatus.FAILED)


def test_start_run_reuses_existing_experiment(
    config: TrackingConfig, clean_repo: Path, data_dir: Path
) -> None:
    experiment_ids = []
    for _ in range(2):
        with start_run("test-reuse", config=config, repo_dir=clean_repo, data_dir=data_dir) as run:
            experiment_ids.append(run.info.experiment_id)

    assert experiment_ids[0] == experiment_ids[1]
