"""Shared fixtures for unit tests that exercise run lineage and MLflow tracking."""

import subprocess
from pathlib import Path

import pytest


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


@pytest.fixture
def lineage_dirs(clean_repo: Path, data_dir: Path, tmp_path: Path) -> dict[str, Path]:
    # Every lineage source points at a temp dir, so no test reads the real
    # repository, raw data, or split file.
    return {
        "repo_dir": clean_repo,
        "data_dir": data_dir,
        "processed_dir": tmp_path / "processed",
    }
