"""Shared fixtures: run lineage / MLflow tracking, and a synthetic uplift experiment."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
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


def _sure_things_experiment(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Sure things (b >= 0.5) buy 80% of the time, contacted or not. Persuadables
    # buy 10%, or 40% if contacted. Among treated clients sure things still buy
    # more (80% vs 40%), so a model of who buys ranks exactly the clients the
    # SMS can't move first.
    rng = np.random.default_rng(seed)
    b = rng.random(n)
    features = pd.DataFrame(
        {
            "b": b,
            "noise": rng.random(n),
            "segment": pd.Categorical(rng.choice(["F", "M", "U"], n), categories=["F", "M", "U"]),
        }
    )
    treatment = rng.integers(0, 2, n)
    outcome = (rng.random(n) < np.where(b >= 0.5, 0.8, 0.1 + 0.3 * treatment)).astype(int)
    return features, treatment, outcome


@pytest.fixture(scope="session")
def sure_things_data() -> dict:
    """Train (6k) and holdout (4k) clients from the sure-things experiment."""
    return {
        "train": _sure_things_experiment(6_000, seed=0),
        "holdout": _sure_things_experiment(4_000, seed=1),
    }
