"""Shared fixtures: run lineage / MLflow tracking, a synthetic uplift experiment, and a champion."""

import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
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


@pytest.fixture
def feature_raw_dir(tmp_path: Path) -> Path:
    """Raw clients/products/purchases with every column any feature group reads."""
    raw = tmp_path / "feature_raw"
    raw.mkdir()
    (raw / "clients.csv").write_text(
        "client_id,first_issue_date,first_redeem_date,age,gender\n"
        "c1,2018-01-01 00:00:00,2018-02-01 00:00:00,30,F\n"
        "c2,2018-01-01 00:00:00,,40,M\n"
    )
    (raw / "products.csv").write_text(
        "product_id,level_1,level_2,level_3,level_4,segment_id,brand_id,vendor_id,"
        "netto,is_own_trademark,is_alcohol\n"
        "p1,a,x,c,d,1.0,b,v,1.0,0,0\n"
        "p2,a,y,c,d,1.0,b,v,1.0,1,0\n"
    )
    (raw / "purchases.csv").write_text(
        "client_id,transaction_id,transaction_datetime,regular_points_received,"
        "express_points_received,regular_points_spent,express_points_spent,purchase_sum,"
        "store_id,product_id,product_quantity,trn_sum_from_iss,trn_sum_from_red\n"
        "c1,t1,2019-01-01 10:00:00,10.0,0.0,0.0,0.0,100.0,s1,p1,1.0,60.0,\n"
        "c1,t1,2019-01-01 10:00:00,10.0,0.0,0.0,0.0,100.0,s1,p2,2.0,40.0,\n"
        "c1,t2,2019-01-05 18:00:00,0.0,5.0,-20.0,0.0,50.0,s2,p1,1.0,30.0,50.0\n"
    )
    return raw


@pytest.fixture
def client_features(feature_raw_dir: Path) -> pl.DataFrame:
    """Real feature code on the tiny raw files: c1 has purchases, c2 has none (null
    purchase features), and the table has boolean columns."""
    from promolift.models.dataset import build_model_features

    return build_model_features(feature_raw_dir)


@pytest.fixture
def make_training_frame() -> Callable[[pl.DataFrame, int, int], pd.DataFrame]:
    """Training rows in the layout of ``client_features``, converted as ``model_frame`` does."""

    def make(features: pl.DataFrame, n: int, seed: int) -> pd.DataFrame:
        # Rows resampled from the clients, so a batch of those clients doesn't drift.
        base = (
            features.drop("client_id")
            .with_columns(pl.col(pl.Boolean).cast(pl.Int8))
            .to_pandas()
            .sample(n, replace=True, random_state=seed)
            .reset_index(drop=True)
        )
        base["gender"] = pd.Categorical(base["gender"], categories=["F", "M", "U"])
        return base

    return make


@pytest.fixture
def scoring_champion(
    client_features: pl.DataFrame,
    make_training_frame: Callable,
    lineage_dirs: dict[str, Path],
    tmp_path: Path,
) -> SimpleNamespace:
    """A model for ``client_features`` registered in a local registry as version 1, champion."""
    from promolift.models.registry import build_model
    from promolift.serving.registration import log_packaged_model, register, set_champion
    from promolift.tracking.mlflow_tracking import local_tracking_config

    name = "uplift_scoring_test"
    config = local_tracking_config(tmp_path / "mlruns")
    training = make_training_frame(client_features, 400, 0)
    rng = np.random.default_rng(1)
    model = build_model(
        "class_transformation", seed=0, overrides={"n_estimators": 10, "min_child_samples": 5}
    ).fit(training, rng.integers(0, 2, 400), rng.integers(0, 2, 400))
    packaged = log_packaged_model(
        model,
        training,
        label="ct_test",
        base_model="class_transformation",
        config=config,
        lineage=lineage_dirs,
    )
    set_champion(register(packaged, name, config), name, config)
    return SimpleNamespace(name=name, config=config, packaged=packaged)
