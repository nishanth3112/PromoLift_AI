"""Unit tests for batch scoring: model input, ranking and send flags, and the scoring run."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from mlflow import MlflowClient

from promolift.models.dataset import build_model_features
from promolift.models.registry import build_model
from promolift.serving.pyfunc import ContractError, FeatureContract
from promolift.serving.registration import log_packaged_model, register, set_champion
from promolift.serving.scoring import (
    SEND_LIST_COLUMNS,
    decile_summary,
    load_champion,
    model_input,
    run_batch_scoring,
    send_list,
)
from promolift.tracking.mlflow_tracking import TrackingConfig, local_tracking_config

_NAME = "uplift_scoring_test"
_AT = datetime(2026, 10, 5, 9, 30, tzinfo=UTC)


def _send_list(scores, client_ids=None, share=0.3):
    ids = client_ids if client_ids is not None else [f"c{i:02d}" for i in range(len(scores))]
    return send_list(
        pl.Series(ids),
        np.asarray(scores, dtype=float),
        send_share=share,
        model_name=_NAME,
        model_version="1",
        scored_at=_AT,
        scoring_run_id="run123",
    )


# --- send_list ---------------------------------------------------------------


def test_ranks_by_score_and_flags_the_top_share() -> None:
    targets = _send_list([0.1, 0.5, -0.2, 0.3, 0.0, 0.4, 0.2, -0.1, 0.05, 0.15])

    assert targets["rank"].to_list() == list(range(1, 11))
    assert targets["uplift_score"].is_sorted(descending=True)
    assert targets["client_id"][:3].to_list() == ["c01", "c05", "c03"]
    assert targets["send"].to_list() == [True] * 3 + [False] * 7
    assert targets["decile"].to_list() == list(range(1, 11))
    assert tuple(targets.columns) == SEND_LIST_COLUMNS


def test_ties_are_broken_by_client_id() -> None:
    targets = _send_list([0.2, 0.2, 0.2], client_ids=["b", "c", "a"], share=1 / 3)

    assert targets["client_id"].to_list() == ["a", "b", "c"]
    assert targets["send"].to_list() == [True, False, False]


def test_deciles_split_the_ranking_into_tenths() -> None:
    targets = _send_list(np.linspace(1, 0, 25))

    counts = decile_summary(targets)["clients"].to_list()
    assert len(counts) == 10
    assert sum(counts) == 25
    assert max(counts) - min(counts) <= 1
    assert targets["decile"].is_sorted()


@pytest.mark.parametrize(("share", "expected"), [(0.0, 0), (1.0, 10), (0.38, 4), (0.35, 4)])
def test_the_number_sent_is_the_rounded_share(share: float, expected: int) -> None:
    assert int(_send_list(np.arange(10), share=share)["send"].sum()) == expected


def test_send_list_records_its_provenance() -> None:
    targets = _send_list([0.3, 0.1])

    row = targets.row(0, named=True)
    assert (row["model_name"], row["model_version"], row["scoring_run_id"]) == (
        _NAME,
        "1",
        "run123",
    )
    assert row["scored_at"] == _AT


def test_send_list_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        send_list(
            pl.Series(["a"]),
            np.array([0.1, 0.2]),
            send_share=0.5,
            model_name=_NAME,
            model_version="1",
            scored_at=_AT,
            scoring_run_id="r",
        )
    with pytest.raises(ValueError, match="NaN"):
        _send_list([0.1, np.nan])
    with pytest.raises(ValueError, match="send_share"):
        _send_list([0.1, 0.2], share=1.5)


# --- model input and end-to-end scoring ------------------------------------------


@pytest.fixture
def features(feature_raw_dir: Path) -> pl.DataFrame:
    # Real feature code on tiny raw files: c1 has purchases, c2 has none
    # (null purchase features), and the table has boolean columns.
    return build_model_features(feature_raw_dir)


def _training_frame(features: pl.DataFrame, n: int, seed: int) -> pd.DataFrame:
    # Converted exactly as models.dataset.model_frame converts training data.
    rng = np.random.default_rng(seed)
    base = (
        features.drop("client_id")
        .with_columns(pl.col(pl.Boolean).cast(pl.Int8))
        .to_pandas()
        .sample(n, replace=True, random_state=seed)
        .reset_index(drop=True)
    )
    base["gender"] = pd.Categorical(rng.choice(["F", "M", "U"], n), categories=["F", "M", "U"])
    return base


@pytest.fixture
def config(tmp_path: Path) -> TrackingConfig:
    return local_tracking_config(tmp_path / "mlruns")


@pytest.fixture
def champion(features: pl.DataFrame, config: TrackingConfig, lineage_dirs: dict[str, Path]):
    training = _training_frame(features, 400, seed=0)
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
    version = register(packaged, _NAME, config)
    set_champion(version, _NAME, config)
    return packaged


def test_model_input_matches_the_contract_layout(features: pl.DataFrame) -> None:
    contract = FeatureContract.from_frame(_training_frame(features, 50, seed=0))

    frame = model_input(features, contract)

    assert tuple(frame.columns) == contract.columns
    numeric = [c for c in contract.columns if c != "gender"]
    assert (frame.dtypes[numeric] == "float64").all()
    assert set(frame["has_redeemed"].dropna()) <= {0.0, 1.0}
    assert frame["gender"].tolist() == ["F", "M"]
    # c2 has no purchases: its purchase features are missing, not zero.
    assert np.isnan(frame.loc[1, "monetary_avg"])


def test_model_input_rejects_a_table_missing_contract_columns(features: pl.DataFrame) -> None:
    contract = FeatureContract.from_frame(_training_frame(features, 50, seed=0))

    with pytest.raises(ContractError, match="lacks contract columns"):
        model_input(features.drop("frequency"), contract)


def test_load_champion_resolves_the_alias_to_a_version(champion, config: TrackingConfig) -> None:
    loaded = load_champion(_NAME, config)

    assert loaded.version == "1"
    assert loaded.contract == champion.contract


def test_scores_every_client_and_writes_the_batch(
    champion, features: pl.DataFrame, config: TrackingConfig, lineage_dirs, tmp_path: Path
) -> None:
    result = run_batch_scoring(
        features,
        send_share=0.5,
        model_name=_NAME,
        output_dir=tmp_path / "scoring",
        scored_at=_AT,
        config=config,
        lineage=lineage_dirs,
    )

    written = pl.read_parquet(result.path)
    assert written.height == features.height == 2
    assert set(written["client_id"]) == {"c1", "c2"}
    assert written["send"].sum() == 1
    assert written["model_version"].unique().to_list() == ["1"]
    assert written["scoring_run_id"].unique().to_list() == [result.run_id]
    assert np.isfinite(written["uplift_score"].to_numpy()).all()

    run = MlflowClient(tracking_uri=config.tracking_uri).get_run(result.run_id)
    assert run.data.tags["run_type"] == "batch_scoring"
    assert run.data.tags["model_version"] == "1"
    assert run.data.metrics["n_clients"] == 2
    assert run.data.metrics["n_send"] == 1
    assert run.data.params["send_share"] == "0.5"


def test_batches_are_kept_and_follow_the_champion(
    champion, features: pl.DataFrame, config: TrackingConfig, lineage_dirs, tmp_path: Path
) -> None:
    output = tmp_path / "scoring"
    first = run_batch_scoring(
        features, send_share=0.5, model_name=_NAME, output_dir=output, scored_at=_AT,
        config=config, lineage=lineage_dirs,
    )  # fmt: skip
    set_champion(register(champion, _NAME, config), _NAME, config)
    second = run_batch_scoring(
        features, send_share=0.5, model_name=_NAME, output_dir=output,
        scored_at=datetime(2026, 10, 12, tzinfo=UTC), config=config, lineage=lineage_dirs,
    )  # fmt: skip

    assert (first.model_version, second.model_version) == ("1", "2")
    assert first.path.exists()
    assert second.path.exists()
    assert len(list(output.glob("send_list_*.parquet"))) == 2
