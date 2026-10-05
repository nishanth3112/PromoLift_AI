"""Unit tests for the packaged model: contract, drift reference, logging, registry, champion."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from promolift.models.registry import build_model
from promolift.serving.pyfunc import (
    ContractError,
    FeatureContract,
    UpliftPyfunc,
    drift_reference,
)
from promolift.serving.registration import (
    CHAMPION_ALIAS,
    ParityError,
    champion_uri,
    check_reload_parity,
    log_packaged_model,
    register,
    registry_uri_for,
    set_champion,
)
from promolift.tracking.mlflow_tracking import TrackingConfig, local_tracking_config

_NAME = "uplift_test"


def _training(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Mirrors the real feature table's dtypes: ints, an int8 flag, a float with
    # nulls, and a categorical with a level that never occurs ("U").
    rng = np.random.default_rng(seed)
    b = rng.random(n)
    features = pd.DataFrame(
        {
            "frequency": rng.integers(1, 50, n),
            "has_redeemed": rng.integers(0, 2, n).astype("int8"),
            "monetary_avg": np.where(rng.random(n) < 0.1, np.nan, b * 500),
            "gender": pd.Categorical(rng.choice(["F", "M"], n), categories=["F", "M", "U"]),
        }
    )
    treatment = rng.integers(0, 2, n)
    outcome = (rng.random(n) < np.where(b < 0.3, 0.1 + 0.4 * treatment, 0.4)).astype(int)
    return features, treatment, outcome


@pytest.fixture(scope="module")
def trained() -> tuple:
    features, treatment, outcome = _training(3_000, seed=0)
    model = build_model("class_transformation", seed=0, overrides={"n_estimators": 30})
    return model.fit(features, treatment, outcome), features


@pytest.fixture
def config(tmp_path: Path) -> TrackingConfig:
    return local_tracking_config(tmp_path / "mlruns")


# --- FeatureContract ----------------------------------------------------------


def test_contract_records_columns_in_order_and_category_levels(trained) -> None:
    _, features = trained

    contract = FeatureContract.from_frame(features)

    assert contract.columns == ("frequency", "has_redeemed", "monetary_avg", "gender")
    assert contract.categories == {"gender": ("F", "M", "U")}
    assert FeatureContract.from_dict(contract.as_dict()) == contract


def test_canonical_input_scores_exactly_like_the_training_layout(trained) -> None:
    model, features = trained
    contract = FeatureContract.from_frame(features)

    canonical = contract.to_input(features)

    assert (canonical.dtypes[["frequency", "has_redeemed", "monetary_avg"]] == "float64").all()
    assert canonical["gender"].dtype == object
    np.testing.assert_array_equal(
        model.predict_uplift(contract.prepare(canonical)), model.predict_uplift(features)
    )


def test_prepare_restores_column_order(trained) -> None:
    model, features = trained
    contract = FeatureContract.from_frame(features)
    shuffled = contract.to_input(features)[["gender", "monetary_avg", "frequency", "has_redeemed"]]

    prepared = contract.prepare(shuffled)

    assert tuple(prepared.columns) == contract.columns
    np.testing.assert_array_equal(model.predict_uplift(prepared), model.predict_uplift(features))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda f: f.drop(columns="frequency"), "missing \\['frequency'\\]"),
        (lambda f: f.assign(client_id="c1"), "extra \\['client_id'\\]"),
        (lambda f: f.assign(frequency="many"), "must be numeric"),
        (lambda f: f.assign(gender="X"), "unknown categories \\['X'\\]"),
    ],
)
def test_prepare_rejects_frames_that_break_the_contract(trained, change, message: str) -> None:
    _, features = trained
    contract = FeatureContract.from_frame(features)

    with pytest.raises(ContractError, match=message):
        contract.prepare(change(contract.to_input(features)))


def test_prepare_keeps_nulls_as_missing(trained) -> None:
    _, features = trained
    contract = FeatureContract.from_frame(features)
    frame = contract.to_input(features.head(3))
    frame.loc[frame.index[0], "gender"] = None

    prepared = contract.prepare(frame)

    assert pd.isna(prepared["gender"].iloc[0])
    assert prepared["monetary_avg"].isna().sum() == features.head(3)["monetary_avg"].isna().sum()


def test_contract_rejects_unsupported_training_columns() -> None:
    with pytest.raises(ContractError, match="unsupported dtype"):
        FeatureContract.from_frame(pd.DataFrame({"name": ["a", "b"]}))


def test_drift_reference_summarizes_each_feature(trained) -> None:
    _, features = trained
    contract = FeatureContract.from_frame(features)

    reference = drift_reference(features, contract)

    assert reference["n_clients"] == len(features)
    numeric = reference["features"]["monetary_avg"]
    assert numeric["kind"] == "numeric"
    assert numeric["null_share"] == pytest.approx(features["monetary_avg"].isna().mean())
    quantiles = list(numeric["quantiles"].values())
    assert len(quantiles) == 11
    assert quantiles == sorted(quantiles)
    gender = reference["features"]["gender"]
    assert gender["kind"] == "categorical"
    assert set(gender["shares"]) == {"F", "M", "U"}
    assert gender["shares"]["U"] == 0
    assert sum(gender["shares"].values()) == pytest.approx(1.0)


def test_pyfunc_predicts_through_the_contract(trained) -> None:
    model, features = trained
    contract = FeatureContract.from_frame(features)

    scores = UpliftPyfunc(model, contract).predict(None, contract.to_input(features))

    np.testing.assert_array_equal(scores, model.predict_uplift(features))


# --- logging, registry, champion ------------------------------------------------


@pytest.fixture
def packaged(trained, config: TrackingConfig, lineage_dirs: dict[str, Path]):
    model, features = trained
    return log_packaged_model(
        model,
        features,
        label="ct_tuned",
        base_model="class_transformation",
        tags={"fit_split": "train+val", "final_evaluation_run_id": "final123"},
        config=config,
        lineage=lineage_dirs,
    )


def test_logs_the_model_contract_and_drift_reference(packaged, config: TrackingConfig) -> None:
    client = MlflowClient(tracking_uri=config.tracking_uri)
    run = client.get_run(packaged.run_id)

    assert run.data.tags["run_type"] == "model_package"
    assert run.data.tags["model_name"] == "ct_tuned"
    assert run.data.params["model"] == "class_transformation"
    artifacts = {a.path for a in client.list_artifacts(packaged.run_id, "contract")}
    assert artifacts == {"contract/feature_contract.json", "contract/drift_reference.json"}
    signature = mlflow.models.get_model_info(packaged.model_uri).signature
    assert [c.name for c in signature.inputs.inputs] == list(packaged.contract.columns)
    assert signature.outputs is not None


def test_the_reloaded_model_scores_identically(packaged) -> None:
    check_reload_parity(packaged)


def test_the_reloaded_model_accepts_missing_values(packaged) -> None:
    frame = packaged.model_input.head(4).copy()
    frame.loc[frame.index[0], "gender"] = None
    frame.loc[frame.index[1], "monetary_avg"] = np.nan

    scores = mlflow.pyfunc.load_model(packaged.model_uri).predict(frame)

    assert len(scores) == 4
    assert np.isfinite(scores).all()


def test_the_reloaded_model_rejects_unknown_categories(packaged) -> None:
    frame = packaged.model_input.head(2).copy()
    frame["gender"] = "X"

    with pytest.raises(Exception, match="unknown categories"):
        mlflow.pyfunc.load_model(packaged.model_uri).predict(frame)


def test_a_parity_mismatch_is_an_error(packaged) -> None:
    tampered = type(packaged)(
        packaged.run_id,
        packaged.model_uri,
        packaged.contract,
        packaged.model_input,
        packaged.expected_scores + 1e-9,
    )

    with pytest.raises(ParityError, match="differ"):
        check_reload_parity(tampered)


def test_registering_creates_versions_traceable_to_their_run(
    packaged, config: TrackingConfig
) -> None:
    first = register(packaged, _NAME, config)
    second = register(packaged, _NAME, config)

    assert (first, second) == ("1", "2")
    tags = MlflowClient(tracking_uri=config.tracking_uri).get_model_version(_NAME, "1").tags
    assert tags["model_name"] == "ct_tuned"
    assert tags["final_evaluation_run_id"] == "final123"
    assert tags["git_dirty"] == "false"
    assert {"git_commit", "split_sha256", "raw_data_digest"} <= set(tags)


def test_registering_does_not_move_the_champion(packaged, config: TrackingConfig) -> None:
    register(packaged, _NAME, config)

    aliases = MlflowClient(tracking_uri=config.tracking_uri).get_registered_model(_NAME).aliases

    assert CHAMPION_ALIAS not in aliases


def test_the_champion_loads_and_scores_like_the_logged_model(
    packaged, config: TrackingConfig
) -> None:
    register(packaged, _NAME, config)
    version = register(packaged, _NAME, config)

    set_champion(version, _NAME, config)

    client = MlflowClient(tracking_uri=config.tracking_uri)
    assert str(client.get_model_version_by_alias(_NAME, CHAMPION_ALIAS).version) == version
    scores = mlflow.pyfunc.load_model(champion_uri(_NAME)).predict(packaged.model_input)
    np.testing.assert_array_equal(scores, packaged.expected_scores)


@pytest.mark.parametrize(
    ("tracking", "registry"),
    [
        ("databricks://promolift", "databricks-uc://promolift"),
        ("databricks", "databricks-uc"),
        ("sqlite:///x/mlflow.db", "sqlite:///x/mlflow.db"),
    ],
)
def test_registry_uri_follows_the_tracking_store(tracking: str, registry: str) -> None:
    assert registry_uri_for(tracking) == registry
