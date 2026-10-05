"""Log the packaged model, prove it reloads identically, register it, and promote it.

A model is logged in a ``promolift-models`` run (with the usual lineage
tags), reloaded from the store, and checked to score exactly as the
in-memory model does; only then is it registered as a new version. The
``champion`` alias -- what batch scoring loads -- moves only when asked.

On Databricks, models register in Unity Catalog (the workspace's legacy
registry is disabled); elsewhere the tracking store's own registry is used.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

import promolift
from promolift.models.base import UpliftModel
from promolift.serving.pyfunc import FeatureContract, UpliftPyfunc, drift_reference
from promolift.tracking.mlflow_tracking import (
    Experiment,
    TrackingConfig,
    default_tracking_config,
    start_run,
)

DEFAULT_MODEL_NAME = "promolift_ai.models.uplift_targeting"
CHAMPION_ALIAS = "champion"
MODEL_ARTIFACT = "model"
# Run tags copied onto each registered version, so a version is traceable on its own.
VERSION_TAGS = (
    "git_commit",
    "git_dirty",
    "split_sha256",
    "raw_data_digest",
    "uv_lock_sha256",
    "model_name",
    "base_model",
    "fit_split",
    "feature_code_sha256",
    "leaderboard_run_id",
    "final_evaluation_run_id",
)
# What the pickled model needs at load time; versions pinned to this environment.
_RUNTIME_PACKAGES = (
    "mlflow",
    "scikit-uplift",
    "lightgbm",
    "scikit-learn",
    "pandas",
    "numpy",
    "cloudpickle",
)


class ParityError(RuntimeError):
    """The reloaded model doesn't score exactly as the in-memory model."""


def registry_uri_for(tracking_uri: str) -> str:
    """Unity Catalog for a Databricks tracking URI (same profile), else the tracking store."""
    if tracking_uri == "databricks" or tracking_uri.startswith("databricks://"):
        return tracking_uri.replace("databricks", "databricks-uc", 1)
    return tracking_uri


def pip_requirements() -> list[str]:
    """Pinned requirements for loading the model elsewhere, from the installed versions."""
    return [f"{name}=={importlib.metadata.version(name)}" for name in _RUNTIME_PACKAGES]


@dataclass(frozen=True)
class PackagedModel:
    """A logged model: where it lives, its contract, and the scores it must reproduce."""

    run_id: str
    model_uri: str
    contract: FeatureContract
    model_input: pd.DataFrame
    expected_scores: np.ndarray


def log_packaged_model(
    model: UpliftModel,
    features: pd.DataFrame,
    *,
    label: str,
    base_model: str,
    tags: dict[str, str] | None = None,
    config: TrackingConfig | None = None,
    lineage: dict[str, Path] | None = None,
) -> PackagedModel:
    """Log ``model`` with its contract, signature, and drift reference in one run.

    Args:
        model: A model already fitted on ``features``' clients.
        features: The fitting frame's features (``ModelFrame.features``).
        label: The variant's name, e.g. ``class_transformation_tuned``.
        tags: Extra run tags (e.g. ``fit_split``, ``final_evaluation_run_id``).

    Raises:
        ParityError: If the canonical input doesn't reproduce the model's
            scores on the training layout -- the contract would change predictions.
    """
    contract = FeatureContract.from_frame(features)
    model_input = contract.to_input(features)
    expected = np.asarray(model.predict_uplift(features), dtype=float)
    through_contract = np.asarray(model.predict_uplift(contract.prepare(model_input)), dtype=float)
    if not np.array_equal(expected, through_contract):
        raise ParityError("Scoring the canonical input changes the model's predictions")

    with start_run(
        Experiment.MODELS,
        run_name=label,
        tags={"run_type": "model_package", "model_name": label, "base_model": base_model}
        | (tags or {}),
        config=config,
        **(lineage or {}),
    ) as run:
        mlflow.log_params(model.params())
        mlflow.log_metric("n_fit_clients", len(features))
        info = mlflow.pyfunc.log_model(
            name=MODEL_ARTIFACT,
            python_model=UpliftPyfunc(model, contract),
            signature=infer_signature(model_input.head(1_000), expected[:1_000]),
            input_example=model_input.head(5),
            code_paths=[str(Path(promolift.__file__).parent)],
            pip_requirements=pip_requirements(),
        )
        mlflow.log_dict(contract.as_dict(), "contract/feature_contract.json")
        mlflow.log_dict(drift_reference(features, contract), "contract/drift_reference.json")
        return PackagedModel(run.info.run_id, info.model_uri, contract, model_input, expected)


def check_reload_parity(packaged: PackagedModel) -> None:
    """Reload the logged model from the store and require bit-identical scores.

    Raises:
        ParityError: If any score differs.
    """
    loaded = mlflow.pyfunc.load_model(packaged.model_uri)
    scores = np.asarray(loaded.predict(packaged.model_input), dtype=float)
    if not np.array_equal(scores, packaged.expected_scores):
        gap = float(np.nanmax(np.abs(scores - packaged.expected_scores)))
        raise ParityError(f"The reloaded model's scores differ (max abs difference {gap:.3g})")


def _client(config: TrackingConfig) -> MlflowClient:
    return MlflowClient(
        tracking_uri=config.tracking_uri, registry_uri=registry_uri_for(config.tracking_uri)
    )


def register(
    packaged: PackagedModel, name: str = DEFAULT_MODEL_NAME, config: TrackingConfig | None = None
) -> str:
    """Register the logged model as a new version of ``name``; return the version.

    The version carries the run's lineage tags (``VERSION_TAGS``), so it can
    be traced without opening the run.
    """
    config = config if config is not None else default_tracking_config()
    client = _client(config)
    run_tags = client.get_run(packaged.run_id).data.tags
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_registry_uri(registry_uri_for(config.tracking_uri))
    version = mlflow.register_model(
        packaged.model_uri,
        name,
        tags={key: run_tags[key] for key in VERSION_TAGS if key in run_tags},
    )
    return str(version.version)


def set_champion(
    version: str, name: str = DEFAULT_MODEL_NAME, config: TrackingConfig | None = None
) -> None:
    """Point the ``champion`` alias -- what batch scoring loads -- at ``version``."""
    config = config if config is not None else default_tracking_config()
    _client(config).set_registered_model_alias(name, CHAMPION_ALIAS, version)


def champion_uri(name: str = DEFAULT_MODEL_NAME) -> str:
    """The model URI batch scoring loads."""
    return f"models:/{name}@{CHAMPION_ALIAS}"
