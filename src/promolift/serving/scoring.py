"""Batch scoring: rank every client with the champion model and flag the top share to text.

The champion is resolved to a concrete version once and loaded by that
version, so a batch can never mix models if the alias moves mid-run. Features
are built for every client in ``clients.csv`` -- not only the experiment's --
and converted to the model's canonical input exactly as training converted
them (booleans to 0/1, then the contract's float64 / string layout).

Each batch is one row per client: uplift score, rank (1 = highest), decile
(1 = top 10%), the send flag, and the model version, scoring time, and run
that produced it. Batches are written as separate files and never
overwritten, so every past decision can be compared with what happened.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import polars as pl
from mlflow import MlflowClient

from promolift.data.loader import project_root
from promolift.serving.pyfunc import ContractError, FeatureContract
from promolift.serving.registration import CHAMPION_ALIAS, DEFAULT_MODEL_NAME, registry_uri_for
from promolift.tracking.mlflow_tracking import (
    Experiment,
    TrackingConfig,
    default_tracking_config,
    start_run,
)

SCORING_OUTPUT_DIR = Path("data") / "interim" / "scoring"
SEND_LIST_COLUMNS = (
    "client_id",
    "uplift_score",
    "rank",
    "decile",
    "send",
    "model_name",
    "model_version",
    "scored_at",
    "scoring_run_id",
)


@dataclass(frozen=True)
class ChampionModel:
    """The champion resolved to one version, loaded, with its feature contract."""

    name: str
    version: str
    model: mlflow.pyfunc.PyFuncModel
    contract: FeatureContract


def load_champion(
    name: str = DEFAULT_MODEL_NAME, config: TrackingConfig | None = None
) -> ChampionModel:
    """Resolve ``name@champion`` to its version and load that exact version."""
    config = config if config is not None else default_tracking_config()
    registry_uri = registry_uri_for(config.tracking_uri)
    client = MlflowClient(tracking_uri=config.tracking_uri, registry_uri=registry_uri)
    version = str(client.get_model_version_by_alias(name, CHAMPION_ALIAS).version)
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    return ChampionModel(name, version, model, model.unwrap_python_model().contract)


def model_input(features: pl.DataFrame, contract: FeatureContract) -> pd.DataFrame:
    """The contract's canonical input for a client feature table.

    Booleans become 0/1 first, as ``models.dataset.model_frame`` does for
    training, so a client scores the same here as in training.

    Raises:
        ContractError: If the table lacks a contract column.
    """
    missing = [column for column in contract.columns if column not in features.columns]
    if missing:
        raise ContractError(f"The feature table lacks contract columns {missing}")
    selected = features.select(contract.columns).with_columns(pl.col(pl.Boolean).cast(pl.Int8))
    return contract.to_input(selected.to_pandas())


def send_list(
    client_ids: pl.Series,
    scores: np.ndarray,
    *,
    send_share: float,
    model_name: str,
    model_version: str,
    scored_at: datetime,
    scoring_run_id: str,
) -> pl.DataFrame:
    """Rank clients by score and flag the top ``send_share`` to text.

    Ties are broken by ``client_id`` so the ranking is deterministic. The
    number sent is ``round(send_share * n)``; deciles split the ranking into
    tenths (1 = highest).

    Raises:
        ValueError: If lengths differ, a score is not finite, or
            ``send_share`` is outside [0, 1].
    """
    if len(client_ids) != len(scores):
        raise ValueError("client_ids and scores must be the same length")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinite values")
    if not 0 <= send_share <= 1:
        raise ValueError("send_share must be in [0, 1]")
    n = len(scores)
    n_send = round(send_share * n)
    ranked = (
        pl.DataFrame({"client_id": client_ids, "uplift_score": np.asarray(scores, dtype=float)})
        .sort(["uplift_score", "client_id"], descending=[True, False])
        .with_row_index("rank", offset=1)
    )
    return ranked.select(
        "client_id",
        "uplift_score",
        pl.col("rank").cast(pl.Int64),
        ((pl.col("rank") - 1) * 10 // max(n, 1) + 1).cast(pl.Int8).alias("decile"),
        (pl.col("rank") <= n_send).alias("send"),
        pl.lit(model_name).alias("model_name"),
        pl.lit(model_version).alias("model_version"),
        pl.lit(scored_at).dt.replace_time_zone("UTC").alias("scored_at"),
        pl.lit(scoring_run_id).alias("scoring_run_id"),
    )


def decile_summary(targets: pl.DataFrame) -> pl.DataFrame:
    """Clients, sends, and score range per decile -- a quick check of a batch."""
    return (
        targets.group_by("decile")
        .agg(
            pl.len().alias("clients"),
            pl.col("send").sum().alias("sent"),
            pl.col("uplift_score").mean().alias("mean_score"),
            pl.col("uplift_score").min().alias("min_score"),
            pl.col("uplift_score").max().alias("max_score"),
        )
        .sort("decile")
    )


@dataclass
class ScoringResult:
    """A scored batch, where it was written, and the run that records it."""

    run_id: str
    model_version: str
    targets: pl.DataFrame
    path: Path


def run_batch_scoring(
    features: pl.DataFrame,
    *,
    send_share: float,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: Path | None = None,
    scored_at: datetime | None = None,
    config: TrackingConfig | None = None,
    lineage: dict[str, Path] | None = None,
) -> ScoringResult:
    """Score every client in ``features`` with the champion, in one MLflow run.

    The batch is written to ``output_dir/send_list_<timestamp>_<run>.parquet``
    (gitignored ``data/interim/scoring`` by default); the run records the
    model version, counts, and a per-decile summary, not the client list.

    Args:
        features: One row per client: ``client_id`` plus the contract columns
            (``models.dataset.build_model_features``).
        send_share: Fraction of clients to text (``business.yaml``).
        lineage: ``repo_dir`` / ``data_dir`` / ``processed_dir`` overrides (for tests).
    """
    config = config if config is not None else default_tracking_config()
    output_dir = output_dir if output_dir is not None else project_root() / SCORING_OUTPUT_DIR
    scored_at = scored_at if scored_at is not None else datetime.now(tz=UTC)
    champion = load_champion(model_name, config)
    scores = np.asarray(
        champion.model.predict(model_input(features, champion.contract)), dtype=float
    )

    with start_run(
        Experiment.SCORING,
        run_name="batch_scoring",
        tags={
            "run_type": "batch_scoring",
            "model_name": model_name,
            "model_version": champion.version,
            "scored_at": scored_at.isoformat(),
        },
        config=config,
        **(lineage or {}),
    ) as run:
        targets = send_list(
            features["client_id"],
            scores,
            send_share=send_share,
            model_name=model_name,
            model_version=champion.version,
            scored_at=scored_at,
            scoring_run_id=run.info.run_id,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = scored_at.strftime("%Y%m%dT%H%M%SZ")
        path = output_dir / f"send_list_{stamp}_{run.info.run_id[:8]}.parquet"
        targets.write_parquet(path)

        sent = targets.filter(pl.col("send"))
        mlflow.log_params({"send_share": send_share})
        mlflow.log_metrics(
            {
                "n_clients": targets.height,
                "n_send": sent.height,
                "mean_score": float(targets["uplift_score"].mean()),
                "mean_score_sent": float(sent["uplift_score"].mean()) if sent.height else math.nan,
                "min_score_sent": float(sent["uplift_score"].min()) if sent.height else math.nan,
            }
        )
        mlflow.log_text(decile_summary(targets).write_csv(), "scoring/decile_summary.csv")
        return ScoringResult(run.info.run_id, champion.version, targets, path)
