"""Databricks job entry point: score every client and append the batch to a Delta table.

Runs the same batch scoring as ``scripts/score_clients.py``, with what a
serverless wheel task has instead of a repository checkout: raw data in a
Unity Catalog volume, the business config from the bundle's synced files,
and Spark to write the table. The deployed git commit arrives as a parameter
(the installed wheel has no ``.git``) and is recorded on the run and in every
row.

Each batch is appended to ``promolift_ai.scoring.targets``; the
``targets_latest`` view always shows the most recent batch only.

Every ``promolift`` import here is at module level on purpose: loading the
model puts its bundled copy of ``promolift`` first on ``sys.path``, so a
late import could pick up that older snapshot instead of this code.
"""

from __future__ import annotations

import argparse
import re
import tempfile
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl

from promolift.models.dataset import build_model_features
from promolift.monitoring.drift import DriftThresholds, load_drift_thresholds
from promolift.optimization.business import load_business_config
from promolift.serving.registration import DEFAULT_MODEL_NAME
from promolift.serving.scoring import ScoringResult, run_batch_scoring
from promolift.tracking.mlflow_tracking import TrackingConfig

DEFAULT_TABLE = "promolift_ai.scoring.targets"
DEFAULT_LATEST_VIEW = "promolift_ai.scoring.targets_latest"
DEFAULT_DRIFT_TABLE = "promolift_ai.scoring.drift"
_THREE_LEVEL_NAME = re.compile(r"^[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")
_LATEST_VIEW_SQL = (
    "CREATE OR REPLACE VIEW {view} "
    "COMMENT 'The most recent PromoLift send list (one scoring batch).' AS "
    "SELECT * FROM {table} WHERE scoring_run_id = "
    "(SELECT scoring_run_id FROM {table} ORDER BY scored_at DESC LIMIT 1)"
)


def _checked_name(name: str) -> str:
    """A Unity Catalog ``catalog.schema.object`` name, safe to put into SQL."""
    if not _THREE_LEVEL_NAME.match(name):
        raise ValueError(f"Expected catalog.schema.name of letters, digits, '_': {name!r}")
    return name


def write_targets(
    targets: pl.DataFrame, *, table: str, latest_view: str, git_commit: str, spark: Any
) -> None:
    """Append a batch to ``table`` and point ``latest_view`` at the newest batch.

    Args:
        targets: A send list from ``scoring.send_list``.
        git_commit: The deployed commit, stored in a ``deployed_git_commit`` column.
        spark: An active ``SparkSession``.
    """
    table, latest_view = _checked_name(table), _checked_name(latest_view)
    rows = targets.with_columns(pl.lit(git_commit).alias("deployed_git_commit")).to_pandas()
    spark.createDataFrame(rows).write.mode("append").saveAsTable(table)
    spark.sql(_LATEST_VIEW_SQL.format(view=latest_view, table=table))


def write_drift(drift: pl.DataFrame, *, table: str, git_commit: str, spark: Any) -> None:
    """Append a batch's per-feature drift report (``scoring.drift_rows``) to ``table``."""
    rows = drift.with_columns(pl.lit(git_commit).alias("deployed_git_commit")).to_pandas()
    spark.createDataFrame(rows).write.mode("append").saveAsTable(_checked_name(table))


def run_job(
    raw_dir: Path,
    business_config: Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    table: str = DEFAULT_TABLE,
    latest_view: str = DEFAULT_LATEST_VIEW,
    drift_table: str = DEFAULT_DRIFT_TABLE,
    drift_thresholds: DriftThresholds | None = None,
    git_commit: str = "unknown",
    tracking: TrackingConfig,
    spark: Any,
    output_dir: Path | None = None,
) -> ScoringResult:
    """Build features from ``raw_dir``, check drift, score, write the batch to ``table``.

    The drift report is appended to ``drift_table`` even when severe drift
    stops the batch.

    Raises:
        ValueError: If the business config has no ``send_share``.
        DriftError: If the batch drifts past the failure threshold.
    """
    config = load_business_config(business_config)
    if config.send_share is None:
        raise ValueError(f"{business_config} has no send_share")
    features = build_model_features(raw_dir)
    return run_batch_scoring(
        features,
        send_share=config.send_share,
        model_name=model_name,
        output_dir=output_dir if output_dir is not None else Path(tempfile.mkdtemp()),
        config=tracking,
        lineage={"data_dir": raw_dir},
        tags={"deployed_git_commit": git_commit, "targets_table": table, "trigger": "job"},
        sink=partial(
            write_targets, table=table, latest_view=latest_view, git_commit=git_commit, spark=spark
        ),
        drift_thresholds=drift_thresholds,
        drift_sink=partial(write_drift, table=drift_table, git_commit=git_commit, spark=spark),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--business-config", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--latest-view", default=DEFAULT_LATEST_VIEW)
    parser.add_argument("--drift-table", default=DEFAULT_DRIFT_TABLE)
    parser.add_argument(
        "--monitoring-config", type=Path, help="drift thresholds; default warn 0.1, fail 0.25"
    )
    parser.add_argument("--git-commit", default="unknown")
    parser.add_argument("--tracking-uri", default="databricks")
    parser.add_argument("--experiment-root", default="/Shared/promolift")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Console entry point ``promolift-score-job`` (the bundle's wheel task)."""
    from pyspark.sql import SparkSession

    args = parse_args(argv)
    result = run_job(
        args.raw_dir,
        args.business_config,
        model_name=args.model_name,
        table=args.table,
        latest_view=args.latest_view,
        drift_table=args.drift_table,
        drift_thresholds=(
            load_drift_thresholds(args.monitoring_config) if args.monitoring_config else None
        ),
        git_commit=args.git_commit,
        tracking=TrackingConfig(args.tracking_uri, experiment_root=args.experiment_root),
        spark=SparkSession.builder.getOrCreate(),
    )
    sent = int(result.targets["send"].sum())
    worst = result.drift.row(0, named=True)
    print(
        f"Drift: max PSI {worst['psi']:.4f} ({worst['feature']}, {worst['status']}). "
        f"Model version {result.model_version}: {result.targets.height:,} clients scored, "
        f"{sent:,} flagged; appended to {args.table}; run {result.run_id}"
    )
