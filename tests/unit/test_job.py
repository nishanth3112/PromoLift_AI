"""Unit tests for the Databricks scoring job: table writes, arguments, and an end-to-end run."""

from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from mlflow import MlflowClient

from promolift.serving.job import (
    DEFAULT_LATEST_VIEW,
    DEFAULT_TABLE,
    parse_args,
    run_job,
    write_targets,
)


class FakeSpark:
    """Records what the job would write, so tests need neither Spark nor Databricks."""

    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []
        self.writes: list[tuple[str, str]] = []
        self.statements: list[str] = []

    def createDataFrame(self, frame: pd.DataFrame) -> "FakeSpark":
        self.frames.append(frame)
        self._mode = None
        return self

    @property
    def write(self) -> "FakeSpark":
        return self

    def mode(self, mode: str) -> "FakeSpark":
        self._mode = mode
        return self

    def saveAsTable(self, table: str) -> None:
        self.writes.append((self._mode, table))

    def sql(self, statement: str) -> None:
        self.statements.append(statement)


def _targets() -> pl.DataFrame:
    return pl.DataFrame(
        {"client_id": ["a", "b"], "uplift_score": [0.2, 0.1], "send": [True, False]}
    )


def test_write_targets_appends_the_batch_and_refreshes_the_latest_view() -> None:
    spark = FakeSpark()

    write_targets(
        _targets(), table="cat.sch.targets", latest_view="cat.sch.latest", git_commit="abc123",
        spark=spark,
    )  # fmt: skip

    assert spark.writes == [("append", "cat.sch.targets")]
    written = spark.frames[0]
    assert written["deployed_git_commit"].tolist() == ["abc123", "abc123"]
    assert written["client_id"].tolist() == ["a", "b"]
    (statement,) = spark.statements
    assert statement.startswith("CREATE OR REPLACE VIEW cat.sch.latest")
    assert "FROM cat.sch.targets" in statement
    assert "ORDER BY scored_at DESC LIMIT 1" in statement


@pytest.mark.parametrize("name", ["targets", "a.b", "a.b.c; DROP TABLE x", "a.b.c-d"])
def test_write_targets_rejects_names_that_are_not_plain_three_level(name: str) -> None:
    spark = FakeSpark()

    with pytest.raises(ValueError, match=r"catalog\.schema\.name"):
        write_targets(
            _targets(), table=name, latest_view="cat.sch.latest", git_commit="x", spark=spark
        )
    assert spark.writes == []


def test_parse_args_defaults_to_databricks_and_the_project_table() -> None:
    args = parse_args(["--raw-dir", "/Volumes/c/s/raw", "--business-config", "/w/business.yaml"])

    assert args.raw_dir == Path("/Volumes/c/s/raw")
    assert (args.table, args.latest_view) == (DEFAULT_TABLE, DEFAULT_LATEST_VIEW)
    assert args.tracking_uri == "databricks"
    assert args.experiment_root == "/Shared/promolift"
    assert args.git_commit == "unknown"


def test_parse_args_requires_the_raw_data_and_config() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--raw-dir", "/Volumes/c/s/raw"])


def _business_config(tmp_path: Path, send_share: str) -> Path:
    path = tmp_path / "business.yaml"
    path.write_text(
        "currency: RUB\nsms_cost: 3.0\ngross_margin: 0.24\nmargin_per_purchase: null\n"
        "budget: null\ncampaign_clients: null\nbreak_even_grid: [0.03]\n"
        f"send_share: {send_share}\n"
    )
    return path


def test_run_job_scores_the_raw_data_and_writes_the_table(
    scoring_champion, feature_raw_dir: Path, tmp_path: Path
) -> None:
    spark = FakeSpark()

    result = run_job(
        feature_raw_dir,
        _business_config(tmp_path, "0.5"),
        model_name=scoring_champion.name,
        table="cat.sch.targets",
        latest_view="cat.sch.latest",
        git_commit="abc123",
        tracking=scoring_champion.config,
        spark=spark,
        output_dir=tmp_path / "out",
    )

    assert spark.writes == [("append", "cat.sch.targets")]
    written = spark.frames[0]
    assert len(written) == 2
    assert written["send"].sum() == 1
    assert set(written["scoring_run_id"]) == {result.run_id}
    assert set(written["deployed_git_commit"]) == {"abc123"}
    run = MlflowClient(tracking_uri=scoring_champion.config.tracking_uri).get_run(result.run_id)
    assert run.data.tags["deployed_git_commit"] == "abc123"
    assert run.data.tags["targets_table"] == "cat.sch.targets"
    assert run.data.tags["trigger"] == "job"


def test_run_job_needs_a_send_share(feature_raw_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no send_share"):
        run_job(
            feature_raw_dir,
            _business_config(tmp_path, "null"),
            tracking=None,
            spark=FakeSpark(),
        )
