"""Unit tests for batch scoring: model input, ranking and send flags, and the scoring run."""

from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pytest
from mlflow import MlflowClient

from promolift.monitoring.drift import DriftError, DriftThresholds
from promolift.serving.pyfunc import ContractError, FeatureContract
from promolift.serving.registration import register, set_champion
from promolift.serving.scoring import (
    SEND_LIST_COLUMNS,
    decile_summary,
    load_champion,
    model_input,
    run_batch_scoring,
    send_list,
)

_NAME = "uplift_scoring_test"
_AT = datetime(2026, 10, 5, 9, 30, tzinfo=UTC)


def _send_list(scores, client_ids=None, share=0.3, holdout=0.0, run_id="run123"):
    ids = client_ids if client_ids is not None else [f"c{i:02d}" for i in range(len(scores))]
    return send_list(
        pl.Series(ids),
        np.asarray(scores, dtype=float),
        send_share=share,
        holdout_share=holdout,
        model_name=_NAME,
        model_version="1",
        scored_at=_AT,
        scoring_run_id=run_id,
    )


# --- send_list ---------------------------------------------------------------


def test_ranks_by_score_and_flags_the_top_share() -> None:
    targets = _send_list([0.1, 0.5, -0.2, 0.3, 0.0, 0.4, 0.2, -0.1, 0.05, 0.15])

    assert targets["rank"].to_list() == list(range(1, 11))
    assert targets["uplift_score"].is_sorted(descending=True)
    assert targets["client_id"][:3].to_list() == ["c01", "c05", "c03"]
    assert targets["send"].to_list() == [True] * 3 + [False] * 7
    assert not targets["holdout"].any()
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


# --- holdout -------------------------------------------------------------------------


def test_holdout_is_taken_from_the_selected_clients_and_never_sent() -> None:
    targets = _send_list(np.linspace(1, 0, 100), share=0.4, holdout=0.25)

    selected = targets.filter(pl.col("rank") <= 40)
    assert selected["holdout"].sum() == 10
    assert selected["send"].sum() == 30
    # Every selected client is either texted or held back, never both.
    assert (selected["send"] ^ selected["holdout"]).all()
    # Nobody outside the selection is texted or held back.
    unselected = targets.filter(pl.col("rank") > 40)
    assert not unselected["send"].any()
    assert not unselected["holdout"].any()


def test_holdout_is_random_not_the_lowest_ranked_selected() -> None:
    targets = _send_list(np.linspace(1, 0, 1000), share=0.5, holdout=0.2)

    held_ranks = targets.filter(pl.col("holdout"))["rank"]
    assert held_ranks.min() < 100
    assert held_ranks.max() > 400
    # Spread across the selection: every fifth of it holds some of the 100 held-back
    # clients (20 expected; the bounds leave room for the seeded draw's noise).
    per_fifth = np.histogram(held_ranks.to_numpy(), bins=5, range=(1, 501))[0]
    assert per_fifth.min() > 5
    assert per_fifth.max() < 40


def test_holdout_is_reproducible_per_batch_and_new_for_each_batch() -> None:
    scores = np.linspace(1, 0, 200)

    first = _send_list(scores, share=0.5, holdout=0.2, run_id="runA")
    again = _send_list(scores, share=0.5, holdout=0.2, run_id="runA")
    other = _send_list(scores, share=0.5, holdout=0.2, run_id="runB")

    assert first["holdout"].to_list() == again["holdout"].to_list()
    assert first["holdout"].to_list() != other["holdout"].to_list()
    assert first["holdout"].sum() == other["holdout"].sum() == 20


def test_holdout_does_not_depend_on_the_scores() -> None:
    # The draw is positions in the ranking, so a different score distribution with the
    # same run id holds back the same ranks.
    low = _send_list(np.linspace(0.1, 0, 100), share=0.5, holdout=0.2)
    high = _send_list(np.linspace(5, -5, 100), share=0.5, holdout=0.2)

    assert low["holdout"].to_list() == high["holdout"].to_list()


def test_a_zero_holdout_texts_everyone_selected() -> None:
    targets = _send_list(np.linspace(1, 0, 20), share=0.5, holdout=0.0)

    assert targets["send"].sum() == 10
    assert targets["holdout"].sum() == 0


def test_the_decile_summary_counts_held_back_clients() -> None:
    targets = _send_list(np.linspace(1, 0, 100), share=0.4, holdout=0.25)

    summary = decile_summary(targets)

    assert summary["held_back"].sum() == 10
    assert summary["sent"].sum() == 30
    assert summary["held_back"].to_list()[4:] == [0] * 6


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
    with pytest.raises(ValueError, match="holdout_share"):
        _send_list([0.1, 0.2], holdout=1.0)


# --- model input and end-to-end scoring ------------------------------------------
# client_features, make_training_frame, and scoring_champion live in conftest.py.


def test_model_input_matches_the_contract_layout(client_features, make_training_frame) -> None:
    contract = FeatureContract.from_frame(make_training_frame(client_features, 50, 0))

    frame = model_input(client_features, contract)

    assert tuple(frame.columns) == contract.columns
    numeric = [c for c in contract.columns if c != "gender"]
    assert (frame.dtypes[numeric] == "float64").all()
    assert set(frame["has_redeemed"].dropna()) <= {0.0, 1.0}
    assert frame["gender"].tolist() == ["F", "M"]
    # c2 has no purchases: its purchase features are missing, not zero.
    assert np.isnan(frame.loc[1, "monetary_avg"])


def test_model_input_rejects_a_table_missing_contract_columns(
    client_features, make_training_frame
) -> None:
    contract = FeatureContract.from_frame(make_training_frame(client_features, 50, 0))

    with pytest.raises(ContractError, match="lacks contract columns"):
        model_input(client_features.drop("frequency"), contract)


def test_load_champion_resolves_the_alias_to_a_version(scoring_champion) -> None:
    loaded = load_champion(scoring_champion.name, scoring_champion.config)

    assert loaded.version == "1"
    assert loaded.contract == scoring_champion.packaged.contract


def test_scores_every_client_and_writes_the_batch(
    scoring_champion, client_features, lineage_dirs, tmp_path: Path
) -> None:
    config = scoring_champion.config
    result = run_batch_scoring(
        client_features,
        send_share=0.5,
        model_name=scoring_champion.name,
        output_dir=tmp_path / "scoring",
        scored_at=_AT,
        config=config,
        lineage=lineage_dirs,
    )

    written = pl.read_parquet(result.path)
    assert written.height == client_features.height == 2
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
    assert run.data.metrics["n_holdout"] == 0
    assert run.data.params["send_share"] == "0.5"
    assert run.data.params["holdout_share"] == "0.0"


def test_a_scoring_run_holds_back_and_records_the_control_group(
    scoring_champion, client_features, lineage_dirs, tmp_path: Path
) -> None:
    result = run_batch_scoring(
        client_features,
        send_share=1.0,
        holdout_share=0.5,
        model_name=scoring_champion.name,
        output_dir=tmp_path / "scoring",
        scored_at=_AT,
        config=scoring_champion.config,
        lineage=lineage_dirs,
    )

    written = pl.read_parquet(result.path)
    assert (written["send"].sum(), written["holdout"].sum()) == (1, 1)
    run = MlflowClient(tracking_uri=scoring_champion.config.tracking_uri).get_run(result.run_id)
    assert run.data.metrics["n_send"] == 1
    assert run.data.metrics["n_holdout"] == 1
    assert run.data.params["holdout_share"] == "0.5"


def test_the_sink_writes_inside_the_run_with_extra_tags(
    scoring_champion, client_features, lineage_dirs, tmp_path: Path
) -> None:
    received = []

    def sink(targets: pl.DataFrame) -> None:
        received.append((targets.height, mlflow.active_run().info.run_id))

    result = run_batch_scoring(
        client_features,
        send_share=0.5,
        model_name=scoring_champion.name,
        output_dir=tmp_path / "scoring",
        config=scoring_champion.config,
        lineage=lineage_dirs,
        tags={"trigger": "test"},
        sink=sink,
    )

    assert received == [(2, result.run_id)]
    run = MlflowClient(tracking_uri=scoring_champion.config.tracking_uri).get_run(result.run_id)
    assert run.data.tags["trigger"] == "test"


def test_batches_are_kept_and_follow_the_champion(
    scoring_champion, client_features, lineage_dirs, tmp_path: Path
) -> None:
    name, config, output = scoring_champion.name, scoring_champion.config, tmp_path / "scoring"
    first = run_batch_scoring(
        client_features, send_share=0.5, model_name=name, output_dir=output, scored_at=_AT,
        config=config, lineage=lineage_dirs,
    )  # fmt: skip
    set_champion(register(scoring_champion.packaged, name, config), name, config)
    second = run_batch_scoring(
        client_features, send_share=0.5, model_name=name, output_dir=output,
        scored_at=datetime(2026, 10, 12, tzinfo=UTC), config=config, lineage=lineage_dirs,
    )  # fmt: skip

    assert (first.model_version, second.model_version) == ("1", "2")
    assert first.path.exists()
    assert second.path.exists()
    assert len(list(output.glob("send_list_*.parquet"))) == 2


# --- drift ---------------------------------------------------------------------------


def test_scoring_logs_the_drift_of_every_feature(
    scoring_champion, client_features, lineage_dirs, tmp_path: Path
) -> None:
    result = run_batch_scoring(
        client_features,
        send_share=0.5,
        model_name=scoring_champion.name,
        output_dir=tmp_path / "scoring",
        config=scoring_champion.config,
        lineage=lineage_dirs,
    )

    contract_columns = set(scoring_champion.packaged.contract.columns)
    assert set(result.drift["feature"]) == contract_columns
    assert set(result.drift["scoring_run_id"]) == {result.run_id}
    run = MlflowClient(tracking_uri=scoring_champion.config.tracking_uri).get_run(result.run_id)
    assert run.data.tags["drift_status"] in {"ok", "warn"}
    assert {f"psi_{c}" for c in contract_columns} <= set(run.data.metrics)
    assert run.data.params["drift_fail_psi"] == "0.25"


def test_severe_drift_stops_the_batch_but_records_the_drift(
    scoring_champion, client_features, lineage_dirs, tmp_path: Path
) -> None:
    # Every client's gender is now "U", a level the training clients never had.
    drifted = client_features.with_columns(pl.lit("U").alias("gender"))
    recorded = []
    written = []

    with pytest.raises(DriftError, match="gender"):
        run_batch_scoring(
            drifted,
            send_share=0.5,
            model_name=scoring_champion.name,
            output_dir=tmp_path / "scoring",
            config=scoring_champion.config,
            lineage=lineage_dirs,
            sink=written.append,
            drift_sink=recorded.append,
        )

    assert written == []
    assert not (tmp_path / "scoring").exists()
    (report,) = recorded
    assert report.filter(pl.col("feature") == "gender")["status"][0] == "fail"
    client = MlflowClient(tracking_uri=scoring_champion.config.tracking_uri)
    (run,) = client.search_runs([client.get_experiment_by_name("promolift-scoring").experiment_id])
    assert run.info.status == "FAILED"
    assert run.data.tags["drift_status"] == "fail"


def test_drift_thresholds_decide_what_blocks(
    scoring_champion, client_features, lineage_dirs, tmp_path: Path
) -> None:
    drifted = client_features.with_columns(pl.lit("U").alias("gender"))

    result = run_batch_scoring(
        drifted,
        send_share=0.5,
        model_name=scoring_champion.name,
        output_dir=tmp_path / "scoring",
        config=scoring_champion.config,
        lineage=lineage_dirs,
        drift_thresholds=DriftThresholds(warn_psi=0.1, fail_psi=100.0),
    )

    assert result.targets.height == 2
    run = MlflowClient(tracking_uri=scoring_champion.config.tracking_uri).get_run(result.run_id)
    assert run.data.tags["drift_status"] == "warn"
