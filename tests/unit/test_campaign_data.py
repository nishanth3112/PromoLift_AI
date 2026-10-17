"""Unit tests for loading, validating, and matching campaign results."""

from pathlib import Path

import polars as pl
import pytest

from promolift.monitoring.campaign_data import (
    RESULT_COLUMNS,
    CampaignResultsError,
    find_send_list,
    load_results,
    match_results,
    validate_results,
)


def _results(batch: pl.DataFrame, *, run_id: str = "run-a", buy: int = 1) -> pl.DataFrame:
    """A result for every targeted client of the batch; everyone buys (or not) as ``buy``."""
    targeted = batch.filter(pl.col("send") | pl.col("holdout"))
    return pl.DataFrame(
        {
            "client_id": targeted["client_id"],
            "treated": targeted["send"].cast(pl.Int8),
            "purchased": [buy] * targeted.height,
            "scoring_run_id": [run_id] * targeted.height,
        }
    )


# --- load and validate ----------------------------------------------------------------


def test_loads_csv_and_parquet_results_with_the_same_content(
    make_campaign_batch, tmp_path: Path
) -> None:
    results = _results(make_campaign_batch(200))
    results.write_csv(tmp_path / "results.csv")
    results.write_parquet(tmp_path / "results.parquet")

    from_csv = load_results(tmp_path / "results.csv")
    from_parquet = load_results(tmp_path / "results.parquet")

    assert from_csv.equals(from_parquet.select(RESULT_COLUMNS))
    assert from_csv.height == 100
    assert from_csv.schema["treated"] == pl.Int8


def test_csv_ids_that_look_like_numbers_stay_text(tmp_path: Path) -> None:
    (tmp_path / "results.csv").write_text(
        "client_id,treated,purchased,scoring_run_id\n001,1,0,r\n002,0,1,r\n"
    )

    assert load_results(tmp_path / "results.csv")["client_id"].to_list() == ["001", "002"]


def test_booleans_are_accepted_as_flags() -> None:
    results = pl.DataFrame(
        {
            "client_id": ["a", "b"],
            "treated": [True, False],
            "purchased": [False, True],
            "scoring_run_id": ["r", "r"],
        }
    )

    clean = validate_results(results)

    assert clean["treated"].to_list() == [1, 0]
    assert clean["purchased"].to_list() == [0, 1]


def test_load_rejects_a_missing_file_and_an_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "nope.csv")
    (tmp_path / "results.xlsx").write_text("x")
    with pytest.raises(CampaignResultsError, match=r"\.csv or \.parquet"):
        load_results(tmp_path / "results.xlsx")


def _blank_first(results: pl.DataFrame, column: str, value) -> pl.DataFrame:
    """The results with ``column`` replaced by ``value`` in the first row only."""
    first = pl.int_range(pl.len()) == 0
    replaced = pl.when(first).then(pl.lit(value)).otherwise(pl.col(column))
    return results.with_columns(replaced.alias(column))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.drop("purchased"), "lack columns"),
        (lambda r: r.clear(), "empty"),
        (lambda r: _blank_first(r, "purchased", None), "missing values"),
        (lambda r: r.with_columns(pl.col("purchased") * 2), "purchased must be 0/1"),
        (lambda r: r.with_columns(pl.lit("yes").alias("treated")), "treated must be 0/1"),
        (lambda r: pl.concat([r, r.head(1)]), "more than once"),
        (lambda r: _blank_first(r, "scoring_run_id", "other"), "one scoring run"),
    ],
)
def test_validate_rejects_results_that_cannot_be_compared(
    make_campaign_batch, change, message: str
) -> None:
    with pytest.raises(CampaignResultsError, match=message):
        validate_results(change(_results(make_campaign_batch(200))))


# --- match to the batch --------------------------------------------------------------


def test_matches_every_targeted_client_with_its_arm_and_prediction(make_campaign_batch) -> None:
    batch = make_campaign_batch(1_000, send_share=0.5, holdout_share=0.2)

    data = match_results(batch, validate_results(_results(batch)))

    assert data.frame.height == 500
    assert (data.n_treated, data.n_control) == (400, 100)
    assert (data.n_missing_treated, data.n_missing_control) == (0, 0)
    assert data.scoring_run_id == "run-a"
    assert set(data.frame.columns) == {
        "client_id",
        "uplift_score",
        "decile",
        "treated",
        "purchased",
    }
    # The prediction comes from the batch, not the results file.
    scores = dict(zip(batch["client_id"], batch["uplift_score"], strict=True))
    row = data.frame.row(0, named=True)
    assert row["uplift_score"] == scores[row["client_id"]]


def test_clients_without_results_are_counted_per_arm(make_campaign_batch) -> None:
    batch = make_campaign_batch(1_000, send_share=0.5, holdout_share=0.2)
    results = validate_results(_results(batch))
    dropped = results.filter(pl.col("treated") == 1).head(40)["client_id"].to_list()

    data = match_results(batch, results.filter(~pl.col("client_id").is_in(dropped)))

    assert (data.n_missing_treated, data.n_missing_control) == (40, 0)
    assert data.n_treated == 360


def test_only_the_results_run_is_used_when_the_batch_holds_several(make_campaign_batch) -> None:
    first = make_campaign_batch(500, run_id="run-a", seed=1)
    second = make_campaign_batch(500, run_id="run-b", seed=2)
    both = pl.concat([first, second])

    data = match_results(both, validate_results(_results(second, run_id="run-b")))

    assert data.scoring_run_id == "run-b"
    assert data.frame.height == 250


def test_results_for_clients_the_batch_did_not_target_are_rejected(make_campaign_batch) -> None:
    batch = make_campaign_batch(500, send_share=0.5)
    untargeted = batch.filter(~pl.col("send") & ~pl.col("holdout")).head(1)
    extra = pl.DataFrame(
        {
            "client_id": untargeted["client_id"],
            "treated": pl.Series([1], dtype=pl.Int8),
            "purchased": pl.Series([1], dtype=pl.Int8),
            "scoring_run_id": ["run-a"],
        }
    )
    results = pl.concat([_results(batch), extra], how="vertical_relaxed")

    with pytest.raises(CampaignResultsError, match="1 result clients were not targeted"):
        match_results(batch, validate_results(results))


def test_an_arm_that_disagrees_with_the_batch_is_rejected(make_campaign_batch) -> None:
    batch = make_campaign_batch(500)
    results = _results(batch)
    flipped = results.with_columns(
        pl.when(pl.int_range(pl.len()) < 3).then(1 - pl.col("treated")).otherwise(pl.col("treated"))
        .cast(pl.Int8)
        .alias("treated")
    )  # fmt: skip

    with pytest.raises(CampaignResultsError, match=r"disagrees .* for 3 clients"):
        match_results(batch, validate_results(flipped))


def test_a_batch_with_no_control_group_cannot_be_evaluated(make_campaign_batch) -> None:
    batch = make_campaign_batch(500, holdout_share=0.0)

    with pytest.raises(CampaignResultsError, match="held nobody back"):
        match_results(batch, validate_results(_results(batch)))


def test_both_arms_need_results(make_campaign_batch) -> None:
    batch = make_campaign_batch(500)
    only_treated = validate_results(_results(batch).filter(pl.col("treated") == 1))

    with pytest.raises(CampaignResultsError, match="both the treated and the control"):
        match_results(batch, only_treated)


def test_a_run_missing_from_the_batch_is_reported(make_campaign_batch) -> None:
    batch = make_campaign_batch(500)

    with pytest.raises(CampaignResultsError, match="no targeted clients from run run-z"):
        match_results(batch, validate_results(_results(batch, run_id="run-z")))


def test_a_batch_without_the_holdout_column_is_explained(make_campaign_batch) -> None:
    batch = make_campaign_batch(500)

    with pytest.raises(CampaignResultsError, match="without a holdout"):
        match_results(batch.drop("holdout"), validate_results(_results(batch)))


# --- finding the send list ---------------------------------------------------------------


def _touch(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"")
    return path


def test_finds_the_send_list_of_a_run_by_the_start_of_its_id(tmp_path: Path) -> None:
    _touch(tmp_path, "send_list_20261001T000000Z_aaaaaaaa.parquet")
    wanted = _touch(tmp_path, "send_list_20261005T000000Z_bbbbbbbb.parquet")
    _touch(tmp_path, "send_list_20261009T000000Z_cccccccc.parquet")

    assert find_send_list(tmp_path, "bbbbbbbb1234567890abcdef") == wanted


def test_the_latest_send_list_is_the_default(tmp_path: Path) -> None:
    _touch(tmp_path, "send_list_20261001T000000Z_aaaaaaaa.parquet")
    latest = _touch(tmp_path, "send_list_20261009T000000Z_cccccccc.parquet")
    _touch(tmp_path, "notes.txt")

    assert find_send_list(tmp_path) == latest


def test_a_missing_send_list_is_reported(tmp_path: Path) -> None:
    _touch(tmp_path, "send_list_20261001T000000Z_aaaaaaaa.parquet")

    with pytest.raises(FileNotFoundError, match="run zzzzzzzz"):
        find_send_list(tmp_path, "zzzzzzzz")
    with pytest.raises(FileNotFoundError, match="any send list"):
        find_send_list(tmp_path / "empty")
