"""Campaign results: load and check them, and match them to the send list that was run.

A scoring batch selects the clients to text (``send``) and holds a random
share back as the control group (``holdout``). Once the campaign has run, the
results say who bought. This module turns a results file into the table the
evaluation needs -- every targeted client with its arm, its predicted uplift,
and its outcome -- and refuses anything that would make the comparison
invalid: results for clients the batch never targeted, or an arm that
disagrees with what the batch assigned.

Results file (CSV or parquet), one row per targeted client with a result:

    client_id, treated, purchased, scoring_run_id

``treated`` and ``purchased`` are 0/1 (or true/false); ``scoring_run_id`` is
the batch the campaign ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

SEND_LIST_PATTERN = "send_list_*.parquet"
RESULT_COLUMNS = ("client_id", "treated", "purchased", "scoring_run_id")
_BATCH_COLUMNS = ("client_id", "uplift_score", "decile", "send", "holdout", "scoring_run_id")
_SHOWN_EXAMPLES = 3


class CampaignResultsError(ValueError):
    """The results (or the batch they are matched to) can't support a valid comparison."""


@dataclass(frozen=True)
class CampaignData:
    """Targeted clients that have a result, and how many targeted clients don't.

    ``frame`` has ``client_id``, ``uplift_score``, ``decile``, ``treated``,
    and ``purchased``. A client missing from the results is neither a buyer
    nor a non-buyer, so the missing counts per arm are kept: unequal gaps
    between the arms would bias the comparison.
    """

    scoring_run_id: str
    frame: pl.DataFrame
    n_targeted_treated: int
    n_targeted_control: int

    @property
    def n_treated(self) -> int:
        return int(self.frame["treated"].sum())

    @property
    def n_control(self) -> int:
        return self.frame.height - self.n_treated

    @property
    def n_missing_treated(self) -> int:
        return self.n_targeted_treated - self.n_treated

    @property
    def n_missing_control(self) -> int:
        return self.n_targeted_control - self.n_control


def find_send_list(directory: Path, scoring_run_id: str | None = None) -> Path:
    """The send list written by scoring run ``scoring_run_id``, or the latest one if None.

    Batches are named ``send_list_<timestamp>_<first 8 of the run id>.parquet``
    (``serving.scoring.run_batch_scoring``), so the timestamp orders them.

    Raises:
        FileNotFoundError: If the directory holds no matching send list.
    """
    pattern = (
        SEND_LIST_PATTERN
        if scoring_run_id is None
        else (f"send_list_*_{scoring_run_id[:8]}.parquet")
    )
    found = sorted(Path(directory).glob(pattern))
    if not found:
        which = "any send list" if scoring_run_id is None else f"run {scoring_run_id}"
        raise FileNotFoundError(f"No send list for {which} in {directory}")
    return found[-1]


def load_results(path: Path) -> pl.DataFrame:
    """Read and validate a campaign results file (``.csv`` or ``.parquet``).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        CampaignResultsError: If the format or contents are invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".csv":
        # Ids stay text even when every id in the file looks like a number.
        raw = pl.read_csv(
            path, schema_overrides={"client_id": pl.String, "scoring_run_id": pl.String}
        )
    elif path.suffix == ".parquet":
        raw = pl.read_parquet(path)
    else:
        raise CampaignResultsError(f"Results must be a .csv or .parquet file, got {path.name}")
    return validate_results(raw)


def _binary(results: pl.DataFrame, column: str) -> pl.Series:
    series = results[column]
    if series.dtype == pl.Boolean:
        return series.cast(pl.Int8)
    if not series.dtype.is_numeric() or not series.is_in([0, 1]).all():
        raise CampaignResultsError(f"{column} must be 0/1 or true/false")
    return series.cast(pl.Int8)


def validate_results(results: pl.DataFrame) -> pl.DataFrame:
    """The results in canonical types (text ids, 0/1 integers), or an error saying what's wrong.

    Raises:
        CampaignResultsError: If a column is missing, a value is missing or
            not 0/1, a client appears twice, the table is empty, or it spans
            more than one scoring run.
    """
    missing = [column for column in RESULT_COLUMNS if column not in results.columns]
    if missing:
        raise CampaignResultsError(f"Results lack columns {missing}")
    results = results.select(RESULT_COLUMNS)
    if results.height == 0:
        raise CampaignResultsError("Results are empty")
    if results.null_count().sum_horizontal().item() > 0:
        raise CampaignResultsError("Results contain missing values")
    if results["client_id"].n_unique() != results.height:
        raise CampaignResultsError("Results list a client more than once")
    if results["scoring_run_id"].n_unique() != 1:
        raise CampaignResultsError("Results must come from one scoring run")
    return results.with_columns(
        pl.col("client_id").cast(pl.String),
        pl.col("scoring_run_id").cast(pl.String),
        _binary(results, "treated").alias("treated"),
        _binary(results, "purchased").alias("purchased"),
    )


def match_results(batch: pl.DataFrame, results: pl.DataFrame) -> CampaignData:
    """Match validated results to the targeted clients of the batch they came from.

    The arms come from the batch: ``send`` clients are the treated arm,
    ``holdout`` clients the control. Each result's ``treated`` must agree.

    Args:
        batch: A send list (``serving.scoring.send_list``); may hold several
            scoring runs -- the results' run is selected.
        results: Output of ``validate_results`` / ``load_results``.

    Raises:
        CampaignResultsError: If the batch lacks the run or a column, a result
            is for a client the batch didn't target, a result's ``treated``
            disagrees with the batch, or an arm has no results.
    """
    missing = [column for column in _BATCH_COLUMNS if column not in batch.columns]
    if missing:
        raise CampaignResultsError(
            f"The send list lacks columns {missing}; was it scored without a holdout?"
        )
    run_id = results["scoring_run_id"][0]
    targeted = batch.filter(
        (pl.col("scoring_run_id") == run_id) & (pl.col("send") | pl.col("holdout"))
    )
    if targeted.height == 0:
        raise CampaignResultsError(f"The send list has no targeted clients from run {run_id}")
    if targeted.filter(pl.col("holdout")).height == 0:
        raise CampaignResultsError(
            f"Run {run_id} held nobody back, so there is no control group to compare with"
        )

    unknown = results.join(targeted.select("client_id"), on="client_id", how="anti")
    if unknown.height:
        examples = ", ".join(unknown["client_id"].head(_SHOWN_EXAMPLES))
        raise CampaignResultsError(
            f"{unknown.height} result clients were not targeted by run {run_id} (e.g. {examples})"
        )

    matched = targeted.join(results.select("client_id", "treated", "purchased"), on="client_id")
    disagree = matched.filter(pl.col("send") != (pl.col("treated") == 1))
    if disagree.height:
        raise CampaignResultsError(
            f"treated disagrees with the batch's send/holdout assignment for {disagree.height} "
            f"clients (e.g. {', '.join(disagree['client_id'].head(_SHOWN_EXAMPLES))})"
        )
    frame = matched.select("client_id", "uplift_score", "decile", "treated", "purchased")
    data = CampaignData(
        scoring_run_id=run_id,
        frame=frame,
        n_targeted_treated=int(targeted["send"].sum()),
        n_targeted_control=int(targeted["holdout"].sum()),
    )
    if data.n_treated == 0 or data.n_control == 0:
        raise CampaignResultsError("Need results for both the treated and the control clients")
    return data
