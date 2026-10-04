"""Choose the final model from a validation leaderboard, by a rule fixed before test is seen.

On val the top uplift variants are statistically tied, so ranking them by
point estimate would pick the luckiest one (winner's curse). The rule is
parsimony: among the uplift variants whose paired Qini difference with the
best includes zero, take the cheapest to fit. Reference rankings (random, the
response model) are never candidates.

The rule reads only artifacts a leaderboard run already logged
(``leaderboard.csv`` and ``paired_comparisons.json``), so the choice is
reproducible from that run alone.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from mlflow import MlflowClient

from promolift.models.base import Params
from promolift.models.training import ModelSpec
from promolift.tracking.mlflow_tracking import TrackingConfig, default_tracking_config

# Base models that are reference rankings, not uplift models.
REFERENCE_BASE_MODELS = frozenset({"random", "response"})
SELECTION_RULE = "cheapest_tied_with_best"
_METRIC = "qini_auc"


@dataclass(frozen=True)
class Leaderboard:
    """A leaderboard run's ranked table, paired comparisons, and lineage tags."""

    run_id: str
    table: pl.DataFrame
    comparisons: list[dict]
    tags: dict[str, str]


@dataclass(frozen=True)
class Selection:
    """The chosen variant and the evidence behind it."""

    chosen: str
    best: str
    tied: tuple[str, ...]
    rule: str = SELECTION_RULE

    def as_dict(self) -> dict:
        return {
            "chosen": self.chosen,
            "best": self.best,
            "tied": list(self.tied),
            "rule": self.rule,
        }


def load_leaderboard(run_id: str, config: TrackingConfig | None = None) -> Leaderboard:
    """Download a leaderboard run's table and paired comparisons."""
    config = config if config is not None else default_tracking_config()
    client = MlflowClient(tracking_uri=config.tracking_uri)
    run = client.get_run(run_id)
    if run.data.tags.get("run_type") != "leaderboard":
        raise ValueError(f"Run {run_id} is not a leaderboard run")
    table = pl.read_csv(client.download_artifacts(run_id, "leaderboard/leaderboard.csv"))
    comparisons = json.loads(
        Path(client.download_artifacts(run_id, "leaderboard/paired_comparisons.json")).read_text()
    )
    return Leaderboard(run_id, table, comparisons, dict(run.data.tags))


def _difference_with(best: str, other: str, comparisons: Sequence[dict]) -> dict:
    for record in comparisons:
        if {record["model_a"], record["model_b"]} == {best, other}:
            return record[_METRIC]
    raise ValueError(f"The leaderboard has no paired comparison of {best!r} and {other!r}")


def select_final_model(table: pl.DataFrame, comparisons: Sequence[dict]) -> Selection:
    """The cheapest uplift variant statistically tied with the best on val Qini.

    "Tied" means the paired bootstrap CI of the Qini difference includes zero
    (the best is tied with itself). Fit time breaks ties between candidates;
    equal fit times fall back to higher Qini, then name, so the result is
    deterministic.

    Args:
        table: ``leaderboard.csv`` (``model``, ``base_model``, ``qini_auc``,
            ``fit_seconds`` columns).
        comparisons: ``paired_comparisons.json`` records.

    Raises:
        ValueError: If there is no uplift variant, or a variant has no paired
            comparison with the best.
    """
    candidates = table.filter(~pl.col("base_model").is_in(list(REFERENCE_BASE_MODELS)))
    if candidates.is_empty():
        raise ValueError("The leaderboard has no uplift model to choose from")
    best = candidates.sort(_METRIC, descending=True)["model"][0]

    def is_tied(model: str) -> bool:
        if model == best:
            return True
        difference = _difference_with(best, model, comparisons)
        return difference["lower"] <= 0 <= difference["upper"]

    tied = candidates.filter(pl.col("model").map_elements(is_tied, return_dtype=pl.Boolean)).sort(
        ["fit_seconds", _METRIC, "model"], descending=[False, True, False]
    )
    names = tuple(tied["model"].to_list())
    return Selection(chosen=names[0], best=best, tied=names)


def leaderboard_spec(table: pl.DataFrame, label: str, tuned_params: dict[str, Params]) -> ModelSpec:
    """Rebuild the ``ModelSpec`` a leaderboard row was trained with.

    Tuned variants get their overrides from the committed tuned-params file,
    the same source the leaderboard run used.

    Raises:
        ValueError: If ``label`` isn't on the leaderboard, or a tuned variant
            has no entry in ``tuned_params``.
    """
    rows = table.filter(pl.col("model") == label)
    if rows.is_empty():
        raise ValueError(f"{label!r} is not on the leaderboard")
    base_model, variant = rows["base_model"][0], rows["variant"][0]
    if variant != "tuned":
        return ModelSpec(label, base_model, variant=variant)
    if base_model not in tuned_params:
        raise ValueError(f"No tuned parameters for {base_model!r}")
    return ModelSpec(label, base_model, tuned_params[base_model], variant=variant)
