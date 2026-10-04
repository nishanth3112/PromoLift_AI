"""Unit tests for the final-model selection rule and reading a leaderboard run."""

from pathlib import Path

import mlflow
import polars as pl
import pytest

from promolift.models.selection import (
    leaderboard_spec,
    load_leaderboard,
    select_final_model,
)
from promolift.tracking.mlflow_tracking import TrackingConfig, local_tracking_config, start_run


def _table(rows: list[tuple[str, str, str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema=["model", "base_model", "variant", "qini_auc", "fit_seconds"], orient="row"
    )


def _diff(a: str, b: str, lower: float, upper: float) -> dict:
    return {
        "model_a": a,
        "model_b": b,
        "qini_auc": {"difference": (lower + upper) / 2, "lower": lower, "upper": upper},
    }


# best "ct_tuned" (slow-ish); "s_tuned" tied and cheapest uplift model; "t" clearly
# worse; "random" is cheaper than everything but is a reference, never a candidate.
_TABLE = _table(
    [
        ("ct_tuned", "class_transformation", "tuned", 0.016, 2.0),
        ("s_tuned", "s_learner", "tuned", 0.015, 1.0),
        ("t", "t_learner", "default", 0.009, 0.5),
        ("random", "random", "default", -0.004, 0.0),
        ("response", "response", "default", -0.010, 0.1),
    ]
)
_COMPARISONS = [
    _diff("ct_tuned", "s_tuned", -0.003, 0.005),
    _diff("ct_tuned", "t", 0.001, 0.012),
    _diff("ct_tuned", "random", 0.010, 0.030),
    _diff("ct_tuned", "response", 0.015, 0.035),
]


def test_picks_the_cheapest_model_tied_with_the_best() -> None:
    selection = select_final_model(_TABLE, _COMPARISONS)

    assert selection.best == "ct_tuned"
    assert selection.chosen == "s_tuned"
    assert selection.tied == ("s_tuned", "ct_tuned")


def test_reference_models_are_never_candidates() -> None:
    # random has the best Qini and the lowest fit time here, yet isn't eligible.
    table = _TABLE.with_columns(
        pl.when(pl.col("model") == "random")
        .then(0.5)
        .otherwise(pl.col("qini_auc"))
        .alias("qini_auc")
    )

    selection = select_final_model(table, _COMPARISONS)

    assert selection.best == "ct_tuned"
    assert "random" not in selection.tied


def test_the_best_is_chosen_when_it_is_also_the_cheapest() -> None:
    table = _TABLE.with_columns(
        pl.when(pl.col("model") == "ct_tuned")
        .then(0.2)
        .otherwise(pl.col("fit_seconds"))
        .alias("fit_seconds")
    )

    assert select_final_model(table, _COMPARISONS).chosen == "ct_tuned"


def test_comparison_orientation_does_not_matter() -> None:
    flipped = [_diff("s_tuned", "ct_tuned", -0.005, 0.003), *_COMPARISONS[1:]]

    assert select_final_model(_TABLE, flipped).chosen == "s_tuned"


def test_equal_fit_times_fall_back_to_higher_qini() -> None:
    table = _TABLE.with_columns(pl.lit(1.0).alias("fit_seconds"))

    assert select_final_model(table, _COMPARISONS).chosen == "ct_tuned"


def test_a_candidate_without_a_comparison_with_the_best_is_an_error() -> None:
    with pytest.raises(ValueError, match="no paired comparison"):
        select_final_model(_TABLE, _COMPARISONS[1:])


def test_a_leaderboard_of_only_references_is_an_error() -> None:
    with pytest.raises(ValueError, match="no uplift model"):
        select_final_model(_TABLE.filter(pl.col("base_model").is_in(["random", "response"])), [])


def test_leaderboard_spec_restores_tuned_overrides() -> None:
    tuned = {"class_transformation": {"num_leaves": 8}}

    spec = leaderboard_spec(_TABLE, "ct_tuned", tuned)
    default = leaderboard_spec(_TABLE, "t", tuned)

    assert (spec.label, spec.model_name, spec.variant) == (
        "ct_tuned",
        "class_transformation",
        "tuned",
    )
    assert spec.overrides == {"num_leaves": 8}
    assert (default.model_name, default.overrides, default.variant) == (
        "t_learner",
        None,
        "default",
    )


def test_leaderboard_spec_rejects_unknown_labels_and_missing_tuned_params() -> None:
    with pytest.raises(ValueError, match="not on the leaderboard"):
        leaderboard_spec(_TABLE, "x_learner", {})
    with pytest.raises(ValueError, match="No tuned parameters"):
        leaderboard_spec(_TABLE, "s_tuned", {})


@pytest.fixture
def config(tmp_path: Path) -> TrackingConfig:
    return local_tracking_config(tmp_path / "mlruns")


def test_load_leaderboard_reads_the_logged_artifacts(
    config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    with start_run(
        "leaderboard-test", tags={"run_type": "leaderboard"}, config=config, **lineage_dirs
    ) as run:
        mlflow.log_text(_TABLE.write_csv(), "leaderboard/leaderboard.csv")
        mlflow.log_dict(_COMPARISONS, "leaderboard/paired_comparisons.json")

    leaderboard = load_leaderboard(run.info.run_id, config)

    assert leaderboard.table["model"].to_list() == _TABLE["model"].to_list()
    assert leaderboard.comparisons == _COMPARISONS
    assert leaderboard.tags["run_type"] == "leaderboard"
    assert "split_sha256" in leaderboard.tags


def test_load_leaderboard_rejects_other_runs(
    config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    with start_run(
        "leaderboard-test", tags={"run_type": "model"}, config=config, **lineage_dirs
    ) as run:
        pass

    with pytest.raises(ValueError, match="not a leaderboard run"):
        load_leaderboard(run.info.run_id, config)
