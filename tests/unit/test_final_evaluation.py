"""Unit tests for the one-time final evaluation: frames, preflight, the one-look guard, logging."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import pytest
from mlflow import MlflowClient

from promolift.data.split import split_assignment_path, write_split
from promolift.models.dataset import ModelFrame
from promolift.models.final_evaluation import (
    DRY_RUN_TYPE,
    FINAL_RUN_TYPE,
    FinalEvaluationExistsError,
    campaign_table,
    combine_frames,
    evaluate_final,
    final_runs,
    fit_models,
    preflight_problems,
)
from promolift.models.selection import Selection
from promolift.models.training import ModelSpec
from promolift.tracking.lineage import GitState, split_sha256
from promolift.tracking.mlflow_tracking import TrackingConfig, local_tracking_config

_FAST = {"n_bootstrap": 20, "n_rankings": 30}
_SELECTION = Selection(chosen="t_learner", best="t_learner", tied=("t_learner",))
_SPECS = [ModelSpec(name, name) for name in ("t_learner", "response", "random")]


def _frame(split: str, n: int, seed: int) -> ModelFrame:
    # Sure things (b >= 0.5) buy 80% regardless; persuadables 10% -> 40% if treated.
    rng = np.random.default_rng(seed)
    b = rng.random(n)
    treatment = rng.integers(0, 2, n)
    outcome = (rng.random(n) < np.where(b >= 0.5, 0.8, 0.1 + 0.3 * treatment)).astype(int)
    return ModelFrame(
        split=split,
        client_ids=pl.Series([f"{split}{i}" for i in range(n)]),
        features=pd.DataFrame(
            {
                "b": b,
                "segment": pd.Categorical(rng.choice(["F", "M"], n), categories=["F", "M", "U"]),
            }
        ),
        treatment=treatment,
        outcome=outcome,
    )


@pytest.fixture
def config(tmp_path: Path) -> TrackingConfig:
    return local_tracking_config(tmp_path / "mlruns")


@pytest.fixture
def lineage_with_split(lineage_dirs: dict[str, Path]) -> dict[str, Path]:
    assignment = pl.DataFrame({"client_id": ["a", "b", "c"], "split": ["train", "val", "test"]})
    write_split(assignment, split_assignment_path(lineage_dirs["processed_dir"]))
    return lineage_dirs


@pytest.fixture(scope="module")
def fitted() -> list:
    return fit_models(_SPECS, _frame("train", 3_000, seed=0))


def _evaluate(fitted, config, lineage, *, final_evaluation: bool, loader=None, max_workers=1):
    split = "test" if final_evaluation else "val"
    return evaluate_final(
        fitted,
        loader or (lambda: _frame(split, 2_000, seed=1)),
        selection=_SELECTION,
        leaderboard_run_id="leaderboard123",
        fit_split="train",
        n_fit_clients=3_000,
        final_evaluation=final_evaluation,
        max_workers=max_workers,
        config=config,
        lineage=lineage,
        **_FAST,
    )


# --- combine_frames ----------------------------------------------------------


def test_combine_frames_stacks_clients_and_keeps_categoricals() -> None:
    train, val = _frame("train", 30, seed=0), _frame("val", 20, seed=1)

    combined = combine_frames([train, val], "train+val")

    assert combined.split == "train+val"
    assert len(combined.features) == len(combined.treatment) == len(combined.outcome) == 50
    assert combined.client_ids.to_list() == train.client_ids.to_list() + val.client_ids.to_list()
    assert np.array_equal(combined.outcome, np.concatenate([train.outcome, val.outcome]))
    assert list(combined.features["segment"].cat.categories) == ["F", "M", "U"]


def test_combine_frames_rejects_shared_clients() -> None:
    frame = _frame("train", 10, seed=0)

    with pytest.raises(ValueError, match="share clients"):
        combine_frames([frame, frame], "train+train")


def test_combine_frames_rejects_different_columns() -> None:
    train, val = _frame("train", 10, seed=0), _frame("val", 10, seed=1)
    val.features = val.features.rename(columns={"b": "c"})

    with pytest.raises(ValueError, match="same feature columns"):
        combine_frames([train, val], "train+val")


def test_combine_frames_rejects_mismatched_categories() -> None:
    train, val = _frame("train", 10, seed=0), _frame("val", 10, seed=1)
    val.features["segment"] = val.features["segment"].cat.remove_unused_categories()

    with pytest.raises(ValueError, match="different categories"):
        combine_frames([train, val], "train+val")


# --- preflight ---------------------------------------------------------------

_TAGS = {"split_sha256": "abc", "raw_data_digest": "def"}
_SHARED = TrackingConfig("databricks://promolift")


def _problems(**overrides) -> list[str]:
    kwargs = {
        "git": GitState("c0ffee", is_dirty=False),
        "split_hash": "abc",
        "raw_data_digest": "def",
        "leaderboard_tags": _TAGS,
        "config": _SHARED,
    } | overrides
    return preflight_problems(**kwargs)


def test_preflight_passes_on_a_clean_commit_with_matching_data() -> None:
    assert _problems() == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"git": GitState("c0ffee", is_dirty=True)}, "uncommitted"),
        ({"git": GitState("unknown", is_dirty=None)}, "uncommitted"),
        ({"config": TrackingConfig("sqlite:///x/mlflow.db")}, "not the shared store"),
        ({"split_hash": None}, "split file is missing"),
        ({"split_hash": "other"}, "split differs"),
        ({"raw_data_digest": "other"}, "raw data differs"),
    ],
)
def test_preflight_reports_each_problem(overrides: dict, expected: str) -> None:
    problems = _problems(**overrides)

    assert len(problems) == 1
    assert expected in problems[0]


# --- evaluate_final ----------------------------------------------------------


def test_dry_run_logs_the_chosen_model_first_with_references_and_artifacts(
    fitted: list, config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    final = _evaluate(fitted, config, lineage_dirs, final_evaluation=False)

    assert [r.name for r in final.results] == ["t_learner", "response", "random"]
    assert {(c["model_a"], c["model_b"]) for c in final.comparisons} == {
        ("t_learner", "response"),
        ("t_learner", "random"),
    }
    client = MlflowClient(tracking_uri=config.tracking_uri)
    run = client.get_run(final.run_id)
    assert run.data.tags["run_type"] == DRY_RUN_TYPE
    assert run.data.tags["chosen_model"] == "t_learner"
    assert run.data.tags["reference_models"] == "response,random"
    assert run.data.tags["leaderboard_run_id"] == "leaderboard123"
    assert run.data.tags["final_evaluation"] == "false"
    assert run.data.metrics["val_qini_auc"] == pytest.approx(
        final.results[0].report.metrics.qini_auc
    )
    assert "response_val_qini_auc" in run.data.metrics
    assert "qini_diff_t_learner_vs_response" in run.data.metrics
    artifacts = {a.path for a in client.list_artifacts(final.run_id, "final")}
    assert artifacts == {
        "final/selection.json",
        "final/results.csv",
        "final/paired_comparisons.json",
        "final/campaign.csv",
        "final/qini_curves.png",
    }
    selection = json.loads(
        Path(client.download_artifacts(final.run_id, "final/selection.json")).read_text()
    )
    assert selection["chosen"] == "t_learner"


def test_parallel_evaluation_matches_sequential_exactly(
    fitted: list, config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    sequential = _evaluate(fitted, config, lineage_dirs, final_evaluation=False)
    parallel = _evaluate(fitted, config, lineage_dirs, final_evaluation=False, max_workers=2)

    for a, b in zip(sequential.results, parallel.results, strict=True):
        assert a.name == b.name
        assert a.report.bootstrap == b.report.bootstrap
        assert a.report.noise_floor == b.report.noise_floor
    assert sequential.comparisons == parallel.comparisons


def test_the_evaluation_frame_is_loaded_only_inside_the_run(
    fitted: list, config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    active = []

    def loader() -> ModelFrame:
        active.append(mlflow.active_run())
        return _frame("val", 2_000, seed=1)

    final = _evaluate(fitted, config, lineage_dirs, final_evaluation=False, loader=loader)

    assert len(active) == 1
    assert active[0] is not None
    assert active[0].info.run_id == final.run_id


def test_a_second_official_run_is_refused_before_loading_test(
    fitted: list, config: TrackingConfig, lineage_with_split: dict[str, Path]
) -> None:
    first = _evaluate(fitted, config, lineage_with_split, final_evaluation=True)
    loads = []

    with pytest.raises(FinalEvaluationExistsError, match=first.run_id):
        _evaluate(
            fitted,
            config,
            lineage_with_split,
            final_evaluation=True,
            loader=lambda: loads.append(1) or _frame("test", 2_000, seed=1),
        )

    assert loads == []
    tags = MlflowClient(tracking_uri=config.tracking_uri).get_run(first.run_id).data.tags
    assert tags["run_type"] == FINAL_RUN_TYPE
    assert tags["final_evaluation"] == "true"


def test_a_crash_after_loading_test_still_blocks_a_second_run(
    fitted: list, config: TrackingConfig, lineage_with_split: dict[str, Path]
) -> None:
    def crashing_loader() -> ModelFrame:
        raise RuntimeError("crashed after opening the run")

    with pytest.raises(RuntimeError, match="crashed"):
        _evaluate(fitted, config, lineage_with_split, final_evaluation=True, loader=crashing_loader)

    with pytest.raises(FinalEvaluationExistsError):
        _evaluate(fitted, config, lineage_with_split, final_evaluation=True)


def test_dry_runs_do_not_count_as_the_final_evaluation(
    fitted: list, config: TrackingConfig, lineage_with_split: dict[str, Path]
) -> None:
    _evaluate(fitted, config, lineage_with_split, final_evaluation=False)

    assert final_runs(split_sha256(lineage_with_split["processed_dir"]), config) == []
    _evaluate(fitted, config, lineage_with_split, final_evaluation=True)


def test_deleting_the_final_run_by_hand_is_the_only_way_to_rerun(
    fitted: list, config: TrackingConfig, lineage_with_split: dict[str, Path]
) -> None:
    first = _evaluate(fitted, config, lineage_with_split, final_evaluation=True)
    MlflowClient(tracking_uri=config.tracking_uri).delete_run(first.run_id)

    second = _evaluate(fitted, config, lineage_with_split, final_evaluation=True)

    assert second.run_id != first.run_id


def test_the_guard_is_per_split(
    fitted: list, config: TrackingConfig, lineage_with_split: dict[str, Path]
) -> None:
    first = _evaluate(fitted, config, lineage_with_split, final_evaluation=True)

    assert final_runs(split_sha256(lineage_with_split["processed_dir"]), config) == [first.run_id]
    assert final_runs("another-split", config) == []


def test_an_official_run_without_the_split_file_is_refused(
    fitted: list, config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    with pytest.raises(ValueError, match="without the split file"):
        _evaluate(fitted, config, lineage_dirs, final_evaluation=True)


def test_the_chosen_model_must_be_fitted(
    fitted: list, config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    with pytest.raises(ValueError, match="was not fitted"):
        _evaluate(fitted[1:], config, lineage_dirs, final_evaluation=False)


def test_campaign_table_converts_uplift_at_k_to_purchases_per_1000(
    fitted: list, config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    final = _evaluate(fitted, config, lineage_dirs, final_evaluation=False)

    table = campaign_table(final.results)

    report = final.results[0].report
    row = table.filter((pl.col("model") == "t_learner") & (pl.col("targeted_pct") == 30))
    assert row["extra_purchases_per_1000"][0] == pytest.approx(
        report.bootstrap.uplift_at_k[0.3].estimate * 1000
    )
    assert row["random_targeting_per_1000"][0] == pytest.approx(report.ate * 1000)
    assert table.height == 3 * len(report.bootstrap.uplift_at_k)
