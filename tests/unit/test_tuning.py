"""Unit tests for cross-validated hyperparameter tuning on synthetic data."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from mlflow import MlflowClient

from promolift.models.dataset import ModelFrame
from promolift.models.tuning import (
    cross_validated_score,
    cv_folds,
    load_tuned_params,
    log_tuning,
    objective_for,
    promote_tuned_params,
    tune_model,
    write_tuned_params,
)
from promolift.tracking.mlflow_tracking import TrackingConfig, local_tracking_config


@pytest.fixture(scope="module")
def train(sure_things_data: dict) -> ModelFrame:
    features, treatment, outcome = sure_things_data["train"]
    return ModelFrame(
        split="train",
        client_ids=pl.Series([f"c{i}" for i in range(len(features))]),
        features=features,
        treatment=treatment,
        outcome=outcome,
    )


def test_folds_are_disjoint_cover_every_client_and_keep_strata(train: ModelFrame) -> None:
    folds = cv_folds(train.treatment, train.outcome, n_folds=3, seed=0)

    scored = np.concatenate([score for _, score in folds])
    assert len(folds) == 3
    assert sorted(scored) == list(range(len(train.treatment)))
    for fit, score in folds:
        assert not set(fit) & set(score)
        assert train.treatment[score].mean() == pytest.approx(train.treatment.mean(), abs=0.01)


def test_cross_validated_qini_reports_every_fold(train: ModelFrame) -> None:
    folds = cv_folds(train.treatment, train.outcome, n_folds=3, seed=0)

    result = cross_validated_score("t_learner", train, folds, overrides=None, seed=0)

    assert len(result.per_fold) == 3
    assert result.mean == pytest.approx(np.mean(result.per_fold))
    assert result.mean > 0.05


def test_optuna_tuning_never_ends_below_the_defaults(train: ModelFrame) -> None:
    result = tune_model("s_learner", train, n_trials=4, seed=0)

    assert result.search == "optuna"
    assert result.trials.height == 4
    # Defaults are enqueued as the first trial, so tuning can only match or beat them.
    assert result.trials["overrides"][0] == "{}"
    assert result.best_cv_score >= result.default_cv_score
    assert result.objective == "qini_auc"


def test_forest_tuning_searches_the_supplied_grid(train: ModelFrame) -> None:
    grid = [{}, {"min_samples_leaf": 50}]

    result = tune_model("causal_forest", train, grid=grid, seed=0)

    assert result.search == "grid"
    assert result.trials.height == 2
    assert result.best_params in grid


def test_tuned_params_round_trip_through_yaml(train: ModelFrame, tmp_path: Path) -> None:
    result = tune_model("class_transformation", train, n_trials=2, seed=0)
    path = tmp_path / "tuned_params.yaml"

    write_tuned_params([result], path, metadata={"git_commit": "abc123"})

    loaded = load_tuned_params(path)
    assert loaded == {"class_transformation": result.best_params}


def test_loading_missing_tuned_params_explains_how_to_create_them(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tune_models"):
        load_tuned_params(tmp_path / "absent.yaml")


@pytest.fixture
def config(tmp_path: Path) -> TrackingConfig:
    return local_tracking_config(tmp_path / "mlruns")


def test_log_tuning_records_best_and_default_scores_and_trials(
    train: ModelFrame, config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    result = tune_model("response", train, n_trials=2, seed=0)

    run_id = log_tuning(result, config=config, lineage=lineage_dirs)

    client = MlflowClient(tracking_uri=config.tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["run_type"] == "tuning"
    assert run.data.tags["objective"] == "treated_roc_auc"
    assert run.data.metrics["best_cv_score"] == pytest.approx(result.best_cv_score)
    assert run.data.metrics["default_cv_score"] == pytest.approx(result.default_cv_score)
    assert [a.path for a in client.list_artifacts(run_id, "tuning")] == ["tuning/trials.csv"]


def test_log_tuning_records_extra_tags_without_overriding_its_own(
    train: ModelFrame, config: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    result = tune_model("response", train, n_trials=2, seed=0)

    run_id = log_tuning(
        result,
        tags={"feature_groups": "demographics,basket_store", "run_type": "other"},
        config=config,
        lineage=lineage_dirs,
    )

    tags = MlflowClient(tracking_uri=config.tracking_uri).get_run(run_id).data.tags
    assert tags["feature_groups"] == "demographics,basket_store"
    assert tags["run_type"] == "tuning"


def test_unknown_models_cannot_be_tuned(train: ModelFrame) -> None:
    with pytest.raises(ValueError, match="random"):
        tune_model("random", train, n_trials=2, seed=0)


def test_frames_without_pandas_features_are_supported_by_iloc(train: ModelFrame) -> None:
    # cv folds index positionally; a non-default index must not break fold slicing.
    shifted = ModelFrame(
        split="train",
        client_ids=train.client_ids,
        features=train.features.set_index(pd.Index(range(10_000, 10_000 + len(train.features)))),
        treatment=train.treatment,
        outcome=train.outcome,
    )
    folds = cv_folds(shifted.treatment, shifted.outcome, n_folds=3, seed=0)

    assert cross_validated_score("t_learner", shifted, folds, overrides=None, seed=0).mean > 0.05


def test_response_model_is_tuned_on_its_own_goal_not_on_uplift(train: ModelFrame) -> None:
    result = tune_model("response", train, n_trials=2, seed=0)

    assert objective_for("response") == "treated_roc_auc"
    assert objective_for("s_learner") == "qini_auc"
    assert result.objective == "treated_roc_auc"
    # A purchase AUC, not a Qini: on sure-things data P(buy | treated) is very predictable.
    assert 0.6 < result.best_cv_score <= 1.0


def test_staged_results_reach_the_output_only_when_promoted(
    train: ModelFrame, tmp_path: Path
) -> None:
    # Writing into the repo mid-run would make later runs' git_dirty tag true;
    # results stage in a gitignored file and are promoted once at the end.
    staging, output = tmp_path / "interim" / "partial.yaml", tmp_path / "configs" / "tuned.yaml"
    first = tune_model("class_transformation", train, n_trials=2, seed=0)
    second = tune_model("s_learner", train, n_trials=2, seed=0)

    write_tuned_params([first], staging, metadata={})
    write_tuned_params([second], staging, metadata={})
    assert not output.exists()

    promote_tuned_params(staging, output)

    assert load_tuned_params(output) == {
        "class_transformation": first.best_params,
        "s_learner": second.best_params,
    }
    assert not staging.exists()


def test_promotion_keeps_output_entries_for_models_not_retuned(
    train: ModelFrame, tmp_path: Path
) -> None:
    staging, output = tmp_path / "partial.yaml", tmp_path / "tuned.yaml"
    kept = tune_model("class_transformation", train, n_trials=2, seed=0)
    retuned = tune_model("s_learner", train, n_trials=2, seed=0)
    write_tuned_params([kept], output, metadata={})

    write_tuned_params([retuned], staging, metadata={})
    promote_tuned_params(staging, output)

    assert set(load_tuned_params(output)) == {"class_transformation", "s_learner"}
