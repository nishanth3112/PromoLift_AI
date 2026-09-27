"""Hyperparameter tuning by cross-validation on the training split only.

Validation stays an untouched selection holdout: every trial is scored by
3-fold CV inside train, stratified on treatment x outcome. LightGBM-based
models get an Optuna search whose first trial is always the defaults, so the
tuned result can only match or beat them on CV. The slow forests get a small
fixed grid that includes their defaults.

Uplift models are tuned on Qini AUC. The response model is tuned on its own
goal -- ROC AUC of P(buy) on treated clients -- because that is how the
incumbent is built in practice; picking it by uplift would turn it into an
uplift model and stop it representing the traditional approach.

Tuned settings are written to ``configs/tuned_params.yaml`` for review and
commit, so a training run is reproducible from the commit alone.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import optuna
import polars as pl
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from promolift.data.loader import project_root
from promolift.evaluation.ranking_metrics import ranking_metrics
from promolift.models.base import Params
from promolift.models.dataset import ModelFrame
from promolift.models.registry import DEFAULT_SEED, build_model
from promolift.tracking.mlflow_tracking import Experiment, TrackingConfig, start_run

DEFAULT_N_FOLDS = 3
DEFAULT_N_TRIALS = 40
TUNED_PARAMS_PATH = Path("configs") / "tuned_params.yaml"
QINI_OBJECTIVE = "qini_auc"
TREATED_AUC_OBJECTIVE = "treated_roc_auc"
_OBJECTIVES = {"response": TREATED_AUC_OBJECTIVE}

OPTUNA_MODELS = (
    "response",
    "s_learner",
    "t_learner",
    "class_transformation",
    "x_learner",
    "dr_learner",
)
# Each grid's first entry is the model's defaults. Sized from measured cost:
# one 3-fold CV trial takes ~77s (causal forest) and ~232s (uplift RF).
FOREST_GRIDS: dict[str, list[Params]] = {
    "causal_forest": [
        {},
        {"min_samples_leaf": 50},
        {"min_samples_leaf": 200},
        {"max_samples": 0.3},
        {"min_samples_leaf": 50, "max_samples": 0.3},
        {"min_samples_leaf": 200, "max_samples": 0.3},
    ],
    "uplift_rf": [
        {},
        {"min_samples_leaf": 100},
        {"max_depth": 6},
        {"max_depth": 6, "min_samples_leaf": 100},
    ],
}


@dataclass(frozen=True)
class CVScore:
    """The tuning objective on each held-out fold, and its mean and spread."""

    mean: float
    std: float
    per_fold: tuple[float, ...]


@dataclass
class TuningResult:
    """The outcome of tuning one model."""

    model_name: str
    search: str
    objective: str
    n_folds: int
    seed: int
    best_params: Params
    best_cv_score: float
    best_cv_std: float
    default_cv_score: float
    default_cv_std: float
    trials: pl.DataFrame


def cv_folds(
    treatment: np.ndarray, outcome: np.ndarray, *, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Positional (fit, score) index pairs, stratified on the treatment x outcome strata."""
    strata = np.asarray(treatment) * 2 + np.asarray(outcome)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(strata)), strata))


def objective_for(model_name: str) -> str:
    """The metric a model is tuned on: its own goal for the response model, else Qini."""
    return _OBJECTIVES.get(model_name, QINI_OBJECTIVE)


def _fold_score(objective: str, scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray):
    if objective == TREATED_AUC_OBJECTIVE:
        treated = treatment == 1
        return roc_auc_score(outcome[treated], scores[treated])
    return ranking_metrics(scores, treatment, outcome).qini_auc


def cross_validated_score(
    model_name: str,
    frame: ModelFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    overrides: Params | None,
    seed: int,
    objective: str = QINI_OBJECTIVE,
) -> CVScore:
    """Fit on each fold's training part, score ``objective`` on its held-out part."""
    scores = []
    for fit_idx, score_idx in folds:
        model = build_model(model_name, seed=seed, overrides=overrides).fit(
            frame.features.iloc[fit_idx], frame.treatment[fit_idx], frame.outcome[fit_idx]
        )
        predictions = model.predict_uplift(frame.features.iloc[score_idx])
        scores.append(
            float(
                _fold_score(
                    objective, predictions, frame.treatment[score_idx], frame.outcome[score_idx]
                )
            )
        )
    return CVScore(float(np.mean(scores)), float(np.std(scores)), tuple(scores))


def _suggest_lgbm_params(trial: optuna.Trial) -> Params:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 96, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 2000, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def _trial_row(number: int, overrides: Params, score: CVScore) -> dict:
    return {
        "trial": number,
        "overrides": json.dumps(overrides, sort_keys=True),
        "cv_score": score.mean,
        "cv_score_std": score.std,
    }


def tune_model(
    model_name: str,
    train: ModelFrame,
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    grid: Sequence[Params] | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
) -> TuningResult:
    """Tune one model by its cross-validated objective (see ``objective_for``) on ``train``.

    Args:
        model_name: An Optuna-tuned LightGBM model or a grid-tuned forest.
        n_trials: Optuna trials (LightGBM models); the first is the defaults.
        grid: Override the forest grid (mainly for tests).

    Raises:
        ValueError: If ``model_name`` has no search space.
    """
    folds = cv_folds(train.treatment, train.outcome, n_folds=n_folds, seed=seed)
    objective = objective_for(model_name)
    rows: list[dict] = []

    def evaluate(overrides: Params) -> float:
        score = cross_validated_score(
            model_name, train, folds, overrides=overrides, seed=seed, objective=objective
        )
        rows.append(_trial_row(len(rows), overrides, score))
        return score.mean

    if model_name in OPTUNA_MODELS:
        search = "optuna"

        def optuna_objective(trial: optuna.Trial) -> float:
            # Trial 0 is the untouched defaults: the baseline tuning must beat.
            return evaluate({} if trial.number == 0 else _suggest_lgbm_params(trial))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(optuna_objective, n_trials=n_trials)
    elif model_name in FOREST_GRIDS or grid is not None:
        search = "grid"
        for overrides in grid if grid is not None else FOREST_GRIDS[model_name]:
            evaluate(dict(overrides))
    else:
        tunable = [*OPTUNA_MODELS, *FOREST_GRIDS]
        raise ValueError(f"No search space for {model_name!r}; tunable models: {tunable}")

    trials = pl.DataFrame(rows)
    best = trials.sort("cv_score", descending=True).row(0, named=True)
    default = trials.row(0, named=True)
    return TuningResult(
        model_name=model_name,
        search=search,
        objective=objective,
        n_folds=n_folds,
        seed=seed,
        best_params=json.loads(best["overrides"]),
        best_cv_score=best["cv_score"],
        best_cv_std=best["cv_score_std"],
        default_cv_score=default["cv_score"],
        default_cv_std=default["cv_score_std"],
        trials=trials,
    )


def log_tuning(
    result: TuningResult,
    *,
    config: TrackingConfig | None = None,
    lineage: dict[str, Path] | None = None,
) -> str:
    """Log one model's search (objective, best/default CV scores, all trials) as a tuning run."""
    with start_run(
        Experiment.MODEL_TUNING,
        run_name=f"tune-{result.model_name}",
        tags={
            "run_type": "tuning",
            "model_name": result.model_name,
            "search": result.search,
            "objective": result.objective,
        },
        config=config,
        **(lineage or {}),
    ) as run:
        mlflow.log_params(
            {
                "n_trials": result.trials.height,
                "n_folds": result.n_folds,
                "seed": result.seed,
                **{f"best_{key}": value for key, value in result.best_params.items()},
            }
        )
        mlflow.log_metrics(
            {
                "best_cv_score": result.best_cv_score,
                "best_cv_score_std": result.best_cv_std,
                "default_cv_score": result.default_cv_score,
                "default_cv_score_std": result.default_cv_std,
                "cv_gain_over_default": result.best_cv_score - result.default_cv_score,
            }
        )
        mlflow.log_text(result.trials.write_csv(), "tuning/trials.csv")
        return run.info.run_id


def write_tuned_params(
    results: Sequence[TuningResult], path: Path | None = None, *, metadata: dict
) -> Path:
    """Write (or update) the reviewable, committed tuned-parameters file.

    Entries for models not in ``results`` are kept, so models can be re-tuned
    one at a time. Each entry records its CV evidence and provenance.
    """
    path = path if path is not None else project_root() / TUNED_PARAMS_PATH
    existing = yaml.safe_load(path.read_text()) if path.exists() else None
    models = (existing or {}).get("models", {})
    for result in results:
        models[result.model_name] = {
            "params": result.best_params,
            "search": result.search,
            "objective": result.objective,
            "n_trials": result.trials.height,
            "n_folds": result.n_folds,
            "seed": result.seed,
            "best_cv_score": round(result.best_cv_score, 6),
            "best_cv_score_std": round(result.best_cv_std, 6),
            "default_cv_score": round(result.default_cv_score, 6),
            "default_cv_score_std": round(result.default_cv_std, 6),
            **metadata,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Generated by scripts/tune_models.py -- review, then commit.\n"
    path.write_text(header + yaml.safe_dump({"models": models}, sort_keys=False))
    return path


def load_tuned_params(path: Path | None = None) -> dict[str, Params]:
    """Model name -> tuned overrides, from the committed tuned-parameters file."""
    path = path if path is not None else project_root() / TUNED_PARAMS_PATH
    if not path.exists():
        msg = f"No tuned parameters at {path}; run `scripts/tune_models.py` first."
        raise FileNotFoundError(msg)
    models = (yaml.safe_load(path.read_text()) or {}).get("models", {})
    return {name: entry["params"] for name, entry in models.items()}
