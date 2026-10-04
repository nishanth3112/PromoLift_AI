"""The one-time final evaluation: refit the chosen model, then score it on test exactly once.

Order matters. Models are fitted before the MLflow run opens, so a crash
while fitting costs nothing. The run then opens *before* the evaluation
frame is loaded, so any look at test is on record: a run that crashes after
loading test still counts as the one look. ``assert_no_final_run`` refuses a
second official run for the same split. There is deliberately no override;
the only recovery is deleting the run by hand in MLflow.

A dry run follows the same path on train -> val with its own run type, so it
exercises everything without touching test or tripping the guard.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlflow
import pandas as pd
import polars as pl
from mlflow import MlflowClient

from promolift.evaluation.plots import qini_curve_figure
from promolift.evaluation.ranking_metrics import uplift_at_k_name
from promolift.evaluation.report import evaluate_ranking, log_evaluation
from promolift.evaluation.uncertainty import (
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_N_RANKINGS,
    DEFAULT_SEED,
    paired_comparison,
    random_ranking_noise_floor,
)
from promolift.models.base import UpliftModel
from promolift.models.dataset import ModelFrame
from promolift.models.registry import build_model
from promolift.models.selection import Selection
from promolift.models.training import (
    ModelSpec,
    TrainedModel,
    comparison_record,
    leaderboard_row,
)
from promolift.tracking.lineage import GitState, split_sha256
from promolift.tracking.mlflow_tracking import (
    Experiment,
    TrackingConfig,
    default_tracking_config,
    start_run,
)

FINAL_RUN_TYPE = "final_evaluation"
DRY_RUN_TYPE = "final_evaluation_dry_run"
_SHARED_STORE_PREFIX = "databricks"


class FinalEvaluationExistsError(RuntimeError):
    """An official final evaluation was already run for this split."""


def combine_frames(frames: Sequence[ModelFrame], split: str) -> ModelFrame:
    """Stack model frames (e.g. train and val) into one fitting frame labelled ``split``.

    Raises:
        ValueError: If the frames have different feature columns or share a client.
    """
    columns = list(frames[0].features.columns)
    if any(list(frame.features.columns) != columns for frame in frames):
        raise ValueError("Frames must have the same feature columns to be combined")
    client_ids = pl.concat([frame.client_ids for frame in frames])
    if client_ids.n_unique() != len(client_ids):
        raise ValueError("Frames to combine share clients")
    features = pd.concat([frame.features for frame in frames], ignore_index=True)
    # Every frame's categoricals come from the whole feature table, so they
    # share levels; a mismatch would silently turn the column into object.
    for column in frames[0].features.select_dtypes("category"):
        if not isinstance(features[column].dtype, pd.CategoricalDtype):
            raise ValueError(f"Column {column!r} has different categories across frames")
    return ModelFrame(
        split=split,
        client_ids=client_ids,
        features=features,
        treatment=pd.concat([pd.Series(f.treatment) for f in frames]).to_numpy(),
        outcome=pd.concat([pd.Series(f.outcome) for f in frames]).to_numpy(),
    )


def final_runs(split_hash: str, config: TrackingConfig | None = None) -> list[str]:
    """Ids of official final-evaluation runs for ``split_hash``; deleted runs don't count."""
    config = config if config is not None else default_tracking_config()
    client = MlflowClient(tracking_uri=config.tracking_uri)
    experiment = client.get_experiment_by_name(config.experiment_name(Experiment.FINAL_EVALUATION))
    if experiment is None:
        return []
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.run_type = '{FINAL_RUN_TYPE}' and tags.split_sha256 = '{split_hash}'"
        ),
    )
    return [run.info.run_id for run in runs]


def assert_no_final_run(split_hash: str, config: TrackingConfig | None = None) -> None:
    """Refuse a second look at test: one official final evaluation per split, ever.

    Raises:
        FinalEvaluationExistsError: If a final-evaluation run for ``split_hash``
            exists, finished or not.
    """
    if existing := final_runs(split_hash, config):
        msg = (
            f"A final evaluation for split {split_hash[:12]}... already exists "
            f"(run(s) {', '.join(existing)}). Test is evaluated once; there is no override."
        )
        raise FinalEvaluationExistsError(msg)


def preflight_problems(
    *,
    git: GitState,
    split_hash: str | None,
    raw_data_digest: str,
    leaderboard_tags: dict[str, str],
    config: TrackingConfig,
) -> list[str]:
    """Reasons the official run must not start; empty when it may.

    The run has to be reproducible from its commit, logged where the team
    (and the one-look guard) can see it, and on the same data and split the
    model was selected on.
    """
    problems = []
    if git.is_dirty is not False:
        problems.append(
            "the working tree has uncommitted or untracked changes; commit first "
            f"(git_dirty={git.is_dirty})"
        )
    if not config.tracking_uri.startswith(_SHARED_STORE_PREFIX):
        problems.append(
            f"tracking to {config.tracking_uri}, not the shared store; "
            "run with `uv run --env-file .env ...`"
        )
    if split_hash is None:
        problems.append("the split file is missing; run scripts/fetch_split.py")
    elif split_hash != leaderboard_tags.get("split_sha256"):
        problems.append("the split differs from the one the leaderboard was evaluated on")
    if raw_data_digest != leaderboard_tags.get("raw_data_digest"):
        problems.append("the raw data differs from the data the leaderboard was trained on")
    return problems


@dataclass
class FittedModel:
    """A variant fitted on the final fitting frame."""

    spec: ModelSpec
    model: UpliftModel
    fit_seconds: float


def fit_models(
    specs: Sequence[ModelSpec], frame: ModelFrame, *, seed: int = DEFAULT_SEED
) -> list[FittedModel]:
    """Fit each variant on ``frame``, as ``train_and_evaluate`` does but without logging."""
    fitted = []
    for spec in specs:
        model = build_model(spec.model_name, seed=seed, overrides=spec.overrides)
        started = time.perf_counter()
        model.fit(frame.features, frame.treatment, frame.outcome)
        fitted.append(FittedModel(spec, model, time.perf_counter() - started))
    return fitted


def campaign_table(results: Sequence[TrainedModel]) -> pl.DataFrame:
    """Extra purchases caused per 1,000 clients targeted, at each targeting depth.

    Uplift@k is the treated-minus-control conversion among the top k%, i.e.
    extra purchases per targeted client; sending to a random k% earns the ATE.
    """
    rows = []
    for result in results:
        report = result.report
        for k, interval in report.bootstrap.uplift_at_k.items():
            rows.append(
                {
                    "model": result.name,
                    "targeted_pct": round(k * 100),
                    "extra_purchases_per_1000": interval.estimate * 1000,
                    "ci_lower": interval.lower * 1000,
                    "ci_upper": interval.upper * 1000,
                    "random_targeting_per_1000": report.ate * 1000,
                }
            )
    return pl.DataFrame(rows)


def _run_all(calls: Sequence[partial], max_workers: int | None) -> list:
    """Run independent calls, in worker processes unless ``max_workers`` is 1.

    Each call carries its own seed, so results are identical either way; the
    bootstrap loops are CPU-bound Python, so threads wouldn't help.
    """
    if max_workers == 1:
        return [call() for call in calls]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(call) for call in calls]
        return [future.result() for future in futures]


@dataclass
class FinalEvaluation:
    """The evaluation run, every model's result (chosen first), and chosen-vs-reference tests."""

    run_id: str
    results: list[TrainedModel]
    comparisons: list[dict]


def evaluate_final(
    fitted: Sequence[FittedModel],
    load_frame: Callable[[], ModelFrame],
    *,
    selection: Selection,
    leaderboard_run_id: str,
    fit_split: str,
    n_fit_clients: int,
    final_evaluation: bool,
    seed: int = DEFAULT_SEED,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    n_rankings: int = DEFAULT_N_RANKINGS,
    max_workers: int | None = None,
    config: TrackingConfig | None = None,
    lineage: dict[str, Path] | None = None,
) -> FinalEvaluation:
    """Score fitted models on the evaluation frame inside one MLflow run.

    The chosen model's report is logged as the run's headline metrics
    (``test_qini_auc``, ...); every model's row, the chosen-vs-reference paired
    comparisons, the campaign table, and Qini curves are logged as artifacts.

    Args:
        fitted: Fitted variants; must include ``selection.chosen``.
        load_frame: Loads the evaluation frame. Called only after the run is
            open, so a look at test is always recorded.
        final_evaluation: True for the official test run (guarded, one per
            split); False for a dry run on val.
        fit_split: Label of what the models were fitted on, e.g. ``train+val``.
        max_workers: Processes for the bootstrap evaluations and comparisons;
            None uses every core, 1 runs them in this process.
        lineage: ``repo_dir`` / ``data_dir`` / ``processed_dir`` overrides (for tests).

    Raises:
        ValueError: If the chosen model isn't among ``fitted``, or the split
            hash is missing for an official run.
        FinalEvaluationExistsError: If an official run already exists for this split.
    """
    by_label = {f.spec.label: f for f in fitted}
    if selection.chosen not in by_label:
        raise ValueError(f"The chosen model {selection.chosen!r} was not fitted")
    references = [label for label in by_label if label != selection.chosen]

    config = config if config is not None else default_tracking_config()
    if final_evaluation:
        split_hash = split_sha256((lineage or {}).get("processed_dir"))
        if split_hash is None:
            raise ValueError("Cannot guard the final evaluation without the split file")
        assert_no_final_run(split_hash, config)

    run_type = FINAL_RUN_TYPE if final_evaluation else DRY_RUN_TYPE
    with start_run(
        Experiment.FINAL_EVALUATION,
        run_name=run_type,
        tags={
            "run_type": run_type,
            "chosen_model": selection.chosen,
            "best_val_model": selection.best,
            "selection_rule": selection.rule,
            "reference_models": ",".join(references),
            "leaderboard_run_id": leaderboard_run_id,
            "fit_split": fit_split,
        },
        config=config,
        **(lineage or {}),
    ) as run:
        frame = load_frame()
        labels = (selection.chosen, *references)
        scores = {label: by_label[label].model.predict_uplift(frame.features) for label in labels}
        t, y = frame.treatment, frame.outcome

        # The comparisons and the noise floor are independent; the reports need the floor.
        noise_floor, *paired = _run_all(
            [
                partial(random_ranking_noise_floor, t, y, n_rankings=n_rankings, seed=seed),
                *(
                    partial(
                        paired_comparison,
                        scores[selection.chosen],
                        scores[other],
                        t,
                        y,
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                    )
                    for other in references
                ),
            ],
            max_workers,
        )
        reports = _run_all(
            [
                partial(
                    evaluate_ranking,
                    label,
                    scores[label],
                    t,
                    y,
                    split=frame.split,
                    final_evaluation=final_evaluation,
                    noise_floor=noise_floor,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                for label in labels
            ],
            max_workers,
        )
        results = [
            TrainedModel(
                label,
                by_label[label].spec.model_name,
                by_label[label].spec.variant,
                run.info.run_id,
                by_label[label].fit_seconds,
                scores[label],
                frame,
                report,
            )
            for label, report in zip(labels, reports, strict=True)
        ]
        chosen = results[0]
        comparisons = [
            comparison_record(chosen.name, other, comparison)
            for other, comparison in zip(references, paired, strict=True)
        ]

        log_evaluation(chosen.report)
        # The model's params carry its seed; log_evaluation logs n_bootstrap.
        mlflow.log_params(by_label[chosen.name].model.params())
        prefix = frame.split
        mlflow.log_metrics(
            {
                "fit_seconds": chosen.fit_seconds,
                "n_fit_clients": n_fit_clients,
                **{f"{r.name}_{prefix}_qini_auc": r.report.metrics.qini_auc for r in results},
                **{
                    f"{r.name}_{prefix}_{uplift_at_k_name(k)}": v
                    for r in results
                    for k, v in r.report.metrics.uplift_at_k.items()
                },
                **{
                    f"qini_diff_{c['model_a']}_vs_{c['model_b']}": c["qini_auc"]["difference"]
                    for c in comparisons
                },
            }
        )
        mlflow.log_dict(selection.as_dict(), "final/selection.json")
        mlflow.log_text(
            pl.DataFrame([leaderboard_row(r) for r in results]).write_csv(), "final/results.csv"
        )
        mlflow.log_dict(comparisons, "final/paired_comparisons.json")
        mlflow.log_text(campaign_table(results).write_csv(), "final/campaign.csv")
        mlflow.log_figure(qini_curve_figure([r.report for r in results]), "final/qini_curves.png")
        return FinalEvaluation(run.info.run_id, results, comparisons)
