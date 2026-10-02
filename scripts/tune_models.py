"""Tune model hyperparameters by 3-fold cross-validation on the training split.

Optuna for the LightGBM-based models (the first trial is always the defaults),
a small fixed grid for the slow forests. Uplift models are tuned on Qini; the
response model on purchase AUC among treated clients, its own goal. Each
model's search is logged as one run in ``promolift-model-tuning``. Results
stage in gitignored ``data/interim/`` after every model (so a crash loses at
most one model; re-run the missing ones with ``--models``) and are promoted to
``configs/tuned_params.yaml`` at the end, for review and commit.
Validation and test are never touched. Run from a clean commit:

    uv run --env-file .env python scripts/tune_models.py [--models ...] [--n-trials N]

``--feature-set all`` tunes on every feature group (the cached feature table)
instead of the 19 base features, writing to
``configs/tuned_params_all_features.yaml`` so the base params stay untouched:

    caffeinate -is uv run --env-file .env python scripts/tune_models.py --feature-set all \
        --models class_transformation s_learner dr_learner

~45 minutes for all models at the default 40 trials (forests dominate), on an
awake machine: on macOS run it under ``caffeinate -i`` and keep the lid open --
system sleep pauses the run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from promolift.data.loader import project_root
from promolift.data.split import Split
from promolift.features.build import ALL_GROUPS
from promolift.features.store import load_feature_table
from promolift.models.dataset import build_model_features, model_frame
from promolift.models.registry import DEFAULT_SEED
from promolift.models.tuning import (
    ALL_FEATURES_PARAMS_PATH,
    ALL_FEATURES_STAGED_PATH,
    DEFAULT_N_TRIALS,
    FOREST_GRIDS,
    OPTUNA_MODELS,
    STAGED_PARAMS_PATH,
    TUNED_PARAMS_PATH,
    log_tuning,
    promote_tuned_params,
    tune_model,
    write_tuned_params,
)
from promolift.tracking.lineage import git_state, split_sha256
from promolift.tracking.mlflow_tracking import default_tracking_config

_TUNABLE = [*OPTUNA_MODELS, *FOREST_GRIDS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=_TUNABLE, choices=_TUNABLE)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--feature-set", choices=["base", "all"], default="base")
    parser.add_argument("--output", type=Path, help="defaults by --feature-set")
    parser.add_argument("--staging", type=Path, help="defaults by --feature-set")
    args = parser.parse_args()
    all_features = args.feature_set == "all"
    output = args.output or project_root() / (
        ALL_FEATURES_PARAMS_PATH if all_features else TUNED_PARAMS_PATH
    )
    staging = args.staging or project_root() / (
        ALL_FEATURES_STAGED_PATH if all_features else STAGED_PARAMS_PATH
    )

    config = default_tracking_config()
    if config.artifact_root is not None:
        print("WARNING: logging to the LOCAL store; use `uv run --env-file .env ...` to share.")
    git = git_state()
    metadata = {"git_commit": git.commit, "split_sha256": split_sha256() or "missing"}
    if git.is_dirty:
        print("WARNING: uncommitted changes; the tuned params won't be reproducible from a commit.")

    tags: dict[str, str] = {}
    if all_features:
        print("Loading every feature group (first build ~2 min, then cached)...")
        stored = load_feature_table(ALL_GROUPS)
        tags = stored.lineage_tags(ALL_GROUPS)
        metadata |= {
            "feature_groups": tags["feature_groups"],
            "feature_cache_key": stored.cache_key,
        }
        train = model_frame(stored.table.select(ALL_GROUPS), Split.TRAIN)
    else:
        print("Building features (scans purchases.csv, ~30s)...")
        train = model_frame(build_model_features(), Split.TRAIN)
    print(f"train {train.features.shape[0]:,} clients x {train.features.shape[1]} features")

    print(f"\n{'model':22s} {'objective':16s} {'default CV':>18s} {'tuned CV':>18s} {'gain':>9s}")
    for name in args.models:
        result = tune_model(name, train, n_trials=args.n_trials, seed=args.seed)
        log_tuning(result, tags=tags, config=config)
        write_tuned_params([result], staging, metadata=metadata)
        gain = result.best_cv_score - result.default_cv_score
        print(
            f"{name:22s} {result.objective:16s} "
            f"{result.default_cv_score:+.4f} ± {result.default_cv_std:.4f} "
            f"{result.best_cv_score:+.4f} ± {result.best_cv_std:.4f} {gain:+.4f}",
            flush=True,
        )
    promote_tuned_params(staging, output)
    print(f"\nWrote {output}; review and commit it before training with tuned params.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
