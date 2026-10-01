"""Compare all feature groups vs base features, each with hyperparameters tuned for it.

The feature ablation held the base-tuned params fixed, which could handicap
the 66-column table. This is the fairness check: base features with
``configs/tuned_params.yaml`` vs every group with
``configs/tuned_params_all_features.yaml`` (``tune_models.py --feature-set
all``), same 3 folds inside train and paired out-of-fold bootstrap as the
ablation. Val and test are never touched. Logs a ``tuned_comparison`` run to
``promolift-feature-ablation``. Run from a clean commit, on an awake machine:

    caffeinate -is uv run --env-file .env python scripts/compare_tuned_features.py
"""

from __future__ import annotations

import argparse
import sys

from promolift.data.loader import project_root
from promolift.data.split import Split
from promolift.evaluation.uncertainty import DEFAULT_N_BOOTSTRAP
from promolift.features.build import ALL_GROUPS, BASE_GROUPS
from promolift.features.store import load_feature_table
from promolift.models.ablation import (
    ABLATION_MODELS,
    Arm,
    compare_arms,
    log_arm_comparison,
)
from promolift.models.dataset import model_frame
from promolift.models.registry import DEFAULT_SEED
from promolift.models.tuning import ALL_FEATURES_PARAMS_PATH, load_tuned_params
from promolift.tracking.mlflow_tracking import default_tracking_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(ABLATION_MODELS))
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    base_params = load_tuned_params()
    all_params = load_tuned_params(project_root() / ALL_FEATURES_PARAMS_PATH)
    missing = [m for m in args.models if m not in base_params or m not in all_params]
    if missing:
        print(f"ERROR: no tuned params for {missing} in both files; tune them first.")
        return 1

    config = default_tracking_config()
    if config.artifact_root is not None:
        print("WARNING: logging to the LOCAL store; use `uv run --env-file .env ...` to share.")

    stored = load_feature_table(ALL_GROUPS)
    print(f"Feature table {stored.table.frame.shape}, key {stored.cache_key}")
    baseline = Arm("base_tuned", BASE_GROUPS, {m: base_params[m] for m in args.models})
    challenger = Arm("all_tuned", ALL_GROUPS, {m: all_params[m] for m in args.models})

    result = compare_arms(
        lambda groups: model_frame(stored.table.select(groups), Split.TRAIN),
        challenger,
        baseline,
        models=args.models,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        progress=print,
    )
    run_id = log_arm_comparison(result, feature_tags=stored.lineage_tags(ALL_GROUPS), config=config)

    gain = result.gain
    print(
        f"\nall_tuned vs base_tuned, paired out-of-fold Qini (mean across models, 95% CI): "
        f"{gain.difference:+.4f} [{gain.lower:+.4f}, {gain.upper:+.4f}]  "
        f"win {gain.win_rate:.2f}  -> {result.verdict}"
    )
    for model, diff in gain.per_model.items():
        print(f"  {model:22s} {diff:+.4f}")
    print(f"Comparison run {run_id} on {config.tracking_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
