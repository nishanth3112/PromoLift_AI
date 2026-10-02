"""Choose the feature set by cross-validated feature-group ablation inside train.

Fits class_transformation, s_learner, and dr_learner (Phase 10 tuned
hyperparameters, held fixed) with 3-fold CV inside the training split on
base, base + each new feature group, and all groups, then applies the
decision rule: keep a group if its paired out-of-fold Qini gain over base has
a 95% CI above 0, and prefer the kept set unless all groups are significantly
better. Val and test are never touched. Logs one ``cv-<config>`` run per
config and an ``ablation`` summary run (gains, decision) to
``promolift-feature-ablation``. Run from a clean commit, on an awake machine:

    caffeinate -is uv run --env-file .env python scripts/run_ablation.py

The first run builds and caches the full feature table (~2 min). Pass e.g.
``--n-bootstrap 200`` for a quick look. Requires the raw data and the
canonical split (``scripts/fetch_split.py``).
"""

from __future__ import annotations

import argparse
import sys

from promolift.data.split import Split
from promolift.evaluation.uncertainty import DEFAULT_N_BOOTSTRAP
from promolift.features.build import ALL_GROUPS
from promolift.features.store import load_feature_table
from promolift.models.ablation import ABLATION_MODELS, log_ablation, run_ablation
from promolift.models.dataset import model_frame
from promolift.models.registry import DEFAULT_SEED
from promolift.models.tuning import load_tuned_params
from promolift.tracking.mlflow_tracking import default_tracking_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(ABLATION_MODELS))
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--rebuild-features", action="store_true", help="rebuild the cached feature table"
    )
    args = parser.parse_args()

    config = default_tracking_config()
    if config.artifact_root is not None:
        print("WARNING: logging to the LOCAL store; use `uv run --env-file .env ...` to share.")

    print("Loading the feature table (first build scans purchases.csv, ~2 min)...")
    stored = load_feature_table(ALL_GROUPS, rebuild=args.rebuild_features)
    source = "cache" if stored.from_cache else "fresh build"
    print(f"  {stored.table.frame.shape} from {source}, key {stored.cache_key}")

    tuned = load_tuned_params()
    overrides = {name: tuned[name] for name in args.models if name in tuned}
    print(f"Cross-validating {len(args.models)} models on train (tuned: {sorted(overrides)})...")
    result = run_ablation(
        lambda groups: model_frame(stored.table.select(groups), Split.TRAIN),
        models=args.models,
        overrides=overrides,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        progress=print,
    )
    summary_run = log_ablation(result, feature_tags=stored.lineage_tags(ALL_GROUPS), config=config)

    print("\nPaired out-of-fold Qini gain (mean across models, 95% CI):")
    for gain in result.reported_gains():
        verdict = "better" if gain.lower > 0 else "worse" if gain.upper < 0 else "tied"
        print(
            f"  {gain.config:40s} vs {gain.baseline:22s} {gain.difference:+.4f} "
            f"[{gain.lower:+.4f}, {gain.upper:+.4f}]  win {gain.win_rate:.2f}  {verdict}"
        )
    kept = ", ".join(g.value for g in result.kept_groups) or "none"
    print(f"\nKept groups: {kept}")
    print(f"Chosen feature set: {result.chosen.name} ({result.reason})")
    print(f"Ablation run {summary_run} on {config.tracking_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
