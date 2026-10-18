"""Refit the champion without age and compare: what does age add, and who gets texted?

SHAP (``scripts/explain_model.py``) shows the champion leans on ``age`` more than
on any other feature; it cannot say how much worse the model would be without
it. This fits the chosen model (the leaderboard rule of ``models.selection``,
with its tuned parameters held fixed) twice on the same train + val folds --
with every feature, and without ``age`` and ``age_is_valid`` -- scores every
client out of fold, and compares the two rankings on the same bootstrap
resamples: Qini AUC, uplift at the deployed targeting depth, the age mix of the
texted clients under each, and how strongly each score still follows age.
One run is logged to ``promolift-feature-ablation``:

    uv run --env-file .env python scripts/age_ablation.py

The test split is never loaded. Removing ``age`` does not remove age
information (tenure, spend, and points correlate with it); the score-age
correlation says how much is left. Takes several minutes: one scan of
purchases.csv, ten model fits, and the paired bootstrap. ``--no-log`` prints
without recording a run.
"""

from __future__ import annotations

import argparse
import sys

from promolift.data.split import Split
from promolift.evaluation.uncertainty import DEFAULT_N_BOOTSTRAP
from promolift.explainability.feature_removal import (
    DEFAULT_DROPPED,
    log_feature_removal,
    run_feature_removal,
    summary,
)
from promolift.models.dataset import build_model_features, model_frame
from promolift.models.final_evaluation import combine_frames
from promolift.models.registry import DEFAULT_SEED
from promolift.models.selection import (
    DEFAULT_LEADERBOARD_RUN_ID,
    leaderboard_spec,
    load_leaderboard,
    select_final_model,
)
from promolift.models.tuning import DEFAULT_N_FOLDS, load_tuned_params
from promolift.optimization.business import load_business_config
from promolift.tracking.mlflow_tracking import TrackingConfig, default_tracking_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leaderboard-run", default=DEFAULT_LEADERBOARD_RUN_ID)
    parser.add_argument(
        "--leaderboard-tracking-uri",
        help="where the leaderboard run lives, if not the tracking store",
    )
    parser.add_argument("--drop", nargs="+", default=list(DEFAULT_DROPPED), help="features to drop")
    parser.add_argument("--depth", type=float, help="targeting depth; default: send_share")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-log", action="store_true", help="print only; record no run")
    args = parser.parse_args()

    business = load_business_config()
    depth = args.depth if args.depth is not None else business.send_share
    if depth is None:
        print("REFUSED -- pass --depth or set send_share in configs/business.yaml.")
        return 1
    tracking = default_tracking_config()
    if tracking.artifact_root is not None:
        print(
            "WARNING: using the LOCAL store; use `uv run --env-file .env ...` for the leaderboard."
        )

    leaderboard_config = (
        TrackingConfig(args.leaderboard_tracking_uri) if args.leaderboard_tracking_uri else tracking
    )
    leaderboard = load_leaderboard(args.leaderboard_run, leaderboard_config)
    chosen = select_final_model(leaderboard.table, leaderboard.comparisons).chosen
    spec = leaderboard_spec(leaderboard.table, chosen, load_tuned_params())
    print(f"Model: {chosen}; dropping {', '.join(args.drop)}; depth {depth:.0%}")

    print("Building features (scans purchases.csv, ~30s)...")
    features = build_model_features()
    frame = combine_frames(
        [model_frame(features, Split.TRAIN), model_frame(features, Split.VAL)], "train+val"
    )
    try:
        result = run_feature_removal(
            frame,
            spec,
            depth=depth,
            dropped=args.drop,
            n_folds=args.n_folds,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            progress=print,
        )
    except ValueError as error:
        print(f"REFUSED -- {error}")
        return 1
    print(f"\n{summary(result)}")

    if args.no_log:
        print("\n--no-log: nothing was recorded.")
        return 0
    run_id = log_feature_removal(result, config=tracking)
    print(f"\nRun {run_id} on {tracking.tracking_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
