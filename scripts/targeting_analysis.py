"""Budget targeting: how deep to text down the chosen model's ranking for the most profit.

The chosen model (the leaderboard rule of ``models.selection``) and the
reference rankings score every train + val client out of fold. Profit per
1,000 clients is computed at every depth from 0% to 100% with the economics
in ``configs/business.yaml``; the chosen depth is the shallowest one whose
profit is statistically tied with the maximum. A sensitivity table repeats
the decision for each break-even uplift (SMS cost / margin) in the config.
Test is never loaded. Logs one run to ``promolift-targeting``:

    uv run --env-file .env python scripts/targeting_analysis.py

Takes a few minutes: one scan of purchases.csv for the average basket, one
for the features, then 5-fold CV of three cheap models and the bootstrap.
"""

from __future__ import annotations

import argparse
import sys

from promolift.data.split import Split
from promolift.evaluation.uncertainty import DEFAULT_N_BOOTSTRAP
from promolift.models.dataset import build_model_features, model_frame
from promolift.models.final_evaluation import combine_frames
from promolift.models.registry import DEFAULT_SEED
from promolift.models.selection import (
    DEFAULT_LEADERBOARD_RUN_ID,
    leaderboard_spec,
    load_leaderboard,
    select_final_model,
)
from promolift.models.tuning import load_tuned_params
from promolift.optimization.analysis import (
    DEFAULT_N_FOLDS,
    out_of_fold_rankings,
    run_targeting_analysis,
)
from promolift.optimization.business import average_transaction_value, load_business_config
from promolift.tracking.mlflow_tracking import TrackingConfig, default_tracking_config

REFERENCE_MODELS = ("response", "random")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leaderboard-run", default=DEFAULT_LEADERBOARD_RUN_ID)
    parser.add_argument(
        "--leaderboard-tracking-uri",
        help="where the leaderboard run lives, if not the tracking store",
    )
    parser.add_argument("--references", nargs="+", default=list(REFERENCE_MODELS))
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    config = load_business_config()
    tracking = default_tracking_config()
    if tracking.artifact_root is not None:
        print("WARNING: logging to the LOCAL store; use `uv run --env-file .env ...` to share.")
    leaderboard_config = (
        TrackingConfig(args.leaderboard_tracking_uri) if args.leaderboard_tracking_uri else tracking
    )
    leaderboard = load_leaderboard(args.leaderboard_run, leaderboard_config)
    chosen = select_final_model(leaderboard.table, leaderboard.comparisons).chosen
    labels = list(dict.fromkeys([chosen, *args.references]))
    tuned_params = load_tuned_params()
    specs = [leaderboard_spec(leaderboard.table, label, tuned_params) for label in labels]
    print(f"Chosen model: {chosen}; references: {', '.join(labels[1:])}")

    print("Measuring the average basket (scans purchases.csv)...")
    average_transaction = average_transaction_value()
    margin = config.margin(average_transaction)
    print(
        f"  average transaction {average_transaction:,.2f} {config.currency}; margin per extra "
        f"purchase {margin:,.2f}; SMS {config.sms_cost:,.2f}; break-even uplift "
        f"{config.sms_cost / margin:.2%}"
    )

    print("Building features (scans purchases.csv, ~30s)...")
    features = build_model_features()
    frame = combine_frames(
        [model_frame(features, Split.TRAIN), model_frame(features, Split.VAL)], "train+val"
    )
    print(f"Scoring {len(frame.outcome):,} clients out of fold ({args.n_folds} folds)...")
    rankings = out_of_fold_rankings(specs, frame, n_folds=args.n_folds, seed=args.seed)

    print("Computing profit curves (bootstrap)...")
    analysis = run_targeting_analysis(
        rankings,
        frame,
        chosen=chosen,
        config=config,
        average_transaction=average_transaction,
        leaderboard_run_id=leaderboard.run_id,
        n_folds=args.n_folds,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        tracking=tracking,
    )

    choice = analysis.choice
    print(
        f"\nDecision ({config.currency}, per 1,000 clients): text the top "
        f"{choice.chosen_depth:.0%} -> profit {choice.profit_per_1000:,.0f} "
        f"[{choice.profit_lower:,.0f}, {choice.profit_upper:,.0f}]"
    )
    best = analysis.curves[chosen].profit_per_1000.max()
    print(
        f"  max-profit depth {choice.best_depth:.0%} ({best:,.0f}); "
        f"budget cap {choice.max_depth:.0%}"
    )
    print(f"  random targeting (all or nothing): {analysis.random_targeting_profit:,.0f}")
    for label in labels[1:]:
        print(f"  {label} at its best depth: {analysis.curves[label].profit_per_1000.max():,.0f}")

    print("\nSensitivity (break-even uplift = SMS cost / margin):")
    for row in analysis.sensitivity.iter_rows(named=True):
        print(
            f"  {row['break_even_uplift']:6.1%}: text top {row['chosen_depth']:4.0%} "
            f"(max at {row['best_depth']:4.0%}), profit {row['profit_per_1000']:8,.0f} "
            f"[{row['profit_ci_lower']:8,.0f}, {row['profit_ci_upper']:8,.0f}]; "
            f"random {row['random_targeting_profit_per_1000']:8,.0f}"
        )
    print(f"\nRun {analysis.run_id} on {tracking.tracking_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
