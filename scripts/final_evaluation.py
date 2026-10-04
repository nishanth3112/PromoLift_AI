"""Final evaluation: choose the model by the leaderboard rule, refit it, score test once.

The final model is chosen from the validation leaderboard by a rule fixed
before test is seen (``models.selection``: the cheapest uplift variant tied
with the best). It and the reference models are refitted on train + val with
the leaderboard's parameters and evaluated once on test, in one run in
``promolift-final-evaluation``. The references are context only; the choice
never changes after test is seen.

The official run refuses to start from a dirty tree, a local store, or data
and a split that differ from the leaderboard's, and refuses outright if a
final evaluation already exists for this split -- there is no override:

    uv run --env-file .env python scripts/final_evaluation.py

``--dry-run`` takes the same path on train -> val (test is never loaded, no
guard) and prints the leaderboard's val Qini beside each model, which a
correct pipeline reproduces exactly. It is a wiring check, so it defaults to
20 bootstrap resamples and 20 random rankings (point estimates don't depend
on them; only the CIs get coarser). Send it to a scratch store:

    $env:MLFLOW_TRACKING_URI = "sqlite:///<scratch>/mlflow.db"
    uv run python scripts/final_evaluation.py --dry-run --leaderboard-tracking-uri databricks://promolift

Requires the raw data and the canonical split (``scripts/fetch_split.py``).
"""

from __future__ import annotations

import argparse
import sys

from promolift.data.split import Split
from promolift.evaluation.uncertainty import DEFAULT_N_BOOTSTRAP, DEFAULT_N_RANKINGS
from promolift.models.dataset import build_model_features, model_frame
from promolift.models.final_evaluation import (
    FinalEvaluationExistsError,
    assert_no_final_run,
    combine_frames,
    evaluate_final,
    fit_models,
    preflight_problems,
)
from promolift.models.registry import DEFAULT_SEED
from promolift.models.selection import (
    leaderboard_spec,
    load_leaderboard,
    select_final_model,
)
from promolift.models.tuning import load_tuned_params
from promolift.tracking.lineage import (
    fingerprint_digest,
    git_state,
    raw_data_fingerprint,
    split_sha256,
)
from promolift.tracking.mlflow_tracking import TrackingConfig, default_tracking_config

# The Phase 10 validation leaderboard (docs/advanced_models_results.md).
LEADERBOARD_RUN_ID = "6fc6717a9b6a46d5b112c83120f2694d"
REFERENCE_MODELS = ("s_learner_tuned", "causal_forest", "response", "random")
DRY_RUN_RESAMPLES = 20


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leaderboard-run", default=LEADERBOARD_RUN_ID)
    parser.add_argument(
        "--leaderboard-tracking-uri",
        help="where the leaderboard run lives, if not the tracking store (for dry runs)",
    )
    parser.add_argument("--references", nargs="+", default=list(REFERENCE_MODELS))
    parser.add_argument(
        "--dry-run", action="store_true", help="fit on train, evaluate on val; never loads test"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        help=f"default {DEFAULT_N_BOOTSTRAP}, or {DRY_RUN_RESAMPLES} with --dry-run",
    )
    parser.add_argument(
        "--n-rankings",
        type=int,
        help=f"noise-floor rankings; default {DEFAULT_N_RANKINGS}, or {DRY_RUN_RESAMPLES} dry",
    )
    parser.add_argument(
        "--workers", type=int, help="processes for the evaluations (default: every core)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.n_bootstrap is None:
        args.n_bootstrap = DRY_RUN_RESAMPLES if args.dry_run else DEFAULT_N_BOOTSTRAP
    if args.n_rankings is None:
        args.n_rankings = DRY_RUN_RESAMPLES if args.dry_run else DEFAULT_N_RANKINGS

    config = default_tracking_config()
    leaderboard_config = (
        TrackingConfig(args.leaderboard_tracking_uri) if args.leaderboard_tracking_uri else config
    )
    leaderboard = load_leaderboard(args.leaderboard_run, leaderboard_config)
    selection = select_final_model(leaderboard.table, leaderboard.comparisons)
    print(
        f"Leaderboard run {leaderboard.run_id}: {len(selection.tied)} variants tied with the best"
    )
    print(f"  tied (cheapest first): {', '.join(selection.tied)}")
    print(f"  chosen: {selection.chosen}  (rule: {selection.rule})")

    if not args.dry_run:
        split_hash = split_sha256()
        problems = preflight_problems(
            git=git_state(),
            split_hash=split_hash,
            raw_data_digest=fingerprint_digest(raw_data_fingerprint()),
            leaderboard_tags=leaderboard.tags,
            config=config,
        )
        if problems:
            print("\nREFUSED -- the official run can't start:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        try:
            assert_no_final_run(split_hash, config)
        except FinalEvaluationExistsError as error:
            print(f"\nREFUSED -- {error}")
            return 1
    elif config.tracking_uri.startswith("databricks"):
        print("WARNING: dry run logging to the SHARED store; point MLFLOW_TRACKING_URI at scratch.")

    labels = list(dict.fromkeys([selection.chosen, *args.references]))
    tuned_params = load_tuned_params()
    specs = [leaderboard_spec(leaderboard.table, label, tuned_params) for label in labels]

    print("\nBuilding features (scans purchases.csv, ~30s)...")
    features = build_model_features()
    train = model_frame(features, Split.TRAIN)
    if args.dry_run:
        fit_frame, eval_split = train, Split.VAL
    else:
        fit_frame = combine_frames([train, model_frame(features, Split.VAL)], "train+val")
        eval_split = Split.TEST
    print(
        f"Fitting {len(specs)} models on {fit_frame.split} ({len(fit_frame.features):,} clients)..."
    )
    fitted = fit_models(specs, fit_frame, seed=args.seed)

    print(f"Evaluating on {eval_split.value}...")
    final = evaluate_final(
        fitted,
        lambda: model_frame(features, eval_split, final_evaluation=not args.dry_run),
        selection=selection,
        leaderboard_run_id=leaderboard.run_id,
        fit_split=fit_frame.split,
        n_fit_clients=len(fit_frame.features),
        final_evaluation=not args.dry_run,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        n_rankings=args.n_rankings,
        max_workers=args.workers,
        config=config,
    )

    floor = final.results[0].report.noise_floor
    expected = dict(zip(leaderboard.table["model"], leaderboard.table["qini_auc"], strict=True))
    print(
        f"\n{eval_split.value} results (random-ranking band: Qini "
        f"[{floor.qini_auc.lower:+.4f}, {floor.qini_auc.upper:+.4f}])"
    )
    for result in final.results:
        qini, up30 = result.report.bootstrap.qini_auc, result.report.bootstrap.uplift_at_k[0.3]
        line = (
            f"  {result.name:26s} Qini {qini.estimate:+.4f} [{qini.lower:+.4f}, {qini.upper:+.4f}]"
            f"  uplift@30% {up30.estimate * 100:+.2f}pp"
        )
        if args.dry_run:
            line += f"  (leaderboard val Qini {expected[result.name]:+.4f})"
        print(line)
    for record in final.comparisons:
        diff = record["qini_auc"]
        print(
            f"  {record['model_a']} - {record['model_b']}: Qini {diff['difference']:+.4f} "
            f"[{diff['lower']:+.4f}, {diff['upper']:+.4f}], P(better) {diff['win_rate']:.3f}"
        )
    print(f"\nRun {final.run_id} on {config.tracking_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
