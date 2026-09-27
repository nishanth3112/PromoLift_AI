"""Train the baseline uplift models on the canonical split and log a validation leaderboard.

Each model is fitted on train and evaluated on val (bootstrap CIs, shared
noise floor, plots) in its own MLflow run; a final ``leaderboard`` run holds
the ranked table, overlaid Qini curves, and paired comparisons. The test split
is never touched. Run from a clean commit against the shared tracking server:

    uv run --env-file .env python scripts/train_baselines.py [--models ...] [--n-bootstrap N]

~8-9 minutes at the default 1,000 bootstrap resamples (mostly paired
comparisons); pass e.g. ``--n-bootstrap 200`` for a quick exploratory run.
Requires the raw data and the canonical split (``scripts/fetch_split.py``).
"""

from __future__ import annotations

import argparse
import sys

from promolift.data.split import Split
from promolift.evaluation.uncertainty import DEFAULT_N_BOOTSTRAP
from promolift.models.dataset import build_model_features, model_frame
from promolift.models.registry import DEFAULT_SEED, available_models
from promolift.models.training import log_leaderboard, train_and_evaluate
from promolift.tracking.mlflow_tracking import default_tracking_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models", nargs="+", default=available_models(), choices=available_models()
    )
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    config = default_tracking_config()
    if config.artifact_root is not None:
        print("WARNING: logging to the LOCAL store; use `uv run --env-file .env ...` to share.")

    print("Building features (scans purchases.csv, ~30s)...")
    features = build_model_features()
    train = model_frame(features, Split.TRAIN)
    val = model_frame(features, Split.VAL)
    print(f"train {len(train.features):,} clients, val {len(val.features):,} clients")

    results = train_and_evaluate(
        args.models, train, val, seed=args.seed, n_bootstrap=args.n_bootstrap, config=config
    )
    print("Comparing models (paired bootstrap)...")
    leaderboard_run = log_leaderboard(
        results, n_bootstrap=args.n_bootstrap, seed=args.seed, config=config
    )

    floor = results[0].report.noise_floor
    print(
        f"\nValidation leaderboard (random-ranking band: Qini "
        f"[{floor.qini_auc.lower:+.4f}, {floor.qini_auc.upper:+.4f}], uplift@30% "
        f"[{floor.uplift_at_k[0.3].lower * 100:+.2f}, {floor.uplift_at_k[0.3].upper * 100:+.2f}]pp)"
    )
    for result in sorted(results, key=lambda r: r.report.metrics.qini_auc, reverse=True):
        qini, up30 = result.report.bootstrap.qini_auc, result.report.bootstrap.uplift_at_k[0.3]
        if result.report.beats_noise_floor["qini_auc"]:
            verdict = "beats random"
        elif qini.estimate < floor.qini_auc.lower:
            verdict = "WORSE than random"
        else:
            verdict = "within noise"
        print(
            f"  {result.name:22s} Qini {qini.estimate:+.4f} [{qini.lower:+.4f}, {qini.upper:+.4f}]"
            f"  uplift@30% {up30.estimate * 100:+.2f}pp "
            f"[{up30.lower * 100:+.2f}, {up30.upper * 100:+.2f}]  {verdict}"
        )
    print(f"\nLeaderboard run {leaderboard_run} on {config.tracking_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
