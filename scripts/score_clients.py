"""Score every client with the champion model and write the send list.

Builds the base features for every client in ``clients.csv``, loads
``promolift_ai.models.uplift_targeting@champion`` (resolved to one version),
scores and ranks every client, and flags the top ``send_share`` of
``configs/business.yaml`` to text. The batch is written to the gitignored
``data/interim/scoring/`` (never overwriting earlier batches) and recorded as
a run in ``promolift-scoring``:

    uv run --env-file .env python scripts/score_clients.py

Takes about a minute: one scan of purchases.csv for the features, then
scoring. Test labels are never read; scoring needs features only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from promolift.models.dataset import build_model_features
from promolift.optimization.business import load_business_config
from promolift.serving.registration import DEFAULT_MODEL_NAME
from promolift.serving.scoring import decile_summary, run_batch_scoring
from promolift.tracking.mlflow_tracking import default_tracking_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, help="default: data/interim/scoring")
    args = parser.parse_args()

    config = load_business_config()
    if config.send_share is None:
        print("REFUSED -- set send_share in configs/business.yaml (from the targeting analysis).")
        return 1
    tracking = default_tracking_config()
    if tracking.artifact_root is not None:
        print(
            "WARNING: using the LOCAL store; use `uv run --env-file .env ...` for the UC champion."
        )

    print("Building features for every client (scans purchases.csv, ~30s)...")
    features = build_model_features()
    print(f"  {features.height:,} clients")

    print(f"Scoring with {args.model_name}@champion...")
    result = run_batch_scoring(
        features,
        send_share=config.send_share,
        model_name=args.model_name,
        output_dir=args.output_dir,
        config=tracking,
    )
    sent = int(result.targets["send"].sum())
    print(
        f"  model version {result.model_version}: {result.targets.height:,} clients scored, "
        f"{sent:,} flagged to text (top {config.send_share:.0%})"
    )
    print("\nBy decile (1 = highest predicted uplift):")
    for row in decile_summary(result.targets).iter_rows(named=True):
        print(
            f"  {row['decile']:2d}: {row['clients']:7,} clients, {row['sent']:7,} sent, "
            f"score {row['min_score']:+.4f} .. {row['max_score']:+.4f} "
            f"(mean {row['mean_score']:+.4f})"
        )
    print(f"\nSend list: {result.path}")
    print(f"Run {result.run_id} on {tracking.tracking_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
