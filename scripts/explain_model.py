"""Explain the champion uplift model: which features drive a client's predicted uplift.

Loads ``promolift_ai.models.uplift_targeting@champion`` (resolved to one
version), takes a random sample of train + val clients, and splits each
client's predicted uplift among the features with exact TreeSHAP. Reports
global importance with direction, importance within predicted-uplift tiers,
gender and age bands, and how the score moves along the leading features.
One run is logged to ``promolift-explainability``:

    uv run --env-file .env python scripts/explain_model.py

The test split is never loaded, and no client-level contributions are logged.
Takes about a minute: one scan of purchases.csv for the features, then SHAP
on the sample. ``--no-log`` prints without recording a run.
"""

from __future__ import annotations

import argparse
import sys

from promolift.data.split import Split
from promolift.explainability.report import (
    FIT_POPULATION,
    build_report,
    log_explanation,
    summary,
)
from promolift.explainability.shap_values import (
    DEFAULT_N_ROWS,
    DEFAULT_SEED,
    ExplanationError,
    explain_registered,
)
from promolift.models.dataset import build_model_features, model_frame
from promolift.models.final_evaluation import combine_frames
from promolift.serving.registration import DEFAULT_MODEL_NAME
from promolift.serving.scoring import load_champion
from promolift.tracking.mlflow_tracking import default_tracking_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--n-rows", type=int, default=DEFAULT_N_ROWS, help="clients to explain")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-log", action="store_true", help="print only; record no run")
    args = parser.parse_args()

    tracking = default_tracking_config()
    if tracking.artifact_root is not None:
        print(
            "WARNING: using the LOCAL store; use `uv run --env-file .env ...` for the UC champion."
        )

    print(f"Loading {args.model_name}@champion...")
    champion = load_champion(args.model_name, tracking)
    print(f"  version {champion.version}")

    print("Building features (scans purchases.csv, ~30s)...")
    features = build_model_features()
    frame = combine_frames(
        [model_frame(features, Split.TRAIN), model_frame(features, Split.VAL)], FIT_POPULATION
    )

    available = len(frame.features)
    print(f"Explaining {min(args.n_rows, available):,} of {available:,} clients...")
    try:
        explanation = explain_registered(
            champion, frame.features, n_rows=args.n_rows, seed=args.seed
        )
    except ExplanationError as error:
        print(f"REFUSED -- {error}")
        return 1
    report = build_report(explanation)
    text = summary(explanation, report, model_name=args.model_name, model_version=champion.version)
    print(f"\n{text}")

    if args.no_log:
        print("\n--no-log: nothing was recorded.")
        return 0
    run_id = log_explanation(
        explanation,
        report,
        model_name=args.model_name,
        model_version=champion.version,
        seed=args.seed,
        config=tracking,
    )
    print(f"\nRun {run_id} on {tracking.tracking_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
