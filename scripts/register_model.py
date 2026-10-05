"""Package the final uplift model and register it in Unity Catalog.

Refits the final model -- chosen by the leaderboard rule of
``models.selection``, with its tuned parameters -- on train + val: the same
fit that was evaluated once on test. The test split is never loaded. The
model is logged with its feature contract, signature, and drift reference in
``promolift-models``, reloaded and checked to score bit-identically, and
registered as a new version of ``promolift_ai.models.uplift_targeting``.
The ``champion`` alias, which batch scoring loads, moves only with
``--set-champion``:

    uv run --env-file .env python scripts/register_model.py --set-champion

Against the shared store the working tree must be clean, so every version
traces to a commit. Requires the raw data, the canonical split, and on
Databricks USE CATALOG / USE SCHEMA / CREATE MODEL on ``promolift_ai.models``.
"""

from __future__ import annotations

import argparse
import sys

from promolift.data.split import Split
from promolift.features.store import feature_code_sha256
from promolift.models.dataset import build_model_features, model_frame
from promolift.models.final_evaluation import combine_frames, fit_models
from promolift.models.registry import DEFAULT_SEED
from promolift.models.selection import (
    DEFAULT_LEADERBOARD_RUN_ID,
    leaderboard_spec,
    load_leaderboard,
    select_final_model,
)
from promolift.models.tuning import load_tuned_params
from promolift.serving.registration import (
    DEFAULT_MODEL_NAME,
    ParityError,
    champion_uri,
    check_reload_parity,
    log_packaged_model,
    register,
    registry_uri_for,
    set_champion,
)
from promolift.tracking.lineage import git_state
from promolift.tracking.mlflow_tracking import TrackingConfig, default_tracking_config

# The one-time test evaluation of this exact fit (docs/final_results.md).
FINAL_EVALUATION_RUN_ID = "62d1975014fc4f5d83d55cc80d21f3ac"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--set-champion",
        action="store_true",
        help="point the 'champion' alias at the new version (after the parity check passes)",
    )
    parser.add_argument("--leaderboard-run", default=DEFAULT_LEADERBOARD_RUN_ID)
    parser.add_argument(
        "--leaderboard-tracking-uri",
        help="where the leaderboard run lives, if not the tracking store",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    tracking = default_tracking_config()
    shared = tracking.tracking_uri.startswith("databricks")
    if shared and git_state().is_dirty is not False:
        print("REFUSED -- commit first: a registered version must trace to a clean commit.")
        return 1
    if not shared:
        print("WARNING: registering in the LOCAL store; use `uv run --env-file .env ...` for UC.")

    leaderboard_config = (
        TrackingConfig(args.leaderboard_tracking_uri) if args.leaderboard_tracking_uri else tracking
    )
    leaderboard = load_leaderboard(args.leaderboard_run, leaderboard_config)
    chosen = select_final_model(leaderboard.table, leaderboard.comparisons).chosen
    spec = leaderboard_spec(leaderboard.table, chosen, load_tuned_params())
    print(f"Model: {chosen} ({spec.model_name}, {spec.variant})")

    print("Building features (scans purchases.csv, ~30s)...")
    features = build_model_features()
    frame = combine_frames(
        [model_frame(features, Split.TRAIN), model_frame(features, Split.VAL)], "train+val"
    )
    print(f"Fitting on {frame.split} ({len(frame.outcome):,} clients)...")
    (fitted,) = fit_models([spec], frame, seed=args.seed)

    print("Logging the packaged model...")
    packaged = log_packaged_model(
        fitted.model,
        frame.features,
        label=chosen,
        base_model=spec.model_name,
        tags={
            "fit_split": frame.split,
            "feature_code_sha256": feature_code_sha256(),
            "leaderboard_run_id": leaderboard.run_id,
            "final_evaluation_run_id": FINAL_EVALUATION_RUN_ID,
        },
        config=tracking,
    )
    print(f"  run {packaged.run_id}: {packaged.model_uri}")

    print("Reloading and checking scores are identical...")
    try:
        check_reload_parity(packaged)
    except ParityError as error:
        print(f"REFUSED -- not registered: {error}")
        return 1
    print(f"  {len(packaged.expected_scores):,} scores identical")

    print(f"Registering in {registry_uri_for(tracking.tracking_uri)}...")
    version = register(packaged, args.model_name, tracking)
    print(f"  {args.model_name} version {version}")
    if args.set_champion:
        set_champion(version, args.model_name, tracking)
        print(
            f"  champion -> version {version}; batch scoring loads {champion_uri(args.model_name)}"
        )
    else:
        print("  champion unchanged (pass --set-champion to promote this version)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
