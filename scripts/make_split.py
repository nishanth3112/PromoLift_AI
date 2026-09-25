"""Generate, validate, and publish the canonical train/val/test split.

Stratified 60/20/20 on treatment_flg x target over uplift_train. Refuses to
write or publish a split that isn't stratified, or in which randomization
doesn't hold within every split. Run once, against the shared tracking server,
from a clean commit:

    uv run --env-file .env python scripts/make_split.py [--force]

Requires the raw files under ``data/raw/``; not run in CI.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict

from promolift.data.loader import Dataset, load_lazy
from promolift.data.split import (
    DEFAULT_SEED,
    EXPECTED_FRACTIONS,
    generate_split,
    split_assignment_path,
    validate_split,
    write_split,
)
from promolift.features.demographics import build_demographic_features
from promolift.tracking.mlflow_tracking import default_tracking_config
from promolift.tracking.split_registry import publish_split
from promolift.validation.experiment_validity import split_randomization_check
from promolift.validation.referential_integrity import transaction_date_integrity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--force", action="store_true", help="replace an existing split (invalidates results)"
    )
    args = parser.parse_args()

    path = split_assignment_path()
    if path.exists() and not args.force:
        print(f"{path} already exists; pass --force to replace the canonical split.")
        return 1

    labels = load_lazy(Dataset.UPLIFT_TRAIN).collect()
    assignment = generate_split(labels, seed=args.seed)
    validation = validate_split(assignment, labels)

    # Same covariate construction as scripts/log_data_validation.py: cleaned
    # (implausible-nulled) age, so raw outliers like -7491 don't mask imbalance.
    reference_date = transaction_date_integrity().max_date
    demographics = build_demographic_features(reference_date).select("client_id", "age", "gender")
    frame = assignment.join(labels, on="client_id").join(
        demographics.collect(), on="client_id", how="left"
    )
    randomization = split_randomization_check(frame)

    problems = []
    if not validation.matches_expected_fractions:
        problems.append(f"fractions {validation.fractions} deviate from 60/20/20")
    if not validation.is_stratified:
        problems.append(f"max stratum deviation {validation.max_stratum_deviation:.6f}")
    problems += [f"covariate imbalance in {r.split}" for r in randomization if not r.is_balanced]
    if problems:
        print("Split rejected, nothing written or published:\n  " + "\n  ".join(problems))
        return 1

    write_split(assignment, path, force=args.force)

    metrics = {
        "max_fraction_deviation": validation.max_fraction_deviation,
        "max_stratum_deviation": validation.max_stratum_deviation,
    }
    for r in randomization:
        metrics[f"{r.split}_n_clients"] = r.ate.treatment_count + r.ate.control_count
        metrics[f"{r.split}_ate"] = r.ate.ate
        metrics[f"{r.split}_ate_ci_half_width"] = (r.ate.ci_upper - r.ate.ci_lower) / 2
    config = default_tracking_config()
    run_id = publish_split(
        {"validation": asdict(validation), "randomization": [asdict(r) for r in randomization]},
        params={
            "seed": args.seed,
            "fractions": "/".join(f"{f:g}" for f in EXPECTED_FRACTIONS.values()),
            "stratify_on": "treatment_flg x target",
        },
        metrics=metrics,
        config=config,
    )

    print(f"Wrote {path}")
    for r in randomization:
        print(
            f"  {r.split:5s} n={metrics[f'{r.split}_n_clients']:>7,}  "
            f"ATE {r.ate.ate * 100:+.2f}pp "
            f"(+/-{metrics[f'{r.split}_ate_ci_half_width'] * 100:.2f}pp)"
        )
    print(f"Published canonical split: run {run_id} on {config.tracking_uri}")
    if config.artifact_root is not None:
        print(
            "WARNING: published to the LOCAL store only. For the shared canonical copy, "
            "re-run with `uv run --env-file .env ... --force`."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
