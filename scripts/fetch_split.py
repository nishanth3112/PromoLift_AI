"""Download the canonical train/val/test split from the tracking server and verify it.

The download is checked against the content hash recorded on the canonical
run; if the raw data is present, the split is also re-validated against
uplift_train. Safe to re-run: an already-matching local split is left as-is.

    uv run --env-file .env python scripts/fetch_split.py [--force]
"""

from __future__ import annotations

import argparse
import sys

from promolift.data.loader import Dataset, load_lazy, path_for
from promolift.data.split import load_split, validate_split
from promolift.tracking.mlflow_tracking import default_tracking_config
from promolift.tracking.split_registry import SplitRegistryError, fetch_canonical_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force", action="store_true", help="replace a different local split (invalidates results)"
    )
    args = parser.parse_args()

    config = default_tracking_config()
    if config.artifact_root is not None:
        print(
            "WARNING: fetching from the LOCAL store. For the shared canonical copy, "
            "run with `uv run --env-file .env ...`."
        )
    try:
        result = fetch_canonical_split(config=config, force=args.force)
    except (SplitRegistryError, FileExistsError) as error:
        print(error)
        return 1

    status = "Downloaded" if result.changed else "Already up to date:"
    print(f"{status} {result.path} (run {result.run_id}, split_sha256 {result.split_sha256[:16]})")

    if path_for(Dataset.UPLIFT_TRAIN).exists():
        report = validate_split(load_split(result.path), load_lazy(Dataset.UPLIFT_TRAIN).collect())
        fractions = {name: round(value, 3) for name, value in report.fractions.items()}
        print(
            f"Matches local uplift_train: {report.n_clients:,} clients, "
            f"fractions {fractions}, stratified={report.is_stratified}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
