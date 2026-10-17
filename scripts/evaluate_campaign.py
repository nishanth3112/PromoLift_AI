"""Evaluate a finished campaign: realized uplift against what the model predicted.

The scoring batch texted its top clients and held a random share back as a
control group (``holdout_share`` in ``configs/business.yaml``). Given who
bought, this compares the two groups' purchase rates -- the campaign's real
effect on the clients the model chose -- with the predicted uplift, overall
and by decile, and checks that the ranking still holds (Qini AUC, uplift at
k). One run is logged to ``promolift-campaigns``.

The results file (CSV or parquet) has one row per targeted client with a
result: ``client_id, treated, purchased, scoring_run_id``. The send list is
found in ``data/interim/scoring/`` from the file's scoring run, or passed with
``--batch`` (export the batch from the Delta table if it was scored by the job):

    uv run --env-file .env python scripts/evaluate_campaign.py --results results.csv

``--demo`` shows the evaluation on a SIMULATED campaign over the latest local
send list (``--effect-scale``: 1 = the model's promise is real, 0 = the texts
do nothing). It prints only and never writes a run, so simulated numbers can't
be mistaken for results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from promolift.data.loader import project_root
from promolift.evaluation.uncertainty import DEFAULT_CONFIDENCE, DEFAULT_N_BOOTSTRAP, DEFAULT_SEED
from promolift.monitoring.campaign import DEFAULT_RANKING_BOOTSTRAP, evaluate_campaign
from promolift.monitoring.campaign_data import (
    CampaignResultsError,
    find_send_list,
    load_results,
    match_results,
    validate_results,
)
from promolift.monitoring.campaign_report import log_campaign_evaluation, summary
from promolift.monitoring.campaign_simulation import simulate_campaign
from promolift.serving.scoring import SCORING_OUTPUT_DIR
from promolift.tracking.mlflow_tracking import default_tracking_config


def _print_deciles(deciles: pl.DataFrame) -> None:
    print("\nBy decile (1 = highest predicted uplift), realized vs predicted uplift:")
    for row in deciles.iter_rows(named=True):
        if row["realized_uplift"] is None:
            print(f"  {row['decile']:2d}: no clients in both arms")
            continue
        print(
            f"  {row['decile']:2d}: {row['n_treated']:7,} texted / "
            f"{row['n_control']:6,} held back, "
            f"realized {row['realized_uplift'] * 100:+6.2f}pp "
            f"({row['realized_lower'] * 100:+6.2f} to {row['realized_upper'] * 100:+6.2f}), "
            f"predicted {row['predicted_uplift'] * 100:+6.2f}pp"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results", type=Path, help="campaign results (.csv or .parquet)")
    source.add_argument("--demo", action="store_true", help="evaluate a simulated campaign")
    parser.add_argument("--batch", type=Path, help="send list parquet (default: found by run id)")
    parser.add_argument("--effect-scale", type=float, default=1.0, help="demo only")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--ranking-bootstrap", type=int, default=DEFAULT_RANKING_BOOTSTRAP)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    scoring_dir = project_root() / SCORING_OUTPUT_DIR
    try:
        if args.demo:
            batch_path = args.batch or find_send_list(scoring_dir)
            batch = pl.read_parquet(batch_path)
            results = validate_results(simulate_campaign(batch, effect_scale=args.effect_scale))
            print(f"SIMULATED campaign (effect scale {args.effect_scale}) on {batch_path.name}")
        else:
            results = load_results(args.results)
            batch_path = args.batch or find_send_list(scoring_dir, results["scoring_run_id"][0])
            batch = pl.read_parquet(batch_path)
        data = match_results(batch, results)
    except (CampaignResultsError, FileNotFoundError) as error:
        print(f"REFUSED -- {error}")
        return 1

    evaluation = evaluate_campaign(
        data,
        n_bootstrap=args.n_bootstrap,
        ranking_bootstrap=args.ranking_bootstrap,
        confidence=args.confidence,
        seed=args.seed,
    )
    print(f"\n{summary(evaluation)}")
    _print_deciles(evaluation.deciles)
    if args.demo:
        print("\nDemo only: nothing was logged.")
        return 0

    tracking = default_tracking_config()
    run_id = log_campaign_evaluation(evaluation, config=tracking, results_name=args.results.name)
    print(f"\nRun {run_id} on {tracking.tracking_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
