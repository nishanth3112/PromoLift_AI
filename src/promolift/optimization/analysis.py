"""Profit analysis of the chosen model's ranking, on out-of-fold scores, logged to MLflow.

Profit curves need scores for clients the model never saw. Test is spent
(Phase 12), and val was used to choose the model, so every client of train +
val is scored by cross-validation instead: each fold's model is fitted on the
other folds, scores are ranked within the fold, and the ranks are pooled
(``models.ablation.out_of_fold_scores``). The reference rankings (response
model, random) are scored the same way, on the same folds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

from promolift.evaluation.plots import profit_curve_figure
from promolift.evaluation.uncertainty import DEFAULT_N_BOOTSTRAP, DEFAULT_SEED
from promolift.models.ablation import out_of_fold_scores
from promolift.models.dataset import ModelFrame
from promolift.models.training import ModelSpec
from promolift.models.tuning import cv_folds
from promolift.optimization.business import BusinessConfig
from promolift.optimization.targeting import (
    DepthChoice,
    Economics,
    ProfitCurve,
    UpliftByDepth,
    budget_depth_cap,
    choose_depth,
    profit_curve,
    random_targeting_profit,
    uplift_by_depth,
)
from promolift.tracking.mlflow_tracking import Experiment, TrackingConfig, start_run

DEFAULT_N_FOLDS = 5
DEPTH_RULE = "shallowest_tied_with_max_profit"


def out_of_fold_rankings(
    specs: Sequence[ModelSpec], frame: ModelFrame, *, n_folds: int, seed: int
) -> dict[str, np.ndarray]:
    """Label -> out-of-fold within-fold ranks, every spec on the same stratified folds."""
    folds = cv_folds(frame.treatment, frame.outcome, n_folds=n_folds, seed=seed)
    return {
        spec.label: out_of_fold_scores(
            spec.model_name, frame, folds, config="base", overrides=spec.overrides, seed=seed
        ).oof_scores
        for spec in specs
    }


def _at(curve: ProfitCurve, depth: float) -> int:
    return int(np.flatnonzero(np.isclose(curve.depths, depth))[0])


def depth_cap(config: BusinessConfig, sms_cost: float) -> float:
    """The budget's depth cap at ``sms_cost``, or 1.0 without a budget."""
    if config.budget is None or config.campaign_clients is None:
        return 1.0
    return budget_depth_cap(config.budget, sms_cost, config.campaign_clients)


def sensitivity_table(
    by_depth: Mapping[str, UpliftByDepth],
    chosen: str,
    margin: float,
    config: BusinessConfig,
) -> pl.DataFrame:
    """The chosen depth and its profit for each break-even uplift in the config.

    Each scenario keeps the margin and sets the SMS cost to break-even x
    margin. References report their best point-estimate profit at any depth
    (their most favourable case); random targeting reports the better of
    texting everyone or no one.
    """
    rows = []
    for break_even in config.break_even_grid:
        economics = Economics(break_even * margin, margin)
        curve = profit_curve(by_depth[chosen], economics)
        choice = choose_depth(curve, max_depth=depth_cap(config, economics.sms_cost))
        i = _at(curve, choice.chosen_depth)
        row = {
            "break_even_uplift": break_even,
            "sms_cost": economics.sms_cost,
            "chosen_depth": choice.chosen_depth,
            "best_depth": choice.best_depth,
            "profit_per_1000": choice.profit_per_1000,
            "profit_ci_lower": choice.profit_lower,
            "profit_ci_upper": choice.profit_upper,
            "sms_per_1000": float(curve.sms_per_1000[i]),
            "extra_purchases_per_1000": float(curve.extra_purchases_per_1000[i]),
            "random_targeting_profit_per_1000": random_targeting_profit(
                by_depth[chosen].ate, economics
            ),
        }
        for label in by_depth:
            if label != chosen:
                reference = profit_curve(by_depth[label], economics)
                row[f"{label}_best_profit_per_1000"] = float(reference.profit_per_1000.max())
        rows.append(row)
    return pl.DataFrame(rows)


def curve_table(curves: Mapping[str, ProfitCurve]) -> pl.DataFrame:
    """Every model's profit curve in long form: one row per model and depth."""
    return pl.concat(
        pl.DataFrame(
            {
                "model": label,
                "depth": curve.depths,
                "sms_per_1000": curve.sms_per_1000,
                "extra_purchases_per_1000": curve.extra_purchases_per_1000,
                "profit_per_1000": curve.profit_per_1000,
                "profit_ci_lower": curve.profit_lower,
                "profit_ci_upper": curve.profit_upper,
                "roi": curve.roi,
            }
        )
        for label, curve in curves.items()
    )


@dataclass
class TargetingAnalysis:
    """The configured scenario's decision, every curve, and the sensitivity table."""

    run_id: str
    economics: Economics
    choice: DepthChoice
    curves: dict[str, ProfitCurve]
    sensitivity: pl.DataFrame
    random_targeting_profit: float


def run_targeting_analysis(
    rankings: Mapping[str, np.ndarray],
    frame: ModelFrame,
    *,
    chosen: str,
    config: BusinessConfig,
    average_transaction: float,
    leaderboard_run_id: str,
    n_folds: int,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    tracking: TrackingConfig | None = None,
    lineage: dict[str, Path] | None = None,
) -> TargetingAnalysis:
    """Profit curves, the depth decision, and the sensitivity table, in one MLflow run.

    Args:
        rankings: Label -> out-of-fold scores for ``frame``'s clients; must
            include ``chosen``.
        frame: The clients the rankings score (train + val).
        average_transaction: Mean basket value, used when the config leaves
            ``margin_per_purchase`` null.
        lineage: ``repo_dir`` / ``data_dir`` / ``processed_dir`` overrides (for tests).

    Raises:
        ValueError: If ``chosen`` has no ranking.
    """
    if chosen not in rankings:
        raise ValueError(f"No ranking for the chosen model {chosen!r}")
    labels = [chosen, *(label for label in rankings if label != chosen)]
    by_depth = {
        label: uplift_by_depth(
            rankings[label], frame.treatment, frame.outcome, n_bootstrap=n_bootstrap, seed=seed
        )
        for label in labels
    }
    economics = config.economics(average_transaction)
    curves = {label: profit_curve(by_depth[label], economics) for label in labels}
    choice = choose_depth(curves[chosen], max_depth=depth_cap(config, economics.sms_cost))
    sensitivity = sensitivity_table(by_depth, chosen, economics.margin_per_purchase, config)
    random_profit = random_targeting_profit(by_depth[chosen].ate, economics)

    with start_run(
        Experiment.TARGETING,
        run_name="targeting",
        tags={
            "run_type": "targeting",
            "chosen_model": chosen,
            "reference_models": ",".join(labels[1:]),
            "leaderboard_run_id": leaderboard_run_id,
            "depth_rule": DEPTH_RULE,
            "scored_clients": frame.split,
            "currency": config.currency,
        },
        config=tracking,
        **(lineage or {}),
    ) as run:
        mlflow.log_params(
            {
                "sms_cost": config.sms_cost,
                "gross_margin": config.gross_margin,
                "margin_per_purchase_configured": config.margin_per_purchase,
                "budget": config.budget,
                "campaign_clients": config.campaign_clients,
                "n_folds": n_folds,
                "n_bootstrap": n_bootstrap,
                "seed": seed,
            }
        )
        i = _at(curves[chosen], choice.chosen_depth)
        mlflow.log_metrics(
            {
                "average_transaction": average_transaction,
                "margin_per_purchase": economics.margin_per_purchase,
                "break_even_uplift": economics.break_even_uplift,
                "ate": by_depth[chosen].ate,
                "n_clients": len(frame.outcome),
                "chosen_depth": choice.chosen_depth,
                "best_depth": choice.best_depth,
                "max_depth": choice.max_depth,
                "profit_per_1000": choice.profit_per_1000,
                "profit_per_1000_ci_lower": choice.profit_lower,
                "profit_per_1000_ci_upper": choice.profit_upper,
                "best_depth_profit_per_1000": float(curves[chosen].profit_per_1000.max()),
                "sms_per_1000": float(curves[chosen].sms_per_1000[i]),
                "extra_purchases_per_1000": float(curves[chosen].extra_purchases_per_1000[i]),
                "random_targeting_profit_per_1000": random_profit,
                **{
                    f"{label}_best_profit_per_1000": float(curves[label].profit_per_1000.max())
                    for label in labels[1:]
                },
            }
        )
        mlflow.log_dict(
            {
                **choice.as_dict(),
                "rule": DEPTH_RULE,
                "chosen_model": chosen,
                "sms_cost": economics.sms_cost,
                "margin_per_purchase": economics.margin_per_purchase,
                "break_even_uplift": economics.break_even_uplift,
            },
            "targeting/decision.json",
        )
        mlflow.log_text(curve_table(curves).write_csv(), "targeting/profit_curves.csv")
        mlflow.log_text(sensitivity.write_csv(), "targeting/sensitivity.csv")
        mlflow.log_figure(
            profit_curve_figure(curves, chosen=chosen, chosen_depth=choice.chosen_depth),
            "targeting/profit_curves.png",
        )
        return TargetingAnalysis(
            run.info.run_id, economics, choice, curves, sensitivity, random_profit
        )
