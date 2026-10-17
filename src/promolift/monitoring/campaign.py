"""Realized uplift of a finished campaign, against what the model predicted.

The send list texted the clients with the highest predicted uplift and held a
random share of them back as a control group, so the difference in purchase
rate between the two is the campaign's real effect on exactly the clients the
model chose. This module measures it and compares it with the prediction:

- **Overall:** realized uplift (treated minus control purchase rate) with a
  bootstrap interval, the mean predicted uplift of the same clients, and the
  gap between them. A prediction outside the realized interval is flagged.
- **By decile:** the same comparison within each predicted-uplift decile of
  the batch, to show where the model over- or under-promises.
- **Ranking:** Qini AUC and uplift at k among the targeted clients, from the
  real outcomes, with bootstrap intervals -- does the order still hold?

Intervals follow the project's evaluation rules: percentile bootstrap, 95%,
and never a point estimate alone. The realized-uplift bootstrap draws each
arm's buyers from a binomial at the observed rate, which is the bootstrap
distribution of a proportion computed directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from promolift.evaluation.ranking_metrics import DEFAULT_K_GRID
from promolift.evaluation.uncertainty import (
    DEFAULT_CONFIDENCE,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_SEED,
    BootstrapMetrics,
    Interval,
    bootstrap_metrics,
)
from promolift.monitoring.campaign_data import CampaignData

# Ranking metrics re-rank the whole campaign on every resample, which costs far
# more than the binomial draws, so they get fewer by default.
DEFAULT_RANKING_BOOTSTRAP = 200


@dataclass
class CampaignEvaluation:
    """Realized against predicted uplift for one campaign."""

    scoring_run_id: str
    confidence: float
    n_treated: int
    n_control: int
    n_missing_treated: int
    n_missing_control: int
    treated_rate: float
    control_rate: float
    realized_uplift: Interval
    predicted_uplift: float
    deviation: Interval
    incremental_purchases: float
    ranking: BootstrapMetrics
    predicted_uplift_at_k: dict[float, float]
    deciles: pl.DataFrame

    @property
    def prediction_within_interval(self) -> bool:
        """True if the predicted uplift lies inside the realized uplift's interval."""
        return self.realized_uplift.lower <= self.predicted_uplift <= self.realized_uplift.upper


def uplift_interval(
    treated_buyers: int,
    n_treated: int,
    control_buyers: int,
    n_control: int,
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    confidence: float,
) -> Interval:
    """Treated minus control purchase rate, with a percentile bootstrap interval."""
    treated_rate, control_rate = treated_buyers / n_treated, control_buyers / n_control
    draws = (
        rng.binomial(n_treated, treated_rate, n_bootstrap) / n_treated
        - rng.binomial(n_control, control_rate, n_bootstrap) / n_control
    )
    alpha = 1 - confidence
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(treated_rate - control_rate, float(lower), float(upper))


def _predicted_uplift_at_k(scores: np.ndarray, k_grid: Sequence[float]) -> dict[float, float]:
    """Mean predicted uplift of the top share ``k`` of the targeted clients."""
    ranked = np.sort(scores)[::-1]
    return {k: float(ranked[: int(len(ranked) * k)].mean()) for k in k_grid}


def decile_comparison(
    frame: pl.DataFrame, *, rng: np.random.Generator, n_bootstrap: int, confidence: float
) -> pl.DataFrame:
    """Realized against predicted uplift in each decile of the batch that was targeted.

    A decile without clients in both arms has no realized uplift (null).
    """
    rows = []
    for decile in sorted(frame["decile"].unique().to_list()):
        part = frame.filter(pl.col("decile") == decile)
        treated, control = part.filter(pl.col("treated") == 1), part.filter(pl.col("treated") == 0)
        predicted = float(part["uplift_score"].mean())
        row: dict = {
            "decile": decile,
            "n_treated": treated.height,
            "n_control": control.height,
            "treated_rate": None,
            "control_rate": None,
            "realized_uplift": None,
            "realized_lower": None,
            "realized_upper": None,
            "predicted_uplift": predicted,
            "deviation": None,
        }
        if treated.height and control.height:
            interval = uplift_interval(
                int(treated["purchased"].sum()),
                treated.height,
                int(control["purchased"].sum()),
                control.height,
                rng=rng,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
            )
            row |= {
                "treated_rate": float(treated["purchased"].mean()),
                "control_rate": float(control["purchased"].mean()),
                "realized_uplift": interval.estimate,
                "realized_lower": interval.lower,
                "realized_upper": interval.upper,
                "deviation": interval.estimate - predicted,
            }
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def evaluate_campaign(
    data: CampaignData,
    *,
    k_grid: Sequence[float] = DEFAULT_K_GRID,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ranking_bootstrap: int = DEFAULT_RANKING_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> CampaignEvaluation:
    """Compare the campaign's realized uplift with the model's prediction.

    Args:
        data: Matched results (``campaign_data.match_results``).
        k_grid: Shares of the targeted clients for uplift at k.
        n_bootstrap: Resamples for the realized-uplift intervals.
        ranking_bootstrap: Resamples for the Qini and uplift-at-k intervals.
        confidence: Interval coverage.
        seed: Bootstrap seed, so a re-run reproduces the intervals.
    """
    frame = data.frame
    rng = np.random.default_rng(seed)
    treated = frame.filter(pl.col("treated") == 1)
    control = frame.filter(pl.col("treated") == 0)
    realized = uplift_interval(
        int(treated["purchased"].sum()),
        treated.height,
        int(control["purchased"].sum()),
        control.height,
        rng=rng,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )
    scores = frame["uplift_score"].to_numpy()
    predicted = float(scores.mean())
    return CampaignEvaluation(
        scoring_run_id=data.scoring_run_id,
        confidence=confidence,
        n_treated=data.n_treated,
        n_control=data.n_control,
        n_missing_treated=data.n_missing_treated,
        n_missing_control=data.n_missing_control,
        treated_rate=float(treated["purchased"].mean()),
        control_rate=float(control["purchased"].mean()),
        realized_uplift=realized,
        predicted_uplift=predicted,
        deviation=Interval(
            realized.estimate - predicted, realized.lower - predicted, realized.upper - predicted
        ),
        incremental_purchases=realized.estimate * treated.height,
        ranking=bootstrap_metrics(
            scores,
            frame["treated"].to_numpy(),
            frame["purchased"].to_numpy(),
            k_grid=k_grid,
            n_bootstrap=ranking_bootstrap,
            confidence=confidence,
            seed=seed,
        ),
        predicted_uplift_at_k=_predicted_uplift_at_k(scores, k_grid),
        deciles=decile_comparison(frame, rng=rng, n_bootstrap=n_bootstrap, confidence=confidence),
    )
