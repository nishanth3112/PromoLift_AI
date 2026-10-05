"""How deep to target: the profit of texting the top k% of clients ranked by uplift score.

Texting the top share ``k`` of the population costs one SMS per client and
causes ``k * uplift@k`` extra purchases per client of the population, where
``uplift@k`` is the treated-minus-control conversion rate among those clients.
With margin ``m`` per extra purchase and cost ``c`` per SMS,

    profit(k) = k * (m * uplift@k - c)        per client of the population,

so the best depth depends only on the break-even uplift ``r = c / m``: keep
texting while the next clients' uplift exceeds ``r``. Profits are reported per
1,000 clients of the population, so a curve estimated on one set of clients
applies to a campaign of any size. Depth 0 (text nobody, profit exactly 0) is
always a candidate: if no depth is reliably profitable, the answer is to not send.

Uplift by depth is computed once per ranking, with bootstrap replicates;
profit for any cost/margin is then a cheap linear transform, so sensitivity
tables over many scenarios cost nothing extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from promolift.evaluation.uncertainty import (
    DEFAULT_CONFIDENCE,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_SEED,
    stratified_resample_indices,
)

# Targeting depths 0%, 1%, ..., 100% of the population.
DEFAULT_DEPTHS = tuple(round(i / 100, 2) for i in range(101))
PER_CLIENTS = 1_000


@dataclass(frozen=True)
class Economics:
    """What one SMS costs and what one extra purchase earns, in the same currency."""

    sms_cost: float
    margin_per_purchase: float

    def __post_init__(self) -> None:
        if self.sms_cost < 0:
            raise ValueError("sms_cost must be non-negative")
        if self.margin_per_purchase <= 0:
            raise ValueError("margin_per_purchase must be positive")

    @property
    def break_even_uplift(self) -> float:
        """Uplift a client must have for texting them to pay: cost / margin."""
        return self.sms_cost / self.margin_per_purchase


@dataclass(frozen=True)
class UpliftByDepth:
    """Observed uplift among the top share of clients, at each depth, with bootstrap replicates.

    ``share`` is the fraction actually targeted at each depth (ties and
    rounding make it differ slightly from the nominal depth); ``samples``
    holds one row of (share, uplift) per bootstrap replicate.
    """

    depths: np.ndarray
    share: np.ndarray
    uplift: np.ndarray
    sample_share: np.ndarray
    sample_uplift: np.ndarray
    ate: float


def _uplift_at_depths(
    treated: np.ndarray,
    converted: np.ndarray,
    weights: np.ndarray,
    depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(share targeted, uplift) at each depth, for clients already sorted best-first.

    ``weights`` are bootstrap multiplicities (all ones for the full data), so
    one sort serves every replicate. The top ``k`` of a resample is the
    prefix holding ``k`` of its total weight.
    """
    cum_weight = np.cumsum(weights)
    cum_treated = np.cumsum(weights * treated)
    cum_treated_converted = np.cumsum(weights * treated * converted)
    cum_control = np.cumsum(weights * (1 - treated))
    cum_control_converted = np.cumsum(weights * (1 - treated) * converted)
    total = cum_weight[-1]

    share = np.zeros(len(depths))
    uplift = np.zeros(len(depths))
    positive = depths > 0
    ends = np.searchsorted(cum_weight, depths[positive] * total - 1e-9)
    ends = np.minimum(ends, len(cum_weight) - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_treated = cum_treated_converted[ends] / cum_treated[ends]
        rate_control = cum_control_converted[ends] / cum_control[ends]
    share[positive] = cum_weight[ends] / total
    uplift[positive] = rate_treated - rate_control
    return share, uplift


def uplift_by_depth(
    scores: ArrayLike,
    treatment: ArrayLike,
    outcome: ArrayLike,
    *,
    depths: Sequence[float] = DEFAULT_DEPTHS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> UpliftByDepth:
    """Uplift among the top share of clients ranked by ``scores``, at each depth.

    Bootstrap replicates resample within the treated and control arms (as
    every other CI in the project does) and keep the ranking fixed.

    Raises:
        ValueError: If inputs are misaligned, depths fall outside [0, 1], or
            some positive depth has no treated or no control client.
    """
    s, t, y = np.asarray(scores, dtype=float), np.asarray(treatment), np.asarray(outcome)
    if not (s.shape == t.shape == y.shape) or s.ndim != 1:
        raise ValueError("scores, treatment, and outcome must be 1-D and the same length")
    grid = np.asarray(depths, dtype=float)
    if grid.min() < 0 or grid.max() > 1:
        raise ValueError("depths must lie in [0, 1]")

    order = np.argsort(-s, kind="stable")
    treated, converted = t[order].astype(float), y[order].astype(float)
    share, uplift = _uplift_at_depths(treated, converted, np.ones(len(s)), grid)
    if np.isnan(uplift).any():
        raise ValueError("some depth has no treated or no control client; use a coarser grid")

    rng = np.random.default_rng(seed)
    sample_share = np.empty((n_bootstrap, len(grid)))
    sample_uplift = np.empty((n_bootstrap, len(grid)))
    for b in range(n_bootstrap):
        weights = np.bincount(stratified_resample_indices(t, rng), minlength=len(s))[order]
        sample_share[b], sample_uplift[b] = _uplift_at_depths(treated, converted, weights, grid)
    # A replicate whose top k lost one arm entirely contributes no uplift there.
    sample_uplift = np.nan_to_num(sample_uplift, nan=0.0)

    return UpliftByDepth(
        depths=grid,
        share=share,
        uplift=uplift,
        sample_share=sample_share,
        sample_uplift=sample_uplift,
        ate=float(y[t == 1].mean() - y[t == 0].mean()),
    )


@dataclass(frozen=True)
class ProfitCurve:
    """Per-1,000-client economics of texting each depth, with bootstrap CIs on profit."""

    depths: np.ndarray
    sms_per_1000: np.ndarray
    extra_purchases_per_1000: np.ndarray
    profit_per_1000: np.ndarray
    profit_lower: np.ndarray
    profit_upper: np.ndarray
    sample_profit: np.ndarray
    economics: Economics

    @property
    def roi(self) -> np.ndarray:
        """Profit per unit spent on SMS at each depth (NaN at depth 0 or free SMS)."""
        spend = self.sms_per_1000 * self.economics.sms_cost
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(spend > 0, self.profit_per_1000 / spend, np.nan)


def _profit(share: np.ndarray, uplift: np.ndarray, economics: Economics) -> np.ndarray:
    margin, cost = economics.margin_per_purchase, economics.sms_cost
    return PER_CLIENTS * share * (margin * uplift - cost)


def profit_curve(
    by_depth: UpliftByDepth, economics: Economics, *, confidence: float = DEFAULT_CONFIDENCE
) -> ProfitCurve:
    """Profit per 1,000 clients of the population at each targeting depth."""
    samples = _profit(by_depth.sample_share, by_depth.sample_uplift, economics)
    alpha = 1 - confidence
    return ProfitCurve(
        depths=by_depth.depths,
        sms_per_1000=PER_CLIENTS * by_depth.share,
        extra_purchases_per_1000=PER_CLIENTS * by_depth.share * by_depth.uplift,
        profit_per_1000=_profit(by_depth.share, by_depth.uplift, economics),
        profit_lower=np.quantile(samples, alpha / 2, axis=0),
        profit_upper=np.quantile(samples, 1 - alpha / 2, axis=0),
        sample_profit=samples,
        economics=economics,
    )


def random_targeting_profit(ate: float, economics: Economics) -> float:
    """Profit per 1,000 clients of the best untargeted policy: text everyone or no one.

    A random share k earns k * (m * ATE - c), linear in k, so the best random
    policy is all-or-nothing.
    """
    return max(0.0, PER_CLIENTS * (economics.margin_per_purchase * ate - economics.sms_cost))


def budget_depth_cap(budget: float, sms_cost: float, campaign_clients: int) -> float:
    """Largest share of ``campaign_clients`` a ``budget`` can text."""
    if budget < 0 or campaign_clients <= 0:
        raise ValueError("budget must be non-negative and campaign_clients positive")
    if sms_cost == 0:
        return 1.0
    return min(1.0, budget / (sms_cost * campaign_clients))


@dataclass(frozen=True)
class DepthChoice:
    """The profit-maximizing depth, and the shallowest depth statistically tied with it."""

    depth: float
    profit_per_1000: float
    profit_lower: float
    profit_upper: float
    shallowest_tied_depth: float
    shallowest_tied_profit: float
    tied_depths: tuple[float, ...]
    max_depth: float

    def as_dict(self) -> dict:
        return {
            "depth": self.depth,
            "profit_per_1000": self.profit_per_1000,
            "profit_lower": self.profit_lower,
            "profit_upper": self.profit_upper,
            "shallowest_tied_depth": self.shallowest_tied_depth,
            "shallowest_tied_profit": self.shallowest_tied_profit,
            "tied_depths": list(self.tied_depths),
            "max_depth": self.max_depth,
        }


def choose_depth(
    curve: ProfitCurve, *, max_depth: float = 1.0, confidence: float = DEFAULT_CONFIDENCE
) -> DepthChoice:
    """The depth with the highest expected profit, up to ``max_depth`` (a budget cap).

    Equal profits go to the shallower depth. Depth 0 (don't send) wins when
    no depth has positive expected profit.

    Also reported: the shallowest depth whose profit is statistically tied
    with the best (the paired bootstrap CI of the difference includes zero),
    a lower-spend option. It is not the decision: the profit curve is noisy,
    so the tied range is wide, and its shallow end can earn less in
    expectation than texting everyone.
    """
    allowed = np.flatnonzero(curve.depths <= max_depth + 1e-9)
    best = allowed[np.argmax(curve.profit_per_1000[allowed])]
    alpha = 1 - confidence
    differences = curve.sample_profit[:, allowed] - curve.sample_profit[:, [best]]
    lower = np.quantile(differences, alpha / 2, axis=0)
    upper = np.quantile(differences, 1 - alpha / 2, axis=0)
    # The best is always tied with itself (its difference is identically zero).
    tied = allowed[(lower <= 0) & (upper >= 0)]
    shallowest = int(tied.min())
    return DepthChoice(
        depth=float(curve.depths[best]),
        profit_per_1000=float(curve.profit_per_1000[best]),
        profit_lower=float(curve.profit_lower[best]),
        profit_upper=float(curve.profit_upper[best]),
        shallowest_tied_depth=float(curve.depths[shallowest]),
        shallowest_tied_profit=float(curve.profit_per_1000[shallowest]),
        tied_depths=tuple(float(curve.depths[i]) for i in tied),
        max_depth=max_depth,
    )
