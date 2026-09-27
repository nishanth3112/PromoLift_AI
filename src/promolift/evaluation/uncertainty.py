"""Uncertainty for uplift metrics: bootstrap CIs, a random-ranking noise floor, paired tests.

Uplift differences between models are small relative to sampling noise (on
the 40k-client validation split the ATE CI alone is +/-0.95pp), so a point
estimate without an interval is not evidence. Three complementary views:

- ``bootstrap_metrics``: sampling uncertainty of one model's metrics.
- ``random_ranking_noise_floor``: how good a metric looks by pure chance on
  this exact data; a model only "works" if it clears this band.
- ``paired_comparison``: whether model A beats model B, scored on the same
  resamples so shared noise cancels out.

All three compute metrics through ``ranking_metrics``, so they can never
disagree with a point estimate on what a metric means.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from promolift.evaluation.ranking_metrics import (
    DEFAULT_K_GRID,
    RankingMetrics,
    ranking_metrics,
    uplift_at_k_name,
)

DEFAULT_N_BOOTSTRAP = 1_000
DEFAULT_N_RANKINGS = 200
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42


@dataclass(frozen=True)
class Interval:
    """A metric estimate with a percentile interval."""

    estimate: float
    lower: float
    upper: float


@dataclass
class BootstrapMetrics:
    """Full-data estimates with bootstrap percentile CIs."""

    n_bootstrap: int
    confidence: float
    qini_auc: Interval
    uplift_auc: Interval
    uplift_at_k: dict[float, Interval]


@dataclass
class NoiseFloor:
    """Distribution of each metric under random rankings: median and central range."""

    n_rankings: int
    n_clients: int
    confidence: float
    qini_auc: Interval
    uplift_auc: Interval
    uplift_at_k: dict[float, Interval]


@dataclass(frozen=True)
class PairedDifference:
    """Metric of model A minus model B, with its CI and P(A > B) over resamples."""

    difference: float
    lower: float
    upper: float
    win_rate: float


@dataclass
class PairedComparison:
    """Per-metric paired differences between two models on the same clients."""

    n_bootstrap: int
    confidence: float
    qini_auc: PairedDifference
    uplift_auc: PairedDifference
    uplift_at_k: dict[float, PairedDifference]


def stratified_resample_indices(treatment: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap indices drawn separately within treated and control clients.

    Arm sizes stay fixed, mirroring the randomized experiment's design, and
    every resample is guaranteed to contain both arms.
    """
    treated = np.flatnonzero(treatment == 1)
    control = np.flatnonzero(treatment == 0)
    return np.concatenate(
        [rng.choice(treated, size=len(treated)), rng.choice(control, size=len(control))]
    )


def _vector(metrics: RankingMetrics) -> np.ndarray:
    return np.array(
        [metrics.qini_auc, metrics.uplift_auc, *metrics.uplift_at_k.values()], dtype=float
    )


def _interval_band(samples: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1 - confidence
    return (
        np.quantile(samples, alpha / 2, axis=0),
        np.quantile(samples, 1 - alpha / 2, axis=0),
    )


def _split_by_metric(values: list, k_grid: Sequence[float]) -> dict:
    return {
        "qini_auc": values[0],
        "uplift_auc": values[1],
        "uplift_at_k": dict(zip(k_grid, values[2:], strict=True)),
    }


def bootstrap_metrics(
    scores: ArrayLike,
    treatment: ArrayLike,
    outcome: ArrayLike,
    *,
    k_grid: Sequence[float] = DEFAULT_K_GRID,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> BootstrapMetrics:
    """Full-data metric estimates with stratified-bootstrap percentile CIs."""
    s, t, y = np.asarray(scores, dtype=float), np.asarray(treatment), np.asarray(outcome)
    estimate = _vector(ranking_metrics(s, t, y, k_grid=k_grid))
    rng = np.random.default_rng(seed)
    samples = np.array(
        [
            _vector(ranking_metrics(s[idx], t[idx], y[idx], k_grid=k_grid))
            for idx in (stratified_resample_indices(t, rng) for _ in range(n_bootstrap))
        ]
    )
    lower, upper = _interval_band(samples, confidence)
    intervals = [
        Interval(float(e), float(lo), float(hi))
        for e, lo, hi in zip(estimate, lower, upper, strict=True)
    ]
    return BootstrapMetrics(
        n_bootstrap=n_bootstrap, confidence=confidence, **_split_by_metric(intervals, k_grid)
    )


def random_ranking_noise_floor(
    treatment: ArrayLike,
    outcome: ArrayLike,
    *,
    k_grid: Sequence[float] = DEFAULT_K_GRID,
    n_rankings: int = DEFAULT_N_RANKINGS,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> NoiseFloor:
    """Median and central range of each metric over uniformly random rankings.

    Qini/uplift AUC center on 0; uplift@k centers on the ATE, because a random
    top-k% is just a random sample -- so clearing the uplift@k band means
    beating untargeted sending, not just beating zero.
    """
    t, y = np.asarray(treatment), np.asarray(outcome)
    rng = np.random.default_rng(seed)
    samples = np.array(
        [
            _vector(ranking_metrics(rng.random(len(t)), t, y, k_grid=k_grid))
            for _ in range(n_rankings)
        ]
    )
    median = np.median(samples, axis=0)
    lower, upper = _interval_band(samples, confidence)
    intervals = [
        Interval(float(m), float(lo), float(hi))
        for m, lo, hi in zip(median, lower, upper, strict=True)
    ]
    return NoiseFloor(
        n_rankings=n_rankings,
        n_clients=len(t),
        confidence=confidence,
        **_split_by_metric(intervals, k_grid),
    )


def beats_noise_floor(metrics: RankingMetrics, floor: NoiseFloor) -> dict[str, bool]:
    """Whether each metric's estimate lies above the noise floor's upper bound."""
    bounds = {
        "qini_auc": floor.qini_auc.upper,
        "uplift_auc": floor.uplift_auc.upper,
        **{uplift_at_k_name(k): band.upper for k, band in floor.uplift_at_k.items()},
    }
    return {name: value > bounds[name] for name, value in metrics.as_flat_dict().items()}


def paired_comparison(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    treatment: ArrayLike,
    outcome: ArrayLike,
    *,
    k_grid: Sequence[float] = DEFAULT_K_GRID,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> PairedComparison:
    """Metric differences (A - B) with bootstrap CIs and win rates.

    Both models are scored on the same resampled clients in every replicate,
    so client-level noise shared by both rankings cancels out of the
    difference -- far more sensitive than comparing two independent CIs.
    """
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    t, y = np.asarray(treatment), np.asarray(outcome)
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must be the same length")

    def difference(idx: np.ndarray | slice) -> np.ndarray:
        return _vector(ranking_metrics(a[idx], t[idx], y[idx], k_grid=k_grid)) - _vector(
            ranking_metrics(b[idx], t[idx], y[idx], k_grid=k_grid)
        )

    estimate = difference(slice(None))
    rng = np.random.default_rng(seed)
    samples = np.array(
        [difference(stratified_resample_indices(t, rng)) for _ in range(n_bootstrap)]
    )
    lower, upper = _interval_band(samples, confidence)
    win_rate = (samples > 0).mean(axis=0)
    differences = [
        PairedDifference(float(e), float(lo), float(hi), float(w))
        for e, lo, hi, w in zip(estimate, lower, upper, win_rate, strict=True)
    ]
    return PairedComparison(
        n_bootstrap=n_bootstrap, confidence=confidence, **_split_by_metric(differences, k_grid)
    )
