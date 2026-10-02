"""Uplift ranking metrics: how well a score ranks clients by their treatment effect.

Individual uplift is never observed, so every metric compares treated vs
control outcomes within groups of clients ranked by the score. Thin wrappers
over scikit-uplift, with inputs validated up front so a bad array fails
loudly instead of producing a plausible-looking number.

Model selection uses ``qini_auc`` (normalized Qini: ranking quality across
every targeting depth, independent of a budget); ``uplift_at_k`` is the
business view -- the effect among the top k% of clients a campaign would target.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklift.metrics import (
    qini_auc_score,
    qini_curve,
    uplift_at_k,
    uplift_auc_score,
    uplift_by_percentile,
)

DEFAULT_K_GRID = (0.1, 0.2, 0.3, 0.5)
# "overall" ranks the whole population and takes its top k%, which is what a
# campaign does; "by_group" would take the top k% of each arm separately.
_STRATEGY = "overall"


def uplift_at_k_name(k: float) -> str:
    """Flat metric name for uplift at targeting depth ``k``, e.g. 0.3 -> ``uplift_at_30pct``."""
    return f"uplift_at_{round(k * 100)}pct"


@dataclass
class RankingMetrics:
    """Point estimates of ranking quality for one score on one set of clients."""

    n_clients: int
    qini_auc: float
    uplift_auc: float
    uplift_at_k: dict[float, float]

    def as_flat_dict(self) -> dict[str, float]:
        """All metrics under flat names, e.g. for comparison or MLflow logging."""
        return {
            "qini_auc": self.qini_auc,
            "uplift_auc": self.uplift_auc,
            **{uplift_at_k_name(k): v for k, v in self.uplift_at_k.items()},
        }


def _validated(
    scores: ArrayLike, treatment: ArrayLike, outcome: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s, t, y = np.asarray(scores, dtype=float), np.asarray(treatment), np.asarray(outcome)
    if not (s.shape == t.shape == y.shape) or s.ndim != 1:
        raise ValueError("scores, treatment, and outcome must be 1-D and the same length")
    if not np.isin(t, (0, 1)).all():
        raise ValueError("treatment must be binary (0/1)")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("outcome must be binary (0/1)")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    if t.min() == t.max():
        raise ValueError("need both treated and control clients to measure uplift")
    return s, t.astype(int), y.astype(int)


def qini_auc(scores: ArrayLike, treatment: ArrayLike, outcome: ArrayLike) -> float:
    """Normalized Qini AUC alone -- for bootstrap loops that need nothing else."""
    s, t, y = _validated(scores, treatment, outcome)
    return float(qini_auc_score(y, s, t))


def ranking_metrics(
    scores: ArrayLike,
    treatment: ArrayLike,
    outcome: ArrayLike,
    *,
    k_grid: Sequence[float] = DEFAULT_K_GRID,
) -> RankingMetrics:
    """Normalized Qini AUC, uplift AUC, and uplift at each targeting depth in ``k_grid``.

    Args:
        scores: Predicted uplift (or any ranking score); higher = target first.
        treatment: 1 if the client was treated, else 0.
        outcome: 1 if the client converted, else 0.
        k_grid: Fractions of the population to target, e.g. 0.3 = top 30%.
    """
    s, t, y = _validated(scores, treatment, outcome)
    return RankingMetrics(
        n_clients=len(s),
        qini_auc=float(qini_auc_score(y, s, t)),
        uplift_auc=float(uplift_auc_score(y, s, t)),
        uplift_at_k={k: float(uplift_at_k(y, s, t, strategy=_STRATEGY, k=k)) for k in k_grid},
    )


@dataclass
class QiniCurve:
    """Cumulative incremental conversions as the targeted share of clients grows."""

    fraction_targeted: np.ndarray
    incremental_conversions: np.ndarray


def qini_curve_points(
    scores: ArrayLike, treatment: ArrayLike, outcome: ArrayLike, *, max_points: int = 201
) -> QiniCurve:
    """The Qini curve from 0% to 100% targeted, downsampled to at most ``max_points``.

    Downsampling keeps reports and plots small (the raw curve has a point per
    client) without changing the curve's shape at plotting resolution.
    """
    s, t, y = _validated(scores, treatment, outcome)
    n_targeted, incremental = qini_curve(y, s, t)
    if n_targeted[0] != 0:
        n_targeted, incremental = np.r_[0, n_targeted], np.r_[0.0, incremental]
    keep = np.unique(np.linspace(0, len(n_targeted) - 1, min(len(n_targeted), max_points)).round())
    keep = keep.astype(int)
    return QiniCurve(
        fraction_targeted=n_targeted[keep] / n_targeted[-1],
        incremental_conversions=np.asarray(incremental[keep], dtype=float),
    )


def uplift_by_decile(
    scores: ArrayLike, treatment: ArrayLike, outcome: ArrayLike, *, bins: int = 10
) -> pl.DataFrame:
    """Treated/control conversion and uplift per score bucket, highest scores first.

    A well-ranked score shows uplift falling from the first bucket to the last.
    """
    s, t, y = _validated(scores, treatment, outcome)
    table = uplift_by_percentile(y, s, t, strategy=_STRATEGY, bins=bins, total=False)
    return pl.from_pandas(table.reset_index()).rename(
        {
            "percentile": "bucket",
            "response_rate_treatment": "treatment_rate",
            "response_rate_control": "control_rate",
        }
    )
