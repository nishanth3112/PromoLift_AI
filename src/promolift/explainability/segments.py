"""Do different kinds of clients get their predicted uplift from different features?

Global importance averages over everyone, which can hide that the clients the
campaign targets are scored for different reasons than the rest. This splits
clients into segments -- by predicted-uplift tier, gender, and age band -- and
reports each feature's importance inside each one.

Segments are descriptive groupings of the explained sample; they are not
fairness tests, and a small segment's numbers are noisy (``n`` is reported).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from promolift.explainability.shap_values import Explanation

# Predicted-uplift tiers by cumulative share of clients, highest scores first. The
# campaign texts roughly the top 40%, so the tiers split that targeted group.
TIERS = ((0.10, "top 10%"), (0.40, "next 30%"), (1.00, "rest"))
AGE_EDGES = (-np.inf, 30, 45, 60, np.inf)
AGE_LABELS = ("under 30", "30-44", "45-59", "60 and over")
AGE_NOT_VALID = "age not valid"
TIER_SEGMENT = "predicted_uplift_tier"
GENDER_SEGMENT = "gender"
AGE_SEGMENT = "age_band"


_NATURAL_ORDER = {
    TIER_SEGMENT: [label for _, label in TIERS],
    AGE_SEGMENT: [*AGE_LABELS, AGE_NOT_VALID],
}


def _ordered(segment_type: str, present: list) -> list:
    """Segment labels in their natural order (tiers high to low, ages young to old), else sorted."""
    natural = _NATURAL_ORDER.get(segment_type, [])
    known = [label for label in natural if label in present]
    return known + sorted((label for label in present if label not in natural), key=str)


def _tier(scores: np.ndarray, index: pd.Index) -> pd.Series:
    share_above = pd.Series(scores, index=index).rank(ascending=False, method="first") / len(scores)
    labels = pd.Series("", index=index, dtype=object)
    lower = 0.0
    for upper, label in TIERS:
        labels[(share_above > lower) & (share_above <= upper)] = label
        lower = upper
    return labels


def age_band(age: pd.Series, is_valid: pd.Series | None = None) -> pd.Series:
    """Age-band label per client; missing or flagged-invalid ages are ``AGE_NOT_VALID``.

    Args:
        age: Age in years, NaN where unknown.
        is_valid: 1 where the age is believed (``age_is_valid``); None trusts every known age.
    """
    trusted = age.notna() if is_valid is None else (is_valid == 1) & age.notna()
    band = pd.cut(age, bins=list(AGE_EDGES), right=False, labels=list(AGE_LABELS))
    return band.astype(object).where(trusted, AGE_NOT_VALID)


def default_segments(explanation: Explanation) -> dict[str, pd.Series]:
    """Segment labels per client, aligned to the explanation, for the groupings that apply.

    The tier segmentation always applies; gender and age bands only if the
    explained features include ``gender`` / ``age``.
    """
    features = explanation.features
    segments = {TIER_SEGMENT: _tier(explanation.scores, features.index)}
    if "gender" in features.columns:
        segments[GENDER_SEGMENT] = features["gender"].astype(object).fillna("missing").astype(str)
    if "age" in features.columns:
        valid = features["age_is_valid"] if "age_is_valid" in features.columns else None
        segments[AGE_SEGMENT] = age_band(features["age"], valid)
    return segments


def segment_importance(explanation: Explanation, segments: dict[str, pd.Series]) -> pl.DataFrame:
    """Feature importance within each segment.

    One row per (segment type, segment, feature): ``n`` and ``mean_score`` of the
    segment, the feature's ``mean_abs_contribution`` and ``share`` of the
    segment's total, and its signed ``mean_contribution``.

    Raises:
        ValueError: If a segment labelling isn't aligned to the explanation.
    """
    contributions = explanation.contributions
    scores = pd.Series(explanation.scores, index=contributions.index)
    rows = []
    for segment_type, labels in segments.items():
        if not labels.index.equals(contributions.index):
            raise ValueError(f"Segments {segment_type!r} are not aligned to the explanation")
        for segment in _ordered(segment_type, list(labels.unique())):
            members = labels == segment
            mean_abs = contributions[members].abs().mean()
            total = float(mean_abs.sum())
            for feature in contributions.columns:
                rows.append(
                    {
                        "segment_type": segment_type,
                        "segment": str(segment),
                        "n": int(members.sum()),
                        "mean_score": float(scores[members].mean()),
                        "feature": feature,
                        "mean_abs_contribution": float(mean_abs[feature]),
                        "share": float(mean_abs[feature]) / total if total > 0 else 0.0,
                        "mean_contribution": float(contributions.loc[members, feature].mean()),
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def top_features(table: pl.DataFrame, k: int = 3) -> pl.DataFrame:
    """The ``k`` most important features of every segment, with their shares, as one row each."""
    rows = []
    for (segment_type, segment), group in table.group_by(
        ["segment_type", "segment"], maintain_order=True
    ):
        top = group.sort("mean_abs_contribution", descending=True).head(k)
        rows.append(
            {
                "segment_type": segment_type,
                "segment": segment,
                "n": group["n"][0],
                "mean_score": group["mean_score"][0],
                "top_features": ", ".join(
                    f"{feature} ({share:.0%})"
                    for feature, share in zip(top["feature"], top["share"], strict=True)
                ),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)
