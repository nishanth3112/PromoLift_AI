"""Global feature importance of the uplift model, from SHAP contributions.

Importance is the mean absolute contribution to the predicted uplift, in
uplift units (0.01 = one percentage point): how far, on average, a feature
moves a client's score away from the average client's. ``share`` is that as a
fraction of the total over all features.

Direction is the rank correlation between a feature's value and its
contribution: positive means higher values raise the predicted uplift. It
says nothing for features that act in both directions (U-shapes, thresholds),
which the dependence table shows bin by bin. A feature with no contribution at
all is "unused by the model".
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from promolift.explainability.shap_values import Explanation

# Rank correlations inside this band are reported as "mixed", not as a direction.
DIRECTION_THRESHOLD = 0.2
# Features with at most this many distinct values are shown value by value in
# dependence tables; others are cut into quantile bins.
MAX_DISTINCT_VALUES = 10
DEFAULT_BINS = 10
MISSING_LABEL = "missing"


UNUSED = "unused by the model"


def _direction(values: pd.Series, contribution: pd.Series) -> tuple[float | None, str]:
    """Spearman correlation of a numeric feature with its contribution, and its reading.

    A feature the model never splits on contributes the same (zero) to every
    client, so there is no direction to read.
    """
    if contribution.nunique() < 2:
        return None, UNUSED
    if not pd.api.types.is_numeric_dtype(values.dtype):
        return None, "categorical"
    present = values.notna()
    if values[present].nunique() < 2:
        return None, "constant"
    correlation = values[present].corr(contribution[present], method="spearman")
    if pd.isna(correlation):
        return None, "constant"
    if correlation > DIRECTION_THRESHOLD:
        return float(correlation), "higher value, higher uplift"
    if correlation < -DIRECTION_THRESHOLD:
        return float(correlation), "higher value, lower uplift"
    return float(correlation), "mixed"


def global_importance(explanation: Explanation) -> pl.DataFrame:
    """One row per feature, most important first.

    Columns: ``rank``, ``feature``, ``mean_abs_contribution``, ``share``,
    ``mean_contribution`` (signed), ``direction_corr`` and ``direction``,
    ``missing_share``, and ``mean_when_missing`` (the mean contribution among
    clients missing the feature; null if none are).
    """
    contributions = explanation.contributions
    mean_abs = contributions.abs().mean()
    total = float(mean_abs.sum())
    rows = []
    for feature in contributions.columns:
        values = explanation.features[feature]
        contribution = contributions[feature]
        corr, direction = _direction(values, contribution)
        missing = values.isna()
        rows.append(
            {
                "feature": feature,
                "mean_abs_contribution": float(mean_abs[feature]),
                "share": float(mean_abs[feature]) / total if total > 0 else 0.0,
                "mean_contribution": float(contribution.mean()),
                "direction_corr": corr,
                "direction": direction,
                "missing_share": float(missing.mean()),
                "mean_when_missing": float(contribution[missing].mean()) if missing.any() else None,
            }
        )
    table = pl.DataFrame(rows, infer_schema_length=None).sort(
        "mean_abs_contribution", descending=True
    )
    return table.with_row_index("rank", offset=1).with_columns(pl.col("rank").cast(pl.Int64))


def dependence_table(
    explanation: Explanation, feature: str, *, bins: int = DEFAULT_BINS
) -> pl.DataFrame:
    """Mean contribution of ``feature`` by its value: how the score moves along the feature.

    Numeric features with few distinct values are shown value by value, others
    in quantile bins labelled by their median value; categorical features by
    level. Clients missing the feature form their own last row.

    Columns: ``label``, ``n``, ``mean_contribution``, ``mean_value`` (null for
    categorical and missing rows).

    Raises:
        KeyError: If ``feature`` isn't one of the explained features.
    """
    if feature not in explanation.contributions.columns:
        raise KeyError(f"{feature!r} is not an explained feature")
    values = explanation.features[feature]
    contribution = explanation.contributions[feature]
    present = values.notna()
    is_numeric = pd.api.types.is_numeric_dtype(values.dtype)

    if not is_numeric:
        groups = values[present].astype(str)
        order = sorted(groups.unique())
    elif values[present].nunique() <= MAX_DISTINCT_VALUES:
        groups = values[present]
        order = sorted(groups.unique())
    else:
        groups = pd.qcut(values[present], q=bins, duplicates="drop")
        order = list(groups.cat.categories)

    rows = []
    for group in order:
        in_group = present & (groups.reindex(values.index) == group).fillna(False)
        mean_value = float(values[in_group].median()) if is_numeric else None
        label = f"{mean_value:.4g}" if is_numeric else str(group)
        rows.append(
            {
                "label": label,
                "n": int(in_group.sum()),
                "mean_contribution": float(contribution[in_group].mean()),
                "mean_value": mean_value,
            }
        )
    if (~present).any():
        rows.append(
            {
                "label": MISSING_LABEL,
                "n": int((~present).sum()),
                "mean_contribution": float(contribution[~present].mean()),
                "mean_value": None,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "label": pl.String,
            "n": pl.Int64,
            "mean_contribution": pl.Float64,
            "mean_value": pl.Float64,
        },
    )
