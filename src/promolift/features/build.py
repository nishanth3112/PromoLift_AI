"""Combines client-level feature groups into a single feature table.

Starts from the full client universe in clients.csv (not just clients in
uplift_train) so any client can be scored later, not just those in the
training experiment.

Nulls from purchase-derived columns (e.g. a client with no transactions) are
left as-is rather than imputed here. Imputation is a modeling-pipeline
concern, not a feature-store concern: tree-based models (LightGBM, CatBoost)
handle nulls natively, while linear-based causal learners (econml, causalml)
will need their own explicit imputation step, which may differ by model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import polars as pl

from promolift.features.basket_store import build_basket_store_features
from promolift.features.category_spend import build_category_spend_features
from promolift.features.demographics import build_demographic_features
from promolift.features.product_mix import build_product_mix_features
from promolift.features.promo_responsiveness import build_promo_responsiveness_features
from promolift.features.purchase_behavior import build_purchase_behavior_features
from promolift.features.purchase_dynamics import build_purchase_dynamics_features


class FeatureGroup(StrEnum):
    """Independently buildable feature groups, the unit of the feature ablation."""

    DEMOGRAPHICS = "demographics"
    PURCHASE_BEHAVIOR = "purchase_behavior"
    PRODUCT_MIX = "product_mix"
    PROMO_RESPONSIVENESS = "promo_responsiveness"
    PURCHASE_DYNAMICS = "purchase_dynamics"
    BASKET_STORE = "basket_store"
    CATEGORY_SPEND = "category_spend"


# The 19 features every model through Phase 10 was trained on.
BASE_GROUPS = (FeatureGroup.DEMOGRAPHICS, FeatureGroup.PURCHASE_BEHAVIOR, FeatureGroup.PRODUCT_MIX)
ALL_GROUPS = tuple(FeatureGroup)

_BUILDERS: dict[FeatureGroup, Callable[[datetime, Path | None], pl.LazyFrame]] = {
    FeatureGroup.DEMOGRAPHICS: build_demographic_features,
    FeatureGroup.PURCHASE_BEHAVIOR: build_purchase_behavior_features,
    FeatureGroup.PRODUCT_MIX: lambda _reference_date, base_dir: build_product_mix_features(
        base_dir
    ),
    FeatureGroup.PROMO_RESPONSIVENESS: build_promo_responsiveness_features,
    FeatureGroup.PURCHASE_DYNAMICS: build_purchase_dynamics_features,
    FeatureGroup.BASKET_STORE: build_basket_store_features,
    FeatureGroup.CATEGORY_SPEND: build_category_spend_features,
}


@dataclass(frozen=True)
class FeatureTable:
    """One row per client, and which feature columns each group contributed."""

    frame: pl.DataFrame
    groups: dict[FeatureGroup, tuple[str, ...]]

    def columns(self, groups: Sequence[FeatureGroup | str]) -> list[str]:
        """Feature columns of ``groups``, in table order."""
        wanted = {FeatureGroup(group) for group in groups}
        missing = wanted - set(self.groups)
        if missing:
            msg = f"feature table has no group(s) {sorted(missing)}"
            raise KeyError(msg)
        return [column for group in self.groups if group in wanted for column in self.groups[group]]

    def select(self, groups: Sequence[FeatureGroup | str]) -> pl.DataFrame:
        """``client_id`` plus the feature columns of ``groups``."""
        return self.frame.select("client_id", *self.columns(groups))


def build_feature_table(
    reference_date: datetime,
    base_dir: Path | None = None,
    groups: Sequence[FeatureGroup | str] = BASE_GROUPS,
) -> FeatureTable:
    """Build the client-level feature table from ``groups``.

    Each group is collected on its own and then joined, so peak memory is one
    purchases scan rather than every group's scan at once.

    Args:
        reference_date: Fixed point in time for recency/tenure features.
            Should be at or before the campaign/treatment date to avoid
            leakage. The entirety of purchases.csv is documented as
            pre-treatment purchase history, so its own max transaction date
            is a safe default (see notebooks/03_feature_engineering.ipynb).
        base_dir: Optional override of the raw data directory (for tests).
        groups: Feature groups to include; must include demographics, which
            defines the client universe the others are left-joined onto.

    Returns one row per client in clients.csv. Purchase-derived columns are
    null for any client with no transactions.

    Raises:
        ValueError: If demographics is missing, or two groups share a column name.
    """
    ordered = list(dict.fromkeys(FeatureGroup(group) for group in groups))
    if FeatureGroup.DEMOGRAPHICS not in ordered:
        msg = "feature groups must include demographics (the client universe)"
        raise ValueError(msg)
    ordered.remove(FeatureGroup.DEMOGRAPHICS)

    frame = _BUILDERS[FeatureGroup.DEMOGRAPHICS](reference_date, base_dir).collect()
    columns = {FeatureGroup.DEMOGRAPHICS: tuple(c for c in frame.columns if c != "client_id")}
    for group in ordered:
        part = _BUILDERS[group](reference_date, base_dir).collect()
        new_columns = tuple(c for c in part.columns if c != "client_id")
        clashes = set(frame.columns).intersection(new_columns)
        if clashes:
            msg = f"{group} repeats feature column(s) {sorted(clashes)}"
            raise ValueError(msg)
        columns[group] = new_columns
        frame = frame.join(part, on="client_id", how="left")
    return FeatureTable(frame=frame, groups=columns)
