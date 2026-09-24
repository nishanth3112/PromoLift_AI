"""Combines all client-level feature groups into a single feature table.

Starts from the full client universe in clients.csv (not just clients in
uplift_train) so any client can be scored later, not just those in the
training experiment.

Nulls from purchase- and product-mix-derived columns (e.g. a client with no
transactions) are left as-is rather than imputed here. Imputation is a
modeling-pipeline concern, not a feature-store concern: tree-based models
(LightGBM, CatBoost) handle nulls natively, while linear-based causal
learners (econml, causalml) will need their own explicit imputation step,
which may differ by model.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from promolift.features.demographics import build_demographic_features
from promolift.features.product_mix import build_product_mix_features
from promolift.features.purchase_behavior import build_purchase_behavior_features


def build_feature_table(reference_date: datetime, base_dir: Path | None = None) -> pl.DataFrame:
    """Build the full client-level feature table.

    Args:
        reference_date: Fixed point in time for recency/tenure features.
            Should be at or before the campaign/treatment date to avoid
            leakage. The entirety of purchases.csv is documented as
            pre-treatment purchase history, so its own max transaction date
            is a safe default (see notebooks/03_feature_engineering.ipynb).
        base_dir: Optional override of the raw data directory (for tests).

    Returns one row per client in clients.csv, with demographic, purchase
    behavior, and product mix columns joined in. Purchase- and product-mix-
    derived columns are null for any client with no transactions (none exist
    in the current data, but the left join is defensive against future data
    with new/never-purchased customers).
    """
    demographics = build_demographic_features(reference_date, base_dir)
    purchase_behavior = build_purchase_behavior_features(reference_date, base_dir)
    product_mix = build_product_mix_features(base_dir)

    return (
        demographics.join(purchase_behavior, on="client_id", how="left")
        .join(product_mix, on="client_id", how="left")
        .collect()
    )
