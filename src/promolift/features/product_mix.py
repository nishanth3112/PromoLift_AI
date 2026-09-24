"""Client-level product mix features from purchases.csv joined to products.csv.

Unlike purchase_behavior.py, this operates at the line-item level -- no
transaction-level de-duplication is needed since product_id and product
attributes are already one row per line item, not repeated per transaction.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from promolift.data.loader import Dataset, load_lazy


def build_product_mix_features(base_dir: Path | None = None) -> pl.LazyFrame:
    """Build client-level product mix features.

    Rates are unweighted (a simple share of purchased line items), not
    weighted by quantity or spend.

    Returns one row per client_id with:
        - ``n_distinct_categories``: number of distinct ``level_2`` categories purchased.
          ``level_1`` has only 4 values total across the whole catalog (too
          coarse to give meaningful variance); ``level_2`` has 43, which
          gives a much more useful diversity signal.
        - ``alcohol_purchase_rate``: fraction of purchased line items that are alcohol products
        - ``own_trademark_rate``: fraction of purchased line items that are own-trademark products
    """
    purchases = load_lazy(Dataset.PURCHASES, base_dir).select("client_id", "product_id")
    products = load_lazy(Dataset.PRODUCTS, base_dir).select(
        "product_id", "level_2", "is_alcohol", "is_own_trademark"
    )

    joined = purchases.join(products, on="product_id", how="left")

    return joined.group_by("client_id").agg(
        pl.col("level_2").n_unique().alias("n_distinct_categories"),
        pl.col("is_alcohol").mean().alias("alcohol_purchase_rate"),
        pl.col("is_own_trademark").mean().alias("own_trademark_rate"),
    )
