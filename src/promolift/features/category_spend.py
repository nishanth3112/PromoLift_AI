"""Client-level spend shares across ``level_2`` product categories.

``n_distinct_categories`` (product_mix.py) says how many categories a client
buys; this says *where the money goes*. Spend is very concentrated -- the 15
largest of the 43 categories hold ~95% of it -- so the top ``top_k`` get their
own share column and the long tail is pooled into ``cat_share_other``. The top
categories are picked by total spend across all clients, which uses no labels
or splits, so it can't leak.

Line spend is ``trn_sum_from_red`` when present -- the full price, recorded
only on transactions that spent points -- else ``trn_sum_from_iss``, the
amount paid in money: both mean the line's full price before any points
discount. Lines whose product has no ``level_2`` (~0.02% of spend) are
dropped.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from promolift.data.loader import Dataset, load_lazy
from promolift.features.transactions import load_line_items

TOP_K = 15
SHARE_PREFIX = "cat_share_"
OTHER_SHARE = f"{SHARE_PREFIX}other"


def build_category_spend_features(
    reference_date: datetime, base_dir: Path | None = None, top_k: int = TOP_K
) -> pl.LazyFrame:
    """Build client-level category spend shares and spend concentration.

    Collects a client x category spend table (at most 43 rows per client) to
    pick the top categories, so unlike the other builders this one is eager
    underneath; the result is returned lazy for a uniform interface.

    Returns one row per client_id with:
        - ``cat_share_<level_2>`` for the ``top_k`` categories with the most
          total spend (ties broken by category id): share of the client's spend
        - ``cat_share_other``: share of spend in every other category
        - ``category_spend_entropy``: Shannon entropy (nats) of the client's
          spend over all categories -- 0 for a single-category client
        All null for a client whose total spend is 0.
    """
    products = load_lazy(Dataset.PRODUCTS, base_dir).select("product_id", "level_2")
    spend = (
        load_line_items(
            reference_date, ["product_id", "trn_sum_from_iss", "trn_sum_from_red"], base_dir
        )
        .join(products, on="product_id", how="left")
        .filter(pl.col("level_2").is_not_null())
        .group_by("client_id", "level_2")
        .agg(
            pl.coalesce(pl.col("trn_sum_from_red").cast(pl.Float64), pl.col("trn_sum_from_iss"))
            .sum()
            .alias("_spend")
        )
        .collect()
    )

    top = (
        spend.group_by("level_2")
        .agg(pl.col("_spend").sum())
        .sort(["_spend", "level_2"], descending=[True, False])
        .head(top_k)["level_2"]
        .to_list()
    )
    shares = spend.with_columns(
        (pl.col("_spend") / pl.col("_spend").sum().over("client_id")).alias("_share")
    ).with_columns(
        # A zero-spend client divides 0 by 0; its shares are undefined, not NaN.
        pl.col("_share").fill_nan(None)
    )

    summary = shares.group_by("client_id").agg(
        pl.col("_spend").sum().alias("_total"),
        pl.col("_share").filter(~pl.col("level_2").is_in(top)).sum().alias(OTHER_SHARE),
        (-(pl.col("_share") * pl.col("_share").log()).filter(pl.col("_share") > 0))
        .sum()
        .alias("category_spend_entropy"),
    )
    wide = shares.filter(pl.col("level_2").is_in(top)).pivot(
        on="level_2", index="client_id", values="_share"
    )
    share_columns = [f"{SHARE_PREFIX}{category}" for category in top]
    wide = wide.rename({category: f"{SHARE_PREFIX}{category}" for category in top})

    has_spend = pl.col("_total") > 0
    return (
        summary.join(wide, on="client_id", how="left")
        .with_columns(
            pl.when(has_spend).then(pl.col(column).fill_null(0.0)).otherwise(None)
            for column in [*share_columns, OTHER_SHARE, "category_spend_entropy"]
        )
        .select("client_id", *share_columns, OTHER_SHARE, "category_spend_entropy")
        .lazy()
    )
