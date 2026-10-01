"""Model-ready data: the feature table joined to one split's clients and labels.

The test-split lock is enforced here, at the data level, so test rows can't
even be loaded outside the final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from promolift.data.loader import Dataset, load_lazy
from promolift.data.split import Split, assert_split_access, load_split
from promolift.features.build import build_feature_table
from promolift.validation.referential_integrity import transaction_date_integrity


@dataclass
class ModelFrame:
    """One split's clients, in ``client_id`` order, as model inputs."""

    split: str
    client_ids: pl.Series
    features: pd.DataFrame
    treatment: np.ndarray
    outcome: np.ndarray


def build_model_features(base_dir: Path | None = None) -> pl.DataFrame:
    """The full client feature table at the end of the purchase window.

    Scans ``purchases.csv`` (~29s), so build it once per process and pass it
    to ``model_frame`` for every split.
    """
    reference_date = transaction_date_integrity(base_dir).max_date
    return build_feature_table(reference_date, base_dir).frame


def _to_pandas(frame: pl.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    numeric = frame.with_columns(pl.col(pl.Boolean).cast(pl.Int8)).drop(list(categories))
    result = numeric.to_pandas()
    for column, levels in categories.items():
        result[column] = pd.Categorical(frame[column].to_list(), categories=levels)
    return result[frame.columns]


def model_frame(
    features: pl.DataFrame,
    split: Split | str,
    *,
    assignment: pl.DataFrame | None = None,
    labels: pl.DataFrame | None = None,
    final_evaluation: bool = False,
) -> ModelFrame:
    """Features, treatment, and outcome for every client in ``split``.

    String columns become pandas categoricals whose category set comes from
    the *whole* feature table, not the split: a category missing from one
    split would otherwise shift the codes, and a tree model would silently
    read val's "M" as train's "U". Booleans become 0/1; nulls stay missing
    (LightGBM handles them natively).

    Args:
        features: Output of ``build_model_features`` (one row per client).
        split: Which split to load; ``test`` requires ``final_evaluation=True``.
        assignment: ``client_id -> split``; defaults to the canonical split file.
        labels: ``uplift_train`` rows; defaults to the raw file.

    Raises:
        ValueError: If any client in the split has no feature row.
    """
    assert_split_access(split, final_evaluation=final_evaluation)
    split_name = Split(split).value
    assignment = assignment if assignment is not None else load_split()
    labels = labels if labels is not None else load_lazy(Dataset.UPLIFT_TRAIN).collect()

    clients = (
        assignment.filter(pl.col("split") == split_name)
        .join(labels.select("client_id", "treatment_flg", "target"), on="client_id")
        .sort("client_id")
    )
    joined = clients.join(
        features.with_columns(pl.lit(True).alias("_has_features")), on="client_id", how="left"
    )
    n_missing = joined["_has_features"].null_count()
    if n_missing:
        raise ValueError(f"{n_missing} {split_name} client(s) have no features")

    feature_columns = [c for c in features.columns if c != "client_id"]
    categories = {
        column: sorted(features[column].drop_nulls().unique().to_list())
        for column in feature_columns
        if features.schema[column] == pl.String
    }
    return ModelFrame(
        split=split_name,
        client_ids=joined["client_id"],
        features=_to_pandas(joined.select(feature_columns), categories),
        treatment=joined["treatment_flg"].to_numpy(),
        outcome=joined["target"].to_numpy(),
    )
