"""Train-fitted encoding to a finite numeric matrix, for models that can't take NaN or categoricals.

LightGBM-based models use the model frame as-is. The econml/causalml forests
need a dense float matrix, so this is the per-model imputation step deferred
at feature engineering. Everything is learned from the training split only
and applied unchanged to other splits, so no statistics leak from val/test.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd


class NumericEncoder:
    """Median imputation + missing flags for numerics, one-hot for categoricals.

    A 0/1 ``<column>_missing`` flag is added for every numeric column with
    nulls in the training data: missingness often carries signal here (a null
    ``days_to_redeem`` means "never redeemed"), which imputation alone erases.
    The flag set is fixed at fit time, so every split gets identical columns.
    """

    def __init__(self) -> None:
        self._columns: list[str] | None = None
        self._medians: dict[str, float] = {}
        self._flagged: list[str] = []
        self._categories: dict[str, list[str]] = {}
        self.feature_names: list[str] = []

    def fit(self, features: pd.DataFrame) -> Self:
        """Learn medians, the missing-flag columns, and one-hot levels from training data."""
        self._columns = list(features.columns)
        self._categories = {
            column: list(features[column].cat.categories)
            for column in self._columns
            if isinstance(features[column].dtype, pd.CategoricalDtype)
        }
        numeric = [c for c in self._columns if c not in self._categories]
        # An all-null training column has no median; 0 keeps the matrix finite
        # and its missing flag (always 1 in train) carries whatever signal exists.
        self._medians = {
            c: float(m) if pd.notna(m := features[c].median()) else 0.0 for c in numeric
        }
        self._flagged = [c for c in numeric if features[c].isna().any()]
        self.feature_names = [
            *numeric,
            *(f"{c}_missing" for c in self._flagged),
            *(f"{c}_{level}" for c, levels in self._categories.items() for level in levels),
        ]
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Encode any split with the rules learned at fit time.

        Raises:
            RuntimeError: If called before ``fit``.
            ValueError: If the columns differ from those seen at fit time.
        """
        if self._columns is None:
            raise RuntimeError("NumericEncoder must be fit before transform")
        if list(features.columns) != self._columns:
            msg = f"Expected columns {self._columns}, got {list(features.columns)}"
            raise ValueError(msg)

        parts = [
            features[c].astype(float).fillna(median).to_numpy()
            for c, median in self._medians.items()
        ]
        parts += [features[c].isna().to_numpy(dtype=float) for c in self._flagged]
        parts += [
            (features[c] == level).to_numpy(dtype=float)
            for c, levels in self._categories.items()
            for level in levels
        ]
        return np.column_stack(parts)

    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        """``fit`` then ``transform`` on the same (training) data."""
        return self.fit(features).transform(features)
