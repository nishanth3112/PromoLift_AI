"""The interface every uplift model implements, so one harness can train and score them all."""

from __future__ import annotations

from typing import Protocol, Self

import numpy as np
import pandas as pd

# One LightGBM configuration shared by every baseline, so differences between
# models come from how they estimate uplift, not from tuning. Deliberately
# conservative (small learning rate, large leaves): uplift signal is weak
# relative to outcome noise. deterministic + row-wise make refits identical
# for a seed -- models are reproduced by refitting, not stored.
LGBM_PARAMS: dict[str, float | int | bool] = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


class UpliftModel(Protocol):
    """Fits on (features, treatment, outcome); scores clients by predicted uplift."""

    name: str

    def fit(self, features: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> Self:
        """Fit on training clients and return self."""
        ...

    def predict_uplift(self, features: pd.DataFrame) -> np.ndarray:
        """Ranking score per client; higher = target first."""
        ...

    def params(self) -> dict[str, str | int | float | bool]:
        """Flat, MLflow-loggable description of the model's configuration."""
        ...
