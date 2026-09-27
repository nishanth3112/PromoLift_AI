"""Advanced uplift models: X-learner, DR-learner, causal forest (econml), uplift RF (causalml).

These libraries need a finite numeric matrix, so each model fits a
``NumericEncoder`` on its training data inside ``fit`` and reuses it at
prediction time -- no statistics from the scoring split ever leak in.

Treatment was randomized 50/50, so the propensity model is a constant
(``DummyClassifier(strategy="prior")``: the training treated share, ~0.5)
rather than a fitted classifier, which could only add noise.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
from causalml.inference.tree import UpliftRandomForestClassifier
from econml.dml import CausalForestDML
from econml.dr import DRLearner
from econml.metalearners import XLearner
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyClassifier

from promolift.models.base import LGBM_PARAMS
from promolift.models.preprocessing import NumericEncoder

# Starting points, not tuned: sized so each forest fits in under a minute on
# the 120k training clients (27s / 55s measured) with leaves large enough to
# average out outcome noise in a +3pp average effect.
CAUSAL_FOREST_PARAMS: dict[str, int] = {"n_estimators": 200, "min_samples_leaf": 100, "cv": 2}
UPLIFT_RF_PARAMS: dict[str, int] = {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 200}
_DR_CROSS_FIT_FOLDS = 2


def _regressor(seed: int) -> LGBMRegressor:
    return LGBMRegressor(**LGBM_PARAMS, random_state=seed)


def _known_propensity() -> DummyClassifier:
    return DummyClassifier(strategy="prior")


class _EncodedModel:
    """Fits a train-only ``NumericEncoder`` and hands the matrix to a library model."""

    name: str

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._encoder = NumericEncoder()

    def _fit(self, x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> None:
        raise NotImplementedError

    def _predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _model_params(self) -> dict[str, Any]:
        return {}

    def fit(self, features: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> Self:
        x = self._encoder.fit_transform(features)
        self._fit(x, np.asarray(treatment).astype(int), np.asarray(outcome).astype(int))
        return self

    def predict_uplift(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._predict(self._encoder.transform(features)), dtype=float).ravel()

    def params(self) -> dict[str, str | int | float | bool]:
        return {"model": self.name, "seed": self.seed, **self._model_params()}


def _lgbm_params() -> dict[str, Any]:
    return {f"lgbm_{key}": value for key, value in LGBM_PARAMS.items()}


class XLearnerModel(_EncodedModel):
    """Per-arm outcome models, then imputed individual effects regressed on features."""

    name = "x_learner"

    def _fit(self, x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> None:
        self._model = XLearner(
            models=_regressor(self.seed),
            cate_models=_regressor(self.seed),
            propensity_model=_known_propensity(),
        )
        self._model.fit(outcome, treatment, X=x)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.effect(x)

    def _model_params(self) -> dict[str, Any]:
        return _lgbm_params()


class DRLearnerModel(_EncodedModel):
    """Doubly robust pseudo-outcomes (cross-fitted), regressed on features."""

    name = "dr_learner"

    def _fit(self, x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> None:
        self._model = DRLearner(
            model_propensity=_known_propensity(),
            model_regression=_regressor(self.seed),
            model_final=_regressor(self.seed),
            cv=_DR_CROSS_FIT_FOLDS,
            random_state=self.seed,
        )
        self._model.fit(outcome, treatment, X=x)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.effect(x)

    def _model_params(self) -> dict[str, Any]:
        return {"cv": _DR_CROSS_FIT_FOLDS, **_lgbm_params()}


class CausalForestModel(_EncodedModel):
    """Generalized random forest on residualized outcome and treatment (econml)."""

    name = "causal_forest"

    def _fit(self, x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> None:
        self._model = CausalForestDML(
            model_y=_regressor(self.seed),
            model_t=_known_propensity(),
            discrete_treatment=True,
            random_state=self.seed,
            n_jobs=-1,
            **CAUSAL_FOREST_PARAMS,
        )
        self._model.fit(outcome, treatment, X=x)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.effect(x)

    def _model_params(self) -> dict[str, Any]:
        return {f"forest_{key}": value for key, value in CAUSAL_FOREST_PARAMS.items()}


class UpliftRandomForestModel(_EncodedModel):
    """Forest of trees split directly on treatment-effect divergence (causalml, KL criterion)."""

    name = "uplift_rf"
    _CONTROL = "0"

    def _fit(self, x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> None:
        self._model = UpliftRandomForestClassifier(
            control_name=self._CONTROL,
            random_state=self.seed,
            n_jobs=-1,
            **UPLIFT_RF_PARAMS,
        )
        self._model.fit(x, treatment.astype(str), outcome)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)

    def _model_params(self) -> dict[str, Any]:
        return {f"forest_{key}": value for key, value in UPLIFT_RF_PARAMS.items()}
