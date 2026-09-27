"""Advanced uplift models: X-learner, DR-learner, causal forest (econml), uplift RF (causalml).

These libraries need a finite numeric matrix, so each model fits a
``NumericEncoder`` on its training data inside ``fit`` and reuses it at
prediction time -- no statistics from the scoring split ever leak in.

Treatment was randomized 50/50, so the propensity model is a constant
(``DummyClassifier(strategy="prior")``: the training treated share, ~0.5)
rather than a fitted classifier, which could only add noise.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from causalml.inference.tree import UpliftRandomForestClassifier
from econml.dml import CausalForestDML
from econml.dr import DRLearner
from econml.metalearners import XLearner
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyClassifier

from promolift.models.base import LGBM_PARAMS, LGBM_TUNABLE_KEYS, Params, merge_params
from promolift.models.baselines import lgbm_param_log
from promolift.models.preprocessing import NumericEncoder

# Starting points, not tuned: sized so each forest fits in under a minute on
# the 120k training clients (27s / 55s measured) with leaves large enough to
# average out outcome noise in a +3pp average effect.
CAUSAL_FOREST_PARAMS: Params = {"n_estimators": 200, "min_samples_leaf": 100, "cv": 2}
UPLIFT_RF_PARAMS: Params = {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 200}
_CAUSAL_FOREST_KEYS = frozenset(CAUSAL_FOREST_PARAMS) | {"max_samples", "max_depth"}
_UPLIFT_RF_KEYS = frozenset(UPLIFT_RF_PARAMS) | {"max_features"}
_DR_CROSS_FIT_FOLDS = 2


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

    def _model_params(self) -> Params:
        return {}

    def fit(self, features: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> Self:
        x = self._encoder.fit_transform(features)
        self._fit(x, np.asarray(treatment).astype(int), np.asarray(outcome).astype(int))
        return self

    def predict_uplift(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._predict(self._encoder.transform(features)), dtype=float).ravel()

    def params(self) -> Params:
        return {"model": self.name, "seed": self.seed, **self._model_params()}


class _EncodedLgbmModel(_EncodedModel):
    """Encoded model whose nuisance / effect models all share one LightGBM configuration."""

    def __init__(self, seed: int, overrides: Params | None = None) -> None:
        super().__init__(seed)
        self.lgbm_params = merge_params(LGBM_PARAMS, overrides, LGBM_TUNABLE_KEYS)

    def _regressor(self) -> LGBMRegressor:
        return LGBMRegressor(**self.lgbm_params, random_state=self.seed)

    def _model_params(self) -> Params:
        return lgbm_param_log(self.lgbm_params)


class XLearnerModel(_EncodedLgbmModel):
    """Per-arm outcome models, then imputed individual effects regressed on features."""

    name = "x_learner"

    def _fit(self, x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> None:
        self._model = XLearner(
            models=self._regressor(),
            cate_models=self._regressor(),
            propensity_model=_known_propensity(),
        )
        self._model.fit(outcome, treatment, X=x)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.effect(x)


class DRLearnerModel(_EncodedLgbmModel):
    """Doubly robust pseudo-outcomes (cross-fitted), regressed on features."""

    name = "dr_learner"

    def _fit(self, x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> None:
        self._model = DRLearner(
            model_propensity=_known_propensity(),
            model_regression=self._regressor(),
            model_final=self._regressor(),
            cv=_DR_CROSS_FIT_FOLDS,
            random_state=self.seed,
        )
        self._model.fit(outcome, treatment, X=x)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.effect(x)

    def _model_params(self) -> Params:
        return {"cv": _DR_CROSS_FIT_FOLDS, **super()._model_params()}


class CausalForestModel(_EncodedModel):
    """Generalized random forest on residualized outcome and treatment (econml).

    Overrides apply to the forest; the outcome nuisance model keeps the shared
    LightGBM defaults.
    """

    name = "causal_forest"

    def __init__(self, seed: int, overrides: Params | None = None) -> None:
        super().__init__(seed)
        self.forest_params = merge_params(CAUSAL_FOREST_PARAMS, overrides, _CAUSAL_FOREST_KEYS)

    def _fit(self, x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> None:
        self._model = CausalForestDML(
            model_y=LGBMRegressor(**LGBM_PARAMS, random_state=self.seed),
            model_t=_known_propensity(),
            discrete_treatment=True,
            random_state=self.seed,
            n_jobs=-1,
            **self.forest_params,
        )
        self._model.fit(outcome, treatment, X=x)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.effect(x)

    def _model_params(self) -> Params:
        return {f"forest_{key}": value for key, value in self.forest_params.items()}


class UpliftRandomForestModel(_EncodedModel):
    """Forest of trees split directly on treatment-effect divergence (causalml, KL criterion)."""

    name = "uplift_rf"
    _CONTROL = "0"

    def __init__(self, seed: int, overrides: Params | None = None) -> None:
        super().__init__(seed)
        self.forest_params = merge_params(UPLIFT_RF_PARAMS, overrides, _UPLIFT_RF_KEYS)

    def _fit(self, x: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> None:
        self._model = UpliftRandomForestClassifier(
            control_name=self._CONTROL,
            random_state=self.seed,
            n_jobs=-1,
            **self.forest_params,
        )
        self._model.fit(x, treatment.astype(str), outcome)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)

    def _model_params(self) -> Params:
        return {f"forest_{key}": value for key, value in self.forest_params.items()}
