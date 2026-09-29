"""Baseline uplift models: a random floor, the traditional response model, and meta-learners.

The response model is the incumbent this project argues against: it predicts
who will buy when contacted, and so targets "sure things" who buy anyway. The
S-learner, T-learner, and class transformation wrap scikit-uplift's reference
implementations around the shared LightGBM configuration.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklift.models import ClassTransformation, SoloModel, TwoModels

from promolift.models.base import LGBM_PARAMS, LGBM_TUNABLE_KEYS, Params, merge_params


def lgbm_param_log(lgbm_params: Params) -> Params:
    """LightGBM settings under ``lgbm_`` names, for MLflow."""
    return {f"lgbm_{key}": value for key, value in lgbm_params.items()}


class RandomModel:
    """Uniformly random scores: a sanity check that should land inside the noise floor."""

    name = "random"

    def __init__(self, seed: int, overrides: Params | None = None) -> None:
        merge_params({}, overrides, allowed=frozenset())
        self.seed = seed

    def fit(self, features: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> Self:
        return self

    def predict_uplift(self, features: pd.DataFrame) -> np.ndarray:
        return np.random.default_rng(self.seed).random(len(features))

    def params(self) -> Params:
        return {"model": self.name, "seed": self.seed}


class _LgbmModel:
    """Shared LightGBM configuration, optionally overridden (e.g. by tuning)."""

    name: str

    def __init__(self, seed: int, overrides: Params | None = None) -> None:
        self.seed = seed
        self.lgbm_params = merge_params(LGBM_PARAMS, overrides, LGBM_TUNABLE_KEYS)

    def _classifier(self) -> LGBMClassifier:
        return LGBMClassifier(**self.lgbm_params, random_state=self.seed)

    def params(self) -> Params:
        return {"model": self.name, "seed": self.seed, **lgbm_param_log(self.lgbm_params)}


class ResponseModel(_LgbmModel):
    """P(buy | contacted), trained on treated clients only -- the classic campaign model.

    Its score is a purchase probability, not an uplift: ranking by it targets
    clients likely to buy, whether or not the SMS changes their behavior.
    """

    name = "response"

    def fit(self, features: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> Self:
        treated = np.asarray(treatment) == 1
        self._model = self._classifier().fit(features[treated], np.asarray(outcome)[treated])
        return self

    def predict_uplift(self, features: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(features)[:, 1]

    def params(self) -> Params:
        return {**super().params(), "trained_on": "treated"}


class _SkliftModel(_LgbmModel):
    """Adapter from scikit-uplift's ``fit(X, y, treatment)`` to ``UpliftModel``."""

    def _build(self) -> SoloModel | TwoModels | ClassTransformation:
        raise NotImplementedError

    def fit(self, features: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> Self:
        self._model = self._build()
        self._model.fit(features, np.asarray(outcome), np.asarray(treatment))
        return self

    def predict_uplift(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(features), dtype=float)


class SLearner(_SkliftModel):
    """One classifier with treatment as a feature; uplift = P(y | t=1) - P(y | t=0)."""

    name = "s_learner"

    def _build(self) -> SoloModel:
        return SoloModel(estimator=self._classifier(), method="dummy")


class TLearner(_SkliftModel):
    """Separate classifiers for treated and control; uplift = the difference of their P(y)."""

    name = "t_learner"

    def _build(self) -> TwoModels:
        return TwoModels(
            estimator_trmnt=self._classifier(), estimator_ctrl=self._classifier(), method="vanilla"
        )


class ClassTransformationModel(_SkliftModel):
    """One classifier on z = 1 when (treated, bought) or (control, didn't); uplift = 2P(z) - 1.

    Valid only when P(treated) = 0.5 for every client -- true here: a 50/50
    randomized experiment, 50.0% treated in every split.
    """

    name = "class_transformation"

    def _build(self) -> ClassTransformation:
        return ClassTransformation(estimator=self._classifier())
