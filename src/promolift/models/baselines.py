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

from promolift.models.base import LGBM_PARAMS


def _classifier(seed: int) -> LGBMClassifier:
    return LGBMClassifier(**LGBM_PARAMS, random_state=seed)


def _lgbm_params() -> dict[str, float | int | bool]:
    return {f"lgbm_{key}": value for key, value in LGBM_PARAMS.items()}


class RandomModel:
    """Uniformly random scores: a sanity check that should land inside the noise floor."""

    name = "random"

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def fit(self, features: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> Self:
        return self

    def predict_uplift(self, features: pd.DataFrame) -> np.ndarray:
        return np.random.default_rng(self.seed).random(len(features))

    def params(self) -> dict[str, str | int | float | bool]:
        return {"model": self.name, "seed": self.seed}


class ResponseModel:
    """P(buy | contacted), trained on treated clients only -- the classic campaign model.

    Its score is a purchase probability, not an uplift: ranking by it targets
    clients likely to buy, whether or not the SMS changes their behavior.
    """

    name = "response"

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._classifier = _classifier(seed)

    def fit(self, features: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> Self:
        treated = np.asarray(treatment) == 1
        self._classifier.fit(features[treated], np.asarray(outcome)[treated])
        return self

    def predict_uplift(self, features: pd.DataFrame) -> np.ndarray:
        return self._classifier.predict_proba(features)[:, 1]

    def params(self) -> dict[str, str | int | float | bool]:
        return {"model": self.name, "seed": self.seed, "trained_on": "treated", **_lgbm_params()}


class _SkliftModel:
    """Adapter from scikit-uplift's ``fit(X, y, treatment)`` to ``UpliftModel``."""

    name: str

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._model = self._build(seed)

    def _build(self, seed: int) -> SoloModel | TwoModels | ClassTransformation:
        raise NotImplementedError

    def fit(self, features: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> Self:
        self._model.fit(features, np.asarray(outcome), np.asarray(treatment))
        return self

    def predict_uplift(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(features), dtype=float)

    def params(self) -> dict[str, str | int | float | bool]:
        return {"model": self.name, "seed": self.seed, **_lgbm_params()}


class SLearner(_SkliftModel):
    """One classifier with treatment as a feature; uplift = P(y | t=1) - P(y | t=0)."""

    name = "s_learner"

    def _build(self, seed: int) -> SoloModel:
        return SoloModel(estimator=_classifier(seed), method="dummy")


class TLearner(_SkliftModel):
    """Separate classifiers for treated and control; uplift = the difference of their P(y)."""

    name = "t_learner"

    def _build(self, seed: int) -> TwoModels:
        return TwoModels(
            estimator_trmnt=_classifier(seed), estimator_ctrl=_classifier(seed), method="vanilla"
        )


class ClassTransformationModel(_SkliftModel):
    """One classifier on z = 1 when (treated, bought) or (control, didn't); uplift = 2P(z) - 1.

    Valid only when P(treated) = 0.5 for every client -- true here: a 50/50
    randomized experiment, 50.0% treated in every split.
    """

    name = "class_transformation"

    def _build(self, seed: int) -> ClassTransformation:
        return ClassTransformation(estimator=_classifier(seed))
