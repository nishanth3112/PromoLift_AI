"""Which features drive the champion's predicted uplift: exact TreeSHAP contributions.

The champion is a class-transformation model: one LightGBM classifier of
z = 1 when (texted, bought) or (held back, didn't buy), scoring a client as
``uplift = 2 * P(z = 1) - 1``. LightGBM's own TreeSHAP (``pred_contrib``,
exact for trees) splits the classifier's log-odds margin ``f`` among the
features: ``f = f0 + sum(phi_j)``.

Uplift is a monotone function of ``f``, so a feature that raises ``f`` raises
the uplift, but ``phi_j`` is in log-odds. To read contributions in uplift units
(0.01 = one percentage point) each client's ``phi`` is rescaled by the one
factor that makes them add up exactly: ``base + sum(contributions) = uplift``.
Scores sit within a few points of zero, where the link is nearly linear, so the
rescaling is close to uniform and the ranking of features is the log-odds one.

SHAP describes what the model does, not what the features cause: correlated
features share credit, and a feature the model leans on may not move buying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit

from promolift.serving.scoring import ChampionModel

CLASS_TRANSFORMATION = "class_transformation"
DEFAULT_N_ROWS = 20_000
DEFAULT_SEED = 42
# A client whose margin equals the average margin has no contribution to rescale.
_MARGIN_EPSILON = 1e-9


class ExplanationError(ValueError):
    """The model can't be explained this way, or the features don't match it."""


@dataclass(frozen=True)
class Explanation:
    """Per-client feature contributions to the predicted uplift, in uplift units.

    ``features`` (the model's layout) and ``contributions`` share an index and
    the feature columns. ``base_value`` is the uplift at the model's average
    margin; ``base_value + contributions.sum(axis=1) == scores`` for every client.
    """

    features: pd.DataFrame
    contributions: pd.DataFrame
    base_value: float
    scores: np.ndarray

    @property
    def additivity_error(self) -> float:
        """Largest gap between a client's score and base value plus contributions."""
        rebuilt = self.base_value + self.contributions.to_numpy().sum(axis=1)
        return float(np.abs(rebuilt - self.scores).max())


def classifier_of(model: Any) -> LGBMClassifier:
    """The fitted LightGBM classifier inside a class-transformation model.

    Raises:
        ExplanationError: If ``model`` is any other kind of uplift model.
    """
    name = getattr(model, "name", None)
    if name != CLASS_TRANSFORMATION:
        raise ExplanationError(
            f"Only the {CLASS_TRANSFORMATION} model can be explained, got {name!r}: other "
            "models score uplift as a difference of predictions, which TreeSHAP can't split"
        )
    return model._model.estimator


def explain(model: Any, features: pd.DataFrame) -> Explanation:
    """TreeSHAP contributions of ``model``'s uplift score for each row of ``features``.

    Args:
        model: A fitted class-transformation model.
        features: Rows in the model's training layout (column order and
            categorical levels), e.g. ``FeatureContract.prepare`` output.

    Raises:
        ExplanationError: If the model isn't a class transformation or the
            columns differ from the ones it was fitted on.
    """
    classifier = classifier_of(model)
    expected = list(classifier.booster_.feature_name())
    if list(features.columns) != expected:
        raise ExplanationError(
            f"Feature columns {list(features.columns)} differ from the model's {expected}"
        )
    raw = np.asarray(classifier.predict(features, pred_contrib=True), dtype=float)
    margins_by_feature, base_margin = raw[:, :-1], float(raw[0, -1])

    margin = margins_by_feature.sum(axis=1) + base_margin
    scores = 2 * expit(margin) - 1
    base_value = 2 * expit(base_margin) - 1
    margin_shift = margin - base_margin
    # Where the margin hasn't moved, fall back to the link's slope at the base.
    slope_at_base = 2 * expit(base_margin) * (1 - expit(base_margin))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(
            np.abs(margin_shift) > _MARGIN_EPSILON,
            (scores - base_value) / margin_shift,
            slope_at_base,
        )
    contributions = pd.DataFrame(
        margins_by_feature * scale[:, None], index=features.index, columns=features.columns
    )
    return Explanation(features, contributions, base_value, scores)


def sample_rows(features: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    """A reproducible random sample of at most ``n_rows`` rows (all of them if fewer)."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    return features.sample(n=min(n_rows, len(features)), random_state=seed).reset_index(drop=True)


def explain_registered(
    champion: ChampionModel,
    features: pd.DataFrame,
    *,
    n_rows: int = DEFAULT_N_ROWS,
    seed: int = DEFAULT_SEED,
) -> Explanation:
    """Explain the registered champion on a sample of clients' features.

    The features are put into the champion's contract first, exactly as
    scoring does, so the explanation is of the model that actually scores.
    """
    sample = champion.contract.prepare(sample_rows(features, n_rows, seed))
    return explain(champion.model.unwrap_python_model().model, sample)
