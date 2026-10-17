"""Unit tests for the TreeSHAP contributions of the class-transformation model."""

import numpy as np
import pandas as pd
import pytest

from promolift.explainability.shap_values import (
    ExplanationError,
    classifier_of,
    explain,
    explain_registered,
    sample_rows,
)
from promolift.models.registry import build_model
from promolift.serving.scoring import load_champion

# --- contributions in uplift units -----------------------------------------------------------


def test_contributions_add_up_to_each_clients_score(explained_model) -> None:
    explanation = explain(explained_model.model, explained_model.features)

    assert explanation.additivity_error < 1e-9
    assert explanation.contributions.shape == explained_model.features.shape
    assert list(explanation.contributions.columns) == list(explained_model.features.columns)
    assert explanation.contributions.index.equals(explained_model.features.index)


def test_the_scores_are_the_models_uplift_predictions(explained_model) -> None:
    explanation = explain(explained_model.model, explained_model.features)

    expected = explained_model.model.predict_uplift(explained_model.features)
    np.testing.assert_allclose(explanation.scores, expected, atol=1e-9)


def test_the_base_value_is_the_uplift_of_an_average_client(explained_model) -> None:
    explanation = explain(explained_model.model, explained_model.features)

    assert explanation.base_value == pytest.approx(explanation.scores.mean(), abs=0.03)


def test_the_feature_with_the_real_effect_dominates(explained_model) -> None:
    contributions = explain(explained_model.model, explained_model.features).contributions

    importance = contributions.abs().mean()
    assert importance.idxmax() == "b"
    # The small model fits a little noise, so require a clear margin, not an absolute one.
    assert importance["b"] > 2 * importance["noise"]
    assert importance["b"] > 2 * importance["segment"]


def test_contributions_push_the_way_the_data_does(explained_model) -> None:
    explanation = explain(explained_model.model, explained_model.features)

    persuadable = explained_model.features["b"] < 0.5
    # Persuadables gain from the SMS, sure things don't: b raises uplift when low.
    assert explanation.contributions.loc[persuadable, "b"].mean() > 0
    assert explanation.contributions.loc[~persuadable, "b"].mean() < 0


def test_a_client_at_the_average_margin_keeps_exact_additivity(explained_model) -> None:
    # Duplicating one row makes identical margins; the rescaling must stay finite.
    row = explained_model.features.head(1)
    explanation = explain(explained_model.model, pd.concat([row, row], ignore_index=True))

    assert np.isfinite(explanation.contributions.to_numpy()).all()
    assert explanation.additivity_error < 1e-9


# --- what can be explained ----------------------------------------------------------------


def test_other_uplift_models_are_refused(sure_things_data) -> None:
    features, treatment, outcome = sure_things_data["train"]
    s_learner = build_model(
        "s_learner", seed=0, overrides={"n_estimators": 10, "min_child_samples": 50}
    ).fit(features, treatment, outcome)

    with pytest.raises(ExplanationError, match="Only the class_transformation model"):
        classifier_of(s_learner)
    with pytest.raises(ExplanationError, match="Only the class_transformation model"):
        explain(s_learner, features)


def test_features_in_another_layout_are_refused(explained_model) -> None:
    shuffled = explained_model.features[["noise", "b", "segment"]]

    with pytest.raises(ExplanationError, match="differ from the model's"):
        explain(explained_model.model, shuffled)


# --- sampling -----------------------------------------------------------------------------


def test_sampling_is_reproducible_and_capped(explained_model) -> None:
    features = explained_model.features

    first = sample_rows(features, 100, seed=1)
    again = sample_rows(features, 100, seed=1)
    other = sample_rows(features, 100, seed=2)

    assert len(first) == 100
    assert first.equals(again)
    assert not first.equals(other)
    assert len(sample_rows(features, 10**9, seed=1)) == len(features)
    with pytest.raises(ValueError, match="positive"):
        sample_rows(features, 0, seed=1)


# --- the registered champion --------------------------------------------------------------


def test_explains_the_registered_champion_in_its_own_contract(
    scoring_champion, client_features, make_training_frame
) -> None:
    champion = load_champion(scoring_champion.name, scoring_champion.config)
    frame = make_training_frame(client_features, 60, 1)

    explanation = explain_registered(champion, frame, n_rows=40, seed=3)

    assert len(explanation.scores) == 40
    assert list(explanation.contributions.columns) == list(champion.contract.columns)
    assert explanation.additivity_error < 1e-9
    # The same scores the deployed model gives these clients, fed as scoring feeds it.
    sample = sample_rows(frame, 40, 3)
    deployed = champion.model.predict(champion.contract.to_input(sample))
    np.testing.assert_allclose(explanation.scores, deployed, atol=1e-9)
