"""Unit tests for the advanced uplift models, on the synthetic sure-things experiment."""

import numpy as np
import pytest

from promolift.evaluation.ranking_metrics import ranking_metrics
from promolift.models.registry import build_model

_ADVANCED = ("x_learner", "dr_learner", "causal_forest", "uplift_rf")


@pytest.fixture(scope="module")
def fitted(sure_things_data: dict) -> dict[str, dict]:
    # Fit each model once and share it: the forests take seconds per fit.
    x, t, y = sure_things_data["holdout"]
    results = {}
    for name in (*_ADVANCED, "response"):
        model = build_model(name)
        returned = model.fit(*sure_things_data["train"])
        scores = model.predict_uplift(x)
        results[name] = {
            "model": model,
            "returned": returned,
            "scores": scores,
            "qini": ranking_metrics(scores, t, y).qini_auc,
        }
    return results


@pytest.mark.parametrize("name", _ADVANCED)
def test_advanced_models_find_the_persuadables(name: str, fitted: dict) -> None:
    assert fitted[name]["qini"] > 0.05
    assert fitted[name]["qini"] > fitted["response"]["qini"]


@pytest.mark.parametrize("name", _ADVANCED)
def test_scores_every_client_with_a_finite_value(name: str, fitted: dict) -> None:
    assert fitted[name]["returned"] is fitted[name]["model"]
    assert fitted[name]["scores"].shape == (4_000,)
    assert np.isfinite(fitted[name]["scores"]).all()


@pytest.mark.parametrize("name", ["causal_forest", "uplift_rf"])
def test_parallel_forests_are_reproducible_for_a_seed(name: str, sure_things_data: dict) -> None:
    x = sure_things_data["holdout"][0]
    first = build_model(name, seed=3).fit(*sure_things_data["train"]).predict_uplift(x)
    second = build_model(name, seed=3).fit(*sure_things_data["train"]).predict_uplift(x)

    np.testing.assert_allclose(first, second)


def test_params_describe_forest_settings_for_logging() -> None:
    params = build_model("causal_forest", seed=5).params()

    assert params["model"] == "causal_forest"
    assert params["seed"] == 5
    assert "forest_n_estimators" in params


def test_forest_overrides_apply_to_the_forest() -> None:
    params = build_model("uplift_rf", overrides={"max_depth": 6}).params()

    assert params["forest_max_depth"] == 6
    assert params["forest_min_samples_leaf"] == 200


def test_forest_overrides_reject_lightgbm_keys() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        build_model("causal_forest", overrides={"learning_rate": 0.1})
