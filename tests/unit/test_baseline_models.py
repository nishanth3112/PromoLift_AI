"""Unit tests for the baseline uplift models, on synthetic data with a known effect."""

import numpy as np
import pytest

from promolift.evaluation.ranking_metrics import ranking_metrics
from promolift.models.registry import available_models, build_model

_UPLIFT_MODELS = ("s_learner", "t_learner", "class_transformation")


@pytest.fixture(scope="module")
def data(sure_things_data: dict) -> dict:
    return sure_things_data


def _holdout_qini(name: str, data: dict) -> float:
    model = build_model(name).fit(*data["train"])
    x, t, y = data["holdout"]
    return ranking_metrics(model.predict_uplift(x), t, y).qini_auc


def test_registry_lists_every_baseline() -> None:
    assert {"random", "response", *_UPLIFT_MODELS} <= set(available_models())


def test_unknown_model_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="t_learner"):
        build_model("x_learner_typo")


@pytest.mark.parametrize("name", sorted(["random", "response", *_UPLIFT_MODELS]))
def test_every_model_scores_every_client(name: str, data: dict) -> None:
    model = build_model(name)

    returned = model.fit(*data["train"])
    scores = model.predict_uplift(data["holdout"][0])

    assert returned is model
    assert scores.shape == (4_000,)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("name", _UPLIFT_MODELS)
def test_uplift_models_find_the_persuadables(name: str, data: dict) -> None:
    assert _holdout_qini(name, data) > 0.05


def test_response_model_targets_sure_things_and_does_worse_than_random(data: dict) -> None:
    # The project's core claim in miniature: ranking by P(buy | treated)
    # targets clients who buy anyway, so it is worse than random targeting.
    assert _holdout_qini("response", data) < 0
    assert all(
        _holdout_qini(name, data) > _holdout_qini("response", data) for name in _UPLIFT_MODELS
    )


def test_models_are_reproducible_for_a_seed(data: dict) -> None:
    x = data["holdout"][0]
    first = build_model("t_learner", seed=3).fit(*data["train"]).predict_uplift(x)
    second = build_model("t_learner", seed=3).fit(*data["train"]).predict_uplift(x)

    np.testing.assert_array_equal(first, second)


def test_random_model_changes_with_seed(data: dict) -> None:
    x = data["holdout"][0]
    first = build_model("random", seed=1).fit(*data["train"]).predict_uplift(x)
    second = build_model("random", seed=2).fit(*data["train"]).predict_uplift(x)

    assert not np.array_equal(first, second)


def test_params_describe_the_model_for_logging() -> None:
    params = build_model("s_learner", seed=7).params()

    assert params["model"] == "s_learner"
    assert params["seed"] == 7
    assert "lgbm_n_estimators" in params
