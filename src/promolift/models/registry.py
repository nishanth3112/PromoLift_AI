"""Name -> model factory, so scripts select models by name and every model is built the same way."""

from __future__ import annotations

from collections.abc import Callable

from promolift.models.advanced import (
    CausalForestModel,
    DRLearnerModel,
    UpliftRandomForestModel,
    XLearnerModel,
)
from promolift.models.base import UpliftModel
from promolift.models.baselines import (
    ClassTransformationModel,
    RandomModel,
    ResponseModel,
    SLearner,
    TLearner,
)

DEFAULT_SEED = 42

_REGISTRY: dict[str, Callable[[int], UpliftModel]] = {
    model.name: model
    for model in (
        RandomModel,
        ResponseModel,
        SLearner,
        TLearner,
        ClassTransformationModel,
        XLearnerModel,
        DRLearnerModel,
        CausalForestModel,
        UpliftRandomForestModel,
    )
}


def available_models() -> list[str]:
    """Registered model names, in registration order."""
    return list(_REGISTRY)


def build_model(name: str, *, seed: int = DEFAULT_SEED) -> UpliftModel:
    """A fresh, unfitted model.

    Raises:
        ValueError: If ``name`` isn't registered.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model {name!r}; available: {available_models()}")
    return _REGISTRY[name](seed)
