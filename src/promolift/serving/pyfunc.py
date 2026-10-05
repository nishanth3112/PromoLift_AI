"""The packaged uplift model: a fitted model, the input it accepts, and its training reference.

``FeatureContract`` pins the columns a model was trained on, in order, and
the levels of each categorical column. ``prepare`` turns a scoring frame into
exactly the training layout -- or refuses it. A missing or extra column, text
in a numeric column, or an unseen category would otherwise score silently
wrong: a tree model reads an unknown category as missing, and a shifted
column order mixes features up.

Models are logged with one canonical input format (``FeatureContract.to_input``):
numeric columns as float64, categorical columns as strings. MLflow's
signature check refuses implicit int -> float conversion, so batch scoring
must build that format; ``prepare`` then restores the training dtypes.

The wrapper also carries the training clients' drift reference
(``monitoring.drift.build_reference``), so whoever loads the model can check
a scoring batch against the data it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd

from promolift.models.base import UpliftModel


class ContractError(ValueError):
    """A scoring frame doesn't match the columns, types, or categories the model was trained on."""


@dataclass(frozen=True)
class FeatureContract:
    """Feature columns in training order, and the allowed levels of each categorical column."""

    columns: tuple[str, ...]
    categories: dict[str, tuple[str, ...]]

    @classmethod
    def from_frame(cls, features: pd.DataFrame) -> FeatureContract:
        """The contract of a training frame (``ModelFrame.features``).

        Raises:
            ContractError: If a column is neither numeric, boolean, nor categorical.
        """
        categories = {}
        for column in features.columns:
            dtype = features[column].dtype
            if isinstance(dtype, pd.CategoricalDtype):
                categories[column] = tuple(str(level) for level in dtype.categories)
            elif not (pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)):
                raise ContractError(f"Column {column!r} has unsupported dtype {dtype}")
        return cls(tuple(features.columns), categories)

    def as_dict(self) -> dict:
        return {
            "columns": list(self.columns),
            "categories": {column: list(levels) for column, levels in self.categories.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> FeatureContract:
        return cls(
            tuple(data["columns"]),
            {column: tuple(levels) for column, levels in data["categories"].items()},
        )

    def _check_columns(self, frame: pd.DataFrame) -> None:
        missing = [c for c in self.columns if c not in frame.columns]
        extra = [c for c in frame.columns if c not in self.columns]
        if missing or extra:
            raise ContractError(
                f"Columns don't match the contract: missing {missing}, extra {extra}"
            )

    def to_input(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The canonical logged-model input: numeric -> float64, categorical -> str (nulls kept)."""
        self._check_columns(frame)
        result = {}
        for column in self.columns:
            values = frame[column]
            if column in self.categories:
                text = values.astype(object).where(values.notna(), None)
                result[column] = text.map(lambda v: None if v is None else str(v))
            else:
                result[column] = pd.to_numeric(values, errors="raise").astype("float64")
        return pd.DataFrame(result, index=frame.index)

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """``frame`` in the training layout: contract column order, float numerics, fixed levels.

        Raises:
            ContractError: On missing or extra columns, non-numeric values in a
                numeric column, or categories the model never saw.
        """
        self._check_columns(frame)
        result = {}
        for column in self.columns:
            values = frame[column]
            if column in self.categories:
                levels = self.categories[column]
                text = values.astype(object).where(values.notna(), None)
                unknown = sorted({str(v) for v in text.dropna()} - set(levels))
                if unknown:
                    raise ContractError(f"Column {column!r} has unknown categories {unknown}")
                result[column] = pd.Categorical(
                    [v if v is None else str(v) for v in text], categories=list(levels)
                )
            else:
                dtype = values.dtype
                if not (pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)):
                    raise ContractError(f"Column {column!r} must be numeric, got {dtype}")
                result[column] = values.astype("float64").to_numpy()
        return pd.DataFrame(result, index=frame.index)


class UpliftPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper: checks the input against the contract, returns one uplift score per row.

    ``drift_reference`` is the training clients' binned feature distribution;
    versions registered before it existed load with ``None``.
    """

    def __init__(
        self, model: UpliftModel, contract: FeatureContract, drift_reference: dict | None = None
    ) -> None:
        self.model = model
        self.contract = contract
        self.drift_reference = drift_reference

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> np.ndarray:
        """Uplift score per row of ``model_input`` (canonical input or the training layout)."""
        return np.asarray(
            self.model.predict_uplift(self.contract.prepare(model_input)), dtype=float
        )
