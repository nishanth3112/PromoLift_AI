"""Verify the causal-ML stack is importable and (optionally) actually works.

Usage:
    uv run python scripts/check_env.py          # import check only
    uv run python scripts/check_env.py --fit    # also run tiny mini-fits
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from importlib import import_module
from importlib.metadata import version

_PACKAGES = [
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "pandera",
    "sklearn",
    "lightgbm",
    "catboost",
    "optuna",
    "shap",
    "sklift",
    "causalml",
    "econml",
    "dowhy",
    "ortools",
    "mlflow",
    "matplotlib",
    "seaborn",
    "duckdb",
    "yaml",
]

_DISTRIBUTION_NAME = {
    "sklearn": "scikit-learn",
    "sklift": "scikit-uplift",
    "yaml": "pyyaml",
}


def check_imports() -> bool:
    """Import every package in the stack and print its version. Returns True if all succeed."""
    all_ok = True
    for module_name in _PACKAGES:
        dist_name = _DISTRIBUTION_NAME.get(module_name, module_name)
        try:
            import_module(module_name)
            print(f"  OK  {dist_name:<16} {version(dist_name)}")
        except Exception as e:  # report every failure, do not stop at the first
            all_ok = False
            print(f"FAIL  {dist_name:<16} {type(e).__name__}: {e}")
    return all_ok


def run_mini_fits() -> bool:
    """Fit each causal/optimization library on tiny synthetic data to confirm it actually works."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=(n, 3))
    treatment = rng.integers(0, 2, size=n)
    y = treatment * 0.5 + x[:, 0] + rng.normal(scale=0.1, size=n)

    all_ok = True

    def _run(name: str, fn: callable) -> None:
        nonlocal all_ok
        try:
            fn()
            print(f"  OK  {name}")
        except Exception as e:
            all_ok = False
            print(f"FAIL  {name}: {type(e).__name__}: {e}")

    def _fit_sklift() -> None:
        from sklearn.linear_model import LogisticRegression
        from sklift.models import SoloModel

        SoloModel(LogisticRegression()).fit(x, y > y.mean(), treatment)

    def _fit_causalml() -> None:
        from causalml.inference.meta import BaseXRegressor
        from sklearn.linear_model import LinearRegression

        BaseXRegressor(learner=LinearRegression()).fit(x, treatment, y)

    def _fit_econml() -> None:
        from econml.dml import LinearDML

        LinearDML(discrete_treatment=True).fit(y, treatment, X=x)

    def _fit_dowhy() -> None:
        import pandas as pd
        from dowhy import CausalModel

        df = pd.DataFrame(x, columns=["x0", "x1", "x2"])
        df["treatment"] = treatment
        df["y"] = y
        model = CausalModel(
            data=df,
            treatment="treatment",
            outcome="y",
            common_causes=["x0", "x1", "x2"],
        )
        model.identify_effect(proceed_when_unidentifiable=True)

    def _solve_ortools() -> None:
        from ortools.linear_solver import pywraplp

        solver = pywraplp.Solver.CreateSolver("CBC")
        var = solver.NumVar(0, 10, "x")
        solver.Maximize(var)
        solver.Solve()

    def _run_mlflow() -> None:
        import mlflow

        with tempfile.TemporaryDirectory() as tmp_dir:
            mlflow.set_tracking_uri(f"sqlite:///{tmp_dir}/mlflow.db")
            with mlflow.start_run():
                mlflow.log_metric("dummy_metric", 1.0)

    _run("scikit-uplift SoloModel", _fit_sklift)
    _run("causalml BaseXRegressor", _fit_causalml)
    _run("econml LinearDML", _fit_econml)
    _run("dowhy CausalModel.identify_effect", _fit_dowhy)
    _run("ortools CBC solver", _solve_ortools)
    _run("mlflow run + log_metric", _run_mlflow)

    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true", help="also run tiny mini-fits")
    args = parser.parse_args()

    print("Checking imports...")
    imports_ok = check_imports()

    fits_ok = True
    if args.fit:
        print("\nRunning mini-fits...")
        fits_ok = run_mini_fits()

    if imports_ok and fits_ok:
        print("\nAll checks passed.")
        return 0
    print("\nSome checks FAILED (see above).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
