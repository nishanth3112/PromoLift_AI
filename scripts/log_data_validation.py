"""Log the experiment-validity checks on the real X5 data as an MLflow run.

Records the randomization evidence (naive ATE with CI, covariate balance,
propensity AUC and overlap) that every later uplift model is built on, so
model runs have a tracked baseline to be compared against. Requires the raw
files under ``data/raw/``; not run in CI.

Usage:
    uv run python scripts/log_data_validation.py
"""

from __future__ import annotations

from dataclasses import asdict

import mlflow

from promolift.data.loader import Dataset, load_lazy
from promolift.features.demographics import build_demographic_features
from promolift.tracking.mlflow_tracking import Experiment, start_run
from promolift.validation.experiment_validity import (
    categorical_covariate_balance,
    numeric_covariate_balance,
    outcome_distribution,
    propensity_check,
    raw_ate_sanity_check,
)
from promolift.validation.referential_integrity import transaction_date_integrity

_CONFIDENCE = 0.95
_PROPENSITY_COVARIATES = ["age", "gender"]
_N_FOLDS = 5
_TAIL_PERCENTILE = 0.01
_RANDOM_STATE = 42


def main() -> None:
    with start_run(Experiment.DATA_VALIDATION, run_name="experiment-validity"):
        mlflow.log_params(
            {
                "confidence": _CONFIDENCE,
                "propensity_covariates": ",".join(_PROPENSITY_COVARIATES),
                "propensity_n_folds": _N_FOLDS,
                "propensity_tail_percentile": _TAIL_PERCENTILE,
                "random_state": _RANDOM_STATE,
            }
        )

        ate = raw_ate_sanity_check(confidence=_CONFIDENCE)
        outcome = outcome_distribution()
        age = numeric_covariate_balance("age")
        gender = categorical_covariate_balance("gender")

        # Same covariate construction as notebooks/02_data_validation.ipynb:
        # demographics at the end of the purchase window, implausible ages nulled.
        reference_date = transaction_date_integrity().max_date
        demographics = build_demographic_features(reference_date).collect()
        joined = (
            load_lazy(Dataset.UPLIFT_TRAIN).collect().join(demographics, on="client_id", how="left")
        )
        propensity = propensity_check(
            joined.select(_PROPENSITY_COVARIATES),
            joined["treatment_flg"],
            n_folds=_N_FOLDS,
            tail_percentile=_TAIL_PERCENTILE,
            random_state=_RANDOM_STATE,
        )

        metrics = {
            "ate": ate.ate,
            "ate_standard_error": ate.standard_error,
            "ate_ci_lower": ate.ci_lower,
            "ate_ci_upper": ate.ci_upper,
            "treatment_rate": ate.treatment_rate,
            "control_rate": ate.control_rate,
            "overall_conversion_rate": outcome.overall_rate,
            "n_treatment": ate.treatment_count,
            "n_control": ate.control_count,
            "age_smd": age.standardized_mean_diff,
            "gender_max_proportion_diff": gender.max_proportion_diff,
            "propensity_auc": propensity.auc,
        }
        if propensity.overlap_range is not None:
            metrics["propensity_overlap_lower"] = propensity.overlap_range[0]
            metrics["propensity_overlap_upper"] = propensity.overlap_range[1]
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "age_is_balanced": str(age.is_balanced).lower(),
                "gender_is_balanced": str(gender.is_balanced).lower(),
                "propensity_overlap": str(propensity.overlap_range is not None).lower(),
            }
        )
        mlflow.log_dict(
            {
                "ate": asdict(ate),
                "outcome_distribution": asdict(outcome),
                "age_balance": asdict(age),
                "gender_balance": asdict(gender),
                "propensity": asdict(propensity),
            },
            "reports/experiment_validity.json",
        )

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")


if __name__ == "__main__":
    main()
