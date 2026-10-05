"""Unit tests for feature drift: reference bins, PSI, statuses, and thresholds."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from promolift.monitoring.drift import (
    REFERENCE_VERSION,
    DriftStatus,
    DriftThresholds,
    bin_shares,
    build_reference,
    check_reference,
    drift_report,
    load_drift_thresholds,
    overall_status,
    psi,
)

_CATEGORIES = {"gender": ("F", "M", "U")}


def _batch(n: int, seed: int, *, valid_share: float = 0.95, spend_shift: float = 0.0):
    # Like the real canonical input: a continuous float, a 0/1 flag that is
    # almost always 1, a small count, and gender as text.
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "monetary_avg": rng.lognormal(6.0 + spend_shift, 0.5, n),
            "age_is_valid": (rng.random(n) < valid_share).astype("float64"),
            "recent_frequency_30d": rng.poisson(3, n).astype("float64"),
            "gender": rng.choice(["F", "M", "U"], n, p=[0.37, 0.17, 0.46]),
        }
    )


@pytest.fixture(scope="module")
def reference() -> dict:
    return build_reference(_batch(20_000, seed=0), _CATEGORIES)


def _psi(report, feature: str) -> float:
    return report.filter(report["feature"] == feature)["psi"][0]


# --- reference -------------------------------------------------------------------


def test_reference_bins_each_kind_of_feature(reference: dict) -> None:
    specs = reference["features"]

    assert reference["version"] == REFERENCE_VERSION
    assert reference["n_clients"] == 20_000
    assert specs["monetary_avg"]["binning"] == "quantile"
    assert len(specs["monetary_avg"]["edges"]) == 9
    # A 95/5 flag keeps both of its values, not one collapsed decile bin.
    assert specs["age_is_valid"]["binning"] == "values"
    assert specs["age_is_valid"]["values"] == [0.0, 1.0]
    assert specs["age_is_valid"]["expected_shares"][0] == pytest.approx(0.05, abs=0.01)
    assert specs["gender"]["levels"] == ["F", "M", "U"]
    for spec in specs.values():
        assert sum(spec["expected_shares"]) == pytest.approx(1.0)


def test_decile_bins_hold_a_tenth_each(reference: dict) -> None:
    shares = reference["features"]["monetary_avg"]["expected_shares"]

    # 10 decile bins, then the missing bin.
    assert shares[:-1] == pytest.approx([0.1] * 10, abs=0.002)
    assert shares[-1] == 0


def test_bin_shares_put_missing_and_unseen_values_in_their_own_bins(reference: dict) -> None:
    spec = reference["features"]["age_is_valid"]

    shares = bin_shares(pd.Series([1.0, 1.0, 0.0, 7.0, np.nan]), spec)

    # bins: 0, 1, other, missing
    assert shares.tolist() == pytest.approx([0.2, 0.4, 0.2, 0.2])


def test_unseen_categories_land_in_other(reference: dict) -> None:
    shares = bin_shares(pd.Series(["F", "M", "X", None]), reference["features"]["gender"])

    # bins: F, M, U, other, missing
    assert shares.tolist() == pytest.approx([0.25, 0.25, 0.0, 0.25, 0.25])


# --- PSI and the report ------------------------------------------------------------


def test_psi_is_zero_for_identical_shares_and_grows_with_the_shift() -> None:
    base = np.array([0.25, 0.25, 0.25, 0.25])

    small = psi(base, np.array([0.3, 0.25, 0.25, 0.2]))
    large = psi(base, np.array([0.7, 0.1, 0.1, 0.1]))

    assert psi(base, base) == 0
    assert 0 < small < large


def test_psi_tolerates_empty_bins() -> None:
    assert np.isfinite(psi(np.array([0.5, 0.5, 0.0]), np.array([0.4, 0.4, 0.2])))


def test_a_fresh_sample_of_the_same_population_is_stable(reference: dict) -> None:
    report = drift_report(_batch(20_000, seed=1), reference, DriftThresholds())

    assert report["psi"].max() < 0.01
    assert set(report["status"]) == {"ok"}
    assert overall_status(report) is DriftStatus.OK


def test_a_spending_shift_fails_and_leaves_other_features_stable(reference: dict) -> None:
    report = drift_report(_batch(20_000, seed=2, spend_shift=0.5), reference, DriftThresholds())

    assert _psi(report, "monetary_avg") > 0.25
    assert report["feature"][0] == "monetary_avg"  # worst first
    assert _psi(report, "gender") < 0.01
    assert overall_status(report) is DriftStatus.FAIL


def test_a_flag_whose_share_moves_is_detected(reference: dict) -> None:
    # 95% -> 80% valid ages: invisible to decile bins, caught by value bins.
    report = drift_report(_batch(20_000, seed=3, valid_share=0.8), reference, DriftThresholds())

    assert _psi(report, "age_is_valid") > 0.1


def test_a_feature_that_starts_coming_back_empty_is_detected(reference: dict) -> None:
    batch = _batch(20_000, seed=4)
    batch.loc[batch.index[:6_000], "recent_frequency_30d"] = np.nan

    report = drift_report(batch, reference, DriftThresholds())

    row = report.filter(report["feature"] == "recent_frequency_30d").row(0, named=True)
    assert row["status"] == "fail"
    assert row["null_share"] == pytest.approx(0.3)
    assert row["reference_null_share"] == 0


def test_statuses_follow_the_thresholds() -> None:
    thresholds = DriftThresholds(warn_psi=0.1, fail_psi=0.25)

    assert thresholds.status(0.05) is DriftStatus.OK
    assert thresholds.status(0.1) is DriftStatus.OK
    assert thresholds.status(0.2) is DriftStatus.WARN
    assert thresholds.status(0.3) is DriftStatus.FAIL
    with pytest.raises(ValueError):
        DriftThresholds(warn_psi=0.3, fail_psi=0.2)


def test_the_report_needs_every_reference_feature(reference: dict) -> None:
    with pytest.raises(ValueError, match="lacks reference features"):
        drift_report(_batch(100, seed=5).drop(columns="gender"), reference, DriftThresholds())


def test_references_without_bin_shares_are_rejected() -> None:
    old_style = {"n_clients": 10, "features": {"x": {"kind": "numeric", "quantiles": {}}}}

    with pytest.raises(ValueError, match="re-register"):
        check_reference(old_style)
    with pytest.raises(ValueError, match="re-register"):
        check_reference(None)


def test_thresholds_load_from_the_monitoring_config(tmp_path: Path) -> None:
    path = tmp_path / "monitoring.yaml"
    path.write_text("drift:\n  warn_psi: 0.05\n  fail_psi: 0.2\n")

    assert load_drift_thresholds(path) == DriftThresholds(0.05, 0.2)


def test_the_committed_monitoring_config_is_valid() -> None:
    thresholds = load_drift_thresholds(
        Path(__file__).resolve().parents[2] / "configs" / "monitoring.yaml"
    )

    assert thresholds == DriftThresholds(0.1, 0.25)
