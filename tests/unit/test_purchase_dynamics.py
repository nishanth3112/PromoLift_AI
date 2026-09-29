"""Unit tests for spend and visit dynamics features."""

from datetime import datetime
from pathlib import Path

import pytest

from promolift.features.purchase_dynamics import build_purchase_dynamics_features

_HEADER = "client_id,transaction_id,transaction_datetime,purchase_sum\n"


def _build(tmp_path: Path, rows: str, reference_date: datetime) -> dict:
    (tmp_path / "purchases.csv").write_text(_HEADER + rows)
    result = build_purchase_dynamics_features(reference_date, tmp_path).collect()
    return {row["client_id"]: row for row in result.iter_rows(named=True)}


def test_deduplicates_line_items_to_transactions(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-09 10:00:00,100.0\nc1,t1,2019-01-09 10:00:00,100.0\n",
        datetime(2019, 1, 10),
    )

    assert rows["c1"]["recent_frequency_7d"] == 1
    assert rows["c1"]["recent_monetary_7d"] == 100.0


def test_interpurchase_gaps_are_between_distinct_purchase_days(tmp_path: Path) -> None:
    # Two baskets on Jan 1 are one visit day; gaps are 2 and 4 days.
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-01 09:00:00,10.0\n"
        "c1,t2,2019-01-01 18:00:00,10.0\n"
        "c1,t3,2019-01-03 12:00:00,10.0\n"
        "c1,t4,2019-01-07 08:00:00,10.0\n",
        datetime(2019, 1, 10),
    )

    assert rows["c1"]["active_days"] == 3
    assert rows["c1"]["interpurchase_days_mean"] == 3.0
    assert rows["c1"]["interpurchase_days_std"] == pytest.approx(2**0.5)
    assert rows["c1"]["interpurchase_days_cv"] == pytest.approx(2**0.5 / 3.0)


def test_interpurchase_stats_are_null_with_too_few_visit_days(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-01 09:00:00,10.0\n"
        "c2,t2,2019-01-01 09:00:00,10.0\n"
        "c2,t3,2019-01-05 09:00:00,10.0\n",
        datetime(2019, 1, 10),
    )

    assert rows["c1"]["interpurchase_days_mean"] is None
    assert rows["c1"]["interpurchase_days_std"] is None
    assert rows["c2"]["interpurchase_days_mean"] == 4.0
    # One gap has no spread.
    assert rows["c2"]["interpurchase_days_std"] is None
    assert rows["c2"]["interpurchase_days_cv"] is None


def test_recent_windows_count_transactions_and_spend(tmp_path: Path) -> None:
    reference = datetime(2019, 4, 1)
    rows = _build(
        tmp_path,
        "c1,t1,2019-03-30 00:00:00,1.0\n"  # 2 days before
        "c1,t2,2019-03-22 00:00:00,10.0\n"  # 10 days before
        "c1,t3,2019-02-20 00:00:00,100.0\n"  # 40 days before
        "c1,t4,2019-01-11 00:00:00,1000.0\n"  # 80 days before
        "c1,t5,2018-12-02 00:00:00,10000.0\n",  # 120 days before
        reference,
    )

    c1 = rows["c1"]
    assert [c1[f"recent_frequency_{d}d"] for d in (7, 14, 60, 90)] == [1, 2, 3, 4]
    assert [c1[f"recent_monetary_{d}d"] for d in (7, 14, 60, 90)] == [1.0, 11.0, 111.0, 1111.0]


def test_recent_windows_are_zero_not_null_without_recent_activity(tmp_path: Path) -> None:
    rows = _build(tmp_path, "c1,t1,2018-12-01 00:00:00,50.0\n", datetime(2019, 4, 1))

    assert rows["c1"]["recent_frequency_7d"] == 0
    assert rows["c1"]["recent_monetary_90d"] == 0.0


def test_trend_slopes_are_positive_for_growing_and_negative_for_shrinking(
    tmp_path: Path,
) -> None:
    # Four 28-day bins back from the reference date; bin 0 is the most recent.
    reference = datetime(2019, 4, 1)
    rows = _build(
        tmp_path,
        # Growing: 10, 20, 30, 40 spend and 1, 1, 2, 2 visits, oldest to newest.
        "up,u1,2018-12-20 00:00:00,10.0\n"  # 102 days before: bin 3
        "up,u2,2019-01-20 00:00:00,20.0\n"  # 71 days: bin 2
        "up,u3,2019-02-15 00:00:00,15.0\n"  # 45 days: bin 1
        "up,u4,2019-02-16 00:00:00,15.0\n"
        "up,u5,2019-03-20 00:00:00,20.0\n"  # 12 days: bin 0
        "up,u6,2019-03-21 00:00:00,20.0\n"
        # Shrinking: all spend in the oldest bin.
        "down,d1,2018-12-20 00:00:00,40.0\n"
        # Flat: the same spend and visits in every bin.
        "flat,f1,2018-12-20 00:00:00,10.0\n"
        "flat,f2,2019-01-20 00:00:00,10.0\n"
        "flat,f3,2019-02-15 00:00:00,10.0\n"
        "flat,f4,2019-03-20 00:00:00,10.0\n",
        reference,
    )

    assert rows["up"]["spend_trend_slope"] > 0
    assert rows["up"]["visit_trend_slope"] > 0
    assert rows["down"]["spend_trend_slope"] < 0
    assert rows["down"]["visit_trend_slope"] < 0
    assert rows["flat"]["spend_trend_slope"] == pytest.approx(0.0)
    assert rows["flat"]["visit_trend_slope"] == pytest.approx(0.0)


def test_trend_slope_is_ordinary_least_squares_over_zero_filled_bins(tmp_path: Path) -> None:
    # Spend by bin, oldest to newest: 0, 0, 0, 30 -> OLS slope on x = 0..3 is 9.
    rows = _build(tmp_path, "c1,t1,2019-03-30 00:00:00,30.0\n", datetime(2019, 4, 1))

    assert rows["c1"]["spend_trend_slope"] == pytest.approx(9.0)
    assert rows["c1"]["visit_trend_slope"] == pytest.approx(0.3)


def test_trend_ignores_transactions_older_than_the_trend_window(tmp_path: Path) -> None:
    # 120 days before the reference date is outside the 4 x 28 = 112-day window.
    rows = _build(tmp_path, "c1,t1,2018-12-02 00:00:00,50.0\n", datetime(2019, 4, 1))

    assert rows["c1"]["spend_trend_slope"] == 0.0
    assert rows["c1"]["visit_trend_slope"] == 0.0


def test_transactions_after_reference_date_are_excluded(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-05 00:00:00,10.0\nc1,t2,2019-01-20 00:00:00,99.0\n",
        datetime(2019, 1, 10),
    )

    assert rows["c1"]["active_days"] == 1
    assert rows["c1"]["recent_monetary_90d"] == 10.0
