"""Unit tests for promotion responsiveness features."""

from datetime import datetime
from pathlib import Path

import pytest

from promolift.features.promo_responsiveness import build_promo_responsiveness_features

_HEADER = (
    "client_id,transaction_id,transaction_datetime,regular_points_received,"
    "express_points_received,regular_points_spent,express_points_spent,purchase_sum\n"
)


def _build(tmp_path: Path, rows: str, reference_date: datetime) -> dict:
    (tmp_path / "purchases.csv").write_text(_HEADER + rows)
    result = build_promo_responsiveness_features(reference_date, tmp_path).collect()
    return {row["client_id"]: row for row in result.iter_rows(named=True)}


def test_deduplicates_transaction_level_points_across_line_items(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-01 00:00:00,0.0,5.0,-20.0,-10.0,100.0\n"
        "c1,t1,2019-01-01 00:00:00,0.0,5.0,-20.0,-10.0,100.0\n",
        datetime(2019, 1, 2),
    )

    assert rows["c1"]["express_points_received_total"] == 5.0
    assert rows["c1"]["express_points_spent_total"] == 10.0
    assert rows["c1"]["points_discount_share"] == pytest.approx(0.3)


def test_transaction_rates_are_shares_of_distinct_transactions(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-01 00:00:00,1.0,5.0,0.0,0.0,100.0\n"
        "c1,t2,2019-01-02 00:00:00,1.0,0.0,-20.0,0.0,100.0\n"
        "c1,t3,2019-01-03 00:00:00,1.0,0.0,0.0,-10.0,100.0\n"
        "c1,t4,2019-01-04 00:00:00,1.0,0.0,0.0,0.0,100.0\n",
        datetime(2019, 1, 5),
    )

    assert rows["c1"]["express_received_tx_rate"] == 0.25
    assert rows["c1"]["express_spent_tx_rate"] == 0.25
    # Any redemption -- regular or express -- counts.
    assert rows["c1"]["redeem_tx_rate"] == 0.5


def test_discount_share_is_points_spent_over_spend(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-01 00:00:00,0.0,0.0,-30.0,0.0,100.0\n"
        "c1,t2,2019-01-02 00:00:00,0.0,0.0,0.0,-10.0,300.0\n",
        datetime(2019, 1, 3),
    )

    assert rows["c1"]["points_discount_share"] == pytest.approx(40.0 / 400.0)


def test_discount_share_is_null_without_spend(tmp_path: Path) -> None:
    rows = _build(tmp_path, "c1,t1,2019-01-01 00:00:00,0.0,0.0,0.0,0.0,0.0\n", datetime(2019, 1, 2))

    assert rows["c1"]["points_discount_share"] is None


def test_days_since_last_redeem_uses_latest_redeeming_transaction(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-01 00:00:00,0.0,0.0,-5.0,0.0,100.0\n"
        "c1,t2,2019-01-04 00:00:00,0.0,0.0,0.0,-5.0,100.0\n"
        "c1,t3,2019-01-08 00:00:00,0.0,0.0,0.0,0.0,100.0\n",
        datetime(2019, 1, 10),
    )

    assert rows["c1"]["days_since_last_redeem"] == 6


def test_days_since_last_redeem_is_null_for_never_redeemed(tmp_path: Path) -> None:
    rows = _build(
        tmp_path, "c1,t1,2019-01-01 00:00:00,1.0,0.0,0.0,0.0,100.0\n", datetime(2019, 1, 2)
    )

    assert rows["c1"]["days_since_last_redeem"] is None
    assert rows["c1"]["redeem_tx_rate"] == 0.0


def test_transactions_after_reference_date_are_excluded(tmp_path: Path) -> None:
    # A redemption after the reference date is post-treatment information.
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-01 00:00:00,0.0,0.0,0.0,0.0,100.0\n"
        "c1,t2,2019-02-01 00:00:00,0.0,9.0,-50.0,-9.0,100.0\n"
        "c2,t3,2019-02-01 00:00:00,0.0,0.0,-50.0,0.0,100.0\n",
        datetime(2019, 1, 15),
    )

    assert rows["c1"]["redeem_tx_rate"] == 0.0
    assert rows["c1"]["express_points_received_total"] == 0.0
    assert rows["c1"]["days_since_last_redeem"] is None
    assert "c2" not in rows


def test_one_row_per_client(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-01 00:00:00,0.0,0.0,0.0,0.0,100.0\n"
        "c2,t2,2019-01-01 00:00:00,0.0,0.0,0.0,0.0,100.0\n"
        "c2,t3,2019-01-02 00:00:00,0.0,0.0,0.0,0.0,100.0\n",
        datetime(2019, 1, 3),
    )

    assert sorted(rows) == ["c1", "c2"]
