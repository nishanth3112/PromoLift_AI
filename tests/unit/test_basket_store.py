"""Unit tests for basket composition and store/time habit features."""

from datetime import datetime
from pathlib import Path

import pytest

from promolift.features.basket_store import build_basket_store_features

_HEADER = "client_id,transaction_id,transaction_datetime,store_id,product_id,product_quantity\n"


def _build(tmp_path: Path, rows: str, reference_date: datetime) -> dict:
    (tmp_path / "purchases.csv").write_text(_HEADER + rows)
    result = build_basket_store_features(reference_date, tmp_path).collect()
    return {row["client_id"]: row for row in result.iter_rows(named=True)}


def test_basket_size_is_averaged_per_transaction(tmp_path: Path) -> None:
    # t1: 3 lines, 5 units; t2: 1 line, 1 unit.
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-01 10:00:00,s1,p1,2.0\n"
        "c1,t1,2019-01-01 10:00:00,s1,p2,2.0\n"
        "c1,t1,2019-01-01 10:00:00,s1,p3,1.0\n"
        "c1,t2,2019-01-02 10:00:00,s1,p1,1.0\n",
        datetime(2019, 1, 3),
    )

    assert rows["c1"]["basket_lines_mean"] == 2.0
    assert rows["c1"]["basket_quantity_mean"] == 3.0


def test_product_variety_counts_distinct_products_over_line_items(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-01 10:00:00,s1,p1,1.0\n"
        "c1,t1,2019-01-01 10:00:00,s1,p2,1.0\n"
        "c1,t2,2019-01-02 10:00:00,s1,p1,1.0\n"
        "c1,t3,2019-01-03 10:00:00,s1,p1,1.0\n",
        datetime(2019, 1, 4),
    )

    assert rows["c1"]["distinct_products"] == 2
    assert rows["c1"]["product_variety_ratio"] == 0.5


def test_store_loyalty_uses_transactions_not_line_items(tmp_path: Path) -> None:
    # s1 has more line items, but s2 has more transactions.
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-01 10:00:00,s1,p1,1.0\n"
        "c1,t1,2019-01-01 10:00:00,s1,p2,1.0\n"
        "c1,t1,2019-01-01 10:00:00,s1,p3,1.0\n"
        "c1,t2,2019-01-02 10:00:00,s2,p1,1.0\n"
        "c1,t3,2019-01-03 10:00:00,s2,p1,1.0\n",
        datetime(2019, 1, 4),
    )

    assert rows["c1"]["distinct_stores"] == 2
    assert rows["c1"]["top_store_tx_share"] == pytest.approx(2 / 3)


def test_time_habits_are_shares_of_transactions(tmp_path: Path) -> None:
    # 2019-01-05 is a Saturday, 2019-01-06 a Sunday, 2019-01-07 a Monday.
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-05 09:00:00,s1,p1,1.0\n"
        "c1,t1,2019-01-05 09:00:00,s1,p2,1.0\n"
        "c1,t2,2019-01-06 19:30:00,s1,p1,1.0\n"
        "c1,t3,2019-01-07 12:00:00,s1,p1,1.0\n"
        "c1,t4,2019-01-07 18:00:00,s1,p1,1.0\n",
        datetime(2019, 1, 8),
    )

    assert rows["c1"]["weekend_tx_rate"] == 0.5
    # Hours 9, 19, 12, 18 -- one per transaction, not per line item.
    assert rows["c1"]["tx_hour_mean"] == 14.5
    assert rows["c1"]["tx_hour_std"] == pytest.approx(23.0**0.5)


def test_hour_spread_is_null_for_a_single_transaction(tmp_path: Path) -> None:
    rows = _build(tmp_path, "c1,t1,2019-01-01 10:00:00,s1,p1,1.0\n", datetime(2019, 1, 2))

    assert rows["c1"]["tx_hour_mean"] == 10.0
    assert rows["c1"]["tx_hour_std"] is None


def test_transactions_after_reference_date_are_excluded(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-01 10:00:00,s1,p1,1.0\n"
        "c1,t2,2019-01-20 10:00:00,s2,p2,9.0\n"
        "c1,t2,2019-01-20 10:00:00,s2,p3,9.0\n",
        datetime(2019, 1, 10),
    )

    assert rows["c1"]["distinct_products"] == 1
    assert rows["c1"]["distinct_stores"] == 1
    assert rows["c1"]["basket_quantity_mean"] == 1.0


def test_one_row_per_client(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        "c1,t1,2019-01-01 10:00:00,s1,p1,1.0\nc2,t2,2019-01-01 10:00:00,s1,p1,1.0\n",
        datetime(2019, 1, 2),
    )

    assert sorted(rows) == ["c1", "c2"]
