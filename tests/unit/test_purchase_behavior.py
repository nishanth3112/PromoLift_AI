"""Unit tests for purchase behavior feature building."""

from datetime import datetime
from pathlib import Path

import pytest

from promolift.features.purchase_behavior import build_purchase_behavior_features


def test_deduplicates_transaction_level_fields_across_line_items(tmp_path: Path) -> None:
    # Same transaction, two line items -- points/purchase_sum repeat on both
    # rows (as they do in the real file). Must not be double-counted.
    (tmp_path / "purchases.csv").write_text(
        "client_id,transaction_id,transaction_datetime,regular_points_received,"
        "express_points_received,regular_points_spent,express_points_spent,purchase_sum\n"
        "c1,t1,2019-01-01 00:00:00,10.0,0.0,0.0,0.0,100.0\n"
        "c1,t1,2019-01-01 00:00:00,10.0,0.0,0.0,0.0,100.0\n"
    )

    result = build_purchase_behavior_features(datetime(2019, 1, 2), tmp_path).collect()

    assert result["frequency"].to_list() == [1]
    assert result["monetary_total"].to_list() == [100.0]
    assert result["points_received_total"].to_list() == [10.0]


def test_frequency_counts_distinct_transactions(tmp_path: Path) -> None:
    (tmp_path / "purchases.csv").write_text(
        "client_id,transaction_id,transaction_datetime,regular_points_received,"
        "express_points_received,regular_points_spent,express_points_spent,purchase_sum\n"
        "c1,t1,2019-01-01 00:00:00,0.0,0.0,0.0,0.0,50.0\n"
        "c1,t2,2019-01-02 00:00:00,0.0,0.0,0.0,0.0,75.0\n"
    )

    result = build_purchase_behavior_features(datetime(2019, 1, 3), tmp_path).collect()

    assert result["frequency"].to_list() == [2]
    assert result["monetary_total"].to_list() == [125.0]
    assert result["monetary_avg"].to_list() == [62.5]


def test_recency_days_relative_to_reference_date(tmp_path: Path) -> None:
    (tmp_path / "purchases.csv").write_text(
        "client_id,transaction_id,transaction_datetime,regular_points_received,"
        "express_points_received,regular_points_spent,express_points_spent,purchase_sum\n"
        "c1,t1,2019-01-01 00:00:00,0.0,0.0,0.0,0.0,50.0\n"
        "c1,t2,2019-01-05 00:00:00,0.0,0.0,0.0,0.0,50.0\n"
    )

    result = build_purchase_behavior_features(datetime(2019, 1, 10), tmp_path).collect()

    # recency is measured from the *last* (most recent) transaction: Jan 5
    assert result["recency_days"].to_list() == [5]


def test_monetary_std_is_null_for_single_transaction_client(tmp_path: Path) -> None:
    (tmp_path / "purchases.csv").write_text(
        "client_id,transaction_id,transaction_datetime,regular_points_received,"
        "express_points_received,regular_points_spent,express_points_spent,purchase_sum\n"
        "c1,t1,2019-01-01 00:00:00,0.0,0.0,0.0,0.0,50.0\n"
    )

    result = build_purchase_behavior_features(datetime(2019, 1, 2), tmp_path).collect()

    assert result["monetary_std"].to_list() == [None]


def test_points_spent_total_sums_absolute_values(tmp_path: Path) -> None:
    (tmp_path / "purchases.csv").write_text(
        "client_id,transaction_id,transaction_datetime,regular_points_received,"
        "express_points_received,regular_points_spent,express_points_spent,purchase_sum\n"
        "c1,t1,2019-01-01 00:00:00,0.0,0.0,-30.0,-5.0,50.0\n"
    )

    result = build_purchase_behavior_features(datetime(2019, 1, 2), tmp_path).collect()

    assert result["points_spent_total"].to_list() == pytest.approx([35.0])
