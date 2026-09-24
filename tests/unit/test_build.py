"""Unit tests for the combined feature table builder."""

from datetime import datetime
from pathlib import Path

from promolift.features.build import build_feature_table


def _write_minimal_fixtures(tmp_path: Path) -> None:
    (tmp_path / "clients.csv").write_text(
        "client_id,first_issue_date,first_redeem_date,age,gender\n"
        "c1,2018-01-01 00:00:00,2018-02-01 00:00:00,30,F\n"
        "c2,2018-01-01 00:00:00,,40,M\n"
    )
    (tmp_path / "purchases.csv").write_text(
        "client_id,transaction_id,transaction_datetime,regular_points_received,"
        "express_points_received,regular_points_spent,express_points_spent,purchase_sum,product_id\n"
        "c1,t1,2019-01-01 00:00:00,10.0,0.0,0.0,0.0,100.0,p1\n"
    )
    (tmp_path / "products.csv").write_text(
        "product_id,level_1,level_2,level_3,level_4,segment_id,brand_id,vendor_id,"
        "netto,is_own_trademark,is_alcohol\n"
        "p1,a,x,c,d,1.0,b,v,1.0,0,0\n"
    )


def test_build_feature_table_includes_every_client(tmp_path: Path) -> None:
    _write_minimal_fixtures(tmp_path)

    result = build_feature_table(datetime(2019, 1, 2), tmp_path)

    assert result.height == 2
    assert set(result["client_id"].to_list()) == {"c1", "c2"}


def test_client_with_no_purchase_history_gets_nulls_not_dropped(tmp_path: Path) -> None:
    _write_minimal_fixtures(tmp_path)

    result = build_feature_table(datetime(2019, 1, 2), tmp_path)
    c2 = result.filter(result["client_id"] == "c2")

    # c2 has no rows in purchases.csv -- must still be present, with nulls
    # for purchase/product-mix columns, not silently dropped by the join.
    assert c2.height == 1
    assert c2["frequency"].to_list() == [None]
    assert c2["monetary_total"].to_list() == [None]
    assert c2["n_distinct_categories"].to_list() == [None]
    # demographic columns are unaffected
    assert c2["age"].to_list() == [40]


def test_client_with_purchase_history_gets_all_feature_groups(tmp_path: Path) -> None:
    _write_minimal_fixtures(tmp_path)

    result = build_feature_table(datetime(2019, 1, 2), tmp_path)
    c1 = result.filter(result["client_id"] == "c1")

    assert c1["frequency"].to_list() == [1]
    assert c1["monetary_total"].to_list() == [100.0]
    assert c1["n_distinct_categories"].to_list() == [1]
    assert c1["has_redeemed"].to_list() == [True]
