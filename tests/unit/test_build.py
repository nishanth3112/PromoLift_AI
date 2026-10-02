"""Unit tests for the combined feature table builder."""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from promolift.features import build
from promolift.features.build import ALL_GROUPS, BASE_GROUPS, FeatureGroup, build_feature_table


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

    result = build_feature_table(datetime(2019, 1, 2), tmp_path).frame

    assert result.height == 2
    assert set(result["client_id"].to_list()) == {"c1", "c2"}


def test_client_with_no_purchase_history_gets_nulls_not_dropped(tmp_path: Path) -> None:
    _write_minimal_fixtures(tmp_path)

    result = build_feature_table(datetime(2019, 1, 2), tmp_path).frame
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

    result = build_feature_table(datetime(2019, 1, 2), tmp_path).frame
    c1 = result.filter(result["client_id"] == "c1")

    assert c1["frequency"].to_list() == [1]
    assert c1["monetary_total"].to_list() == [100.0]
    assert c1["n_distinct_categories"].to_list() == [1]
    assert c1["has_redeemed"].to_list() == [True]


def test_default_groups_are_the_base_feature_set(feature_raw_dir: Path) -> None:
    table = build_feature_table(datetime(2019, 1, 10), feature_raw_dir)

    assert list(table.groups) == list(BASE_GROUPS)
    assert len(table.frame.columns) == 20  # client_id + the 19 Phase 10 features


def test_group_columns_partition_the_table(feature_raw_dir: Path) -> None:
    table = build_feature_table(datetime(2019, 1, 10), feature_raw_dir, groups=ALL_GROUPS)

    listed = [c for cols in table.groups.values() for c in cols]
    assert ["client_id", *listed] == table.frame.columns
    assert table.frame.height == 2
    assert table.groups[FeatureGroup.PROMO_RESPONSIVENESS][0] == "express_points_received_total"


def test_select_returns_client_id_and_the_chosen_groups_columns(feature_raw_dir: Path) -> None:
    table = build_feature_table(datetime(2019, 1, 10), feature_raw_dir, groups=ALL_GROUPS)

    selected = table.select([FeatureGroup.DEMOGRAPHICS, "basket_store"])

    assert selected.columns == [
        "client_id",
        *table.groups[FeatureGroup.DEMOGRAPHICS],
        *table.groups[FeatureGroup.BASKET_STORE],
    ]


def test_select_rejects_groups_the_table_was_not_built_with(feature_raw_dir: Path) -> None:
    table = build_feature_table(datetime(2019, 1, 10), feature_raw_dir)

    with pytest.raises(KeyError, match="category_spend"):
        table.select([FeatureGroup.CATEGORY_SPEND])


def test_demographics_always_comes_first(feature_raw_dir: Path) -> None:
    table = build_feature_table(
        datetime(2019, 1, 10),
        feature_raw_dir,
        groups=[FeatureGroup.PURCHASE_DYNAMICS, FeatureGroup.DEMOGRAPHICS],
    )

    assert list(table.groups) == [FeatureGroup.DEMOGRAPHICS, FeatureGroup.PURCHASE_DYNAMICS]


def test_groups_must_include_demographics(feature_raw_dir: Path) -> None:
    with pytest.raises(ValueError, match="demographics"):
        build_feature_table(
            datetime(2019, 1, 10), feature_raw_dir, groups=[FeatureGroup.PURCHASE_BEHAVIOR]
        )


def test_duplicate_column_across_groups_is_rejected(
    feature_raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A clashing name would otherwise come back from the join as "age_right".
    clashing = lambda _date, _dir: pl.LazyFrame({"client_id": ["c1"], "age": [1]})  # noqa: E731
    monkeypatch.setitem(build._BUILDERS, FeatureGroup.BASKET_STORE, clashing)

    with pytest.raises(ValueError, match="age"):
        build_feature_table(
            datetime(2019, 1, 10),
            feature_raw_dir,
            groups=[FeatureGroup.DEMOGRAPHICS, FeatureGroup.BASKET_STORE],
        )
