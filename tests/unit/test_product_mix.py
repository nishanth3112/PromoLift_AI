"""Unit tests for product mix feature building."""

from pathlib import Path

import pytest

from promolift.features.product_mix import build_product_mix_features


def test_counts_distinct_level_2_categories(tmp_path: Path) -> None:
    (tmp_path / "products.csv").write_text(
        "product_id,level_1,level_2,level_3,level_4,segment_id,brand_id,vendor_id,"
        "netto,is_own_trademark,is_alcohol\n"
        "p1,a,cat_x,c,d,1.0,b,v,1.0,0,0\n"
        "p2,a,cat_y,c,d,1.0,b,v,1.0,0,0\n"
        "p3,a,cat_x,c,d,1.0,b,v,1.0,0,0\n"
    )
    (tmp_path / "purchases.csv").write_text("client_id,product_id\nc1,p1\nc1,p2\nc1,p3\n")

    result = build_product_mix_features(tmp_path).collect()

    # p1 and p3 share cat_x, p2 is cat_y -- 2 distinct categories, not 3
    assert result["n_distinct_categories"].to_list() == [2]


def test_alcohol_purchase_rate_is_line_item_share(tmp_path: Path) -> None:
    (tmp_path / "products.csv").write_text(
        "product_id,level_1,level_2,level_3,level_4,segment_id,brand_id,vendor_id,"
        "netto,is_own_trademark,is_alcohol\n"
        "p1,a,x,c,d,1.0,b,v,1.0,0,1\n"
        "p2,a,x,c,d,1.0,b,v,1.0,0,0\n"
        "p3,a,x,c,d,1.0,b,v,1.0,0,0\n"
        "p4,a,x,c,d,1.0,b,v,1.0,0,0\n"
    )
    (tmp_path / "purchases.csv").write_text("client_id,product_id\nc1,p1\nc1,p2\nc1,p3\nc1,p4\n")

    result = build_product_mix_features(tmp_path).collect()

    assert result["alcohol_purchase_rate"].to_list() == pytest.approx([0.25])


def test_own_trademark_rate_is_line_item_share(tmp_path: Path) -> None:
    (tmp_path / "products.csv").write_text(
        "product_id,level_1,level_2,level_3,level_4,segment_id,brand_id,vendor_id,"
        "netto,is_own_trademark,is_alcohol\n"
        "p1,a,x,c,d,1.0,b,v,1.0,1,0\n"
        "p2,a,x,c,d,1.0,b,v,1.0,0,0\n"
    )
    (tmp_path / "purchases.csv").write_text("client_id,product_id\nc1,p1\nc1,p2\n")

    result = build_product_mix_features(tmp_path).collect()

    assert result["own_trademark_rate"].to_list() == pytest.approx([0.5])


def test_multiple_clients_get_independent_rows(tmp_path: Path) -> None:
    (tmp_path / "products.csv").write_text(
        "product_id,level_1,level_2,level_3,level_4,segment_id,brand_id,vendor_id,"
        "netto,is_own_trademark,is_alcohol\n"
        "p1,a,x,c,d,1.0,b,v,1.0,0,1\n"
        "p2,a,y,c,d,1.0,b,v,1.0,1,0\n"
    )
    (tmp_path / "purchases.csv").write_text("client_id,product_id\nc1,p1\nc2,p2\n")

    result = build_product_mix_features(tmp_path).sort("client_id").collect()

    assert result["client_id"].to_list() == ["c1", "c2"]
    assert result["alcohol_purchase_rate"].to_list() == pytest.approx([1.0, 0.0])
    assert result["own_trademark_rate"].to_list() == pytest.approx([0.0, 1.0])
