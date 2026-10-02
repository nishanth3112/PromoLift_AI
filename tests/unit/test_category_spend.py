"""Unit tests for level_2 category spend share features."""

import math
from datetime import datetime
from pathlib import Path

import pytest

from promolift.features.category_spend import build_category_spend_features

_PURCHASES_HEADER = (
    "client_id,transaction_id,transaction_datetime,product_id,trn_sum_from_iss,trn_sum_from_red\n"
)
_PRODUCTS = "product_id,level_2\npa,A\npb,B\npc,C\npd,D\npx,\n"


def _build(
    tmp_path: Path, rows: str, reference_date: datetime = datetime(2019, 2, 1), top_k: int = 2
) -> dict:
    (tmp_path / "purchases.csv").write_text(_PURCHASES_HEADER + rows)
    (tmp_path / "products.csv").write_text(_PRODUCTS)
    result = build_category_spend_features(reference_date, tmp_path, top_k=top_k).collect()
    return {row["client_id"]: row for row in result.iter_rows(named=True)}


# Total spend: A 600, B 300, C 90, D 10 -> with top_k=2, A and B get columns.
_MARKET = (
    "m,t1,2019-01-01 10:00:00,pa,600.0,\n"
    "m,t2,2019-01-01 10:00:00,pb,300.0,\n"
    "m,t3,2019-01-01 10:00:00,pc,90.0,\n"
    "m,t4,2019-01-01 10:00:00,pd,10.0,\n"
)


def test_top_categories_by_total_spend_get_share_columns(tmp_path: Path) -> None:
    rows = _build(tmp_path, _MARKET)

    assert [k for k in rows["m"] if k.startswith("cat_share_")] == [
        "cat_share_A",
        "cat_share_B",
        "cat_share_other",
    ]
    assert rows["m"]["cat_share_A"] == pytest.approx(0.6)
    assert rows["m"]["cat_share_B"] == pytest.approx(0.3)
    assert rows["m"]["cat_share_other"] == pytest.approx(0.1)


def test_shares_are_spend_weighted_not_line_counts(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        _MARKET + "c1,t5,2019-01-02 10:00:00,pa,30.0,\n"
        "c1,t5,2019-01-02 10:00:00,pb,5.0,\n"
        "c1,t5,2019-01-02 10:00:00,pb,5.0,\n",
    )

    assert rows["c1"]["cat_share_A"] == pytest.approx(0.75)
    assert rows["c1"]["cat_share_B"] == pytest.approx(0.25)
    assert rows["c1"]["cat_share_other"] == 0.0


def test_line_spend_is_full_price_when_points_were_redeemed(tmp_path: Path) -> None:
    # red is the full price (100); iss is only the part paid in money (60).
    rows = _build(
        tmp_path,
        _MARKET + "c1,t5,2019-01-02 10:00:00,pa,60.0,100.0\nc1,t5,2019-01-02 10:00:00,pb,100.0,\n",
    )

    assert rows["c1"]["cat_share_A"] == pytest.approx(0.5)


def test_client_missing_a_top_category_has_zero_share(tmp_path: Path) -> None:
    rows = _build(tmp_path, _MARKET + "c1,t5,2019-01-02 10:00:00,pc,50.0,\n")

    assert rows["c1"]["cat_share_A"] == 0.0
    assert rows["c1"]["cat_share_other"] == 1.0


def test_entropy_measures_spend_concentration(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        _MARKET + "even,t5,2019-01-02 10:00:00,pc,50.0,\n"
        "even,t5,2019-01-02 10:00:00,pd,50.0,\n"
        "single,t6,2019-01-02 10:00:00,pc,50.0,\n",
    )

    # Entropy runs over every category, not only the top-k ones.
    assert rows["even"]["category_spend_entropy"] == pytest.approx(math.log(2))
    assert rows["single"]["category_spend_entropy"] == 0.0


def test_zero_spend_client_has_null_shares(tmp_path: Path) -> None:
    rows = _build(tmp_path, _MARKET + "c1,t5,2019-01-02 10:00:00,pa,0.0,\n")

    assert rows["c1"]["cat_share_A"] is None
    assert rows["c1"]["cat_share_other"] is None
    assert rows["c1"]["category_spend_entropy"] is None


def test_products_without_a_category_are_ignored(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        _MARKET + "c1,t5,2019-01-02 10:00:00,pa,50.0,\n"
        "c1,t5,2019-01-02 10:00:00,px,50.0,\n"
        "c2,t6,2019-01-02 10:00:00,px,50.0,\n",
    )

    assert rows["c1"]["cat_share_A"] == 1.0
    assert "c2" not in rows


def test_ties_in_top_categories_break_by_category_id(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        "m,t1,2019-01-01 10:00:00,pd,50.0,\n"
        "m,t2,2019-01-01 10:00:00,pc,50.0,\n"
        "m,t3,2019-01-01 10:00:00,pb,50.0,\n",
    )

    assert {k for k in rows["m"] if k.startswith("cat_share_")} == {
        "cat_share_B",
        "cat_share_C",
        "cat_share_other",
    }


def test_purchases_after_reference_date_are_excluded(tmp_path: Path) -> None:
    rows = _build(
        tmp_path,
        _MARKET + "c1,t5,2019-01-02 10:00:00,pa,50.0,\nc1,t6,2019-03-01 10:00:00,pb,50.0,\n",
    )

    assert rows["c1"]["cat_share_A"] == 1.0
