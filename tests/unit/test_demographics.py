"""Unit tests for demographic feature building."""

from datetime import datetime
from pathlib import Path

from promolift.features.demographics import build_demographic_features


def test_valid_age_passes_through(tmp_path: Path) -> None:
    (tmp_path / "clients.csv").write_text(
        "client_id,first_issue_date,first_redeem_date,age,gender\nc1,2018-01-01 00:00:00,,45,F\n"
    )

    result = build_demographic_features(datetime(2019, 1, 1), tmp_path).collect()

    assert result["age"].to_list() == [45]
    assert result["age_is_valid"].to_list() == [True]


def test_implausible_age_is_nulled_and_flagged(tmp_path: Path) -> None:
    (tmp_path / "clients.csv").write_text(
        "client_id,first_issue_date,first_redeem_date,age,gender\n"
        "c1,2018-01-01 00:00:00,,-7491,F\n"
        "c2,2018-01-01 00:00:00,,1901,M\n"
        "c3,2018-01-01 00:00:00,,150,M\n"
    )

    result = build_demographic_features(datetime(2019, 1, 1), tmp_path).collect()

    assert result["age"].to_list() == [None, None, None]
    assert result["age_is_valid"].to_list() == [False, False, False]


def test_tenure_days_computed_relative_to_reference_date(tmp_path: Path) -> None:
    (tmp_path / "clients.csv").write_text(
        "client_id,first_issue_date,first_redeem_date,age,gender\nc1,2018-01-01 00:00:00,,30,F\n"
    )

    result = build_demographic_features(datetime(2018, 1, 11), tmp_path).collect()

    assert result["tenure_days"].to_list() == [10]


def test_has_redeemed_and_days_to_redeem_for_redeemed_client(tmp_path: Path) -> None:
    (tmp_path / "clients.csv").write_text(
        "client_id,first_issue_date,first_redeem_date,age,gender\n"
        "c1,2018-01-01 00:00:00,2018-01-15 00:00:00,30,F\n"
    )

    result = build_demographic_features(datetime(2019, 1, 1), tmp_path).collect()

    assert result["has_redeemed"].to_list() == [True]
    assert result["days_to_redeem"].to_list() == [14]


def test_has_redeemed_false_for_client_who_never_redeemed(tmp_path: Path) -> None:
    (tmp_path / "clients.csv").write_text(
        "client_id,first_issue_date,first_redeem_date,age,gender\nc1,2018-01-01 00:00:00,,30,F\n"
    )

    result = build_demographic_features(datetime(2019, 1, 1), tmp_path).collect()

    assert result["has_redeemed"].to_list() == [False]
    assert result["days_to_redeem"].to_list() == [None]
