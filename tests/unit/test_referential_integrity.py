"""Unit tests for referential integrity and date-sanity checks."""

from datetime import datetime
from pathlib import Path

from promolift.validation.referential_integrity import (
    client_foreign_key_integrity,
    product_foreign_key_integrity,
    transaction_date_integrity,
)


def test_client_foreign_key_integrity_detects_orphans(tmp_path: Path) -> None:
    (tmp_path / "clients.csv").write_text("client_id\nc1\nc2\n")
    (tmp_path / "purchases.csv").write_text("client_id\nc1\nc2\nc1\nc3\n")

    report = client_foreign_key_integrity(tmp_path)

    assert report.total_rows == 4
    assert report.orphan_count == 1
    assert report.orphan_rate == 0.25


def test_client_foreign_key_integrity_reports_no_orphans_for_clean_data(tmp_path: Path) -> None:
    (tmp_path / "clients.csv").write_text("client_id\nc1\nc2\n")
    (tmp_path / "purchases.csv").write_text("client_id\nc1\nc2\nc1\n")

    report = client_foreign_key_integrity(tmp_path)

    assert report.orphan_count == 0
    assert report.orphan_rate == 0.0


def test_product_foreign_key_integrity_detects_orphans(tmp_path: Path) -> None:
    (tmp_path / "products.csv").write_text("product_id\np1\np2\n")
    (tmp_path / "purchases.csv").write_text("product_id\np1\np2\np9\n")

    report = product_foreign_key_integrity(tmp_path)

    assert report.total_rows == 3
    assert report.orphan_count == 1
    assert report.orphan_rate == 1 / 3


def test_transaction_date_integrity_parses_valid_dates(tmp_path: Path) -> None:
    (tmp_path / "purchases.csv").write_text(
        "transaction_datetime\n2018-11-21 21:02:33\n2019-03-18 23:40:03\n"
    )

    report = transaction_date_integrity(tmp_path, as_of=datetime(2026, 1, 1))

    assert report.total_rows == 2
    assert report.unparseable_count == 0
    assert report.min_date == datetime(2018, 11, 21, 21, 2, 33)
    assert report.max_date == datetime(2019, 3, 18, 23, 40, 3)
    assert report.future_date_count == 0


def test_transaction_date_integrity_detects_unparseable_dates(tmp_path: Path) -> None:
    (tmp_path / "purchases.csv").write_text(
        "transaction_datetime\n2018-11-21 21:02:33\nnot-a-date\n"
    )

    report = transaction_date_integrity(tmp_path, as_of=datetime(2026, 1, 1))

    assert report.total_rows == 2
    assert report.unparseable_count == 1


def test_transaction_date_integrity_detects_future_dates(tmp_path: Path) -> None:
    (tmp_path / "purchases.csv").write_text(
        "transaction_datetime\n2018-11-21 21:02:33\n2099-01-01 00:00:00\n"
    )

    report = transaction_date_integrity(tmp_path, as_of=datetime(2026, 1, 1))

    assert report.future_date_count == 1
