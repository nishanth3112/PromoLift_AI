"""Unit tests for core data-audit utilities, using synthetic CSV fixtures."""

from pathlib import Path

import pytest

from promolift.data.loader import Dataset
from promolift.validation.data_audit import audit_dataset, column_cardinality, inventory


@pytest.fixture
def populated_raw_dir(tmp_path: Path) -> Path:
    (tmp_path / "clients.csv").write_text(
        "client_id,first_issue_date,first_redeem_date,age,gender\n"
        "c1,2020-01-01,2020-02-01,30,F\n"
        "c2,2020-01-02,,40,M\n"
        "c3,2020-01-03,2020-02-03,,U\n"
    )
    (tmp_path / "uplift_train.csv").write_text(
        "client_id,treatment_flg,target\nc1,0,1\nc2,1,1\nc2,1,0\n"
    )
    return tmp_path


def test_inventory_reports_existing_and_missing_files(populated_raw_dir: Path) -> None:
    entries = {entry.name: entry for entry in inventory(populated_raw_dir)}

    assert entries["clients"].exists is True
    assert entries["clients"].size_bytes is not None
    assert entries["clients"].size_bytes > 0
    assert entries["purchases"].exists is False
    assert entries["purchases"].size_bytes is None


def test_audit_dataset_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        audit_dataset(Dataset.CLIENTS, tmp_path)


def test_audit_dataset_computes_row_and_column_counts(populated_raw_dir: Path) -> None:
    audit = audit_dataset(Dataset.CLIENTS, populated_raw_dir)

    assert audit.row_count == 3
    assert audit.column_count == 5


def test_audit_dataset_computes_null_counts_and_rates(populated_raw_dir: Path) -> None:
    audit = audit_dataset(Dataset.CLIENTS, populated_raw_dir)
    by_name = {column.name: column for column in audit.columns}

    assert by_name["first_redeem_date"].null_count == 1
    assert by_name["age"].null_count == 1
    assert by_name["age"].null_rate == pytest.approx(1 / 3)
    assert by_name["gender"].null_count == 0


def test_audit_dataset_computes_numeric_range_and_skips_strings(populated_raw_dir: Path) -> None:
    audit = audit_dataset(Dataset.CLIENTS, populated_raw_dir)
    by_name = {column.name: column for column in audit.columns}

    assert by_name["age"].min_value == 30
    assert by_name["age"].max_value == 40
    assert by_name["client_id"].min_value is None
    assert by_name["client_id"].max_value is None


def test_audit_dataset_detects_duplicate_keys(populated_raw_dir: Path) -> None:
    audit = audit_dataset(Dataset.UPLIFT_TRAIN, populated_raw_dir, key_columns=["client_id"])

    assert audit.duplicate_key_count == 1


def test_audit_dataset_skips_duplicate_check_without_key_columns(populated_raw_dir: Path) -> None:
    audit = audit_dataset(Dataset.UPLIFT_TRAIN, populated_raw_dir)

    assert audit.duplicate_key_count is None


def test_column_cardinality_counts_unique_values(populated_raw_dir: Path) -> None:
    result = column_cardinality(Dataset.CLIENTS, ["client_id", "gender"], populated_raw_dir)

    assert result == {"client_id": 3, "gender": 3}
