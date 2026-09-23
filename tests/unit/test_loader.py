"""Unit tests for the dataset registry and lazy loader."""

from pathlib import Path

import polars as pl
import pytest

from promolift.data.loader import Dataset, load_lazy, path_for, raw_data_dir, resolve_path


def test_raw_data_dir_defaults_under_project_root() -> None:
    path = raw_data_dir()

    assert path.name == "raw"
    assert path.parent.name == "data"


def test_raw_data_dir_respects_override(tmp_path: Path) -> None:
    assert raw_data_dir(tmp_path) == tmp_path


def test_path_for_maps_each_dataset_to_its_expected_filename(tmp_path: Path) -> None:
    assert path_for(Dataset.CLIENTS, tmp_path) == tmp_path / "clients.csv"
    assert path_for(Dataset.PRODUCTS, tmp_path) == tmp_path / "products.csv"
    assert path_for(Dataset.PURCHASES, tmp_path) == tmp_path / "purchases.csv"
    assert path_for(Dataset.UPLIFT_TRAIN, tmp_path) == tmp_path / "uplift_train.csv"
    assert path_for(Dataset.UPLIFT_TEST, tmp_path) == tmp_path / "uplift_test.csv"
    assert (
        path_for(Dataset.UPLIFT_SAMPLE_SUBMISSION, tmp_path)
        == tmp_path / "uplift_sample_submission.csv"
    )


def test_resolve_path_raises_clear_error_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="clients"):
        resolve_path(Dataset.CLIENTS, tmp_path)


def test_resolve_path_returns_path_when_file_exists(tmp_path: Path) -> None:
    target = tmp_path / "clients.csv"
    target.write_text("client_id\nabc\n")

    assert resolve_path(Dataset.CLIENTS, tmp_path) == target


def test_load_lazy_returns_lazyframe_with_correct_schema(tmp_path: Path) -> None:
    target = tmp_path / "uplift_train.csv"
    target.write_text("client_id,treatment_flg,target\nc1,0,1\nc2,1,0\n")

    lf = load_lazy(Dataset.UPLIFT_TRAIN, tmp_path)

    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect_schema().names() == ["client_id", "treatment_flg", "target"]
    assert lf.collect().height == 2


def test_load_lazy_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_lazy(Dataset.PURCHASES, tmp_path)
