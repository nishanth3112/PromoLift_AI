"""Unit tests for the cached, hash-verified feature table store."""

import json
import logging
from pathlib import Path

import polars as pl
import pytest

from promolift.features import store
from promolift.features.build import ALL_GROUPS, BASE_GROUPS, FeatureGroup, FeatureTable
from promolift.features.store import (
    feature_code_sha256,
    feature_table_sha256,
    load_feature_table,
)


def _load(raw: Path, cache: Path, **kwargs):
    return load_feature_table(ALL_GROUPS, base_dir=raw, cache_dir=cache, **kwargs)


def _forbid_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("feature table was rebuilt instead of read from cache")

    monkeypatch.setattr(store, "build_feature_table", fail)


def test_first_load_builds_and_writes_table_and_manifest(
    feature_raw_dir: Path, tmp_path: Path
) -> None:
    stored = _load(feature_raw_dir, tmp_path / "cache")

    assert not stored.from_cache
    assert stored.table.frame.height == 2
    assert stored.path.exists()
    manifest = json.loads(stored.manifest_path.read_text())
    assert manifest["feature_table_sha256"] == stored.sha256
    assert manifest["feature_cache_key"] == stored.cache_key
    assert stored.cache_key in stored.path.name
    assert manifest["reference_date"] == "2019-01-05T18:00:00"
    assert list(manifest["groups"]) == [g.value for g in ALL_GROUPS]


def test_second_load_reads_the_cache_without_rebuilding(
    feature_raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _load(feature_raw_dir, tmp_path / "cache")
    _forbid_rebuild(monkeypatch)

    second = _load(feature_raw_dir, tmp_path / "cache")

    assert second.from_cache
    assert second.sha256 == first.sha256
    assert second.cache_key == first.cache_key
    assert second.table.groups == first.table.groups
    assert second.table.frame.equals(first.table.frame)


def test_rebuild_flag_ignores_the_cache(feature_raw_dir: Path, tmp_path: Path) -> None:
    _load(feature_raw_dir, tmp_path / "cache")

    assert not _load(feature_raw_dir, tmp_path / "cache", rebuild=True).from_cache


def test_changed_feature_code_misses_the_cache(
    feature_raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _load(feature_raw_dir, tmp_path / "cache")
    monkeypatch.setattr(store, "feature_code_sha256", lambda: "edited")

    second = _load(feature_raw_dir, tmp_path / "cache")

    assert not second.from_cache
    assert second.cache_key != first.cache_key


def test_changed_raw_data_misses_the_cache(feature_raw_dir: Path, tmp_path: Path) -> None:
    first = _load(feature_raw_dir, tmp_path / "cache")
    with (feature_raw_dir / "clients.csv").open("a") as clients:
        clients.write("c3,2018-01-01 00:00:00,,50,U\n")

    second = _load(feature_raw_dir, tmp_path / "cache")

    assert not second.from_cache
    assert second.table.frame.height == 3
    assert second.sha256 != first.sha256


def test_different_groups_are_cached_separately(feature_raw_dir: Path, tmp_path: Path) -> None:
    everything = _load(feature_raw_dir, tmp_path / "cache")
    base = load_feature_table(BASE_GROUPS, base_dir=feature_raw_dir, cache_dir=tmp_path / "cache")

    assert base.path != everything.path
    assert list(base.table.groups) == list(BASE_GROUPS)


def test_tampered_cache_is_detected_and_rebuilt(
    feature_raw_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first = _load(feature_raw_dir, tmp_path / "cache")
    first.table.frame.with_columns(pl.lit(99).alias("age")).write_parquet(first.path)

    with caplog.at_level(logging.WARNING):
        second = _load(feature_raw_dir, tmp_path / "cache")

    assert not second.from_cache
    assert second.sha256 == first.sha256
    assert "does not match its manifest" in caplog.text


def test_lineage_tags_name_the_table_hash_and_groups(feature_raw_dir: Path, tmp_path: Path) -> None:
    stored = _load(feature_raw_dir, tmp_path / "cache")

    tags = stored.lineage_tags([FeatureGroup.DEMOGRAPHICS, FeatureGroup.CATEGORY_SPEND])

    assert tags == {
        "feature_cache_key": stored.cache_key,
        "feature_table_sha256": stored.sha256,
        "feature_groups": "demographics,category_spend",
    }


def test_table_hash_ignores_row_order_but_not_values() -> None:
    groups = {FeatureGroup.DEMOGRAPHICS: ("age",)}
    table = FeatureTable(pl.DataFrame({"client_id": ["a", "b"], "age": [1, 2]}), groups)
    shuffled = FeatureTable(pl.DataFrame({"client_id": ["b", "a"], "age": [2, 1]}), groups)
    edited = FeatureTable(pl.DataFrame({"client_id": ["a", "b"], "age": [1, 3]}), groups)

    assert feature_table_sha256(table) == feature_table_sha256(shuffled)
    assert feature_table_sha256(table) != feature_table_sha256(edited)


def test_table_hash_covers_group_membership() -> None:
    frame = pl.DataFrame({"client_id": ["a"], "age": [1], "frequency": [2]})
    one = FeatureTable(frame, {FeatureGroup.DEMOGRAPHICS: ("age", "frequency")})
    two = FeatureTable(
        frame,
        {FeatureGroup.DEMOGRAPHICS: ("age",), FeatureGroup.PURCHASE_BEHAVIOR: ("frequency",)},
    )

    assert feature_table_sha256(one) != feature_table_sha256(two)


def test_code_hash_ignores_line_endings(tmp_path: Path) -> None:
    unix, windows = tmp_path / "unix", tmp_path / "windows"
    unix.mkdir()
    windows.mkdir()
    (unix / "a.py").write_bytes(b"x = 1\ny = 2\n")
    (windows / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")

    assert feature_code_sha256([unix / "a.py"], root=unix) != feature_code_sha256([], root=unix)
    assert feature_code_sha256([unix / "a.py"], root=unix) == feature_code_sha256(
        [windows / "a.py"], root=windows
    )
