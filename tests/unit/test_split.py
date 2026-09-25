"""Unit tests for the canonical train/val/test split, using small synthetic assignments."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from promolift.data.split import (
    Split,
    SplitIntegrityError,
    generate_split,
    load_split,
    split_assignment_path,
    split_content_sha256,
    validate_split,
    write_split,
)


def _labels(n_per_stratum: int = 10) -> pl.DataFrame:
    # Every (treatment_flg, target) stratum equally represented.
    rows = [(f"c{t}{y}{i:03d}", t, y) for t in (0, 1) for y in (0, 1) for i in range(n_per_stratum)]
    return pl.DataFrame(rows, schema=["client_id", "treatment_flg", "target"], orient="row")


def _stratified_assignment(labels: pl.DataFrame) -> pl.DataFrame:
    # 6/2/2 of every 10 clients within each stratum -> exact 60/20/20, exact strata.
    pattern = [Split.TRAIN] * 6 + [Split.VAL] * 2 + [Split.TEST] * 2
    return (
        labels.with_columns(pl.int_range(pl.len()).over("treatment_flg", "target").alias("_rank"))
        .with_columns(
            pl.col("_rank")
            .map_elements(lambda r: pattern[r % 10].value, return_dtype=pl.String)
            .alias("split")
        )
        .select("client_id", "split")
    )


def test_validate_split_accepts_exact_stratified_split() -> None:
    labels = _labels()

    report = validate_split(_stratified_assignment(labels), labels)

    assert report.n_clients == 40
    assert report.fractions == pytest.approx({"train": 0.6, "val": 0.2, "test": 0.2})
    assert report.max_fraction_deviation == pytest.approx(0.0)
    assert report.max_stratum_deviation == pytest.approx(0.0)
    assert report.matches_expected_fractions is True
    assert report.is_stratified is True


def test_validate_split_flags_unstratified_split() -> None:
    labels = _labels()
    # All treated clients in train, all control clients spread over val/test.
    assignment = labels.select(
        "client_id",
        pl.when(pl.col("treatment_flg") == 1)
        .then(pl.lit(Split.TRAIN.value))
        .when(pl.col("target") == 1)
        .then(pl.lit(Split.VAL.value))
        .otherwise(pl.lit(Split.TEST.value))
        .alias("split"),
    )

    report = validate_split(assignment, labels)

    assert report.is_stratified is False
    assert report.matches_expected_fractions is False


def test_validate_split_rejects_missing_clients() -> None:
    labels = _labels()
    assignment = _stratified_assignment(labels).head(39)

    with pytest.raises(SplitIntegrityError, match="missing from the split"):
        validate_split(assignment, labels)


def test_validate_split_rejects_unknown_clients() -> None:
    labels = _labels()
    extra = pl.DataFrame({"client_id": ["stranger"], "split": ["train"]})
    assignment = pl.concat([_stratified_assignment(labels), extra])

    with pytest.raises(SplitIntegrityError, match="not in uplift_train"):
        validate_split(assignment, labels)


def test_validate_split_rejects_duplicate_clients() -> None:
    labels = _labels()
    assignment = _stratified_assignment(labels)
    duplicate = assignment.head(1).with_columns(pl.lit(Split.TEST.value).alias("split"))

    with pytest.raises(SplitIntegrityError, match="more than once"):
        validate_split(pl.concat([assignment, duplicate]), labels)


def test_validate_split_rejects_unknown_split_labels() -> None:
    labels = _labels()
    assignment = _stratified_assignment(labels).with_columns(
        pl.when(pl.col("split") == Split.VAL.value)
        .then(pl.lit("valid"))
        .otherwise(pl.col("split"))
        .alias("split")
    )

    with pytest.raises(SplitIntegrityError, match="valid"):
        validate_split(assignment, labels)


def test_split_content_sha256_ignores_row_order() -> None:
    assignment = _stratified_assignment(_labels())

    shuffled = assignment.sample(fraction=1.0, shuffle=True, seed=0)

    assert split_content_sha256(shuffled) == split_content_sha256(assignment)


def test_split_content_sha256_changes_when_one_client_moves() -> None:
    assignment = _stratified_assignment(_labels())
    first = assignment["client_id"][0]
    moved = assignment.with_columns(
        pl.when(pl.col("client_id") == first)
        .then(pl.lit(Split.TEST.value))
        .otherwise(pl.col("split"))
        .alias("split")
    )

    assert split_content_sha256(moved) != split_content_sha256(assignment)


def test_write_split_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    assignment = _stratified_assignment(_labels())
    path = split_assignment_path(tmp_path)
    write_split(assignment, path)

    with pytest.raises(FileExistsError, match="--force"):
        write_split(assignment, path)


def test_write_split_overwrites_with_force(tmp_path: Path) -> None:
    assignment = _stratified_assignment(_labels())
    path = split_assignment_path(tmp_path)
    write_split(assignment.head(5), path)

    write_split(assignment, path, force=True)

    assert load_split(path).height == assignment.height


def test_load_split_round_trips_content(tmp_path: Path) -> None:
    assignment = _stratified_assignment(_labels())
    path = split_assignment_path(tmp_path)

    write_split(assignment.sample(fraction=1.0, shuffle=True, seed=1), path)

    assert split_content_sha256(load_split(path)) == split_content_sha256(assignment)


def test_load_split_explains_how_to_get_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fetch_split"):
        load_split(split_assignment_path(tmp_path))


def _random_labels(n: int = 1_000, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "client_id": [f"id{i:05d}" for i in range(n)],
            "treatment_flg": rng.integers(0, 2, n),
            # Unequal strata, like the real data (~62% conversion).
            "target": (rng.random(n) < 0.62).astype(int),
        }
    )


def test_generate_split_is_stratified_60_20_20_over_all_clients() -> None:
    labels = _random_labels()

    report = validate_split(generate_split(labels), labels)

    assert report.n_clients == labels.height
    assert report.matches_expected_fractions is True
    # Tiny strata at n=1000 round to whole clients, so allow more slack than
    # the production default tuned for 200k rows.
    assert report.max_stratum_deviation < 0.01


def test_generate_split_is_deterministic_for_a_seed() -> None:
    labels = _random_labels()

    assert split_content_sha256(generate_split(labels, seed=7)) == split_content_sha256(
        generate_split(labels, seed=7)
    )


def test_generate_split_ignores_input_row_order() -> None:
    labels = _random_labels()
    shuffled = labels.sample(fraction=1.0, shuffle=True, seed=3)

    assert split_content_sha256(generate_split(shuffled)) == split_content_sha256(
        generate_split(labels)
    )


def test_generate_split_changes_with_seed() -> None:
    labels = _random_labels()

    assert split_content_sha256(generate_split(labels, seed=1)) != split_content_sha256(
        generate_split(labels, seed=2)
    )
