"""Unit tests for run lineage capture (git state, lockfile hash, raw data fingerprint)."""

import os
import subprocess
from pathlib import Path

import pytest

from promolift.tracking.lineage import (
    UNKNOWN,
    fingerprint_digest,
    git_state,
    lockfile_sha256,
    raw_data_fingerprint,
)

_GIT_IDENTITY = ["-c", "user.name=test", "-c", "user.email=test@example.com"]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *_GIT_IDENTITY, "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def committed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "module.py").write_text("x = 1\n")
    _git(repo, "add", "module.py")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def test_git_state_reports_commit_and_clean_tree(committed_repo: Path) -> None:
    state = git_state(committed_repo)

    assert len(state.commit) == 40
    assert state.is_dirty is False


def test_git_state_detects_uncommitted_changes(committed_repo: Path) -> None:
    (committed_repo / "module.py").write_text("x = 2\n")

    assert git_state(committed_repo).is_dirty is True


def test_git_state_treats_untracked_files_as_dirty(committed_repo: Path) -> None:
    # An untracked module a run imports is just as unreproducible as an edit.
    (committed_repo / "new_module.py").write_text("y = 1\n")

    assert git_state(committed_repo).is_dirty is True


def test_git_state_is_unknown_outside_a_repository(tmp_path: Path) -> None:
    state = git_state(tmp_path)

    assert state.commit == UNKNOWN
    assert state.is_dirty is None


def test_lockfile_sha256_ignores_line_ending_differences(tmp_path: Path) -> None:
    lf_repo, crlf_repo = tmp_path / "lf", tmp_path / "crlf"
    lf_repo.mkdir()
    crlf_repo.mkdir()
    (lf_repo / "uv.lock").write_bytes(b"version = 1\nname = 'a'\n")
    (crlf_repo / "uv.lock").write_bytes(b"version = 1\r\nname = 'a'\r\n")

    assert lockfile_sha256(lf_repo) == lockfile_sha256(crlf_repo)


def test_lockfile_sha256_is_none_when_missing(tmp_path: Path) -> None:
    assert lockfile_sha256(tmp_path) is None


def test_raw_data_fingerprint_records_present_and_missing_files(tmp_path: Path) -> None:
    (tmp_path / "clients.csv").write_text("client_id\nc1\n")

    by_name = {fp.filename: fp for fp in raw_data_fingerprint(tmp_path)}

    assert by_name["clients.csv"].size_bytes == len("client_id\nc1\n")
    assert by_name["clients.csv"].modified_at is not None
    assert by_name["purchases.csv"].size_bytes is None
    assert by_name["purchases.csv"].modified_at is None


def test_fingerprint_digest_ignores_modification_time(tmp_path: Path) -> None:
    # Identical copies on two machines have different mtimes but must share a digest.
    path = tmp_path / "clients.csv"
    path.write_text("client_id\nc1\n")
    before = fingerprint_digest(raw_data_fingerprint(tmp_path))

    os.utime(path, (0, 0))

    assert fingerprint_digest(raw_data_fingerprint(tmp_path)) == before


def test_fingerprint_digest_changes_when_file_size_changes(tmp_path: Path) -> None:
    path = tmp_path / "clients.csv"
    path.write_text("client_id\nc1\n")
    before = fingerprint_digest(raw_data_fingerprint(tmp_path))

    path.write_text("client_id\nc1\nc2\n")

    assert fingerprint_digest(raw_data_fingerprint(tmp_path)) != before
