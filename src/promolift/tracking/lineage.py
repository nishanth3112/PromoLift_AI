"""Reproducibility lineage recorded on every tracked run: code, environment, and data.

Deliberately cheap to compute so it can run on every run: git state via the
``git`` CLI, a hash of ``uv.lock``, a size/mtime fingerprint of the raw X5
files instead of a content hash (hashing ~4.3 GB would add tens of seconds to
every run, and raw data is read-only by project rule), and a content hash of
the small persisted train/val/test split.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from promolift.data.loader import Dataset, path_for, project_root
from promolift.data.split import load_split, split_assignment_path, split_content_sha256

UNKNOWN = "unknown"


@dataclass(frozen=True)
class GitState:
    """Commit and working-tree state; ``is_dirty`` is None when git is unavailable."""

    commit: str
    is_dirty: bool | None


@dataclass(frozen=True)
class RawFileFingerprint:
    """Size and modification time of one raw file; both None if the file is missing."""

    filename: str
    size_bytes: int | None
    modified_at: str | None


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def git_state(repo_dir: Path | None = None) -> GitState:
    """Return the HEAD commit and whether the working tree has uncommitted changes.

    Untracked files count as dirty: an uncommitted module a run imports is as
    unreproducible as an uncommitted edit.
    """
    cwd = repo_dir if repo_dir is not None else project_root()
    try:
        commit = _git(["rev-parse", "HEAD"], cwd)
        status = _git(["status", "--porcelain"], cwd)
    except (OSError, subprocess.CalledProcessError):
        return GitState(commit=UNKNOWN, is_dirty=None)
    return GitState(commit=commit, is_dirty=bool(status))


def lockfile_sha256(repo_dir: Path | None = None) -> str | None:
    """SHA-256 of ``uv.lock``, or None if it doesn't exist."""
    path = (repo_dir if repo_dir is not None else project_root()) / "uv.lock"
    if not path.exists():
        return None
    # Normalize line endings so a Windows checkout (core.autocrlf) of the same
    # lockfile hashes identically to the macOS/Linux one.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def split_sha256(processed_dir: Path | None = None) -> str | None:
    """Content hash of the persisted train/val/test split, or None if it doesn't exist."""
    path = split_assignment_path(processed_dir)
    if not path.exists():
        return None
    return split_content_sha256(load_split(path))


def raw_data_fingerprint(base_dir: Path | None = None) -> list[RawFileFingerprint]:
    """Size and UTC modification time of every registered raw X5 file."""
    fingerprints = []
    for dataset in Dataset:
        path = path_for(dataset, base_dir)
        if path.exists():
            stat = path.stat()
            modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
            fingerprints.append(RawFileFingerprint(path.name, stat.st_size, modified_at))
        else:
            fingerprints.append(RawFileFingerprint(path.name, None, None))
    return fingerprints


def fingerprint_digest(fingerprints: list[RawFileFingerprint]) -> str:
    """Short digest over filenames and sizes, for filtering runs by data version.

    Excludes modification times: identical copies of the data on two machines
    have different mtimes but should compare as the same data version.
    """
    payload = json.dumps(
        sorted((fp.filename, fp.size_bytes) for fp in fingerprints), separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def fingerprints_as_dicts(fingerprints: list[RawFileFingerprint]) -> list[dict]:
    """JSON-serializable form of a fingerprint list."""
    return [asdict(fp) for fp in fingerprints]
