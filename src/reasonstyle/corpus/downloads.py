"""Verifying and unpacking downloaded reference-source files.

Raw source files live under the gitignored ``data/sources/raw/``. What is
committed is the manifest: each file's official URL, size, the checksum its
provider publishes, and the SHA-256 we compute ourselves. Anyone can re-fetch
from the pinned URLs and confirm they hold the same bytes.

Two safety rules apply to ZIP archives. Every member path is checked before
anything is written, so no member can escape the target directory. And pickle
members are never extracted: loading a pickle can execute arbitrary code, and
the same data is available as JSON.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

__all__ = [
    "ChecksumMismatch",
    "UnsafeArchive",
    "ExtractionResult",
    "file_digest",
    "git_blob_sha1",
    "provider_digest",
    "safe_extract",
]

PICKLE_SUFFIXES = (".pkl", ".pickle")


class ChecksumMismatch(ValueError):
    """A downloaded file does not match the checksum its provider publishes."""


class UnsafeArchive(ValueError):
    """A ZIP member would be written outside the target directory."""


def file_digest(data: bytes, algorithm: str) -> str:
    return hashlib.new(algorithm, data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    """The SHA-1 git assigns to a file's contents, as GitHub's API reports it."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def provider_digest(data: bytes, algorithm: str) -> str:
    if algorithm == "git_blob_sha1":
        return git_blob_sha1(data)
    if algorithm in ("md5", "sha1", "sha256"):
        return file_digest(data, algorithm)
    raise ValueError(f"unsupported provider checksum algorithm {algorithm!r}")


def verify(data: bytes, *, size: int, algorithm: str, expected: str) -> str:
    """Check size and provider checksum; return our own SHA-256."""
    if len(data) != size:
        raise ChecksumMismatch(f"size {len(data)} bytes, expected {size}")
    actual = provider_digest(data, algorithm)
    if actual != expected:
        raise ChecksumMismatch(f"{algorithm} {actual}, expected {expected}")
    return file_digest(data, "sha256")


@dataclass(frozen=True)
class ExtractionResult:
    extracted: tuple[str, ...]
    skipped_pickles: tuple[str, ...]


def _member_is_safe(name: str) -> bool:
    path = PurePosixPath(name)
    return not (path.is_absolute() or ".." in path.parts or name.startswith("\\")
                or (len(name) > 1 and name[1] == ":"))


def safe_extract(archive: str | Path, target: str | Path) -> ExtractionResult:
    """Extract every non-pickle member, refusing the whole archive if any
    member path is unsafe. Nothing is written until every path is checked."""
    target = Path(target).resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for info in members:
            if not _member_is_safe(info.filename):
                raise UnsafeArchive(f"unsafe member path {info.filename!r}")
            if not (target / info.filename).resolve().is_relative_to(target):
                raise UnsafeArchive(f"member {info.filename!r} escapes the target directory")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise UnsafeArchive(f"member {info.filename!r} is a symbolic link")

        extracted, skipped = [], []
        for info in members:
            if info.filename.lower().endswith(PICKLE_SUFFIXES):
                skipped.append(info.filename)
                continue
            zf.extract(info, target)
            if not info.is_dir():
                extracted.append(info.filename)
    return ExtractionResult(tuple(extracted), tuple(skipped))
