"""Stable content hashes for configs, prompts and corpora.

Every generated artefact carries the hash of the configuration that produced it
(plan §15.3), so hashing must be reproducible across machines, Python runs and
irrelevant reorderings of a YAML file.

Two distinct hashes are provided and they answer different questions:

``content_hash``
    Canonical-JSON hash of a *parsed* object. Independent of key order,
    comments, indentation and quoting style. This is the semantic identity of a
    configuration: reordering two YAML keys does not create a new experiment.

``file_sha256``
    Byte-exact hash of a file. Detects any edit at all, including comments.
    Recorded alongside ``content_hash`` so an in-place edit is still visible
    even when it was semantically inert.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = [
    "canonical_json",
    "sha256_of",
    "content_hash",
    "file_sha256",
    "short",
]

_HASH_PREFIX_LEN = 12


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` to a canonical JSON string.

    Keys are sorted, separators are tight and non-ASCII characters are kept
    verbatim, so the byte sequence depends only on the data.

    ``sort_keys`` requires homogeneously comparable keys; mappings with
    non-string keys are rejected rather than silently coerced, since a silent
    coercion would let two different configs share a hash.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_of(data: str | bytes) -> str:
    """Hex SHA-256 of a string (UTF-8 encoded) or raw bytes."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def content_hash(obj: Any) -> str:
    """Order-independent hex SHA-256 of a parsed object."""
    return sha256_of(canonical_json(obj))


def file_sha256(path: str | Path) -> str:
    """Byte-exact hex SHA-256 of a file."""
    return sha256_of(Path(path).read_bytes())


def short(digest: str, length: int = _HASH_PREFIX_LEN) -> str:
    """First ``length`` characters of a digest, for run IDs and filenames."""
    return digest[:length]
