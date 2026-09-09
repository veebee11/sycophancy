"""JSONL input and output for the scenario corpus.

The corpus is stored one JSON object per line. Writing goes through
:func:`reasonstyle.hashing.canonical_json`, so the bytes depend only on the
data: two runs that produce the same records produce the same file, and
:func:`corpus_content_hash` is stable across machines and key orderings.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import ValidationError

from .hashing import canonical_json, content_hash, file_sha256
from .schemas import ScenarioRecord

__all__ = [
    "CorpusError",
    "corpus_content_hash",
    "corpus_file_sha256",
    "dumps_record",
    "load_corpus",
    "save_corpus",
]


class CorpusError(ValueError):
    """A corpus file could not be read as scenario records."""


def dumps_record(record: ScenarioRecord) -> str:
    """One canonical JSON line for a record."""
    return canonical_json(record.model_dump(mode="json", exclude_none=False))


def load_corpus(path: str | Path) -> list[ScenarioRecord]:
    """Read a JSONL corpus, reporting the line number of any bad record."""
    path = Path(path)
    records: list[ScenarioRecord] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
        try:
            records.append(ScenarioRecord.model_validate(payload))
        except ValidationError as exc:
            raise CorpusError(f"{path}:{lineno}: not a valid scenario record:\n{exc}") from exc
    return records


def save_corpus(records: Iterable[ScenarioRecord], path: str | Path) -> Path:
    """Write records as canonical JSONL. Byte-stable for equal input."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{dumps_record(r)}\n" for r in records), encoding="utf-8")
    return path


def corpus_content_hash(records: Sequence[ScenarioRecord]) -> str:
    """Order-independent? No — record order is part of the corpus identity."""
    return content_hash([r.model_dump(mode="json") for r in records])


def corpus_file_sha256(path: str | Path) -> str:
    return file_sha256(path)
