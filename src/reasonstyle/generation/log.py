"""The generation log: one line per call, and the raw traffic beside it.

Two artefacts, written before anything is parsed into a corpus record:

``<raw_dir>/<call_id>.json``
    The request as sent and the response as received, verbatim. Never edited.

``generation_log.jsonl``
    One line per call — what was asked, of which model, under which config,
    brief and allocation, and what came back. A refusal or an error is logged
    like any other call. Nothing is silently retried: a retry is a new attempt
    with its own line.

No credential ever reaches either file: the request payload is stored without
headers, and the key is read from the environment by the backend alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..hashing import sha256_of

__all__ = ["GenerationLog", "LogEntry", "utc_now"]


def utc_now() -> str:
    """Wall-clock time, used only in the log and in generation metadata.

    Generated review artefacts never carry a timestamp; a record of what was
    sent to a provider, and when, is the one place a real clock belongs.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class LogEntry:
    call_id: str
    kind: str
    attempt: int
    decision_id: str
    variant_id: int
    supported_option: str | None
    template_name: str
    template_sha256: str
    prompt_sha256: str
    model: str                       # the repository id we asked for
    model_returned: str | None       # what the server says it served
    request_fields: dict[str, Any]   # everything sent except the prompt itself
    config_content_hash: str
    topic_bank_content_hash: str
    allocation_content_hash: str | None
    response_sha256: str | None
    stop_reason: str | None
    usage: dict[str, Any] | None
    status: str                      # ok | rejected | error | refused | validation_failed
    error: str | None
    generated_at: str
    outcome: str | None = None       # e.g. needs_manual_review
    #: What the validator made of the parsed response, when the caller ran it.
    #: A response that parsed but failed validation is NOT an accepted result,
    #: so the status says so and the codes are kept here rather than only on a
    #: terminal that scrolls away.
    validation: dict[str, Any] | None = None
    #: The weights that actually produced this, and the machine that ran them.
    #: The seed is part of the record, not a reproducibility guarantee.
    model_revision: str | None = None
    seed: int | None = None
    gpu: str | None = None
    runtime: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenerationLog:
    """Append-only JSONL log with its raw-traffic directory."""

    def __init__(self, path: str | Path, raw_dir: str | Path | None = None):
        self.path = Path(path)
        self.raw_dir = Path(raw_dir) if raw_dir else self.path.parent / "raw"

    def append(self, entry: LogEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    def store_raw(self, call_id: str, request_payload: dict[str, Any],
                  response: Any, *, prompt: str, meta: dict[str, Any] | None = None) -> Path:
        """Store the verbatim request and response for one call.

        ``meta`` records what the backend reported about the call itself — the
        finish reason, the model it served, token usage — beside the untouched
        response. A recovery after a crash reads it rather than guessing at the
        shape of a response it did not receive itself.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / f"{call_id}.json"
        path.write_text(json.dumps(
            {"call_id": call_id, "prompt": prompt,
             "request_payload": request_payload, "call_meta": meta or {},
             "response": response},
            indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def entry_for(self, call_id: str) -> dict[str, Any] | None:
        """The logged line for one call, if it was written before a crash."""
        return next((e for e in self.entries() if e["call_id"] == call_id), None)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def attempts_for(self, decision_id: str, variant_id: int, option: str) -> int:
        """How many calls this group has already consumed."""
        return sum(1 for e in self.entries()
                   if e["decision_id"] == decision_id and e["variant_id"] == variant_id
                   and e["supported_option"] == option and e["kind"] in ("group", "repair"))

    @staticmethod
    def response_digest(response: Any) -> str:
        return sha256_of(json.dumps(response, sort_keys=True, ensure_ascii=False))
