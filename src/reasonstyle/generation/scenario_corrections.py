"""Audited human corrections to scenario drafts.

Generated evidence is immutable: a correction never rewrites a result or raw
response.  It names the exact model call and text being corrected, records the
replacement and its curator, and is applied only after the replacement passes
the ordinary scenario validator.  The scenario keeps the source call id while
its approval binds the corrected text hash, so both the model source and the
human edit remain visible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ..config import ExperimentConfig
from ..corpus.segmentation import Segmenter
from ..corpus.validate import validate_scenario_text
from ..hashing import sha256_of

__all__ = [
    "ScenarioCorrection",
    "ScenarioCorrectionError",
    "apply_scenario_corrections",
    "load_scenario_corrections",
    "save_scenario_corrections",
]

_STATES = ("approved", "pending", "rejected")
_FIELDS = (
    "scenario_id", "original_call_id", "original_text", "original_text_sha256",
    "corrected_text", "corrected_text_sha256", "editor", "reason", "decided_at",
    "approval_state",
)


class ScenarioCorrectionError(ValueError):
    """A scenario correction is incomplete, stale or machine-invalid."""


@dataclass(frozen=True, slots=True)
class ScenarioCorrection:
    """One human edit, bound to the exact generated scenario it corrects."""

    scenario_id: str
    original_call_id: str
    original_text: str
    corrected_text: str
    editor: str
    reason: str
    decided_at: date
    approval_state: str = "pending"
    validation_error_codes: tuple[str, ...] = ()
    validation_warning_codes: tuple[str, ...] = ()

    @property
    def original_text_sha256(self) -> str:
        return sha256_of(self.original_text)

    @property
    def corrected_text_sha256(self) -> str:
        return sha256_of(self.corrected_text)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["decided_at"] = self.decided_at.isoformat()
        out["original_text_sha256"] = self.original_text_sha256
        out["corrected_text_sha256"] = self.corrected_text_sha256
        out["validation_error_codes"] = list(self.validation_error_codes)
        out["validation_warning_codes"] = list(self.validation_warning_codes)
        return out


def _from(raw: dict[str, Any]) -> ScenarioCorrection:
    missing = [name for name in _FIELDS if raw.get(name) in (None, "")]
    if missing:
        raise ScenarioCorrectionError(f"a scenario correction is missing {missing}")
    if raw["approval_state"] not in _STATES:
        raise ScenarioCorrectionError(
            f"approval_state {raw['approval_state']!r} is not one of {list(_STATES)}")
    if raw["original_text"] == raw["corrected_text"]:
        raise ScenarioCorrectionError(f"{raw['scenario_id']}: the correction changes nothing")
    original_hash = sha256_of(raw["original_text"])
    corrected_hash = sha256_of(raw["corrected_text"])
    if raw["original_text_sha256"] != original_hash:
        raise ScenarioCorrectionError(
            f"{raw['scenario_id']}: original_text_sha256 does not match original_text")
    if raw["corrected_text_sha256"] != corrected_hash:
        raise ScenarioCorrectionError(
            f"{raw['scenario_id']}: corrected_text_sha256 does not match corrected_text")
    decided_at = raw["decided_at"]
    return ScenarioCorrection(
        scenario_id=raw["scenario_id"], original_call_id=raw["original_call_id"],
        original_text=raw["original_text"], corrected_text=raw["corrected_text"],
        editor=raw["editor"], reason=raw["reason"],
        decided_at=(decided_at if isinstance(decided_at, date)
                    else date.fromisoformat(decided_at)),
        approval_state=raw["approval_state"],
        validation_error_codes=tuple(raw.get("validation_error_codes") or ()),
        validation_warning_codes=tuple(raw.get("validation_warning_codes") or ()),
    )


def load_scenario_corrections(path: str | Path) -> list[ScenarioCorrection]:
    """Load the correction ledger; an absent file means no corrections."""
    path = Path(path)
    if not path.is_file():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    except yaml.YAMLError as exc:
        raise ScenarioCorrectionError(f"{path} is not readable YAML: {exc}") from exc
    if not isinstance(raw, list):
        raise ScenarioCorrectionError(f"{path} must be a list of correction records")
    corrections = [_from(item) for item in raw]
    seen: set[str] = set()
    for correction in corrections:
        if correction.scenario_id in seen:
            raise ScenarioCorrectionError(
                f"two current corrections for {correction.scenario_id}; history belongs in git")
        seen.add(correction.scenario_id)
    return corrections


def save_scenario_corrections(corrections: list[ScenarioCorrection],
                              path: str | Path) -> Path:
    """Write a deterministic correction ledger."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump([c.as_dict() for c in corrections], sort_keys=True,
                                   allow_unicode=True), encoding="utf-8")
    return path


def apply_scenario_corrections(
    scenarios: dict[str, dict[str, Any]],
    corrections: list[ScenarioCorrection],
    cfg: ExperimentConfig,
    segmenter: Segmenter,
) -> dict[str, dict[str, Any]]:
    """Return current scenario records with approved corrections applied.

    Every correction must match the exact current model text and call.  It is
    then checked by the same scenario validator used at generation time.  A
    human decision cannot override a machine error.
    """
    out = {scenario_id: dict(record) for scenario_id, record in scenarios.items()}
    for correction in corrections:
        record = out.get(correction.scenario_id)
        if not record or not record.get("scenario_text"):
            raise ScenarioCorrectionError(
                f"{correction.scenario_id}: no generated scenario exists to correct")
        if correction.original_call_id != record.get("call_id"):
            raise ScenarioCorrectionError(
                f"{correction.scenario_id}: correction names call "
                f"{correction.original_call_id[:12]}, current source is "
                f"{str(record.get('call_id'))[:12]}")
        if correction.original_text != record["scenario_text"]:
            raise ScenarioCorrectionError(
                f"{correction.scenario_id}: original_text is not what the model returned")
        if correction.approval_state != "approved":
            raise ScenarioCorrectionError(
                f"{correction.scenario_id}: correction is {correction.approval_state}, "
                f"not approved")
        findings = validate_scenario_text(
            correction.corrected_text, cfg, segmenter,
            loc={"scenario_id": correction.scenario_id})
        errors = sorted({f.code for f in findings if f.severity == "error"})
        warnings = sorted({f.code for f in findings if f.severity == "warning"})
        if errors:
            raise ScenarioCorrectionError(
                f"{correction.scenario_id}: corrected text has machine errors {errors}")
        out[correction.scenario_id] = {
            **record,
            "scenario_text": correction.corrected_text,
            "outcome": "accepted",
            "error_codes": [],
            "warning_codes": warnings,
            "scenario_correction": correction.as_dict(),
        }
    return out
