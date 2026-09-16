"""The curator-approval gate: a scenario is drafted on only after a person says so.

A group's four counterarguments are built *from* a scenario. If the scenario is
later found wanting, everything drafted on top of it goes with it — so the
approval comes first, and it binds to the exact text that was read.

**An approval binds four things**, and a change to any one of them makes it
stale rather than silently still valid:

``scenario_text_sha256``   the exact text the curator read, byte for byte
``call_id``                the accepted drafting call it came from
``config_content_hash``    the configuration it was drafted under
``topic_bank_content_hash``the briefs it was drafted from

An edited scenario therefore has no approval at all, which is the point: there
is no way to approve one text and generate from another.

**Approval is a human judgement and never overrides a machine check.** A
scenario that fails validation cannot be approved into use: the gate reports it
as ``blocked_by_machine_errors`` however the file reads. Approval adds a
requirement, it never removes one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ..hashing import sha256_of

__all__ = [
    "APPROVED",
    "ApprovalError",
    "BLOCKED",
    "NEEDS_MANUAL_REVIEW",
    "PENDING",
    "REDRAFT",
    "STALE",
    "ScenarioApproval",
    "approval_status",
    "gate_problems",
    "load_approvals",
    "save_approvals",
]

#: Gate states. Only APPROVED lets a group be drafted.
APPROVED = "approved"
PENDING = "pending"                  # no record at all
STALE = "stale"                      # a record, but its bindings no longer hold
REDRAFT = "redraft"                  # the curator asked for another draft
NEEDS_MANUAL_REVIEW = "needs_manual_review"
BLOCKED = "blocked_by_machine_errors"

#: Every judgement must be true before a scenario may be approved. They are the
#: scenario-level questions no validator can answer.
REQUIRED_JUDGEMENTS = (
    "both_facts_stated",
    "no_added_facts_or_quantities",
    "options_neutral_and_equal_weight",
    "no_recommendation_or_hint",
    "self_contained",
    "no_letter_or_list_labels",
    "length_without_filler",
)

#: Decisions a file may record. ``pending`` is what a generated template
#: carries: it loads, so a partly reviewed file is usable while the curator
#: works through it, and it never satisfies the gate.
_DECISIONS = (APPROVED, PENDING, REDRAFT, NEEDS_MANUAL_REVIEW)


class ApprovalError(ValueError):
    """An approvals file could not be read, or contradicts itself."""


@dataclass(frozen=True, slots=True)
class ScenarioApproval:
    """One curator decision about one scenario, bound to its exact text."""

    scenario_id: str
    scenario_text_sha256: str
    call_id: str
    config_content_hash: str
    topic_bank_content_hash: str
    decision: str
    judgements: dict[str, bool | None]
    decided_by: str | None
    decided_at: date | None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["decided_at"] = self.decided_at.isoformat() if self.decided_at else None
        return out

    @property
    def judgements_complete(self) -> bool:
        return all(self.judgements.get(name) is True for name in REQUIRED_JUDGEMENTS)


def _approval_from(scenario_id: str, raw: dict[str, Any]) -> ScenarioApproval:
    decision = raw.get("decision")
    if decision not in _DECISIONS:
        raise ApprovalError(f"{scenario_id}: decision {decision!r} is not one of "
                            f"{list(_DECISIONS)}")
    required = ("scenario_text_sha256", "call_id", "config_content_hash",
                "topic_bank_content_hash", "decision", "judgements")
    if decision == APPROVED:
        # An approval is a person's act, and says who made it and when.
        required += ("decided_by", "decided_at")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ApprovalError(f"{scenario_id}: a {decision!r} record is missing {missing}")
    if decision == APPROVED and not (raw.get("decided_by") and raw.get("decided_at")):
        raise ApprovalError(f"{scenario_id}: an approval needs its reviewer and date")
    if decision in (REDRAFT, NEEDS_MANUAL_REVIEW) and not raw.get("reason"):
        raise ApprovalError(f"{scenario_id}: a decision of {decision!r} needs a reason")
    decided_at = raw.get("decided_at")
    return ScenarioApproval(
        scenario_id=scenario_id,
        scenario_text_sha256=raw["scenario_text_sha256"],
        call_id=raw["call_id"],
        config_content_hash=raw["config_content_hash"],
        topic_bank_content_hash=raw["topic_bank_content_hash"],
        decision=raw["decision"],
        judgements=dict(raw["judgements"] or {}),
        decided_by=raw.get("decided_by"),
        decided_at=(decided_at if isinstance(decided_at, date) or decided_at is None
                    else date.fromisoformat(decided_at)),
        reason=raw.get("reason"))


def load_approvals(path: str | Path) -> dict[str, ScenarioApproval]:
    """Read the curator's approvals file. Absent is empty, not an error."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ApprovalError(f"{path} is not readable YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ApprovalError(f"{path} must map scenario_id -> approval")
    return {scenario_id: _approval_from(scenario_id, record)
            for scenario_id, record in sorted(raw.items())}


def save_approvals(approvals: dict[str, ScenarioApproval], path: str | Path) -> Path:
    """Write approvals back, deterministically. Used by tests and tooling; the
    curator's own file is written by hand."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {scenario_id: {k: v for k, v in approval.as_dict().items() if k != "scenario_id"}
            for scenario_id, approval in sorted(approvals.items())}
    path.write_text(yaml.safe_dump(body, sort_keys=True, allow_unicode=True), encoding="utf-8")
    return path


def approval_status(scenario_id: str, scenario_text: str, call_id: str,
                    approvals: dict[str, ScenarioApproval], *, config_content_hash: str,
                    topic_bank_content_hash: str,
                    machine_errors: int = 0) -> tuple[str, tuple[str, ...]]:
    """``(state, reasons)`` for one scenario.

    ``machine_errors`` comes from the validator. It is checked first and it
    wins: a human cannot approve a scenario past a machine error.
    """
    if machine_errors:
        return BLOCKED, (f"{machine_errors} machine error(s): approval cannot override the "
                         f"validator; fix or redraft the scenario",)
    approval = approvals.get(scenario_id)
    if approval is None:
        return PENDING, ("no approval recorded",)
    if approval.decision == PENDING:
        return PENDING, ("recorded but not yet reviewed",)
    if approval.decision != APPROVED:
        return approval.decision, (approval.reason or "",)
    if not (approval.decided_by and approval.decided_at):
        return NEEDS_MANUAL_REVIEW, ("the approval does not say who approved it, or when",)

    reasons = []
    if approval.scenario_text_sha256 != sha256_of(scenario_text):
        reasons.append("the scenario text has changed since it was approved")
    if approval.call_id != call_id:
        reasons.append(f"approved call {approval.call_id[:12]} is not the accepted call "
                       f"{call_id[:12]}")
    if approval.config_content_hash != config_content_hash:
        reasons.append("approved under a different configuration")
    if approval.topic_bank_content_hash != topic_bank_content_hash:
        reasons.append("approved against a different topic bank")
    if reasons:
        return STALE, tuple(reasons)
    if not approval.judgements_complete:
        missing = [n for n in REQUIRED_JUDGEMENTS if approval.judgements.get(n) is not True]
        return NEEDS_MANUAL_REVIEW, (f"judgements not all affirmed: {missing}",)
    return APPROVED, ()


def gate_problems(scenarios: dict[str, dict[str, Any]],
                  approvals: dict[str, ScenarioApproval], *, config_content_hash: str,
                  topic_bank_content_hash: str) -> list[str]:
    """Why group drafting may not begin. Empty means every scenario is approved.

    ``scenarios`` maps scenario_id to ``{"scenario_text": …, "call_id": …,
    "machine_errors": int}``. The gate is all-or-nothing by design: excluding a
    scenario would unbalance the marker allocation, which is built over every
    decision at once.
    """
    problems = []
    for scenario_id, info in sorted(scenarios.items()):
        state, reasons = approval_status(
            scenario_id, info["scenario_text"], info["call_id"], approvals,
            config_content_hash=config_content_hash,
            topic_bank_content_hash=topic_bank_content_hash,
            machine_errors=info.get("machine_errors", 0))
        if state != APPROVED:
            detail = "; ".join(r for r in reasons if r)
            problems.append(f"{scenario_id}: {state}" + (f" ({detail})" if detail else ""))
    return problems
