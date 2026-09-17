"""Redrafting the scenarios a curator rejected — one call each, and no more.

A scenario the curator marked ``redraft`` is not broken in a way a machine can
describe: it passed every machine check. What is wrong with it is what the
reviewer wrote down, so that is what the redraft request carries, alongside the
same frozen brief the original was drafted from. The reviewer's *reason* goes to
the model; the reviewer's own preferred wording does not, because the model is
the drafter and text a curator wrote would no longer be generated material.

**The template is versioned and hashed here, not in the experiment
configuration.** Adding it to ``configs/experiment.yaml`` would change the
configuration hash, and every one of the curator's existing approvals binds that
hash — so recording a redraft template there would make fifteen approvals stale
and invalidate a live run. Instead the file's SHA-256 is computed at build time
and recorded in the request, the log and the result, which is where provenance
belongs anyway. Nothing about the experiment's own configuration moves.

**Supersession is explicit.** A redraft records ``supersedes_call_id``: the
scenario it replaces. The current text of a scenario is resolved by following
that chain from the original, never by taking whichever call happens to be
latest in the log.

One call per scenario. There is no automatic second redraft, and a redraft that
comes back unchanged, machine-invalid or rejected leaves the scenario blocking
the gate, for a person to decide about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from ..config import ExperimentConfig
from ..corpus.segmentation import Segmenter
from ..corpus.validate import validate_scenario_text
from ..hashing import file_sha256, sha256_of
from .approvals import REDRAFT, ScenarioApproval
from .pipeline import (
    ACCEPTED,
    NEEDS_MANUAL_REVIEW,
    REFUSED,
    Attempt,
    CallStore,
    PipelineAbort,
    StageResult,
    _send_or_resume,
    _stored_meta,
    recorded_scenarios,
)
from .requests import DraftRequest, RequestError

__all__ = [
    "REDRAFT_TEMPLATE",
    "RedraftTarget",
    "current_scenarios",
    "redraft_request",
    "redraft_targets",
    "redraft_template_sha256",
    "run_redraft_stage",
]

#: Versioned outside the experiment configuration, for the reason in the module
#: docstring: the configured hash is what fifteen approvals are bound to.
REDRAFT_TEMPLATE = Path("prompts/scenario_redraft_v1.txt")
REDRAFT_TEMPLATE_NAME = "scenario_redraft_v1"

#: The response schema, identical in shape to a scenario draft's.
REDRAFT_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["scenario_text"], "properties": {"scenario_text": {"type": "string"}},
}

_PLACEHOLDERS = ("decision_framing", "variant_context", "option_text_opt_1",
                 "option_text_opt_2", "fact_opt_1", "fact_opt_2",
                 "original_scenario_text", "failure_reason", "min_words", "max_words")


@dataclass(frozen=True, slots=True)
class RedraftTarget:
    """One scenario the curator asked to have drafted again."""

    scenario_id: str
    decision_id: str
    variant_id: int
    original_call_id: str
    original_text: str
    failure_reason: str
    failed_judgements: tuple[str, ...]

    @property
    def original_text_sha256(self) -> str:
        return sha256_of(self.original_text)


def redraft_template_sha256(path: str | Path = REDRAFT_TEMPLATE) -> str:
    """The template's own hash, recorded with every redraft it produces."""
    return file_sha256(Path(path))


def redraft_targets(approvals: dict[str, ScenarioApproval],
                    scenarios: dict[str, dict[str, Any]]) -> list[RedraftTarget]:
    """Exactly the scenarios currently marked ``redraft``, and nothing else.

    The set comes from the curator's file. It is never narrowed by a command
    line argument: which scenarios need drafting again is the reviewer's
    finding, not an operator's selection.
    """
    targets = []
    for scenario_id, approval in sorted(approvals.items()):
        if approval.decision != REDRAFT:
            continue
        record = scenarios.get(scenario_id) or {}
        text = record.get("scenario_text")
        if not text:
            raise RequestError(f"{scenario_id} is marked redraft but has no recorded "
                               f"scenario text to replace")
        if approval.call_id != record.get("call_id"):
            raise RequestError(
                f"{scenario_id}: the redraft decision names call {approval.call_id[:12]}, "
                f"but the recorded scenario came from {str(record.get('call_id'))[:12]}")
        if approval.scenario_text_sha256 != sha256_of(text):
            raise RequestError(f"{scenario_id}: the recorded text is not the text the "
                               f"curator reviewed; the decision is stale")
        if not approval.reason:
            raise RequestError(f"{scenario_id}: a redraft needs the reviewer's reason")
        decision_id, _, variant = scenario_id.rpartition("_v")
        targets.append(RedraftTarget(
            scenario_id=scenario_id, decision_id=decision_id, variant_id=int(variant),
            original_call_id=approval.call_id, original_text=text,
            failure_reason=approval.reason,
            failed_judgements=tuple(sorted(
                name for name, value in approval.judgements.items() if value is False))))
    return targets


def redraft_request(topic, target: RedraftTarget, cfg: ExperimentConfig, *,
                    template: str | Path = REDRAFT_TEMPLATE) -> DraftRequest:
    """One redraft call: the frozen brief, the original, and what was wrong.

    The reviewer's reason is passed verbatim; no replacement prose is. The model
    drafts the scenario, as it did the first time.
    """
    path = Path(template)
    if not path.is_file():
        raise RequestError(f"the redraft template {path} is missing")
    variant = topic.variants[f"v{target.variant_id}"]
    facts = {option: getattr(variant.scenario_facts, option) for option in ("opt_1", "opt_2")}
    for option, values in facts.items():
        if len(values) != 1:
            raise RequestError(f"{target.scenario_id}: the pilot gives each option exactly "
                               f"one fact; {option} has {len(values)}")
    words = cfg.raw["corpus"]["scenario_words"]
    body = path.read_text(encoding="utf-8")
    values = {
        "decision_framing": topic.decision_framing,
        "variant_context": variant.context,
        "option_text_opt_1": topic.options["opt_1"],
        "option_text_opt_2": topic.options["opt_2"],
        "fact_opt_1": facts["opt_1"][0],
        "fact_opt_2": facts["opt_2"][0],
        "original_scenario_text": target.original_text,
        "failure_reason": target.failure_reason,
        "min_words": words["min"],
        "max_words": words["max"],
    }
    missing = [name for name in _PLACEHOLDERS if f"${{{name}}}" not in body]
    if missing:
        raise RequestError(f"{path} does not use {missing}")
    prompt = Template(body).substitute(values)
    if "${" in prompt:
        raise RequestError(f"{path} left a placeholder unfilled")

    return DraftRequest(
        kind="scenario_redraft", template_name=REDRAFT_TEMPLATE_NAME,
        template_sha256=file_sha256(path), prompt=prompt, response_schema=REDRAFT_SCHEMA,
        decision_id=target.decision_id, variant_id=target.variant_id,
        # Attempt 2 of this scenario: the original draft was attempt 1.
        attempt=2,
        context={
            "domain": topic.domain,
            "scenario_id": target.scenario_id,
            "supersedes_call_id": target.original_call_id,
            "original_text_sha256": target.original_text_sha256,
            "failure_reason": target.failure_reason,
            "failed_judgements": list(target.failed_judgements),
            "template_path": str(path),
        })


def current_scenarios(store: CallStore,
                      approvals: dict[str, ScenarioApproval] | None = None,
                      ) -> dict[str, dict[str, Any]]:
    """The text that currently stands for each scenario, by supersession.

    A redraft replaces the scenario it names in ``supersedes_call_id``, and
    only that one. Nothing is replaced merely by being the most recent call in
    the log: an unaccepted redraft, or a call for some other scenario, leaves
    the original standing — which is what keeps the curator's fifteen approvals
    valid while the other nine are worked on.
    """
    originals = recorded_scenarios(store)
    redrafts: dict[str, dict[str, Any]] = {}
    for entry in store.log.entries():
        if entry["kind"] != "scenario_redraft" or entry["status"] != "ok":
            continue
        if entry.get("outcome") != ACCEPTED:
            continue                              # a rejected redraft replaces nothing
        scenario_id = f"{entry['decision_id']}_v{entry['variant_id']}"
        superseded = (entry.get("extra") or {}).get("supersedes_call_id")
        original = originals.get(scenario_id) or {}
        if not superseded or superseded != original.get("call_id"):
            continue                              # it supersedes some other draft, not this
        path = store.result_path(entry["call_id"])
        if not path.is_file():
            continue
        fields = json.loads(path.read_text(encoding="utf-8")).get("fields") or {}
        if not fields.get("scenario_text"):
            continue
        redrafts[scenario_id] = {
            **original, "scenario_text": fields["scenario_text"],
            "call_id": entry["call_id"], "outcome": entry.get("outcome"),
            "error_codes": list((entry.get("validation") or {}).get("error_codes") or []),
            "supersedes_call_id": superseded,
            "superseded_text_sha256": (entry.get("extra") or {}).get("original_text_sha256"),
        }
    return {**originals, **redrafts}


def run_redraft_stage(topics, approvals: dict[str, ScenarioApproval], cfg: ExperimentConfig,
                      segmenter: Segmenter, backend, store: CallStore, *,
                      allow_live: bool = False) -> list[StageResult]:
    """One call for each scenario the curator marked ``redraft``. No more.

    No group is drafted here, no scenario outside that set is touched, and there
    is no second redraft: a redraft that comes back unchanged, machine-invalid or
    rejected leaves its scenario blocking the gate for a person to decide about.
    Every redraft, accepted or not, is a new call with its own id, recorded
    against the call it supersedes.
    """
    by_decision = {topic.decision_id: topic for topic in topics}
    targets = redraft_targets(approvals, recorded_scenarios(store))
    results: list[StageResult] = []

    for target in targets:
        topic = by_decision.get(target.decision_id)
        if topic is None:
            raise PipelineAbort(f"no brief for {target.decision_id} in this topic bank")
        request = redraft_request(topic, target, cfg)
        status, fields, error, reused, response = _send_or_resume(
            store, request, backend, cfg, allow_live=allow_live)

        if status == "error":
            store.record_transport_failure(request, error or "transport failure")
            raise PipelineAbort(f"{target.scenario_id}: {error}")
        if status == "refused":
            results.append(StageResult("scenario_redraft", target.decision_id,
                                       target.variant_id, None, REFUSED,
                                       (Attempt(request.call_id, request.kind, 2, status,
                                                REFUSED, request.prompt_sha256, error=error,
                                                reused=reused),)))
            continue

        findings, errors, warnings = (), (), ()
        unchanged = False
        if status == "ok":
            text = fields["scenario_text"]
            unchanged = text.strip() == target.original_text.strip()
            findings = tuple(validate_scenario_text(
                text, cfg, segmenter,
                loc={"decision_id": target.decision_id,
                     "scenario_id": target.scenario_id}))
            errors = tuple(sorted({f.code for f in findings if f.severity == "error"}))
            warnings = tuple(sorted({f.code for f in findings if f.severity == "warning"}))
        # A redraft that returns the original changed nothing the reviewer asked
        # to have changed. It is recorded, and it does not become the scenario.
        outcome = (ACCEPTED if status == "ok" and not errors and not unchanged
                   else NEEDS_MANUAL_REVIEW)
        note = error
        if unchanged:
            note = "the redraft returned the original scenario unchanged"

        store.record(request, status=status, outcome=outcome, fields=fields, error=note,
                     error_codes=errors, warning_codes=warnings, response=response,
                     meta=_stored_meta(store, request) if reused else None,
                     extra={"supersedes_call_id": target.original_call_id,
                            "original_text_sha256": target.original_text_sha256,
                            "redrafted_text_sha256": (sha256_of(fields["scenario_text"])
                                                      if fields else None),
                            "failure_reason": target.failure_reason,
                            "failed_judgements": list(target.failed_judgements),
                            "template_name": REDRAFT_TEMPLATE_NAME,
                            "template_sha256": request.template_sha256,
                            "unchanged_from_original": unchanged})
        results.append(StageResult(
            "scenario_redraft", target.decision_id, target.variant_id, None, outcome,
            (Attempt(request.call_id, request.kind, 2, status, outcome,
                     request.prompt_sha256, errors, warnings, note, reused,
                     no_progress=unchanged),),
            payload=fields if outcome == ACCEPTED else None, findings=findings))
    return results
