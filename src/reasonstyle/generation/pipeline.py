"""Bounded drafting: one scenario call, one group draft, at most two repairs.

The controller that turns requests into *recorded attempts*. It knows four
things and nothing else: what to send next, when to stop, what to write down,
and what counts as machine-valid.

**Budget.** A group gets `corpus.repair.max_calls_per_group` calls — one draft
plus two repairs. A group that is still failing after them is
``needs_manual_review`` and is never hand-corrected here. A scenario gets one
call: there is no scenario repair template, and a scenario that fails its
machine checks stops its own groups from being drafted at all.

**What consumes budget.** A response that arrived — parsed or schema-rejected —
consumes one call. A transport failure or a missing authorisation does not: the
first aborts the run for a human to look at, the second sends nothing.

**Machine-valid is not approved.** ``accepted`` here means the validator found
no *errors*. Every group still carries its unconditional human-review codes,
and no code in this module can discharge them.

**Resumable.** Every call is content-addressed by ``call_id``, and its raw
traffic and its result are written to disk before the next call begins. A
restart re-reads them instead of asking the model again, so an interrupted run
never pays for the same call twice — and never silently replaces a text that
sampled decoding would not reproduce.

Nothing here contacts a server by itself: it drives whatever backend it is
given, and the live authorisations are checked by the backend and the caller.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..corpus.findings import Finding
from ..corpus.schemas import CORE_CONDITIONS, Cell, DirectionBlock
from ..corpus.segmentation import Segmenter
from ..corpus.validate import validate_group, validate_scenario_text
from .allocation import GroupAllocation
from .backends import BackendError, BackendUnavailable, LiveCallRefused, vllm_payload
from .environment import CachedModel, describe_run
from .log import GenerationLog, LogEntry, utc_now
from .requests import DraftRequest, ResponseRejected, group_request, parse_response, repair_request
from .requests import scenario_request

__all__ = [
    "ABORTED",
    "ACCEPTED",
    "Attempt",
    "CallStore",
    "NEEDS_MANUAL_REVIEW",
    "PipelineAbort",
    "REFUSED",
    "TRANSPORT_ERROR",
    "StageResult",
    "group_diagnostics",
    "draft_group",
    "draft_scenario",
    "run_pilot",
]

#: Research outcomes. Kept separate from the call ``status`` (ok, rejected,
#: error, refused): a response can arrive intact and still not be usable.
ACCEPTED = "accepted"
REPAIR_NEEDED = "repair_needed"
NEEDS_MANUAL_REVIEW = "needs_manual_review"
ABORTED = "aborted"
REFUSED = "refused"
#: A transport failure: audited, never a completed generation attempt.
TRANSPORT_ERROR = "transport_error"
#: The outcome written when a crashed call is re-derived from its stored raw
#: response. The controller re-judges it and records the research outcome.
RECOVERED = "recovered_from_raw"
SKIPPED_SCENARIO_NOT_ACCEPTED = "skipped_scenario_not_accepted"

#: Statuses that consume one call of a budget. A transport error or a refusal
#: never does: nothing usable came back, and nothing was decided.
_CONSUMES_BUDGET = ("ok", "rejected")


class PipelineAbort(RuntimeError):
    """The run stopped for a reason a person must look at, mid-way."""


@dataclass(frozen=True, slots=True)
class Attempt:
    """One call: what was asked, what came back, and what was decided."""

    call_id: str
    kind: str                       # scenario | group | repair
    attempt: int
    status: str                     # ok | rejected | error | refused
    outcome: str
    prompt_sha256: str
    error_codes: tuple[str, ...] = ()
    warning_codes: tuple[str, ...] = ()
    error: str | None = None
    #: True when a restart re-read this call instead of sending it again.
    reused: bool = False
    #: True when a repair returned the four bodies it was given, unchanged.
    no_progress: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StageResult:
    """What became of one scenario or one group."""

    kind: str                       # scenario | group
    decision_id: str
    variant_id: int
    supported_option: str | None
    outcome: str
    attempts: tuple[Attempt, ...] = ()
    #: The accepted content: ``{"scenario_text": …}`` or the four bodies.
    payload: dict[str, Any] | None = None
    findings: tuple[Finding, ...] = ()

    @property
    def calls_made(self) -> int:
        return len(self.attempts)

    @property
    def accepted(self) -> bool:
        return self.outcome == ACCEPTED

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "decision_id": self.decision_id,
                "variant_id": self.variant_id, "supported_option": self.supported_option,
                "outcome": self.outcome, "calls_made": self.calls_made,
                "attempts": [a.as_dict() for a in self.attempts],
                "payload": self.payload,
                "error_codes": sorted({c for a in self.attempts for c in a.error_codes})}


def _meta_from(response: Any) -> dict[str, Any]:
    """What the backend said about the call, kept beside the raw response."""
    if response is None:
        return {}
    return {"stop_reason": getattr(response, "stop_reason", None),
            "model_returned": getattr(response, "model_returned", None),
            "usage": getattr(response, "usage", None)}


def _content_from_raw(raw: Any) -> Any:
    """The model's structured answer, out of a stored raw response.

    Two shapes are stored: an OpenAI-compatible reply from the local server,
    and the stand-in a test backend records. Both are read here so that a
    resumed run can recover a call without sending it again.
    """
    if isinstance(raw, dict):
        if "choices" in raw:
            message = (raw["choices"][0] or {}).get("message") or {}
            content = message.get("content")
            return json.loads(content) if isinstance(content, str) else content
        if "content" in raw:
            return raw["content"]
    return None


@dataclass
class CallStore:
    """Where one run's calls are written, and where a restart reads them.

    ``raw/<call_id>.json``      the request and the response, verbatim
    ``results/<call_id>.json``  the parsed fields, status, outcome and codes
    ``generation_log.jsonl``    one appended line per call

    The raw file is written before the result and the log line, so a crash in
    between leaves evidence that the call happened; the result is rebuilt from
    it on the next run rather than re-sent.
    """

    directory: Path
    cfg: ExperimentConfig
    model: str = ""
    #: Provenance of the weights and the machine that served them. Empty in an
    #: offline test; required by the live entry points, which check the three
    #: revisions agree before the first call.
    cached: CachedModel | None = None
    server: dict[str, Any] | None = None
    endpoint: str | None = None
    topic_bank_content_hash: str | None = None
    allocation_content_hash: str | None = None

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.log = GenerationLog(self.directory / "generation_log.jsonl",
                                 raw_dir=self.directory / "raw")
        self.model = self.model or self.cfg.raw["models"]["generator"]["model"]["repo_id"]

    # -- where things live ---------------------------------------------------
    def result_path(self, call_id: str) -> Path:
        return self.directory / "results" / f"{call_id}.json"

    def raw_path(self, call_id: str) -> Path:
        return self.log.raw_dir / f"{call_id}.json"

    def completed_entry_for(self, call_id: str) -> dict[str, Any] | None:
        """The log line for a call that actually produced a response.

        A transport failure is logged under the same ``call_id`` — the request
        was identical — but it completed nothing, so it must not stop the
        retried attempt from recording its own outcome.
        """
        return next((e for e in self.log.entries()
                     if e["call_id"] == call_id and e.get("outcome") != TRANSPORT_ERROR), None)

    def _read_raw(self, call_id: str) -> dict[str, Any] | None:
        path = self.raw_path(call_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def recover(self, request: DraftRequest) -> dict[str, Any] | None:
        """A call that already happened, re-derived from what is on disk.

        A response that was received is never asked for again, whichever
        artefact a crash left behind:

        * result and log line present  -> returned as it is, nothing appended;
        * result but no log line       -> the log line is rebuilt from it;
        * raw response only            -> the stored response is re-parsed with
          this request's own schema, and the result and log line are written.

        Re-parsing matters: a truncated or schema-breaking response recovers as
        ``rejected``, exactly as it would have been at the time, and never as an
        accepted draft. Transport errors leave no result file at all, so they
        recover as "nothing happened" and the call may be made again.
        """
        path = self.result_path(request.call_id)
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            if self.completed_entry_for(request.call_id) is None:
                # Crash after the result, before the log: the decision is
                # already made, so the missing line is rebuilt from it.
                self._append_log(request, result, recovered=True)
            return result
        if self._read_raw(request.call_id) is not None:
            # Crash after the raw response, before anything else. The stored
            # response is re-judged here; the caller records the research
            # outcome, which writes the result and the single log line.
            return self._result_from_raw(request)
        return None

    def _result_from_raw(self, request: DraftRequest) -> dict[str, Any]:
        """Re-judge a stored response under the request's own schema."""
        raw = self._read_raw(request.call_id) or {}
        meta = raw.get("call_meta") or {}
        stop_reason = meta.get("stop_reason")
        content = _content_from_raw(raw.get("response"))
        status, fields, error = "ok", None, None
        if stop_reason not in (None, "stop"):
            status, error = "rejected", f"finish_reason={stop_reason!r}, not 'stop'"
        else:
            try:
                fields = parse_response(request, content)
            except (ResponseRejected, TypeError) as exc:
                status, fields, error = "rejected", None, str(exc)
        result = self._result(request, status=status, outcome=RECOVERED, fields=fields,
                              error=error, meta=meta, recovered=True)
        self._write_result(result)
        return result

    # -- writing -------------------------------------------------------------
    def _result(self, request: DraftRequest, *, status: str, outcome: str, fields: Any = None,
                error: str | None = None, error_codes: tuple[str, ...] = (),
                warning_codes: tuple[str, ...] = (), meta: dict[str, Any] | None = None,
                recovered: bool = False) -> dict[str, Any]:
        return {"call_id": request.call_id, "kind": request.kind,
                "attempt": request.attempt, "decision_id": request.decision_id,
                "variant_id": request.variant_id,
                "supported_option": request.supported_option,
                "prompt_sha256": request.prompt_sha256, "status": status,
                "outcome": outcome, "error": error,
                "error_codes": list(error_codes), "warning_codes": list(warning_codes),
                "fields": fields, "call_meta": meta or {}, "recovered": recovered,
                "recorded_at": utc_now()}

    def _write_result(self, result: dict[str, Any]) -> None:
        path = self.result_path(result["call_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    def _provenance(self, request: DraftRequest, *, response_sha256: str | None = None,
                    usage: dict[str, Any] | None = None) -> dict[str, Any]:
        """What is known about a call before its response — which is everything
        except what the response itself carries.

        Built once and used by both a completed call and a failed transport
        attempt: a call that never returned is still fully described by the
        request that was sent, the weights that were loaded and the machine that
        was serving them.
        """
        env = describe_run(
            self.cfg, cached=self.cached,
            endpoint=self.endpoint or self.cfg.raw["models"]["generator"]["vllm"]["base_url"],
            server=self.server, prompt_sha256=request.prompt_sha256,
            response_sha256=response_sha256,
            input_tokens=(usage or {}).get("prompt_tokens"),
            output_tokens=(usage or {}).get("completion_tokens"))
        payload = {k: v for k, v in vllm_payload(request, self.cfg).items() if k != "messages"}
        return {
            "request_fields": {**payload, "attempt": request.attempt, "kind": request.kind},
            "config_content_hash": self.cfg.content_hash,
            "topic_bank_content_hash": str(self.topic_bank_content_hash),
            "allocation_content_hash": self.allocation_content_hash,
            "model_revision": self.cached.revision if self.cached else None,
            "seed": self.cfg.raw["models"]["generator"]["decoding"]["seed"],
            "gpu": (self.server or {}).get("gpu_name"),
            "runtime": env.as_dict(),
        }

    def _append_log(self, request: DraftRequest, result: dict[str, Any], *,
                    recovered: bool = False) -> None:
        """One append-only line per call, carrying the whole provenance block."""
        meta = result.get("call_meta") or {}
        fields = result.get("fields")
        response_sha = GenerationLog.response_digest(fields) if fields else None
        provenance = self._provenance(request, response_sha256=response_sha,
                                      usage=meta.get("usage"))
        self.log.append(LogEntry(
            call_id=request.call_id, kind=request.kind, attempt=request.attempt,
            decision_id=request.decision_id, variant_id=request.variant_id,
            supported_option=request.supported_option,
            template_name=request.template_name, template_sha256=request.template_sha256,
            prompt_sha256=request.prompt_sha256, model=self.model,
            model_returned=meta.get("model_returned"),
            response_sha256=response_sha,
            stop_reason=meta.get("stop_reason"), usage=meta.get("usage"),
            status=result["status"], error=result.get("error"), generated_at=utc_now(),
            **provenance,
            validation={"error_codes": result.get("error_codes", []),
                        "warning_codes": result.get("warning_codes", []),
                        "machine_valid": result["status"] == "ok"
                        and not result.get("error_codes"),
                        "no_progress": bool(result.get("no_progress")),
                        "recovered_after_interruption": recovered},
            outcome=result["outcome"]))

    def record(self, request: DraftRequest, *, status: str, outcome: str,
               fields: Any = None, error: str | None = None,
               error_codes: tuple[str, ...] = (), warning_codes: tuple[str, ...] = (),
               response: Any = None, meta: dict[str, Any] | None = None,
               no_progress: bool = False) -> dict[str, Any]:
        """Write the result file and append exactly one log line for a call."""
        meta = meta or _meta_from(response)
        result = self._result(request, status=status, outcome=outcome, fields=fields,
                              error=error, error_codes=error_codes,
                              warning_codes=warning_codes, meta=meta,
                              recovered=bool(meta) and response is None)
        result["no_progress"] = no_progress
        self._write_result(result)
        if self.completed_entry_for(request.call_id) is None:
            self._append_log(request, result)
        # Else: this call is already in the append-only log, from this run or an
        # earlier one. The result file keeps the current decision; the log is
        # never appended to twice for one call_id.
        return result

    def record_transport_failure(self, request: DraftRequest, error: str) -> None:
        """Audit a transport failure WITHOUT completing the attempt.

        No result file is written, so the same attempt may contact the backend
        again after a restart and does not lose one of its budget positions.
        The log keeps the aborted transport event, so the failure stays visible.
        """
        self.log.append(LogEntry(
            call_id=request.call_id, kind=request.kind, attempt=request.attempt,
            decision_id=request.decision_id, variant_id=request.variant_id,
            supported_option=request.supported_option,
            template_name=request.template_name, template_sha256=request.template_sha256,
            prompt_sha256=request.prompt_sha256, model=self.model, model_returned=None,
            # Everything known before the request went out. Only what a response
            # would have carried — model returned, stop reason, usage, response
            # hash — is null, because none of it exists.
            **self._provenance(request),
            response_sha256=None, stop_reason=None, usage=None,
            status="error", error=error, generated_at=utc_now(),
            validation={"error_codes": [], "warning_codes": [], "machine_valid": False,
                        "consumed_budget": False},
            outcome=TRANSPORT_ERROR))


def _send_or_resume(store: CallStore, request: DraftRequest, backend, cfg: ExperimentConfig,
                    *, allow_live: bool) -> tuple[str, Any, str | None, bool, Any]:
    """``(status, fields, error, reused, response)`` for one call.

    A completed call is never sent again: its ``call_id`` is content-addressed,
    so the same request in a resumed run resolves to the same recorded result.
    """
    done = store.recover(request)
    if done is not None:
        return (done["status"], done.get("fields"), done.get("error"), True, None)

    try:
        response = backend.send(request, cfg, allow_live=allow_live)
    except LiveCallRefused as exc:
        return ("refused", None, str(exc), False, None)
    except (BackendUnavailable, BackendError) as exc:
        return ("error", None, str(exc), False, None)

    # The raw traffic and what the backend reported about it, written BEFORE the
    # result and the log line, so a crash in between leaves enough to recover.
    payload = {k: v for k, v in vllm_payload(request, cfg).items() if k != "messages"}
    store.log.store_raw(request.call_id, payload, response.raw, prompt=request.prompt,
                        meta=_meta_from(response))
    if response.stop_reason not in (None, "stop"):
        # Truncation is not a draft: there is nothing complete to repair.
        return ("rejected", None, f"finish_reason={response.stop_reason!r}, not 'stop'",
                False, response)
    try:
        fields = parse_response(request, response.content)
    except ResponseRejected as exc:
        return ("rejected", None, str(exc), False, response)
    return ("ok", fields, None, False, response)


def _stored_meta(store: CallStore, request: DraftRequest) -> dict[str, Any]:
    """The backend metadata a recovered call already has on disk."""
    raw = store._read_raw(request.call_id) or {}
    return raw.get("call_meta") or {}


def _block_from(bodies: dict[str, str], allocation: GroupAllocation) -> DirectionBlock:
    """The four cells, with the marker fields taken from the ALLOCATION.

    Never from the response: the marker assignment is part of the design, and a
    model does not get to change it by saying something different.
    """
    return DirectionBlock(
        supported_option=allocation.supported_option,
        marker_family=allocation.marker_family,
        marker_string=allocation.marker_string,
        marker_realization_id=allocation.marker_realization_id,
        cells={c: Cell(condition=c, body=bodies[c], markers_present=c in ("RS", "NS"),
                       marker_family=allocation.marker_family if c in ("RS", "NS") else None)
               for c in CORE_CONDITIONS})


def _codes(findings, severity: str) -> tuple[str, ...]:
    return tuple(sorted({f.code for f in findings if f.severity == severity}))


def group_diagnostics(bodies: dict[str, str], opening: str, cfg: ExperimentConfig,
                      segmenter: Segmenter) -> str:
    """The measurements behind the findings, per condition.

    A repair that is handed only error codes has to guess what to change. These
    are the same numbers the validator used: body sentence counts against the
    configured rule, body word counts against the configured ratios, and which
    conditions are the long and the short ones. Body and full-text counts are
    labelled separately, because the opening sentence makes every full-text
    count one higher and it is the BODY the rules are about.

    Two ratios, and they are not interchangeable. ``ratio_warn`` (1.10) is the
    matching target the drafting prompts ask for; ``ratio_fail`` (1.15) is where
    the validator raises an error. The actionable advice — what to shorten, what
    to lengthen, by how much — is computed from the **target**, so that a repair
    aiming at it lands inside the rule rather than on its edge. Both numbers are
    stated, and neither is weakened.
    """
    import re as _re
    word_re = _re.compile(cfg.parsed.matching.words.word_regex)
    required = cfg.raw["corpus"]["body_sentences"]
    ratio_warn = cfg.parsed.matching.words.ratio_warn
    ratio_fail = cfg.parsed.matching.words.ratio_fail

    body_sentences, full_sentences, body_words, full_words = {}, {}, {}, {}
    for condition in CORE_CONDITIONS:
        body = bodies[condition]
        full = f"{opening} {body}"
        body_sentences[condition] = segmenter.segment(body).count
        full_sentences[condition] = segmenter.segment(full).count
        body_words[condition] = len(word_re.findall(body))
        full_words[condition] = len(word_re.findall(full))

    shortest, longest = min(body_words.values()), max(body_words.values())
    ratio = longest / shortest if shortest else float("inf")
    # Aim at the target, not at the error threshold.
    target_max = int(shortest * ratio_warn)          # keeping the shortest body
    target_min = math.ceil(longest / ratio_warn)     # or lengthening the short ones
    hard_max = int(shortest * ratio_fail)
    too_long = [c for c in CORE_CONDITIONS if body_words[c] > target_max]
    too_short = [c for c in CORE_CONDITIONS if body_words[c] < target_min]

    lines = [
        "  BODY sentences (the rule: exactly "
        f"{required} per body): " + ", ".join(
            f"{c} {body_sentences[c]}" for c in CORE_CONDITIONS),
        "  full-text sentences, opening included, for reference only: " + ", ".join(
            f"{c} {full_sentences[c]}" for c in CORE_CONDITIONS),
    ]
    wrong = [c for c in CORE_CONDITIONS if body_sentences[c] != required]
    if wrong:
        lines.append(f"  -> wrong body sentence count: {', '.join(wrong)}; each must be "
                     f"exactly {required} sentences.")
    lines += [
        "  BODY words: " + ", ".join(f"{c} {body_words[c]}" for c in CORE_CONDITIONS),
        "  full-text words, opening included: " + ", ".join(
            f"{c} {full_words[c]}" for c in CORE_CONDITIONS),
        f"  shortest body {shortest} words, longest {longest}, current ratio {ratio:.2f}.",
        f"  target ratio {ratio_warn} (what to aim for); hard-error ceiling {ratio_fail} "
        f"(where the check fails).",
        f"  to reach the {ratio_warn} target: keep the shortest body at {shortest} words and "
        f"bring every body to {target_max} words or fewer, or keep the longest at {longest} "
        f"and bring every body to at least {target_min} words.",
        f"  for reference, the hard ceiling alone would allow up to {hard_max} words against "
        f"a {shortest}-word shortest body; do not aim there.",
    ]
    if too_long:
        lines.append(f"  -> shorten to meet the {ratio_warn} target: {', '.join(too_long)}")
    if too_short:
        lines.append(f"  -> or lengthen to meet it: {', '.join(too_short)}")
    return "\n".join(lines)


def _repair_findings(findings) -> list[str]:
    """What a repair is asked to fix: the validator's own errors, sorted.

    Warnings never trigger a repair — they are for the human reviewer — and no
    advice is composed here, which would make the repair prompt per-item.
    """
    return sorted({f"{f.code}: {f.message}" for f in findings if f.severity == "error"})


def draft_scenario(topic, variant_id: int, cfg: ExperimentConfig, segmenter: Segmenter,
                   backend, store: CallStore, *, allow_live: bool = False) -> StageResult:
    """One scenario, one call. No redraft: there is no scenario repair template.

    A scenario that fails its machine checks is ``needs_manual_review``, and
    its groups are not drafted at all — counterarguments built on a rejected
    scenario would have to be thrown away with it.
    """
    request = scenario_request(topic, variant_id, cfg)
    status, fields, error, reused, response = _send_or_resume(
        store, request, backend, cfg, allow_live=allow_live)

    if status == "error":
        store.record_transport_failure(request, error or "transport failure")
        raise PipelineAbort(f"{topic.decision_id} v{variant_id}: {error}")
    if status == "refused":
        attempt = Attempt(request.call_id, request.kind, request.attempt, status, REFUSED,
                          request.prompt_sha256, error=error, reused=reused)
        return StageResult("scenario", topic.decision_id, variant_id, None, REFUSED, (attempt,))

    findings: tuple[Finding, ...] = ()
    if status == "ok":
        findings = tuple(validate_scenario_text(
            fields["scenario_text"], cfg, segmenter,
            loc={"decision_id": topic.decision_id,
                 "scenario_id": f"{topic.decision_id}_v{variant_id}"}))
    errors, warnings = _codes(findings, "error"), _codes(findings, "warning")
    outcome = ACCEPTED if status == "ok" and not errors else NEEDS_MANUAL_REVIEW
    store.record(request, status=status, outcome=outcome, fields=fields, error=error,
                 error_codes=errors, warning_codes=warnings, response=response,
                 meta=_stored_meta(store, request) if reused else None)
    attempt = Attempt(request.call_id, request.kind, request.attempt, status, outcome,
                      request.prompt_sha256, errors, warnings, error, reused)
    return StageResult("scenario", topic.decision_id, variant_id, None, outcome, (attempt,),
                       payload=fields if outcome == ACCEPTED else None, findings=findings)


def draft_group(topic, variant_id: int, scenario_text: str, allocation: GroupAllocation,
                cfg: ExperimentConfig, segmenter: Segmenter, backend, store: CallStore,
                *, allow_live: bool = False) -> StageResult:
    """One group: a draft, then validator-driven repairs until the budget ends.

    Stops the moment the group is machine-valid, and after
    ``corpus.repair.max_calls_per_group`` calls at the latest. A group that is
    still failing is ``needs_manual_review`` and stops there.
    """
    budget = cfg.raw["corpus"]["repair"]["max_calls_per_group"]
    option = allocation.supported_option
    loc = {"decision_id": topic.decision_id,
           "scenario_id": f"{topic.decision_id}_v{variant_id}"}
    opening = cfg.raw["corpus"]["counterargument_opening"]

    attempts: list[Attempt] = []
    bodies: dict[str, str] | None = None
    repair_notes: list[str] = []
    findings: tuple[Finding, ...] = ()
    history: list[str] = []
    previous_repair_prompt: str | None = None

    while len(attempts) < budget:
        attempt_no = len(attempts) + 1
        if bodies is None:
            request = group_request(topic, variant_id, scenario_text, allocation, cfg,
                                    attempt=attempt_no)
        else:
            request = repair_request(
                topic, variant_id, scenario_text, allocation, bodies, repair_notes,
                attempt_no, cfg,
                diagnostics=group_diagnostics(bodies, opening, cfg, segmenter),
                history="\n".join(history))
            if request.prompt_sha256 == previous_repair_prompt:
                # A repair that is byte-identical to the one that just failed
                # asks a deterministic server the same question twice. It
                # happened live on 2026-09-16 and is what this guard prevents.
                raise PipelineAbort(
                    f"{topic.decision_id} v{variant_id} {option}: the next repair would "
                    f"repeat the previous failed repair request exactly "
                    f"({request.prompt_sha256[:16]}); refusing to spend a call on it")
            previous_repair_prompt = request.prompt_sha256

        status, fields, error, reused, response = _send_or_resume(
            store, request, backend, cfg, allow_live=allow_live)

        if status == "error":
            store.record_transport_failure(request, error or "transport failure")
            raise PipelineAbort(f"{topic.decision_id} v{variant_id} {option}: {error}")
        if status == "refused":
            attempts.append(Attempt(request.call_id, request.kind, attempt_no, status, REFUSED,
                                    request.prompt_sha256, error=error, reused=reused))
            return StageResult("group", topic.decision_id, variant_id, option, REFUSED,
                               tuple(attempts))

        last_call = len(attempts) + 1 >= budget
        if status == "rejected":
            # Nothing usable came back, so there is nothing to repair: the next
            # call, if any, is a fresh draft.
            outcome = NEEDS_MANUAL_REVIEW if last_call else REPAIR_NEEDED
            bodies, repair_notes, findings = None, [], ()
            errors = warnings = ()
            no_progress = False
        else:
            block = _block_from(fields, allocation)
            findings = tuple(validate_group(scenario_text, opening, block, cfg, segmenter,
                                            loc=loc))
            errors, warnings = _codes(findings, "error"), _codes(findings, "warning")
            # A repair that returns exactly what it was given changed nothing.
            # It is recorded as such and the next one is told so explicitly.
            no_progress = bodies is not None and dict(fields) == bodies
            if no_progress:
                history.append(
                    f"Repair attempt {attempt_no} returned the four bodies UNCHANGED: it made "
                    f"no progress, and the same findings stand. Do not return these bodies "
                    f"again. Make a materially different correction this time, using the "
                    f"measurements below, while keeping every substantive rule.")
            if not errors:
                outcome = ACCEPTED
            else:
                outcome = NEEDS_MANUAL_REVIEW if last_call else REPAIR_NEEDED
                bodies, repair_notes = dict(fields), _repair_findings(findings)

        store.record(request, status=status, outcome=outcome, fields=fields, error=error,
                     error_codes=errors, warning_codes=warnings, response=response,
                     meta=_stored_meta(store, request) if reused else None,
                     no_progress=no_progress)
        attempts.append(Attempt(request.call_id, request.kind, attempt_no, status, outcome,
                                request.prompt_sha256, errors, warnings, error, reused,
                                no_progress=no_progress))

        if outcome == ACCEPTED:
            return StageResult("group", topic.decision_id, variant_id, option, ACCEPTED,
                               tuple(attempts), payload=dict(fields), findings=findings)

    return StageResult("group", topic.decision_id, variant_id, option, NEEDS_MANUAL_REVIEW,
                       tuple(attempts), findings=findings)


def run_pilot(topics, allocation_groups, cfg: ExperimentConfig, segmenter: Segmenter,
              backend, store: CallStore, *, allow_live: bool = False,
              variants: tuple[int, ...] = (1, 2)) -> list[StageResult]:
    """Scenarios first, then the groups of every scenario that was accepted.

    Strictly sequential and deterministic: decisions in sorted order, variants
    in order, ``opt_1`` before ``opt_2``. A scenario that is not accepted takes
    its own groups out of the run — and nothing here approves anything: a
    machine-valid group still needs the human judgements the validator lists.
    """
    by_group = {(g.decision_id, g.variant_id, g.supported_option): g for g in allocation_groups}
    results: list[StageResult] = []

    for topic in sorted(topics, key=lambda t: t.decision_id):
        for variant_id in variants:
            scenario = draft_scenario(topic, variant_id, cfg, segmenter, backend, store,
                                      allow_live=allow_live)
            results.append(scenario)
            if not scenario.accepted:
                for option in ("opt_1", "opt_2"):
                    results.append(StageResult(
                        "group", topic.decision_id, variant_id, option,
                        SKIPPED_SCENARIO_NOT_ACCEPTED))
                continue
            for option in ("opt_1", "opt_2"):
                group = by_group.get((topic.decision_id, variant_id, option))
                if group is None:
                    raise PipelineAbort(
                        f"no marker allocation for {topic.decision_id} v{variant_id} {option}")
                results.append(draft_group(
                    topic, variant_id, scenario.payload["scenario_text"], group, cfg,
                    segmenter, backend, store, allow_live=allow_live))
    return results
