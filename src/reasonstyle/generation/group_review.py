"""A read-only reading view of a recorded group run. It inspects; it decides nothing.

The group stage records, for every one of the pilot's 48 four-condition groups,
one draft and at most two repairs. Some pass their machine checks; the rest end
``needs_manual_review`` and need a person to read them. This module turns that
recorded evidence into deterministic Markdown so they can be read.

Every count, every command and every claim in the output is derived from the run
being read. Nothing is written for one particular generator's result: the same
code produced the Qwen pilot's 2-of-48 reading view and the hosted pilot's
36-of-48 one, and neither page states the other's numbers.

**What it is not.** It makes no call, sends nothing, repairs nothing, approves
nothing, corrects nothing and writes nothing into the run directory or the
corpus. It proposes no wording: a correction is a separate, separately approved
act (``design_notes.md``, *Assembly and counterargument correction*), and a tool
that drafted one here would be making the curator's decision for them. The
manual-correction ledger is deliberately neither generated nor populated.

**Where the numbers come from.** The findings, the counts and the measurements
are recomputed by :func:`~reasonstyle.corpus.validate.validate_group` and the
validator's own ``_measure``, over the group the assembler would build — the
marker fields from the frozen allocation, never from the response. Nothing here
re-implements a rule, so a review can never show a group as passing a check the
corpus would fail it on. The codes each call recorded at the time are printed
beside the recomputed ones, and a divergence between them is reported rather
than quietly resolved.

**Which text stands.** The scenario shown is the one that currently stands:
model draft, superseded by an accepted redraft where there is one, and then the
approved human correction where there is one. That is the same resolution the
group stage itself used, so the review reads a group against the exact scenario
it was drafted from.

**Which attempt stands.** The final recorded attempt of each group — the last
call that consumed a budget position. Earlier attempts are kept and shown as a
history, because what a repair did, or failed to do, is part of what a reviewer
is reading.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..corpus.findings import Finding
from ..corpus.schemas import CORE_CONDITIONS, SEMANTIC_OPTIONS, Measurements
from ..corpus.segmentation import Segmenter
from ..corpus.validate import HUMAN_REVIEW_CODES, validate_group
# The validator's own per-cell measurement function. Imported rather than
# reproduced: a second implementation of the counting could drift from the rules
# the corpus is actually held to, which is the one thing a review must not do.
from ..corpus.validate import _measure
from ..hashing import canonical_json, sha256_of
from .allocation import GroupAllocation
from .approvals import approval_status
# The same block the controller and the assembler build: marker fields taken
# from the allocation, never from what the model happened to return.
from .pipeline import ACCEPTED, CallStore, _block_from

__all__ = [
    "AttemptRecord",
    "GroupReview",
    "GroupReviewExport",
    "MACHINE_VALID",
    "NEEDS_CORRECTION",
    "NOT_RECORDED",
    "build_group_review",
    "group_attempts",
]

#: A group whose final recorded attempt was accepted by the machine checks. It
#: is *not* an approved group: the human judgements below it are outstanding.
MACHINE_VALID = "machine_valid"
#: A group that ended its budget still failing. It needs inspection and an
#: audited correction; nothing here performs one.
NEEDS_CORRECTION = "needs_correction"
#: An expected group with no recorded attempt at all.
NOT_RECORDED = "not_recorded"

#: Statuses that consumed one call of a group's budget, mirroring the
#: controller. A transport failure completed nothing and is not an attempt.
_CONSUMED_BUDGET = ("ok", "rejected")

_CONDITION_GLOSS = {
    "RS": "reason present · explicit style",
    "RP": "reason present · plain",
    "NS": "no reason · explicit style",
    "NP": "no reason · plain",
}

#: What the page says about itself. The command names the configuration that was
#: actually loaded and the run that was actually read, so a reader can reproduce
#: this export rather than the one some other generator would produce. A config
#: built in memory has no path; the note then stays generator-neutral instead of
#: naming a file that does not exist.
def _read_only_note(cfg: ExperimentConfig, run: Path) -> str:
    config = cfg.path.as_posix() if getattr(cfg, "path", None) else None
    command = (f"`uv run python scripts/pilot.py group-review --config {config} "
               f"--out {run.as_posix()}`" if config else
               "`scripts/pilot.py group-review`, with the configuration and run directory "
               "named above")
    return (
        "> **Generated file — read only.** Regenerate with\n"
        f"> {command}.\n"
        "> It reads the recorded run and writes only here: no call, no repair, no approval,\n"
        "> no correction, no corpus write. Corrections belong in their own approved ledger,\n"
        "> and nothing written into this Markdown is ever read back."
    )


# --------------------------------------------------------------------------
# Reading the recorded run
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One recorded call of one group: what was asked, and what came back."""

    call_id: str
    kind: str                       # group | repair
    attempt: int
    status: str                     # ok | rejected
    outcome: str
    prompt_sha256: str
    recorded_error_codes: tuple[str, ...] = ()
    recorded_warning_codes: tuple[str, ...] = ()
    #: What the controller wrote down: a repair that returned the four bodies it
    #: was handed, unchanged.
    recorded_no_progress: bool = False
    bodies: dict[str, str] | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None

    @property
    def bodies_sha256(self) -> str | None:
        return sha256_of(canonical_json(self.bodies)) if self.bodies else None


def group_attempts(store: CallStore) -> dict[tuple[str, str], tuple[AttemptRecord, ...]]:
    """Every recorded attempt of every group, in order, by (scenario_id, option).

    Ordered by attempt number, so the last element is the final recorded
    attempt — the one whose text and findings stand for the group. A transport
    failure consumed no budget position and completed nothing, so it is not an
    attempt here; the retry that followed it is.
    """
    out: dict[tuple[str, str], list[AttemptRecord]] = {}
    for entry in store.log.entries():
        if entry["kind"] not in ("group", "repair"):
            continue
        if entry["status"] not in _CONSUMED_BUDGET:
            continue
        key = (f"{entry['decision_id']}_v{entry['variant_id']}", entry["supported_option"])
        path = store.result_path(entry["call_id"])
        fields = None
        if path.is_file():
            fields = json.loads(path.read_text(encoding="utf-8")).get("fields") or None
        validation = entry.get("validation") or {}
        out.setdefault(key, []).append(AttemptRecord(
            call_id=entry["call_id"], kind=entry["kind"], attempt=int(entry["attempt"]),
            status=entry["status"], outcome=entry.get("outcome") or "?",
            prompt_sha256=entry["prompt_sha256"],
            recorded_error_codes=tuple(validation.get("error_codes") or ()),
            recorded_warning_codes=tuple(validation.get("warning_codes") or ()),
            recorded_no_progress=bool(validation.get("no_progress")),
            bodies=dict(fields) if fields else None,
            stop_reason=entry.get("stop_reason"), usage=entry.get("usage"),
            error=entry.get("error")))
    return {key: tuple(sorted(rows, key=lambda a: a.attempt)) for key, rows in out.items()}


@dataclass(frozen=True, slots=True)
class GroupReview:
    """One expected group, as the recorded run leaves it."""

    scenario_id: str
    decision_id: str
    variant_id: int
    domain: str
    supported_option: str
    marker_family: str
    marker_string: str
    marker_realization_id: str
    realization_description: str
    options: dict[str, str]
    scenario_text: str
    #: The corpus-wide opening the renderer prepends to every counterargument.
    opening: str
    scenario_call_id: str | None
    scenario_approval_state: str
    scenario_supersedes_call_id: str | None
    scenario_correction: dict[str, Any] | None
    attempts: tuple[AttemptRecord, ...]
    #: Recomputed by the validator on the final attempt's bodies.
    findings: tuple[Finding, ...]
    measurements: dict[str, Measurements]
    #: Recomputed findings per attempt, by call id — the history's own diagnoses.
    attempt_findings: dict[str, tuple[Finding, ...]]
    state: str

    @property
    def final(self) -> AttemptRecord | None:
        return self.attempts[-1] if self.attempts else None

    def rendered(self, condition: str) -> str:
        """The complete counterargument a reader sees: opening, then body.

        Composed exactly as :class:`~reasonstyle.corpus.schemas.ScenarioRecord`
        and the validator compose it, so what is shown here is what is measured.
        """
        final = self.final
        body = (final.bodies or {}).get(condition) if final else None
        return f"{self.opening} {body}" if body else ""

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "warning")

    @property
    def human_review(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "human_review")

    def unchanged_from_previous(self, attempt: AttemptRecord) -> bool:
        """Whether this attempt returned exactly the text of the one before it."""
        index = self.attempts.index(attempt)
        if index == 0 or attempt.bodies is None:
            return False
        return attempt.bodies == self.attempts[index - 1].bodies

    @property
    def identical_throughout(self) -> bool:
        """Every recorded attempt returned the same four bodies."""
        hashes = {a.bodies_sha256 for a in self.attempts}
        return len(self.attempts) > 1 and len(hashes) == 1 and None not in hashes

    def distinct_error_codes(self, attempt: AttemptRecord) -> tuple[str, ...]:
        """The distinct machine-error codes recomputed for one attempt."""
        return tuple(sorted({f.code for f in self.attempt_findings.get(attempt.call_id, ())
                             if f.severity == "error"}))

    @property
    def earlier_attempt_with_fewer_errors(self) -> AttemptRecord | None:
        """An earlier attempt with strictly fewer distinct error codes, if any.

        **Informational only.** The final recorded attempt remains canonical
        here and everywhere else; this selects nothing, approves nothing, and
        changes no text. Nor does it rank the two drafts: a smaller set of
        machine-error codes is a statement about form — counts, ratios, pattern
        matches — and says nothing about whether the earlier text supports its
        option, preserves its pair's propositions or keeps its no-reason cells
        reason-free. Those are the human judgements, and they are outstanding
        for every attempt of every group.

        Among the qualifying earlier attempts the one with the fewest codes is
        named, the earliest of them on a tie, so the report is deterministic.
        An attempt that returned nothing usable was never validated and cannot
        qualify by having no findings.
        """
        final = self.final
        if final is None or final.call_id not in self.attempt_findings:
            return None
        target = len(self.distinct_error_codes(final))
        candidates = [a for a in self.attempts[:-1]
                      if a.call_id in self.attempt_findings
                      and len(self.distinct_error_codes(a)) < target]
        if not candidates:
            return None
        return min(candidates, key=lambda a: (len(self.distinct_error_codes(a)), a.attempt))

    @property
    def recorded_codes_agree(self) -> bool:
        """Whether the final attempt's stored codes match the recomputation."""
        final = self.final
        if final is None or final.bodies is None:
            return True
        return (sorted({f.code for f in self.errors}) == sorted(set(final.recorded_error_codes))
                and sorted({f.code for f in self.warnings})
                == sorted(set(final.recorded_warning_codes)))


def _review_one(topic, variant_id: int, allocation: GroupAllocation, scenario: dict[str, Any],
                attempts: tuple[AttemptRecord, ...], cfg: ExperimentConfig,
                segmenter: Segmenter, *, approval_state: str) -> GroupReview:
    scenario_id = f"{topic.decision_id}_v{variant_id}"
    text = scenario.get("scenario_text") or ""
    opening = cfg.raw["corpus"]["counterargument_opening"]
    realizations = cfg.marker_realizations()

    findings: tuple[Finding, ...] = ()
    measurements: dict[str, Measurements] = {}
    attempt_findings: dict[str, tuple[Finding, ...]] = {}
    loc = {"decision_id": topic.decision_id, "scenario_id": scenario_id}
    for attempt in attempts:
        if attempt.bodies is None or not text:
            continue
        block = _block_from(attempt.bodies, allocation)
        attempt_findings[attempt.call_id] = tuple(
            validate_group(text, opening, block, cfg, segmenter, loc=loc))

    final = attempts[-1] if attempts else None
    if final is not None and final.bodies is not None and text:
        findings = attempt_findings[final.call_id]
        block = _block_from(final.bodies, allocation)
        word_re = re.compile(cfg.parsed.matching.words.word_regex)
        measurements = {condition: _measure(opening, block, condition, segmenter, word_re)[0]
                        for condition in CORE_CONDITIONS}

    if final is None:
        state = NOT_RECORDED
    elif final.outcome == ACCEPTED and not any(f.severity == "error" for f in findings):
        state = MACHINE_VALID
    else:
        state = NEEDS_CORRECTION

    return GroupReview(
        scenario_id=scenario_id, decision_id=topic.decision_id, variant_id=variant_id,
        domain=topic.domain, supported_option=allocation.supported_option,
        marker_family=allocation.marker_family, marker_string=allocation.marker_string,
        marker_realization_id=allocation.marker_realization_id,
        realization_description=(realizations.get(allocation.marker_realization_id) or {}
                                 ).get("description", ""),
        options=dict(topic.options), scenario_text=text, opening=opening,
        scenario_call_id=scenario.get("call_id"), scenario_approval_state=approval_state,
        scenario_supersedes_call_id=scenario.get("supersedes_call_id"),
        scenario_correction=scenario.get("scenario_correction"),
        attempts=attempts, findings=findings, measurements=measurements,
        attempt_findings=attempt_findings, state=state)


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def _cell(text: str) -> str:
    """One table cell. Pipes and newlines would otherwise break the row."""
    return text.replace("|", r"\|").replace("\n", " ") if text else "—"


def _detail_lines(finding: Finding) -> list[str]:
    """The measurements behind a finding, exactly as the validator recorded them."""
    if not finding.detail:
        return []
    return [f"  - `{key}`: {canonical_json(finding.detail[key])}"
            for key in sorted(finding.detail)]


def _findings_block(findings: tuple[Finding, ...], severity: str, none_message: str) -> list[str]:
    rows = [f for f in findings if f.severity == severity]
    if not rows:
        return [f"*{none_message}*", ""]
    lines = []
    for finding in sorted(rows, key=lambda f: (f.code, f.condition or "", f.scope)):
        lines.append(f"- **{finding.severity}** `{finding.code}` "
                     f"({finding.condition or finding.scope}) — {finding.message}")
        lines += _detail_lines(finding)
    return lines + [""]


def _human_review_block(findings: tuple[Finding, ...]) -> list[str]:
    grouped: dict[str, set[str]] = {}
    for finding in findings:
        if finding.severity != "human_review":
            continue
        where = finding.condition or ("/".join(finding.detail["pair"])
                                      if "pair" in finding.detail else finding.scope)
        grouped.setdefault(finding.code, set()).add(where)
    if not grouped:
        return ["*No group was validated, so no human judgement is listed for it.*", ""]
    return [f"- `{code}` ({', '.join(sorted(where))}) — {HUMAN_REVIEW_CODES[code]}"
            for code, where in sorted(grouped.items())] + [""]


def _provenance_block(cfg: ExperimentConfig, run: Path, bank_hash: str,
                      allocation_hash: str | None, segmenter: Segmenter) -> list[str]:
    return [
        f"> **Config** {cfg.config_version} `{cfg.content_hash[:12]}` · **Topic bank** "
        f"`{bank_hash[:12]}` · **Allocation** `{(allocation_hash or '')[:12]}` · "
        f"**Segmenter** {segmenter.info.library} {segmenter.info.version}",
        f"> Recorded run `{run.as_posix()}` — read, never written.",
        _read_only_note(cfg, run),
        "",
    ]


def _state_heading(review: GroupReview) -> str:
    return {
        MACHINE_VALID: "machine-valid, no correction needed",
        NEEDS_CORRECTION: f"NEEDS CORRECTION — {len(review.errors)} machine error(s)",
        NOT_RECORDED: "NO ATTEMPT RECORDED",
    }[review.state]


def _conditions_table(review: GroupReview) -> list[str]:
    final = review.final
    bodies = (final.bodies if final and final.bodies else {})
    rows = [
        "| | RS | RP | NS | NP |",
        "|---|---|---|---|---|",
        "| reason | present | present | absent | absent |",
        "| explicit style | yes | no | yes | no |",
        "| marker carried | yes | no | yes | no |",
    ]
    if review.measurements:
        m = review.measurements
        rows += [
            "| body words | " + " | ".join(str(m[c].word_count_body)
                                           for c in CORE_CONDITIONS) + " |",
            "| full-text words | " + " | ".join(str(m[c].word_count_full)
                                                for c in CORE_CONDITIONS) + " |",
            "| body sentences | " + " | ".join(str(m[c].sentence_count_body)
                                               for c in CORE_CONDITIONS) + " |",
            "| full-text sentences | " + " | ".join(str(m[c].sentence_count_full)
                                                    for c in CORE_CONDITIONS) + " |",
        ]
    rows.append("| body | " + " | ".join(_cell(bodies.get(c, "")) for c in CORE_CONDITIONS)
                + " |")
    return rows + [""]


def _attempt_history(review: GroupReview) -> list[str]:
    lines = ["#### Attempt history", "",
             "| # | kind | call | prompt | returned | recorded errors |",
             "|---|---|---|---|---|---|"]
    for attempt in review.attempts:
        unchanged = review.unchanged_from_previous(attempt)
        returned = ("**UNCHANGED from the previous attempt**" if unchanged
                    else "new text" if attempt.attempt > 1 else "first draft")
        lines.append(f"| {attempt.attempt} | `{attempt.kind}` | `{attempt.call_id[:12]}` "
                     f"| `{attempt.prompt_sha256[:12]}` | {returned} "
                     f"| {', '.join(attempt.recorded_error_codes) or 'none'} |")
    lines.append("")
    if review.identical_throughout:
        repairs = sum(1 for a in review.attempts if a.kind == "repair")
        lines += [f"**Every recorded attempt returned the same four bodies.** "
                  f"{repairs} diagnosed repair(s) changed nothing, so there is no later "
                  f"draft to prefer: what a reviewer is reading is the model's only answer "
                  f"to this group.", ""]
    for attempt in review.attempts:
        unchanged = review.unchanged_from_previous(attempt)
        header = (f"**Attempt {attempt.attempt} — {attempt.kind}** · call "
                  f"`{attempt.call_id}` · prompt `{attempt.prompt_sha256}` · outcome "
                  f"`{attempt.outcome}`")
        lines += [header, ""]
        if unchanged:
            lines += ["> **This repair returned the previous four bodies verbatim.** "
                      f"Bodies sha256 `{(attempt.bodies_sha256 or '')[:16]}`, unchanged. "
                      "It made no progress, and the controller recorded that as "
                      f"`no_progress: {str(attempt.recorded_no_progress).lower()}`.", ""]
        elif attempt.recorded_no_progress:
            lines += ["> The controller recorded `no_progress: true` for this attempt.", ""]
        if attempt.bodies:
            lines += ["| condition | body |", "|---|---|"]
            lines += [f"| **{c}** | {_cell(attempt.bodies.get(c, ''))} |"
                      for c in CORE_CONDITIONS]
            lines.append("")
        else:
            lines += ["*No usable bodies came back from this call "
                      f"({attempt.status}: {attempt.error or 'no detail recorded'}).*", ""]
        findings = review.attempt_findings.get(attempt.call_id, ())
        lines += ["*Findings at this attempt:*", ""]
        lines += _findings_block(findings, "error", "No machine errors at this attempt.")
        lines += _findings_block(findings, "warning", "No warnings at this attempt.")
    return lines


def group_markdown(review: GroupReview) -> list[str]:
    """One group's section, used on its scenario page and in the combined file."""
    final = review.final
    lines = [f"### `{review.supported_option}` — {_state_heading(review)}", ""]
    other = next(o for o in SEMANTIC_OPTIONS if o != review.supported_option)
    lines += [
        f"- **supported option** `{review.supported_option}` — {review.options[review.supported_option]}",
        f"- the other option `{other}` — {review.options[other]}",
        f"- **marker family** `{review.marker_family}` · **string** “{review.marker_string}” "
        f"· **realization** `{review.marker_realization_id}`"
        + (f" ({review.realization_description})" if review.realization_description else ""),
    ]
    if final is None:
        lines += ["- **no call was recorded for this group**", ""]
        return lines
    budget = len(review.attempts)
    lines += [
        f"- **final call** `{final.call_id}`",
        f"- **attempt {final.attempt} of {budget} recorded** · kind `{final.kind}` · "
        f"outcome `{final.outcome}` · stop reason `{final.stop_reason}`",
        f"- prompt sha256 `{final.prompt_sha256}`",
        f"- four bodies sha256 `{final.bodies_sha256}`",
        "",
    ]
    if not review.recorded_codes_agree:
        lines += ["> **The recomputed findings differ from the codes this call recorded.** "
                  f"Recorded errors {sorted(set(final.recorded_error_codes))}, recomputed "
                  f"{sorted({f.code for f in review.errors})}. Reported, not resolved.", ""]

    lines += ["#### The four conditions, side by side", ""]
    lines += _conditions_table(review)

    lines += ["#### Verbatim bodies", "",
              "*What the model returned, exactly as recorded. The shared opening is not "
              "part of a body.*", ""]
    for condition in CORE_CONDITIONS:
        lines += [f"**{condition}** — {_CONDITION_GLOSS[condition]}", "",
                  "```text", (final.bodies or {}).get(condition, "(nothing recorded)"),
                  "```", ""]

    lines += ["#### The complete counterargument, opening included", "",
              "*What a reader of the transcript would see: the shared opening the renderer "
              "prepends, then the body.*", ""]
    for condition in CORE_CONDITIONS:
        body = (final.bodies or {}).get(condition)
        lines += [f"**{condition}**", "", "```text",
                  review.rendered(condition) if body else "(nothing recorded)", "```", ""]

    lines += ["#### Machine findings on the final attempt", ""]
    lines += _findings_block(review.findings, "error", "No machine errors.")
    lines += ["*Warnings — for the reviewer, never blocking:*", ""]
    lines += _findings_block(review.findings, "warning", "No warnings.")

    lines += ["#### Outstanding human judgements", "",
              "*Emitted unconditionally. A machine-valid group has every one of these still "
              "outstanding; none of them is answered anywhere in this file.*", ""]
    lines += _human_review_block(review.findings)

    if review.state != MACHINE_VALID or len(review.attempts) > 1:
        lines += _attempt_history(review)
    return lines


def scenario_markdown(scenario_id: str, reviews: list[GroupReview], cfg: ExperimentConfig,
                      run: Path, bank_hash: str, allocation_hash: str | None,
                      segmenter: Segmenter) -> str:
    first = reviews[0]
    lines = [f"# `{scenario_id}` — {first.domain}, variant {first.variant_id}", ""]
    lines += _provenance_block(cfg, run, bank_hash, allocation_hash, segmenter)
    valid = [r for r in reviews if r.state == MACHINE_VALID]
    lines += [f"**Groups on this page** {len(reviews)} · **machine-valid** {len(valid)} · "
              f"**needing correction** {len(reviews) - len(valid)}", ""]

    lines += ["## The scenario these counterarguments were drafted from", ""]
    state = first.scenario_approval_state
    lines += [f"- gate state **{state}** · source call "
              f"`{first.scenario_call_id or '(none)'}`",
              f"- text sha256 `{sha256_of(first.scenario_text) if first.scenario_text else '(none)'}`"]
    if first.scenario_supersedes_call_id:
        lines.append(f"- this text superseded call `{first.scenario_supersedes_call_id}`, "
                     f"which the curator rejected")
    if first.scenario_correction:
        correction = first.scenario_correction
        lines += [f"- **human-corrected** by {correction['editor']} on "
                  f"{correction['decided_at']}, bound to call "
                  f"`{correction['original_call_id']}`",
                  f"- correction reason: {correction['reason']}",
                  f"- original text sha256 `{correction['original_text_sha256']}` → corrected "
                  f"`{correction['corrected_text_sha256']}`"]
    else:
        lines.append("- no human correction: this is the model's own text")
    lines += ["", "```text", first.scenario_text or "(no scenario recorded)", "```", "",
              "## Shared counterargument opening", "", f"> {first.opening}", "",
              "Prepended by the renderer to all eight counterarguments of this scenario, so "
              "the four cells of a group cannot differ in their opening. It is never part of "
              "a body.", ""]

    for review in reviews:
        lines += ["---", ""]
        lines += group_markdown(review)
    return "\n".join(lines).rstrip("\n") + "\n"


def index_markdown(reviews: list[GroupReview], cfg: ExperimentConfig, run: Path,
                   bank_hash: str, allocation_hash: str | None, segmenter: Segmenter,
                   expected: int) -> str:
    valid = [r for r in reviews if r.state == MACHINE_VALID]
    failing = [r for r in reviews if r.state == NEEDS_CORRECTION]
    missing = [r for r in reviews if r.state == NOT_RECORDED]

    lines = ["# Pilot group run — reading view", ""]
    lines += _provenance_block(cfg, run, bank_hash, allocation_hash, segmenter)
    lines += [
        f"**{expected} expected group(s)** · **{len(valid)} machine-valid** · "
        f"**{len(failing)} requiring correction** · **{len(missing)} with no recorded call**",
        "",
        "Machine-valid means the validator found no errors on the final recorded attempt. "
        "It is not approval: every human judgement listed under each group is outstanding, "
        "for every machine-valid group exactly as much as for the rest.",
        "",
        "This pass is inspection only. No wording is proposed here, no correction ledger is "
        "written, and no generator, model or validator decision follows from it.",
        "",
    ]

    lines += ["## Machine-valid groups", ""]
    if valid:
        lines += ["| Scenario | Option | Domain | Marker family | Attempts | Final call | Page |",
                  "|---|---|---|---|---|---|---|"]
        lines += [_index_row(r) for r in valid]
    else:
        lines.append("*None.*")
    lines.append("")

    lines += ["## Groups requiring inspection and an audited correction", ""]
    if failing:
        spent = sorted({len(r.attempts) for r in failing})
        budget = (f"{spent[0]} recorded attempt(s) each" if len(spent) == 1
                  else f"between {spent[0]} and {spent[-1]} recorded attempts each")
        lines += [f"Each of these ended without passing its machine checks, with {budget}. "
                  f"The attempt history on each page shows what every attempt returned.", ""]
    if failing:
        lines += ["| Scenario | Option | Domain | Marker family | Attempts | Final errors | Page |",
                  "|---|---|---|---|---|---|---|"]
        lines += [_index_row(r, errors=True) for r in failing]
    else:
        lines.append("*None.*")
    lines.append("")

    if missing:
        lines += ["## Expected groups with no recorded call", "",
                  "| Scenario | Option | Domain |", "|---|---|---|"]
        lines += [f"| `{r.scenario_id}` | `{r.supported_option}` | {r.domain} |"
                  for r in missing]
        lines.append("")

    counts: dict[str, int] = {}
    for review in failing:
        for code in sorted({f.code for f in review.errors}):
            counts[code] = counts.get(code, 0) + 1
    if counts:
        lines += ["## Final errors, counted over the groups requiring correction", "",
                  "| Code | Groups |", "|---|---|"]
        lines += [f"| `{code}` | {n} |"
                  for code, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        lines.append("")

    repairs = [r for r in failing if len(r.attempts) > 1]
    unchanged_by_attempt: dict[int, int] = {}
    for review in repairs:
        for attempt in review.attempts[1:]:
            if review.unchanged_from_previous(attempt):
                unchanged_by_attempt[attempt.attempt] = (
                    unchanged_by_attempt.get(attempt.attempt, 0) + 1)
    if repairs:
        identical = sum(1 for r in repairs if r.identical_throughout)
        lines += ["## What the repairs did", "",
                  f"Counted over the {len(repairs)} group(s) that were repaired and still "
                  f"require correction.", "",
                  "| Measure | Groups |", "|---|---|"]
        if unchanged_by_attempt:
            lines += [f"| repair returning unchanged text at attempt {attempt} "
                      f"| {count} of {len(repairs)} |"
                      for attempt, count in sorted(unchanged_by_attempt.items())]
        else:
            lines.append(f"| repair returning unchanged text | 0 of {len(repairs)} |")
        lines.append(f"| identical bodies at every recorded attempt "
                     f"| {identical} of {len(repairs)} |")
        lines.append("")
        if not unchanged_by_attempt and not identical:
            lines += ["**Every repair returned different text.** The model acted on the "
                      "measurements it was given at every attempt; these groups are ones "
                      "where doing so did not reach the rules, not ones where the repair "
                      "was ignored.", ""]

    flagged = [r for r in reviews if r.earlier_attempt_with_fewer_errors is not None]
    if flagged:
        lines += ["## Informational — an earlier attempt recorded fewer error codes", "",
                  "For these groups, some attempt before the last one came back with strictly "
                  "fewer distinct machine-error codes than the final recorded attempt did.",
                  "",
                  "**The final recorded attempt remains canonical.** It is the text this "
                  "review shows, measures and reports for the group, here and on every page.",
                  "",
                  "**This flag selects nothing and approves nothing.** It does not promote the "
                  "earlier attempt, does not change which text stands, and is not a "
                  "recommendation to use it. Choosing text is a curator's act, recorded as an "
                  "approved correction bound to the exact call it replaces.",
                  "",
                  "**Fewer machine-error codes does not mean better.** The codes count "
                  "*form* — sentence counts, word ratios, marker presence, pattern matches. "
                  "They say nothing about substantive support, support direction, no-reason "
                  "integrity, proposition preservation or naturalness, which are human "
                  "judgements and are outstanding for every attempt of every group. An "
                  "earlier draft with one code may be worse in every way that matters.",
                  "",
                  "| Scenario | Option | Earlier attempt | Earlier call | Earlier error codes "
                  "| Final attempt | Final call | Final error codes |",
                  "|---|---|---|---|---|---|---|---|"]
        for review in flagged:
            earlier, final = review.earlier_attempt_with_fewer_errors, review.final
            lines.append(
                f"| `{review.scenario_id}` | `{review.supported_option}` "
                f"| {earlier.attempt} | `{earlier.call_id}` "
                f"| {', '.join(review.distinct_error_codes(earlier)) or 'none'} "
                f"| {final.attempt} | `{final.call_id}` "
                f"| {', '.join(review.distinct_error_codes(final)) or 'none'} |")
        lines.append("")

    lines += ["## Every expected group", "",
              "| Scenario | Option | Domain | Variant | State | Page |",
              "|---|---|---|---|---|---|"]
    for review in reviews:
        lines.append(f"| `{review.scenario_id}` | `{review.supported_option}` "
                     f"| {review.domain} | v{review.variant_id} | {review.state} "
                     f"| [scenarios/{review.scenario_id}.md]"
                     f"(scenarios/{review.scenario_id}.md) |")
    lines += ["", "Everything above is also in one file: "
              "[all_groups.md](all_groups.md).", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def _index_row(review: GroupReview, *, errors: bool = False) -> str:
    final = review.final
    last = (", ".join(sorted({f.code for f in review.errors})) or "none") if errors else (
        f"`{final.call_id[:12]}`" if final else "—")
    return (f"| `{review.scenario_id}` | `{review.supported_option}` | {review.domain} "
            f"| `{review.marker_family}` | {len(review.attempts)} | {last} "
            f"| [scenarios/{review.scenario_id}.md](scenarios/{review.scenario_id}.md) |")


def combined_markdown(pages: dict[str, str], cfg: ExperimentConfig, run: Path,
                      bank_hash: str, allocation_hash: str | None,
                      segmenter: Segmenter) -> str:
    lines = ["# All pilot groups — combined searchable view", ""]
    lines += _provenance_block(cfg, run, bank_hash, allocation_hash, segmenter)
    lines += ["## Contents", ""]
    ids = sorted(pages)
    lines += [f"- [`{scenario_id}`](#{scenario_id.replace('_', '-')})" for scenario_id in ids]
    lines.append("")
    for scenario_id in ids:
        body = pages[scenario_id]
        body = body.split("\n", 1)[1] if body.startswith("# ") else body
        lines += [f'<a id="{scenario_id.replace("_", "-")}"></a>', "",
                  f"# `{scenario_id}`", "", body]
    return "\n".join(lines).rstrip("\n") + "\n"


@dataclass(frozen=True)
class GroupReviewExport:
    """Every generated file, as text, plus the manifest and the reviews."""

    files: dict[str, str]
    manifest: dict[str, Any]
    reviews: tuple[GroupReview, ...]

    @property
    def machine_valid(self) -> tuple[GroupReview, ...]:
        return tuple(r for r in self.reviews if r.state == MACHINE_VALID)

    @property
    def needing_correction(self) -> tuple[GroupReview, ...]:
        return tuple(r for r in self.reviews if r.state == NEEDS_CORRECTION)

    def write(self, root: str | Path) -> Path:
        root = Path(root)
        for name, text in sorted(self.files.items()):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        (root / "MANIFEST.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return root


def build_group_review(*, topics, allocation_groups, scenarios: dict[str, dict[str, Any]],
                       store: CallStore, cfg: ExperimentConfig, segmenter: Segmenter,
                       approvals: dict, topic_bank_content_hash: str,
                       allocation_content_hash: str | None = None,
                       variants: tuple[int, ...] = (1, 2)) -> GroupReviewExport:
    """Build every review file in memory. Deterministic for a given recorded run.

    Reads the log, the result files, the approvals and the correction ledger.
    Writes nothing anywhere: :meth:`GroupReviewExport.write` is what puts files
    on disk, and only under the review directory it is given.

    Every *expected* group appears, taken from the frozen allocation rather than
    from what the log happens to contain — a report built from the recorded
    calls alone would show a run that generated nothing as complete.
    """
    attempts = group_attempts(store)
    by_decision = {topic.decision_id: topic for topic in topics}
    expected = [g for g in allocation_groups
                if g.decision_id in by_decision and g.variant_id in variants]

    reviews: list[GroupReview] = []
    for allocation in sorted(expected, key=lambda g: (g.decision_id, g.variant_id,
                                                      g.supported_option)):
        topic = by_decision[allocation.decision_id]
        scenario_id = f"{allocation.decision_id}_v{allocation.variant_id}"
        scenario = scenarios.get(scenario_id) or {}
        state = "not_generated"
        if scenario.get("scenario_text"):
            state = approval_status(
                scenario_id, scenario["scenario_text"], scenario["call_id"], approvals,
                config_content_hash=cfg.content_hash,
                topic_bank_content_hash=topic_bank_content_hash,
                machine_errors=len(scenario.get("error_codes") or []))[0]
        reviews.append(_review_one(
            topic, allocation.variant_id, allocation, scenario,
            attempts.get((scenario_id, allocation.supported_option), ()), cfg, segmenter,
            approval_state=state))

    run = Path(store.directory)
    by_scenario: dict[str, list[GroupReview]] = {}
    for review in reviews:
        by_scenario.setdefault(review.scenario_id, []).append(review)

    files: dict[str, str] = {}
    pages: dict[str, str] = {}
    for scenario_id in sorted(by_scenario):
        text = scenario_markdown(scenario_id, by_scenario[scenario_id], cfg, run,
                                 topic_bank_content_hash, allocation_content_hash, segmenter)
        pages[scenario_id] = text
        files[f"scenarios/{scenario_id}.md"] = text
    files["index.md"] = index_markdown(reviews, cfg, run, topic_bank_content_hash,
                                       allocation_content_hash, segmenter, len(expected))
    files["all_groups.md"] = combined_markdown(pages, cfg, run, topic_bank_content_hash,
                                               allocation_content_hash, segmenter)

    manifest = {
        "kind": "pilot_group_review",
        "read_only": True,
        "config_version": cfg.config_version,
        "config_content_hash": cfg.content_hash,
        "topic_bank_content_hash": topic_bank_content_hash,
        "allocation_content_hash": allocation_content_hash,
        "run_directory": run.as_posix(),
        "segmenter": {"library": segmenter.info.library, "version": segmenter.info.version},
        "expected_groups": len(expected),
        "machine_valid": sum(1 for r in reviews if r.state == MACHINE_VALID),
        "needing_correction": sum(1 for r in reviews if r.state == NEEDS_CORRECTION),
        "not_recorded": sum(1 for r in reviews if r.state == NOT_RECORDED),
        "calls_recorded": sum(len(r.attempts) for r in reviews),
        "groups": [
            {"scenario_id": r.scenario_id, "supported_option": r.supported_option,
             "domain": r.domain, "variant_id": r.variant_id, "state": r.state,
             "marker_family": r.marker_family, "marker_string": r.marker_string,
             "marker_realization_id": r.marker_realization_id,
             "scenario_call_id": r.scenario_call_id,
             "scenario_approval_state": r.scenario_approval_state,
             "scenario_human_corrected": bool(r.scenario_correction),
             "attempts": [
                 {"attempt": a.attempt, "kind": a.kind, "call_id": a.call_id,
                  "prompt_sha256": a.prompt_sha256, "outcome": a.outcome,
                  "bodies_sha256": a.bodies_sha256,
                  "recorded_no_progress": a.recorded_no_progress,
                  "unchanged_from_previous": r.unchanged_from_previous(a),
                  "recorded_error_codes": list(a.recorded_error_codes),
                  "recorded_warning_codes": list(a.recorded_warning_codes)}
                 for a in r.attempts],
             "final_error_codes": sorted({f.code for f in r.errors}),
             "final_warning_codes": sorted({f.code for f in r.warnings}),
             "outstanding_human_review_codes": sorted({f.code for f in r.human_review}),
             "recorded_codes_agree": r.recorded_codes_agree,
             "identical_bodies_throughout": r.identical_throughout,
             # Informational only: the final attempt stays canonical, and fewer
             # machine-error codes is a statement about form, not about quality.
             "earlier_attempt_with_fewer_error_codes": (
                 None if r.earlier_attempt_with_fewer_errors is None else {
                     "attempt": r.earlier_attempt_with_fewer_errors.attempt,
                     "call_id": r.earlier_attempt_with_fewer_errors.call_id,
                     "error_codes": list(r.distinct_error_codes(
                         r.earlier_attempt_with_fewer_errors)),
                     "final_attempt": r.final.attempt,
                     "final_call_id": r.final.call_id,
                     "final_error_codes": list(r.distinct_error_codes(r.final))})}
            for r in reviews],
        "files": {name: sha256_of(text) for name, text in sorted(files.items())},
    }
    return GroupReviewExport(files=files, manifest=manifest, reviews=tuple(reviews))
