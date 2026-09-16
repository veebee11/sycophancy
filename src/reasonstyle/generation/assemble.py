"""Assembly: approved scenarios and machine-valid groups into corpus records.

Three rules govern everything here.

**Machine validity is necessary.** A group with validator errors is never
assembled, however it got them and whoever approves of it. A human correction
does not exempt anything: corrected text is re-validated by the same code, and
if it still fails, it is still refused.

**Human approval is also necessary, and separate.** A scenario is assembled only
if the curator approved that exact text (``approvals.py``). Machine validity
alone assembles nothing, and an assembled record stays ``validation.status:
draft`` until the item, pair and scenario judgements are recorded.

**The generated text is never edited in place.** A correction is a *separate,
auditable record* naming the original call and its original text, the corrected
text, the editor, the reason, the validator result and the approval state. The
raw responses and the generation log keep what the model actually produced, so
what came from the model and what a person changed stay distinguishable
afterwards.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ..config import ExperimentConfig
from ..corpus.schemas import CORE_CONDITIONS, Cell, DirectionBlock, ScenarioRecord
from ..corpus.segmentation import Segmenter
from ..corpus.validate import validate_group
from ..hashing import sha256_of
from .allocation import GroupAllocation
from .approvals import APPROVED, ScenarioApproval, approval_status

__all__ = [
    "AssemblyError",
    "AssemblyRefused",
    "ManualCorrection",
    "assemble_pilot",
    "assemble_scenario",
    "correction_problems",
    "load_corrections",
    "save_corrections",
    "write_pilot_corpus",
]

#: Every field an auditable correction must carry. A correction missing any of
#: them is refused: an unattributed edit to research material is not a record.
CORRECTION_FIELDS = ("scenario_id", "supported_option", "condition", "original_call_id",
                     "original_text", "corrected_text", "editor", "reason", "decided_at",
                     "approval_state")

_APPROVAL_STATES = ("approved", "pending", "rejected")


class AssemblyError(ValueError):
    """The inputs to assembly are malformed."""


class AssemblyRefused(AssemblyError):
    """Assembly stopped because material was not approved, or not machine-valid."""


@dataclass(frozen=True, slots=True)
class ManualCorrection:
    """One recorded human edit to one cell, kept beside the original."""

    scenario_id: str
    supported_option: str
    condition: str
    original_call_id: str
    original_text: str
    corrected_text: str
    editor: str
    reason: str
    decided_at: date
    approval_state: str = "pending"
    #: Filled in when the corrected text is re-validated: the codes it produced.
    validation_error_codes: tuple[str, ...] = ()
    validation_warning_codes: tuple[str, ...] = ()

    @property
    def original_text_sha256(self) -> str:
        return sha256_of(self.original_text)

    @property
    def corrected_text_sha256(self) -> str:
        return sha256_of(self.corrected_text)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.scenario_id, self.supported_option, self.condition)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["decided_at"] = self.decided_at.isoformat()
        out["original_text_sha256"] = self.original_text_sha256
        out["corrected_text_sha256"] = self.corrected_text_sha256
        out["validation_error_codes"] = list(self.validation_error_codes)
        out["validation_warning_codes"] = list(self.validation_warning_codes)
        return out


def _correction_from(raw: dict[str, Any]) -> ManualCorrection:
    missing = [f for f in CORRECTION_FIELDS if raw.get(f) in (None, "")]
    if missing:
        raise AssemblyError(
            f"a manual correction is missing {missing}. A correction is a record: it names "
            f"the original call and text, the corrected text, the editor, the reason, the "
            f"date and its approval state.")
    if raw["approval_state"] not in _APPROVAL_STATES:
        raise AssemblyError(f"approval_state {raw['approval_state']!r} is not one of "
                            f"{list(_APPROVAL_STATES)}")
    if raw["condition"] not in CORE_CONDITIONS:
        raise AssemblyError(f"condition {raw['condition']!r} is not a core condition")
    if raw["corrected_text"] == raw["original_text"]:
        raise AssemblyError(f"{raw['scenario_id']}/{raw['supported_option']}/"
                            f"{raw['condition']}: the correction changes nothing")
    decided_at = raw["decided_at"]
    return ManualCorrection(
        scenario_id=raw["scenario_id"], supported_option=raw["supported_option"],
        condition=raw["condition"], original_call_id=raw["original_call_id"],
        original_text=raw["original_text"], corrected_text=raw["corrected_text"],
        editor=raw["editor"], reason=raw["reason"],
        decided_at=decided_at if isinstance(decided_at, date) else date.fromisoformat(decided_at),
        approval_state=raw["approval_state"],
        validation_error_codes=tuple(raw.get("validation_error_codes") or ()),
        validation_warning_codes=tuple(raw.get("validation_warning_codes") or ()))


def load_corrections(path: str | Path) -> list[ManualCorrection]:
    """Read the manual-correction record. Absent is empty, not an error."""
    path = Path(path)
    if not path.is_file():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    except yaml.YAMLError as exc:
        raise AssemblyError(f"{path} is not readable YAML: {exc}") from exc
    if not isinstance(raw, list):
        raise AssemblyError(f"{path} must be a list of correction records")
    corrections = [_correction_from(item) for item in raw]
    seen: dict[tuple[str, str, str], ManualCorrection] = {}
    for correction in corrections:
        if correction.key in seen:
            raise AssemblyError(f"two corrections for {correction.key}; one cell has one "
                                f"current correction, and its history lives in git")
        seen[correction.key] = correction
    return corrections


def save_corrections(corrections: list[ManualCorrection], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump([c.as_dict() for c in corrections], sort_keys=True,
                                   allow_unicode=True), encoding="utf-8")
    return path


def correction_problems(correction: ManualCorrection, original_text: str,
                        original_call_id: str | None = None) -> list[str]:
    """Why a correction may not be applied, before anything is validated.

    A correction names the call whose output it corrects. If that id is not the
    accepted call for this group, the correction belongs to some other draft —
    an earlier attempt, or another run — and applying it here would silently
    attach a human edit to material it was never written about.
    """
    problems = []
    if correction.original_text != original_text:
        problems.append(
            f"{correction.key}: the recorded original text is not what the model returned; "
            f"the correction is bound to a text that is no longer there")
    if original_call_id is not None and correction.original_call_id != original_call_id:
        problems.append(
            f"{correction.key}: the correction names call "
            f"{correction.original_call_id[:12]}, but this group was accepted from "
            f"{original_call_id[:12]}")
    if correction.approval_state != "approved":
        problems.append(f"{correction.key}: the correction is {correction.approval_state}, "
                        f"not approved")
    return problems


def assemble_scenario(
    *,
    topic,
    variant_id: int,
    scenario_text: str,
    scenario_call_id: str,
    groups: dict[str, dict[str, str]],
    group_call_ids: dict[str, str],
    allocations: dict[str, GroupAllocation],
    approvals: dict[str, ScenarioApproval],
    cfg: ExperimentConfig,
    segmenter: Segmenter,
    corrections: list[ManualCorrection] | None = None,
    topic_bank_content_hash: str,
    generation: dict[str, Any] | None = None,
) -> tuple[ScenarioRecord, list[ManualCorrection]]:
    """One assembled ``ScenarioRecord``, or a refusal.

    Refuses unless the scenario's exact text is approved, every applied
    correction is an approved record bound to the text it replaces, and every
    group — after corrections — is machine-valid. Marker fields come from the
    allocation, never from a response or a correction.

    Returns the record and the corrections that were applied, each carrying the
    validator result of the corrected material.
    """
    scenario_id = f"{topic.decision_id}_v{variant_id}"
    opening = cfg.raw["corpus"]["counterargument_opening"]
    corrections = list(corrections or [])

    state, reasons = approval_status(
        scenario_id, scenario_text, scenario_call_id, approvals,
        config_content_hash=cfg.content_hash,
        topic_bank_content_hash=topic_bank_content_hash)
    if state != APPROVED:
        raise AssemblyRefused(f"{scenario_id}: scenario is {state}"
                              + (f" ({'; '.join(r for r in reasons if r)})" if reasons else ""))

    if set(groups) != {"opt_1", "opt_2"}:
        raise AssemblyRefused(f"{scenario_id}: both supported options must be accepted; "
                              f"got {sorted(groups)}")
    if set(group_call_ids) != {"opt_1", "opt_2"}:
        raise AssemblyRefused(
            f"{scenario_id}: a call id is needed for each supported option, so every cell can "
            f"be traced to the call that produced it; got {sorted(group_call_ids)}")
    for option in ("opt_1", "opt_2"):
        allocation = allocations.get(option)
        if allocation is None:
            raise AssemblyRefused(f"{scenario_id}: no marker allocation for {option}")
        belongs = (allocation.decision_id == topic.decision_id
                   and allocation.variant_id == variant_id
                   and allocation.scenario_id == scenario_id
                   and allocation.supported_option == option)
        if not belongs:
            raise AssemblyRefused(
                f"{scenario_id}/{option}: the allocation belongs to "
                f"{allocation.scenario_id}/{allocation.supported_option}, not to this group; "
                f"marker fields must come from this group's own allocation")

    # A correction naming an option this scenario does not have is a mistake in
    # the record, not something to skip over quietly.
    for correction in corrections:
        if correction.scenario_id != scenario_id:
            continue
        if correction.supported_option not in ("opt_1", "opt_2"):
            raise AssemblyRefused(
                f"{scenario_id}: a correction names supported option "
                f"{correction.supported_option!r}, which is not one of ['opt_1', 'opt_2']")

    applied: list[ManualCorrection] = []
    blocks: dict[str, DirectionBlock] = {}
    for option in ("opt_1", "opt_2"):
        bodies = dict(groups[option])
        if set(bodies) != set(CORE_CONDITIONS):
            raise AssemblyRefused(f"{scenario_id}/{option}: expected the four core cells")
        allocation = allocations[option]

        for correction in corrections:
            if (correction.scenario_id, correction.supported_option) != (scenario_id, option):
                continue
            problems = correction_problems(correction, bodies[correction.condition],
                                           group_call_ids[option])
            if problems:
                raise AssemblyRefused("; ".join(problems))
            bodies[correction.condition] = correction.corrected_text
            applied.append(correction)

        block = DirectionBlock(
            supported_option=option,
            marker_family=allocation.marker_family,
            marker_string=allocation.marker_string,
            marker_realization_id=allocation.marker_realization_id,
            cells={c: Cell(condition=c, body=bodies[c], markers_present=c in ("RS", "NS"),
                           marker_family=allocation.marker_family if c in ("RS", "NS") else None)
                   for c in CORE_CONDITIONS})

        # The same validator, on whatever is about to be assembled. Corrected
        # text earns no exemption: if it still fails, assembly still refuses.
        findings = validate_group(scenario_text, opening, block, cfg, segmenter,
                                  loc={"decision_id": topic.decision_id,
                                       "scenario_id": scenario_id})
        errors = sorted({f.code for f in findings if f.severity == "error"})
        warnings = sorted({f.code for f in findings if f.severity == "warning"})
        if errors:
            corrected_here = [c for c in applied if c.supported_option == option]
            note = (" (after applying its recorded correction(s); a human approval never "
                    "overrides a machine error)" if corrected_here else "")
            raise AssemblyRefused(f"{scenario_id}/{option} has machine errors {errors}{note}")
        applied = [
            c if c.supported_option != option else
            type(c)(**{**{f.name: getattr(c, f.name) for f in c.__dataclass_fields__.values()},
                       "validation_error_codes": tuple(errors),
                       "validation_warning_codes": tuple(warnings)})
            for c in applied]
        blocks[option] = block

    record = ScenarioRecord(
        schema_version="1",
        config_version=cfg.config_version,
        config_content_hash=cfg.content_hash,
        decision_id=topic.decision_id,
        scenario_id=scenario_id,
        variant_id=variant_id,
        domain=topic.domain,
        scenario_text=scenario_text,
        options=dict(topic.options),
        counterargument_opening=opening,
        counterarguments=blocks,
        source_type="constructed",
        generation=generation,
    )
    # Deliberately left at "draft": machine-valid and scenario-approved is not
    # an approved item. The item, pair and scenario judgements come later.
    return record, applied


def assembly_manifest(record: ScenarioRecord, *, scenario_call_id: str,
                      group_call_ids: dict[str, str],
                      corrections: list[ManualCorrection]) -> dict[str, Any]:
    """What this record was built from, by call id, plus any human edits."""
    return {
        "scenario_id": record.scenario_id,
        "scenario_call_id": scenario_call_id,
        "group_call_ids": dict(sorted(group_call_ids.items())),
        "scenario_text_sha256": sha256_of(record.scenario_text),
        "validation_status": record.validation.status,
        "manual_corrections": [c.as_dict() for c in corrections],
    }


# --------------------------------------------------------------------------
# The corpus-wide driver
# --------------------------------------------------------------------------


def assemble_pilot(*, topics, allocation_groups, approvals: dict[str, ScenarioApproval],
                   corrections: list[ManualCorrection], scenarios: dict[str, dict[str, Any]],
                   groups: dict[tuple[str, str], dict[str, Any]], cfg: ExperimentConfig,
                   segmenter: Segmenter, topic_bank_content_hash: str,
                   variants: tuple[int, ...] = (1, 2)) -> tuple[list, dict[str, Any]]:
    """Every record of the pilot, or nothing.

    A corpus assembled from whatever happened to be ready would not be the
    pilot; it would be a subset chosen by which calls succeeded, with a marker
    allocation no longer balanced over the decisions it was built for. So a
    single missing scenario, unaccepted group, unresolved machine error or
    unapproved correction refuses the whole build.

    The records come back validated at pilot scope and still marked ``draft``:
    machine-valid and scenario-approved is not item-approved.
    """
    from ..corpus.validate import validate_corpus

    by_group = {(g.decision_id, g.variant_id, g.supported_option): g
                for g in allocation_groups}
    records, manifests = [], []
    missing: list[str] = []

    for topic in sorted(topics, key=lambda t: t.decision_id):
        for variant_id in variants:
            scenario_id = f"{topic.decision_id}_v{variant_id}"
            scenario = scenarios.get(scenario_id) or {}
            if not scenario.get("scenario_text"):
                missing.append(f"{scenario_id}: no accepted scenario draft")
                continue
            bodies, call_ids, allocations = {}, {}, {}
            for option in ("opt_1", "opt_2"):
                recorded = groups.get((scenario_id, option)) or {}
                if not recorded.get("bodies"):
                    missing.append(f"{scenario_id}/{option}: no usable group response "
                                   f"({recorded.get('outcome') or 'no call recorded'})")
                    continue
                if not recorded.get("accepted"):
                    # A group the model never got right may still be assembled,
                    # but only through recorded corrections bound to this exact
                    # call — and only if the corrected cells then validate.
                    corrected = [c for c in corrections
                                 if (c.scenario_id, c.supported_option) == (scenario_id, option)]
                    if not corrected:
                        missing.append(
                            f"{scenario_id}/{option}: ended "
                            f"{recorded.get('outcome')} and has no recorded correction")
                        continue
                allocation = by_group.get((topic.decision_id, variant_id, option))
                if allocation is None:
                    missing.append(f"{scenario_id}/{option}: no marker allocation")
                    continue
                bodies[option] = recorded["bodies"]
                call_ids[option] = recorded["call_id"]
                allocations[option] = allocation
            if len(bodies) != 2:
                continue

            mine = [c for c in corrections if c.scenario_id == scenario_id]
            record, applied = assemble_scenario(
                topic=topic, variant_id=variant_id,
                scenario_text=scenario["scenario_text"],
                scenario_call_id=scenario["call_id"], groups=bodies,
                group_call_ids=call_ids, allocations=allocations, approvals=approvals,
                cfg=cfg, segmenter=segmenter, corrections=mine,
                topic_bank_content_hash=topic_bank_content_hash)
            records.append(record)
            manifests.append(assembly_manifest(record, scenario_call_id=scenario["call_id"],
                                               group_call_ids=call_ids, corrections=applied))

    if missing:
        raise AssemblyRefused(
            "the pilot is incomplete, so nothing was assembled:\n  - " + "\n  - ".join(missing))

    report = validate_corpus(records, cfg, segmenter, corpus_scope="pilot")
    if report.errors:
        codes = sorted({f.code for f in report.errors})
        raise AssemblyRefused(
            f"full-corpus validation found {len(report.errors)} error(s) {codes}; a machine "
            f"error is never assembled, and no approval overrides one")

    manifest = {
        "config_version": cfg.config_version,
        "config_content_hash": cfg.content_hash,
        "topic_bank_content_hash": topic_bank_content_hash,
        "corpus_scope": report.corpus_scope,
        "n_scenarios": len(records),
        "n_texts": report.n_texts,
        "machine_errors": len(report.errors),
        "machine_warnings": len(report.warnings),
        "outstanding_human_review": len(report.human_review),
        "validation_status": "draft",
        "scenarios": manifests,
        "note": ("Machine-valid and scenario-approved. Every item, pair and scenario "
                 "judgement is still outstanding, and every record stays draft until they "
                 "are recorded."),
    }
    return records, manifest


def corpus_matches_manifest(corpus_path: str | Path,
                            manifest_path: str | Path) -> tuple[bool, str]:
    """``(consistent, why)`` for a corpus and the manifest beside it.

    The manifest records the corpus's SHA-256, so a pair left mismatched by an
    interruption is detectable rather than merely unlikely. Run this before
    trusting a corpus you did not just build.
    """
    corpus_path, manifest_path = Path(corpus_path), Path(manifest_path)
    if not corpus_path.is_file():
        return (False, f"{corpus_path} is missing")
    if not manifest_path.is_file():
        return (False, f"{manifest_path} is missing: the corpus was written but its manifest "
                       f"was not, so rebuild or re-run assembly")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = sha256_of(corpus_path.read_text(encoding="utf-8"))
    if manifest.get("corpus_sha256") != actual:
        return (False, f"{manifest_path} describes corpus {str(manifest.get('corpus_sha256'))[:12]}, "
                       f"but {corpus_path} hashes to {actual[:12]}")
    return (True, "the corpus and its manifest agree")


def write_pilot_corpus(records, manifest: dict[str, Any], *, corpus_path: str | Path,
                       manifest_path: str | Path | None = None,
                       overwrite: bool = False) -> tuple[Path, Path]:
    """Write the corpus, then its manifest, each replaced atomically.

    What this guarantees, exactly: every byte of each file is written to a
    temporary file beside its destination and only then renamed into place, so
    **neither file is ever seen half-written**, and a failure before the first
    rename leaves both destinations untouched.

    What it does not guarantee: the *pair* is not written in one transaction.
    An interruption between the two renames can leave a new corpus beside an
    old or absent manifest. That state is detectable rather than silent — the
    manifest carries the corpus's SHA-256, and
    :func:`corpus_matches_manifest` compares them — and the fix is to run the
    assembly again, which is deterministic.

    An existing corpus is never replaced silently: overwriting is an explicit
    act, because the file it replaces may be what a reviewer has been reading.
    """
    from ..corpus.store import dumps_record

    corpus_path = Path(corpus_path)
    manifest_path = Path(manifest_path or corpus_path.with_suffix(".manifest.json"))
    existing = [p for p in (corpus_path, manifest_path) if p.exists()]
    if existing and not overwrite:
        raise AssemblyRefused(
            f"{', '.join(str(p) for p in existing)} already exists; pass overwrite=True to "
            f"replace it deliberately")

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(dumps_record(record) + "\n" for record in records)
    manifest = {**manifest, "corpus_sha256": sha256_of(body),
                "corpus_file": corpus_path.name}
    temporary = []
    try:
        corpus_tmp = corpus_path.with_name(corpus_path.name + ".tmp")
        corpus_tmp.write_text(body, encoding="utf-8")
        temporary.append(corpus_tmp)
        manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
                                           sort_keys=True) + "\n", encoding="utf-8")
        temporary.append(manifest_tmp)
        corpus_tmp.replace(corpus_path)
        manifest_tmp.replace(manifest_path)
    finally:
        for path in temporary:
            if path.exists():
                path.unlink()
    return corpus_path, manifest_path
