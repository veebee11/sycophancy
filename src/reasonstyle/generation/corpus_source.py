"""The frozen v2 pilot as the seed of the full corpus, imported read-only.

The full corpus is 60 decisions. Twelve of them already exist as a completed,
corrected, assembled and manifest-described corpus — the v2 pilot — and the
whole point is to reuse them rather than pay to regenerate text a person has
already read. This module is the one place that decides whether that reuse is
safe. It mirrors ``scenario_source.py`` one level up: over an assembled
*corpus* rather than a set of scenarios.

Everything is verified before a single record is returned, and every check is
a refusal:

* the seed configuration's content hash is the one the manifest records, and
  the one pinned in the full configuration;
* the corpus and manifest agree, and both match their pinned SHA-256;
* every record validates under the pilot's **own** configuration, and still
  carries that configuration's hash — nothing is restamped;
* the seed is exactly 12 decisions, 24 scenarios, 48 groups and 192 texts;
* the 12 decisions are exactly the curated pilot decisions, and their topic
  definitions in the full bank are the approved pilot definitions;
* every scenario text hash, scenario approval, group call id and correction
  matches the manifest and the committed ledgers;
* the pilot marker allocation is the pinned one, and every record's marker
  fields are its row;
* the seed and the remainder share no decision, and together are exactly the
  60 curated decisions.

The committed corpus, manifest, approvals, correction ledger and allocation are
sufficient. The gitignored run directory, when present, is read as additional
evidence — never written — and its absence invalidates nothing.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, load_config
from ..corpus.schemas import ScenarioRecord
from ..corpus.segmentation import segmenter_from_config
from ..corpus.topics import TopicBank, load_topic_bank
from ..corpus.validate import validate_corpus
from ..hashing import content_hash, sha256_of
from .allocation import AllocationError, MarkerAllocation, load_allocation
from .approvals import APPROVED, load_approvals

__all__ = [
    "SEED_MODE",
    "CorpusPlan",
    "SeedCorpus",
    "SeedCorpusError",
    "apply_seed_overlays",
    "combine_corpus",
    "combined_corpus_problems",
    "load_seed_corpus",
    "plan_full_corpus",
    "seed_corpus_spec",
]

SEED_MODE = "reuse_frozen_v2_pilot"
_CALL_ID = re.compile(r"^[0-9a-f]{64}$")
#: Every pin a seed declaration must carry when it declares any. A pin that is
#: missing, empty or not a SHA-256 is a refusal: a falsy pin would otherwise
#: switch its own check off without anyone noticing.
PIN_KEYS = ("config_content_hash", "corpus_sha256", "manifest_sha256",
            "allocation_content_hash", "corrections_sha256", "topic_bank_content_hash")
#: The correction fields that must survive into the manifest unchanged.
_CORRECTION_FIELDS = ("scenario_id", "supported_option", "condition", "original_call_id",
                      "original_text", "original_text_sha256", "corrected_text",
                      "corrected_text_sha256", "editor", "reason", "decided_at",
                      "approval_state")


class SeedCorpusError(RuntimeError):
    """The declared seed corpus does not verify. Nothing proceeds."""


@dataclass(frozen=True)
class SeedCorpus:
    """The verified seed: its records, their exact bytes, and what to record."""

    records: tuple[ScenarioRecord, ...]
    #: Each record's JSONL line exactly as committed, so a combined corpus can
    #: carry the seed byte for byte rather than re-serialising it.
    lines: dict[str, str]
    manifest: dict[str, Any]
    allocation: MarkerAllocation
    decision_ids: tuple[str, ...]
    provenance: dict[str, Any]

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.lines))


@dataclass(frozen=True)
class CorpusPlan:
    """What the full run must produce, given the seed. Derived, never counted."""

    expected_decisions: tuple[str, ...]
    seed_decisions: tuple[str, ...]
    new_decisions: tuple[str, ...]
    new_scenario_ids: tuple[str, ...]
    new_groups: tuple[tuple[str, str], ...]
    variants: tuple[int, ...]

    @property
    def new_texts(self) -> int:
        return len(self.new_groups) * 4


def seed_corpus_spec(cfg: ExperimentConfig) -> dict[str, Any] | None:
    """The declared seed, or ``None`` for a design that grows from nothing."""
    return cfg.parsed.seed_corpus


def _curated(bank: TopicBank) -> dict[str, Any]:
    return {t.decision_id: t for t in bank.topics if t.status == "curated"}


def load_seed_corpus(cfg: ExperimentConfig, *, bank: TopicBank,
                     root: Path | None = None) -> SeedCorpus:
    """Read and verify the declared seed corpus. Writes nothing, anywhere.

    ``bank`` is the full design's topic bank; the seed's own bank is read from
    the path the seed declares.
    """
    spec = seed_corpus_spec(cfg)
    if not spec:
        raise SeedCorpusError("this configuration declares no seed_corpus")
    if spec.get("mode") != SEED_MODE:
        raise SeedCorpusError(f"unknown seed_corpus mode {spec.get('mode')!r}")
    if spec.get("access") != "read_only":
        raise SeedCorpusError(
            f"seed_corpus.access is {spec.get('access')!r}; a seed is read_only, and a "
            f"design that wrote to it would be altering finished pilot evidence")
    pins = spec.get("pins")
    if pins is not None:
        malformed = [k for k in PIN_KEYS
                     if not (isinstance(pins, dict) and isinstance(pins.get(k), str)
                             and _CALL_ID.match(pins[k]))]
        unknown = sorted(set(pins) - set(PIN_KEYS)) if isinstance(pins, dict) else []
        if malformed or unknown:
            raise SeedCorpusError(
                f"seed_corpus.pins must give every one of {list(PIN_KEYS)} as a SHA-256 "
                f"string; malformed or missing {malformed}, unknown {unknown}")
    pins = pins or {}
    base = Path(root) if root else Path()
    paths = {name: base / spec[name]
             for name in ("config", "corpus", "manifest", "allocation", "corrections", "topics")}
    for name, path in paths.items():
        if not path.is_file():
            raise SeedCorpusError(f"seed {name} {path} does not exist")

    problems: list[str] = []

    # -- configuration --------------------------------------------------------
    seed_cfg = load_config(paths["config"])
    manifest_bytes = paths["manifest"].read_text(encoding="utf-8")
    manifest = json.loads(manifest_bytes)
    if manifest.get("config_content_hash") != seed_cfg.content_hash:
        problems.append(f"{spec['config']} hashes to {seed_cfg.content_hash[:12]}, but the "
                        f"manifest records {str(manifest.get('config_content_hash'))[:12]}")
    if manifest.get("config_version") != seed_cfg.config_version:
        problems.append(f"the manifest names config {manifest.get('config_version')!r}, "
                        f"the seed config is {seed_cfg.config_version!r}")
    if pins.get("config_content_hash") and pins["config_content_hash"] != seed_cfg.content_hash:
        problems.append(f"{spec['config']} hashes to {seed_cfg.content_hash[:12]}, not the "
                        f"pinned {pins['config_content_hash'][:12]}")

    # -- corpus and manifest agree, and match their pins ----------------------
    corpus_text = paths["corpus"].read_text(encoding="utf-8")
    corpus_sha = sha256_of(corpus_text)
    if manifest.get("corpus_sha256") != corpus_sha:
        problems.append(f"the manifest describes corpus "
                        f"{str(manifest.get('corpus_sha256'))[:12]}, but {spec['corpus']} "
                        f"hashes to {corpus_sha[:12]}")
    if pins.get("corpus_sha256") and pins["corpus_sha256"] != corpus_sha:
        problems.append(f"{spec['corpus']} hashes to {corpus_sha[:12]}, not the pinned "
                        f"{pins['corpus_sha256'][:12]}")
    manifest_sha = sha256_of(manifest_bytes)
    if pins.get("manifest_sha256") and pins["manifest_sha256"] != manifest_sha:
        problems.append(f"{spec['manifest']} hashes to {manifest_sha[:12]}, not the pinned "
                        f"{pins['manifest_sha256'][:12]}")
    if problems:
        # Nothing further can be trusted: the records below would be read from
        # a file whose identity is already in doubt.
        raise SeedCorpusError(_refusal(problems))

    lines: dict[str, str] = {}
    records: list[ScenarioRecord] = []
    for lineno, line in enumerate(corpus_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = ScenarioRecord.model_validate_json(line)
        except ValueError as exc:
            raise SeedCorpusError(f"{spec['corpus']}:{lineno} is not a corpus record: {exc}")
        if record.scenario_id in lines:
            problems.append(f"{record.scenario_id} appears twice in the seed corpus")
        lines[record.scenario_id] = line
        records.append(record)

    # -- not restamped, and valid under the pilot's own configuration --------
    for record in records:
        if record.config_content_hash != seed_cfg.content_hash:
            problems.append(f"{record.scenario_id} carries config "
                            f"{record.config_content_hash[:12]}, not the pilot's "
                            f"{seed_cfg.content_hash[:12]}; seed records are never restamped")
    report = validate_corpus(records, seed_cfg, segmenter_from_config(seed_cfg),
                             corpus_scope=manifest.get("corpus_scope") or "pilot")
    if report.errors:
        problems.append(f"the seed has {len(report.errors)} machine error(s) under its own "
                        f"configuration: {sorted({f.code for f in report.errors})}")

    # -- counts ---------------------------------------------------------------
    want = spec.get("contributes") or {}
    groups = [(r.scenario_id, option) for r in records for option in r.counterarguments]
    counts = {"decisions": len({r.decision_id for r in records}),
              "scenarios": len(records), "groups": len(groups), "texts": report.n_texts}
    for key, n in counts.items():
        if key in want and want[key] != n:
            problems.append(f"the seed holds {n} {key}; the configuration declares {want[key]}")
    for key, field in (("scenarios", "n_scenarios"), ("texts", "n_texts")):
        if manifest.get(field) != counts[key]:
            problems.append(f"the manifest records {field} {manifest.get(field)}, the corpus "
                            f"holds {counts[key]}")

    # -- the decisions are exactly the curated pilot decisions ----------------
    pilot_bank = load_topic_bank(paths["topics"])
    pilot_bank_hash = content_hash(pilot_bank.model_dump(mode="json"))
    if manifest.get("topic_bank_content_hash") != pilot_bank_hash:
        problems.append(f"{spec['topics']} hashes to {pilot_bank_hash[:12]}, but the manifest "
                        f"records {str(manifest.get('topic_bank_content_hash'))[:12]}")
    if pins.get("topic_bank_content_hash") and pins["topic_bank_content_hash"] != pilot_bank_hash:
        problems.append(f"{spec['topics']} hashes to {pilot_bank_hash[:12]}, not the pinned "
                        f"{pins['topic_bank_content_hash'][:12]}")
    pilot_curated = _curated(pilot_bank)
    full_curated = _curated(bank)
    seed_ids = sorted({r.decision_id for r in records})
    missing = sorted(set(pilot_curated) - set(seed_ids))
    extra = sorted(set(seed_ids) - set(pilot_curated))
    if missing:
        problems.append(f"the seed is missing curated pilot decision(s) {missing}")
    if extra:
        problems.append(f"the seed carries decision(s) that are not curated pilot decisions "
                        f"{extra}")
    for decision_id in sorted(pilot_curated):
        full = full_curated.get(decision_id)
        if full is None:
            problems.append(f"{decision_id} is not a curated decision in the full topic bank")
        elif full != pilot_curated[decision_id]:
            problems.append(f"{decision_id}: the full topic bank no longer holds the approved "
                            f"pilot definition")
    for record in records:
        topic = pilot_curated.get(record.decision_id)
        if topic is not None and (dict(topic.options) != record.options
                                  or topic.domain != record.domain):
            problems.append(f"{record.scenario_id}: options or domain differ from the "
                            f"approved pilot definition")

    # -- scenarios: text hashes and approvals ---------------------------------
    entries = {e["scenario_id"]: e for e in manifest.get("scenarios") or []}
    if set(entries) != set(lines):
        problems.append(f"the manifest describes scenarios {sorted(set(entries) ^ set(lines))} "
                        f"that the corpus does not, or the reverse")
    source = manifest.get("scenario_source") or {}
    source_scenarios = source.get("scenarios") or {}
    approvals_path = base / source["approvals"] if source.get("approvals") else None
    approvals = load_approvals(approvals_path) if approvals_path and approvals_path.is_file() \
        else None
    if approvals is None:
        problems.append("the manifest's scenario source names no readable approvals file")
    source_cfg_hash = None
    if source.get("config") and (base / source["config"]).is_file():
        source_cfg_hash = load_config(base / source["config"]).content_hash
        if source_cfg_hash != source.get("config_content_hash"):
            problems.append(f"{source['config']} hashes to {source_cfg_hash[:12]}, but the "
                            f"scenarios were approved under "
                            f"{str(source.get('config_content_hash'))[:12]}")
    else:
        problems.append("the manifest's scenario source configuration is not readable")
    for record in records:
        entry = entries.get(record.scenario_id) or {}
        text_sha = sha256_of(record.scenario_text)
        if entry.get("scenario_text_sha256") != text_sha:
            problems.append(f"{record.scenario_id}: the scenario text does not match the "
                            f"manifest's hash")
        recorded = source_scenarios.get(record.scenario_id) or {}
        if recorded.get("scenario_text_sha256") != text_sha or \
                recorded.get("call_id") != entry.get("scenario_call_id"):
            problems.append(f"{record.scenario_id}: the manifest's scenario source does not "
                            f"record this text and call")
        approval = (approvals or {}).get(record.scenario_id)
        if approval is None or approval.decision != APPROVED:
            problems.append(f"{record.scenario_id}: no approved scenario approval")
        elif (approval.call_id != entry.get("scenario_call_id")
              or approval.scenario_text_sha256 != text_sha
              or approval.config_content_hash != source.get("config_content_hash")):
            problems.append(f"{record.scenario_id}: the approval is not bound to this call, "
                            f"text and configuration")

    # -- groups: call ids and corrections -------------------------------------
    call_ids: list[str] = []
    for record in records:
        ids = (entries.get(record.scenario_id) or {}).get("group_call_ids") or {}
        if set(ids) != set(record.counterarguments):
            problems.append(f"{record.scenario_id}: the manifest records group calls for "
                            f"{sorted(ids)}, the record has {sorted(record.counterarguments)}")
        for option, call_id in sorted(ids.items()):
            if not isinstance(call_id, str) or not _CALL_ID.match(call_id):
                problems.append(f"{record.scenario_id}/{option}: {call_id!r} is not a call id")
            call_ids.append(call_id)
    duplicated = sorted(c for c, n in Counter(call_ids).items() if n > 1)
    if duplicated:
        problems.append(f"group call id(s) recorded for more than one group: {duplicated}")

    ledger_text = paths["corrections"].read_text(encoding="utf-8")
    if pins.get("corrections_sha256") and pins["corrections_sha256"] != sha256_of(ledger_text):
        problems.append(f"{spec['corrections']} is not the pinned correction ledger")
    import yaml
    ledger = yaml.safe_load(ledger_text) or []
    manifest_corrections = [c for e in entries.values() for c in e.get("manual_corrections") or []]
    key = lambda c: (c.get("scenario_id"), c.get("supported_option"), c.get("condition"))  # noqa: E731
    by_key = {key(c): c for c in ledger}
    if sorted(map(key, ledger)) != sorted(map(key, manifest_corrections)):
        problems.append("the correction ledger and the manifest name different cells")
    by_scenario = {r.scenario_id: r for r in records}
    for correction in manifest_corrections:
        k = key(correction)
        recorded = by_key.get(k) or {}
        for field in _CORRECTION_FIELDS:
            if str(correction.get(field)) != str(recorded.get(field)):
                problems.append(f"{k}: the manifest's correction field {field!r} differs from "
                                f"the ledger")
        if correction.get("original_text_sha256") != sha256_of(correction.get("original_text", "")) \
                or correction.get("corrected_text_sha256") != sha256_of(
                    correction.get("corrected_text", "")):
            problems.append(f"{k}: a recorded correction hash does not match its text")
        if correction.get("approval_state") != "approved":
            problems.append(f"{k}: the correction is not approved")
        entry = entries.get(correction.get("scenario_id")) or {}
        if (entry.get("group_call_ids") or {}).get(correction.get("supported_option")) != \
                correction.get("original_call_id"):
            problems.append(f"{k}: the correction names a call that is not this group's")
        record = by_scenario.get(correction.get("scenario_id"))
        if record is not None:
            block = record.counterarguments.get(correction.get("supported_option"))
            cell = block.cells.get(correction.get("condition")) if block else None
            if cell is None or cell.body != correction.get("corrected_text"):
                problems.append(f"{k}: the corpus cell is not the corrected text")

    # -- marker allocation ------------------------------------------------------
    try:
        allocation = load_allocation(paths["allocation"])
    except AllocationError as exc:
        raise SeedCorpusError(_refusal(problems + [f"seed allocation: {exc}"])) from exc
    if pins.get("allocation_content_hash") and \
            pins["allocation_content_hash"] != allocation.content_hash:
        problems.append(f"{spec['allocation']} is allocation {allocation.content_hash[:12]}, "
                        f"not the pinned {pins['allocation_content_hash'][:12]}")
    if allocation.config_content_hash != seed_cfg.content_hash:
        problems.append("the seed allocation was not built under the seed configuration")
    rows = {(g.scenario_id, g.supported_option): g for g in allocation.groups}
    if len(rows) != len(allocation.groups) or set(rows) != set(groups):
        problems.append(f"the seed allocation covers {len(rows)} groups; the corpus holds "
                        f"{len(groups)}, and they must be the same groups")
    for record in records:
        for option, block in record.counterarguments.items():
            row = rows.get((record.scenario_id, option))
            if row is None:
                continue
            if (block.marker_family, block.marker_string, block.marker_realization_id) != \
                    (row.marker_family, row.marker_string, row.marker_realization_id):
                problems.append(f"{record.scenario_id}/{option}: the record's marker is not its "
                                f"allocation row")

    # -- the future remainder --------------------------------------------------
    remainder = sorted(set(full_curated) - set(seed_ids))
    union_problems = _union_problems(seed_ids, remainder, cfg, full_curated)
    problems += union_problems

    # -- the gitignored run, when present: read, never written ----------------
    run_evidence = _run_evidence(base / spec["run"], spec["run"], call_ids, seed_cfg,
                                 allocation, problems) if spec.get("run") else "not declared"

    if problems:
        raise SeedCorpusError(_refusal(problems))

    provenance = {
        "role": "seed",
        "mode": spec["mode"],
        "access": "read_only",
        "config": spec["config"],
        "config_version": seed_cfg.config_version,
        # The pilot's own hash. NOT the full design's: a seed record was built,
        # corrected and approved under this configuration and no other.
        "config_content_hash": seed_cfg.content_hash,
        "corpus": spec["corpus"],
        "corpus_sha256": corpus_sha,
        "manifest": spec["manifest"],
        "manifest_sha256": manifest_sha,
        "topic_bank": spec["topics"],
        "topic_bank_content_hash": pilot_bank_hash,
        "allocation": spec["allocation"],
        "allocation_content_hash": allocation.content_hash,
        "corrections": spec["corrections"],
        "corrections_sha256": sha256_of(ledger_text),
        "scenario_source": {k: source.get(k) for k in
                            ("config", "config_version", "config_content_hash", "approvals")},
        "run": spec.get("run"),
        "run_evidence": run_evidence,
        "verified": sorted(spec.get("verify") or ()),
        "pinned": bool(pins),
        "counts": counts,
        "decisions": seed_ids,
        "manual_corrections": len(manifest_corrections),
        "validation_status": manifest.get("validation_status"),
        "scenarios": {e["scenario_id"]: {"scenario_call_id": e["scenario_call_id"],
                                         "scenario_text_sha256": e["scenario_text_sha256"],
                                         "group_call_ids": dict(sorted(e["group_call_ids"].items()))}
                      for e in sorted(entries.values(), key=lambda e: e["scenario_id"])},
    }
    return SeedCorpus(records=tuple(sorted(records, key=lambda r: r.scenario_id)),
                      lines=lines, manifest=manifest, allocation=allocation,
                      decision_ids=tuple(seed_ids), provenance=provenance)


def _union_problems(seed_ids, new_ids, cfg: ExperimentConfig, full_curated) -> list[str]:
    """Seed plus remainder must be exactly the curated decisions, once each."""
    problems = []
    overlap = sorted(set(seed_ids) & set(new_ids))
    if overlap:
        problems.append(f"decision(s) {overlap} are in both the seed and the remainder")
    union = set(seed_ids) | set(new_ids)
    if union != set(full_curated):
        problems.append(f"seed plus remainder is not the curated set: missing "
                        f"{sorted(set(full_curated) - union)}, unexpected "
                        f"{sorted(union - set(full_curated))}")
    wanted = cfg.raw["corpus"]["decisions_full"]
    if len(full_curated) != wanted:
        problems.append(f"the full topic bank holds {len(full_curated)} curated decisions, "
                        f"not {wanted}")
    generate = (seed_corpus_spec(cfg) or {}).get("generate_only") or {}
    if generate.get("decisions") is not None and len(new_ids) != generate["decisions"]:
        problems.append(f"{len(new_ids)} decisions remain to generate; the configuration "
                        f"declares {generate['decisions']}")
    return problems


def _run_evidence(run: Path, declared: str, call_ids, seed_cfg, allocation,
                  problems: list[str]) -> str:
    """Cross-check the gitignored run directory when it exists. Opened for
    reading only; its absence is recorded, not treated as a failure."""
    log = run / "generation_log.jsonl"
    if not log.is_file():
        return "absent (not required: the committed corpus and manifest verify on their own)"
    accepted = {}
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("outcome") == "accepted":
            accepted[entry.get("call_id")] = entry
    for call_id in call_ids:
        entry = accepted.get(call_id)
        if entry is None:
            problems.append(f"run evidence: group call {call_id[:12]} is not an accepted call "
                            f"in {declared}")
        elif entry.get("config_content_hash") != seed_cfg.content_hash or \
                entry.get("allocation_content_hash") != allocation.content_hash:
            problems.append(f"run evidence: call {call_id[:12]} was made under a different "
                            f"configuration or allocation")
    # The declared path, as the configuration writes it: a provenance block that
    # carried this machine's absolute path would not be portable.
    return f"checked read-only: {len(call_ids)} group calls accepted in {declared}"


def _refusal(problems: list[str]) -> str:
    return ("the seed corpus does not verify, so nothing is imported:\n  - "
            + "\n  - ".join(problems))


def plan_full_corpus(cfg: ExperimentConfig, bank: TopicBank, seed: SeedCorpus) -> CorpusPlan:
    """The new material the full run must produce: the curated decisions the seed
    does not supply, with both variants and both supported options each."""
    full_curated = _curated(bank)
    new_ids = sorted(set(full_curated) - set(seed.decision_ids))
    problems = _union_problems(seed.decision_ids, new_ids, cfg, full_curated)
    if problems:
        raise SeedCorpusError(_refusal(problems))
    variants = tuple(range(1, cfg.raw["corpus"]["variants_per_decision"] + 1))
    options = tuple(cfg.raw["corpus"]["supported_options"])
    scenarios = tuple(f"{d}_v{v}" for d in new_ids for v in variants)
    groups = tuple((s, o) for s in scenarios for o in options)
    generate = (seed_corpus_spec(cfg) or {}).get("generate_only") or {}
    for key, n in (("scenarios", len(scenarios)), ("groups", len(groups)),
                   ("texts", len(groups) * 4)):
        if generate.get(key) is not None and generate[key] != n:
            raise SeedCorpusError(f"the plan has {n} new {key}; the configuration declares "
                                  f"{generate[key]}")
    return CorpusPlan(expected_decisions=tuple(sorted(full_curated)),
                      seed_decisions=tuple(seed.decision_ids), new_decisions=tuple(new_ids),
                      new_scenario_ids=scenarios, new_groups=groups, variants=variants)


def combine_corpus(seed: SeedCorpus, plan: CorpusPlan, new_records, new_manifest: dict[str, Any],
                   *, cfg: ExperimentConfig,
                   seed_overlays: dict[str, ScenarioRecord] | None = None
                   ) -> tuple[str, dict[str, Any]]:
    """The combined corpus body and its two-source manifest. Writes nothing.

    Seed records pass through **byte for byte**: their committed lines are
    reused, never re-serialised, and never restamped with the full design's
    hash. The one exception is ``seed_overlays`` (:func:`apply_seed_overlays`):
    a seed record corrected by an approved full-design correction is written
    as corrected, still under the seed's own configuration hash, and the
    caller records that overlay in the manifest. New records must be exactly
    the planned scenarios, built under this design. Order is deterministic: by
    scenario id.
    """
    from ..corpus.store import dumps_record

    new = {r.scenario_id: r for r in new_records}
    problems = []
    clash = sorted(set(new) & set(seed.lines))
    if clash:
        problems.append(f"new records for seed scenario(s) {clash}")
    if set(new) - set(seed.lines) != set(plan.new_scenario_ids):
        problems.append(f"new records do not match the plan: missing "
                        f"{sorted(set(plan.new_scenario_ids) - set(new))}, unexpected "
                        f"{sorted(set(new) - set(plan.new_scenario_ids) - set(seed.lines))}")
    for record in new.values():
        if record.config_content_hash != cfg.content_hash:
            problems.append(f"{record.scenario_id} was not built under this design")
    if problems:
        raise SeedCorpusError("the combined corpus is refused:\n  - " + "\n  - ".join(problems))

    overlays = dict(seed_overlays or {})
    stray = sorted(set(overlays) - set(seed.lines))
    if stray:
        raise SeedCorpusError(f"seed overlays for non-seed scenario(s) {stray}")
    lines = dict(seed.lines)
    lines.update({sid: dumps_record(record) for sid, record in overlays.items()})
    lines.update({sid: dumps_record(record) for sid, record in new.items()})
    body = "".join(lines[sid] + "\n" for sid in sorted(lines))
    manifest = {
        "config_version": cfg.config_version,
        "config_content_hash": cfg.content_hash,
        "corpus_scope": "full",
        "n_scenarios": len(lines),
        "validation_status": "draft",
        "sources": [seed.provenance, {"role": "full_run", **new_manifest}],
        "scenario_sources": {sid: ("seed" if sid in seed.lines else "full_run")
                             for sid in sorted(lines)},
        "corpus_sha256": sha256_of(body),
    }
    return body, manifest


def combined_corpus_problems(seed: SeedCorpus, plan: CorpusPlan, body: str,
                             allocation: MarkerAllocation, cfg: ExperimentConfig,
                             segmenter) -> tuple[list[str], Any]:
    """Every reason the combined corpus ``body`` may not be written, and the
    full-scope validation report it was judged by. Writes nothing.

    The whole corpus is validated at ``full`` scope, so the marker-allocation
    minima and caps and cross-corpus duplicate checks run over all of it. Seed
    records keep the pilot configuration's hash by design (they are carried
    byte for byte, never restamped), so ``E_CONFIG_HASH_MISMATCH`` is expected
    on exactly the seed records, carrying exactly the seed's verified hash, and
    on nothing else. Any other error refuses. Every allocation row must be
    covered exactly once, with its own marker fields, and the counts must be
    exactly the plan's.
    """
    records = [ScenarioRecord.model_validate_json(line) for line in body.splitlines()
               if line.strip()]
    report = validate_corpus(records, cfg, segmenter, corpus_scope="full")
    seed_hash = seed.provenance["config_content_hash"]
    problems: list[str] = []
    expected = 0
    for finding in report.errors:
        if (finding.code == "E_CONFIG_HASH_MISMATCH" and finding.scenario_id in seed.lines
                and finding.detail.get("record") == seed_hash):
            expected += 1
            continue
        problems.append(f"{finding.scenario_id or 'corpus'}: {finding.code}: {finding.message}")
    if expected != len(seed.lines):
        problems.append(f"{expected} seed record(s) carry the seed configuration hash; "
                        f"expected all {len(seed.lines)}")

    rows = {(g.scenario_id, g.supported_option): g for g in allocation.groups}
    if len(rows) != len(allocation.groups):
        problems.append("the allocation names some (scenario, option) more than once")
    found = Counter((r.scenario_id, option) for r in records for option in r.counterarguments)
    duplicated = sorted(k for k, n in found.items() if n > 1)
    missing = sorted(set(rows) - set(found))
    extra = sorted(set(found) - set(rows))
    if duplicated or missing or extra:
        problems.append(f"allocation coverage is not exact: duplicated {duplicated}, "
                        f"missing {missing}, unallocated {extra}")
    for record in records:
        for option, block in record.counterarguments.items():
            row = rows.get((record.scenario_id, option))
            if row is not None and (block.marker_family, block.marker_string,
                                    block.marker_realization_id) != (
                    row.marker_family, row.marker_string, row.marker_realization_id):
                problems.append(f"{record.scenario_id}/{option}: marker fields differ from "
                                f"the allocation")

    n_seed = len(seed.lines)
    want = {"decisions": len(plan.expected_decisions),
            "scenarios": n_seed + len(plan.new_scenario_ids),
            "groups": len(rows), "texts": 4 * len(rows)}
    got = {"decisions": len({r.decision_id for r in records}), "scenarios": len(records),
           "groups": sum(found.values()), "texts": report.n_texts}
    if got != want:
        problems.append(f"the combined corpus has {got}; the plan requires {want}")
    return problems, report


def apply_seed_overlays(seed: SeedCorpus, corrections, segmenter
                        ) -> tuple[dict[str, ScenarioRecord], dict[str, Any]]:
    """Approved full-design corrections to seed records, as overlays. Writes nothing.

    The frozen pilot corpus, its manifest and its own correction ledger are
    never touched: a seed cell is corrected only in the combined full corpus,
    by a correction recorded in the full design's ledger. Each correction must
    name the seed group's recorded call id and the exact seed cell text, and be
    approved (``assemble.correction_problems``); the corrected record keeps the
    seed configuration hash and must still validate under the seed's own
    configuration. Returns the corrected records and their provenance
    (original and corrected line SHA-256, and every applied correction).
    """
    from .assemble import AssemblyRefused, correction_problems

    by_scenario: dict[str, list] = {}
    for correction in corrections:
        if correction.scenario_id in seed.lines:
            by_scenario.setdefault(correction.scenario_id, []).append(correction)
    if not by_scenario:
        return {}, {}
    seed_cfg = load_config(seed.provenance["config"])
    records = {r.scenario_id: r for r in seed.records}
    overlays: dict[str, ScenarioRecord] = {}
    provenance: dict[str, Any] = {}
    for scenario_id, mine in sorted(by_scenario.items()):
        record = records[scenario_id]
        call_ids = (seed.provenance["scenarios"].get(scenario_id) or {}).get("group_call_ids") or {}
        blocks = dict(record.counterarguments)
        problems: list[str] = []
        for correction in mine:
            block = blocks.get(correction.supported_option)
            if block is None or correction.condition not in block.cells:
                problems.append(f"{correction.key}: no such seed cell")
                continue
            cell = block.cells[correction.condition]
            problems += correction_problems(correction, cell.body,
                                            call_ids.get(correction.supported_option))
            cells = dict(block.cells)
            cells[correction.condition] = cell.model_copy(update={"body": correction.corrected_text})
            blocks[correction.supported_option] = block.model_copy(update={"cells": cells})
        if problems:
            raise AssemblyRefused("; ".join(problems))
        corrected = record.model_copy(update={"counterarguments": blocks})
        report = validate_corpus([corrected], seed_cfg, segmenter, corpus_scope="pilot")
        if report.errors:
            raise AssemblyRefused(
                f"{scenario_id}: the corrected seed record has machine errors under the seed's "
                f"own configuration {sorted({f.code for f in report.errors})}")
        from ..corpus.store import dumps_record
        overlays[scenario_id] = corrected
        provenance[scenario_id] = {
            "original_line_sha256": sha256_of(seed.lines[scenario_id]),
            "corrected_line_sha256": sha256_of(dumps_record(corrected)),
            "seed_config_content_hash": corrected.config_content_hash,
            "corrections": [c.as_dict() for c in mine],
        }
    return overlays, provenance
