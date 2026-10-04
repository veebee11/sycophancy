"""Build the v3 stimulus set from the corrected full-v2 corpus. No model call.

Per ``(scenario_id, supported_option)`` group, from the exact v2 bodies:

* ``RP`` — the v2 reason-present plain body, unchanged;
* ``NP`` — the v2 no-reason plain body, unchanged (the exact endorsement);
* the premise — ``RP`` minus ``" " + NP``, required to be exact;
* for each of the group's two allocated markers (one per family):
  ``RS = premise + " " + prefix + NP`` and ``NS = prefix + NP``.

Six unique bodies per group; each RS points to the group's one RP and each NS
to its one NP, which both marker variants share. Every body is rendered once
under each of the three openings: ``opening + " " + body``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..hashing import canonical_json, file_sha256, sha256_of
from .allocation import ALGORITHM, allocation_problems, allocation_rows, solve_allocation
from .spec import FAMILIES, Spec, V3Error

#: Paths v3 must never write: the v2 corpus and everything it rests on.
PROTECTED_PREFIXES = ("data/full/", "data/pilot/", "configs/", "data/topics/", "data/sources/")


@dataclass(frozen=True)
class Build:
    allocation_text: str
    bodies: list[dict[str, Any]]
    stimuli: list[dict[str, Any]]
    source: dict[str, Any]


ARTIFACTS = ("allocation", "bodies", "stimuli", "annotation_units", "reliability_proposal",
             "manifest")


def artifact_texts(built: Build, units: list[dict[str, Any]], reliability: dict[str, Any],
                   manifest: dict[str, Any] | None = None) -> dict[str, str]:
    """Every emitted artifact as the exact text written to disk."""
    texts = {"allocation": built.allocation_text, "bodies": jsonl(built.bodies),
             "stimuli": jsonl(built.stimuli), "annotation_units": jsonl(units),
             "reliability_proposal": json.dumps(reliability, indent=1, sort_keys=True,
                                                ensure_ascii=False) + "\n"}
    if manifest is not None:
        texts["manifest"] = json.dumps(manifest, indent=2, sort_keys=True,
                                       ensure_ascii=False) + "\n"
    return texts


def source_records(spec: Spec, root: Path = Path(".")) -> tuple[list[dict], list[str], dict]:
    """The v2 records and their exact lines, after verifying both pinned hashes."""
    src = spec.source
    corpus, manifest = root / src["corpus"], root / src["manifest"]
    problems = []
    if file_sha256(corpus) != src["source_corpus_sha256"]:
        problems.append(f"{src['corpus']} hashes to {file_sha256(corpus)}, not the pinned "
                        f"{src['source_corpus_sha256']}")
    if file_sha256(manifest) != src["source_manifest_sha256"]:
        problems.append(f"{src['manifest']} is not the pinned source manifest")
    if problems:
        raise V3Error("the v2 source does not verify; nothing was built:\n  - "
                      + "\n  - ".join(problems))
    lines = corpus.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    info = {"corpus": src["corpus"], "source_corpus_sha256": src["source_corpus_sha256"],
            "manifest": src["manifest"], "source_manifest_sha256": src["source_manifest_sha256"],
            "config": src["config"], "config_content_hash": src["config_content_hash"]}
    return records, lines, info


def decisions_of(records: list[dict]) -> list[tuple[str, str]]:
    return sorted({(r["decision_id"], r["domain"]) for r in records})


def scenario_texts_of(records: list[dict]) -> dict[str, str]:
    return {r["scenario_id"]: r["scenario_text"] for r in records}


def allocation_yaml(spec: Spec, decisions: list[tuple[str, str]], source: dict,
                    scenario_texts: dict[str, str]) -> str:
    """The allocation as deterministic YAML, solved from the recorded seed."""
    rows = allocation_rows(decisions, solve_allocation(decisions, spec, spec.seed,
                                                       scenario_texts=scenario_texts))
    problems = allocation_problems(rows, spec, scenario_texts)
    if problems:
        raise V3Error("the solved allocation fails its own checks:\n  - " + "\n  - ".join(problems))
    body = {
        "dataset_version": spec.version,
        "generated": "deterministically by scripts/build_v3.py; do not edit by hand",
        "algorithm": ALGORITHM,
        "seed": spec.seed,
        "source_corpus_sha256": source["source_corpus_sha256"],
        "groups": rows,
    }
    return yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=100)


def load_allocation_rows(text: str, spec: Spec, source: dict,
                         scenario_texts: dict[str, str]) -> list[dict[str, Any]]:
    data = yaml.safe_load(text)
    problems = []
    if data.get("dataset_version") != spec.version or data.get("seed") != spec.seed:
        problems.append("the allocation names a different dataset version or seed")
    if data.get("source_corpus_sha256") != source["source_corpus_sha256"]:
        problems.append("the allocation was made for a different source corpus")
    problems += allocation_problems(data.get("groups") or [], spec, scenario_texts)
    if problems:
        raise V3Error("the allocation does not verify; nothing was built:\n  - "
                      + "\n  - ".join(problems))
    return data["groups"]


def body_id(spec: Spec, scenario_id: str, option: str, condition: str, marker: str | None) -> str:
    return f"{spec.version}.{scenario_id}.{option}.{condition}.{marker or 'none'}"


def build(spec: Spec, allocation_text: str | None = None, root: Path = Path(".")) -> Build:
    """Everything, in memory. ``allocation_text`` is the recorded allocation;
    omitted, it is solved from the seed."""
    records, lines, source = source_records(spec, root)
    decisions = decisions_of(records)
    texts = scenario_texts_of(records)
    if allocation_text is None:
        allocation_text = allocation_yaml(spec, decisions, source, texts)
    rows = load_allocation_rows(allocation_text, spec, source, texts)
    markers = spec.markers
    by_scenario = {r["scenario_id"]: (r, line) for r, line in zip(records, lines)}

    bodies: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: (r["scenario_id"], r["supported_option"])):
        record, line = by_scenario[row["scenario_id"]]
        option = row["supported_option"]
        cells = record["counterarguments"][option]["cells"]
        rp, np_ = cells["RP"]["body"], cells["NP"]["body"]
        if not rp.endswith(" " + np_) or len(rp) <= len(np_) + 1:
            raise V3Error(f"{row['scenario_id']}/{option}: RP is not premise + ' ' + NP; the "
                          f"premise cannot be recovered deterministically")
        premise = rp[: -len(np_) - 1]
        base = {
            "dataset_version": spec.version, "decision_id": record["decision_id"],
            "domain": record["domain"], "scenario_id": record["scenario_id"],
            "variant_id": record["variant_id"], "supported_option": option,
            "scenario_text": record["scenario_text"], "options": dict(record["options"]),
            "endorsement": np_,
            "source": {"source_corpus_sha256": source["source_corpus_sha256"],
                       "source_record_id": record["scenario_id"],
                       "source_record_line_sha256": sha256_of(line),
                       "source_rp_sha256": sha256_of(rp), "source_np_sha256": sha256_of(np_)},
        }
        rp_id = body_id(spec, record["scenario_id"], option, "RP", None)
        np_id = body_id(spec, record["scenario_id"], option, "NP", None)
        styled_ids = {"RP": [], "NP": []}
        group: list[dict[str, Any]] = []
        for family in FAMILIES:
            m = markers[row["markers"][family]]
            marker_fields = {"marker_id": m.marker_id, "marker_string": m.string,
                             "marker_family": m.family, "marker_subtype": m.subtype,
                             "realization_id": m.realization_id, "marker_prefix": m.prefix}
            for condition, text, control_id, control in (
                    ("RS", f"{premise} {m.prefix}{np_}", rp_id, "RP"),
                    ("NS", f"{m.prefix}{np_}", np_id, "NP")):
                bid = body_id(spec, record["scenario_id"], option, condition, m.marker_id)
                styled_ids[control].append(bid)
                group.append({**base, "body_id": bid, "condition": condition,
                              "reason": "present" if condition == "RS" else "absent",
                              "style": "explicit", **marker_fields,
                              "premise": premise if condition == "RS" else None,
                              "body": text, "paired_control_body_id": control_id,
                              "styled_variant_body_ids": None})
        none = {"marker_id": "none", "marker_string": None, "marker_family": "none",
                "marker_subtype": "none", "realization_id": "none", "marker_prefix": None}
        for condition, text, bid in (("RP", rp, rp_id), ("NP", np_, np_id)):
            group.append({**base, "body_id": bid, "condition": condition,
                          "reason": "present" if condition == "RP" else "absent",
                          "style": "plain", **none,
                          "premise": premise if condition == "RP" else None, "body": text,
                          "paired_control_body_id": None,
                          "styled_variant_body_ids": sorted(styled_ids[condition])})
        order = {"RS": 0, "NS": 1, "RP": 2, "NP": 3}
        family_order = {"conclusion_result": 0, "inference_basis": 1, "none": 2}
        bodies += sorted(group, key=lambda b: (order[b["condition"]],
                                               family_order[b["marker_family"]]))

    stimuli: list[dict[str, Any]] = []
    for b in bodies:
        for opening_id, opening in spec.openings.items():
            control = (f"{b['paired_control_body_id']}.{opening_id}"
                       if b["paired_control_body_id"] else None)
            stimuli.append({
                "stimulus_id": f"{b['body_id']}.{opening_id}",
                **{k: v for k, v in b.items() if k not in ("styled_variant_body_ids",)},
                "opening_id": opening_id, "opening": opening,
                "rendered": f"{opening} {b['body']}",
                "paired_control_stimulus_id": control,
            })
    return Build(allocation_text=allocation_text, bodies=bodies, stimuli=stimuli, source=source)


def jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(canonical_json(r) + "\n" for r in rows)


def check_output_paths(spec: Spec, out: dict[str, Path], *, overwrite: bool) -> None:
    """Refuse any v2 or other protected path, and any silent overwrite."""
    problems = []
    for name, path in out.items():
        posix = Path(path).as_posix()
        rel = posix[2:] if posix.startswith("./") else posix
        if any(rel.startswith(p) or f"/{p}" in posix for p in PROTECTED_PREFIXES):
            problems.append(f"{name} {posix} is inside a protected v2 location")
        if Path(path).exists() and not overwrite:
            problems.append(f"{name} {posix} already exists; pass overwrite deliberately")
    if problems:
        raise V3Error("refusing to write; nothing was written:\n  - " + "\n  - ".join(problems))


def write_outputs(spec: Spec, texts: dict[str, str], out: dict[str, Path], *,
                  overwrite: bool = False) -> None:
    """Write every artifact atomically, after refusing protected paths and overwrites."""
    check_output_paths(spec, out, overwrite=overwrite)
    temporary = []
    try:
        for name in ARTIFACTS:
            path = Path(out[name])
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(texts[name], encoding="utf-8")
            temporary.append((tmp, path))
        for tmp, path in temporary:
            tmp.replace(path)
    finally:
        for tmp, _ in temporary:
            if tmp.exists():
                tmp.unlink()


def counts(built: Build) -> dict[str, Any]:
    s = built.stimuli
    return {
        "decisions": len({x["decision_id"] for x in s}),
        "decisions_per_domain": dict(sorted(Counter(
            domain for _, domain in {(x["decision_id"], x["domain"]) for x in s}).items())),
        "scenarios": len({x["scenario_id"] for x in s}),
        "groups": len({(x["scenario_id"], x["supported_option"]) for x in s}),
        "unique_bodies": len(built.bodies),
        "stimuli": len(s),
        "by_condition": dict(sorted(Counter(x["condition"] for x in s).items())),
        "by_opening": dict(sorted(Counter(x["opening_id"] for x in s).items())),
        "by_marker": dict(sorted(Counter(x["marker_id"] for x in s
                                         if x["marker_id"] != "none").items())),
        "by_family": dict(sorted(Counter(x["marker_family"] for x in s
                                         if x["marker_family"] != "none").items())),
        "by_domain": dict(sorted(Counter(x["domain"] for x in s).items())),
        "by_supported_option": dict(sorted(Counter(x["supported_option"] for x in s).items())),
        "per_scenario": sorted(set(Counter(x["scenario_id"] for x in s).values())),
        "per_decision": sorted(set(Counter(x["decision_id"] for x in s).values())),
        "bodies_per_group": sorted(set(Counter((b["scenario_id"], b["supported_option"])
                                               for b in built.bodies).values())),
    }


def _relative(path: Path) -> str:
    """A repository-relative path when ``path`` lies inside the working tree, so
    the manifest does not depend on how the spec path was spelled."""
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def make_manifest(spec: Spec, built: Build, findings, units_summary: dict[str, Any],
                  reliability: dict[str, Any], balance: dict[str, Any],
                  texts: dict[str, str]) -> dict[str, Any]:
    """Provenance, counts, balance, findings, annotation units by level and the
    hash of every other artifact."""
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]
    return {
        "dataset_version": spec.version,
        "status": "draft",
        "spec": _relative(spec.path),
        "spec_sha256": file_sha256(spec.path),
        "source_corpus": built.source["corpus"],
        "source_corpus_sha256": built.source["source_corpus_sha256"],
        "source_manifest_sha256": built.source["source_manifest_sha256"],
        "source_config_content_hash": built.source["config_content_hash"],
        "relationship_to_v2": ("a new derived dataset version; not a correction to v2, which "
                               "is unchanged"),
        "allocation": {"seed": spec.seed, "algorithm": ALGORITHM,
                       "lexical_collision": spec.raw["allocation"]["lexical_collision"],
                       "sha256": sha256_of(built.allocation_text)},
        "artifacts_sha256": {name: sha256_of(texts[name]) for name in ARTIFACTS
                             if name != "manifest"},
        "counts": counts(built),
        "balance": balance,
        "plain_controls": ("RP and NP are shared: each group's two RS marker variants pair with "
                           "its one RP, and its two NS variants with its one NP, so RP and NP "
                           "have half as many rows as RS and NS. This is by design, not missing "
                           "data; the two styled variants of a group are dependent through their "
                           "shared control."),
        "machine_validation": {
            "errors": len(errors), "warnings": len(warnings),
            "warning_detail": [{"code": f.code, "where": f.where, "message": f.message}
                               for f in warnings]},
        "annotation_units": units_summary,
        "annotation_note": ("Counts are per level and per rating and are never summed across "
                            "levels. Every unit is outstanding; no rating is completed, inherited "
                            "or passed. Proposition preservation is rated per rendered pair under "
                            "each opening, because an anaphoric marker may take the opening as "
                            "its antecedent."),
        "reliability_proposal": {
            "status": reliability["status"], "seed": reliability["seed"],
            "reuses_v2_pin": reliability["reuses_v2_pin"],
            "sizes": {k: v["size"] for k, v in reliability["samples"].items()},
            "separation": {k: {a: {x: r[x] for x in ("achieved_all_pairs", "satisfied", "method")}
                               for a, r in v["separation"].items()}
                           for k, v in reliability["samples"].items()},
            "coverage": reliability["coverage"],
            "annotation_begun": reliability["annotation_begun"]},
        "open_semantic_review_flags": [
            "\"This implies that I support ...\" may be pragmatically less natural: a "
            "preference is framed as an implication (route: MC4, MC7)",
            "anaphoric NS markers (\"Given this\", \"For this reason\", \"That is why\", ...) "
            "may point to the scenario or opening and imply unstated support (route: MC8)"],
        "manipulation_checks": ("configs/manipulation_checks_v3_proposed.yaml — proposed, not "
                                "approved; configs/manipulation_checks_v1.yaml unchanged"),
        "formal_annotation": "not started",
        "behavioural_evaluation": "not started",
    }
