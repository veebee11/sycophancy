"""Deterministic machine validation of a v3 build.

Errors refuse the build. Warnings are recorded for review. Every human
judgement a body still needs is emitted as an outstanding code and is never
reported as passed: lexical presence does not show that a marker functions
naturally as an inference marker, or that a pair preserves its propositions.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .allocation import allocation_problems
from .build import Build, counts
from .spec import FAMILIES, Spec

EXPECTED = {
    "decisions": 60, "decisions_per_domain": {"climate": 20, "energy": 20, "technology": 20},
    "scenarios": 120, "groups": 240, "unique_bodies": 1440, "stimuli": 4320,
    "by_condition": {"NP": 720, "NS": 1440, "RP": 720, "RS": 1440},
    "by_opening": {"directive_reconsideration": 1440, "stance_alternative": 1440,
                   "stance_disagreement": 1440},
    "by_family": {"conclusion_result": 1440, "inference_basis": 1440},
    "by_domain": {"climate": 1440, "energy": 1440, "technology": 1440},
    "by_supported_option": {"opt_1": 2160, "opt_2": 2160},
    "per_scenario": [36], "per_decision": [72], "bodies_per_group": [6],
}
_LEAKS = re.compile(r"\b(RS|RP|NS|NP|opt_1|opt_2|conclusion_result|inference_basis|"
                    r"m\d\d_[a-z_]+|option [AB]\b|\(A\)|\(B\)|marker|condition)\b")


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str           # error | warning
    message: str
    where: str = ""


def _count_phrase(text: str, phrase: str) -> int:
    return len(re.findall(rf"(?<![A-Za-z]){re.escape(phrase)}(?![A-Za-z])", text, re.IGNORECASE))


def validate_build(built: Build, spec: Spec, source_records: list[dict], segmenter,
                   ) -> list[Finding]:
    f: list[Finding] = []
    add = lambda code, sev, msg, where="": f.append(Finding(code, sev, msg, where))
    markers = spec.markers
    openings = spec.openings

    # -- counts and identity ---------------------------------------------------
    got = counts(built)
    for key, want in EXPECTED.items():
        if got[key] != want:
            add("E3_COUNT", "error", f"{key} is {got[key]}, expected {want}")
    if got["by_marker"] != {m: 240 for m in markers}:
        add("E3_COUNT", "error", f"styled stimuli per marker {got['by_marker']}, expected 240 each")
    for name, rows, key in (("body", built.bodies, "body_id"),
                            ("stimulus", built.stimuli, "stimulus_id")):
        ids = Counter(r[key] for r in rows)
        dupes = sorted(i for i, n in ids.items() if n > 1)
        if dupes or any(not i for i in ids):
            add("E3_ID", "error", f"duplicate or empty {name} ids: {dupes[:5]}")

    # -- allocation --------------------------------------------------------------
    import yaml
    texts = {r["scenario_id"]: r["scenario_text"] for r in source_records}
    for problem in allocation_problems(yaml.safe_load(built.allocation_text)["groups"], spec, texts):
        add("E3_ALLOCATION", "error", problem)

    # -- source binding ------------------------------------------------------------
    by_scenario = {r["scenario_id"]: r for r in source_records}
    bodies = {b["body_id"]: b for b in built.bodies}
    for b in built.bodies:
        where = b["body_id"]
        src = by_scenario.get(b["scenario_id"])
        if src is None or b["source"]["source_corpus_sha256"] != built.source["source_corpus_sha256"]:
            add("E3_SOURCE", "error", "not bound to the pinned source corpus", where)
            continue
        cells = src["counterarguments"][b["supported_option"]]["cells"]
        rp, np_ = cells["RP"]["body"], cells["NP"]["body"]
        premise = rp[: -len(np_) - 1]
        if (b["scenario_text"] != src["scenario_text"] or b["options"] != src["options"]
                or b["endorsement"] != np_):
            add("E3_SOURCE", "error", "scenario, options or endorsement differ from v2", where)
        if b["condition"] in ("RS", "RP") and b["premise"] != premise:
            add("E3_PREMISE", "error", "premise differs from the corrected v2 premise", where)
        if b["condition"] == "RP" and b["body"] != rp or b["condition"] == "NP" and b["body"] != np_:
            add("E3_SOURCE", "error", "plain control is not the exact v2 body", where)

        # -- markers ---------------------------------------------------------------
        body = b["body"]
        if b["condition"] in ("RS", "NS"):
            m = markers[b["marker_id"]]
            if (b["marker_family"], b["marker_subtype"], b["realization_id"]) != (
                    m.family, m.subtype, m.realization_id):
                add("E3_MARKER_METADATA", "error", "marker family, subtype or realization wrong", where)
            if _count_phrase(body, m.string) != 1:
                add("E3_MARKER_COUNT", "error", f"{m.string!r} occurs "
                    f"{_count_phrase(body, m.string)} times, not exactly once", where)
            others = [o.string for o in markers.values() if o.marker_id != m.marker_id
                      and _count_phrase(body, o.string) and _count_phrase(np_ + premise, o.string) == 0]
            if others:
                add("E3_MARKER_EXTRA", "error", f"other markers present: {others}", where)
            control = bodies.get(b["paired_control_body_id"])
            if control is None or control["condition"] != {"RS": "RP", "NS": "NP"}[b["condition"]] \
                    or (control["scenario_id"], control["supported_option"]) != (
                        b["scenario_id"], b["supported_option"]):
                add("E3_PAIRING", "error", "paired control is not the group's shared plain body", where)
                continue
            stripped = body.replace(m.prefix, "", 1)
            if stripped != control["body"]:
                add("E3_MINIMAL_PAIR", "error", "removing the marker wrapper does not reproduce "
                    "the paired plain body exactly", where)
            expected_start = (premise + " " + m.prefix) if b["condition"] == "RS" else m.prefix
            if not body.startswith(expected_start):
                add("E3_REALIZATION", "error", "marker is not realised by its template", where)
        else:
            present = [o.string for o in markers.values() if _count_phrase(body, o.string)]
            if present:
                add("E3_MARKER_IN_PLAIN", "error", f"plain body contains marker(s) {present}", where)
            variants = b.get("styled_variant_body_ids") or []
            back = sorted(x["body_id"] for x in built.bodies
                          if x.get("paired_control_body_id") == b["body_id"])
            if variants != back or len(back) != 2:
                add("E3_PAIRING", "error", f"shared control maps to {back}, recorded {variants}", where)
        if body.count(b["endorsement"]) != 1 or not body.endswith(b["endorsement"]):
            add("E3_ENDORSEMENT", "error", "exact endorsement is not present once at the end", where)
        if not body[0].isupper() or not body.endswith("."):
            add("E3_PUNCTUATION", "error", "body is not capitalised and full-stopped", where)

        # -- sentence counts -----------------------------------------------------------
        n = segmenter.segment(body).count
        want = 2 if b["condition"] in ("RS", "RP") else 1
        if n != want:
            add("E3_SENTENCES", "error", f"{n} sentence(s), expected {want}", where)

        # Advisory: a scenario that already uses a marker string elsewhere.
        if b["condition"] in ("RS", "NS") and _count_phrase(b["scenario_text"],
                                                           markers[b["marker_id"]].string):
            add("W3_MARKER_STRING_IN_SCENARIO", "warning",
                f"the scenario text itself contains {markers[b['marker_id']].string!r}", where)

    # -- openings and rendering ------------------------------------------------------
    rendered_by_body: dict[str, set[str]] = {}
    for s in built.stimuli:
        where = s["stimulus_id"]
        opening = openings.get(s["opening_id"])
        if opening != s["opening"] or s["rendered"] != f"{opening} {s['body']}":
            add("E3_OPENING", "error", "rendered text is not opening + ' ' + body", where)
        for oid, text in openings.items():
            want = 1 if oid == s["opening_id"] else 0
            if s["rendered"].count(text) != want:
                add("E3_OPENING", "error", f"opening {oid!r} occurs {s['rendered'].count(text)} "
                    f"times, expected {want}", where)
        rendered_by_body.setdefault(s["body_id"], set()).add(s["rendered"][len(opening) + 1:])
        if _LEAKS.search(s["rendered"]):
            add("E3_LEAKAGE", "error", f"condition or metadata leaks into the text: "
                f"{_LEAKS.search(s['rendered']).group(0)!r}", where)
        control = s["paired_control_stimulus_id"]
        if s["condition"] in ("RS", "NS") and control != f"{s['paired_control_body_id']}.{s['opening_id']}":
            add("E3_PAIRING", "error", "paired control stimulus does not share the opening", where)
        n = segmenter.segment(s["rendered"]).count
        want = 3 if s["condition"] in ("RS", "RP") else 2
        if n != want:
            add("E3_SENTENCES", "error", f"rendered text has {n} sentence(s), expected {want}", where)
    for bid, texts in rendered_by_body.items():
        if len(texts) != 1 or texts != {bodies[bid]["body"]}:
            add("E3_OPENING_BODY", "error", "text after the opening differs across openings", bid)
    return f


EXPECTED_UNITS = {
    "stimulus": {"units": 4320, "ratings": {
        **{r: 4320 for r in ("substantive_support", "support_direction_confirmed",
                             "perceived_reasoning_style", "perceived_speaker_commitment",
                             "perceived_naturalness", "perceived_unstated_support", "confidence",
                             "pressure", "politeness", "authority", "credibility")},
        "no_reason_integrity": 2160, "inference_function": 2880}},
    "pair": {"units": 2880, "ratings": {"proposition_preservation": 2880},
             "body_level_relationships": 960},
    "scenario": {"units": 120, "ratings": {"option_feasibility": 120, "option_non_dominance": 120,
                                           "normative_underdetermination": 120},
                 "ratings_proposed": {"option_neutrality": 120}, "inherited_ratings": 0},
}


def unit_problems(summary: dict[str, Any]) -> list[Finding]:
    out = []
    for level, want in EXPECTED_UNITS.items():
        got = summary.get(level, {})
        for key, value in want.items():
            expected = dict(sorted(value.items())) if isinstance(value, dict) else value
            if got.get(key) != expected:
                out.append(Finding("E3_ANNOTATION_UNITS", "error",
                                   f"{level}.{key} is {got.get(key)}, expected {expected}"))
    return out


def balance_tables(built: Build) -> dict[str, Any]:
    rows = [b for b in built.bodies if b["condition"] == "RS"]
    out: dict[str, Any] = {}
    for key in ("supported_option", "variant_id", "domain"):
        out[f"groups_per_marker_by_{key}"] = {
            m: dict(sorted(Counter(str(r[key]) for r in rows if r["marker_id"] == m).items()))
            for m in sorted({r["marker_id"] for r in rows})}
    by_group: dict[tuple, dict[str, str]] = {}
    for r in rows:
        by_group.setdefault((r["scenario_id"], r["supported_option"]), {})[r["marker_family"]] = r["marker_id"]
    cross = Counter((g[FAMILIES[0]], g[FAMILIES[1]]) for g in by_group.values())
    out["cross_family_pair_counts"] = dict(sorted(Counter(cross.values()).items()))
    decision_sets: dict[str, dict[str, set]] = {}
    for r in rows:
        decision_sets.setdefault(r["decision_id"], {}).setdefault(r["marker_family"], set()).add(r["marker_id"])
    pair = Counter(p for sets in decision_sets.values() for fam in sets.values()
                   for p in combinations(sorted(fam), 2))
    out["within_family_pair_cooccurrence"] = dict(sorted(Counter(pair.values()).items()))
    return out
