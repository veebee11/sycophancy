"""Annotation units for v3, one level at a time, never combined.

* **stimulus** — every rendered stimulus (body x opening), because a rating
  that may change with the opening must attach to what a reader actually sees;
* **pair** — every styled stimulus with its plain control under the SAME
  opening (2,880), rated separately per opening because an anaphoric marker may
  take the opening as its antecedent; the 960 body-level relationships are
  recorded as structure only;
* **scenario** — the 120 semantic scenarios, each bound to its v2 scenario-text
  hash. No formal scenario rating has been completed, so none is inherited.

Units list what must be judged. None is completed and none is reported as
passed.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..hashing import sha256_of
from .build import Build
from .spec import Spec


def annotation_units(spec: Spec, built: Build) -> list[dict[str, Any]]:
    levels = spec.raw["annotation_levels"]
    stim = levels["stimulus"]
    out: list[dict[str, Any]] = []
    for s in built.stimuli:
        ratings = list(stim["ratings_all"]) + [
            r for r, conditions in stim["ratings_by_condition"].items()
            if s["condition"] in conditions]
        out.append({"level": "stimulus", "unit_id": s["stimulus_id"],
                    "scenario_id": s["scenario_id"], "decision_id": s["decision_id"],
                    "condition": s["condition"], "marker_id": s["marker_id"],
                    "opening_id": s["opening_id"], "ratings": ratings, "status": "outstanding"})
    by_id = {s["stimulus_id"]: s for s in built.stimuli}
    for s in built.stimuli:
        if s["condition"] not in ("RS", "NS"):
            continue
        control = by_id[s["paired_control_stimulus_id"]]
        out.append({"level": "pair",
                    "unit_id": f"{s['stimulus_id']}~{control['stimulus_id']}",
                    "pair_type": f"{s['condition']}_{control['condition']}",
                    "styled_stimulus_id": s["stimulus_id"],
                    "plain_stimulus_id": control["stimulus_id"],
                    "body_pair_id": f"{s['body_id']}~{control['body_id']}",
                    "scenario_id": s["scenario_id"], "decision_id": s["decision_id"],
                    "marker_id": s["marker_id"], "opening_id": s["opening_id"],
                    "ratings": list(levels["pair"]["ratings"]), "status": "outstanding"})
    seen: dict[str, dict] = {}
    for s in built.stimuli:
        seen.setdefault(s["scenario_id"], s)
    for scenario_id, s in sorted(seen.items()):
        out.append({"level": "scenario", "unit_id": f"{spec.version}.{scenario_id}.scenario",
                    "scenario_id": scenario_id, "decision_id": s["decision_id"],
                    "v2_scenario_text_sha256": sha256_of(s["scenario_text"]),
                    "ratings": list(levels["scenario"]["ratings"]),
                    "ratings_proposed": list(levels["scenario"]["ratings_proposed"]),
                    "inherited_ratings": [], "status": "outstanding"})
    return out


def unit_counts(units: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts by level, and by rating within each level — never summed across levels."""
    out: dict[str, Any] = {}
    for level in ("stimulus", "pair", "scenario"):
        mine = [u for u in units if u["level"] == level]
        out[level] = {"units": len(mine),
                      "ratings": dict(sorted(Counter(r for u in mine for r in u["ratings"]).items()))}
        if level == "pair":
            out[level]["body_level_relationships"] = len({u["body_pair_id"] for u in mine})
        if level == "scenario":
            out[level]["ratings_proposed"] = dict(sorted(Counter(
                r for u in mine for r in u["ratings_proposed"]).items()))
            out[level]["inherited_ratings"] = sum(len(u["inherited_ratings"]) for u in mine)
    return out
