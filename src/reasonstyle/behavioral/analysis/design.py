"""Balanced four-cell blocks from validated behavioural rows.

A block is ``run x scenario_id x order_id x opening_id`` for a non-tied initial
prompt: two RS and two NS marker rows (the same two markers, one per family)
and one shared RP and one shared NP. RS and NS are the unweighted mean of their
two marker rows; RP and NP are entered once. Exact post ties keep their movement
and carry ``None`` flips, which flip estimators skip.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .validate import CONDITIONS, FAMILIES, PLAIN_MARKER, AnalysisError, ValidatedEvidence

STYLED = ("RS", "NS")
PLAIN = ("RP", "NP")


@dataclass(frozen=True)
class Block:
    run: str
    model: str
    initial_id: str
    opening_id: str
    decision_id: str
    domain: str
    scenario_id: str
    order_id: str
    supported_option: str
    initial_option: str
    m_before: float
    markers: dict[str, str]                         # family -> marker_id
    movement: dict[str, float]                      # core cell -> value
    marker_movement: dict[str, dict[str, float]]    # RS/NS -> marker -> value
    flips: dict[str, dict[str, bool | None]]        # cond -> marker ("none" for plain) -> flip
    input_tokens: dict[str, dict[str, int]]         # cond -> marker -> post input tokens

    @property
    def abs_m_before(self) -> float:
        return abs(self.m_before)

    def flip_cell(self, cond: str, marker: str | None = None) -> float | None:
        values = [f for m, f in self.flips[cond].items()
                  if f is not None and (marker is None or m == marker)]
        return sum(1.0 for f in values if f) / len(values) if values else None

    def record(self) -> dict[str, Any]:
        return {"run": self.run, "model": self.model, "initial_id": self.initial_id,
                "opening_id": self.opening_id, "decision_id": self.decision_id,
                "domain": self.domain, "scenario_id": self.scenario_id,
                "order_id": self.order_id, "supported_option": self.supported_option,
                "initial_option": self.initial_option, "m_before": self.m_before,
                "abs_m_before": self.abs_m_before, "markers": dict(self.markers),
                "movement_cells": dict(self.movement),
                "movement_by_marker": {k: dict(v) for k, v in self.marker_movement.items()},
                "flip_rows": {k: dict(v) for k, v in self.flips.items()},
                "flip_cells": {c: self.flip_cell(c) for c in CONDITIONS}}


def build_blocks(evidence: ValidatedEvidence) -> list[Block]:
    blocks: list[Block] = []
    for name, run in evidence.runs.items():
        option = {r["initial_id"]: r["initial_option"] for r in run.initials}
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in run.scores:
            grouped[(row["initial_id"], row["opening_id"])].append(row)
        if len(grouped) != 3 * len({r["initial_id"] for r in run.scores}):
            raise AnalysisError(f"{name}: not every initial has three openings")
        for (initial_id, opening_id), rows in sorted(grouped.items()):
            blocks.append(make_block(name, run.model, initial_id, opening_id, rows,
                                     option[initial_id]))
    return blocks


def make_block(run: str, model: str, initial_id: str, opening_id: str,
               rows: list[dict[str, Any]], initial_option: str) -> Block:
    where = f"{run}:{initial_id}:{opening_id}"
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)
    if len(rows) != 6 or {c: len(v) for c, v in by_cond.items()} != {
            "RS": 2, "NS": 2, "RP": 1, "NP": 1}:
        raise AnalysisError(f"{where}: block is not 2 RS, 2 NS, 1 RP, 1 NP")
    for key in ("decision_id", "domain", "scenario_id", "order_id", "supported_option",
                "m_before"):
        if len({row[key] for row in rows}) != 1:
            raise AnalysisError(f"{where}: rows disagree on {key}")
    plain_ids = {c: by_cond[c][0]["stimulus_id"] for c in PLAIN}
    for cond in PLAIN:
        if by_cond[cond][0]["marker_id"] != PLAIN_MARKER:
            raise AnalysisError(f"{where}: plain control {cond} carries a marker")
    families = {}
    for cond, control in (("RS", "RP"), ("NS", "NP")):
        fam = {row["marker_family"]: row["marker_id"] for row in by_cond[cond]}
        if sorted(fam) != sorted(FAMILIES) or len(set(fam.values())) != 2:
            raise AnalysisError(f"{where}: {cond} does not hold one marker per family")
        for row in by_cond[cond]:
            if row["paired_control_stimulus_id"] != plain_ids[control]:
                raise AnalysisError(f"{where}: {row['branch_id']} is not paired with this "
                                    f"block's {control} control")
        families[cond] = fam
    if families["RS"] != families["NS"]:
        raise AnalysisError(f"{where}: RS and NS use different markers")
    marker_movement = {c: {r["marker_id"]: r["movement_toward_counter"] for r in by_cond[c]}
                       for c in STYLED}
    movement = {}
    for cond in CONDITIONS:
        if cond in STYLED:
            a, b = (marker_movement[cond][families[cond][f]] for f in FAMILIES)
            movement[cond] = (a + b) / 2.0
        else:
            movement[cond] = by_cond[cond][0]["movement_toward_counter"]
    first = rows[0]
    return Block(
        run=run, model=model, initial_id=initial_id, opening_id=opening_id,
        decision_id=first["decision_id"], domain=first["domain"],
        scenario_id=first["scenario_id"], order_id=first["order_id"],
        supported_option=first["supported_option"], initial_option=initial_option,
        m_before=first["m_before"], markers=dict(sorted(families["RS"].items())),
        movement=movement, marker_movement=marker_movement,
        flips={c: {r["marker_id"]: r["flip"] for r in by_cond[c]} for c in CONDITIONS},
        input_tokens={c: {r["marker_id"]: r.get("input_tokens") for r in by_cond[c]}
                      for c in CONDITIONS})


def valid_initials(evidence: ValidatedEvidence, run: str) -> set[str]:
    return {r["initial_id"] for r in evidence.runs[run].initials if not r["excluded"]}


def common_valid_initials(evidence: ValidatedEvidence, runs=("base", "instruct")) -> set[str]:
    return set.intersection(*(valid_initials(evidence, r) for r in runs))


def common_semantic_choice_initials(evidence: ValidatedEvidence, runs: list[str]) -> set[str]:
    """Initials valid in every listed run with the identical initial semantic option."""
    choice = {name: {r["initial_id"]: r["initial_option"] for r in evidence.runs[name].initials
                     if not r["excluded"]} for name in runs}
    ids = set.intersection(*(set(c) for c in choice.values()))
    return {i for i in ids if len({choice[n][i] for n in runs}) == 1}
