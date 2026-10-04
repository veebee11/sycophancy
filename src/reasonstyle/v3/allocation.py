"""Balanced incomplete-block marker allocation for v3, by constrained search.

Every decision has four groups, one per position
``(variant 1, opt_1), (1, opt_2), (2, opt_1), (2, opt_2)``. For each of the two
six-marker families, each of the 15 four-of-six subsets is used by exactly four
decisions (60 blocks), and the decision's four groups receive its four markers,
one each. The block structure is an invariant of every move, never a soft
target. A seeded simulated-annealing search over (a) which block each decision
receives and (b) the order of a block's markers over the four positions then
drives three balance targets to zero violation:

* each marker in 10 groups at each of the four positions;
* each marker in 13/13/14 decisions per domain, the k-th marker of a family
  receiving 14 in ``sorted(domains)[k % 3]``;
* each cross-family (conclusion_result, inference_basis) pair in 6 or 7 groups;
* no lexical collision: a marker is never allocated to a group whose scenario
  text already contains that marker's string.

The search is deterministic for a seed. :func:`allocation_problems`
independently re-checks every constraint, including the ones that hold by
construction.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from itertools import combinations
from typing import Any

from .spec import FAMILIES, Spec, V3Error

POSITIONS: tuple[tuple[int, str], ...] = ((1, "opt_1"), (1, "opt_2"), (2, "opt_1"), (2, "opt_2"))
ALGORITHM = "v3_bib_annealing_v2"


def _targets(spec: Spec, domains: list[str]) -> dict[str, dict[str, int]]:
    out = {}
    for family in FAMILIES:
        for k, marker in enumerate(spec.family_markers(family)):
            out[marker] = {d: (14 if d == domains[k % 3] else 13) for d in domains}
    return out


def scenario_collisions(spec: Spec, scenario_texts: dict[str, str]) -> dict[str, set[str]]:
    """``{scenario_id: marker ids whose string already occurs in that scenario}``."""
    import re
    out: dict[str, set[str]] = {}
    for scenario_id, text in scenario_texts.items():
        hits = {m.marker_id for m in spec.markers.values()
                if re.search(rf"(?<![A-Za-z]){re.escape(m.string)}(?![A-Za-z])", text, re.I)}
        if hits:
            out[scenario_id] = hits
    return out


def solve_allocation(decisions: list[tuple[str, str]], spec: Spec, seed: int,
                     scenario_texts: dict[str, str] | None = None,
                     max_iterations: int = 2_000_000) -> dict[str, list[list[str]]]:
    """``{family: [[marker at each position] for each decision]}``, decisions in
    the order given. Raises :class:`V3Error` if no zero-violation allocation is
    found within ``max_iterations`` (which would not prove infeasibility)."""
    domains = sorted({d for _, d in decisions})
    if len(decisions) != 60 or Counter(d for _, d in decisions) != Counter({d: 20 for d in domains}):
        raise V3Error("v3 allocation needs 60 decisions, 20 per domain")
    rng = random.Random(seed)
    target = _targets(spec, domains)
    dom_of = [d for _, d in decisions]
    collide = scenario_collisions(spec, scenario_texts or {})
    banned = [[collide.get(f"{decision_id}_v{variant}", set()) for variant, _ in POSITIONS]
              for decision_id, _ in decisions]
    state: dict[str, list[list[str]]] = {}
    for family in FAMILIES:
        blocks = [list(s) for s in combinations(spec.family_markers(family), 4) for _ in range(4)]
        rng.shuffle(blocks)
        for block in blocks:
            rng.shuffle(block)
        state[family] = blocks

    dom: Counter = Counter()
    pos: Counter = Counter()
    cross: Counter = Counter()
    for family in FAMILIES:
        for i, block in enumerate(state[family]):
            for p, m in enumerate(block):
                dom[(m, dom_of[i])] += 1
                pos[(m, p)] += 1
    cr, ib = FAMILIES
    for i in range(60):
        for p in range(4):
            cross[(state[cr][i][p], state[ib][i][p])] += 1

    def cost_dom(keys):
        return sum(abs(dom[k] - target[k[0]][k[1]]) for k in keys)

    def cost_pos(keys):
        return sum(abs(pos[k] - 10) for k in keys)

    def cost_cross(keys):
        return sum(max(0, cross[k] - 7) + max(0, 6 - cross[k]) for k in keys)

    all_dom = [(m, d) for m in target for d in domains]
    all_pos = [(m, p) for m in target for p in range(4)]
    all_cross = [(c, i) for c in spec.family_markers(cr) for i in spec.family_markers(ib)]
    def cost_ban(family, rows):
        return sum(state[family][i][p] in banned[i][p] for i in rows for p in range(4))

    total = (cost_dom(all_dom) + cost_pos(all_pos) + cost_cross(all_cross)
             + sum(cost_ban(f, range(60)) for f in FAMILIES))

    def update(family, rows, sign):
        """Remove (-1) or add (+1) the counter contributions of ``rows``."""
        for i in rows:
            for p, m in enumerate(state[family][i]):
                dom[(m, dom_of[i])] += sign
                pos[(m, p)] += sign
                cross[(state[cr][i][p], state[ib][i][p])] += sign

    def cross_keys(rows):
        return {(state[cr][i][p], state[ib][i][p]) for i in rows for p in range(4)}

    def cost(keys_dom, keys_pos, keys_cross):
        return cost_dom(keys_dom) + cost_pos(keys_pos) + cost_cross(keys_cross)

    temperature = 2.0
    for _ in range(max_iterations):
        if total == 0:
            return state
        family = FAMILIES[rng.randrange(2)]
        if rng.random() < 0.5:
            rows = tuple(rng.sample(range(60), 2))
            a, b = rows

            def move(a=a, b=b, family=family):
                state[family][a], state[family][b] = state[family][b], state[family][a]
        else:
            a = rng.randrange(60)
            p, q = rng.sample(range(4), 2)
            rows = (a,)

            def move(a=a, p=p, q=q, family=family):
                block = state[family][a]
                block[p], block[q] = block[q], block[p]
        # Every counter key the move can touch, before and after it (a swap
        # carries each block's markers into the other decision's domain).
        markers = {m for i in rows for m in state[family][i]}
        keys_dom = {(m, dom_of[i]) for i in rows for m in markers}
        keys_pos = {(m, p2) for m in markers for p2 in range(4)}
        keys_cross = cross_keys(rows)
        move()
        keys_cross |= cross_keys(rows)
        move()                                   # a swap is its own inverse
        before = cost(keys_dom, keys_pos, keys_cross) + cost_ban(family, rows)
        update(family, rows, -1)
        move()
        update(family, rows, +1)
        delta = cost(keys_dom, keys_pos, keys_cross) + cost_ban(family, rows) - before
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            total += delta
        else:
            update(family, rows, -1)
            move()
            update(family, rows, +1)
        temperature = max(0.05, temperature * 0.99995)
    raise V3Error(f"no zero-violation allocation found within {max_iterations} iterations "
                  f"(seed {seed}); this does not prove the constraints infeasible")


def allocation_rows(decisions: list[tuple[str, str]], state: dict[str, list[list[str]]]
                    ) -> list[dict[str, Any]]:
    """One row per group, in decision order then position order."""
    rows = []
    for i, (decision_id, domain) in enumerate(decisions):
        for p, (variant, option) in enumerate(POSITIONS):
            rows.append({
                "decision_id": decision_id, "domain": domain,
                "scenario_id": f"{decision_id}_v{variant}", "variant_id": variant,
                "supported_option": option,
                "markers": {family: state[family][i][p] for family in FAMILIES},
            })
    return rows


def allocation_problems(rows: list[dict[str, Any]], spec: Spec,
                        scenario_texts: dict[str, str] | None = None) -> list[str]:
    """Every constraint, re-checked independently of the search. With
    ``scenario_texts``, also the lexical-collision constraint."""
    problems: list[str] = []
    if scenario_texts is not None:
        collide = scenario_collisions(spec, scenario_texts)
        for r in rows:
            clash = set(r["markers"].values()) & collide.get(r["scenario_id"], set())
            if clash:
                problems.append(f"{r['scenario_id']}/{r['supported_option']}: allocated "
                                f"{sorted(clash)}, whose string already occurs in the scenario")
    domains = sorted({r["domain"] for r in rows})
    target = _targets(spec, domains) if len(domains) == 3 else {}
    keys = [(r["scenario_id"], r["supported_option"]) for r in rows]
    if len(rows) != 240 or len(set(keys)) != 240:
        problems.append(f"expected 240 distinct groups, got {len(rows)} ({len(set(keys))} distinct)")
    by_decision: dict[str, list[dict]] = {}
    for r in rows:
        by_decision.setdefault(r["decision_id"], []).append(r)
    if len(by_decision) != 60:
        problems.append(f"expected 60 decisions, got {len(by_decision)}")
    if Counter(rs[0]["domain"] for rs in by_decision.values()) != Counter({d: 20 for d in domains}):
        problems.append("decisions are not 20 per domain")
    for decision_id, rs in by_decision.items():
        positions = sorted((r["variant_id"], r["supported_option"]) for r in rs)
        if positions != sorted(POSITIONS):
            problems.append(f"{decision_id}: groups are not exactly the four positions")
        all_markers = [m for r in rs for m in r["markers"].values()]
        if len(all_markers) != len(set(all_markers)):
            problems.append(f"{decision_id}: a marker repeats within the decision")
    for family in FAMILIES:
        members = set(spec.family_markers(family))
        for r in rows:
            if r["markers"].get(family) not in members:
                problems.append(f"{r['scenario_id']}/{r['supported_option']}: {family} marker "
                                f"{r['markers'].get(family)!r} is not in the family")
        blocks = Counter(frozenset(r["markers"][family] for r in rs)
                         for rs in by_decision.values())
        expected = Counter({frozenset(s): 4 for s in combinations(sorted(members), 4)})
        if blocks != expected:
            problems.append(f"{family}: the 15 four-marker subsets are not each used exactly 4 times")
        groups = Counter(r["markers"][family] for r in rows)
        decisions = Counter(m for rs in by_decision.values()
                            for m in {r["markers"][family] for r in rs})
        pairs = Counter(pair for rs in by_decision.values()
                        for pair in combinations(sorted({r["markers"][family] for r in rs}), 2))
        for m in sorted(members):
            ms = [r for r in rows if r["markers"][family] == m]
            if groups[m] != 40 or decisions[m] != 40:
                problems.append(f"{m}: {groups[m]} groups / {decisions[m]} decisions, not 40 / 40")
            opt = Counter(r["supported_option"] for r in ms)
            var = Counter(r["variant_id"] for r in ms)
            cell = Counter((r["variant_id"], r["supported_option"]) for r in ms)
            if opt != Counter({"opt_1": 20, "opt_2": 20}):
                problems.append(f"{m}: supported options {dict(opt)}, not 20/20")
            if var != Counter({1: 20, 2: 20}):
                problems.append(f"{m}: variants {dict(var)}, not 20/20")
            if cell != Counter({p: 10 for p in POSITIONS}):
                problems.append(f"{m}: variant x option {dict(cell)}, not 10 each")
            dom = Counter(r["domain"] for r in ms)
            if target and dict(dom) != target[m]:
                problems.append(f"{m}: domains {dict(dom)}, not {target[m]}")
        for a, b in combinations(sorted(members), 2):
            if pairs[(a, b)] != 24:
                problems.append(f"{family}: {a}+{b} co-occur in {pairs[(a, b)]} decisions, not 24")
    cr, ib = FAMILIES
    cross = Counter((r["markers"][cr], r["markers"][ib]) for r in rows)
    for c in spec.family_markers(cr):
        for i in spec.family_markers(ib):
            if cross[(c, i)] not in (6, 7):
                problems.append(f"cross pair {c}+{i} occurs {cross[(c, i)]} times, not 6 or 7")
    for scenario_id in {r["scenario_id"] for r in rows}:
        ms = [m for r in rows if r["scenario_id"] == scenario_id for m in r["markers"].values()]
        if len(ms) != 4 or len(set(ms)) != 4:
            problems.append(f"{scenario_id}: not four distinct markers")
    return problems
