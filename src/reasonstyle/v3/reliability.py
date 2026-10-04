"""A PROPOSED, deterministic reliability-sample design for v3. Not approved; no
annotation has begun. It does not reuse the v2 pin.

Siblings are keyed by ``decision_id``: one decision holds every closely related
version of a text — the same body under the other openings, the two marker
variants sharing one plain control, the opposite supported-option group of the
same scenario, and the other scenario variant. Each annotator's order keeps
siblings at least ``min_sibling_separation`` positions apart (``|i - j| >= 20``:
20 positions apart, meaning at least 19 intervening items), found by the exact
search in ``reasonstyle.corpus.review`` and re-checked over every pair.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from ..corpus.review import order_with_separation
from .build import Build
from .spec import Spec, V3Error

ANNOTATORS = ("annotator_1", "annotator_2")


def _rng(seed: int, salt: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"{seed}:{salt}".encode()).hexdigest(), 16) % 2**32)


def _draw(strata: dict[tuple, list[dict]], per: dict[str, int], used: set[str], body_key,
          rng: random.Random) -> list[dict]:
    """One draw per stratum, most-constrained strata first, never reusing a body."""
    keys = sorted(strata)
    rng.shuffle(keys)
    keys.sort(key=lambda k: len(strata[k]))
    chosen = []
    for key in keys:
        candidates = sorted((c for c in strata[key] if body_key(c) not in used),
                            key=lambda c: c["unit_id"])
        rng.shuffle(candidates)
        need = per[key[0]]
        if len(candidates) < need:
            raise V3Error(f"stratum {key} has only {len(candidates)} unused candidate(s) for {need}")
        for c in candidates[:need]:
            used.add(body_key(c))
            chosen.append(c)
    return chosen


def _all_pairs_min(order: list[str], decision_of: dict[str, str]) -> int | None:
    d = [abs(i - j) for i, j in combinations(range(len(order)), 2)
         if decision_of[order[i]] == decision_of[order[j]]]
    return min(d) if d else None


def propose_reliability(spec: Spec, built: Build, units: list[dict[str, Any]]) -> dict[str, Any]:
    cfg = spec.raw["reliability"]
    seed, minimum = int(cfg["seed"]), int(cfg["min_sibling_separation"])
    stim_by_id = {s["stimulus_id"]: s for s in built.stimuli}

    # -- stimulus sample ---------------------------------------------------------
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for u in (u for u in units if u["level"] == "stimulus"):
        s = stim_by_id[u["unit_id"]]
        if s["condition"] in ("RS", "NS"):
            key = ("styled", s["marker_id"], s["condition"], s["domain"], s["opening_id"],
                   s["supported_option"])
        else:
            key = ("plain", s["condition"], s["domain"], s["opening_id"], s["supported_option"])
        strata[key].append({**u, "body_id": s["body_id"]})
    per = {"styled": cfg["stimulus_sample"]["styled_per_stratum"],
           "plain": cfg["stimulus_sample"]["plain_per_stratum"]}
    items = _draw(strata, per, set(), lambda c: c["body_id"], _rng(seed, "stimulus-sample"))

    # -- pair sample ----------------------------------------------------------------
    pstrata: dict[tuple, list[dict]] = defaultdict(list)
    for u in (u for u in units if u["level"] == "pair"):
        s = stim_by_id[u["styled_stimulus_id"]]
        pstrata[("pair", u["marker_id"], u["pair_type"], u["opening_id"], s["domain"])].append(
            {**u, "styled_body_id": s["body_id"]})
    pairs = _draw(pstrata, {"pair": cfg["pair_sample"]["per_stratum"]}, set(),
                  lambda c: c["styled_body_id"], _rng(seed, "pair-sample"))

    # -- scenario sample: at most one scenario per decision, 8 per domain -------------
    srng = _rng(seed, "scenario-sample")
    scenario_units = [u for u in units if u["level"] == "scenario"]
    domain_of = {s["decision_id"]: s["domain"] for s in built.stimuli}
    scenarios = []
    for domain in sorted(set(domain_of.values())):
        decisions = sorted(d for d, dom in domain_of.items() if dom == domain)
        srng.shuffle(decisions)
        for decision in sorted(decisions[: cfg["scenario_sample"]["per_domain"]]):
            options = sorted((u for u in scenario_units if u["decision_id"] == decision),
                             key=lambda u: u["unit_id"])
            scenarios.append(srng.choice(options))

    # -- blind ids, labels, orders ---------------------------------------------------
    def keyed(rows, prefix, salt, kind):
        rng = _rng(seed, f"{salt}-labels")
        out = []
        for n, u in enumerate(sorted(rows, key=lambda r: r["unit_id"]), start=1):
            entry = {"blind_id": f"{prefix}{n:04d}", "unit_id": u["unit_id"],
                     "decision_id": u["decision_id"]}
            labels = ["opt_1", "opt_2"]
            rng.shuffle(labels)
            if kind == "pair":
                sides = [u["styled_stimulus_id"], u["plain_stimulus_id"]]
                rng.shuffle(sides)
                entry["sides"] = {"1": sides[0], "2": sides[1]}
            else:
                entry["option_labels"] = {"P": labels[0], "Q": labels[1]}
            out.append(entry)
        return out

    sample: dict[str, Any] = {}
    for name, rows, prefix, kind in (("stimulus", items, "v3i", "stimulus"),
                                     ("pair", pairs, "v3p", "pair"),
                                     ("scenario", scenarios, "v3s", "scenario")):
        members = keyed(rows, prefix, name, kind)
        decision_of = {m["blind_id"]: m["decision_id"] for m in members}
        orders, separation = {}, {}
        for annotator in ANNOTATORS:
            order, result = order_with_separation(
                [m["blind_id"] for m in members], lambda b: decision_of[b], minimum,
                _rng(seed, f"{name}-order-{annotator}"))
            achieved = _all_pairs_min(order, decision_of)
            orders[annotator] = order
            separation[annotator] = {
                "requested": minimum, "achieved_all_pairs": achieved,
                "satisfied": result.satisfied and (achieved is None or achieved >= minimum),
                "status": result.status, "method": result.method,
                "sibling_pairs": result.n_sibling_pairs,
                "distance": (f"|i - j| >= {minimum}: {minimum} positions apart, meaning at "
                             f"least {minimum - 1} intervening items")}
        sample[name] = {"size": len(members), "members": members, "orders": orders,
                        "separation": separation}

    return {
        "dataset_version": spec.version,
        "status": "proposed_not_approved",
        "annotation_begun": False,
        "reuses_v2_pin": False,
        "seed": seed,
        "independent_annotators": list(ANNOTATORS),
        "sibling_key": cfg["sibling_key"],
        "sibling_rationale": cfg["sibling_rationale"],
        "design": {k: cfg[k] for k in ("stimulus_sample", "pair_sample", "scenario_sample")},
        "samples": sample,
        "coverage": coverage(built, sample),
    }


def coverage(built: Build, sample: dict[str, Any]) -> dict[str, Any]:
    stim = {s["stimulus_id"]: s for s in built.stimuli}
    rows = [stim[m["unit_id"]] for m in sample["stimulus"]["members"]]
    out = {"stimulus": {k: dict(sorted(Counter(str(r[k]) for r in rows).items()))
                        for k in ("domain", "condition", "marker_family", "marker_id",
                                  "opening_id", "supported_option")}}
    out["stimulus"]["decisions_covered"] = len({r["decision_id"] for r in rows})
    out["stimulus"]["max_per_decision"] = max(Counter(r["decision_id"] for r in rows).values())
    out["stimulus"]["distinct_bodies"] = len({r["body_id"] for r in rows})
    prow = [stim[m["unit_id"].split("~")[0]] for m in sample["pair"]["members"]]
    out["pair"] = {k: dict(sorted(Counter(str(r[k]) for r in prow).items()))
                   for k in ("domain", "condition", "marker_id", "opening_id")}
    out["pair"]["distinct_styled_bodies"] = len({r["body_id"] for r in prow})
    out["scenario"] = {"decisions_covered": len({m["decision_id"] for m in sample["scenario"]["members"]}),
                       "by_domain": dict(sorted(Counter(stim_dom for stim_dom in (
                           next(s["domain"] for s in built.stimuli if s["decision_id"] == m["decision_id"])
                           for m in sample["scenario"]["members"])).items()))}
    return out
