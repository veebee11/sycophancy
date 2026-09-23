"""Deterministic marker allocation for the pilot.

The unit is the **group** — ``(scenario_id, supported_option)`` — because
``marker_family``, ``marker_string`` and ``marker_realization_id`` are
group-level fields: RP and NP carry no marker of their own.

The scheme, fixed before any drafting call:

1. Each decision's **two variants take two different families**, so no family
   is confounded with a particular decision. Twenty-four (decision, variant)
   slots, eight per confirmatory family.
2. Within a slot, **both directions share the family and the marker string**
   and take the **two different realizations**, one each.
3. Which direction gets which realization alternates, so within a family each
   realization appears eight times, split evenly across ``opt_1`` and ``opt_2``.
4. A family's eight slots split four/four between its two pilot strings, so
   each string covers four distinct decisions — the configured minimum.

Determinism comes from ``determinism.seeds.marker_family_assignment`` and the
sorted decision ids. It is not alphabetical: decisions are shuffled within
their domain by the seeded generator, so the first decision of a domain has no
privileged marker.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import ExperimentConfig
from ..corpus.schemas import SEMANTIC_OPTIONS, SemanticOption
from ..corpus.topics import TopicBank
from ..hashing import content_hash

__all__ = [
    "AllocationError",
    "GroupAllocation",
    "MarkerAllocation",
    "SeededAllocation",
    "allocate_full_markers",
    "allocate_markers",
    "allocation_problems",
    "balance_table",
    "full_allocation_problems",
    "load_allocation",
    "render_full_allocation",
    "save_allocation",
]

_HEADER = """\
# Marker allocation for the pilot. GENERATED — do not edit by hand.
#
# Rebuild with:
#   uv run python scripts/allocate_markers.py --config configs/experiment.yaml \\
#       --topics data/topics/pilot_topics.yaml --out data/pilot/marker_allocation.yaml
#
# Fixed before drafting: the drafting scripts read this file and never choose a
# marker themselves. Editing it by hand changes its content hash, which the
# generation log records with every call.
"""


class AllocationError(ValueError):
    """The allocation could not be built, or the file is malformed."""


@dataclass(frozen=True, slots=True)
class GroupAllocation:
    decision_id: str
    domain: str
    variant_id: int
    scenario_id: str
    supported_option: SemanticOption
    marker_family: str
    marker_string: str
    marker_realization_id: str

    @property
    def slot(self) -> tuple[str, int]:
        return (self.decision_id, self.variant_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "domain": self.domain,
            "variant_id": self.variant_id,
            "scenario_id": self.scenario_id,
            "supported_option": self.supported_option,
            "marker_family": self.marker_family,
            "marker_string": self.marker_string,
            "marker_realization_id": self.marker_realization_id,
        }


@dataclass(frozen=True, slots=True)
class MarkerAllocation:
    groups: tuple[GroupAllocation, ...]
    seed: int
    config_content_hash: str
    topic_bank_content_hash: str

    @property
    def content_hash(self) -> str:
        """Order-independent hash of the allocation itself, excluding the
        hashes of the inputs, so the same allocation is recognisable across
        unrelated config edits."""
        return content_hash([g.as_dict() for g in self.groups])

    def for_group(self, scenario_id: str, option: str) -> GroupAllocation:
        for g in self.groups:
            if g.scenario_id == scenario_id and g.supported_option == option:
                return g
        raise AllocationError(f"no allocation for ({scenario_id}, {option})")

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "config_content_hash": self.config_content_hash,
            "topic_bank_content_hash": self.topic_bank_content_hash,
            "allocation_content_hash": self.content_hash,
            "groups": [g.as_dict() for g in self.groups],
        }


def _curated(bank: TopicBank, cfg: ExperimentConfig) -> list[tuple[str, str]]:
    """``(decision_id, domain)`` for the curated decisions, sorted."""
    curated = sorted((t.decision_id, t.domain) for t in bank.topics if t.status == "curated")
    expected = cfg.raw["corpus"]["decisions_pilot"]
    if len(curated) != expected:
        raise AllocationError(
            f"the pilot allocation needs exactly {expected} curated decisions; got {len(curated)}")
    per_domain = Counter(domain for _, domain in curated)
    want = cfg.raw["domains"]["decisions_per_domain_pilot"]
    off = {d: n for d, n in per_domain.items() if n != want}
    if off or set(per_domain) != set(cfg.raw["domains"]["ids"]):
        raise AllocationError(
            f"the pilot must be {want} curated decisions per domain; got {dict(per_domain)}")
    return curated


def _realizations(cfg: ExperimentConfig, family: str) -> list[str]:
    """The realizations available to a family: two to alternate, or one shared.

    With two, the scheme gives the two directions of a slot one each, so
    realization is balanced across directions. With one — which is what a design
    restricted to sentence-initial markers has, because a semicolon-medial
    marker needs a clause before the semicolon and a no-premise cell has none —
    both directions take it, and realization simply stops varying.
    """
    registry = cfg.raw["markers"]["realization"]["registry"]
    found = sorted(rid for rid, spec in registry.items() if spec["family"] == family)
    if len(found) not in (1, 2):
        raise AllocationError(
            f"{family}: the scheme uses one or two realizations per family; found {found}")
    return found * 2 if len(found) == 1 else found


def allocate_markers(bank: TopicBank, cfg: ExperimentConfig) -> MarkerAllocation:
    """Build the pilot allocation. Pure: same inputs, same output."""
    alloc_cfg = cfg.raw["markers"]["allocation"]
    families: list[str] = list(alloc_cfg["pilot_families"])
    if len(families) < 2:
        raise AllocationError(
            "the slot scheme gives a decision's two variants different families, so it "
            f"needs at least two; got {families}")
    strings = {f: list(alloc_cfg["pilot_strings"][f]) for f in families}
    realizations = {f: _realizations(cfg, f) for f in families}

    curated = _curated(bank, cfg)
    domains = list(cfg.raw["domains"]["ids"])
    variants = [1, 2]

    seed = cfg.raw["determinism"]["seeds"]["marker_family_assignment"]
    rng = random.Random(seed)

    # -- 1. two different families per decision -----------------------------
    slot_family: dict[tuple[str, int], str] = {}
    n_families = len(families)
    for d_index, domain in enumerate(domains):
        in_domain = [did for did, dom in curated if dom == domain]
        rng.shuffle(in_domain)
        rotated = [families[(i + d_index) % n_families] for i in range(n_families)]
        # Consecutive cyclic pairs, repeated to cover the decisions in this
        # domain. With three families this is exactly the sequence the pilot was
        # built with. With two, the cyclic list would itself alternate the lead
        # and cancel the alternation applied below, leaving every v1 on one
        # family and every v2 on the other — family perfectly confounded with
        # variant. So with two the pair is constant and the swap below does the
        # alternating, which is what it is there for.
        if n_families == 2:
            pairs = [(rotated[0], rotated[1])] * len(in_domain)
        else:
            cyclic = [(rotated[i], rotated[(i + 1) % n_families])
                      for i in range(n_families)]
            pairs = [cyclic[i % n_families] for i in range(len(in_domain))]
        for position, decision_id in enumerate(in_domain):
            first, second = pairs[position % len(pairs)]
            if (position + d_index) % 2:          # which variant leads alternates
                first, second = second, first
            slot_family[(decision_id, variants[0])] = first
            slot_family[(decision_id, variants[1])] = second

    domain_of = dict(curated)

    # -- 2. strings: four slots each, spread across domains ------------------
    slot_string: dict[tuple[str, int], str] = {}
    for family in families:
        slots = sorted(s for s, f in slot_family.items() if f == family)
        by_domain: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
        for slot in slots:
            by_domain[domain_of[slot[0]]].append(slot)
        turn = 0
        for domain in domains:                     # a running turn across domains
            here = by_domain[domain]
            rng.shuffle(here)
            for slot in here:
                slot_string[slot] = strings[family][turn % 2]
                turn += 1

    # -- 3. realizations alternate across the directions ---------------------
    groups: list[GroupAllocation] = []
    for family in families:
        for index, slot in enumerate(sorted(s for s, f in slot_family.items() if f == family)):
            decision_id, variant_id = slot
            first, second = realizations[family]
            if index % 2:
                first, second = second, first
            for option, realization in zip(SEMANTIC_OPTIONS, (first, second), strict=True):
                groups.append(GroupAllocation(
                    decision_id=decision_id,
                    domain=domain_of[decision_id],
                    variant_id=variant_id,
                    scenario_id=f"{decision_id}_v{variant_id}",
                    supported_option=option,
                    marker_family=family,
                    marker_string=slot_string[slot],
                    marker_realization_id=realization,
                ))

    groups.sort(key=lambda g: (g.decision_id, g.variant_id, g.supported_option))
    return MarkerAllocation(
        groups=tuple(groups),
        seed=seed,
        config_content_hash=cfg.content_hash,
        topic_bank_content_hash=content_hash(bank.model_dump(mode="json")),
    )


def allocation_problems(alloc: MarkerAllocation, cfg: ExperimentConfig) -> list[str]:
    """Every rule the allocation must satisfy, checked on the built result.

    Written as an independent check rather than trusting the construction: a
    future change to the scheme must still satisfy the same constraints.
    """
    rules = cfg.raw["markers"]["allocation"]
    problems: list[str] = []

    expected_groups = (cfg.raw["corpus"]["decisions_pilot"]
                       * cfg.raw["corpus"]["variants_per_decision"]
                       * len(cfg.raw["corpus"]["supported_options"]))
    if len(alloc.groups) != expected_groups:
        problems.append(f"expected {expected_groups} groups; got {len(alloc.groups)}")

    by_slot: defaultdict[tuple[str, int], list[GroupAllocation]] = defaultdict(list)
    for g in alloc.groups:
        by_slot[g.slot].append(g)

    for slot, pair in sorted(by_slot.items()):
        if {g.supported_option for g in pair} != set(SEMANTIC_OPTIONS):
            problems.append(f"{slot}: both semantic options must be allocated")
            continue
        if len({g.marker_family for g in pair}) != 1:
            problems.append(f"{slot}: both directions share the family")
        if len({g.marker_string for g in pair}) != 1:
            problems.append(f"{slot}: both directions share the marker string")
        # Two realizations per family means the directions take one each, so
        # realization is balanced across directions. A family with only one
        # available — a design restricted to sentence-initial markers, where a
        # semicolon-medial one cannot realize a no-premise cell — has nothing to
        # alternate, and the two directions share it. The rule is that the pair
        # uses every realization the family has, not that it uses two.
        available = len({rid for rid, spec
                         in cfg.raw["markers"]["realization"]["registry"].items()
                         if spec["family"] == pair[0].marker_family})
        used = len({g.marker_realization_id for g in pair})
        if used != min(available, 2):
            problems.append(
                f"{slot}: the two directions must use all {min(available, 2)} realization(s) "
                f"available to {pair[0].marker_family}; they use {used}")

    by_decision: defaultdict[str, set[str]] = defaultdict(set)
    for slot, pair in by_slot.items():
        by_decision[slot[0]].add(pair[0].marker_family)
    for decision_id, fams in sorted(by_decision.items()):
        if len(fams) != 2:
            problems.append(f"{decision_id}: its two variants must take different families")

    allowed = set(rules["pilot_families"])
    used = {g.marker_family for g in alloc.groups}
    if used - allowed:
        problems.append(f"families outside the pilot scope: {sorted(used - allowed)}")
    per_family = Counter(g.marker_family for g in alloc.groups)
    if len(set(per_family.values())) > 1:
        problems.append(f"families are unevenly allocated: {dict(per_family)}")

    registry = cfg.raw["markers"]["realization"]["registry"]
    for g in alloc.groups:
        if registry.get(g.marker_realization_id, {}).get("family") != g.marker_family:
            problems.append(f"{g.scenario_id}/{g.supported_option}: realization "
                            f"{g.marker_realization_id!r} is not a {g.marker_family} realization")
        if g.marker_string not in rules["pilot_strings"][g.marker_family]:
            problems.append(f"{g.scenario_id}/{g.supported_option}: {g.marker_string!r} "
                            f"is not a pilot string for {g.marker_family}")

    for family in sorted(used):
        counts = Counter(g.marker_realization_id for g in alloc.groups
                         if g.marker_family == family)
        if len(set(counts.values())) > 1:
            problems.append(f"{family}: realizations are unevenly allocated: {dict(counts)}")
        for realization, by_option in _by(alloc.groups, "marker_realization_id",
                                          "supported_option", family=family).items():
            if len(set(by_option.values())) > 1:
                problems.append(f"{realization}: unevenly split across the directions: "
                                f"{dict(by_option)}")

    # -- the same caps the corpus validator applies to the full corpus -------
    decisions_per_marker: defaultdict[str, set[str]] = defaultdict(set)
    domains_per_marker: defaultdict[str, Counter] = defaultdict(Counter)
    options_per_marker: defaultdict[str, Counter] = defaultdict(Counter)
    for g in alloc.groups:
        decisions_per_marker[g.marker_string].add(g.decision_id)
        domains_per_marker[g.marker_string][g.domain] += 1
        options_per_marker[g.marker_string][g.supported_option] += 1

    for marker, decisions in sorted(decisions_per_marker.items()):
        if len(decisions) < rules["min_decisions_per_marker"]:
            problems.append(f"marker {marker!r} appears in {len(decisions)} decision(s); "
                            f"at least {rules['min_decisions_per_marker']} are required")
    for marker, counts in sorted(domains_per_marker.items()):
        share = max(counts.values()) / sum(counts.values())
        if share > rules["max_share_within_one_domain"]:
            problems.append(f"marker {marker!r} has {share:.0%} of its uses in one domain; "
                            f"the cap is {rules['max_share_within_one_domain']:.0%}")
    for marker, counts in sorted(options_per_marker.items()):
        share = max(counts.values()) / sum(counts.values())
        if share > rules["max_share_within_one_supported_option"]:
            problems.append(f"marker {marker!r} has {share:.0%} of its uses on one semantic "
                            f"option; the cap is "
                            f"{rules['max_share_within_one_supported_option']:.0%}")
    return problems


def _by(groups, key: str, sub: str, *, family: str) -> dict[str, Counter]:
    out: defaultdict[str, Counter] = defaultdict(Counter)
    for g in groups:
        if g.marker_family == family:
            out[getattr(g, key)][getattr(g, sub)] += 1
    return dict(out)


def save_allocation(alloc: MarkerAllocation, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(alloc.as_dict(), sort_keys=False, allow_unicode=True, width=100)
    path.write_text(_HEADER + body, encoding="utf-8")


def load_allocation(path: str | Path) -> MarkerAllocation:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AllocationError(f"{path}: {exc}") from exc
    if not isinstance(raw, dict) or "groups" not in raw:
        raise AllocationError(f"{path}: not a marker allocation file")
    groups = tuple(GroupAllocation(**g) for g in raw["groups"])
    alloc = MarkerAllocation(groups=groups, seed=raw["seed"],
                             config_content_hash=raw["config_content_hash"],
                             topic_bank_content_hash=raw["topic_bank_content_hash"])
    if alloc.content_hash != raw["allocation_content_hash"]:
        raise AllocationError(
            f"{path}: the file was edited after it was generated "
            f"(recorded {raw['allocation_content_hash'][:12]}, "
            f"actual {alloc.content_hash[:12]})")
    return alloc


# --------------------------------------------------------------------------
# The full corpus: 60 decisions grown from the frozen v2 pilot
# --------------------------------------------------------------------------
#
# The pilot's 48 groups are already drafted, corrected and assembled under
# their allocation, so their assignments are fixed facts, not choices. The full
# allocation imports them exactly and allocates only the 192 new groups, so
# that the WHOLE corpus is balanced: 60 groups per marker string, 20 per
# domain, 30 per supported option and 30 per variant; 120 per family, 40 per
# domain and 60 per option and per variant. The pilot allocation above is not
# touched: this is a separate mode, and the pilot files still rebuild byte for
# byte.

_FULL_HEADER = """\
# Marker allocation for the full v2 corpus. GENERATED — do not edit by hand.
#
# Rebuild with:
#   uv run python scripts/allocate_markers.py --config configs/experiment_v2_full.draft.yaml \\
#       --topics data/topics/full_topics_v2.yaml \\
#       --seed-allocation data/pilot/marker_allocation_v2.yaml \\
#       --out data/full/marker_allocation_full_v2.yaml
#   ... --check   rebuild and fail unless this file is byte-identical
#
# The 48 groups named under seed_allocation are the frozen v2 pilot's own
# assignments, imported exactly; the 192 under new_allocation were allocated
# around them. Fixed before drafting: the drafting scripts read this file and
# never choose a marker themselves.
"""


@dataclass(frozen=True, slots=True)
class SeededAllocation:
    """A full allocation and where each of its rows came from."""

    allocation: MarkerAllocation
    seed_allocation: MarkerAllocation
    seed_allocation_path: str

    @property
    def seed_keys(self) -> set[tuple[str, str]]:
        return {(g.scenario_id, g.supported_option) for g in self.seed_allocation.groups}

    def as_dict(self) -> dict[str, Any]:
        alloc = self.allocation
        seed_scenarios = sorted({g.scenario_id for g in self.seed_allocation.groups})
        new_scenarios = sorted({g.scenario_id for g in alloc.groups} - set(seed_scenarios))
        return {
            "seed": alloc.seed,
            "config_content_hash": alloc.config_content_hash,
            "topic_bank_content_hash": alloc.topic_bank_content_hash,
            "allocation_content_hash": alloc.content_hash,
            "seed_allocation": {
                "path": self.seed_allocation_path,
                "allocation_content_hash": self.seed_allocation.content_hash,
                "config_content_hash": self.seed_allocation.config_content_hash,
                "topic_bank_content_hash": self.seed_allocation.topic_bank_content_hash,
                "groups": len(self.seed_allocation.groups),
                "scenario_ids": seed_scenarios,
            },
            "new_allocation": {
                "groups": len(alloc.groups) - len(self.seed_allocation.groups),
                "scenario_ids": new_scenarios,
            },
            "groups": [g.as_dict() for g in alloc.groups],
        }


def _full_curated(bank: TopicBank, cfg: ExperimentConfig) -> dict[str, str]:
    curated = {t.decision_id: t.domain for t in bank.topics if t.status == "curated"}
    expected = cfg.raw["corpus"]["decisions_full"]
    if len(curated) != expected:
        raise AllocationError(
            f"the full allocation needs exactly {expected} curated decisions; got {len(curated)}")
    per_domain = Counter(curated.values())
    want = cfg.raw["domains"]["decisions_per_domain_full"]
    if set(per_domain) != set(cfg.raw["domains"]["ids"]) or set(per_domain.values()) != {want}:
        raise AllocationError(f"the full corpus must be {want} curated decisions per domain; "
                              f"got {dict(per_domain)}")
    return curated


def _split(slots_by_domain, need_first_by_domain, need_first_v1, *, family, first):
    """How many of each domain's variant-1 slots take the first string.

    Exact constraints: domain d gives ``need_first_by_domain[d]`` of its slots
    to the first string, and across domains ``need_first_v1`` of the first
    string's slots are variant 1. Every feasible split is enumerated and the
    most proportional one kept, ties broken by the sorted domain order, so the
    result is deterministic and never a matter of search order.
    """
    domains = sorted(slots_by_domain)
    ranges = []
    for d in domains:
        a = sum(1 for _, v in slots_by_domain[d] if v == 1)
        b = len(slots_by_domain[d]) - a
        need = need_first_by_domain[d]
        lo, hi = max(0, need - b), min(a, need)
        if lo > hi:
            raise AllocationError(
                f"{family}/{first}: domain {d} needs {need} new slots of {first} but has "
                f"{a} variant-1 and {b} variant-2 new slots; no split exists")
        ideal = need * a / (a + b) if a + b else 0
        ranges.append((d, lo, hi, ideal))
    best = None
    def walk(i, chosen, total):
        nonlocal best
        if i == len(ranges):
            if total != need_first_v1:
                return
            cost = sum((x - r[3]) ** 2 for x, r in zip(chosen, ranges))
            key = (round(cost, 9), tuple(chosen))
            if best is None or key < best[0]:
                best = (key, dict(zip(domains, chosen)))
            return
        _, lo, hi, _ = ranges[i]
        for x in range(lo, hi + 1):
            walk(i + 1, chosen + [x], total + x)
    walk(0, [], 0)
    if best is None:
        raise AllocationError(
            f"{family}/{first}: no split gives {first} exactly {need_first_v1} new variant-1 "
            f"slots while meeting every domain's count — the frozen pilot assignments and the "
            f"required balance cannot coexist")
    return best[1]


def allocate_full_markers(bank: TopicBank, cfg: ExperimentConfig, seed: MarkerAllocation,
                          *, seed_allocation_path: str) -> SeededAllocation:
    """The full allocation: the seed's rows exactly, and the rest allocated so the
    whole corpus meets every balance. Pure: same inputs, same output."""
    alloc_cfg = cfg.raw["markers"]["allocation"]
    families: list[str] = list(alloc_cfg["pilot_families"])
    if len(families) != 2:
        raise AllocationError(f"the full scheme balances exactly two families; got {families}")
    strings = {f: list(alloc_cfg["pilot_strings"][f]) for f in families}
    if any(len(s) != 2 for s in strings.values()):
        raise AllocationError(f"the full scheme uses two strings per family; got {strings}")
    realizations = {f: _realizations(cfg, f) for f in families}
    curated = _full_curated(bank, cfg)
    domains = list(cfg.raw["domains"]["ids"])
    per_domain = cfg.raw["domains"]["decisions_per_domain_full"]
    n_decisions = len(curated)

    seed_slots: dict[tuple[str, int], GroupAllocation] = {}
    for g in seed.groups:
        if curated.get(g.decision_id) != g.domain:
            raise AllocationError(f"seed row {g.scenario_id}/{g.supported_option} is not a "
                                  f"curated decision of domain {g.domain} in the full bank")
        seed_slots.setdefault(g.slot, g)
    seed_decisions = {d for d, _ in seed_slots}
    for decision_id in seed_decisions:
        if {(decision_id, 1), (decision_id, 2)} - set(seed_slots):
            raise AllocationError(f"the seed allocation does not cover both variants of "
                                  f"{decision_id}")
    new = {d: dom for d, dom in curated.items() if d not in seed_decisions}

    seed_value = cfg.raw["determinism"]["seeds"]["marker_family_assignment"]
    rng = random.Random(seed_value)

    # -- 1. which variant leads with the first family, per domain -------------
    # Each decision's two variants take different families, so a family's slots
    # are one per decision; balance by variant means half the decisions lead
    # with each family. Done per domain, so every domain is balanced too.
    lead = families[0]
    slot_family: dict[tuple[str, int], str] = {s: g.marker_family for s, g in seed_slots.items()}
    for domain in domains:
        seeded = sum(1 for (d, v), f in slot_family.items()
                     if v == 1 and f == lead and curated[d] == domain)
        in_domain = sorted(d for d, dom in new.items() if dom == domain)
        k = per_domain // 2 - seeded
        if not 0 <= k <= len(in_domain):
            raise AllocationError(
                f"{domain}: {seeded} seed decisions already lead with {lead}; the "
                f"{len(in_domain)} new ones cannot bring it to {per_domain // 2}")
        rng.shuffle(in_domain)
        for i, decision_id in enumerate(in_domain):
            first, second = (families[0], families[1]) if i < k else (families[1], families[0])
            slot_family[(decision_id, 1)] = first
            slot_family[(decision_id, 2)] = second

    # -- 2. strings within each family ----------------------------------------
    slot_string: dict[tuple[str, int], str] = {s: g.marker_string for s, g in seed_slots.items()}
    per_string_domain = per_domain // 2          # 10: a family has 20 slots per domain
    per_string_variant = n_decisions // 4        # 15: a string has 30 slots, half per variant
    for family in families:
        first, second = strings[family]
        new_by_domain = {d: sorted(s for s, f in slot_family.items()
                                   if f == family and s not in seed_slots and curated[s[0]] == d)
                         for d in domains}
        need_by_domain = {d: per_string_domain - sum(
            1 for s, g in seed_slots.items()
            if g.marker_string == first and curated[s[0]] == d) for d in domains}
        need_v1 = per_string_variant - sum(1 for s, g in seed_slots.items()
                                           if g.marker_string == first and s[1] == 1)
        split = _split(new_by_domain, need_by_domain, need_v1, family=family, first=first)
        for domain in domains:
            for variant, take in ((1, split[domain]),
                                  (2, need_by_domain[domain] - split[domain])):
                here = [s for s in new_by_domain[domain] if s[1] == variant]
                rng.shuffle(here)
                for i, slot in enumerate(here):
                    slot_string[slot] = first if i < take else second

    # -- 3. rows: the seed's exactly, the new ones built -----------------------
    groups: list[GroupAllocation] = list(seed.groups)
    for family in families:
        new_slots = sorted(s for s, f in slot_family.items() if f == family and s not in seed_slots)
        for index, slot in enumerate(new_slots):
            decision_id, variant_id = slot
            one, two = realizations[family]
            if index % 2:
                one, two = two, one
            for option, realization in zip(SEMANTIC_OPTIONS, (one, two), strict=True):
                groups.append(GroupAllocation(
                    decision_id=decision_id, domain=curated[decision_id],
                    variant_id=variant_id, scenario_id=f"{decision_id}_v{variant_id}",
                    supported_option=option, marker_family=family,
                    marker_string=slot_string[slot], marker_realization_id=realization))
    groups.sort(key=lambda g: (g.decision_id, g.variant_id, g.supported_option))
    return SeededAllocation(
        allocation=MarkerAllocation(
            groups=tuple(groups), seed=seed_value, config_content_hash=cfg.content_hash,
            topic_bank_content_hash=content_hash(bank.model_dump(mode="json"))),
        seed_allocation=seed, seed_allocation_path=seed_allocation_path)


def balance_table(groups) -> dict[str, dict[str, dict[str, int]]]:
    """``{marker_string | marker_family: {value: {dimension: count}}}``."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for level in ("marker_string", "marker_family"):
        table: dict[str, dict[str, int]] = {}
        for g in groups:
            row = table.setdefault(getattr(g, level), Counter())
            row["groups"] += 1
            row[g.domain] += 1
            row[g.supported_option] += 1
            row[f"v{g.variant_id}"] += 1
        out[level] = {k: dict(v) for k, v in sorted(table.items())}
    return out


def full_allocation_problems(seeded: SeededAllocation, bank: TopicBank,
                             cfg: ExperimentConfig) -> list[str]:
    """Every rule the full allocation must meet, checked on the result alone."""
    alloc, seed = seeded.allocation, seeded.seed_allocation
    rules = cfg.raw["markers"]["allocation"]
    target = rules.get("full_target") or {}
    problems: list[str] = []
    curated = {t.decision_id: t.domain for t in bank.topics if t.status == "curated"}
    domains = list(cfg.raw["domains"]["ids"])
    variants = range(1, cfg.raw["corpus"]["variants_per_decision"] + 1)

    expected = {(f"{d}_v{v}", o) for d in curated for v in variants for o in SEMANTIC_OPTIONS}
    keys = [(g.scenario_id, g.supported_option) for g in alloc.groups]
    if len(keys) != len(set(keys)):
        problems.append("a group is allocated more than once")
    if set(keys) != expected:
        problems.append(f"{len(set(keys))} groups allocated; exactly {len(expected)} are "
                        f"required (every curated decision, both variants, both options)")
    if target.get("groups_total") and len(alloc.groups) != target["groups_total"]:
        problems.append(f"{len(alloc.groups)} groups; the target is {target['groups_total']}")
    for g in alloc.groups:
        if curated.get(g.decision_id) != g.domain or \
                g.scenario_id != f"{g.decision_id}_v{g.variant_id}":
            problems.append(f"{g.scenario_id}/{g.supported_option}: identity fields disagree "
                            f"with the topic bank")

    # -- the seed, exactly ------------------------------------------------------
    rows = {(g.scenario_id, g.supported_option): g for g in alloc.groups}
    for g in seed.groups:
        if rows.get((g.scenario_id, g.supported_option)) != g:
            problems.append(f"seed row {g.scenario_id}/{g.supported_option} was not "
                            f"preserved exactly")
    if target.get("pilot_assignments_fixed") and \
            len(seed.groups) != target["pilot_assignments_fixed"]:
        problems.append(f"{len(seed.groups)} seed rows; the target fixes "
                        f"{target['pilot_assignments_fixed']}")

    # -- scenario pairs and decision variants ---------------------------------
    by_slot: dict[tuple[str, int], list[GroupAllocation]] = {}
    for g in alloc.groups:
        by_slot.setdefault(g.slot, []).append(g)
    for slot, pair in sorted(by_slot.items()):
        if len({(g.marker_family, g.marker_string, g.marker_realization_id) for g in pair}) != 1:
            problems.append(f"{slot}: both supported options must share family, string and "
                            f"realization")
    fams_by_decision: dict[str, set[str]] = {}
    for (decision_id, _), pair in by_slot.items():
        fams_by_decision.setdefault(decision_id, set()).add(pair[0].marker_family)
    for decision_id, fams in sorted(fams_by_decision.items()):
        if len(fams) != 2:
            problems.append(f"{decision_id}: its two variants must take different families")

    # -- only the permitted families, strings and realizations ----------------
    selectable = set(rules.get("selectable_families") or rules["pilot_families"])
    refused = set(rules.get("refused_families") or ())
    registry = cfg.raw["markers"]["realization"]["registry"]
    for g in alloc.groups:
        where = f"{g.scenario_id}/{g.supported_option}"
        if g.marker_family in refused or g.marker_family not in selectable:
            problems.append(f"{where}: family {g.marker_family!r} may not be allocated")
        if g.marker_string not in rules["pilot_strings"].get(g.marker_family, ()):
            problems.append(f"{where}: {g.marker_string!r} is not a string of {g.marker_family}")
        spec = registry.get(g.marker_realization_id) or {}
        if spec.get("family") != g.marker_family or spec.get("position") != "sentence_initial":
            problems.append(f"{where}: {g.marker_realization_id!r} is not a sentence-initial "
                            f"{g.marker_family} realization")

    # -- whole-corpus balance -----------------------------------------------------
    n = len(curated)
    table = balance_table(alloc.groups)
    markers = list(target.get("markers") or [s for f in rules["pilot_strings"].values() for s in f])
    per_marker = target.get("groups_per_marker") or len(expected) // len(markers)
    want_marker = {"groups": per_marker, **{d: per_marker // len(domains) for d in domains},
                   **{o: per_marker // 2 for o in SEMANTIC_OPTIONS}, "v1": per_marker // 2,
                   "v2": per_marker // 2}
    want_family = {"groups": 2 * n, **{d: 2 * n // len(domains) for d in domains},
                   **{o: n for o in SEMANTIC_OPTIONS}, "v1": n, "v2": n}
    for level, wanted_keys, want in (("marker_string", markers, want_marker),
                                     ("marker_family", sorted(selectable), want_family)):
        got_keys = sorted(table[level])
        if got_keys != sorted(wanted_keys):
            problems.append(f"{level}: allocated {got_keys}; expected {sorted(wanted_keys)}")
        for key in wanted_keys:
            row = table[level].get(key, {})
            for dim, value in want.items():
                if row.get(dim, 0) != value:
                    problems.append(f"{level} {key!r}: {dim} = {row.get(dim, 0)}, "
                                    f"required {value}")
    return problems


def render_full_allocation(seeded: SeededAllocation) -> str:
    """The exact bytes of the full allocation file. A rebuild is compared
    against these, so any hand edit — rows or metadata — is detected."""
    body = yaml.safe_dump(seeded.as_dict(), sort_keys=False, allow_unicode=True, width=100)
    return _FULL_HEADER + body
