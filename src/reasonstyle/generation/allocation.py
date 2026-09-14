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
    "allocate_markers",
    "allocation_problems",
    "load_allocation",
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
    registry = cfg.raw["markers"]["realization"]["registry"]
    found = sorted(rid for rid, spec in registry.items() if spec["family"] == family)
    if len(found) != 2:
        raise AllocationError(
            f"{family}: the scheme pairs two realizations per family; found {found}")
    return found


def allocate_markers(bank: TopicBank, cfg: ExperimentConfig) -> MarkerAllocation:
    """Build the pilot allocation. Pure: same inputs, same output."""
    alloc_cfg = cfg.raw["markers"]["allocation"]
    families: list[str] = list(alloc_cfg["pilot_families"])
    if len(families) != 3:
        raise AllocationError("the slot scheme pairs families two at a time across three")
    strings = {f: list(alloc_cfg["pilot_strings"][f]) for f in families}
    realizations = {f: _realizations(cfg, f) for f in families}

    curated = _curated(bank, cfg)
    domains = list(cfg.raw["domains"]["ids"])
    variants = [1, 2]

    seed = cfg.raw["determinism"]["seeds"]["marker_family_assignment"]
    rng = random.Random(seed)

    # -- 1. two different families per decision -----------------------------
    slot_family: dict[tuple[str, int], str] = {}
    for d_index, domain in enumerate(domains):
        in_domain = [did for did, dom in curated if dom == domain]
        rng.shuffle(in_domain)
        rotated = [families[(i + d_index) % 3] for i in range(3)]
        pairs = [(rotated[0], rotated[1]), (rotated[1], rotated[2]),
                 (rotated[2], rotated[0]), (rotated[0], rotated[1])]
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
        if len({g.marker_realization_id for g in pair}) != 2:
            problems.append(f"{slot}: the two directions take different realizations")

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
