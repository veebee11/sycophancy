"""Marker allocation: the scheme, its constraints, and its determinism."""

from __future__ import annotations

from collections import Counter

import pytest

from reasonstyle.generation.allocation import (
    AllocationError,
    allocate_markers,
    allocation_problems,
    load_allocation,
    save_allocation,
)


@pytest.fixture
def alloc(cfg, pilot_bank):
    return allocate_markers(pilot_bank, cfg)


def test_every_group_is_allocated_once(alloc, cfg):
    expected = (cfg.raw["corpus"]["decisions_pilot"] * cfg.raw["corpus"]["variants_per_decision"]
                * len(cfg.raw["corpus"]["supported_options"]))
    assert len(alloc.groups) == expected
    keys = [(g.scenario_id, g.supported_option) for g in alloc.groups]
    assert len(set(keys)) == len(keys)


def test_the_allocation_satisfies_its_own_rules(alloc, cfg):
    assert allocation_problems(alloc, cfg) == []


def test_a_decision_covers_two_families(alloc):
    """Otherwise a family would be confounded with particular decisions."""
    by_decision: dict[str, set[str]] = {}
    for g in alloc.groups:
        by_decision.setdefault(g.decision_id, set()).add(g.marker_family)
    assert all(len(families) == 2 for families in by_decision.values())


def test_both_directions_of_a_slot_share_family_and_marker(alloc):
    by_slot: dict[tuple, list] = {}
    for g in alloc.groups:
        by_slot.setdefault(g.slot, []).append(g)
    for pair in by_slot.values():
        assert len({g.marker_family for g in pair}) == 1
        assert len({g.marker_string for g in pair}) == 1
        # ... and take different realizations, so realization is not confounded
        # with the direction being supported.
        assert len({g.marker_realization_id for g in pair}) == 2


def test_realizations_are_balanced_across_directions(alloc):
    for realization in {g.marker_realization_id for g in alloc.groups}:
        by_option = Counter(g.supported_option for g in alloc.groups
                            if g.marker_realization_id == realization)
        assert len(set(by_option.values())) == 1


def test_each_marker_reaches_the_minimum_decisions(alloc, cfg):
    floor = cfg.raw["markers"]["allocation"]["min_decisions_per_marker"]
    for marker in {g.marker_string for g in alloc.groups}:
        assert len({g.decision_id for g in alloc.groups if g.marker_string == marker}) >= floor


def test_no_marker_is_concentrated_in_one_domain_or_direction(alloc, cfg):
    rules = cfg.raw["markers"]["allocation"]
    for marker in {g.marker_string for g in alloc.groups}:
        uses = [g for g in alloc.groups if g.marker_string == marker]
        by_domain = Counter(g.domain for g in uses)
        by_option = Counter(g.supported_option for g in uses)
        assert max(by_domain.values()) / len(uses) <= rules["max_share_within_one_domain"]
        assert (max(by_option.values()) / len(uses)
                <= rules["max_share_within_one_supported_option"])


def test_the_exploratory_family_is_not_in_the_pilot(alloc, cfg):
    exploratory = set(cfg.raw["markers"]["exploratory_families"])
    assert not {g.marker_family for g in alloc.groups} & exploratory


def test_allocation_is_deterministic(cfg, pilot_bank):
    first = allocate_markers(pilot_bank, cfg)
    second = allocate_markers(pilot_bank, cfg)
    assert first.content_hash == second.content_hash
    assert [g.as_dict() for g in first.groups] == [g.as_dict() for g in second.groups]


def test_allocation_is_not_alphabetical(alloc):
    """A seeded shuffle, not decision order: the first decision of a domain
    must not always take the first family."""
    first_of_domain = {}
    for g in sorted(alloc.groups, key=lambda g: (g.domain, g.decision_id, g.variant_id)):
        first_of_domain.setdefault(g.domain, g.marker_family)
    assert len(set(first_of_domain.values())) > 1


def test_a_hand_edited_allocation_is_refused(alloc, tmp_path):
    path = tmp_path / "alloc.yaml"
    save_allocation(alloc, path)
    assert load_allocation(path).content_hash == alloc.content_hash
    path.write_text(path.read_text().replace("because", "therefore", 1), encoding="utf-8")
    with pytest.raises(AllocationError, match="edited"):
        load_allocation(path)


def test_an_uncurated_bank_cannot_be_allocated(cfg, pilot_bank):
    raw = pilot_bank.model_dump(mode="json")
    raw["topics"][0]["status"] = "proposed"
    from reasonstyle.corpus.topics import TopicBank
    with pytest.raises(AllocationError, match="curated decisions"):
        allocate_markers(TopicBank.model_validate(raw), cfg)
