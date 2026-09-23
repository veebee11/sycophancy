"""The full v2 marker allocation: the pilot's 48 rows fixed, 192 allocated around
them, and the whole 240-group corpus balanced exactly."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from reasonstyle.config import load_config
from reasonstyle.corpus.topics import load_topic_bank
from reasonstyle.generation.allocation import (
    AllocationError,
    _split,
    allocate_full_markers,
    allocation_problems,
    balance_table,
    full_allocation_problems,
    load_allocation,
    render_full_allocation,
)

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "configs/frozen/v2_full.yaml"
BANK_PATH = ROOT / "data/topics/full_topics_v2.yaml"
SEED_PATH = "data/pilot/marker_allocation_v2.yaml"
STORED = ROOT / "data/full/marker_allocation_full_v2.yaml"
SCRIPT = ROOT / "scripts/allocate_markers.py"
MARKERS = ("therefore", "consequently", "it follows that", "this implies")
FAMILIES = ("conclusion_indicator", "metadiscursive_inference")


@pytest.fixture(scope="module")
def cfg():
    return load_config(FULL)


@pytest.fixture(scope="module")
def bank():
    return load_topic_bank(BANK_PATH)


@pytest.fixture(scope="module")
def seed():
    return load_allocation(ROOT / SEED_PATH)


@pytest.fixture(scope="module")
def seeded(cfg, bank, seed):
    return allocate_full_markers(bank, cfg, seed, seed_allocation_path=SEED_PATH)


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True,
                          text=True, cwd=ROOT)


def full_args(out):
    return ["--config", FULL, "--topics", BANK_PATH, "--seed-allocation", SEED_PATH, "--out", out]


# --- the pilot rows ------------------------------------------------------------------


def test_all_48_pilot_rows_are_preserved_exactly(seeded, seed):
    rows = {(g.scenario_id, g.supported_option): g for g in seeded.allocation.groups}
    assert len(seed.groups) == 48
    for g in seed.groups:
        assert rows[(g.scenario_id, g.supported_option)].as_dict() == g.as_dict()


def test_the_new_rows_are_exactly_the_48_new_decisions(seeded, seed, bank):
    seed_decisions = {g.decision_id for g in seed.groups}
    new = [g for g in seeded.allocation.groups if g.decision_id not in seed_decisions]
    assert len(new) == 192
    assert len({g.decision_id for g in new}) == 48
    curated = {t.decision_id for t in bank.topics if t.status == "curated"}
    assert {g.decision_id for g in seeded.allocation.groups} == curated


def test_the_file_identifies_imported_and_new_rows(seeded):
    data = seeded.as_dict()
    assert data["seed_allocation"]["groups"] == 48
    assert len(data["seed_allocation"]["scenario_ids"]) == 24
    assert data["new_allocation"]["groups"] == 192
    assert len(data["new_allocation"]["scenario_ids"]) == 96
    assert not set(data["seed_allocation"]["scenario_ids"]) & set(
        data["new_allocation"]["scenario_ids"])
    assert data["seed_allocation"]["allocation_content_hash"] == \
        "973ffb2ea6bfac1b09b00d628fea16c540d3031cd44f29acd0c06094771f02cc"


def test_the_file_records_every_input_hash(seeded, cfg):
    data = seeded.as_dict()
    assert data["config_content_hash"] == cfg.content_hash
    assert data["topic_bank_content_hash"] and data["allocation_content_hash"]
    assert data["seed"] == cfg.raw["determinism"]["seeds"]["marker_family_assignment"]


# --- determinism and the stored file ----------------------------------------------------


def test_the_full_allocation_is_deterministic(cfg, bank, seed, seeded):
    again = allocate_full_markers(bank, cfg, seed, seed_allocation_path=SEED_PATH)
    assert render_full_allocation(again) == render_full_allocation(seeded)


def test_the_stored_full_allocation_is_a_fresh_build(seeded):
    assert STORED.read_text(encoding="utf-8") == render_full_allocation(seeded)
    assert load_allocation(STORED).content_hash == seeded.allocation.content_hash


def test_check_passes_on_the_stored_file():
    result = run(*full_args(STORED), "--check")
    assert result.returncode == 0, result.stderr
    assert "byte for byte" in result.stdout


@pytest.mark.parametrize(("old", "new"), [
    ("marker_string: therefore", "marker_string: consequently"),        # a row
    ("  groups: 48\n", "  groups: 47\n"),                               # provenance
    ("seed: 13\n", "seed: 14\n"),                                       # metadata
])
def test_check_rejects_any_hand_edit(tmp_path, old, new):
    copy = tmp_path / "alloc.yaml"
    text = STORED.read_text(encoding="utf-8")
    assert old in text
    copy.write_text(text.replace(old, new, 1), encoding="utf-8")
    result = run(*full_args(copy), "--check")
    assert result.returncode == 1


def test_a_fresh_rebuild_is_byte_identical(tmp_path):
    out = tmp_path / "full.yaml"
    assert run(*full_args(out)).returncode == 0
    assert out.read_bytes() == STORED.read_bytes()


# --- balance ---------------------------------------------------------------------------


def test_both_independent_rule_checks_pass(seeded, bank, cfg):
    assert full_allocation_problems(seeded, bank, cfg) == []
    assert allocation_problems(seeded.allocation, cfg) == []


def test_every_marker_is_balanced(seeded):
    table = balance_table(seeded.allocation.groups)["marker_string"]
    assert sorted(table) == sorted(MARKERS)
    for marker in MARKERS:
        assert table[marker] == {"groups": 60, "climate": 20, "energy": 20, "technology": 20,
                                 "opt_1": 30, "opt_2": 30, "v1": 30, "v2": 30}, marker


def test_every_family_is_balanced(seeded):
    table = balance_table(seeded.allocation.groups)["marker_family"]
    assert sorted(table) == sorted(FAMILIES)
    for family in FAMILIES:
        assert table[family] == {"groups": 120, "climate": 40, "energy": 40, "technology": 40,
                                 "opt_1": 60, "opt_2": 60, "v1": 60, "v2": 60}, family


def test_each_decisions_variants_take_different_families(seeded):
    fams: dict[str, set[str]] = {}
    for g in seeded.allocation.groups:
        fams.setdefault(g.decision_id, set()).add(g.marker_family)
    assert len(fams) == 60 and all(len(f) == 2 for f in fams.values())


def test_both_options_of_a_scenario_share_family_string_and_realization(seeded):
    by_scenario: dict[str, set] = {}
    for g in seeded.allocation.groups:
        by_scenario.setdefault(g.scenario_id, set()).add(
            (g.marker_family, g.marker_string, g.marker_realization_id))
    assert len(by_scenario) == 120 and all(len(v) == 1 for v in by_scenario.values())


def test_only_the_two_v2_sentence_initial_realizations_are_used(seeded):
    assert Counter(g.marker_realization_id for g in seeded.allocation.groups) == {
        "sentence_initial_conclusion_v1": 120, "sentence_initial_metadiscursive_v1": 120}


def test_no_forbidden_family_or_marker_appears(seeded):
    families = {g.marker_family for g in seeded.allocation.groups}
    strings = {g.marker_string for g in seeded.allocation.groups}
    assert families == set(FAMILIES)
    assert not families & {"premise_indicator", "concession_contrast"}
    assert not strings & {"because", "given that", "considering that", "however", "even so",
                          "nevertheless", "despite this"}


def test_a_forbidden_row_is_reported(seeded, bank, cfg):
    from dataclasses import replace
    from reasonstyle.generation.allocation import MarkerAllocation, SeededAllocation
    groups = list(seeded.allocation.groups)
    groups[-1] = replace(groups[-1], marker_family="premise_indicator", marker_string="because")
    bad = SeededAllocation(allocation=MarkerAllocation(
        groups=tuple(groups), seed=13, config_content_hash="x", topic_bank_content_hash="y"),
        seed_allocation=seeded.seed_allocation, seed_allocation_path=SEED_PATH)
    problems = full_allocation_problems(bad, bank, cfg)
    assert any("may not be allocated" in p for p in problems)
    assert any("marker_string" in p for p in problems)


def test_an_altered_seed_row_is_reported(seeded, bank, cfg):
    from dataclasses import replace
    from reasonstyle.generation.allocation import MarkerAllocation, SeededAllocation
    groups = [replace(g, marker_string="this implies" if g.marker_string == "it follows that"
                      else g.marker_string)
              if g.scenario_id == "climate_01_v1" else g for g in seeded.allocation.groups]
    bad = SeededAllocation(allocation=MarkerAllocation(
        groups=tuple(groups), seed=13, config_content_hash="x", topic_bank_content_hash="y"),
        seed_allocation=seeded.seed_allocation, seed_allocation_path=SEED_PATH)
    assert any("was not preserved exactly" in p for p in full_allocation_problems(bad, bank, cfg))


def test_an_impossible_split_is_refused_not_relaxed():
    slots = {"climate": [("a", 1), ("b", 1)], "energy": [("c", 2)], "technology": [("d", 2)]}
    with pytest.raises(AllocationError, match="cannot coexist"):
        _split(slots, {"climate": 1, "energy": 1, "technology": 0}, 3,
               family="f", first="s")


# --- the pilot allocations are untouched ------------------------------------------------


@pytest.mark.parametrize(("config", "stored"), [
    ("configs/experiment.yaml", "data/pilot/marker_allocation.yaml"),
    ("configs/experiment_v2_pilot.yaml", "data/pilot/marker_allocation_v2.yaml"),
])
def test_pilot_allocations_still_rebuild_byte_for_byte(tmp_path, config, stored):
    out = tmp_path / "alloc.yaml"
    result = run("--config", ROOT / config, "--topics", ROOT / "data/topics/pilot_topics.yaml",
                 "--out", out)
    assert result.returncode == 0, result.stderr
    assert out.read_bytes() == (ROOT / stored).read_bytes()
    assert run("--config", ROOT / config, "--topics", ROOT / "data/topics/pilot_topics.yaml",
               "--out", ROOT / stored, "--check").returncode == 0
