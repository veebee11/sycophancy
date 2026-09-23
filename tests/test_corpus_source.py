"""The frozen v2 pilot as the seed of the full corpus: verified before import.

Every tampering test copies the committed seed evidence into a temporary tree
and changes one thing. The gitignored run directory is deliberately not copied:
the committed corpus, manifest, approvals, ledger and allocation must verify on
their own.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
import yaml

from reasonstyle.config import load_config
from reasonstyle.corpus.store import dumps_record
from reasonstyle.corpus.topics import load_topic_bank
from reasonstyle.generation.corpus_source import (
    SeedCorpusError,
    _union_problems,
    combine_corpus,
    load_seed_corpus,
    plan_full_corpus,
)
from reasonstyle.hashing import sha256_of

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "configs/frozen/v2_full.yaml"
BANK = load_topic_bank(ROOT / "data" / "topics" / "full_topics_v2.yaml")
SEED_FILES = [
    "configs/experiment_v2_pilot.yaml",
    "configs/experiment_openai_pilot.yaml",
    "data/pilot/corpus_v2.jsonl",
    "data/pilot/corpus_v2.manifest.json",
    "data/pilot/marker_allocation_v2.yaml",
    "data/pilot/manual_corrections_v2.yaml",
    "data/pilot/scenario_approvals_openai.yaml",
    "data/topics/pilot_topics.yaml",
]


@pytest.fixture(scope="module")
def cfg():
    return load_config(FULL)


@pytest.fixture(scope="module")
def seed(cfg):
    return load_seed_corpus(cfg, bank=BANK, root=ROOT)


@pytest.fixture()
def tree(tmp_path):
    """A copy of the committed seed evidence, with no run directory."""
    for rel in SEED_FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, tmp_path / rel)
    return tmp_path


def full_cfg(tmp_path, *, pins: bool = True, edit=None):
    """The full configuration, optionally without its pins or with one edit."""
    text = FULL.read_text(encoding="utf-8")
    if not pins:
        text = re.sub(r"\n  # The seed's identity.*?\n  verify:\n", "\n  verify:\n", text,
                      flags=re.S)
        assert "pins:" not in text
    if edit:
        text = edit(text)
    # A frozen configuration must live in configs/frozen/ under its version
    # name; a copy of one is no exception.
    path = tmp_path / "configs" / "frozen" / "v2_full.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def refused(cfg, root, match):
    with pytest.raises(SeedCorpusError, match=match):
        load_seed_corpus(cfg, bank=BANK, root=root)


def manifest(root) -> dict:
    return json.loads((root / "data/pilot/corpus_v2.manifest.json").read_text())


def write_manifest(root, data, *, reseal=False):
    if reseal:
        data["corpus_sha256"] = sha256_of((root / "data/pilot/corpus_v2.jsonl").read_text())
    (root / "data/pilot/corpus_v2.manifest.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def corpus_lines(root) -> list[str]:
    return (root / "data/pilot/corpus_v2.jsonl").read_text().splitlines()


def write_corpus(root, lines):
    (root / "data/pilot/corpus_v2.jsonl").write_text("".join(l + "\n" for l in lines))


# --- a valid import ---------------------------------------------------------------


def test_the_committed_seed_verifies(seed):
    assert seed.provenance["counts"] == {"decisions": 12, "scenarios": 24, "groups": 48,
                                         "texts": 192}
    assert len(seed.records) == 24 and len(seed.decision_ids) == 12
    assert seed.provenance["manual_corrections"] == 8


def test_seed_records_keep_the_pilot_configuration_hash(seed):
    pilot_hash = load_config(ROOT / "configs/experiment_v2_pilot.yaml").content_hash
    assert {r.config_content_hash for r in seed.records} == {pilot_hash}
    assert seed.provenance["config_content_hash"] == pilot_hash


def test_seed_decisions_are_exactly_the_curated_pilot_decisions(seed):
    pilot = load_topic_bank(ROOT / "data/topics/pilot_topics.yaml")
    assert set(seed.decision_ids) == {t.decision_id for t in pilot.topics if t.status == "curated"}


def test_the_seed_verifies_without_the_gitignored_run_directory(tree, cfg):
    assert not (tree / "data/pilot/run_v2").exists()
    seed = load_seed_corpus(cfg, bank=BANK, root=tree)
    assert seed.provenance["run_evidence"].startswith("absent")
    assert seed.provenance["counts"]["texts"] == 192


def test_the_provenance_block_is_json_and_portable(seed):
    text = json.dumps(seed.provenance, sort_keys=True)
    assert str(ROOT) not in text
    assert len(seed.provenance["scenarios"]) == 24


# --- the corpus and manifest ---------------------------------------------------------


def test_an_altered_corpus_is_refused(tree, cfg):
    lines = corpus_lines(tree)
    lines[0] = lines[0].replace("stricter", "tighter", 1)
    write_corpus(tree, lines)
    refused(cfg, tree, "hashes to")


def test_an_altered_manifest_is_refused(tree, cfg):
    data = manifest(tree)
    data["outstanding_human_review"] = 0
    write_manifest(tree, data)
    refused(cfg, tree, "not the pinned")


def test_a_corpus_and_manifest_rewritten_together_still_fail_the_pins(tree, cfg):
    lines = corpus_lines(tree)
    lines[0] = lines[0].replace("stricter", "tighter", 1)
    write_corpus(tree, lines)
    write_manifest(tree, manifest(tree), reseal=True)
    refused(cfg, tree, "not the pinned")


def test_a_wrong_pilot_config_hash_is_refused(tree, cfg):
    path = tree / "configs/experiment_v2_pilot.yaml"
    text = path.read_text()
    changed = text.replace("counterargument_opening: I disagree with that choice.",
                           "counterargument_opening: I do not agree with that choice.", 1)
    assert changed != text
    path.write_text(changed)
    refused(cfg, tree, "manifest records|pinned")


# --- decisions ----------------------------------------------------------------------


def test_a_missing_seed_decision_is_refused(tree, tmp_path):
    cfg = full_cfg(tmp_path, pins=False)
    write_corpus(tree, [l for l in corpus_lines(tree) if '"decision_id":"climate_01"' not in l])
    data = manifest(tree)
    data["scenarios"] = [e for e in data["scenarios"] if not e["scenario_id"].startswith("climate_01")]
    write_manifest(tree, data, reseal=True)
    refused(cfg, tree, r"missing curated pilot decision\(s\) \['climate_01'\]")


def test_an_extra_seed_decision_is_refused(tree, tmp_path):
    cfg = full_cfg(tmp_path, pins=False)
    lines = corpus_lines(tree)
    extra = lines[0].replace('"decision_id":"climate_01"', '"decision_id":"climate_06"') \
                    .replace('"scenario_id":"climate_01_v1"', '"scenario_id":"climate_06_v1"')
    write_corpus(tree, lines + [extra])
    write_manifest(tree, manifest(tree), reseal=True)
    refused(cfg, tree, "not curated pilot decisions")


def test_a_changed_pilot_definition_in_the_full_bank_is_refused(tree, cfg):
    changed = BANK.model_copy(deep=True)
    topic = next(t for t in changed.topics if t.decision_id == "energy_02")
    object.__setattr__(topic, "decision_framing", topic.decision_framing + " Now.")
    with pytest.raises(SeedCorpusError, match="approved pilot definition"):
        load_seed_corpus(cfg, bank=changed, root=tree)


def test_overlapping_seed_and_new_ids_are_refused(cfg):
    curated = {t.decision_id: t for t in BANK.topics if t.status == "curated"}
    seed_ids = ["climate_01", "climate_02"]
    new_ids = sorted(set(curated) - set(seed_ids)) + ["climate_01"]
    assert any("in both the seed and the remainder" in p
               for p in _union_problems(seed_ids, new_ids, cfg, curated))


def test_an_incomplete_union_is_refused(cfg):
    curated = {t.decision_id: t for t in BANK.topics if t.status == "curated"}
    seed_ids = sorted(curated)[:12]
    new_ids = sorted(curated)[12:-1]
    assert any("not the curated set" in p for p in _union_problems(seed_ids, new_ids, cfg, curated))


# --- scenarios, groups and corrections ------------------------------------------------


def test_a_wrong_scenario_text_hash_is_refused(tree, tmp_path):
    cfg = full_cfg(tmp_path, pins=False)
    data = manifest(tree)
    data["scenarios"][3]["scenario_text_sha256"] = "0" * 64
    write_manifest(tree, data)
    refused(cfg, tree, "does not match the manifest's hash")


def test_a_wrong_group_call_id_is_refused_by_the_pins(tree, cfg):
    data = manifest(tree)
    data["scenarios"][0]["group_call_ids"]["opt_1"] = "a" * 64
    write_manifest(tree, data)
    refused(cfg, tree, "not the pinned")


def test_a_wrong_group_call_id_is_refused_by_correction_provenance(tree, tmp_path):
    cfg = full_cfg(tmp_path, pins=False)
    data = manifest(tree)
    entry = next(e for e in data["scenarios"] if e["scenario_id"] == "climate_01_v2")
    entry["group_call_ids"]["opt_2"] = "b" * 64
    write_manifest(tree, data)
    refused(cfg, tree, "names a call that is not this group's")


def test_a_wrong_group_call_id_is_refused_by_run_evidence_when_present(tree, tmp_path, seed):
    cfg = full_cfg(tmp_path, pins=False)
    run = tree / "data/pilot/run_v2"
    run.mkdir(parents=True)
    pilot_hash = seed.provenance["config_content_hash"]
    lines = [json.dumps({"call_id": c, "outcome": "accepted", "config_content_hash": pilot_hash,
                         "allocation_content_hash": seed.allocation.content_hash})
             for s in seed.provenance["scenarios"].values() for c in s["group_call_ids"].values()]
    (run / "generation_log.jsonl").write_text("\n".join(lines) + "\n")
    assert load_seed_corpus(cfg, bank=BANK, root=tree).provenance["run_evidence"].startswith(
        "checked")
    data = manifest(tree)
    entry = next(e for e in data["scenarios"] if e["scenario_id"] == "energy_01_v1")
    entry["group_call_ids"]["opt_1"] = "c" * 64
    write_manifest(tree, data)
    refused(cfg, tree, "is not an accepted call")


def test_a_wrong_correction_hash_is_refused(tree, tmp_path):
    cfg = full_cfg(tmp_path, pins=False)
    path = tree / "data/pilot/manual_corrections_v2.yaml"
    ledger = yaml.safe_load(path.read_text())
    ledger[0]["corrected_text_sha256"] = "d" * 64
    path.write_text(yaml.safe_dump(ledger, sort_keys=True, allow_unicode=True))
    refused(cfg, tree, "corrected_text_sha256")


def test_an_altered_correction_ledger_fails_its_pin(tree, cfg):
    path = tree / "data/pilot/manual_corrections_v2.yaml"
    path.write_text(path.read_text().replace("would", "might", 1))
    refused(cfg, tree, "not the pinned correction ledger")


def test_correction_metadata_and_original_hashes_are_preserved(seed):
    ledger = yaml.safe_load((ROOT / "data/pilot/manual_corrections_v2.yaml").read_text())
    recorded = [c for e in seed.manifest["scenarios"] for c in e["manual_corrections"]]
    assert len(recorded) == len(ledger) == 8
    for c in recorded:
        assert c["original_text_sha256"] == sha256_of(c["original_text"])
        assert c["editor"] and c["reason"] and c["approval_state"] == "approved"


# --- the seed allocation ----------------------------------------------------------------


def test_a_wrong_seed_allocation_hash_is_refused(tree, tmp_path):
    cfg = full_cfg(tmp_path, edit=lambda t: t.replace(
        "allocation_content_hash: 973ffb2ea6bfac1b09b00d628fea16c540d3031cd44f29acd0c06094771f02cc",
        "allocation_content_hash: " + "e" * 64))
    refused(cfg, tree, "not the pinned")


@pytest.mark.parametrize("value", ["'" + "0" * 64 + "'", "0" * 64, "''", "null", "abc"])
def test_a_malformed_pin_is_refused_not_ignored(tree, tmp_path, value):
    cfg = full_cfg(tmp_path, edit=lambda t: t.replace(
        "corpus_sha256: 7e0dae8415abe0499321bffb19742b031dea8eede08fca1fca20340373e625d8",
        "corpus_sha256: " + value, 1))
    if value.startswith("'0"):
        refused(cfg, tree, "not the pinned")         # well-formed but wrong
    else:
        refused(cfg, tree, "must give every one of")  # 0, empty, null or not a SHA-256


def test_a_missing_pin_is_refused(tree, tmp_path):
    cfg = full_cfg(tmp_path, edit=lambda t: t.replace(
        "    corrections_sha256: 3d3f56005eacbe5b379ef893bf274d6bc6e1327f0cb584fc25153431d1af441c\n",
        "", 1))
    refused(cfg, tree, "must give every one of")


def test_a_hand_edited_seed_allocation_is_refused(tree, cfg):
    path = tree / "data/pilot/marker_allocation_v2.yaml"
    path.write_text(path.read_text().replace("marker_string: therefore",
                                             "marker_string: consequently", 1))
    refused(cfg, tree, "edited after it was generated")


# --- planning and combination -----------------------------------------------------------


def test_the_plan_skips_the_seed(cfg, seed):
    plan = plan_full_corpus(cfg, BANK, seed)
    assert len(plan.seed_decisions) == 12 and len(plan.new_decisions) == 48
    assert not set(plan.seed_decisions) & set(plan.new_decisions)
    assert len(plan.new_scenario_ids) == 96 and len(plan.new_groups) == 192
    assert plan.new_texts == 768
    assert set(plan.seed_decisions) | set(plan.new_decisions) == set(plan.expected_decisions)
    assert len(plan.expected_decisions) == 60


def _new_records(cfg, seed, plan):
    template = seed.records[0]
    out = []
    for sid in plan.new_scenario_ids:
        decision_id, variant = sid.rsplit("_v", 1)
        out.append(template.model_copy(update={
            "decision_id": decision_id, "scenario_id": sid, "variant_id": int(variant),
            "config_content_hash": cfg.content_hash, "config_version": cfg.config_version}))
    return out


def test_combined_assembly_carries_seed_records_byte_for_byte(cfg, seed):
    plan = plan_full_corpus(cfg, BANK, seed)
    body, manifest_ = combine_corpus(seed, plan, _new_records(cfg, seed, plan),
                                     {"config_content_hash": cfg.content_hash}, cfg=cfg)
    lines = body.splitlines()
    assert len(lines) == 120
    committed = set(corpus_lines(ROOT))
    assert committed <= set(lines)                       # every seed line, byte for byte
    assert [json.loads(l)["scenario_id"] for l in lines] == sorted(
        json.loads(l)["scenario_id"] for l in lines)     # deterministic order
    roles = [s["role"] for s in manifest_["sources"]]
    assert roles == ["seed", "full_run"]
    assert sum(v == "seed" for v in manifest_["scenario_sources"].values()) == 24
    assert manifest_["sources"][0]["config_content_hash"] != cfg.content_hash
    again, _ = combine_corpus(seed, plan, list(reversed(_new_records(cfg, seed, plan))),
                              {"config_content_hash": cfg.content_hash}, cfg=cfg)
    assert again == body


def test_combined_assembly_refuses_a_new_record_for_a_seed_scenario(cfg, seed):
    plan = plan_full_corpus(cfg, BANK, seed)
    records = _new_records(cfg, seed, plan)
    records.append(seed.records[0].model_copy(update={"config_content_hash": cfg.content_hash}))
    with pytest.raises(SeedCorpusError, match="seed scenario"):
        combine_corpus(seed, plan, records, {}, cfg=cfg)


def test_combined_assembly_refuses_an_incomplete_remainder(cfg, seed):
    plan = plan_full_corpus(cfg, BANK, seed)
    with pytest.raises(SeedCorpusError, match="do not match the plan"):
        combine_corpus(seed, plan, _new_records(cfg, seed, plan)[:-1], {}, cfg=cfg)


def test_combined_assembly_refuses_a_record_restamped_or_built_elsewhere(cfg, seed):
    plan = plan_full_corpus(cfg, BANK, seed)
    records = _new_records(cfg, seed, plan)
    records[0] = records[0].model_copy(update={"config_content_hash": "f" * 64})
    with pytest.raises(SeedCorpusError, match="not built under this design"):
        combine_corpus(seed, plan, records, {}, cfg=cfg)


def test_the_seed_is_read_only(cfg):
    before = {rel: (ROOT / rel).read_bytes() for rel in SEED_FILES}
    load_seed_corpus(cfg, bank=BANK, root=ROOT)
    assert {rel: (ROOT / rel).read_bytes() for rel in SEED_FILES} == before
    assert dumps_record  # the combined corpus re-serialises only new records
