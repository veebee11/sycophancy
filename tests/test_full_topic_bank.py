"""The full topic bank as a whole: counts, uniqueness, pilot preservation,
locators, curation records, near duplicates, and nothing generated yet.

The committed-bank tests read the real files. Locator checks against the pinned
sources run only where those (gitignored) files have been fetched. Every other
test mutates the committed bank's text, so each check is shown to fire.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from reasonstyle.config import load_config
from reasonstyle.corpus.full_bank import check_full_bank, check_locator, record_blocks
from reasonstyle.corpus.sources import load_registry
from reasonstyle.corpus.topics import TopicBank, load_topic_bank

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment_v2_full.draft.yaml"
PILOT = ROOT / "data" / "topics" / "pilot_topics.yaml"
BANK = ROOT / "data" / "topics" / "full_topics_v2.yaml"
REGISTRY = ROOT / "data" / "sources" / "registry.yaml"
RAW = ROOT / "data" / "sources" / "raw"
HAVE_SOURCES = (RAW / "POLIANNA_v1_1").exists() and any(RAW.glob("eurlex_*_EN.txt"))

CFG = load_config(CONFIG)
BANK_TEXT = BANK.read_text(encoding="utf-8")
PILOT_TEXT = PILOT.read_text(encoding="utf-8")


def check(bank_text=BANK_TEXT, *, raw_dir=None, root=None, blocked=True, paths=None):
    bank = TopicBank.model_validate(yaml.safe_load(bank_text))
    return check_full_bank(
        bank=bank, bank_text=bank_text,
        pilot=load_topic_bank(PILOT), pilot_text=PILOT_TEXT,
        domains=list(CFG.raw["domains"]["ids"]),
        per_domain=CFG.raw["topics"]["curated_per_domain_for_drafting"],
        registry_keys=load_registry(REGISTRY).datasets,
        raw_dir=raw_dir, output_paths=paths or CFG.raw["paths"],
        root=root or ROOT, generation_blocked=blocked,
    )


def codes(findings, severity="error"):
    return sorted({f.code for f in findings if f.severity == severity})


def drop_record(text, decision_id):
    blocks = record_blocks(text)
    return text.replace(blocks[decision_id], "", 1)


# --- the committed bank -------------------------------------------------------


def test_the_full_config_points_at_the_full_bank():
    assert Path(CFG.raw["topics"]["bank"]) == Path("data/topics/full_topics_v2.yaml")


def test_the_committed_bank_has_the_design_counts():
    bank = load_topic_bank(BANK)
    active = [t for t in bank.topics if t.status != "rejected"]
    assert len(active) == 60
    for domain in ("climate", "energy", "technology"):
        assert sum(t.domain == domain for t in active) == 20
    assert sum(t.status == "curated" for t in bank.topics) == 60
    assert sum(t.status == "proposed" for t in bank.topics) == 0
    assert sorted(t.decision_id for t in bank.topics if t.status == "rejected") == [
        "climate_03", "energy_04", "technology_05"]


def test_the_added_decisions_are_numbered_06_to_21_in_each_domain():
    bank = load_topic_bank(BANK)
    pilot_ids = {t.decision_id for t in load_topic_bank(PILOT).topics}
    added = sorted(t.decision_id for t in bank.topics if t.decision_id not in pilot_ids)
    assert added == sorted(f"{d}_{i:02d}" for d in ("climate", "energy", "technology")
                           for i in range(6, 22))


def test_every_added_decision_carries_the_recorded_approval():
    pilot_ids = {t.decision_id for t in load_topic_bank(PILOT).topics}
    for t in load_topic_bank(BANK).topics:
        if t.decision_id in pilot_ids:
            continue
        assert t.status == "curated", t.decision_id
        assert all(v is True for v in t.curation.recorded().values()), t.decision_id
        assert t.curation.curated_by == "Vidhi Bhutani"
        assert str(t.curation.curated_at) == "2026-09-21"


def test_the_committed_bank_passes_without_sources():
    findings = check()
    assert codes(findings) == []


@pytest.mark.skipif(not HAVE_SOURCES, reason="pinned source files not fetched")
def test_every_committed_locator_points_at_something_real():
    assert codes(check(raw_dir=RAW)) == []


def test_every_pilot_record_is_carried_over_byte_for_byte():
    pilot, full = record_blocks(PILOT_TEXT), record_blocks(BANK_TEXT)
    assert len(pilot) == 15
    for did, block in pilot.items():
        assert full[did] == block, did


# --- each check fires ------------------------------------------------------------


def test_a_changed_pilot_record_is_caught():
    text = BANK_TEXT.replace("faster cuts in road-transport emissions",
                             "quicker cuts in road-transport emissions", 1)
    assert "E_BANK_PILOT_CHANGED" in codes(check(text))


def test_a_missing_pilot_record_is_caught():
    found = codes(check(drop_record(BANK_TEXT, "energy_02")))
    assert "E_BANK_PILOT_MISSING" in found
    assert "E_BANK_DOMAIN_COUNT" in found


def test_a_missing_added_decision_breaks_the_counts():
    found = codes(check(drop_record(BANK_TEXT, "technology_21")))
    assert {"E_BANK_COUNT", "E_BANK_DOMAIN_COUNT"} <= set(found)


def test_a_duplicate_id_is_caught():
    text = BANK_TEXT.replace("decision_id: energy_21", "decision_id: energy_20", 1)
    assert "E_BANK_DUPLICATE_ID" in codes(check(text))


def test_a_repeated_proposition_is_caught():
    bank = load_topic_bank(BANK)
    a = next(t for t in bank.topics if t.decision_id == "energy_20").options["opt_1"]
    b = next(t for t in bank.topics if t.decision_id == "energy_21").options["opt_1"]
    text = BANK_TEXT.replace(f'opt_1: "{b}"', f'opt_1: "{a}"', 1)
    assert "E_BANK_DUPLICATE_PROPOSITION" in codes(check(text))


def test_a_near_duplicate_decision_is_caught():
    bank = load_topic_bank(BANK)
    t20 = next(t for t in bank.topics if t.decision_id == "energy_20")
    t21 = next(t for t in bank.topics if t.decision_id == "energy_21")
    text = BANK_TEXT.replace(f'decision_framing: "{t21.decision_framing}"',
                             f'decision_framing: "{t20.decision_framing[:-1]} soon."', 1)
    text = text.replace(f'opt_1: "{t21.options["opt_1"]}"',
                        f'opt_1: "{t20.options["opt_1"][:-1]} quickly."', 1)
    text = text.replace(f'opt_2: "{t21.options["opt_2"]}"',
                        f'opt_2: "{t20.options["opt_2"][:-1]} first."', 1)
    assert "E_BANK_NEAR_DUPLICATE" in codes(check(text))


def test_an_unknown_source_key_is_caught():
    text = BANK_TEXT.replace("key: eurlex_lulucf", "key: eurlex_forests", 1)
    assert "E_BANK_UNKNOWN_SOURCE" in codes(check(text))


def as_proposed(text, decision_id):
    """The bank with one added decision put back to proposed, curation cleared."""
    block = record_blocks(text)[decision_id]
    new = block.replace("status: curated", "status: proposed")
    new = re.sub(r"(\n      [a-z_]+): true", r"\1: null", new)
    new = re.sub(r'\n      curated_by: .*\n      curated_at: .*', "", new)
    return text.replace(block, new, 1)


def test_a_proposed_decision_with_no_curation_passes():
    assert "E_BANK_NEW_CURATION" not in codes(check(as_proposed(BANK_TEXT, "climate_21")))


def test_a_curated_decision_needs_every_judgement():
    block = record_blocks(BANK_TEXT)["climate_21"]
    text = BANK_TEXT.replace(block, block.replace("underdetermined: true",
                                                  "underdetermined: null", 1), 1)
    assert "E_BANK_NEW_CURATION" in codes(check(text))


def test_a_curated_decision_needs_a_curator():
    block = record_blocks(BANK_TEXT)["energy_06"]
    text = BANK_TEXT.replace(block, block.replace('      curated_by: "Vidhi Bhutani"\n', ""), 1)
    assert "E_BANK_NEW_CURATION" in codes(check(text))


def test_a_proposed_decision_may_not_carry_curation():
    text = as_proposed(BANK_TEXT, "technology_06")
    block = record_blocks(text)["technology_06"]
    text = text.replace(block, block.replace("underdetermined: null",
                                             "underdetermined: true", 1), 1)
    assert "E_BANK_NEW_CURATION" in codes(check(text))


def test_an_added_decision_may_not_be_rejected_silently():
    block = record_blocks(BANK_TEXT)["technology_06"]
    text = BANK_TEXT.replace(block, block.replace("status: curated", "status: rejected"), 1)
    assert "E_BANK_NEW_CURATION" in codes(check(text))


def test_anything_generated_from_the_draft_bank_is_caught(tmp_path):
    (tmp_path / "data" / "full" / "run_v2").mkdir(parents=True)
    assert "E_BANK_DOWNSTREAM_EXISTS" in codes(check(root=tmp_path))


def test_the_pre_drafting_allocation_may_exist(tmp_path):
    (tmp_path / "data" / "full").mkdir(parents=True)
    (tmp_path / "data" / "full" / "marker_allocation_full_v2.yaml").write_text("groups: []\n")
    assert "E_BANK_DOWNSTREAM_EXISTS" not in codes(check(root=tmp_path))


@pytest.mark.parametrize("name", ["corpus_full_v2.jsonl", "scenario_approvals_full_v2.yaml",
                                  "manual_corrections_full_v2.yaml"])
def test_generated_material_beside_the_allocation_is_still_caught(tmp_path, name):
    (tmp_path / "data" / "full").mkdir(parents=True)
    (tmp_path / "data" / "full" / name).write_text("x\n")
    assert "E_BANK_DOWNSTREAM_EXISTS" in codes(check(root=tmp_path))


def test_nothing_generated_passes(tmp_path):
    assert "E_BANK_DOWNSTREAM_EXISTS" not in codes(check(root=tmp_path))


def test_generation_must_stay_blocked_while_decisions_are_proposed():
    text = as_proposed(BANK_TEXT, "energy_21")
    assert "E_BANK_GENERATION_UNBLOCKED" in codes(check(text, blocked=False))


def test_the_committed_full_config_keeps_generation_blocked():
    assert CFG.raw["corpus"]["generation_block"]["blocked"] is True


# --- locators against synthetic sources -------------------------------------------


@pytest.fixture()
def raw(tmp_path):
    (tmp_path / "eurlex_02099R0001-20990101_EN.txt").write_text(
        "Article 5  \ntext\nArticle 10a  \nCHAPTER IVa\n", encoding="utf-8")
    articles = tmp_path / "POLIANNA_v1_1" / "POLIANNA_v1_1" / "03b_processed_to_json"
    (articles / "EU_32099L0001_Title_0_Chapter_1_Section_0_Article_07").mkdir(parents=True)
    (tmp_path / "Export_PSTW_GENAI_AnnexII_usecases_dataset.csv").write_text(
        "ID;Process type;Interaction;Responsible organisation category\n"
        "X-1;Public services and engagement;G2C;Private sector\n"
        "X-2;Internal management;G2G;Consortium\n", encoding="utf-8")
    (tmp_path / "Export_PSTW_GENAI_AnnexI_guidelines_dataset.csv").write_text(
        "Document ID;Transparency;Accountability\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("locator", [
    "CELEX 02099R0001-20990101, art. 5 (something)",
    "CELEX 02099R0001-20990101, arts. 5 and 10a (something)",
    "CELEX 02099R0001-20990101, art. 10a(8) and Chapter IVa (something)",
    "CELEX 32099L0001, art. 7 (something)",
    "Annex II process type: internal management; Annex I principle: accountability",
    "Annex II interaction: government to citizen; organisation categories: private sector, consortium",
])
def test_a_real_locator_passes(raw, locator):
    assert check_locator(locator, raw) == []


@pytest.mark.parametrize(("locator", "problem"), [
    ("CELEX 02099R0001-20990101, art. 6 (x)", "no Article 6"),
    ("CELEX 02099R0001-20990101, art. 5 and Chapter IX (x)", "no Chapter IX"),
    ("CELEX 02099R0002-20990101, art. 5 (x)", "no pinned text"),
    ("CELEX 32099L0001, art. 8 (x)", "no CELEX 32099L0001 article 8"),
    ("Annex II process type: space exploration", "no Annex II process type"),
    ("Annex II process type: internal management; Annex I principle: speed", "no Annex I principle"),
    ("Annex II mood: happy; Annex I principle: transparency", "unknown JRC locator field"),
    ("See the directive", "unrecognised locator format"),
])
def test_a_locator_that_points_at_nothing_is_caught(raw, locator, problem):
    problems = check_locator(locator, raw)
    assert any(re.search(re.escape(problem), p) for p in problems), problems
