"""Topic bank: the brief's rules, the curation gate, and the review export.

Everything here is synthetic. The fixture cites only a synthetic registry, so no
test claims a real dataset inspired a topic.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from reasonstyle.config import load_config
from reasonstyle.corpus.sources import load_registry
from reasonstyle.corpus.topic_review import build_topic_export
from reasonstyle.corpus.topics import (
    CURATION_JUDGEMENTS,
    TopicBank,
    TopicBankError,
    check_topics,
    load_topic_bank,
)

ROOT = Path(__file__).resolve().parents[1]
TOPICS = ROOT / "data" / "fixtures" / "topics.yaml"
REGISTRY = ROOT / "data" / "fixtures" / "registry.yaml"
CONFIG = ROOT / "configs" / "experiment.yaml"


@pytest.fixture(scope="module")
def registry():
    return load_registry(REGISTRY)


@pytest.fixture(scope="module")
def cfg_two_facts(tmp_path_factory):
    """The rules must still hold if the pilot's one-fact limit is later raised."""
    raw = yaml.safe_load(CONFIG.read_text())
    raw["topics"]["facts_per_option"]["max"] = 2
    path = tmp_path_factory.mktemp("cfg") / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


@pytest.fixture(scope="module")
def raw_topics():
    return yaml.safe_load(TOPICS.read_text())["topics"]


def curated(raw_topics, **changes):
    """The fixture's curated topic, with changes applied."""
    topic = copy.deepcopy(next(t for t in raw_topics if t["status"] == "curated"))
    topic.update(changes)
    return topic


def check(topics, cfg, registry, source_texts=None):
    return check_topics(TopicBank.model_validate({"topics": topics}), cfg, registry,
                        source_texts=source_texts or {})


def codes(report):
    return {f.code for f in report.findings}


def words(n):
    return " ".join(["word"] * n)


# --- the fixture -------------------------------------------------------------


def test_the_fixture_is_machine_valid(cfg, registry):
    report = check_topics(load_topic_bank(TOPICS), cfg, registry, source_texts={})
    assert report.errors == () and report.warnings == ()


# --- structure ---------------------------------------------------------------


def test_options_and_goals_need_exactly_both_semantic_options(raw_topics):
    with pytest.raises(Exception, match="opt_1"):
        TopicBank.model_validate({"topics": [curated(raw_topics, options={"opt_1": "x"})]})


def test_retained_source_excerpts_are_not_part_of_the_format(raw_topics, tmp_path):
    """No source wording is kept or passed on, so there is no field for it."""
    topic = curated(raw_topics, retained_excerpts=[{"text": "copied"}])
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump({"topics": [topic]}))
    with pytest.raises(TopicBankError):
        load_topic_bank(path)


def test_a_source_reference_is_a_key_locator_and_summary(raw_topics):
    ref = curated(raw_topics)["source_references"][0]
    assert set(ref) == {"key", "locator", "inspiration_summary"}


# --- scenario facts, per variant ---------------------------------------------


@pytest.mark.parametrize("n", [0, 2])
def test_the_pilot_needs_exactly_one_fact_per_option_per_variant(raw_topics, cfg, registry, n):
    topic = curated(raw_topics)
    facts = [f"Fact number {i} about the plant." for i in range(n)]
    topic["variants"]["v1"]["scenario_facts"] = {"opt_1": facts, "opt_2": facts}
    assert "E_TOPIC_FACT_COUNT" in codes(check([topic], cfg, registry))


def test_both_options_get_the_same_number_of_facts_within_a_variant(raw_topics, cfg_two_facts,
                                                                      registry):
    """Checked under a two-fact limit, where unequal counts become possible."""
    topic = curated(raw_topics)
    topic["variants"]["v1"]["scenario_facts"]["opt_1"].append("A second reliability fact.")
    assert "E_TOPIC_FACT_IMBALANCE" in codes(check([topic], cfg_two_facts, registry))


def test_variants_that_share_every_fact_are_rejected(raw_topics, cfg, registry):
    """Differing only in the context sentence is not enough."""
    topic = curated(raw_topics)
    topic["variants"]["v2"]["scenario_facts"] = copy.deepcopy(topic["variants"]["v1"]["scenario_facts"])
    assert "E_TOPIC_VARIANTS_SHARE_ALL_FACTS" in codes(check([topic], cfg, registry))


def test_changing_one_fact_is_enough_for_the_machine_check(raw_topics, cfg, registry):
    """Whether that change is substantive is the curator's judgement."""
    topic = curated(raw_topics)
    topic["variants"]["v2"]["scenario_facts"]["opt_1"] = copy.deepcopy(
        topic["variants"]["v1"]["scenario_facts"]["opt_1"])
    assert "E_TOPIC_VARIANTS_SHARE_ALL_FACTS" not in codes(check([topic], cfg, registry))


# --- word limits -------------------------------------------------------------


def test_a_scenario_fact_is_limited_to_25_words(raw_topics, cfg, registry):
    topic = curated(raw_topics)
    topic["variants"]["v1"]["scenario_facts"]["opt_1"] = [words(25)]
    assert "E_TOPIC_TOO_LONG" not in codes(check([topic], cfg, registry))
    topic["variants"]["v1"]["scenario_facts"]["opt_1"] = [words(26)]
    finding = next(f for f in check([topic], cfg, registry).errors if f.code == "E_TOPIC_TOO_LONG")
    assert finding.detail["field"] == "variants.v1.scenario_facts.opt_1[1]"
    assert finding.detail["limit"] == 25


def test_the_limit_applies_to_each_fact_not_to_the_whole_list(raw_topics, cfg_two_facts, registry):
    topic = curated(raw_topics)
    topic["variants"]["v1"]["scenario_facts"] = {"opt_1": [words(20), words(20)],
                                                 "opt_2": [words(20), words(20)]}
    assert "E_TOPIC_TOO_LONG" not in codes(check([topic], cfg_two_facts, registry))


def test_a_prose_field_over_its_limit_is_rejected(raw_topics, cfg, registry):
    report = check([curated(raw_topics, decision_framing=words(41))], cfg, registry)
    assert "E_TOPIC_TOO_LONG" in codes(report)


# --- content screens ---------------------------------------------------------


def test_an_evidence_claim_cannot_become_a_scenario_fact(raw_topics, cfg, registry):
    topic = curated(raw_topics)
    topic["variants"]["v1"]["scenario_facts"]["opt_1"][0] = "Studies show the extension is cheaper."
    assert "E_TOPIC_FORBIDDEN_PHRASE" in codes(check([topic], cfg, registry))


def test_a_warning_phrase_warns_without_blocking(raw_topics, cfg, registry):
    topic = curated(raw_topics)
    topic["variants"]["v1"]["scenario_facts"]["opt_1"][0] = "The plant uses proven turbine technology."
    report = check([topic], cfg, registry)
    assert "W_TOPIC_FORBIDDEN_PHRASE" in codes(report) and not report.errors


def test_display_labels_are_rejected(raw_topics, cfg, registry):
    topic = curated(raw_topics)
    topic["variants"]["v1"]["scenario_facts"]["opt_1"][0] = "Option A keeps the margin above the threshold."
    assert "E_TOPIC_LABEL_LEAKAGE" in codes(check([topic], cfg, registry))


def test_party_framing_terms_warn_but_never_block(raw_topics, cfg, registry):
    """Policy decisions may be contested; the curator decides."""
    report = check([curated(raw_topics, notes=None,
                            decision_framing="A partisan dispute over how a grid operator covers a shortfall.")],
                   cfg, registry)
    assert "W_TOPIC_FRAMING_TERMS" in codes(report) and not report.errors


def test_naming_who_is_affected_is_not_identity_framing(raw_topics, cfg, registry):
    """The rule bars identity appeals, not neutral descriptions of who bears a
    policy's costs. A trade-off usually cannot be stated without them."""
    topic = curated(raw_topics)
    topic["variants"]["v1"]["scenario_facts"]["opt_1"][0] = (
        "Tenants and households without private parking would keep a supply they can rely on.")
    report = check([topic], cfg, registry)
    assert "W_TOPIC_FRAMING_TERMS" not in codes(report) and not report.errors


# --- numerical specificity ---------------------------------------------------


def _v1_facts(raw_topics, opt_1, opt_2):
    topic = curated(raw_topics)
    topic["variants"]["v1"]["scenario_facts"] = {"opt_1": [opt_1], "opt_2": [opt_2]}
    return topic


@pytest.mark.parametrize("specific", [
    "The plant keeps 40 per cent of winter capacity in reserve.",
    "The storage build halves the region's emissions.",
    "The plant can run for three extra winters.",
    "The tender closes in March.",
])
def test_a_specific_fact_opposite_a_qualitative_one_is_flagged(raw_topics, cfg, registry, specific):
    topic = _v1_facts(raw_topics, specific, "Retiring the plant would substantially cut emissions.")
    report = check([topic], cfg, registry)
    assert "W_TOPIC_SPECIFICITY_MISMATCH" in codes(report) and not report.errors


def test_comparably_specific_facts_are_not_flagged(raw_topics, cfg, registry):
    topic = _v1_facts(raw_topics, "The plant covers two cold spells a winter.",
                      "Retiring the plant cuts emissions by a third.")
    assert "W_TOPIC_SPECIFICITY_MISMATCH" not in codes(check([topic], cfg, registry))


def test_qualitative_facts_on_both_sides_are_not_flagged(raw_topics, cfg, registry):
    assert "W_TOPIC_SPECIFICITY_MISMATCH" not in codes(check([curated(raw_topics)], cfg, registry))


# --- the source-overlap screen ------------------------------------------------

SOURCE = {"doc": "The operator must ensure that storage facilities remain available to every "
                 "market participant on equal terms."}


def test_six_shared_words_are_flagged(raw_topics, cfg, registry):
    topic = curated(raw_topics,
                    decision_framing="A regulator asks whether storage facilities remain available to every market participant.")
    report = check([topic], cfg, registry, source_texts=SOURCE)
    finding = next(f for f in report.warnings if f.code == "W_TOPIC_SOURCE_OVERLAP")
    assert finding.detail["field"] == "decision_framing" and finding.detail["words"] >= 6
    assert not report.errors


def test_five_shared_words_are_not_flagged(raw_topics, cfg, registry):
    topic = curated(raw_topics, decision_framing="A regulator asks whether storage facilities remain open to newcomers.")
    assert "W_TOPIC_SOURCE_OVERLAP" not in codes(check([topic], cfg, registry, source_texts=SOURCE))


def test_the_screen_checks_scenario_facts_too(raw_topics, cfg, registry):
    topic = curated(raw_topics)
    topic["variants"]["v1"]["scenario_facts"]["opt_2"] = [
        "Storage facilities remain available to every market participant on equal terms."]
    assert "W_TOPIC_SOURCE_OVERLAP" in codes(check([topic], cfg, registry, source_texts=SOURCE))


def test_the_screen_is_a_required_argument(raw_topics, cfg, registry):
    """It cannot be skipped by forgetting it."""
    with pytest.raises(TypeError):
        check_topics(TopicBank.model_validate({"topics": [curated(raw_topics)]}), cfg, registry)


def test_the_export_states_whether_the_screen_ran(cfg, registry):
    bank = load_topic_bank(TOPICS)
    unscreened = build_topic_export(bank, check_topics(bank, cfg, registry, source_texts={}),
                                    cfg, registry, TOPICS)
    screened = build_topic_export(bank, check_topics(bank, cfg, registry, source_texts=SOURCE),
                                  cfg, registry, TOPICS)
    assert "Overlap screen: NOT RUN" in unscreened.files["index.md"]
    assert "compared with 1 downloaded source documents" in screened.files["index.md"]


# --- variants and identity ---------------------------------------------------


def test_variants_must_be_exactly_the_configured_two(raw_topics, cfg, registry):
    topic = curated(raw_topics)
    del topic["variants"]["v2"]
    assert "E_TOPIC_VARIANTS" in codes(check([topic], cfg, registry))


def test_variants_must_differ_in_context(raw_topics, cfg, registry):
    topic = curated(raw_topics)
    topic["variants"]["v2"]["context"] = topic["variants"]["v1"]["context"]
    assert "E_TOPIC_VARIANTS_IDENTICAL" in codes(check([topic], cfg, registry))


def test_the_decision_id_must_match_its_domain(raw_topics, cfg, registry):
    assert "E_TOPIC_ID_DOMAIN_MISMATCH" in codes(
        check([curated(raw_topics, decision_id="climate_99")], cfg, registry))
    assert "E_TOPIC_UNKNOWN_DOMAIN" in codes(
        check([curated(raw_topics, domain="transport", decision_id="transport_01")], cfg, registry))


def test_duplicate_decision_ids_are_rejected(raw_topics, cfg, registry):
    a = curated(raw_topics)
    b = curated(raw_topics, decision_framing="A different framing of another grid decision.")
    assert "E_TOPIC_DUPLICATE_ID" in codes(check([a, b], cfg, registry))


# --- sources -----------------------------------------------------------------


@pytest.mark.parametrize("key", ["synthetic_unverified_corpus", "no_such_source"])
def test_a_topic_may_cite_only_citable_sources(raw_topics, cfg, registry, key):
    topic = curated(raw_topics)
    topic["source_references"][0]["key"] = key
    assert "E_TOPIC_SOURCE_NOT_CITABLE" in codes(check([topic], cfg, registry))


def test_a_constructed_topic_may_cite_nothing(raw_topics, cfg, registry):
    assert not check([curated(raw_topics, source_references=[])], cfg, registry).errors


# --- curation ----------------------------------------------------------------


@pytest.mark.parametrize("change", [
    {"can_be_made_self_contained": None},
    {"no_option_dominates_given_the_facts": False},
    {"both_goals_represented_in_every_variant": None},
    {"curated_by": None},
    {"curated_at": None},
])
def test_curated_needs_every_judgement_true_plus_who_and_when(raw_topics, cfg, registry, change):
    topic = curated(raw_topics)
    topic["curation"].update(change)
    assert "E_TOPIC_CURATION_INCOMPLETE" in codes(check([topic], cfg, registry))


def test_a_rejection_needs_a_reason(raw_topics, cfg, registry):
    topic = curated(raw_topics, status="rejected")
    assert "E_TOPIC_CURATION_INCOMPLETE" in codes(check([topic], cfg, registry))


def test_fact_balance_is_judged_as_three_separate_questions():
    for judgement in ("each_fact_supports_its_option",
                      "both_goals_represented_in_every_variant",
                      "no_option_dominates_given_the_facts"):
        assert judgement in CURATION_JUDGEMENTS
    assert "scenario_facts_balanced" not in CURATION_JUDGEMENTS
    assert "variants_differ_substantively" in CURATION_JUDGEMENTS


# --- readiness for drafting --------------------------------------------------


def _bank(raw_topics, per_domain, extra_proposed=0):
    topics = []
    for domain, n in per_domain.items():
        for i in range(1, n + 1):
            topics.append(curated(raw_topics, domain=domain, decision_id=f"{domain}_{i:02d}",
                                  decision_framing=f"Decision {i} in the {domain} domain about capacity."))
    for i in range(extra_proposed):
        topics.append(curated(raw_topics, status="proposed", domain="energy",
                              decision_id=f"energy_{90 + i}",
                              decision_framing=f"Proposed candidate {i} about capacity."))
    return topics


def test_balance_counts_only_curated_topics(raw_topics, cfg, registry):
    """Extra proposed candidates never cause a balance problem."""
    report = check(_bank(raw_topics, {"climate": 4, "energy": 4, "technology": 4}, extra_proposed=3),
                   cfg, registry)
    assert report.curated_per_domain == {"climate": 4, "energy": 4, "technology": 4}
    assert report.ready_for_drafting


@pytest.mark.parametrize("per_domain", [{"climate": 5, "energy": 4, "technology": 4},
                                        {"climate": 3, "energy": 4, "technology": 4}])
def test_drafting_needs_exactly_four_curated_per_domain(raw_topics, cfg, registry, per_domain):
    assert not check(_bank(raw_topics, per_domain), cfg, registry).ready_for_drafting


def test_a_machine_error_blocks_drafting(raw_topics, cfg, registry):
    topics = _bank(raw_topics, {"climate": 4, "energy": 4, "technology": 4})
    topics[0]["decision_framing"] = words(41)
    assert not check(topics, cfg, registry).ready_for_drafting


# --- review export -----------------------------------------------------------


@pytest.fixture(scope="module")
def export(cfg, registry):
    bank = load_topic_bank(TOPICS)
    return build_topic_export(bank, check_topics(bank, cfg, registry, source_texts={}), cfg, registry, TOPICS)


def test_the_export_is_deterministic(cfg, registry, export):
    bank = load_topic_bank(TOPICS)
    again = build_topic_export(bank, check_topics(bank, cfg, registry, source_texts={}), cfg, registry, TOPICS)
    assert again.files == export.files and again.manifest == export.manifest


def test_every_topic_has_a_page_listed_in_the_index(export):
    for decision_id in ("energy_fixture_001", "climate_fixture_001", "technology_fixture_001"):
        assert f"{decision_id}.md" in export.files
        assert f"`{decision_id}`" in export.files["index.md"]
    assert "all_topics.md" in export.files


def test_pages_are_read_only_and_record_their_hashes(export):
    for name, text in export.files.items():
        assert "read only" in text.lower(), name
        assert export.manifest["topic_bank_content_hash"][:12] in text, name
        assert export.manifest["config_content_hash"][:12] in text, name


def test_each_fact_appears_verbatim_under_its_variant(raw_topics, export):
    page = export.files["energy_fixture_001.md"]
    topic = curated(raw_topics)
    for vid, variant in topic["variants"].items():
        section = page.split(f"### `{vid}`")[1].split("###")[0]
        for opt in ("opt_1", "opt_2"):
            for fact in variant["scenario_facts"][opt]:
                assert fact in section


def test_the_export_states_readiness(export):
    assert "not yet ready for drafting" in export.files["index.md"]
    assert export.manifest["ready_for_drafting"] is False
