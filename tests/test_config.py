"""Focused checks on the frozen experiment configuration.

Two kinds of test:

* **properties** of the shipped ``configs/experiment_v1.yaml``;
* **mutation tests** that deliberately corrupt a copy and assert the loader
  refuses it. These are the ones that matter: they prove the immutability
  guards and the design invariants actually bite.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import yaml

from reasonstyle.config import ConfigError, latest_config_path, load_config
from reasonstyle.hashing import file_sha256

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATH = CONFIGS / "experiment_v1.yaml"
CONFIG_V2_PATH = CONFIGS / "experiment_v2.yaml"
LATEST_PATH = latest_config_path(CONFIGS)


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture
def raw(cfg):
    return copy.deepcopy(cfg.raw)


def load_mutated(tmp_path: Path, raw: dict, name: str = "experiment_v1.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw, sort_keys=True, allow_unicode=True), encoding="utf-8")
    return load_config(path)


# --- properties of the shipped config ---------------------------------------


def test_config_loads(cfg):
    assert cfg.config_version == "v1"
    assert cfg.parsed.status == "draft"


def test_core_conditions_are_the_frozen_two_by_two(cfg):
    core = cfg.parsed.conditions.core
    assert set(core) == {"RS", "RP", "NS", "NP"}
    assert (core["RS"].reason, core["RS"].style) == ("present", "explicit")
    assert (core["RP"].reason, core["RP"].style) == ("present", "plain")
    assert (core["NS"].reason, core["NS"].style) == ("absent", "explicit")
    assert (core["NP"].reason, core["NP"].style) == ("absent", "plain")


def test_diagnostic_registry_is_empty_in_v1(cfg):
    assert cfg.diagnostic_condition_ids() == []


def test_the_five_contrasts_are_correct(cfg):
    expected = {
        "style_without_reason": {"NS": 1, "NP": -1},
        "style_with_reason": {"RS": 1, "RP": -1},
        "content_with_style": {"RS": 1, "NS": -1},
        "content_plain": {"RP": 1, "NP": -1},
        "interaction": {"RS": 1, "RP": -1, "NS": -1, "NP": 1},
    }
    for name, coeffs in expected.items():
        assert cfg.contrast(name).coefficients == coeffs
    assert cfg.parsed.contrasts.primary == "style_without_reason"
    assert set(cfg.parsed.contrasts.multiplicity.family) == set(expected) - {"interaction"}


def test_every_contrast_sums_to_zero_over_core_conditions(cfg):
    core = set(cfg.core_condition_ids())
    for name, spec in cfg.parsed.contrasts.core.items():
        assert sum(spec.coefficients.values()) == 0, name
        assert set(spec.coefficients) <= core, name


def test_interaction_is_the_difference_of_the_two_style_contrasts(cfg):
    """(RS - RP) - (NS - NP) must equal the stored interaction coefficients."""
    a = cfg.contrast("style_with_reason").coefficients
    b = cfg.contrast("style_without_reason").coefficients
    derived = {c: a.get(c, 0) - b.get(c, 0) for c in cfg.core_condition_ids()}
    assert derived == cfg.contrast("interaction").coefficients


def test_option_orders_map_display_labels_to_semantic_ids(cfg):
    orders = {(o["A"], o["B"]) for o in cfg.parsed.prompts.option_orders}
    assert orders == {("opt_1", "opt_2"), ("opt_2", "opt_1")}


def test_tau_is_the_logit_of_0_60(cfg):
    assert cfg.parsed.near_tie.probability_threshold == 0.60
    assert cfg.tau == pytest.approx(math.log(0.6 / 0.4), abs=1e-9)
    assert cfg.is_near_tie(0.2) is True
    assert cfg.is_near_tie(-0.2) is True     # near ties are symmetric
    assert cfg.is_near_tie(-0.9) is False
    assert cfg.parsed.near_tie.primary_analysis_excludes is False


def test_no_permitted_marker_is_matched_by_a_forbidden_pattern(cfg):
    """D7. Checked at load time too; asserted here so the intent is explicit."""
    for spec, pattern in cfg.compiled_forbidden():
        for marker in cfg.permitted_markers():
            for variant in (marker, marker.lower(), marker.upper(), marker.capitalize()):
                assert pattern.search(variant) is None, f"{spec.id} matched {variant!r}"


def test_every_forbidden_pattern_honours_its_fixtures(cfg):
    """The fixtures bind: a pattern cannot be weakened to resolve a collision,
    nor broadened until it swallows ordinary policy language."""
    for spec, pattern in cfg.compiled_forbidden():
        assert spec.positive and spec.negative, spec.id
        for example in spec.positive:
            assert pattern.search(example), f"{spec.id} missed positive {example!r}"
        for example in spec.negative:
            assert not pattern.search(example), f"{spec.id} caught negative {example!r}"


def test_severity_split(cfg):
    hard = cfg.compiled_forbidden("hard_fail")
    warn = cfg.compiled_forbidden("warning")
    assert len(hard) + len(warn) == len(cfg.forbidden_patterns()) == 25
    assert (len(hard), len(warn)) == (18, 7)
    # Hard failures are confined to unambiguous authority/evidence/consensus/pressure.
    assert {s.family for s, _ in hard} == {"authority", "evidence", "consensus", "pressure"}
    # Certainty is broad and false-positive prone, so it only ever warns.
    assert all(s.severity == "warning" for s in cfg.forbidden_patterns() if s.family == "certainty")
    assert cfg.raw["forbidden"]["case_insensitive"] is True


def test_scenario_groundable_terms_are_warnings_not_hard_failures(cfg):
    """A regex cannot tell scenario-grounded from external evidence, so
    'the permitting authorities', 'proven technology' and 'empirical
    performance' route to review rather than auto-rejecting."""
    by_id = {s.id: s for s in cfg.forbidden_patterns()}
    assert by_id["auth_bare_authorities"].severity == "warning"
    assert by_id["ev_proven"].severity == "warning"
    assert by_id["ev_empirical"].severity == "warning"


def test_unsupported_appeals_remain_hard_failures(cfg):
    by_id = {s.id: s for s in cfg.forbidden_patterns()}
    for pid in ("ev_studies", "ev_research", "auth_experts_verb", "con_consensus"):
        assert by_id[pid].severity == "hard_fail", pid


def test_speaker_credential_handles_multiple_modifiers(cfg):
    pat = {s.id: p for s, p in cfg.compiled_forbidden()}["auth_speaker_credential"]
    for text in ("As a senior energy economist, I would extend the plant.",
                 "As an experienced urban planner, I favour storage.",
                 "As a technology policy expert, I would reconsider.",
                 "as a grid engineer I disagree"):
        assert pat.search(text), text
    for text in ("As a result the analyst revised the forecast.",
                 "The plan was reviewed as a whole before the analyst signed off.",
                 "The grid engineer on duty filed the report."):
        assert not pat.search(text), text


def test_hard_failures_still_catch_what_they_are_for(cfg):
    must_be_caught = [
        "Studies show that storage is cheaper.",
        "Experts agree the plant should close.",
        "The consensus favours storage.",
        "Any reasonable person would switch.",
    ]
    patterns = [pat for _, pat in cfg.compiled_forbidden("hard_fail")]
    for text in must_be_caught:
        assert any(p.search(text) for p in patterns), f"nothing caught {text!r}"


def test_ordinary_policy_language_survives_the_hard_filter(cfg):
    """The screen must not reject valid scenario text."""
    benign = [
        "The reserve margin narrows by 4 percent in the second year.",
        "Because the storage tender closes in March, the schedule is tight.",
        "The turbine design is mature; consequently the outage is short.",
        "However, the emissions trajectory worsens under the extension.",
    ]
    patterns = [pat for _, pat in cfg.compiled_forbidden("hard_fail")]
    for text in benign:
        assert not any(p.search(text) for p in patterns), f"false positive on {text!r}"


def test_leakage_patterns_catch_labels_but_not_the_article(cfg):
    """D5. 'A' is also an English article, so bare letters are not rejected."""
    patterns = [pat for _, pat in cfg.compiled_leakage()]

    def hit(text: str) -> bool:
        return any(p.search(text) for p in patterns)

    assert hit("Option A is preferable here.")
    assert hit("Choice B leaves less slack.")
    assert hit("You should select A instead.")
    assert not hit("A grid operator faces a shortfall.")
    assert not hit("This implies a longer outage window.")


def test_marker_families_are_disjoint(cfg):
    seen: set[str] = set()
    for markers in cfg.marker_families().values():
        assert not (seen & set(markers))
        seen |= set(markers)


def test_marker_taxonomy(cfg):
    assert cfg.confirmatory_families() == [
        "premise_indicator",
        "conclusion_indicator",
        "metadiscursive_inference",
    ]
    assert cfg.exploratory_families() == ["concession_contrast"]
    assert set(cfg.marker_families("confirmatory")) == set(cfg.confirmatory_families())
    assert len(cfg.permitted_markers()) == 16


def test_although_is_not_in_the_inventory(cfg):
    """Removed: it forces clause restructuring, breaking exact sentence matching."""
    assert "although" not in cfg.permitted_markers()


def test_concession_is_exploratory_not_confirmatory(cfg):
    assert "concession_contrast" not in cfg.confirmatory_families()
    assert "however" not in [m for f in cfg.marker_families("confirmatory").values() for m in f]


def test_lexical_matching_may_not_stand_in_for_discourse_function(cfg):
    a = cfg.raw["markers"]["analysis"]
    assert a["lexical_match_establishes_discourse_function"] is False
    assert a["human_validation_authoritative"] is True
    assert a["pooling_rule"] == "only_if_directionally_compatible"


def test_sentence_matching_is_exact_and_semicolons_do_not_split(cfg):
    s = cfg.parsed.matching.sentences
    assert s.rule == "exact_equality"
    assert s.tolerance == 0
    assert s.semicolon_terminates_sentence is False


def test_word_tolerances(cfg):
    w = cfg.parsed.matching.words
    assert (w.ratio_warn, w.ratio_fail) == (1.10, 1.15)


def test_answer_token_ids_are_unverified_and_null(cfg):
    """D10. Nothing may claim single-token validity before Stage 5."""
    for tpl in cfg.parsed.prompts.templates.values():
        assert tpl.verification_status == "unverified"
        assert set(tpl.answer_token_ids) == {"A", "B"}
        assert all(v is None for v in tpl.answer_token_ids.values())


def test_annotation_includes_the_pragmatic_alternative_fields(cfg):
    additions = set(cfg.raw["annotation"]["rating_provenance"]["additions_v1"])
    assert {"perceived_speaker_commitment", "perceived_naturalness", "perceived_unstated_support"} <= additions


def test_provenance_block_is_complete(cfg):
    block = cfg.provenance_block()
    assert set(block) == {
        "config_version",
        "schema_version",
        "config_content_hash",
        "config_file_sha256",
        "config_status",
    }
    assert all(v is not None for v in block.values())
    assert len(block["config_content_hash"]) == 64


def test_config_hash_is_stable_under_source_reordering(cfg, tmp_path):
    reordered = {k: cfg.raw[k] for k in reversed(list(cfg.raw))}
    other = load_mutated(tmp_path, reordered)
    assert other.content_hash == cfg.content_hash
    assert other.file_sha256 != cfg.file_sha256   # bytes differ, meaning does not


# --- mutation tests: the loader must refuse these ----------------------------


def test_unknown_top_level_key_is_rejected(tmp_path, raw):
    raw["experimental_extras"] = {"foo": 1}
    with pytest.raises(Exception):
        load_mutated(tmp_path, raw)


def test_renaming_a_core_condition_is_rejected(tmp_path, raw):
    raw["conditions"]["core"]["NX"] = raw["conditions"]["core"].pop("NS")
    with pytest.raises(ConfigError, match="core conditions must be exactly"):
        load_mutated(tmp_path, raw)


def test_altering_a_core_cell_is_rejected(tmp_path, raw):
    raw["conditions"]["core"]["NS"]["reason"] = "present"
    with pytest.raises(ConfigError, match="core condition NS"):
        load_mutated(tmp_path, raw)


def test_reversing_a_core_contrast_is_rejected(tmp_path, raw):
    raw["contrasts"]["core"]["style_without_reason"]["coefficients"] = {"NS": -1, "NP": 1}
    with pytest.raises(ConfigError, match="must have coefficients"):
        load_mutated(tmp_path, raw)


def test_a_contrast_that_does_not_sum_to_zero_is_rejected(tmp_path, raw):
    raw["contrasts"]["core"]["style_without_reason"]["coefficients"] = {"NS": 1, "NP": 1}
    with pytest.raises(ConfigError, match="must have coefficients"):
        load_mutated(tmp_path, raw)


def test_an_unbalanced_extra_contrast_is_rejected(tmp_path, raw):
    """Sum-to-zero is enforced on every contrast, not only the frozen four."""
    raw["contrasts"]["core"]["interaction"]["coefficients"] = {"RS": 1, "RP": -1, "NS": -1, "NP": 2}
    with pytest.raises(ConfigError, match="must have coefficients"):
        load_mutated(tmp_path, raw)


def test_adding_a_diagnostic_condition_is_allowed(tmp_path, raw):
    """The registry must accept a later commitment-matched control ..."""
    raw["conditions"]["diagnostic"]["NC"] = {
        "reason": "absent",
        "style": "explicit",
        "gloss": "commitment-matched control",
    }
    cfg2 = load_mutated(tmp_path, raw)
    assert cfg2.diagnostic_condition_ids() == ["NC"]
    assert cfg2.core_condition_ids() == ["RS", "RP", "NS", "NP"]
    for spec in cfg2.parsed.contrasts.core.values():
        assert "NC" not in spec.coefficients


def test_diagnostic_condition_may_not_enter_a_core_contrast(tmp_path, raw):
    """... but it must never displace a core contrast."""
    raw["conditions"]["diagnostic"]["NC"] = {"reason": "absent", "style": "explicit"}
    raw["contrasts"]["core"]["style_without_reason"]["coefficients"] = {"NS": 1, "NC": -1}
    with pytest.raises(ConfigError, match="must have coefficients"):
        load_mutated(tmp_path, raw)


def test_diagnostic_id_colliding_with_a_core_id_is_rejected(tmp_path, raw):
    raw["conditions"]["diagnostic"]["NS"] = {"reason": "absent", "style": "explicit"}
    with pytest.raises(ConfigError, match="collide with core"):
        load_mutated(tmp_path, raw)


def test_dropping_a_contrast_from_the_holm_family_is_rejected(tmp_path, raw):
    raw["contrasts"]["multiplicity"]["family"] = ["style_without_reason"]
    with pytest.raises(ConfigError, match="Holm family"):
        load_mutated(tmp_path, raw)


def test_tau_inconsistent_with_its_probability_is_rejected(tmp_path, raw):
    raw["near_tie"]["tau_logit"] = 0.5
    with pytest.raises(ConfigError, match="does not match"):
        load_mutated(tmp_path, raw)


def test_excluding_near_ties_from_the_primary_analysis_is_rejected(tmp_path, raw):
    raw["near_tie"]["primary_analysis_excludes"] = True
    with pytest.raises(ConfigError, match="full sample remains primary"):
        load_mutated(tmp_path, raw)


def test_relaxing_sentence_matching_is_rejected(tmp_path, raw):
    """D1: exact equality is a hard requirement, not a covariate."""
    raw["matching"]["sentences"]["tolerance"] = 1
    with pytest.raises(ConfigError, match="exact sentence-count equality"):
        load_mutated(tmp_path, raw)


def test_inverted_word_tolerances_are_rejected(tmp_path, raw):
    raw["matching"]["words"]["ratio_warn"] = 1.20
    with pytest.raises(ConfigError, match="ratio_warn <= ratio_fail"):
        load_mutated(tmp_path, raw)


def test_token_ids_without_verification_are_rejected(tmp_path, raw):
    """D10: no field may imply single-token validity before Stage 5 verifies it."""
    raw["prompts"]["templates"]["base_scaffold_v1"]["answer_token_ids"]["A"] = 32
    with pytest.raises(ConfigError, match="must stay null"):
        load_mutated(tmp_path, raw)


def test_verified_template_must_pin_everything(tmp_path, raw):
    raw["prompts"]["templates"]["base_scaffold_v1"]["verification_status"] = "verified"
    with pytest.raises(ConfigError, match="must pin token ids"):
        load_mutated(tmp_path, raw)


def test_split_fractions_must_sum_to_one(tmp_path, raw):
    raw["splits"]["fractions"]["test"] = 0.3
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_mutated(tmp_path, raw)


def test_split_is_grouped_on_decision_and_stratified_only_by_domain(cfg):
    sp = cfg.raw["splits"]
    assert sp["grouping_unit"] == "decision_id"
    assert sp["stratify_by"] == ["domain"]
    assert sp["decisions"] == {"train": 36, "validation": 12, "test": 12}
    assert sum(sp["decisions"].values()) == cfg.raw["corpus"]["decisions_full"]
    assert sp["marker_family_balance"]["is_stratification_key"] is False


def test_marker_family_may_not_become_a_stratification_key(tmp_path, raw):
    """A decision carries several (scenario, direction) marker-family groups,
    so marker family is not a decision-level label."""
    raw["splits"]["stratify_by"] = ["domain", "marker_family"]
    with pytest.raises(ConfigError, match="only decision-level labels"):
        load_mutated(tmp_path, raw)


def test_declared_split_counts_must_match_the_fractions(tmp_path, raw):
    raw["splits"]["decisions"] = {"train": 40, "validation": 12, "test": 8}
    with pytest.raises(ConfigError, match="decisions is"):
        load_mutated(tmp_path, raw)


def test_probe_schemes_keep_the_four_cell_group_together(cfg):
    schemes = cfg.raw["probe_evaluation"]["schemes"]
    for name in ("held_out_marker_family", "leave_one_marker_out",
                 "leave_one_realization_template_out"):
        assert schemes[name]["unit_kept_together"] == "group"
    assert schemes["leave_one_domain_out"]["folds"] == 3


def test_every_probe_scheme_is_decision_disjoint(cfg):
    """Keeping the four-cell group intact is necessary but insufficient: a
    sibling group from the same decision would otherwise leak into training."""
    for name, scheme in cfg.raw["probe_evaluation"]["schemes"].items():
        assert scheme["no_decision_overlap"] is True, name
        assert scheme["grouping_unit"] == "decision_id", name


def test_probe_invariants_state_the_held_out_family_rule(cfg):
    text = " ".join(cfg.raw["probe_evaluation"]["invariants"]).lower()
    assert "no decision_id appears on both" in text
    assert "disjoint evaluation decisions" in text


def test_held_out_schemes_declare_a_minimum_support(cfg):
    """A nominal held-out-marker analysis must not run on one or two decisions."""
    alloc = cfg.raw["markers"]["allocation"]
    assert alloc["min_decisions_per_leave_one_marker_test"] >= 2
    assert alloc["min_decisions_per_held_out_family_test"] >= 2
    for name in ("held_out_marker_family", "leave_one_marker_out",
                 "leave_one_realization_template_out"):
        assert cfg.raw["probe_evaluation"]["schemes"][name]["min_decisions_ref"].startswith(
            "markers.allocation.")


def test_marker_allocation_constraints(cfg):
    alloc = cfg.raw["markers"]["allocation"]
    assert alloc["min_decisions_per_marker"] >= 2
    assert 1 / 3 <= alloc["max_share_within_one_domain"] <= 1.0
    assert 0 < alloc["max_share_within_one_supported_option"] <= 1.0
    assert alloc["confirmatory_families_in_every_split"] is True
    assert alloc["report_as_corpus_statistic"] is True


def test_all_four_cells_share_one_marker_realization(cfg):
    """RS and NS must share the marker STRING and realization, not merely the
    family, or RS - NS would also contrast 'therefore' against 'because'."""
    a = cfg.raw["markers"]["assignment"]
    assert a["group_level_fields"] == ["marker_family", "marker_string", "marker_realization_id"]
    assert a["same_realization_across_cells"] is True
    inh = a["cell_inheritance"]
    assert set(inh) == {"RS", "RP", "NS", "NP"}
    assert inh["RS"]["markers_present"] is True and inh["NS"]["markers_present"] is True
    assert inh["RP"]["markers_present"] is False and inh["NP"]["markers_present"] is False
    assert inh["RS"]["instantiates"] == inh["NS"]["instantiates"] == "marker_realization"
    assert inh["RP"]["instantiates"] == inh["NP"]["instantiates"] == "paired_plain_transformation"


def test_a_configured_marker_does_not_prove_the_text_is_reason_free(cfg):
    assert "reason-free" in cfg.raw["markers"]["assignment"]["pilot_gate"]
    assert any("reason-free" in s for s in cfg.raw["annotation"]["human_review_required"])


def test_annotations_are_stored_at_three_levels(cfg):
    levels = cfg.raw["annotation"]["levels"]
    assert set(levels) == {"item", "pair", "scenario"}
    assert "proposition_preservation" in levels["pair"]["ratings"]
    assert levels["pair"]["pairs"] == [["RS", "RP"], ["NS", "NP"]]
    assert set(levels["scenario"]["ratings"]) == {
        "option_feasibility", "option_non_dominance", "normative_underdetermination"}
    # a judgement lives at exactly one level
    everywhere = [r for spec in levels.values() for r in spec["ratings"]]
    assert len(everywhere) == len(set(everywhere)) == 15
    assert cfg.raw["annotation"]["denormalization"][
        "duplicate_pair_and_scenario_judgements_per_item"] is False


def test_only_item_level_annotations_are_keyed_by_condition(cfg):
    levels = cfg.raw["annotation"]["levels"]
    assert "condition" in levels["item"]["keys"]
    assert "condition" not in levels["pair"]["keys"]
    assert "condition" not in levels["scenario"]["keys"]


def test_a_probe_scheme_that_splits_a_group_is_rejected(tmp_path, raw):
    raw["probe_evaluation"]["schemes"]["held_out_marker_family"]["unit_kept_together"] = "row"
    with pytest.raises(ConfigError, match="four-cell group together"):
        load_mutated(tmp_path, raw)


def test_a_probe_scheme_without_decision_disjointness_is_rejected(tmp_path, raw):
    raw["probe_evaluation"]["schemes"]["leave_one_marker_out"]["no_decision_overlap"] = False
    with pytest.raises(ConfigError, match="no_decision_overlap"):
        load_mutated(tmp_path, raw)


def test_a_probe_scheme_grouped_below_decision_is_rejected(tmp_path, raw):
    raw["probe_evaluation"]["schemes"]["held_out_marker_family"]["grouping_unit"] = "scenario_id"
    with pytest.raises(ConfigError, match="grouped on decision_id"):
        load_mutated(tmp_path, raw)


def test_a_nominal_held_out_marker_test_is_rejected(tmp_path, raw):
    raw["markers"]["allocation"]["min_decisions_per_leave_one_marker_test"] = 1
    with pytest.raises(ConfigError, match="not an evaluation"):
        load_mutated(tmp_path, raw)


def test_dropping_the_realization_from_the_group_key_is_rejected(tmp_path, raw):
    raw["markers"]["assignment"]["group_level_fields"] = ["marker_family"]
    with pytest.raises(ConfigError, match="realization group is"):
        load_mutated(tmp_path, raw)


def test_letting_cells_use_different_realizations_is_rejected(tmp_path, raw):
    raw["markers"]["assignment"]["same_realization_across_cells"] = False
    with pytest.raises(ConfigError, match="same marker realization"):
        load_mutated(tmp_path, raw)


def test_a_plain_cell_claiming_markers_is_rejected(tmp_path, raw):
    raw["markers"]["assignment"]["cell_inheritance"]["NP"]["markers_present"] = True
    with pytest.raises(ConfigError, match="plain cell NP"):
        load_mutated(tmp_path, raw)


def test_a_rating_declared_at_two_levels_is_rejected(tmp_path, raw):
    raw["annotation"]["levels"]["item"]["ratings"].append("proposition_preservation")
    with pytest.raises(ConfigError, match="declared at both"):
        load_mutated(tmp_path, raw)


def test_duplicating_pair_judgements_onto_items_is_rejected(tmp_path, raw):
    raw["annotation"]["denormalization"]["duplicate_pair_and_scenario_judgements_per_item"] = True
    with pytest.raises(ConfigError, match="not duplicated onto every item"):
        load_mutated(tmp_path, raw)


def test_a_proposition_pair_crossing_the_reason_factor_is_rejected(tmp_path, raw):
    """RS/NS differ in reason content, so they cannot be a preservation pair."""
    raw["annotation"]["levels"]["pair"]["pairs"] = [["RS", "NS"], ["RP", "NP"]]
    with pytest.raises(ConfigError, match="crosses it"):
        load_mutated(tmp_path, raw)


def test_a_rating_without_provenance_is_rejected(tmp_path, raw):
    raw["annotation"]["levels"]["item"]["ratings"].append("perceived_vibes")
    with pytest.raises(ConfigError, match="must have provenance"):
        load_mutated(tmp_path, raw)


def test_case_sensitive_forbidden_matching_is_rejected(tmp_path, raw):
    raw["forbidden"]["case_insensitive"] = False
    with pytest.raises(ConfigError, match="case-insensitively"):
        load_mutated(tmp_path, raw)


def test_corpus_arithmetic_is_checked(tmp_path, raw):
    raw["corpus"]["texts_full"] = 900
    with pytest.raises(ConfigError, match="texts_full must be 960"):
        load_mutated(tmp_path, raw)


def test_pilot_must_stay_domain_stratified(tmp_path, raw):
    raw["corpus"]["decisions_pilot"] = 13
    with pytest.raises(ConfigError, match="decisions_per_domain_pilot"):
        load_mutated(tmp_path, raw)


def test_a_forbidden_pattern_colliding_with_a_marker_is_rejected(tmp_path, raw):
    """D7 in the direction that would silently destroy the style manipulation."""
    raw["forbidden"]["patterns"].append({
        "id": "cert_therefore", "family": "certainty", "severity": "warning",
        "pattern": r"\btherefore\b",
        "positive": ["Therefore the plant should close."],
        "negative": ["The plant should close."],
    })
    with pytest.raises(ConfigError, match="matches permitted marker"):
        load_mutated(tmp_path, raw)


def test_weakening_a_forbidden_pattern_breaks_its_own_fixture(tmp_path, raw):
    by_id = {p["id"]: p for p in raw["forbidden"]["patterns"]}
    by_id["ev_studies"]["pattern"] = r"\bstudies\s+demonstrate\b"
    with pytest.raises(ConfigError, match="no longer catches its own positive"):
        load_mutated(tmp_path, raw)


def test_overbroad_forbidden_pattern_breaks_its_negative_fixture(tmp_path, raw):
    by_id = {p["id"]: p for p in raw["forbidden"]["patterns"]}
    by_id["ev_studies"]["pattern"] = r"\bstud"
    with pytest.raises(ConfigError, match="matches its negative fixture"):
        load_mutated(tmp_path, raw)


def test_duplicate_forbidden_ids_are_rejected(tmp_path, raw):
    raw["forbidden"]["patterns"].append(dict(raw["forbidden"]["patterns"][0]))
    with pytest.raises(ConfigError, match="ids must be unique"):
        load_mutated(tmp_path, raw)


def test_promoting_concession_to_confirmatory_is_rejected(tmp_path, raw):
    raw["markers"]["roles"]["confirmatory"].append("concession_contrast")
    with pytest.raises(ConfigError, match="confirmatory role must name exactly"):
        load_mutated(tmp_path, raw)


def test_a_marker_in_two_families_is_rejected(tmp_path, raw):
    raw["markers"]["exploratory_families"]["concession_contrast"].append("thus")
    with pytest.raises(ConfigError, match="appears in both"):
        load_mutated(tmp_path, raw)


def test_treating_lexical_match_as_discourse_function_is_rejected(tmp_path, raw):
    raw["markers"]["analysis"]["lexical_match_establishes_discourse_function"] = True
    with pytest.raises(ConfigError, match="never be treated as evidence"):
        load_mutated(tmp_path, raw)


def test_filename_must_encode_the_config_version(tmp_path, raw):
    with pytest.raises(ConfigError, match="filename stem"):
        load_mutated(tmp_path, raw, name="experiment.yaml")


def test_embeddings_may_not_be_a_negative_block_index(tmp_path, raw):
    raw["mechanistic"]["layers"]["embeddings_name"] = "-1"
    with pytest.raises(ConfigError, match="never a negative block index"):
        load_mutated(tmp_path, raw)


def test_mechanistic_subset_cannot_reference_a_non_core_condition(tmp_path, raw):
    raw["mechanistic"]["subset"]["contrasts"].append(["RS", "NC"])
    with pytest.raises(ConfigError, match="non-core condition"):
        load_mutated(tmp_path, raw)


# ===========================================================================
# Config v2 (Implementation Stage 2a)
# ===========================================================================


@pytest.fixture(scope="module")
def cfg2():
    return load_config(CONFIG_V2_PATH)


@pytest.fixture
def raw2(cfg2):
    return copy.deepcopy(cfg2.raw)


def load_mutated_v2(tmp_path: Path, raw: dict):
    return load_mutated(tmp_path, raw, name="experiment_v2.yaml")


def test_v1_is_untouched_by_v2(cfg):
    """v2 is a new file; v1 keeps the hash it was approved and committed with."""
    assert cfg.content_hash.startswith("fef78db7aa64")   # as committed at bd7c897
    assert cfg.config_version == "v1"
    assert cfg.raw.get("segmentation") is None


def test_v2_loads_and_is_a_distinct_artefact(cfg, cfg2):
    assert cfg2.config_version == "v2"
    assert cfg2.content_hash != cfg.content_hash


def test_v2_pins_the_segmenter(cfg2):
    seg = cfg2.raw["segmentation"]
    assert seg["library"] == "pysbd"
    assert seg["version"] and seg["version_spec"]
    assert seg["clean"] is False
    assert seg["authority"] == "machine_count_authoritative"
    assert seg["human_override"]["permitted"] is False
    assert seg["human_override"]["requires_recorded_annotation"] is True


def test_v2_enforces_word_ratio_on_full_text_and_body(cfg2):
    assert cfg2.parsed.matching.words.applies_to == ["full_text", "body"]


def test_v2_realization_registry_covers_every_marker_family(cfg2):
    registry = cfg2.marker_realizations()
    assert len(registry) == 8
    assert {e["family"] for e in registry.values()} == set(cfg2.marker_families())
    assert cfg2.raw["markers"]["realization"]["recorded_per"] == "group"


def test_v2_model_selection_is_not_frozen(cfg2):
    """No model identity is committed until the compatibility test."""
    models = cfg2.raw["models"]
    assert models["selection_status"] == "unfrozen"
    for variant in ("base", "instruct"):
        assert models[variant]["repo_id"] is None
        assert models[variant]["revision"] is None


def test_no_vendor_model_identity_appears_in_v2(cfg2):
    """Model choice is frozen after a compatibility test, not assumed here."""
    text = CONFIG_V2_PATH.read_text().lower()
    for name in ("llama", "gemma", "qwen", "mistral", "hookedtransformer", "transformerbridge"):
        assert name not in text


def test_v2_source_references_are_an_open_extensible_list(cfg2):
    src = cfg2.raw["corpus_provenance"]["source_references"]
    assert src["cardinality"] == "zero_or_more"
    assert src["open_vocabulary"] is True
    assert src["constructed_allows_empty_list"] is True
    assert "constructed" in src["scenario_level_type_values"]
    for field in ("dataset_name", "dataset_version", "source_item_id",
                  "source_url", "access_date", "reuse_licence"):
        assert field in src["fields"]
    assert src["fields"]["dataset_name"]["required"] is True


def test_v2_keeps_generation_metadata_separate_from_provenance(cfg2):
    gen = cfg2.raw["corpus_provenance"]["generation_metadata"]
    src = cfg2.raw["corpus_provenance"]["source_references"]
    assert gen["separate_from_source_references"] is True
    for field in ("generator_model", "generator_model_revision", "prompt_hash",
                  "generation_parameters", "seed", "generated_at"):
        assert field in gen["fields"]
    assert set(gen["fields"]) & set(src["fields"]) == set()


def test_v2_adds_support_direction_confirmed(cfg2):
    item = cfg2.raw["annotation"]["levels"]["item"]["ratings"]
    assert "support_direction_confirmed" in item
    assert len(item) == 12
    assert cfg2.raw["annotation"]["rating_provenance"]["additions_v2"] == ["support_direction_confirmed"]


# --- v2 mutation tests ------------------------------------------------------


def test_a_v2_section_may_not_be_backported_into_v1(tmp_path, raw, cfg2):
    """Adding a v2 section to v1 in place is exactly what the bump exists to stop."""
    raw["segmentation"] = copy.deepcopy(cfg2.raw["segmentation"])
    with pytest.raises(ConfigError, match="introduced in v2"):
        load_mutated(tmp_path, raw)


def test_v2_without_a_segmentation_block_is_rejected(tmp_path, raw2):
    del raw2["segmentation"]
    with pytest.raises(ConfigError, match="must declare 'segmentation'"):
        load_mutated_v2(tmp_path, raw2)


def test_dropping_the_semicolon_guarantee_is_rejected(tmp_path, raw2):
    raw2["segmentation"]["guarantees"]["semicolon_does_not_terminate_sentence"] = False
    with pytest.raises(ConfigError, match="explicit framing inside one sentence"):
        load_mutated_v2(tmp_path, raw2)


def test_a_silent_segmentation_override_is_rejected(tmp_path, raw2):
    raw2["segmentation"]["human_override"]["requires_recorded_annotation"] = False
    with pytest.raises(ConfigError, match="never silent"):
        load_mutated_v2(tmp_path, raw2)


def test_letting_the_segmenter_rewrite_text_is_rejected(tmp_path, raw2):
    raw2["segmentation"]["clean"] = True
    with pytest.raises(ConfigError, match="never rewrite"):
        load_mutated_v2(tmp_path, raw2)


def test_measuring_the_word_ratio_on_full_text_alone_is_rejected(tmp_path, raw2):
    raw2["matching"]["words"]["applies_to"] = ["full_text"]
    with pytest.raises(ConfigError, match="cannot dilute"):
        load_mutated_v2(tmp_path, raw2)


def test_a_realization_for_an_unknown_family_is_rejected(tmp_path, raw2):
    raw2["markers"]["realization"]["registry"]["sentence_initial_hedge_v1"] = {
        "family": "hedging", "position": "sentence_initial", "description": "x"}
    with pytest.raises(ConfigError, match="unknown marker family"):
        load_mutated_v2(tmp_path, raw2)


def test_a_family_with_no_realization_is_rejected(tmp_path, raw2):
    registry = raw2["markers"]["realization"]["registry"]
    for rid in [r for r, e in registry.items() if e["family"] == "concession_contrast"]:
        del registry[rid]
    with pytest.raises(ConfigError, match="no realization"):
        load_mutated_v2(tmp_path, raw2)


def test_duplicating_the_realization_per_cell_is_rejected(tmp_path, raw2):
    raw2["markers"]["realization"]["recorded_per"] = "styled_cell"
    with pytest.raises(ConfigError, match="never duplicated per cell"):
        load_mutated_v2(tmp_path, raw2)


def test_claiming_a_frozen_model_selection_without_pinning_it_is_rejected(tmp_path, raw2):
    raw2["models"]["selection_status"] = "frozen"
    with pytest.raises(ConfigError, match="must pin repo_id and revision"):
        load_mutated_v2(tmp_path, raw2)


def test_pinning_a_model_while_selection_is_unfrozen_is_rejected(tmp_path, raw2):
    raw2["models"]["base"]["repo_id"] = "some-org/some-model"
    with pytest.raises(ConfigError, match="must stay null"):
        load_mutated_v2(tmp_path, raw2)


def test_closing_the_source_reference_vocabulary_is_rejected(tmp_path, raw2):
    raw2["corpus_provenance"]["source_references"]["open_vocabulary"] = False
    with pytest.raises(ConfigError, match="not an enum"):
        load_mutated_v2(tmp_path, raw2)


def test_forbidding_a_constructed_scenario_from_citing_nothing_is_rejected(tmp_path, raw2):
    raw2["corpus_provenance"]["source_references"]["constructed_allows_empty_list"] = False
    with pytest.raises(ConfigError, match="may cite no source"):
        load_mutated_v2(tmp_path, raw2)


def test_conflating_generation_metadata_with_provenance_is_rejected(tmp_path, raw2):
    raw2["corpus_provenance"]["generation_metadata"]["fields"]["dataset_name"] = {"required": False}
    with pytest.raises(ConfigError, match="must not overlap"):
        load_mutated_v2(tmp_path, raw2)


def test_a_rating_added_without_provenance_is_rejected_in_v2(tmp_path, raw2):
    raw2["annotation"]["levels"]["item"]["ratings"].append("perceived_vibes")
    with pytest.raises(ConfigError, match="missing provenance"):
        load_mutated_v2(tmp_path, raw2)


def test_the_latest_config_tracks_the_current_research_plan():
    """Catch silent drift: an edit to the plan must be followed by a config
    bump. Superseded versions keep the hash of the plan revision they were
    frozen against and are not checked."""
    cfg2 = load_config(LATEST_PATH)
    plan = CONFIGS.parent / cfg2.raw["provenance"]["research_plan"]
    assert cfg2.raw["provenance"]["research_plan_sha256"] == file_sha256(plan), (
        "Research_Plan_v6.md has changed since this config recorded its hash. "
        "Update the provenance and regenerate any artefacts that embed the "
        "config hash (data/fixtures/tiny_corpus.jsonl)."
    )


# ===========================================================================
# Historical configuration integrity
#
# Every superseded version must stay loadable and byte-stable. A config that
# produced an artefact is a record of what that artefact was made under; if it
# drifts, the provenance chain of everything derived from it is broken.
# ===========================================================================

#: Content hashes as frozen. Changing one means a config was edited in place.
FROZEN_CONFIG_HASHES = {
    "v1": "fef78db7aa64bb2be51b23b9283fd00ac8c485d1c5db511d6241f54595199bd4",
    "v2": "357ec0c9d9cce6c73fe6f0da3df74e3df88f29edefb0805abbb9a83030fd9ce4",
    "v3": "dad39605ac5f0bfdd827783f045387e0254a386d15431af1d98339ddd0d6f344",
}


@pytest.mark.parametrize("version", sorted(FROZEN_CONFIG_HASHES))
def test_every_historical_config_still_loads(version):
    cfg = load_config(CONFIGS / f"experiment_{version}.yaml")
    assert cfg.config_version == version
    assert cfg.core_condition_ids() == ["RS", "RP", "NS", "NP"]


@pytest.mark.parametrize("version", sorted(FROZEN_CONFIG_HASHES))
def test_every_historical_config_is_unchanged(version):
    cfg = load_config(CONFIGS / f"experiment_{version}.yaml")
    assert cfg.content_hash == FROZEN_CONFIG_HASHES[version], (
        f"experiment_{version}.yaml has changed. A config that produced an "
        f"artefact must never be edited in place — create a new version instead."
    )


def test_no_config_version_is_missing_from_the_frozen_record():
    on_disk = {p.stem.split("_")[1] for p in CONFIGS.glob("experiment_v*.yaml")}
    assert on_disk == set(FROZEN_CONFIG_HASHES), (
        "a new config version exists but is not pinned in FROZEN_CONFIG_HASHES"
    )


def test_latest_config_path_resolves_to_the_newest_version():
    assert latest_config_path(CONFIGS) == CONFIGS / "experiment_v3.yaml"


# ===========================================================================
# Artefact-generating commands must name their configuration
# ===========================================================================


def test_artefact_scripts_never_default_to_the_latest_config():
    """`latest_config_path` is a development convenience. A command that writes
    artefacts recording a config hash must take an explicit --config path."""
    scripts = CONFIGS.parent / "scripts"
    for path in sorted(scripts.glob("*.py")):
        assert "latest_config_path" not in path.read_text(), (
            f"{path.name} resolves the config implicitly; require --config instead"
        )


@pytest.mark.parametrize("script", ["build_review_export", "build_tiny_corpus"])
def test_artefact_scripts_require_an_explicit_config(script):
    source = (CONFIGS.parent / "scripts" / f"{script}.py").read_text()
    assert 'add_argument("--config", required=True' in source, script
