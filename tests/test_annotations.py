"""Annotation records: three levels, and honest blinding.

A blinded annotator must never be asked a question whose phrasing reveals the
design, and must be able to say "unclear". Unblinding compares the answer with
the hidden key rather than assuming it.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from reasonstyle.corpus.annotations import (
    AnnotationError,
    BlindItemKey,
    BlindItemResponse,
    BlindPairKey,
    BlindPairResponse,
    ItemAnnotation,
    PairAnnotation,
    ScenarioAnnotation,
    load_annotations,
    save_annotations,
    unblind_items,
    unblind_pairs,
)

HASHES = {"config_content_hash": "a" * 64, "corpus_content_hash": "b" * 64}
RATINGS = dict(substantive_support=4, perceived_reasoning_style=5,
               perceived_speaker_commitment=4, perceived_unstated_support=2,
               perceived_naturalness=4, confidence=3, pressure=2, politeness=4,
               authority=1, credibility=3)


def item(**kw):
    base = dict(scenario_id="fixture_001_v1", supported_option="opt_1", condition="NS",
                annotator_id="curator", annotated_at=date(2026, 9, 9), round="full_review",
                support_direction_answer="opt_1", support_direction_confirmed=True,
                supplies_premise=False, **HASHES, **RATINGS)
    return ItemAnnotation(**{**base, **kw})


def response(**kw):
    base = dict(blind_id="i0001", annotator_id="annotator_1", annotated_at=date(2026, 9, 9),
                supports_option="P", supplies_premise=False, **RATINGS)
    return BlindItemResponse(**{**base, **kw})


KEY = BlindItemKey(blind_id="i0001", scenario_id="fixture_001_v1", supported_option="opt_1",
                   condition="NS", option_labels={"P": "opt_1", "Q": "opt_2"})


# --- levels stay separate ---------------------------------------------------


def test_the_three_levels_have_different_keys():
    assert "condition" in ItemAnnotation.model_fields
    assert "condition" not in PairAnnotation.model_fields
    assert "condition" not in ScenarioAnnotation.model_fields
    assert "supported_option" not in ScenarioAnnotation.model_fields
    assert "pair_id" in PairAnnotation.model_fields


def test_proposition_preservation_lives_only_at_pair_level():
    assert "proposition_preservation" in PairAnnotation.model_fields
    assert "proposition_preservation" not in ItemAnnotation.model_fields


def test_scenario_validity_lives_only_at_scenario_level():
    for field in ("option_feasibility", "option_non_dominance", "normative_underdetermination"):
        assert field in ScenarioAnnotation.model_fields
        assert field not in ItemAnnotation.model_fields


# --- derived judgements -----------------------------------------------------


def test_no_reason_integrity_is_derived_for_no_reason_cells_only():
    assert item(condition="NS", supplies_premise=False).no_reason_integrity is True
    assert item(condition="NP", supplies_premise=True).no_reason_integrity is False
    assert item(condition="RS", supplies_premise=True).no_reason_integrity is None


def test_direction_confirmation_must_match_the_answer():
    with pytest.raises(ValidationError, match="support_direction_confirmed must equal"):
        item(support_direction_answer="opt_2", support_direction_confirmed=True)
    assert item(support_direction_answer="opt_2", support_direction_confirmed=False)
    assert item(support_direction_answer="unclear", support_direction_confirmed=False)


def test_ratings_are_bounded():
    with pytest.raises(ValidationError):
        item(confidence=6)
    with pytest.raises(ValidationError):
        item(confidence=0)


# --- blinding ---------------------------------------------------------------


def test_a_blinded_response_carries_no_condition_or_intended_option():
    fields = set(BlindItemResponse.model_fields)
    for leak in ("condition", "supported_option", "scenario_id",
                 "marker_family", "marker_string"):
        assert leak not in fields


def test_a_blinded_response_may_answer_unclear():
    assert response(supports_option="unclear").supports_option == "unclear"


def test_unblinding_compares_the_answer_with_the_key():
    correct = unblind_items([response(supports_option="P")], [KEY], **HASHES)[0]
    assert correct.support_direction_answer == "opt_1"
    assert correct.support_direction_confirmed is True
    assert correct.condition == "NS" and correct.round == "reliability"
    assert correct.blind_id == "i0001"

    wrong = unblind_items([response(supports_option="Q")], [KEY], **HASHES)[0]
    assert wrong.support_direction_answer == "opt_2"
    assert wrong.support_direction_confirmed is False


def test_unblinding_respects_a_flipped_label_mapping():
    """P and Q are randomised per item, so the mapping must come from the key."""
    flipped = KEY.model_copy(update={"option_labels": {"P": "opt_2", "Q": "opt_1"}})
    result = unblind_items([response(supports_option="P")], [flipped], **HASHES)[0]
    assert result.support_direction_answer == "opt_2"
    assert result.support_direction_confirmed is False


def test_unclear_survives_unblinding():
    result = unblind_items([response(supports_option="unclear")], [KEY], **HASHES)[0]
    assert result.support_direction_answer == "unclear"
    assert result.support_direction_confirmed is False


def test_unblinding_without_a_key_entry_fails_loudly():
    with pytest.raises(AnnotationError, match="no key entry"):
        unblind_items([response(blind_id="i9999")], [KEY], **HASHES)


def test_a_duplicate_blind_id_in_the_key_fails():
    with pytest.raises(AnnotationError, match="duplicate blind_id"):
        unblind_items([response()], [KEY, KEY], **HASHES)


def test_option_labels_must_cover_both_semantic_options():
    with pytest.raises(ValidationError, match="P and Q"):
        BlindItemKey(blind_id="i1", scenario_id="s_v1", supported_option="opt_1",
                     condition="RS", option_labels={"P": "opt_1", "Q": "opt_1"})


def test_pair_unblinding_recovers_the_pair_id():
    key = BlindPairKey(blind_id="p0001", scenario_id="fixture_001_v1",
                       supported_option="opt_2", pair_id="NS_NP",
                       side_conditions={"1": "NP", "2": "NS"})
    result = unblind_pairs([BlindPairResponse(blind_id="p0001", annotator_id="a1",
                                              annotated_at=date(2026, 9, 9),
                                              proposition_preservation=5)],
                           [key], **HASHES)[0]
    assert result.pair_id == "NS_NP" and result.round == "reliability"


# --- storage ----------------------------------------------------------------


def test_annotations_round_trip_through_jsonl(tmp_path):
    path = save_annotations([item(), item(condition="NP")], tmp_path / "item.jsonl")
    assert load_annotations(path, ItemAnnotation) == [item(), item(condition="NP")]


def test_a_missing_annotation_file_is_empty_not_an_error(tmp_path):
    assert load_annotations(tmp_path / "absent.jsonl", ItemAnnotation) == []


def test_a_bad_annotation_line_reports_its_number(tmp_path):
    path = tmp_path / "item.jsonl"
    path.write_text('{"nope": 1}\n')
    with pytest.raises(AnnotationError, match=r"item\.jsonl:1"):
        load_annotations(path, ItemAnnotation)


def test_annotations_are_immutable():
    with pytest.raises(ValidationError):
        item().confidence = 5      # type: ignore[misc]
