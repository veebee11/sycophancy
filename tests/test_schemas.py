"""Scenario schema: structural consistency, inheritance, and JSONL round-trip.

The fixture these tests load is **machine-valid**: structurally sound and
compliant with the lexical corpus rules. It is not "valid" in the full sense -
substantive support, no-reason integrity, proposition preservation, naturalness
and pragmatic commitment are human judgements no validator can make.

These tests check *shape*. Nothing here asserts that a reason is relevant,
valid or genuinely supporting — those are human judgements recorded in the
annotation tables, and the corpus rules that can be machine-checked are the
validator's job (Implementation Stage 2c).
"""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from reasonstyle.config import load_config
from reasonstyle.corpus import CorpusError, corpus_content_hash, load_corpus, save_corpus
from reasonstyle.corpus.store import dumps_record
from reasonstyle.corpus.schemas import (
    CORE_CONDITIONS,
    Cell,
    DirectionBlock,
    GenerationMetadata,
    Measurements,
    ScenarioRecord,
    SegmenterRef,
    SourceReference,
    ValidationStatus,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "corpus.jsonl"
CONFIG_V2 = ROOT / "configs" / "experiment.yaml"


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(FIXTURE)


@pytest.fixture
def payload(corpus):
    return json.loads(dumps_record(corpus[0]))


# --- the fixture's declared shape -------------------------------------------


def test_fixture_is_one_decision_two_variants_sixteen_texts(corpus):
    assert len(corpus) == 2
    assert {r.decision_id for r in corpus} == {"fixture_001"}
    assert sorted(r.variant_id for r in corpus) == [1, 2]
    assert [r.scenario_id for r in corpus] == ["fixture_001_v1", "fixture_001_v2"]
    assert all(r.counterargument_count == 8 for r in corpus)
    assert sum(r.counterargument_count for r in corpus) == 16


def test_every_scenario_has_both_directions_and_four_cells(corpus):
    for record in corpus:
        assert set(record.counterarguments) == {"opt_1", "opt_2"}
        for option, block in record.counterarguments.items():
            assert block.supported_option == option
            assert set(block.cells) == set(CORE_CONDITIONS)


def test_semantic_options_are_the_only_identities(corpus):
    """No display letter reaches stored data."""
    text = FIXTURE.read_text()
    assert '"opt_1"' in text and '"opt_2"' in text
    for record in corpus:
        assert set(record.options) == {"opt_1", "opt_2"}
        assert record.opposing_option("opt_1") == "opt_2"
        assert record.opposing_option("opt_2") == "opt_1"


def test_fixture_exercises_every_marker_family_and_realization(corpus):
    """Four families, four distinct realizations, one of each per block."""
    cfg = load_config(CONFIG_V2)
    blocks = [b for r in corpus for b in r.counterarguments.values()]
    assert {b.marker_family for b in blocks} == set(cfg.marker_families())
    assert len({b.marker_realization_id for b in blocks}) == 4
    registry = cfg.marker_realizations()
    for block in blocks:
        assert registry[block.marker_realization_id]["family"] == block.marker_family
        assert block.marker_string in cfg.marker_families()[block.marker_family]


def test_family_ids_and_realization_ids_are_distinct_namespaces(corpus):
    """`premise_indicator` etc. are marker FAMILY ids and must match the frozen
    config families. `clause_initial_premise_v1` etc. are REALIZATION ids. The
    two namespaces never overlap, so a report label cannot be misread."""
    cfg = load_config(CONFIG_V2)
    families = set(cfg.marker_families())
    realizations = set(cfg.marker_realizations())

    assert families == {"premise_indicator", "conclusion_indicator",
                        "metadiscursive_inference", "concession_contrast"}
    assert not (families & realizations)
    for block in (b for r in corpus for b in r.counterarguments.values()):
        assert block.marker_family in families
        assert block.marker_realization_id in realizations
        assert block.marker_realization_id not in families
        assert block.marker_family not in realizations


def test_fixture_declares_itself_constructed_with_no_sources(corpus):
    for record in corpus:
        assert record.source_type == "constructed"
        assert record.source_references == ()
        assert record.generation is None      # hand-written, not LLM-drafted


def test_fixture_carries_its_config_provenance(corpus):
    cfg = load_config(CONFIG_V2)
    for record in corpus:
        assert record.config_content_hash == cfg.content_hash


# --- the shared opening (D6) ------------------------------------------------


def test_the_opening_is_stored_once_and_shared_by_all_eight(corpus):
    for record in corpus:
        rendered = [record.render(o, c) for o in ("opt_1", "opt_2") for c in CORE_CONDITIONS]
        assert len(rendered) == 8
        assert all(text.startswith(record.counterargument_opening) for text in rendered)
        # equality across the eight holds by construction, not by check: the
        # opening appears exactly once in the record, on the scenario itself
        serialized = dumps_record(record)
        assert serialized.count(json.dumps(record.counterargument_opening)) == 1
        assert all("counterargument_opening" not in c
                   for b in record.counterarguments.values() for c in b.cells.values())


def test_render_is_opening_plus_body(corpus):
    record = corpus[0]
    body = record.counterarguments["opt_1"].cells["RS"].body
    assert record.render("opt_1", "RS") == f"{record.counterargument_opening} {body}"


# --- realization stored once, inherited on demand ---------------------------


def test_the_realization_is_stored_on_the_block_not_the_cells(payload):
    block = payload["counterarguments"]["opt_1"]
    assert {"marker_family", "marker_string", "marker_realization_id"} <= set(block)
    for cell in block["cells"].values():
        assert "marker_string" not in cell
        assert "marker_realization_id" not in cell


def test_resolve_composes_the_inherited_view(corpus):
    block = corpus[0].counterarguments["opt_2"]
    for condition in CORE_CONDITIONS:
        resolved = block.resolve(condition)
        assert resolved.inherited_marker_family == block.marker_family
        assert resolved.marker_string == block.marker_string
        assert resolved.marker_realization_id == block.marker_realization_id


def test_rs_and_ns_share_one_realization(corpus):
    """RS - NS is a content contrast because the realization cancels."""
    for record in corpus:
        for block in record.counterarguments.values():
            rs, ns = block.resolve("RS"), block.resolve("NS")
            assert (rs.marker_string, rs.marker_realization_id) == (ns.marker_string, ns.marker_realization_id)
            assert rs.markers_present is ns.markers_present is True


def test_rp_and_np_are_the_plain_cells(corpus):
    """Marker absence defines RP and NP, making RP - NP the plain content contrast."""
    for record in corpus:
        for block in record.counterarguments.values():
            for condition in ("RP", "NP"):
                cell = block.cells[condition]
                assert cell.markers_present is False
                assert cell.marker_family is None                     # §15.1
                assert block.resolve(condition).inherited_marker_family == block.marker_family


def test_resolved_cells_covers_all_sixteen(corpus):
    assert sum(len(r.resolved_cells()) for r in corpus) == 16


# --- round trip -------------------------------------------------------------


def test_round_trip_is_byte_stable(corpus, tmp_path):
    out = save_corpus(corpus, tmp_path / "again.jsonl")
    assert out.read_bytes() == FIXTURE.read_bytes()


def test_round_trip_preserves_the_records(corpus, tmp_path):
    reloaded = load_corpus(save_corpus(corpus, tmp_path / "again.jsonl"))
    assert reloaded == corpus
    assert corpus_content_hash(reloaded) == corpus_content_hash(corpus)


def test_content_hash_is_insensitive_to_key_order_but_not_to_content(corpus, tmp_path):
    shuffled = tmp_path / "shuffled.jsonl"
    lines = []
    for record in corpus:
        obj = json.loads(dumps_record(record))
        lines.append(json.dumps({k: obj[k] for k in reversed(list(obj))}))
    shuffled.write_text("\n".join(lines) + "\n")
    assert corpus_content_hash(load_corpus(shuffled)) == corpus_content_hash(corpus)

    edited = copy.deepcopy(corpus[0]).model_dump()
    edited["scenario_text"] += " One more clause."
    assert corpus_content_hash([ScenarioRecord.model_validate(edited)]) != corpus_content_hash([corpus[0]])


def test_blank_lines_are_ignored(corpus, tmp_path):
    path = tmp_path / "gappy.jsonl"
    path.write_text("\n" + FIXTURE.read_text() + "\n\n")
    assert load_corpus(path) == corpus


def test_malformed_json_reports_its_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\nnot json\n')
    with pytest.raises(CorpusError, match=r"bad\.jsonl:1"):
        load_corpus(path)


def test_an_invalid_record_reports_its_line(corpus, tmp_path):
    path = tmp_path / "bad.jsonl"
    obj = json.loads(dumps_record(corpus[0]))
    del obj["counterarguments"]["opt_2"]
    path.write_text(dumps_record(corpus[0]) + "\n" + json.dumps(obj) + "\n")
    with pytest.raises(CorpusError, match=r"bad\.jsonl:2"):
        load_corpus(path)


# --- structural rejections --------------------------------------------------


def _rebuild(payload: dict) -> ScenarioRecord:
    return ScenarioRecord.model_validate(payload)


def test_a_missing_direction_is_rejected(payload):
    del payload["counterarguments"]["opt_2"]
    with pytest.raises(ValidationError, match="one direction block per semantic option"):
        _rebuild(payload)


def test_a_missing_cell_is_rejected(payload):
    del payload["counterarguments"]["opt_1"]["cells"]["NS"]
    with pytest.raises(ValidationError, match="exactly the four core cells"):
        _rebuild(payload)


def test_a_cell_filed_under_the_wrong_condition_is_rejected(payload):
    payload["counterarguments"]["opt_1"]["cells"]["RS"]["condition"] = "RP"
    with pytest.raises(ValidationError):
        _rebuild(payload)


def test_a_block_filed_under_the_wrong_option_is_rejected(payload):
    payload["counterarguments"]["opt_1"]["supported_option"] = "opt_2"
    with pytest.raises(ValidationError, match="declares supported_option"):
        _rebuild(payload)


def test_a_styled_cell_with_a_different_family_is_rejected(payload):
    payload["counterarguments"]["opt_1"]["cells"]["NS"]["marker_family"] = "concession_contrast"
    with pytest.raises(ValidationError, match="all four cells share one realization"):
        _rebuild(payload)


def test_a_plain_cell_claiming_markers_is_rejected(payload):
    payload["counterarguments"]["opt_1"]["cells"]["NP"]["markers_present"] = True
    with pytest.raises(ValidationError, match="markers_present must be False"):
        _rebuild(payload)


def test_a_plain_cell_naming_a_marker_family_is_rejected(payload):
    payload["counterarguments"]["opt_1"]["cells"]["RP"]["marker_family"] = "premise_indicator"
    with pytest.raises(ValidationError, match=r"marker_family: null"):
        _rebuild(payload)


def test_a_styled_cell_without_markers_is_rejected(payload):
    payload["counterarguments"]["opt_1"]["cells"]["RS"]["markers_present"] = False
    with pytest.raises(ValidationError, match="markers_present must be True"):
        _rebuild(payload)


def test_a_scenario_id_that_disagrees_with_its_decision_is_rejected(payload):
    payload["scenario_id"] = "other_decision_v1"
    with pytest.raises(ValidationError, match="does not belong to decision"):
        _rebuild(payload)


def test_a_scenario_id_that_disagrees_with_its_variant_is_rejected(payload):
    payload["variant_id"] = 2
    with pytest.raises(ValidationError, match="disagrees with variant_id"):
        _rebuild(payload)


def test_an_unknown_field_is_rejected(payload):
    payload["difficulty"] = "hard"
    with pytest.raises(ValidationError):
        _rebuild(payload)


def test_records_are_immutable(corpus):
    with pytest.raises(ValidationError):
        corpus[0].scenario_text = "edited"      # type: ignore[misc]


# --- provenance -------------------------------------------------------------


def test_a_constructed_scenario_may_cite_nothing(payload):
    assert _rebuild(payload).source_references == ()


def test_a_constructed_scenario_may_not_cite_a_source(payload):
    payload["source_references"] = [{"dataset_name": "synthetic-example"}]
    with pytest.raises(ValidationError, match="constructed scenario cites no source"):
        _rebuild(payload)


def test_an_adapted_scenario_must_cite_at_least_one_source(payload):
    payload["source_type"] = "adapted"
    with pytest.raises(ValidationError, match="requires at least one source reference"):
        _rebuild(payload)


def test_a_scenario_may_cite_several_sources(payload):
    payload["source_type"] = "mixed"
    payload["source_references"] = [
        {"dataset_name": "synthetic-corpus-a", "dataset_version": "1.0",
         "source_item_id": "a-17", "source_url": "https://example.invalid/a/17",
         "access_date": "2026-09-09", "reuse_licence": "CC-BY-4.0"},
        {"dataset_name": "synthetic-corpus-b", "notes": "structure only"},
    ]
    record = _rebuild(payload)
    assert len(record.source_references) == 2
    assert record.source_references[0].reuse_licence == "CC-BY-4.0"
    assert record.source_references[1].dataset_version is None


def test_a_source_reference_needs_a_dataset_name():
    with pytest.raises(ValidationError):
        SourceReference(dataset_version="1.0")      # type: ignore[call-arg]


def test_generation_metadata_is_separate_from_source_references(payload):
    """It records how a draft was made. It is not a source and not a licence."""
    payload["generation"] = {
        "generator_model": "synthetic-generator",
        "generator_model_revision": "rev-abc",
        "prompt_hash": "a" * 64,
        "generation_parameters": {"temperature": 0.7},
        "seed": 11,
        "generated_at": "2026-09-09",
    }
    record = _rebuild(payload)
    assert record.generation is not None
    assert record.generation.generator_model == "synthetic-generator"
    assert not hasattr(record.generation, "dataset_name")


def test_generation_metadata_requires_a_prompt_hash():
    with pytest.raises(ValidationError):
        GenerationMetadata(
            generator_model="m", prompt_hash="not-a-hash",
            generation_parameters={}, generated_at=date(2026, 9, 9),
        )


# --- validation status ------------------------------------------------------


def test_a_draft_needs_no_counts(corpus):
    assert corpus[0].validation.status == "draft"
    assert corpus[0].validation.machine_errors is None


def test_machine_validated_requires_counts():
    with pytest.raises(ValidationError, match="requires machine error and warning counts"):
        ValidationStatus(status="machine_validated")


def test_approval_requires_a_reviewer():
    with pytest.raises(ValidationError, match="requires reviewed_by"):
        ValidationStatus(status="approved", machine_errors=0, machine_warnings=0)


def test_approval_is_impossible_with_machine_errors():
    with pytest.raises(ValidationError, match="cannot carry machine errors"):
        ValidationStatus(status="approved", machine_errors=1, machine_warnings=0,
                         reviewed_by="reviewer_1", reviewed_at=date(2026, 9, 9))


def test_approval_is_impossible_with_outstanding_human_review():
    """Machine validation is not approval: the H_ codes must be discharged."""
    with pytest.raises(ValidationError, match="outstanding human review"):
        ValidationStatus(status="approved", machine_errors=0, machine_warnings=0,
                         reviewed_by="reviewer_1", reviewed_at=date(2026, 9, 9),
                         outstanding_human_review=("H_SUPPORT_DIRECTION",))


def test_a_valid_approval(corpus):
    status = ValidationStatus(status="approved", machine_errors=0, machine_warnings=2,
                              reviewed_by="reviewer_1", reviewed_at=date(2026, 9, 9))
    assert status.status == "approved"


# --- measurements -----------------------------------------------------------


def test_measurements_are_absent_until_the_validator_fills_them(corpus):
    for record in corpus:
        for block in record.counterarguments.values():
            assert all(c.measurements is None for c in block.cells.values())


def test_measurements_record_which_text_was_counted():
    m = Measurements(word_count_full=32, word_count_body=22,
                     sentence_count_full=3, sentence_count_body=2,
                     segmenter=SegmenterRef(library="pysbd", version="0.3.4", language="en"))
    assert m.word_count_body < m.word_count_full
    assert m.segmenter.version == "0.3.4"


def test_body_counts_cannot_exceed_full_counts():
    with pytest.raises(ValidationError, match="word_count_body cannot exceed"):
        Measurements(word_count_full=10, word_count_body=11,
                     sentence_count_full=2, sentence_count_body=2,
                     segmenter=SegmenterRef(library="pysbd", version="0.3.4", language="en"))


def test_cell_and_block_can_be_built_directly():
    cells = {c: Cell(condition=c, body=f"body {c}.", markers_present=c in ("RS", "NS"),
                     marker_family="premise_indicator" if c in ("RS", "NS") else None)
             for c in CORE_CONDITIONS}
    block = DirectionBlock(supported_option="opt_1", marker_family="premise_indicator",
                           marker_string="because", marker_realization_id="clause_initial_premise_v1",
                           cells=cells)
    assert len(block.resolved_cells()) == 4
