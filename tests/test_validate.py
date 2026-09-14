"""Validator: the machine-checkable rules, and the limits of what they mean.

Two things are under test. That every corpus rule the validator claims to check
is actually enforced — each invalid fixture isolates one rule. And that a clean
run is never presented as a validated corpus: the ``human_review`` findings are
unconditional, and ``ok`` means "no machine errors" and nothing more.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reasonstyle.config import load_config
from reasonstyle.corpus.store import corpus_content_hash, load_corpus
from reasonstyle.corpus.findings import Finding
from reasonstyle.corpus.segmentation import segmenter_from_config
from reasonstyle.corpus.validate import HUMAN_REVIEW_CODES, validate_corpus, with_measurements

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "corpus.jsonl"





@pytest.fixture(scope="module")
def report(records, cfg, segmenter):
    return validate_corpus(records, cfg, segmenter, corpus_scope="fixture")


# --- the machine-valid fixture ----------------------------------------------


def test_the_fixture_is_machine_valid(report):
    assert report.errors == ()
    assert report.warnings == ()
    assert report.ok is True


def test_machine_valid_is_not_approved(report):
    """`ok` means no machine errors. It is not a claim about quality."""
    assert report.ok is True
    assert report.requires_human_review is True
    assert len(report.human_review) == 86


def test_the_summary_states_its_own_limits(report):
    text = report.summary()
    assert "MACHINE-VALID" in text
    assert "A clean run is not an approved corpus." in text
    for phrase in ("Substantive support", "no-reason integrity",
                   "proposition preservation", "pragmatic commitment"):
        assert phrase.lower() in text.lower()


def test_human_review_is_emitted_at_the_right_level(report):
    counts = report.codes("human_review")
    assert counts == {
        "H_SUPPORT_DIRECTION": 16,          # per cell
        "H_SUBSTANTIVE_SUPPORT": 16,        # per cell
        "H_NATURALNESS": 16,                # per cell
        "H_PRAGMATIC_COMMITMENT": 16,       # per cell
        "H_NO_REASON_INTEGRITY": 8,         # per NS/NP cell only
        "H_PROPOSITION_PRESERVATION": 8,    # per pair: RS/RP and NS/NP
        "H_REALIZATION_YIELDS_REASON_FREE_NS": 4,   # per group
        "H_SCENARIO_VALIDITY": 2,           # per scenario
    }
    assert set(counts) == set(HUMAN_REVIEW_CODES)


def test_no_reason_integrity_is_asked_only_of_the_no_reason_cells(report):
    conditions = {f.condition for f in report.human_review if f.code == "H_NO_REASON_INTEGRITY"}
    assert conditions == {"NS", "NP"}


def test_proposition_preservation_is_a_pair_level_judgement(report):
    findings = [f for f in report.human_review if f.code == "H_PROPOSITION_PRESERVATION"]
    assert all(f.scope == "pair" for f in findings)
    assert {tuple(f.detail["pair"]) for f in findings} == {("RS", "RP"), ("NS", "NP")}
    assert all(f.condition is None for f in findings)


def test_outstanding_review_is_reported_per_scenario(report):
    outstanding = report.outstanding_human_review("energy_fixture_001_v1")
    assert "H_SUPPORT_DIRECTION" in outstanding
    assert "H_SCENARIO_VALIDITY" in outstanding
    assert report.outstanding_human_review("no_such_scenario_v1") == ()


def test_report_carries_its_provenance(report, records, cfg, segmenter):
    assert report.config_content_hash == cfg.content_hash
    assert report.config_version == cfg.config_version
    assert report.corpus_content_hash == corpus_content_hash(records)
    assert report.segmenter == segmenter.info.as_dict()
    assert (report.n_scenarios, report.n_texts) == (2, 16)
    assert report.corpus_scope == "fixture"


def test_report_serializes(report):
    payload = report.as_dict()
    assert payload["ok"] is True and payload["requires_human_review"] is True
    assert len(payload["findings"]) == len(report.findings)
    assert set(payload["findings"][0]) == {
        "code", "severity", "message", "scope", "decision_id",
        "scenario_id", "supported_option", "condition", "detail"}


def test_finding_locator():
    f = Finding(code="E_X", severity="error", message="m", scope="cell",
                scenario_id="s_v1", supported_option="opt_1", condition="RS")
    assert f.locator == "s_v1/opt_1/RS"
    assert str(f).startswith("[error] E_X at s_v1/opt_1/RS")


# --- each invalid fixture isolates its rule ---------------------------------

EXPECTED_ERROR = {
    "sentence_count_mismatch": "E_SENTENCE_COUNT_MISMATCH",
    "word_ratio_fail": "E_WORD_RATIO_BODY",
    "marker_missing_in_styled_cell": "E_MARKER_MISSING_IN_STYLED_CELL",
    "marker_in_plain_cell": "E_MARKER_IN_PLAIN_CELL",
    "forbidden_hard_fail": "E_FORBIDDEN",
    "label_leakage": "E_LABEL_LEAKAGE",
    "unknown_domain": "E_UNKNOWN_DOMAIN",
    "unknown_realization": "E_REALIZATION_UNKNOWN",
    "realization_family_mismatch": "E_REALIZATION_FAMILY_MISMATCH",
    "marker_not_in_family": "E_MARKER_NOT_IN_FAMILY",
    "duplicate_text": "E_DUPLICATE_TEXT",
    "duplicate_scenario_id": "E_DUPLICATE_SCENARIO_ID",
    "prohibited_formatting": "E_PROHIBITED_FORMATTING",
    "config_hash_mismatch": "E_CONFIG_HASH_MISMATCH",
}

EXPECTED_WARNING = {
    "forbidden_warning": "W_FORBIDDEN",
    "word_ratio_warn": "W_WORD_RATIO_BODY",
    "ambiguous_segmentation": "W_AMBIGUOUS_SEGMENTATION",
}


def _validate(name, cfg, segmenter, invalid_corpus):
    return validate_corpus(load_corpus(invalid_corpus(name)), cfg, segmenter, "fixture")


@pytest.mark.parametrize(("name", "code"), sorted(EXPECTED_ERROR.items()))
def test_invalid_fixtures_are_rejected(name, code, cfg, segmenter, invalid_corpus):
    report = _validate(name, cfg, segmenter, invalid_corpus)
    assert report.ok is False, f"{name} should not be machine-valid"
    assert code in report.codes("error"), f"{name}: expected {code}, got {sorted(report.codes('error'))}"


@pytest.mark.parametrize(("name", "code"), sorted(EXPECTED_WARNING.items()))
def test_warning_fixtures_are_flagged_but_not_rejected(name, code, cfg, segmenter, invalid_corpus):
    """A warning routes to a reviewer; it does not block on its own."""
    report = _validate(name, cfg, segmenter, invalid_corpus)
    assert report.ok is True, f"{name} produced errors: {sorted(report.codes('error'))}"
    assert code in report.codes("warning")


@pytest.mark.parametrize("name", sorted(set(EXPECTED_ERROR) - {"prohibited_formatting"}))
def test_each_invalid_fixture_isolates_one_rule(name, cfg, segmenter, invalid_corpus):
    """One broken rule, one error code — so a regression cannot hide behind a
    cascade. `prohibited_formatting` is exempt: a bullet list unavoidably breaks
    the length, sentence and marker rules at the same time."""
    codes = set(_validate(name, cfg, segmenter, invalid_corpus).codes("error"))
    expected = {EXPECTED_ERROR[name]}
    if name == "word_ratio_fail":
        expected.add("E_WORD_RATIO_FULL_TEXT")      # both measurements exceed the cap
    if name == "duplicate_scenario_id":
        expected.add("E_FIXTURE_SHAPE")             # two v1 records is also a shape error
    assert codes == expected


def test_the_body_ratio_catches_what_the_full_text_ratio_misses(cfg, segmenter, invalid_corpus):
    """The point of measuring both. Here the manipulated bodies differ by
    1.143 while the full texts differ by only 1.094, because the shared opening
    dilutes the difference — so the full-text measurement stays silent."""
    report = _validate("word_ratio_warn", cfg, segmenter, invalid_corpus)
    assert "W_WORD_RATIO_BODY" in report.codes("warning")
    assert "W_WORD_RATIO_FULL_TEXT" not in report.codes("warning")
    finding = next(f for f in report.warnings if f.code == "W_WORD_RATIO_BODY")
    assert finding.detail["measurement"] == "body"
    assert 1.10 < finding.detail["ratio"] <= 1.15


def test_forbidden_severity_decides_error_versus_warning(cfg, segmenter, invalid_corpus):
    hard = _validate("forbidden_hard_fail", cfg, segmenter, invalid_corpus)
    soft = _validate("forbidden_warning", cfg, segmenter, invalid_corpus)
    hard_finding = next(f for f in hard.errors if f.code == "E_FORBIDDEN")
    soft_finding = next(f for f in soft.warnings if f.code == "W_FORBIDDEN")
    assert hard_finding.detail["pattern_id"] == "ev_studies"
    assert soft_finding.detail["pattern_id"] == "ev_proven"
    assert hard.ok is False and soft.ok is True


def test_findings_locate_the_offending_cell(cfg, segmenter, invalid_corpus):
    report = _validate("marker_in_plain_cell", cfg, segmenter, invalid_corpus)
    finding = next(f for f in report.errors if f.code == "E_MARKER_IN_PLAIN_CELL")
    assert (finding.scenario_id, finding.supported_option, finding.condition) == (
        "energy_fixture_001_v1", "opt_1", "RP")
    assert "given that" in finding.detail["markers_found"]


def test_sentence_mismatch_reports_all_four_counts(cfg, segmenter, invalid_corpus):
    report = _validate("sentence_count_mismatch", cfg, segmenter, invalid_corpus)
    finding = next(f for f in report.errors if f.code == "E_SENTENCE_COUNT_MISMATCH")
    assert finding.scope == "group"
    assert set(finding.detail["counts"]) == {"RS", "RP", "NS", "NP"}
    assert len(set(finding.detail["counts"].values())) > 1


# --- corpus scope gating ----------------------------------------------------


def test_allocation_checks_are_skipped_outside_the_full_corpus(report):
    """Skipped loudly, never passed silently."""
    skipped = next(f for f in report.info if f.code == "I_ALLOCATION_SKIPPED")
    assert set(skipped.detail["skipped"]) == {
        "E_MARKER_UNDER_ALLOCATED", "E_MARKER_CONCENTRATED_DOMAIN",
        "E_MARKER_CONCENTRATED_OPTION"}
    assert "E_MARKER_UNDER_ALLOCATED" not in report.codes("error")


def test_allocation_checks_run_at_full_scope(records, cfg, segmenter):
    """The same fixture fails as a 'full corpus': one decision cannot satisfy
    the minima, and every marker sits in one domain and one direction."""
    full = validate_corpus(records, cfg, segmenter, corpus_scope="full")
    assert full.ok is False
    assert full.codes("error")["E_MARKER_UNDER_ALLOCATED"] == 4
    assert "E_MARKER_CONCENTRATED_DOMAIN" in full.codes("error")
    assert "E_MARKER_CONCENTRATED_OPTION" in full.codes("error")
    assert "I_ALLOCATION_SKIPPED" not in full.codes("info")


def test_pilot_scope_also_defers_allocation(records, cfg, segmenter):
    pilot = validate_corpus(records, cfg, segmenter, corpus_scope="pilot")
    assert "I_ALLOCATION_SKIPPED" in pilot.codes("info")
    assert "E_FIXTURE_SHAPE" not in pilot.codes("error")


def test_unfrozen_splits_are_reported_as_unchecked(report):
    assert "I_SPLITS_NOT_ASSIGNED" in report.codes("info")


# --- measurements -----------------------------------------------------------


def test_with_measurements_fills_every_cell(records, cfg, segmenter):
    measured = with_measurements(records, cfg, segmenter)
    cells = [c for r in measured for b in r.counterarguments.values() for c in b.cells.values()]
    assert len(cells) == 16
    assert all(c.measurements is not None for c in cells)
    sample = measured[0].counterarguments["opt_1"].cells["RS"].measurements
    assert sample is not None
    assert sample.word_count_body < sample.word_count_full
    assert sample.sentence_count_body == 2 and sample.sentence_count_full == 3
    assert sample.segmenter.library == "pysbd"


def test_with_measurements_leaves_the_originals_untouched(records, cfg, segmenter):
    with_measurements(records, cfg, segmenter)
    assert all(c.measurements is None
               for r in records for b in r.counterarguments.values() for c in b.cells.values())


# --- drafting-shape rules (approved 2026-09-14) ------------------------------


def _mutate(records, cfg, segmenter, change):
    """Validate a copy of the fixture with one change applied."""
    import copy
    from reasonstyle.corpus.schemas import ScenarioRecord
    payload = [json_of(r) for r in records]
    change(payload)
    return validate_corpus([ScenarioRecord.model_validate(p) for p in copy.deepcopy(payload)],
                           cfg, segmenter, corpus_scope="fixture")


def json_of(record):
    import json
    from reasonstyle.corpus.store import dumps_record
    return json.loads(dumps_record(record))


def test_a_scenario_outside_the_word_band_warns(records, cfg, segmenter):
    """The band is a drafting instruction, and the pilot is meant to show
    whether it forced filler — so it is reported, not enforced."""
    report = _mutate(records, cfg, segmenter,
                     lambda p: p[0].__setitem__("scenario_text", "Too short by far."))
    assert "W_SCENARIO_WORDS" in report.codes("warning")
    assert report.errors == ()


def test_the_opening_must_be_the_corpus_wide_one(records, cfg, segmenter):
    report = _mutate(records, cfg, segmenter,
                     lambda p: p[0].__setitem__("counterargument_opening",
                                                "I have read the scenario and would weigh it differently."))
    assert "E_OPENING_NOT_CORPUS_WIDE" in report.codes("error")


def test_a_body_of_the_wrong_length_is_rejected(records, cfg, segmenter):
    """All four cells at three sentences: internally consistent, but not the
    fixed body length. The mismatch error would not catch this."""
    def lengthen(p):
        for condition, cell in p[0]["counterarguments"]["opt_1"]["cells"].items():
            cell["body"] += " The plant can deliver it."
    report = _mutate(records, cfg, segmenter, lengthen)
    assert "E_BODY_SENTENCE_COUNT" in report.codes("error")
    assert "E_SENTENCE_COUNT_MISMATCH" not in report.codes("error")


def test_a_premise_from_outside_the_scenario_is_flagged(records, cfg, segmenter):
    """A lexical screen only: it points the reviewer at a cell whose content
    words are not in its own scenario. The human judgement remains the one that
    decides."""
    report = _mutate(records, cfg, segmenter, lambda p: p[0]["counterarguments"]["opt_1"]
                     ["cells"]["RS"].__setitem__(
                         "body", "The proposed tariff reform lowers connection charges for "
                                 "industry. Because charges fall, the plant extension remains "
                                 "my preferred option."))
    assert "W_PREMISE_NOT_IN_SCENARIO" in report.codes("warning")


def test_every_reason_cell_of_the_fixture_is_contained_by_its_scenario(records, cfg):
    """The fixture satisfies the containment rule the real corpus must satisfy:
    each reason cell's content words come from its own scenario."""
    import re
    from reasonstyle.corpus.validate import _containment
    word = re.compile(cfg.parsed.matching.words.word_regex)
    spec = cfg.raw["corpus"]["premise_containment"]
    for record in records:
        for option, block in record.counterarguments.items():
            for condition in ("RS", "RP"):
                unmatched, coverage = _containment(block.cells[condition].body,
                                                   record.scenario_text, word, spec)
                assert coverage >= spec["min_content_word_coverage"], (
                    record.scenario_id, option, condition, unmatched)


def test_the_fixture_scenarios_state_both_brief_facts(records, cfg, synthetic_bank):
    """The chain brief -> scenario -> counterargument holds end to end: both of
    a variant's facts are stated in the scenario, before any answer is given."""
    import re
    word = re.compile(cfg.parsed.matching.words.word_regex)
    topic = next(t for t in synthetic_bank.topics if t.decision_id == "energy_fixture_001")
    for record in records:
        variant = topic.variants[f"v{record.variant_id}"]
        in_scenario = {w.casefold() for w in word.findall(record.scenario_text)}
        for option in ("opt_1", "opt_2"):
            fact, = getattr(variant.scenario_facts, option)
            missing = [w for w in word.findall(fact)
                       if len(w) >= 4 and w.casefold() not in in_scenario]
            assert not missing, (record.scenario_id, option, missing)
