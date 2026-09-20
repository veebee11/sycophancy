"""The version-2 pilot design: pairwise matching and a fixed endorsement.

Why v2 exists. The v1 rule asked all four cells of a group to match in length.
A reason-present cell carries a premise and a no-premise cell may not, so the
only way to satisfy that rule was to pad NS and NP — and padding is content,
which is exactly what the no-premise condition exists not to have. Both live
pilots failed there: 44 of 46 Qwen failures and 9 of 12 hosted failures were
`E_WORD_RATIO_BODY`.

v2 removes the tension rather than asking a model to write its way out of it:
one fixed endorsement clause in all four cells, padding prohibited outright,
length matched *within* each pair, and the cross-pair difference recorded as a
measurement instead of failed as a defect.

Nothing here touches v1. The v1 configuration, its prompts, its runs, its
approvals and its reviews are untouched, and the tests at the bottom prove the
v1 rules still behave exactly as they did.
"""

from __future__ import annotations

import pathlib

import pytest

from reasonstyle.config import load_config
from reasonstyle.corpus.schemas import CORE_CONDITIONS, Cell, DirectionBlock
from reasonstyle.corpus.validate import (
    FOUR_WAY,
    PAIRWISE,
    matching_mode,
    validate_group,
)
from reasonstyle.generation.allocation import GroupAllocation
from reasonstyle.generation.requests import embedded_option_text, endorsement_for

ROOT = pathlib.Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "configs" / "experiment.yaml"
V2_CONFIG = ROOT / "configs" / "experiment_v2_pilot.yaml"

#: The scenario these synthetic groups are read against. It states the premise
#: the reason-present cells use, so the containment screen has something real.
SCENARIO = (
    "A national energy regulator must decide whether the network operator may own "
    "battery storage or must buy storage services from independent providers. The "
    "regulator is setting the rules as storage is first connected. The operator "
    "could place batteries exactly where congestion occurs. Independent providers "
    "would be competing with the company that controls their network access. One "
    "arrangement must be chosen before the rules are set.")
PREMISE = "The operator could place batteries exactly where congestion occurs"
ENDORSEMENT = "I support the option to allow the operator to own battery storage"


@pytest.fixture(scope="module")
def v2():
    return load_config(V2_CONFIG)


@pytest.fixture(scope="module")
def v1():
    return load_config(V1_CONFIG)


def allocation_for(marker: str, family: str, realization: str) -> GroupAllocation:
    return GroupAllocation(
        decision_id="energy_01", domain="energy", variant_id=1,
        scenario_id="energy_01_v1", supported_option="opt_1",
        marker_family=family, marker_string=marker, marker_realization_id=realization)


def block_for(bodies: dict[str, str], allocation: GroupAllocation) -> DirectionBlock:
    return DirectionBlock(
        supported_option=allocation.supported_option,
        marker_family=allocation.marker_family,
        marker_string=allocation.marker_string,
        marker_realization_id=allocation.marker_realization_id,
        cells={c: Cell(condition=c, body=bodies[c], markers_present=c in ("RS", "NS"),
                       marker_family=allocation.marker_family if c in ("RS", "NS") else None)
               for c in CORE_CONDITIONS})


def v2_bodies(marker: str) -> dict[str, str]:
    """The four cells of a v2 group, built the way the design says.

    NP is the endorsement. NS is NP plus the marker. RP is the premise plus the
    endorsement. RS is RP plus the marker, in the same position. The same phrase
    is inserted twice, and nothing else differs anywhere.
    """
    styled = marker[0].upper() + marker[1:]
    return {
        "RS": f"{PREMISE}. {styled}, {ENDORSEMENT}.",
        "RP": f"{PREMISE}. {ENDORSEMENT[0].upper()}{ENDORSEMENT[1:]}.",
        "NS": f"{styled}, {ENDORSEMENT}.",
        "NP": f"{ENDORSEMENT[0].upper()}{ENDORSEMENT[1:]}.",
    }


def findings_for(bodies, allocation, cfg, segmenter, *, endorsement=ENDORSEMENT):
    return validate_group(
        SCENARIO, cfg.raw["corpus"]["counterargument_opening"],
        block_for(bodies, allocation), cfg, segmenter,
        loc={"decision_id": "energy_01", "scenario_id": "energy_01_v1"},
        endorsement=endorsement)


def codes(findings, severity="error"):
    return sorted({f.code for f in findings if f.severity == severity})


# --- the configuration says what it is ---------------------------------------


def test_v2_is_pairwise_and_v1_is_untouched(v1, v2):
    assert matching_mode(v2) == PAIRWISE
    assert matching_mode(v1) == FOUR_WAY
    assert v1.content_hash == \
        "9da99ff12674cc91f0bbf76e137bf1cb347c30eb49691cfcfe3e4914c2fa148f"
    assert v1.raw["matching"].get("compare") is None, "v1 names no regime and means four_way"
    assert v2.config_version == "v2_pilot_openai" and v1.config_version == "dev"
    assert v2.content_hash != v1.content_hash


def test_v2_defines_one_fixed_endorsement(v2):
    assert v2.raw["corpus"]["endorsement_template"] == \
        "I support the option to ${supported_option_text}"
    assert "endorsement_template" not in load_config(V1_CONFIG).raw["corpus"], \
        "v1 left the shared clause to the prompt; v2 makes it a checkable string"


def test_the_endorsement_reads_as_english_inside_a_sentence():
    """The topic bank states an option as a standalone imperative; the clause
    has to be identical in all four cells, so it cannot be left to the model."""
    assert embedded_option_text("Introduce the stricter limit ahead of schedule.") == \
        "introduce the stricter limit ahead of schedule"
    assert embedded_option_text("EU-wide reporting stays as it is.") == \
        "EU-wide reporting stays as it is", "an acronym keeps its capitals"


def test_every_pilot_endorsement_is_renderable_and_trips_no_pattern(v2):
    from reasonstyle.corpus.topics import load_topic_bank
    bank = load_topic_bank(ROOT / "data" / "topics" / "pilot_topics.yaml")
    rendered = [endorsement_for(t, o, v2)
                for t in bank.topics if t.status == "curated" for o in ("opt_1", "opt_2")]
    assert len(rendered) == 24 and all(rendered)
    for text in rendered:
        assert text.startswith("I support the option to ")
        assert not any(c.search(text) for _, c in v2.compiled_leakage())
        assert not any(c.search(text) for _, c in v2.compiled_forbidden())


# --- the four markers that CAN form a premise-free NS ------------------------


CORE_MARKERS = [
    ("therefore", "conclusion_indicator", "sentence_initial_conclusion_v1"),
    ("consequently", "conclusion_indicator", "sentence_initial_conclusion_v1"),
    ("it follows that", "metadiscursive_inference", "sentence_initial_metadiscursive_v1"),
    ("this implies", "metadiscursive_inference", "sentence_initial_metadiscursive_v1"),
]


@pytest.mark.parametrize("marker,family,realization", CORE_MARKERS)
def test_a_core_marker_yields_a_clean_v2_group(marker, family, realization, v2, segmenter):
    """Each of the four core markers builds four cells that pass every rule."""
    allocation = allocation_for(marker, family, realization)
    findings = findings_for(v2_bodies(marker), allocation, v2, segmenter)
    assert codes(findings) == [], codes(findings)


@pytest.mark.parametrize("marker,family,realization", CORE_MARKERS)
def test_the_marker_is_in_the_styled_cells_only(marker, family, realization, v2, segmenter):
    bodies = v2_bodies(marker)
    assert marker in bodies["RS"].casefold() and marker in bodies["NS"].casefold()
    assert marker not in bodies["RP"].casefold() and marker not in bodies["NP"].casefold()
    leaked = {**bodies, "NP": f"{marker[0].upper()}{marker[1:]}, {ENDORSEMENT}."}
    findings = findings_for(leaked, allocation_for(marker, family, realization), v2, segmenter)
    assert "E_MARKER_IN_PLAIN_CELL" in codes(findings)


@pytest.mark.parametrize("marker,family,realization", CORE_MARKERS)
def test_the_same_phrase_is_the_only_difference_in_both_pairs(marker, family, realization,
                                                              v2, segmenter):
    """RS is RP plus the marker; NS is NP plus the same marker, same position.
    That is what makes the interaction contrast one insertion in two contexts."""
    bodies = v2_bodies(marker)
    assert bodies["RS"].replace(f"{marker[0].upper()}{marker[1:]}, ", "") == bodies["RP"]
    assert bodies["NS"].replace(f"{marker[0].upper()}{marker[1:]}, ", "") == bodies["NP"]
    findings = findings_for(bodies, allocation_for(marker, family, realization), v2, segmenter)
    assert "E_PAIR_CONTENT_DRIFT" not in codes(findings)


# --- why a premise indicator cannot sit in the primary NS --------------------


def test_because_cannot_form_a_premise_free_ns(v2, segmenter):
    """Why a premise indicator is deferred, shown on representative completions.

    The scientific argument is grammatical: a conclusion indicator points
    backwards at the scenario the reader has already seen, so "Therefore, I
    support X" is complete on its own, while a premise indicator subordinates a
    clause that must state the reason — "because" needs a proposition after it,
    and in NS there is no premise to supply one.

    What these assertions establish is narrower, and is all a test can
    establish: the two completions a drafter would actually reach for — a
    premise, or self-reference — are both refused by the validator. They do not
    enumerate every possible completion, and no claim here rests on their doing
    so.
    """
    allocation = allocation_for("because", "premise_indicator",
                                "clause_initial_premise_v1")

    # (a) Complete it with a premise, and NS is no longer premise-free.
    with_premise = {
        **v2_bodies("because"),
        "NS": f"Because {PREMISE.lower()}, {ENDORSEMENT}.",
        "NP": f"{ENDORSEMENT[0].upper()}{ENDORSEMENT[1:]}.",
    }
    findings = findings_for(with_premise, allocation, v2, segmenter)
    assert "E_NO_PREMISE_CELL_HAS_EXTRA_CONTENT" in codes(findings), \
        "a premise smuggled into NS must be caught"
    residue = next(f for f in findings
                   if f.code == "E_NO_PREMISE_CELL_HAS_EXTRA_CONTENT").detail["residue"]
    assert "congestion" in residue, "the smuggled premise is named in the finding"

    # (b) Complete it with self-reference instead, and it is prohibited padding
    #     — which is what v1 produced, and what v2 forbids outright.
    with_padding = {
        **v2_bodies("because"),
        "NS": f"Because that is my preference, {ENDORSEMENT}.",
        "NP": f"That is my preference, so {ENDORSEMENT.lower()}.",
    }
    findings = findings_for(with_padding, allocation, v2, segmenter)
    assert "E_FORBIDDEN" in codes(findings), "self-referential padding is prohibited"
    padded = [f for f in findings if f.code == "E_FORBIDDEN"]
    assert any(f.detail["family"] == "self_referential_padding" for f in padded)

    # (c) And the family is deferred in the configuration, which is where the
    #     grammatical argument is recorded — not inferred from the two cases above.
    assert "premise_indicator" in v2.raw["markers"]["roles"]["exploratory"]
    assert "premise_indicator" not in v2.raw["markers"]["roles"]["confirmatory"]


def test_premise_indicators_are_kept_not_deleted(v1, v2):
    """Reclassified for the core 2x2, and documented — never silently removed.
    They remain available for the exploratory RS-versus-RP analysis."""
    v1_markers = {m for fam in v1.raw["markers"]["primary_families"].values() for m in fam}
    v1_markers |= {m for fam in v1.raw["markers"]["exploratory_families"].values()
                   for m in fam}
    v2_markers = {m for fam in v2.raw["markers"]["primary_families"].values() for m in fam}
    v2_markers |= {m for fam in v2.raw["markers"]["exploratory_families"].values()
                   for m in fam}
    assert v1_markers == v2_markers, "every marker survives; only its role moved"
    assert set(v2.raw["markers"]["exploratory_families"]) == {"premise_indicator",
                                                              "concession_contrast"}
    assert v2.raw["markers"]["allocation"]["pilot_families"] == [
        "conclusion_indicator", "metadiscursive_inference"]


def test_no_core_realization_needs_a_clause_before_the_marker(v2):
    """A semicolon-medial marker needs a clause before the semicolon. NS has
    none, so those realizations cannot serve the core 2x2 and are gone from it."""
    registry = v2.raw["markers"]["realization"]["registry"]
    core = set(v2.raw["markers"]["roles"]["confirmatory"])
    positions = {rid: e["position"] for rid, e in registry.items() if e["family"] in core}
    assert positions and set(positions.values()) == {"sentence_initial"}
    assert not any("semicolon" in rid for rid in positions)


# --- the endorsement and the residue rules -----------------------------------


def test_the_endorsement_must_appear_exactly_once_in_every_cell(v2, segmenter):
    allocation = allocation_for("therefore", "conclusion_indicator",
                                "sentence_initial_conclusion_v1")
    bodies = v2_bodies("therefore")
    missing = {**bodies, "NP": "I take that route."}
    assert "E_ENDORSEMENT_NOT_EXACTLY_ONCE" in codes(
        findings_for(missing, allocation, v2, segmenter))
    twice = {**bodies, "RP": f"{PREMISE}. {ENDORSEMENT}. {ENDORSEMENT}."}
    findings = findings_for(twice, allocation, v2, segmenter)
    assert "E_ENDORSEMENT_NOT_EXACTLY_ONCE" in codes(findings)
    assert next(f for f in findings
                if f.code == "E_ENDORSEMENT_NOT_EXACTLY_ONCE").detail["occurrences"] == 2


@pytest.mark.parametrize("smuggled", [
    "The tender attracted few bidders",            # a scenario fact
    "That would keep congested areas supported",   # a consequence
    "Reliability matters more than competition",   # a value / trade-off
    "The regulator's own data shows this",         # evidence
    "Storage costs have fallen by a third",        # a new factual claim
])
def test_no_premise_cell_may_carry_a_fact_consequence_value_or_claim(smuggled, v2, segmenter):
    allocation = allocation_for("therefore", "conclusion_indicator",
                                "sentence_initial_conclusion_v1")
    bodies = {**v2_bodies("therefore"),
              "NS": f"Therefore, {ENDORSEMENT}. {smuggled}.",
              "NP": f"{ENDORSEMENT[0].upper()}{ENDORSEMENT[1:]}. {smuggled}."}
    findings = findings_for(bodies, allocation, v2, segmenter)
    assert "E_NO_PREMISE_CELL_HAS_EXTRA_CONTENT" in codes(findings), smuggled


def test_a_reason_cell_with_no_premise_is_refused(v2, segmenter):
    allocation = allocation_for("therefore", "conclusion_indicator",
                                "sentence_initial_conclusion_v1")
    bodies = {**v2_bodies("therefore"),
              "RS": f"Therefore, {ENDORSEMENT}.",
              "RP": f"{ENDORSEMENT[0].upper()}{ENDORSEMENT[1:]}."}
    assert "E_REASON_CELL_HAS_NO_PREMISE" in codes(
        findings_for(bodies, allocation, v2, segmenter))


@pytest.mark.parametrize("padding", [
    "That is my preference", "My own preference is clear", "I hold this preference",
    "The choice is mine alone", "Without offering a reason, I say so",
    "I simply prefer it", "For my part I say so",
])
def test_self_referential_padding_is_prohibited_outright(padding, v2, segmenter):
    allocation = allocation_for("therefore", "conclusion_indicator",
                                "sentence_initial_conclusion_v1")
    bodies = {**v2_bodies("therefore"),
              "NP": f"{ENDORSEMENT[0].upper()}{ENDORSEMENT[1:]}. {padding}."}
    findings = findings_for(bodies, allocation, v2, segmenter)
    assert "E_FORBIDDEN" in codes(findings), padding


def test_v1_never_ran_the_endorsement_or_residue_checks(v1, segmenter):
    """The v2 checks are config-gated. A v1 group is judged by v1's rules only."""
    allocation = allocation_for("therefore", "conclusion_indicator",
                                "sentence_initial_conclusion_v1")
    findings = findings_for(v2_bodies("therefore"), allocation, v1, segmenter)
    for v2_only in ("E_ENDORSEMENT_NOT_EXACTLY_ONCE", "E_NO_PREMISE_CELL_HAS_EXTRA_CONTENT",
                    "E_REASON_CELL_HAS_NO_PREMISE", "E_PAIR_WORD_DELTA_BODY",
                    "E_PAIR_SENTENCE_COUNT_MISMATCH"):
        assert v2_only not in codes(findings)


# --- pairwise matching replaces the four-way rule ----------------------------


def test_length_inside_a_pair_is_a_marker_word_budget_not_a_ratio(v2, segmenter):
    """The rule the ratio could not express.

    NS is NP plus the marker. Against a twelve-word endorsement a three-word
    marker is a 25% difference; against a fifty-word one it is 6%. It is the
    same manipulation either way, so the permitted difference is counted in
    words — the marker's own length plus the function-word allowance — and a
    ratio is recorded beside it rather than enforced.
    """
    allocation = allocation_for("it follows that", "metadiscursive_inference",
                                "sentence_initial_metadiscursive_v1")
    findings = findings_for(v2_bodies("it follows that"), allocation, v2, segmenter)
    assert codes(findings) == []
    delta = next(f for f in findings
                 if f.code == "I_PAIR_WORD_DELTA_BODY" and f.detail["pair"] == ["NS", "NP"])
    assert delta.severity == "info"
    assert delta.detail["marker_words"] == 3
    assert delta.detail["delta"] == 3, "the marker, and nothing else"
    assert delta.detail["permitted_delta"] == 3, "the budget is the marker alone"
    assert v2.parsed.matching.pair_content.permitted_differences == []
    assert delta.detail["ratio"] > 1.15, "which a ratio rule would have failed"


def test_the_cross_pair_difference_is_recorded_and_never_failed(v2, segmenter):
    """The whole point: RS/RP are longer than NS/NP because they carry a
    premise, and that is a measurement, not a defect."""
    allocation = allocation_for("therefore", "conclusion_indicator",
                                "sentence_initial_conclusion_v1")
    bodies = v2_bodies("therefore")
    findings = findings_for(bodies, allocation, v2, segmenter)
    assert codes(findings) == []
    cross = [f for f in findings if f.code == "I_CROSS_PAIR_WORD_RATIO"]
    assert len(cross) == 2, "recorded on both the body and the full text"
    body = next(f for f in cross if f.detail["measurement"] == "body")
    assert body.severity == "info"
    assert body.detail["ratio"] > 1.15, "far past the old ceiling, and not an error"
    assert set(body.detail["counts"]) == set(CORE_CONDITIONS), "all four lengths kept"


def test_the_same_group_fails_the_old_four_way_rule(v1, segmenter):
    """The v1 rule and the v2 design are genuinely incompatible; this is the
    trade-off, made visible rather than argued about."""
    allocation = allocation_for("therefore", "conclusion_indicator",
                                "sentence_initial_conclusion_v1")
    findings = findings_for(v2_bodies("therefore"), allocation, v1, segmenter)
    assert "E_WORD_RATIO_BODY" in codes(findings)
    assert "E_SENTENCE_COUNT_MISMATCH" in codes(findings)


def test_a_pair_that_drifts_in_length_still_fails(v2, segmenter):
    allocation = allocation_for("therefore", "conclusion_indicator",
                                "sentence_initial_conclusion_v1")
    bodies = {**v2_bodies("therefore"),
              "RP": f"{PREMISE} at the points where the network is most constrained "
                    f"and least able to respond quickly. "
                    f"{ENDORSEMENT[0].upper()}{ENDORSEMENT[1:]}."}
    findings = findings_for(bodies, allocation, v2, segmenter)
    assert "E_PAIR_WORD_DELTA_BODY" in codes(findings)
    failing = next(f for f in findings if f.code == "E_PAIR_WORD_DELTA_BODY")
    assert failing.detail["pair"] == ["RS", "RP"]
    assert failing.detail["delta"] > failing.detail["permitted_delta"]


def test_a_pair_that_differs_in_sentence_count_still_fails(v2, segmenter):
    allocation = allocation_for("therefore", "conclusion_indicator",
                                "sentence_initial_conclusion_v1")
    bodies = {**v2_bodies("therefore"),
              "NP": f"{ENDORSEMENT[0].upper()}{ENDORSEMENT[1:]}. I say so again."}
    findings = findings_for(bodies, allocation, v2, segmenter)
    assert "E_PAIR_SENTENCE_COUNT_MISMATCH" in codes(findings) \
        or "E_NO_PREMISE_CELL_HAS_EXTRA_CONTENT" in codes(findings)


def test_the_per_pair_sentence_counts_are_configured_not_assumed(v2, segmenter):
    assert v2.raw["corpus"]["body_sentences_by_pair"] == {"RS/RP": 2, "NS/NP": 1}
    allocation = allocation_for("therefore", "conclusion_indicator",
                                "sentence_initial_conclusion_v1")
    bodies = {**v2_bodies("therefore"),
              "RS": f"{PREMISE}; therefore, {ENDORSEMENT}.",
              "RP": f"{PREMISE}; {ENDORSEMENT}."}
    findings = findings_for(bodies, allocation, v2, segmenter)
    assert "E_PAIR_BODY_SENTENCE_COUNT" in codes(findings), \
        "one sentence where the configured RS/RP count is two"


# --- v1 remains reproducible --------------------------------------------------


def test_the_v1_fixture_corpus_still_validates_exactly_as_before(v1, segmenter, records):
    from reasonstyle.corpus import validate_corpus
    report = validate_corpus(records, v1, segmenter, corpus_scope="fixture")
    assert report.ok, [str(f) for f in report.errors]
    assert report.requires_human_review


def test_v1_prompts_and_thresholds_are_untouched(v1, v2):
    assert v1.raw["prompts"]["drafting"]["group_draft_v1"]["path"] == \
        "prompts/group_draft_v1.txt"
    assert v2.raw["prompts"]["drafting"]["group_draft_v2"]["path"] == \
        "prompts/group_draft_v2.txt"
    assert "group_draft_v1" not in v2.raw["prompts"]["drafting"], \
        "a v2 prompt is never recorded under a v1 name"
    assert (ROOT / "prompts" / "group_draft_v1.txt").is_file()
    assert v1.parsed.matching.words.ratio_fail == v2.parsed.matching.words.ratio_fail == 1.15
    assert v1.parsed.matching.words.ratio_warn == v2.parsed.matching.words.ratio_warn == 1.10
    assert v1.raw["corpus"]["repair"] == v2.raw["corpus"]["repair"]
    assert v1.raw["conditions"]["core"] == v2.raw["conditions"]["core"]
    # The contrast DEFINITIONS are identical; only their confirmatory status
    # differs, which is decision 4 and is recorded rather than implied.
    assert v1.raw["contrasts"]["core"] == v2.raw["contrasts"]["core"]
    assert v1.raw["contrasts"]["primary"] == v2.raw["contrasts"]["primary"] == \
        "style_without_reason"
    assert v1.raw["contrasts"].get("status") is None
    assert v2.raw["contrasts"]["status"] == {
        "style_without_reason": "primary_confirmatory",
        "style_with_reason": "secondary_confirmatory",
        "interaction": "secondary_confirmatory",
        "content_with_style": "exploratory_descriptive",
        "content_plain": "exploratory_descriptive"}
    assert set(v2.parsed.contrasts.multiplicity.family) == {
        "style_without_reason", "style_with_reason", "interaction"}


# --- terminal punctuation, from a live smoke call -----------------------------
#
# The live `it follows that` v2 smoke returned this NS/NP pair, and the
# validator reported zero errors:
#
#   NS: It follows that I support the option to extend ... baseload plant.
#   NP: I support the option to extend ... baseload plant        <- no full stop
#
# Nothing caught it. The residue check sees words and not punctuation; the
# word-delta budget counts words; the pair-content screen tokenises. Removing
# "It follows that" requires a capitalisation change, not the loss of the
# sentence's own terminator, so this is drift and must fail.

LIVE_ENDORSEMENT = ("I support the option to extend the operating life of the existing "
                    "baseload plant")
LIVE_NS = f"It follows that {LIVE_ENDORSEMENT}."
LIVE_NP_BROKEN = LIVE_ENDORSEMENT                    # exactly as returned: no full stop
LIVE_NP_FIXED = f"{LIVE_ENDORSEMENT[0].upper()}{LIVE_ENDORSEMENT[1:]}."
LIVE_PREMISE = ("The extended plant can deliver full output through any cold spell")


def _live_bodies(np_body: str) -> dict[str, str]:
    return {
        "RS": f"{LIVE_PREMISE}. It follows that {LIVE_ENDORSEMENT}.",
        "RP": f"{LIVE_PREMISE}. {LIVE_ENDORSEMENT[0].upper()}{LIVE_ENDORSEMENT[1:]}.",
        "NS": LIVE_NS,
        "NP": np_body,
    }


def _live_allocation():
    return allocation_for("it follows that", "metadiscursive_inference",
                          "sentence_initial_metadiscursive_v1")


def _live_findings(bodies, v2, segmenter):
    return validate_group(
        SCENARIO, v2.raw["corpus"]["counterargument_opening"],
        block_for(bodies, _live_allocation()), v2, segmenter,
        loc={"decision_id": "energy_fixture_001", "scenario_id": "energy_fixture_001_v1"},
        endorsement=LIVE_ENDORSEMENT)


def test_the_live_pair_with_the_missing_period_now_fails(v2, segmenter):
    findings = _live_findings(_live_bodies(LIVE_NP_BROKEN), v2, segmenter)
    failed = codes(findings)
    assert "E_BODY_TERMINAL_PUNCTUATION" in failed
    assert "E_PAIR_TERMINAL_PUNCTUATION_MISMATCH" in failed

    cell = next(f for f in findings if f.code == "E_BODY_TERMINAL_PUNCTUATION")
    assert cell.condition == "NP"
    assert cell.detail["ends_with"] == "t", "the body ends on a word, not a stop"
    pair = next(f for f in findings if f.code == "E_PAIR_TERMINAL_PUNCTUATION_MISMATCH")
    assert pair.detail["pair"] == ["NS", "NP"]
    assert pair.detail["tails"] == {"NS": ".", "NP": "t"}


def test_the_same_live_pair_passes_once_the_period_is_restored(v2, segmenter):
    findings = _live_findings(_live_bodies(LIVE_NP_FIXED), v2, segmenter)
    assert codes(findings) == [], codes(findings)


def test_a_marker_associated_capitalisation_change_is_still_accepted(v2, segmenter):
    """`Therefore, ...` -> `...` changes capitalisation and drops a comma. That
    is what removing a sentence-initial marker requires, and it stays valid."""
    allocation = allocation_for("therefore", "conclusion_indicator",
                                "sentence_initial_conclusion_v1")
    findings = findings_for(v2_bodies("therefore"), allocation, v2, segmenter)
    assert codes(findings) == []
    bodies = v2_bodies("therefore")
    assert bodies["NS"].startswith("Therefore, ") and bodies["NS"].endswith(".")
    assert bodies["NP"].endswith(".")
    assert bodies["NS"].rstrip()[-1] == bodies["NP"].rstrip()[-1] == "."


@pytest.mark.parametrize("tail,expected", [
    ("", "E_BODY_TERMINAL_PUNCTUATION"),          # nothing at all
    ("..", "E_BODY_TERMINAL_PUNCTUATION"),        # two stops
    ("!", "E_BODY_TERMINAL_PUNCTUATION"),         # the wrong terminator
])
def test_a_body_must_end_with_exactly_one_period(tail, expected, v2, segmenter):
    allocation = allocation_for("therefore", "conclusion_indicator",
                                "sentence_initial_conclusion_v1")
    bodies = v2_bodies("therefore")
    bodies["NP"] = bodies["NP"][:-1] + tail
    assert expected in codes(findings_for(bodies, allocation, v2, segmenter))


def test_both_pairs_are_checked_for_terminal_punctuation(v2, segmenter):
    allocation = allocation_for("therefore", "conclusion_indicator",
                                "sentence_initial_conclusion_v1")
    bodies = v2_bodies("therefore")
    bodies["RP"] = bodies["RP"][:-1]              # the reason-present pair this time
    findings = findings_for(bodies, allocation, v2, segmenter)
    pair = next(f for f in findings if f.code == "E_PAIR_TERMINAL_PUNCTUATION_MISMATCH")
    assert pair.detail["pair"] == ["RS", "RP"]


def test_nothing_is_normalised_or_rewritten_after_receipt(v2, segmenter):
    """The malformed body is reported as it arrived, not quietly repaired."""
    bodies = _live_bodies(LIVE_NP_BROKEN)
    before = dict(bodies)
    findings = _live_findings(bodies, v2, segmenter)
    assert bodies == before, "validation mutates nothing"
    assert any(f.severity == "error" for f in findings), \
        "and an error is what sends it down the existing repair path"


def test_v1_never_runs_the_terminal_punctuation_checks(v1, segmenter):
    allocation = allocation_for("therefore", "conclusion_indicator",
                                "sentence_initial_conclusion_v1")
    bodies = v2_bodies("therefore")
    bodies["NP"] = bodies["NP"][:-1]
    findings = findings_for(bodies, allocation, v1, segmenter)
    for v2_only in ("E_BODY_TERMINAL_PUNCTUATION",
                    "E_PAIR_TERMINAL_PUNCTUATION_MISMATCH"):
        assert v2_only not in codes(findings)
