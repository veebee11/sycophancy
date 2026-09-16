"""The repair no-progress failure of the live synthetic smoke (2026-09-16).

The evidence is read-only and gitignored at
``data/pilot/smoke/pipeline_repair_2026-09-16_attempt2/``. What happened: the
scenario was accepted, the group draft failed three rules, and both repairs
returned *the same four bodies* — because both repair requests were the same
request. Their prompt hash was identical
(``bcc545c175bbec26848fe1f3d10148b8631454385ea79c6708d03e1c43f2cb5c``), so a deterministic
server was asked the same question twice and the budget ran out at
``needs_manual_review``.

The bodies below are verbatim. They are not edited or reconstructed here.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from reasonstyle.generation import FakeBackend
from reasonstyle.generation.allocation import GroupAllocation
from reasonstyle.generation.pipeline import (
    ACCEPTED,
    NEEDS_MANUAL_REVIEW,
    CallStore,
    PipelineAbort,
    draft_group,
    group_diagnostics,
)

EVIDENCE = (pathlib.Path(__file__).resolve().parents[1] / "data" / "pilot" / "smoke" /
            "pipeline_repair_2026-09-16_attempt2")

#: The prompt hash both repairs shared. The defect, in one value.
REPEATED_REPAIR_PROMPT = "bcc545c175bbec26848fe1f3d10148b8631454385ea79c6708d03e1c43f2cb5c"

#: Verbatim model output from that run. Do not edit: this is evidence.
STUCK = {
    'RS': 'The extended plant can deliver full output through any cold spell of the coming winters because I prefer to extend the operating life of the existing baseload plant.',
    'RP': 'The extended plant can deliver full output through any cold spell of the coming winters. I prefer to extend the operating life of the existing baseload plant.',
    'NS': 'I prefer to extend the operating life of the existing baseload plant because I find that choice more appealing personally.',
    'NP': 'I prefer to extend the operating life of the existing baseload plant. I find that choice more appealing personally.',
}

SCENARIO = "A regional grid operator must decide how to cover a projected shortfall in firm capacity over the next three winters. The operator makes its initial capacity decision for the coming three winters. One option is to extend the operating life of the existing baseload plant, which can deliver full output through any cold spell of the coming winters. The other option is to accelerate the storage build already under tender, which would retire the plant on schedule and cut the region's power-sector emissions substantially. Both courses of action involve trade-offs between reliability and environmental impact. Each approach has distinct implications for the region's energy security and emissions profile, requiring careful evaluation of their respective benefits and consequences."


@pytest.fixture
def topic(synthetic_bank):
    return next(t for t in synthetic_bank.topics if t.decision_id == "energy_fixture_001")


@pytest.fixture
def allocation():
    return GroupAllocation(
        decision_id="energy_fixture_001", domain="energy", variant_id=1,
        scenario_id="energy_fixture_001_v1", supported_option="opt_1",
        marker_family="premise_indicator", marker_string="because",
        marker_realization_id="clause_initial_premise_v1")


@pytest.fixture
def store(tmp_path, cfg):
    return CallStore(tmp_path / "run", cfg)


class Responder:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        reply = self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]
        return reply(request) if callable(reply) else reply


FIXED = {
    "RS": ("The extended plant can deliver full output through any cold spell. Because that "
           "output holds, the plant extension remains my preferred option."),
    "RP": ("The extended plant can deliver full output through any cold spell. That output "
           "holds, and the plant extension remains my preferred option."),
    "NS": ("I would choose the plant extension in this particular case. Because that is my "
           "view, the plant extension remains my preferred option."),
    "NP": ("I would choose the plant extension in this particular case. That is my view, "
           "and the plant extension remains my preferred option."),
}


def _run(topic, cfg, store, responder, allocation, segmenter):
    return draft_group(topic, 1, SCENARIO, allocation, cfg, segmenter,
                       FakeBackend(responder), store)


def test_the_evidence_still_shows_one_repeated_repair_prompt():
    """Guards the facts these tests are built on."""
    if not EVIDENCE.is_dir():
        pytest.skip(f"{EVIDENCE} is not on this machine (gitignored evidence)")
    log = [json.loads(line) for line
           in (EVIDENCE / "generation_log.jsonl").read_text().splitlines()]
    repairs = [e for e in log if e["kind"] == "repair"]
    assert len(repairs) == 2
    assert {e["prompt_sha256"] for e in repairs} == {REPEATED_REPAIR_PROMPT}
    for entry in log:
        if entry["kind"] in ("group", "repair"):
            assert set(entry["validation"]["error_codes"]) == {
                "E_SENTENCE_COUNT_MISMATCH", "E_WORD_RATIO_BODY", "E_WORD_RATIO_FULL_TEXT"}
    assert [e["outcome"] for e in log] == [
        "accepted", "repair_needed", "repair_needed", "needs_manual_review"]


def test_the_stuck_bodies_still_fail_the_same_three_rules(topic, cfg, segmenter, store,
                                                          allocation):
    result = _run(topic, cfg, store, Responder(STUCK, FIXED), allocation, segmenter)
    assert {"E_SENTENCE_COUNT_MISMATCH", "E_WORD_RATIO_BODY",
            "E_WORD_RATIO_FULL_TEXT"} <= set(result.attempts[0].error_codes)


def test_no_progress_is_detected_and_recorded(topic, cfg, segmenter, store, allocation):
    """A repair that returns what it was given changed nothing, and the record
    says so rather than looking like an ordinary failed attempt."""
    result = _run(topic, cfg, store, Responder(STUCK, STUCK, STUCK), allocation, segmenter)
    assert result.outcome == NEEDS_MANUAL_REVIEW
    assert [a.no_progress for a in result.attempts] == [False, True, True]
    entries = store.log.entries()
    assert [e["validation"]["no_progress"] for e in entries] == [False, True, True]


def test_the_two_repairs_are_not_the_same_request(topic, cfg, segmenter, store, allocation):
    """The defect itself: repair 2 and repair 3 had one prompt hash between
    them. They must now differ."""
    responder = Responder(STUCK, STUCK, STUCK)
    result = _run(topic, cfg, store, responder, allocation, segmenter)
    repairs = [r for r in responder.requests if r.kind == "repair"]
    assert len(repairs) == 2
    assert repairs[0].prompt_sha256 != repairs[1].prompt_sha256
    assert repairs[0].prompt != repairs[1].prompt
    assert [a.prompt_sha256 for a in result.attempts[1:]] == [r.prompt_sha256 for r in repairs]


def test_the_later_repair_names_the_unchanged_response(topic, cfg, segmenter, store,
                                                       allocation):
    responder = Responder(STUCK, STUCK, STUCK)
    _run(topic, cfg, store, responder, allocation, segmenter)
    second_repair = [r for r in responder.requests if r.kind == "repair"][1]
    assert "repair attempt 3" in second_repair.prompt
    assert "UNCHANGED" in second_repair.prompt
    assert "no progress" in second_repair.prompt
    assert "materially different correction" in second_repair.prompt
    # ... and the first repair does not claim a history it does not have.
    first_repair = [r for r in responder.requests if r.kind == "repair"][0]
    assert "This is the first repair of this group." in first_repair.prompt
    assert "UNCHANGED" not in first_repair.prompt


def test_the_prompt_carries_correct_sentence_and_word_diagnostics(cfg, segmenter):
    """Measured from the live bodies: RS and NS are one sentence, RP and NP are
    two, and the rule is two BODY sentences."""
    opening = cfg.raw["corpus"]["counterargument_opening"]
    text = group_diagnostics(STUCK, opening, cfg, segmenter)
    assert "BODY sentences (the rule: exactly 2 per body): RS 1, RP 2, NS 1, NP 2" in text
    assert "full-text sentences, opening included, for reference only: RS 2, RP 3, NS 2, NP 3" \
        in text
    assert "wrong body sentence count: RS, NS" in text

    import re
    word_re = re.compile(cfg.parsed.matching.words.word_regex)
    counts = {c: len(word_re.findall(STUCK[c])) for c in ("RS", "RP", "NS", "NP")}
    assert ("BODY words: " + ", ".join(f"{c} {n}" for c, n in counts.items())) in text
    shortest, longest = min(counts.values()), max(counts.values())
    assert f"shortest body {shortest} words, longest {longest}" in text
    assert "-> shorten to meet the 1.1 target: RS, RP" in text
    assert "-> or lengthen to meet it: NS, NP" in text


def test_the_actionable_word_target_is_the_1_10_matching_target_not_1_15(cfg, segmenter):
    """1.10 is what the drafting prompts ask for; 1.15 is where the validator
    errors. Advice computed from 1.15 would aim a repair at the edge of the
    rule. Both are stated; only the target is actionable."""
    import math
    import re
    warn = cfg.parsed.matching.words.ratio_warn
    fail = cfg.parsed.matching.words.ratio_fail
    assert (warn, fail) == (1.10, 1.15)
    text = group_diagnostics(STUCK, cfg.raw["corpus"]["counterargument_opening"], cfg,
                             segmenter)
    word_re = re.compile(cfg.parsed.matching.words.word_regex)
    counts = {c: len(word_re.findall(STUCK[c])) for c in ("RS", "RP", "NS", "NP")}
    shortest, longest = min(counts.values()), max(counts.values())

    target_max, hard_max = int(shortest * warn), int(shortest * fail)
    target_min = math.ceil(longest / warn)
    assert target_max != hard_max, "the fixture must distinguish the two thresholds"
    assert f"target ratio {warn} (what to aim for)" in text
    assert f"hard-error ceiling {fail} (where the check fails)" in text
    assert f"bring every body to {target_max} words or fewer" in text
    assert f"bring every body to at least {target_min} words" in text
    assert f"the hard ceiling alone would allow up to {hard_max} words" in text
    assert "do not aim there" in text

    # The advice itself is computed from the target: a body at the hard-ceiling
    # length is still named as too long.
    at_hard_ceiling = {**STUCK, "RS": " ".join(["word"] * hard_max)}
    advice = group_diagnostics(at_hard_ceiling, cfg.raw["corpus"]["counterargument_opening"],
                               cfg, segmenter)
    assert f"-> shorten to meet the {warn} target:" in advice
    assert "RS" in advice.split("-> shorten to meet")[1].splitlines()[0]


def test_the_diagnostics_distinguish_body_from_full_text(cfg, segmenter):
    """The live finding reported 2/3 full-text sentences while the rule is two
    BODY sentences; the prompt must not leave that implicit."""
    opening = cfg.raw["corpus"]["counterargument_opening"]
    text = group_diagnostics(STUCK, opening, cfg, segmenter)
    assert text.index("BODY sentences") < text.index("full-text sentences")
    assert "opening included" in text


def test_the_pair_structure_rule_is_stated_in_the_prompt(topic, cfg, segmenter, store,
                                                         allocation):
    responder = Responder(STUCK, FIXED)
    _run(topic, cfg, store, responder, allocation, segmenter)
    repair = [r for r in responder.requests if r.kind == "repair"][0]
    assert "same" in repair.prompt and "sentence-boundary structure" in repair.prompt
    assert "Dropping the connective must not split one" in repair.prompt
    # the substantive restrictions are still there
    assert "free of any task-relevant reason" in repair.prompt
    assert "same content words" in repair.prompt


def test_a_repair_that_works_still_works(topic, cfg, segmenter, store, allocation):
    """The ordinary path is unchanged: one failing draft, one repair, accepted."""
    responder = Responder(STUCK, FIXED)
    result = _run(topic, cfg, store, responder, allocation, segmenter)
    assert result.outcome == ACCEPTED and result.calls_made == 2
    assert [a.no_progress for a in result.attempts] == [False, False]


def test_the_budget_is_still_three_calls_for_a_group(topic, cfg, segmenter, store, allocation):
    responder = Responder(STUCK, STUCK, STUCK, FIXED)
    result = _run(topic, cfg, store, responder, allocation, segmenter)
    assert result.calls_made == 3 == cfg.raw["corpus"]["repair"]["max_calls_per_group"]
    assert len(responder.requests) == 3
    assert result.outcome == NEEDS_MANUAL_REVIEW


def test_recovery_does_not_resend_a_completed_stuck_call(topic, cfg, segmenter, store,
                                                         allocation):
    _run(topic, cfg, store, Responder(STUCK, STUCK, STUCK), allocation, segmenter)
    before = [e["call_id"] for e in store.log.entries()]

    again = Responder(lambda request: pytest.fail("a completed call was sent again"))
    result = _run(topic, cfg, store, again, allocation, segmenter)
    assert again.requests == []
    assert result.outcome == NEEDS_MANUAL_REVIEW
    assert [e["call_id"] for e in store.log.entries()] == before


def test_an_identical_repair_request_can_never_be_sent(topic, cfg, segmenter, store,
                                                       allocation, monkeypatch):
    """Belt and braces: if a future edit ever made two repair prompts identical
    again — which is exactly what happened live — the controller refuses to
    spend the call rather than asking the same question twice."""
    import reasonstyle.generation.pipeline as pipeline
    original = pipeline.repair_request
    built = []

    def always_the_same(*args, **kwargs):
        if not built:
            built.append(original(*args, **kwargs))
        return built[0]

    monkeypatch.setattr(pipeline, "repair_request", always_the_same)
    with pytest.raises(PipelineAbort, match="repeat the previous failed repair"):
        _run(topic, cfg, store, Responder(STUCK, STUCK, STUCK), allocation, segmenter)
