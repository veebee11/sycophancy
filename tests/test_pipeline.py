"""The bounded drafting controller: budgets, stopping, recording and resume.

No test reaches a network or a GPU: every call goes through ``FakeBackend``,
whose responder decides what the model "returned" for each attempt. A test
therefore states a failure sequence directly — draft fails, first repair
succeeds — and asserts what the controller did about it.
"""

from __future__ import annotations

import json

import pytest

from reasonstyle.corpus import segmenter_from_config
from reasonstyle.generation import FakeBackend
from reasonstyle.generation.allocation import GroupAllocation
from reasonstyle.generation.pipeline import (
    ACCEPTED,
    NEEDS_MANUAL_REVIEW,
    REFUSED,
    SKIPPED_SCENARIO_NOT_ACCEPTED,
    CallStore,
    PipelineAbort,
    draft_group,
    draft_scenario,
    run_pilot,
)

SCENARIO = ("A regional grid operator must decide how to cover a projected shortfall in firm "
            "capacity over the next three winters. It can extend the operating life of the "
            "existing baseload plant, or it can accelerate the storage build already under "
            "tender. Both routes are funded and technically feasible, and the operating licence "
            "does not state how supply reliability should be set against the region's emissions "
            "trajectory. The extended plant can deliver full output through any cold spell of "
            "the coming winters. Retiring the plant on schedule would cut the region's "
            "power-sector emissions substantially. The operator must choose one route before "
            "the tender closes.")

#: A group that passes every machine rule: two sentences each, matched word
#: counts, the marker in the styled cells only, and pairs that differ solely by
#: the marker and the permitted linking word.
VALID = {
    "RS": ("The extended plant can deliver full output through any cold spell. Because that "
           "output holds, the plant extension remains my preferred option."),
    "RP": ("The extended plant can deliver full output through any cold spell. That output "
           "holds, and the plant extension remains my preferred option."),
    "NS": ("I would choose the plant extension in this particular case. Because that is my "
           "view, the plant extension remains my preferred option."),
    "NP": ("I would choose the plant extension in this particular case. That is my view, "
           "and the plant extension remains my preferred option."),
}

#: One sentence in RS instead of two, and far too short: breaks the sentence
#: rule and both word ratios at once — the shape of the second live smoke call.
BROKEN = {**VALID, "RS": "Because that output holds, the plant extension remains my option."}


@pytest.fixture
def topic(synthetic_bank):
    """The synthetic fixture brief — never a pilot brief."""
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
    """Answers each call from a scripted sequence, and counts what it was asked."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        reply = self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]
        return reply(request) if callable(reply) else reply

    @property
    def kinds(self):
        return [r.kind for r in self.requests]


def _group(topic, cfg, store, responder, allocation, segmenter):
    backend = FakeBackend(responder)
    return draft_group(topic, 1, SCENARIO, allocation, cfg, segmenter, backend, store)


# --- the budget --------------------------------------------------------------


def test_a_clean_draft_is_accepted_in_one_call(topic, cfg, segmenter, store, allocation):
    responder = Responder(VALID)
    result = _group(topic, cfg, store, responder, allocation, segmenter)
    assert result.outcome == ACCEPTED
    assert result.calls_made == 1 and responder.kinds == ["group"]
    assert result.payload == VALID


def test_a_failing_draft_is_repaired_and_the_repair_is_accepted(topic, cfg, segmenter, store,
                                                                allocation):
    """The draft breaks sentence count and word balance; the first repair fixes
    both, and the controller stops there."""
    responder = Responder(BROKEN, VALID)
    result = _group(topic, cfg, store, responder, allocation, segmenter)
    assert responder.kinds == ["group", "repair"]
    assert result.outcome == ACCEPTED and result.calls_made == 2
    first, second = result.attempts
    assert first.outcome == "repair_needed"
    assert {"E_SENTENCE_COUNT_MISMATCH", "E_WORD_RATIO_BODY"} <= set(first.error_codes)
    assert second.attempt == 2 and second.error_codes == ()


def test_the_repair_prompt_carries_the_validator_findings(topic, cfg, segmenter, store,
                                                          allocation):
    responder = Responder(BROKEN, VALID)
    _group(topic, cfg, store, responder, allocation, segmenter)
    repair = responder.requests[1]
    assert repair.kind == "repair"
    assert any("E_SENTENCE_COUNT_MISMATCH" in f for f in repair.context["findings"])
    assert all(f.startswith("E_") for f in repair.context["findings"]), "warnings never repair"
    for body in BROKEN.values():                     # the bodies being repaired
        assert body in repair.prompt


def test_two_failed_repairs_end_in_needs_manual_review_and_no_fourth_call(
        topic, cfg, segmenter, store, allocation):
    responder = Responder(BROKEN, BROKEN, BROKEN, VALID)
    result = _group(topic, cfg, store, responder, allocation, segmenter)
    assert result.outcome == NEEDS_MANUAL_REVIEW
    assert result.calls_made == 3 == cfg.raw["corpus"]["repair"]["max_calls_per_group"]
    assert responder.kinds == ["group", "repair", "repair"], "no fourth call"
    assert len(responder.requests) == 3
    assert result.attempts[-1].outcome == NEEDS_MANUAL_REVIEW


def test_a_schema_rejection_is_followed_by_a_draft_not_a_repair(topic, cfg, segmenter, store,
                                                                allocation):
    """There are no bodies to repair, so the next call is a fresh draft with its
    own call id."""
    responder = Responder({"RS": "only one key"}, VALID)
    result = _group(topic, cfg, store, responder, allocation, segmenter)
    assert responder.kinds == ["group", "group"]
    assert responder.requests[0].call_id != responder.requests[1].call_id
    assert result.attempts[0].status == "rejected"
    assert result.outcome == ACCEPTED


def test_a_transport_failure_is_audited_but_completes_no_attempt(topic, cfg, segmenter, store,
                                                                 allocation):
    """The server was unreachable: nothing was generated, so nothing may be
    recorded as a finished call — but the failure stays visible in the log."""
    from reasonstyle.generation.backends import BackendUnavailable
    from reasonstyle.generation.pipeline import TRANSPORT_ERROR
    responder = Responder(lambda request: BackendUnavailable("no vLLM server"))
    with pytest.raises(PipelineAbort, match="no vLLM server"):
        _group(topic, cfg, store, responder, allocation, segmenter)
    entry = store.log.entries()[0]
    assert entry["status"] == "error" and entry["outcome"] == TRANSPORT_ERROR
    assert entry["validation"]["consumed_budget"] is False
    assert not store.result_path(entry["call_id"]).is_file(), "no completed-call artefact"


def test_a_transport_failure_is_retried_after_a_restart(topic, cfg, segmenter, store,
                                                        allocation):
    """The same attempt may contact the backend again, keeps its budget position
    and, when a response finally arrives, is recorded normally."""
    from reasonstyle.generation.backends import BackendUnavailable
    from reasonstyle.generation.pipeline import TRANSPORT_ERROR
    failing = Responder(lambda request: BackendUnavailable("no vLLM server"))
    with pytest.raises(PipelineAbort):
        _group(topic, cfg, store, failing, allocation, segmenter)

    after = Responder(BROKEN, VALID)
    result = _group(topic, cfg, store, after, allocation, segmenter)
    assert after.kinds == ["group", "repair"], "the failed attempt was sent again"
    assert result.outcome == ACCEPTED
    assert result.calls_made == 2, "the transport failure took no budget position"
    outcomes = [e["outcome"] for e in store.log.entries()]
    assert outcomes == [TRANSPORT_ERROR, "repair_needed", ACCEPTED]


def test_an_unauthorised_run_sends_nothing(topic, cfg, segmenter, store, allocation):
    from reasonstyle.generation import VLLMOpenAIBackend
    backend = VLLMOpenAIBackend(_urlopen=_no_network)
    result = draft_group(topic, 1, SCENARIO, allocation, cfg, segmenter, backend, store)
    assert result.outcome == REFUSED and result.calls_made == 1
    assert result.payload is None


def _no_network(*args, **kwargs):                    # pragma: no cover - must never run
    raise AssertionError("a test reached the network")


# --- repairs may not wander --------------------------------------------------


def test_a_repair_that_changes_unrelated_valid_content_is_caught(topic, cfg, segmenter, store,
                                                                 allocation):
    """The repair fixes RS but quietly rewrites NS, so NS and NP no longer share
    their content words. The group is not accepted on the strength of the fix."""
    wandering = {**VALID,
                 "NS": ("I would choose the plant extension in this particular case. Because "
                        "that is my view, the reliable plant extension remains my option.")}
    responder = Responder(BROKEN, wandering)
    result = _group(topic, cfg, store, responder, allocation, segmenter)
    assert result.outcome != ACCEPTED
    assert "E_PAIR_CONTENT_DRIFT" in result.attempts[1].error_codes


def test_the_marker_fields_come_from_the_allocation_not_the_response(topic, cfg, segmenter,
                                                                     store, allocation):
    """A response cannot change the design: the marker is checked against the
    allocated one, so a group written with a different marker fails."""
    other = {**VALID,
             "RS": ("The extended plant can deliver full output through any cold spell. "
                    "Therefore that output holds, and the plant extension remains my option.")}
    responder = Responder(other, other, other)
    result = _group(topic, cfg, store, responder, allocation, segmenter)
    assert result.outcome == NEEDS_MANUAL_REVIEW
    assert "E_MARKER_MISSING_IN_STYLED_CELL" in result.attempts[0].error_codes


# --- what is written down ----------------------------------------------------


def test_every_call_is_recorded_separately_with_its_attempt_and_outcome(
        topic, cfg, segmenter, store, allocation):
    responder = Responder(BROKEN, VALID)
    result = _group(topic, cfg, store, responder, allocation, segmenter)
    entries = store.log.entries()
    assert [e["attempt"] for e in entries] == [1, 2]
    assert [e["kind"] for e in entries] == ["group", "repair"]
    assert [e["outcome"] for e in entries] == ["repair_needed", ACCEPTED]
    assert len({e["call_id"] for e in entries}) == 2
    for entry, attempt in zip(entries, result.attempts):
        assert entry["prompt_sha256"] == attempt.prompt_sha256
        assert entry["validation"]["error_codes"] == list(attempt.error_codes)
        assert store.raw_path(entry["call_id"]).is_file()
        saved = json.loads(store.result_path(entry["call_id"]).read_text())
        assert saved["outcome"] == entry["outcome"] and saved["attempt"] == entry["attempt"]


def test_machine_valid_is_never_recorded_as_human_approval(topic, cfg, segmenter, store,
                                                           allocation):
    result = _group(topic, cfg, store, Responder(VALID), allocation, segmenter)
    assert result.outcome == ACCEPTED
    entry = store.log.entries()[0]
    assert entry["validation"]["machine_valid"] is True
    assert "approved" not in json.dumps(entry).replace("_approved", "")
    outstanding = {f.code for f in result.findings if f.severity == "human_review"}
    assert {"H_PROPOSITION_PRESERVATION", "H_NO_REASON_INTEGRITY",
            "H_SUPPORT_DIRECTION"} <= outstanding


def test_the_log_line_carries_the_whole_provenance_block(topic, cfg, segmenter, tmp_path,
                                                         allocation, synthetic_bank):
    """Everything needed to say what produced a draft, on every live call."""
    from reasonstyle.generation.environment import CachedModel
    from reasonstyle.hashing import content_hash
    cached = CachedModel(repo_id="Qwen/Qwen3-14B", revision="a" * 40,
                         snapshot_path="/data/vidhi/hf_cache/hub/snap", refs=("main",),
                         size_bytes=1, weight_files=("model.safetensors",))
    server = {"gpu_index": "0", "gpu_name": "NVIDIA RTX A6000", "dtype": "bfloat16",
              "revision": "a" * 40, "generation_config": "vllm", "seed": 20260914,
              "libraries": {"vllm": "0.8.5.post1+cu118", "transformers": "4.51.3",
                            "torch": "2.6.0+cu118"}}
    store = CallStore(tmp_path / "run", cfg, cached=cached, server=server,
                      endpoint="http://127.0.0.1:8011/v1",
                      topic_bank_content_hash=content_hash(synthetic_bank.model_dump(mode="json")),
                      allocation_content_hash="alloc-hash")
    _group(topic, cfg, store, Responder(VALID), allocation, segmenter)
    entry = store.log.entries()[0]

    # the real request fields, not a summary of them
    decoding = cfg.raw["models"]["generator"]["decoding"]
    for field in ("model", "temperature", "top_p", "max_tokens", "seed", "top_k", "min_p",
                  "repetition_penalty", "presence_penalty", "frequency_penalty",
                  "response_format"):
        assert field in entry["request_fields"], field
    assert "messages" not in entry["request_fields"], "the prompt is stored raw, not twice"
    assert entry["request_fields"]["seed"] == decoding["seed"]
    assert entry["request_fields"]["response_format"]["json_schema"]["strict"] is True

    assert entry["model_revision"] == "a" * 40
    assert entry["runtime"]["snapshot_path"].endswith("snap")
    assert entry["runtime"]["server"]["libraries"]["vllm"] == "0.8.5.post1+cu118"
    assert entry["gpu"] == "NVIDIA RTX A6000" and entry["runtime"]["dtype"] == "bfloat16"
    assert entry["seed"] == decoding["seed"]
    assert "HF_HUB_OFFLINE" in entry["runtime"]["offline_env"]
    assert entry["topic_bank_content_hash"] and entry["allocation_content_hash"] == "alloc-hash"
    assert entry["prompt_sha256"] and entry["response_sha256"]
    assert entry["model_returned"] == "fake-backend"
    assert entry["stop_reason"] == "stop" and entry["usage"]["completion_tokens"] > 0


def test_a_logged_call_can_be_checked_back_against_its_brief(topic, cfg, segmenter, store,
                                                             allocation, synthetic_bank):
    """`check_request_provenance` rebuilds the request from the brief and the
    allocation, so a logged call that did not come from them is detectable."""
    from reasonstyle.generation import check_request_provenance
    from reasonstyle.generation.allocation import MarkerAllocation
    from reasonstyle.hashing import content_hash
    store.topic_bank_content_hash = content_hash(synthetic_bank.model_dump(mode="json"))
    alloc = MarkerAllocation(groups=(allocation,), seed=0, config_content_hash=cfg.content_hash,
                             topic_bank_content_hash=store.topic_bank_content_hash)
    store.allocation_content_hash = alloc.content_hash
    _group(topic, cfg, store, Responder(VALID), allocation, segmenter)
    entry = store.log.entries()[0]

    good = check_request_provenance(entry, synthetic_bank, alloc, cfg, scenario_text=SCENARIO)
    assert good.matched, good.problems
    tampered = {**entry, "prompt_sha256": "0" * 64}
    assert not check_request_provenance(tampered, synthetic_bank, alloc, cfg,
                                        scenario_text=SCENARIO).matched


def test_the_synthetic_path_rebuilds_both_its_requests_from_the_brief(
        topic, cfg, segmenter, tmp_path, allocation, synthetic_bank):
    """What `scripts/pipeline_smoke.py` reports: the scenario rebuilds from the
    brief, and the group rebuilds from the brief plus the exact accepted
    scenario text it was given."""
    from reasonstyle.generation import check_request_provenance
    from reasonstyle.generation.allocation import MarkerAllocation
    from reasonstyle.hashing import content_hash
    bank_hash = content_hash(synthetic_bank.model_dump(mode="json"))
    store = CallStore(tmp_path / "run", cfg, topic_bank_content_hash=bank_hash)
    alloc = MarkerAllocation(groups=(allocation,), seed=0, config_content_hash=cfg.content_hash,
                             topic_bank_content_hash=bank_hash)

    responder = Responder({"scenario_text": SCENARIO}, BROKEN, VALID)
    backend = FakeBackend(responder)
    scenario = draft_scenario(topic, 1, cfg, segmenter, backend, store)
    assert scenario.accepted
    group = draft_group(topic, 1, scenario.payload["scenario_text"], allocation, cfg,
                        segmenter, backend, store)
    assert group.outcome == ACCEPTED and [a.kind for a in group.attempts] == ["group", "repair"]

    by_kind = {e["kind"]: e for e in store.log.entries()}
    assert check_request_provenance(by_kind["scenario"], synthetic_bank, alloc, cfg).matched
    checked = check_request_provenance(by_kind["group"], synthetic_bank, alloc, cfg,
                                       scenario_text=scenario.payload["scenario_text"])
    assert checked.matched, checked.problems

    # A different scenario text rebuilds a different prompt, and is caught.
    wrong = check_request_provenance(by_kind["group"], synthetic_bank, alloc, cfg,
                                     scenario_text=SCENARIO + " One more sentence.")
    assert not wrong.matched

    # A repair is NOT rebuildable through this checker: it only compares the
    # hashes and skips the rebuild, so a "matched" repair result would say far
    # less than it appears to. That is why the smoke script prints an explicit
    # "provenance n/a" for repairs instead of "ok" or a blank.
    from reasonstyle.generation.provenance import ProvenanceMismatch, rebuild_request
    with pytest.raises(ProvenanceMismatch, match="carry their own inputs"):
        rebuild_request(by_kind["repair"], synthetic_bank, alloc, cfg,
                        scenario_text=scenario.payload["scenario_text"])


def test_a_transport_failure_records_everything_known_before_the_request(
        topic, cfg, segmenter, tmp_path, allocation, synthetic_bank):
    """A call that never returned is still fully described by what was sent."""
    from reasonstyle.generation.backends import BackendUnavailable
    from reasonstyle.generation.environment import CachedModel
    from reasonstyle.generation.pipeline import TRANSPORT_ERROR
    from reasonstyle.hashing import content_hash
    cached = CachedModel(repo_id="Qwen/Qwen3-14B", revision="a" * 40,
                         snapshot_path="/data/vidhi/hf_cache/hub/snap", refs=("main",),
                         size_bytes=1, weight_files=("model.safetensors",))
    server = {"gpu_index": "0", "gpu_name": "NVIDIA RTX A6000", "dtype": "bfloat16",
              "revision": "a" * 40, "generation_config": "vllm", "seed": 20260914,
              "libraries": {"vllm": "0.8.5.post1+cu118", "transformers": "4.51.3",
                            "torch": "2.6.0+cu118"}}
    store = CallStore(tmp_path / "run", cfg, cached=cached, server=server,
                      endpoint="http://127.0.0.1:8011/v1",
                      topic_bank_content_hash=content_hash(synthetic_bank.model_dump(mode="json")),
                      allocation_content_hash="alloc-hash")

    with pytest.raises(PipelineAbort):
        _group(topic, cfg, store,
               Responder(lambda request: BackendUnavailable("no vLLM server")),
               allocation, segmenter)
    entry = store.log.entries()[0]
    assert entry["outcome"] == TRANSPORT_ERROR
    for field in ("model", "temperature", "top_p", "max_tokens", "seed", "top_k",
                  "response_format"):
        assert field in entry["request_fields"], field
    assert entry["model_revision"] == "a" * 40
    assert entry["runtime"]["snapshot_path"].endswith("snap")
    assert entry["runtime"]["server"]["libraries"]["torch"] == "2.6.0+cu118"
    assert entry["gpu"] == "NVIDIA RTX A6000" and entry["runtime"]["dtype"] == "bfloat16"
    assert entry["seed"] == cfg.raw["models"]["generator"]["decoding"]["seed"]
    assert "HF_HUB_OFFLINE" in entry["runtime"]["offline_env"]
    assert entry["topic_bank_content_hash"] and entry["allocation_content_hash"] == "alloc-hash"
    assert entry["prompt_sha256"]
    assert entry["validation"]["consumed_budget"] is False
    # Only what a response would have carried is missing.
    assert entry["model_returned"] is None and entry["stop_reason"] is None
    assert entry["usage"] is None and entry["response_sha256"] is None

    # And the same budget position is retried successfully after a restart.
    result = _group(topic, cfg, store, Responder(VALID), allocation, segmenter)
    assert result.outcome == ACCEPTED and result.calls_made == 1
    assert [e["outcome"] for e in store.log.entries()] == [TRANSPORT_ERROR, ACCEPTED]


# --- resume ------------------------------------------------------------------


def test_a_restart_does_not_repeat_a_completed_call(topic, cfg, segmenter, store, allocation):
    first = Responder(BROKEN, VALID)
    _group(topic, cfg, store, first, allocation, segmenter)
    assert len(first.requests) == 2

    second = Responder(lambda request: pytest.fail("a completed call was sent again"))
    result = _group(topic, cfg, store, second, allocation, segmenter)
    assert second.requests == []
    assert result.outcome == ACCEPTED and all(a.reused for a in result.attempts)
    assert len(store.log.entries()) == 2, "no second log line for the same call"


def _crash_after_raw(store, call_id):
    """Leave only the raw response: no result file, no log line."""
    store.result_path(call_id).unlink()
    store.log.path.unlink()


def _crash_after_result(store, call_id):
    """Leave the raw response and the result: only the log line is missing."""
    store.log.path.unlink()


def test_a_call_interrupted_after_its_raw_write_is_recovered_not_resent(
        topic, cfg, segmenter, store, allocation):
    """The crash window: raw traffic on disk, no result and no log line."""
    _group(topic, cfg, store, Responder(VALID), allocation, segmenter)
    _crash_after_raw(store, store.log.entries()[0]["call_id"])

    again = Responder(lambda request: pytest.fail("the call was sent again"))
    result = _group(topic, cfg, store, again, allocation, segmenter)
    assert again.requests == []
    assert result.outcome == ACCEPTED and result.payload == VALID
    entries = store.log.entries()
    assert len(entries) == 1 and entries[0]["outcome"] == ACCEPTED, \
        "an accepted recovered call must appear exactly once in the log"


def test_a_call_interrupted_after_its_result_rebuilds_only_the_log_line(
        topic, cfg, segmenter, store, allocation):
    _group(topic, cfg, store, Responder(VALID), allocation, segmenter)
    call_id = store.log.entries()[0]["call_id"]
    _crash_after_result(store, call_id)

    again = Responder(lambda request: pytest.fail("the call was sent again"))
    result = _group(topic, cfg, store, again, allocation, segmenter)
    assert again.requests == [] and result.outcome == ACCEPTED
    entries = store.log.entries()
    assert len(entries) == 1 and entries[0]["call_id"] == call_id
    assert entries[0]["validation"]["recovered_after_interruption"] is True


def test_recovery_reapplies_the_request_schema_to_a_malformed_response(
        topic, cfg, segmenter, store, allocation):
    """A stored response that breaks the schema recovers as `rejected`, never as
    an accepted draft: the crash must not launder it."""
    responder = Responder({"RS": "only one key"}, VALID)
    _group(topic, cfg, store, responder, allocation, segmenter)
    first_call = store.log.entries()[0]["call_id"]
    for call_id in [e["call_id"] for e in store.log.entries()]:
        store.result_path(call_id).unlink()
    store.log.path.unlink()

    again = Responder(lambda request: pytest.fail("a stored response was sent again"))
    result = _group(topic, cfg, store, again, allocation, segmenter)
    assert again.requests == []
    assert result.attempts[0].call_id == first_call
    assert result.attempts[0].status == "rejected", "re-parsed under the same schema"
    assert result.outcome == ACCEPTED, "the second, valid stored response still stands"


def test_recovery_treats_a_truncated_response_as_rejected(topic, cfg, segmenter, store,
                                                          allocation):
    """`finish_reason` other than "stop" is read from the stored call metadata,
    so a truncated response cannot recover as a complete draft."""
    _group(topic, cfg, store, Responder(VALID), allocation, segmenter)
    call_id = store.log.entries()[0]["call_id"]
    raw_path = store.raw_path(call_id)
    raw = json.loads(raw_path.read_text())
    raw["call_meta"]["stop_reason"] = "length"
    raw_path.write_text(json.dumps(raw))
    _crash_after_raw(store, call_id)

    again = Responder(VALID)
    result = _group(topic, cfg, store, again, allocation, segmenter)
    assert result.attempts[0].status == "rejected"
    assert "length" in (result.attempts[0].error or "")
    # The truncated call was re-judged from disk, not re-sent; only the fresh
    # draft that follows it reaches the backend.
    assert [r.attempt for r in again.requests] == [2]
    assert result.outcome == ACCEPTED


def test_a_valid_raw_response_recovers_with_its_provenance_intact(topic, cfg, segmenter,
                                                                  store, allocation):
    _group(topic, cfg, store, Responder(VALID), allocation, segmenter)
    call_id = store.log.entries()[0]["call_id"]
    before = store.log.entries()[0]
    _crash_after_raw(store, call_id)
    _group(topic, cfg, store, Responder(VALID), allocation, segmenter)
    after = store.log.entries()[0]
    for field in ("call_id", "prompt_sha256", "template_sha256", "attempt", "outcome",
                  "response_sha256", "stop_reason"):
        assert after[field] == before[field], field


# --- the curator gate --------------------------------------------------------


def _approval(cfg, bank_hash, scenario_id, text, call_id, **overrides):
    from datetime import date
    from reasonstyle.generation.approvals import REQUIRED_JUDGEMENTS, ScenarioApproval
    from reasonstyle.hashing import sha256_of
    base = dict(scenario_id=scenario_id, scenario_text_sha256=sha256_of(text),
                call_id=call_id, config_content_hash=cfg.content_hash,
                topic_bank_content_hash=bank_hash, decision="approved",
                judgements={n: True for n in REQUIRED_JUDGEMENTS},
                decided_by="Vidhi Bhutani", decided_at=date(2026, 9, 20), reason=None)
    return ScenarioApproval(**{**base, **overrides})


def _both_allocations(allocation):
    return [allocation,
            type(allocation)(**{**allocation.as_dict(), "supported_option": "opt_2"})]


def _two_variant_allocations(allocation):
    out = []
    for variant_id in (1, 2):
        for option in ("opt_1", "opt_2"):
            out.append(type(allocation)(**{
                **allocation.as_dict(), "variant_id": variant_id, "supported_option": option,
                "scenario_id": f"energy_fixture_001_v{variant_id}"}))
    return out


def test_one_unapproved_scenario_stops_the_groups_of_the_approved_one(
        topic, cfg, segmenter, store, allocation, synthetic_bank):
    """All-or-nothing, with two scenarios: v1 is approved against its exact
    text, v2 is not. Neither gets a group call — splitting the set would make
    "the pilot" mean whichever half passed first, and would unbalance the
    marker allocation."""
    from reasonstyle.hashing import content_hash
    bank_hash = content_hash(synthetic_bank.model_dump(mode="json"))
    other = SCENARIO.replace("three winters", "four winters")
    responder = Responder({"scenario_text": SCENARIO}, {"scenario_text": other},
                          VALID, VALID, VALID, VALID)
    backend = FakeBackend(responder)

    first = draft_scenario(topic, 1, cfg, segmenter, backend, store)
    approvals = {"energy_fixture_001_v1": _approval(
        cfg, bank_hash, "energy_fixture_001_v1", SCENARIO, first.attempts[-1].call_id)}

    results = run_pilot([topic], _two_variant_allocations(allocation), cfg, segmenter,
                        backend, store, variants=(1, 2), approvals=approvals,
                        topic_bank_content_hash=bank_hash)
    scenarios = [r for r in results if r.kind == "scenario"]
    groups = [r for r in results if r.kind == "group"]
    assert [r.outcome for r in scenarios] == [ACCEPTED, ACCEPTED]
    assert len(groups) == 4 and not any(g.outcome == ACCEPTED for g in groups)
    # v1 is approved and still waits; v2 says why the set is held up.
    v1 = [g.outcome for g in groups if g.variant_id == 1]
    v2 = [g.outcome for g in groups if g.variant_id == 2]
    assert set(v1) == {"skipped_gate_not_satisfied"}
    assert set(v2) == {"skipped_scenario_not_approved:pending"}
    assert responder.kinds.count("group") == 0


def test_a_stale_approval_in_the_set_also_stops_every_group(topic, cfg, segmenter, store,
                                                            allocation, synthetic_bank):
    from reasonstyle.hashing import content_hash
    bank_hash = content_hash(synthetic_bank.model_dump(mode="json"))
    other = SCENARIO.replace("three winters", "four winters")
    responder = Responder({"scenario_text": SCENARIO}, {"scenario_text": other},
                          VALID, VALID, VALID, VALID)
    backend = FakeBackend(responder)
    first = draft_scenario(topic, 1, cfg, segmenter, backend, store)
    second = draft_scenario(topic, 2, cfg, segmenter, backend, store)
    approvals = {
        "energy_fixture_001_v1": _approval(cfg, bank_hash, "energy_fixture_001_v1", SCENARIO,
                                           first.attempts[-1].call_id),
        # approved against text that is not what was drafted
        "energy_fixture_001_v2": _approval(cfg, bank_hash, "energy_fixture_001_v2",
                                           "a scenario nobody drafted",
                                           second.attempts[-1].call_id),
    }
    results = run_pilot([topic], _two_variant_allocations(allocation), cfg, segmenter,
                        backend, store, variants=(1, 2), approvals=approvals,
                        topic_bank_content_hash=bank_hash)
    groups = [r for r in results if r.kind == "group"]
    assert {g.outcome for g in groups if g.variant_id == 2} == {
        "skipped_scenario_not_approved:stale"}
    assert {g.outcome for g in groups if g.variant_id == 1} == {"skipped_gate_not_satisfied"}
    assert responder.kinds.count("group") == 0


def test_the_two_stage_workflow_resumes_without_resending_scenarios(
        topic, cfg, segmenter, store, allocation, synthetic_bank):
    """First invocation: scenarios only. The curator approves. Second
    invocation: the scenario calls are recovered, not re-sent, and the groups
    are drafted."""
    from reasonstyle.hashing import content_hash
    bank_hash = content_hash(synthetic_bank.model_dump(mode="json"))
    other = SCENARIO.replace("three winters", "four winters")
    first_run = Responder({"scenario_text": SCENARIO}, {"scenario_text": other})
    results = run_pilot([topic], _two_variant_allocations(allocation), cfg, segmenter,
                        FakeBackend(first_run), store, variants=(1, 2))
    assert first_run.kinds == ["scenario", "scenario"], "no group call in stage one"
    assert all(r.outcome.startswith("skipped") for r in results if r.kind == "group")

    scenarios = {r.variant_id: r for r in results if r.kind == "scenario"}
    approvals = {
        f"energy_fixture_001_v{v}": _approval(
            cfg, bank_hash, f"energy_fixture_001_v{v}", text,
            scenarios[v].attempts[-1].call_id)
        for v, text in ((1, SCENARIO), (2, other))}

    second_run = Responder(VALID)
    resumed = run_pilot([topic], _two_variant_allocations(allocation), cfg, segmenter,
                        FakeBackend(second_run), store, variants=(1, 2), approvals=approvals,
                        topic_bank_content_hash=bank_hash)
    assert "scenario" not in second_run.kinds, "scenarios were recovered, not re-sent"
    assert second_run.kinds == ["group"] * 4
    assert [r.outcome for r in resumed if r.kind == "group"] == [ACCEPTED] * 4
    assert all(r.attempts[0].reused for r in resumed if r.kind == "scenario")


@pytest.mark.parametrize("gate", [
    {"approvals": None, "topic_bank_content_hash": "abc"},
    {"approvals": {}, "topic_bank_content_hash": None},
    {"approvals": None, "topic_bank_content_hash": None},
])
def test_a_missing_gate_never_permits_group_drafting(gate, topic, cfg, segmenter, store,
                                                     allocation):
    """No approvals, or no topic-bank hash to bind them to, is a gate failure —
    never an invitation to proceed."""
    responder = Responder({"scenario_text": SCENARIO}, VALID, VALID)
    results = run_pilot([topic], _both_allocations(allocation), cfg, segmenter,
                        FakeBackend(responder), store, variants=(1,), **gate)
    assert responder.kinds == ["scenario"]
    assert all(r.outcome.startswith("skipped") for r in results if r.kind == "group")


def test_groups_are_not_drafted_from_an_unapproved_scenario(topic, cfg, segmenter, store,
                                                            allocation, synthetic_bank):
    """The gate: a scenario the curator has not approved takes its own groups
    out of the run, exactly as a failing one does."""
    from reasonstyle.hashing import content_hash
    bank_hash = content_hash(synthetic_bank.model_dump(mode="json"))
    responder = Responder({"scenario_text": SCENARIO}, VALID, VALID)
    results = run_pilot([topic], [allocation], cfg, segmenter, FakeBackend(responder), store,
                        variants=(1,), approvals={}, topic_bank_content_hash=bank_hash)
    assert results[0].outcome == ACCEPTED, "the scenario itself was machine-valid"
    assert all(r.outcome.startswith("skipped_scenario_not_approved") for r in results[1:])
    assert responder.kinds == ["scenario"], "no group call was made"


def test_an_approved_scenario_lets_its_groups_be_drafted(topic, cfg, segmenter, store,
                                                         allocation, synthetic_bank):
    from reasonstyle.hashing import content_hash
    bank_hash = content_hash(synthetic_bank.model_dump(mode="json"))
    responder = Responder({"scenario_text": SCENARIO}, VALID, VALID)
    backend = FakeBackend(responder)
    scenario = draft_scenario(topic, 1, cfg, segmenter, backend, store)
    approvals = {"energy_fixture_001_v1": _approval(
        cfg, bank_hash, "energy_fixture_001_v1", SCENARIO, scenario.attempts[-1].call_id)}

    both = [allocation,
            type(allocation)(**{**allocation.as_dict(), "supported_option": "opt_2"})]
    results = run_pilot([topic], both, cfg, segmenter, backend, store,
                        variants=(1,), approvals=approvals,
                        topic_bank_content_hash=bank_hash)
    assert [r.outcome for r in results] == [ACCEPTED, ACCEPTED, ACCEPTED]
    assert responder.kinds.count("group") == 2


def test_an_approval_for_different_text_does_not_open_the_gate(topic, cfg, segmenter, store,
                                                               allocation, synthetic_bank):
    """Approving one text and generating from another is what the hash binding
    exists to prevent."""
    from reasonstyle.hashing import content_hash
    bank_hash = content_hash(synthetic_bank.model_dump(mode="json"))
    responder = Responder({"scenario_text": SCENARIO}, VALID, VALID)
    approvals = {"energy_fixture_001_v1": _approval(
        cfg, bank_hash, "energy_fixture_001_v1", "some other scenario text", "z" * 64)}
    results = run_pilot([topic], [allocation], cfg, segmenter, FakeBackend(responder), store,
                        variants=(1,), approvals=approvals,
                        topic_bank_content_hash=bank_hash)
    assert all(r.outcome.endswith("stale") for r in results[1:])
    assert responder.kinds == ["scenario"]


# --- the scenario gate -------------------------------------------------------


def test_a_scenario_is_one_call_and_has_no_repair_path(topic, cfg, segmenter, store):
    responder = Responder({"scenario_text": SCENARIO})
    result = draft_scenario(topic, 1, cfg, segmenter, FakeBackend(responder), store)
    assert result.outcome == ACCEPTED and result.calls_made == 1
    assert responder.kinds == ["scenario"]
    assert result.payload["scenario_text"] == SCENARIO


def test_a_failing_scenario_stops_its_own_groups(topic, cfg, segmenter, store, allocation):
    """A scenario with a display label leaks the experiment's A/B labels. No
    counterargument may be drafted on top of it."""
    bad = SCENARIO + " The operator should pick option A."
    responder = Responder({"scenario_text": bad}, VALID, VALID)
    results = run_pilot([topic], [allocation], cfg, segmenter, FakeBackend(responder), store,
                        variants=(1,))
    scenario, *groups = results
    assert scenario.outcome == NEEDS_MANUAL_REVIEW
    assert "E_LABEL_LEAKAGE" in scenario.attempts[0].error_codes
    assert {g.outcome for g in groups} == {SKIPPED_SCENARIO_NOT_ACCEPTED}
    assert responder.kinds == ["scenario"], "no group call was made"


def test_an_accepted_scenario_feeds_its_own_text_to_the_groups(topic, cfg, segmenter, store,
                                                               allocation, synthetic_bank):
    from reasonstyle.hashing import content_hash
    bank_hash = content_hash(synthetic_bank.model_dump(mode="json"))
    responder = Responder({"scenario_text": SCENARIO}, VALID, VALID)
    backend = FakeBackend(responder)
    drafted = draft_scenario(topic, 1, cfg, segmenter, backend, store)
    approvals = {"energy_fixture_001_v1": _approval(
        cfg, bank_hash, "energy_fixture_001_v1", SCENARIO, drafted.attempts[-1].call_id)}
    results = run_pilot([topic], _both_allocations(allocation),
                        cfg, segmenter, backend, store, variants=(1,), approvals=approvals,
                        topic_bank_content_hash=bank_hash)
    assert [r.outcome for r in results] == [ACCEPTED, ACCEPTED, ACCEPTED]
    assert responder.kinds == ["scenario", "group", "group"]
    for request in responder.requests[1:]:
        assert SCENARIO in request.prompt


def test_the_pilot_stages_need_both_authorisations(cfg):
    """A live stage needs --send, the local-generation key and the pilot key.
    Pilot generation is a separate decision from one smoke call, and the CLI
    tests in tests/test_pipeline_cli.py prove no other command can send."""
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "pilot_cli", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    every_key = {"REASONSTYLE_ALLOW_LOCAL_GENERATION": "1", "HF_HUB_OFFLINE": "1",
                 module.PILOT_AUTHORIZATION_ENV: "1"}
    assert module.live_problems(True, cfg, env=every_key) == []
    assert module.live_problems(False, cfg, env=every_key), "--send is still required"
    for missing in every_key:
        problems = module.live_problems(True, cfg, env={**every_key, missing: ""})
        assert any(missing in p for p in problems), missing
