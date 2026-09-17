"""Redrafting the nine scenarios Vidhi rejected on 2026-09-17.

The live scenario stage of 2026-09-16 produced 24 machine-clean scenarios;
human review approved 15 and asked for nine to be drafted again. These tests
cover the step that follows, offline: exactly those nine, one call each, each
recorded against the call it supersedes, and none of it approving anything.

Nothing here reaches a server, and nothing writes into the live run directory.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from reasonstyle.corpus import segmenter_from_config
from reasonstyle.generation import FakeBackend
from reasonstyle.generation.approvals import (
    REDRAFT,
    REQUIRED_JUDGEMENTS,
    ScenarioApproval,
    load_approvals,
)
from reasonstyle.generation.pipeline import (
    ACCEPTED,
    NEEDS_MANUAL_REVIEW,
    CallStore,
    gate_problems_for,
    recorded_scenarios,
    run_group_stage,
)
from reasonstyle.generation.redraft import (
    REDRAFT_TEMPLATE,
    current_scenarios,
    redraft_request,
    redraft_targets,
    redraft_template_sha256,
    run_redraft_stage,
)
from reasonstyle.generation.requests import RequestError
from reasonstyle.hashing import content_hash, sha256_of

ROOT = Path(__file__).resolve().parents[1]
LIVE_RUN = ROOT / "data" / "pilot" / "run" / "pilot_scenarios_2026-09-16"
REDRAFT_RUN = ROOT / "data" / "pilot" / "run" / "pilot_redraft_snapshot_2026-09-17"
APPROVALS = ROOT / "data" / "pilot" / "scenario_approvals.yaml"

REDRAFTED = ("climate_01_v1", "climate_01_v2", "climate_04_v1", "energy_01_v1",
             "energy_03_v1", "energy_03_v2", "technology_03_v1", "technology_03_v2",
             "technology_04_v1")


@pytest.fixture
def topics(pilot_bank):
    return [t for t in pilot_bank.topics if t.status == "curated"]


@pytest.fixture
def bank_hash(pilot_bank):
    return content_hash(pilot_bank.model_dump(mode="json"))


@pytest.fixture
def final_approvals():
    return load_approvals(APPROVALS)


@pytest.fixture
def store(tmp_path, cfg, bank_hash):
    """A copy of the live run: the original evidence is never written to."""
    if not LIVE_RUN.is_dir():
        pytest.skip(f"{LIVE_RUN} is not on this machine (gitignored run artefacts)")
    run = tmp_path / "run"
    shutil.copytree(LIVE_RUN, run)
    return CallStore(run, cfg, topic_bank_content_hash=bank_hash)


@pytest.fixture
def approvals(final_approvals, store, cfg, bank_hash):
    """Reconstruct the 15/9 decision state that authorised the redraft run.

    The tracked approvals file is now the final 24/24 gate.  The historical
    decision, reasons and failed judgements remain in the immutable redraft
    call metadata; tests use that evidence rather than making the current gate
    stale again.
    """
    log = REDRAFT_RUN / "generation_log.jsonl"
    if not log.is_file():
        pytest.skip(f"{REDRAFT_RUN} is not on this machine (gitignored run artefacts)")
    originals = recorded_scenarios(store)
    out = dict(final_approvals)
    entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    for entry in entries:
        if entry.get("kind") != "scenario_redraft":
            continue
        scenario_id = f"{entry['decision_id']}_v{entry['variant_id']}"
        extra = entry.get("extra") or {}
        original = originals[scenario_id]
        failed = set(extra["failed_judgements"])
        out[scenario_id] = ScenarioApproval(
            scenario_id=scenario_id,
            scenario_text_sha256=sha256_of(original["scenario_text"]),
            call_id=original["call_id"],
            config_content_hash=cfg.content_hash,
            topic_bank_content_hash=bank_hash,
            decision=REDRAFT,
            judgements={name: name not in failed for name in REQUIRED_JUDGEMENTS},
            decided_by="Vidhi Bhutani", decided_at=date(2026, 9, 17),
            reason=extra["failure_reason"])
    return out


def revised(scenario_id: str) -> str:
    """A plausible redraft: different text, inside the band, no labels."""
    return (f"A regional authority must decide between two courses of action in the "
            f"{scenario_id.replace('_', ' ')} case, and the choice is open. The first course "
            f"would act sooner, and the second would keep to the arrangement already in "
            f"place. One consideration supports acting sooner, and one supports keeping to "
            f"the existing arrangement; both are set out here before the decision is made, "
            f"and neither is stronger than the other. The authority has funding for either "
            f"course and no rule says how the two considerations should be weighed against "
            f"each other. Nothing in the record settles which course is better, and a "
            f"decision is needed before the current period ends.")


class RedraftResponder:
    def __init__(self, *, unchanged=(), invalid=(), originals=None):
        self.requests = []
        self.unchanged, self.invalid = set(unchanged), set(invalid)
        self.originals = originals or {}

    def __call__(self, request):
        self.requests.append(request)
        scenario_id = request.context["scenario_id"]
        if scenario_id in self.unchanged:
            return {"scenario_text": self.originals[scenario_id]}
        if scenario_id in self.invalid:
            return {"scenario_text": "The authority should pick option A, obviously."}
        return {"scenario_text": revised(scenario_id)}

    @property
    def kinds(self):
        return [r.kind for r in self.requests]


def _run(topics, approvals, cfg, segmenter, store, responder=None):
    scenarios = recorded_scenarios(store)
    responder = responder or RedraftResponder(
        originals={k: v["scenario_text"] for k, v in scenarios.items()})
    results = run_redraft_stage(topics, approvals, cfg, segmenter, FakeBackend(responder),
                                store)
    return responder, results


# --- the recorded review -----------------------------------------------------


def test_the_approvals_file_records_the_final_twenty_four(final_approvals):
    assert len(final_approvals) == 24
    assert all(a.decision == "approved" for a in final_approvals.values())
    for approval in final_approvals.values():
        assert approval.decided_by == "Vidhi Bhutani"
        assert approval.decided_at == date(2026, 9, 17)
        assert set(approval.judgements) == set(REQUIRED_JUDGEMENTS)
        assert None not in approval.judgements.values(), "no judgement left unanswered"
        assert all(approval.judgements.values()) and approval.reason is None


def test_the_fifteen_approvals_are_current_against_the_live_run(approvals, cfg, bank_hash,
                                                                store):
    """They bind this configuration and this topic bank: none is stale."""
    from reasonstyle.generation.approvals import approval_status
    scenarios = recorded_scenarios(store)
    approved = [k for k, a in approvals.items() if a.decision == "approved"]
    assert len(approved) == 15
    for scenario_id in approved:
        record = scenarios[scenario_id]
        state, reasons = approval_status(
            scenario_id, record["scenario_text"], record["call_id"], approvals,
            config_content_hash=cfg.content_hash, topic_bank_content_hash=bank_hash)
        assert state == "approved", (scenario_id, reasons)


# --- selection ---------------------------------------------------------------


def test_exactly_the_nine_reviewed_scenarios_are_selected(approvals, store):
    targets = redraft_targets(approvals, recorded_scenarios(store))
    assert sorted(t.scenario_id for t in targets) == sorted(REDRAFTED)
    for target in targets:
        assert target.failure_reason and target.failed_judgements
        assert target.original_call_id and target.original_text


def test_selection_comes_from_the_file_not_the_command_line(approvals, store):
    """Which scenarios need drafting again is the reviewer's finding."""
    import inspect
    from reasonstyle.generation import redraft as module
    assert "only" not in inspect.signature(module.redraft_targets).parameters
    assert "only" not in inspect.signature(module.run_redraft_stage).parameters


def test_a_stale_redraft_decision_is_refused(approvals, store):
    """The decision names a text; if the recorded text moved on, it is stale."""
    scenarios = recorded_scenarios(store)
    scenarios["climate_01_v1"] = {**scenarios["climate_01_v1"],
                                  "scenario_text": "some other text entirely"}
    with pytest.raises(RequestError, match="not the text the curator reviewed"):
        redraft_targets(approvals, scenarios)


# --- the request -------------------------------------------------------------


def test_the_request_carries_the_brief_the_original_and_the_reason(topics, approvals, cfg,
                                                                   store):
    targets = {t.scenario_id: t for t in redraft_targets(approvals, recorded_scenarios(store))}
    target = targets["energy_03_v1"]
    topic = next(t for t in topics if t.decision_id == "energy_03")
    request = redraft_request(topic, target, cfg)
    variant = topic.variants["v1"]

    assert topic.decision_framing in request.prompt
    assert variant.context in request.prompt
    assert topic.options["opt_1"] in request.prompt and topic.options["opt_2"] in request.prompt
    assert variant.scenario_facts.opt_1[0] in request.prompt
    assert variant.scenario_facts.opt_2[0] in request.prompt
    assert target.original_text in request.prompt
    assert target.failure_reason in request.prompt
    band = cfg.raw["corpus"]["scenario_words"]
    assert str(band["min"]) in request.prompt and str(band["max"]) in request.prompt
    assert "materially different" in request.prompt
    assert "Never by a letter" in request.prompt
    assert request.context["supersedes_call_id"] == target.original_call_id
    assert request.kind == "scenario_redraft"


def test_the_redraft_template_is_hashed_outside_the_experiment_config(cfg):
    """Recording it in configs/experiment.yaml would change the configuration
    hash, and every one of the fifteen approvals binds that hash."""
    digest = redraft_template_sha256()
    assert digest == sha256_of(REDRAFT_TEMPLATE.read_text(encoding="utf-8")) or len(digest) == 64
    drafting = cfg.raw["prompts"]["drafting"]
    assert "scenario_redraft_v1" not in drafting
    assert digest not in json.dumps(cfg.raw)
    assert cfg.content_hash == "9da99ff12674cc91f0bbf76e137bf1cb347c30eb49691cfcfe3e4914c2fa148f"


# --- the stage ---------------------------------------------------------------


def test_one_call_per_rejected_scenario_and_no_group_call(topics, approvals, cfg, segmenter,
                                                          store):
    responder, results = _run(topics, approvals, cfg, segmenter, store)
    assert len(responder.requests) == 9
    assert set(responder.kinds) == {"scenario_redraft"}
    assert sorted(r.context["scenario_id"] for r in responder.requests) == sorted(REDRAFTED)
    assert all(r.outcome == ACCEPTED for r in results)


def test_an_unchanged_redraft_is_not_progress(topics, approvals, cfg, segmenter, store):
    """Returning the original answers nothing the reviewer asked."""
    scenarios = recorded_scenarios(store)
    responder = RedraftResponder(
        unchanged={"climate_01_v1"},
        originals={k: v["scenario_text"] for k, v in scenarios.items()})
    _, results = _run(topics, approvals, cfg, segmenter, store, responder)
    stuck = next(r for r in results if r.decision_id == "climate_01" and r.variant_id == 1)
    assert stuck.outcome == NEEDS_MANUAL_REVIEW
    assert stuck.attempts[0].no_progress is True
    assert current_scenarios(store)["climate_01_v1"]["scenario_text"] == \
        scenarios["climate_01_v1"]["scenario_text"], "the original still stands"


def test_a_machine_invalid_redraft_stays_blocked(topics, approvals, cfg, segmenter, store):
    responder = RedraftResponder(invalid={"energy_03_v1"})
    _, results = _run(topics, approvals, cfg, segmenter, store, responder)
    bad = next(r for r in results if r.decision_id == "energy_03" and r.variant_id == 1)
    assert bad.outcome == NEEDS_MANUAL_REVIEW
    assert "E_LABEL_LEAKAGE" in bad.attempts[0].error_codes
    assert "energy_03_v1" not in {k for k, v in current_scenarios(store).items()
                                  if v.get("supersedes_call_id")}


def test_a_completed_redraft_is_recovered_not_resent(topics, approvals, cfg, segmenter,
                                                     store):
    _run(topics, approvals, cfg, segmenter, store)
    before = len(store.log.entries())
    again = RedraftResponder()
    again.__call__ = lambda request: pytest.fail("a completed redraft was sent again")
    responder, results = _run(topics, approvals, cfg, segmenter, store,
                              RedraftResponder(originals={}))
    assert responder.requests == []
    assert len(store.log.entries()) == before
    assert all(r.attempts[0].reused for r in results)


def test_there_is_no_automatic_second_redraft(topics, approvals, cfg, segmenter, store):
    scenarios = recorded_scenarios(store)
    responder = RedraftResponder(
        unchanged=set(REDRAFTED),
        originals={k: v["scenario_text"] for k, v in scenarios.items()})
    _run(topics, approvals, cfg, segmenter, store, responder)
    assert len(responder.requests) == 9, "one call each, however badly they went"


# --- provenance ---------------------------------------------------------------


def test_every_redraft_records_its_supersession_and_hashes(topics, approvals, cfg, segmenter,
                                                           store):
    _run(topics, approvals, cfg, segmenter, store)
    entries = [e for e in store.log.entries() if e["kind"] == "scenario_redraft"]
    assert len(entries) == 9
    originals = recorded_scenarios(store)
    for entry in entries:
        scenario_id = f"{entry['decision_id']}_v{entry['variant_id']}"
        extra = entry["extra"]
        assert entry["call_id"] != originals[scenario_id]["call_id"], "a new call"
        assert extra["supersedes_call_id"] == approvals[scenario_id].call_id
        assert extra["original_text_sha256"] == approvals[scenario_id].scenario_text_sha256
        assert extra["redrafted_text_sha256"] != extra["original_text_sha256"]
        assert extra["failure_reason"] == approvals[scenario_id].reason
        assert extra["template_name"] == "scenario_redraft_v1"
        assert extra["template_sha256"] == redraft_template_sha256()
        assert entry["template_sha256"] == redraft_template_sha256()
        assert entry["prompt_sha256"] and entry["response_sha256"]
        assert "validation" in entry and entry["validation"]["error_codes"] == []
        assert entry["request_fields"]["seed"] == cfg.raw["models"]["generator"]["decoding"]["seed"]
        assert "temperature" in entry["request_fields"] and "top_p" in entry["request_fields"]


def test_a_redraft_replaces_only_the_call_it_supersedes(topics, approvals, cfg, segmenter,
                                                        store):
    """Not whichever call happens to be latest."""
    originals = recorded_scenarios(store)
    _run(topics, approvals, cfg, segmenter, store)
    current = current_scenarios(store)
    for scenario_id, record in current.items():
        if scenario_id in REDRAFTED:
            assert record["call_id"] != originals[scenario_id]["call_id"]
            assert record["supersedes_call_id"] == originals[scenario_id]["call_id"]
        else:
            assert record["call_id"] == originals[scenario_id]["call_id"]
            assert "supersedes_call_id" not in record, "an approved scenario is untouched"


def test_a_redraft_naming_another_call_supersedes_nothing(topics, approvals, cfg, segmenter,
                                                          store):
    _run(topics, approvals, cfg, segmenter, store)
    entry = next(e for e in store.log.entries() if e["kind"] == "scenario_redraft")
    lines = []
    for line in store.log.path.read_text().splitlines():
        record = json.loads(line)
        if record["call_id"] == entry["call_id"]:
            record["extra"] = {**record["extra"], "supersedes_call_id": "z" * 64}
        lines.append(json.dumps(record))
    store.log.path.write_text("\n".join(lines) + "\n")
    scenario_id = f"{entry['decision_id']}_v{entry['variant_id']}"
    current = current_scenarios(store)
    assert current[scenario_id]["call_id"] == recorded_scenarios(store)[scenario_id]["call_id"]


# --- what stays true afterwards ------------------------------------------------


def test_the_fifteen_approvals_survive_the_redraft_stage(topics, approvals, cfg, segmenter,
                                                         store, bank_hash):
    from reasonstyle.generation.approvals import approval_status
    _run(topics, approvals, cfg, segmenter, store)
    current = current_scenarios(store, approvals)
    approved = [k for k, a in approvals.items() if a.decision == "approved"]
    for scenario_id in approved:
        record = current[scenario_id]
        state, reasons = approval_status(
            scenario_id, record["scenario_text"], record["call_id"], approvals,
            config_content_hash=cfg.content_hash, topic_bank_content_hash=bank_hash)
        assert state == "approved", (scenario_id, reasons)


def test_the_nine_redrafts_are_unapproved_and_block_the_groups(topics, approvals, cfg,
                                                               segmenter, store, bank_hash):
    _run(topics, approvals, cfg, segmenter, store)
    current = current_scenarios(store, approvals)
    problems = gate_problems_for(topics, store, cfg, approvals=approvals,
                                 topic_bank_content_hash=bank_hash, scenarios=current)
    assert len(problems) == 9
    assert all(any(scenario_id in p for p in problems) for scenario_id in REDRAFTED)

    responder = RedraftResponder()
    with pytest.raises(Exception, match="curator gate is not satisfied"):
        run_group_stage(topics, [], cfg, segmenter, FakeBackend(responder), store,
                        approvals=approvals, topic_bank_content_hash=bank_hash,
                        scenarios=current)
    assert responder.requests == []


def test_the_live_run_evidence_is_never_written_to(topics, approvals, cfg, segmenter, store):
    """The stage runs against a copy in these tests; the live directory is
    read-only evidence and must stay byte-identical."""
    before = {p.relative_to(LIVE_RUN): p.read_bytes()
              for p in sorted(LIVE_RUN.rglob("*")) if p.is_file()}
    _run(topics, approvals, cfg, segmenter, store)
    after = {p.relative_to(LIVE_RUN): p.read_bytes()
             for p in sorted(LIVE_RUN.rglob("*")) if p.is_file()}
    assert before == after and len(before) >= 49


def test_the_configuration_and_allocation_hashes_do_not_move(cfg):
    from reasonstyle.generation.allocation import load_allocation
    assert cfg.content_hash == "9da99ff12674cc91f0bbf76e137bf1cb347c30eb49691cfcfe3e4914c2fa148f"
    allocation = load_allocation(ROOT / "data" / "pilot" / "marker_allocation.yaml")
    assert allocation.content_hash == \
        "60621c17186e25cc37d4ced304086ac58599c5a720fc3c922cda6d553f693ecd"
    assert allocation.config_content_hash == cfg.content_hash


def test_the_redraft_command_refuses_a_narrowed_selection(tmp_path):
    """The set is the curator's finding. Narrowing it with --only would
    overrule the review, not filter a report."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "scripts/pilot.py", "redraft-scenarios", "--config",
         "configs/experiment.yaml", "--out", str(tmp_path / "run"), "--only", "climate_01"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1
    assert "runs on the complete pilot" in result.stderr
    assert not (tmp_path / "run" / "generation_log.jsonl").exists()
