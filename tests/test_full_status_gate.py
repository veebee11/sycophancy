"""``pilot.py status`` on the full configuration must derive its NEW MATERIAL
and next-gate wording from what is actually recorded, not from a stored claim
that predates the scenario stage.

Everything here runs against an isolated copy of the frozen configuration,
with its run directory and approvals file redirected under ``tmp_path``: the
real ``data/full/run_v2`` and ``data/full/scenario_approvals_full_v2.yaml`` are
never opened for writing, only the frozen files this repository ships are read
(the config text itself, the topic bank, the marker allocation, the pilot
seed). No request is made and no credential is read.
"""

from __future__ import annotations

import importlib.util
import re
from datetime import date
from pathlib import Path

import pytest

from reasonstyle.config import load_config
from reasonstyle.corpus import segmenter_from_config
from reasonstyle.corpus.topics import load_topic_bank
from reasonstyle.generation import FakeBackend
from reasonstyle.generation.allocation import load_allocation
from reasonstyle.generation.approvals import (
    APPROVED,
    BLOCKED,
    NEEDS_MANUAL_REVIEW,
    PENDING,
    REDRAFT,
    REQUIRED_JUDGEMENTS,
    STALE,
    ScenarioApproval,
    save_approvals,
)
from reasonstyle.generation.corpus_source import load_seed_corpus, plan_full_corpus
from reasonstyle.generation.pipeline import CallStore, recorded_scenarios, run_scenario_stage
from reasonstyle.hashing import content_hash, sha256_of

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "configs/frozen/v2_full.yaml"
BANK = load_topic_bank(ROOT / "data/topics/full_topics_v2.yaml")
REDRAFT_IDS = ("technology_08_v1", "technology_13_v1")


def _load_pilot_script():
    spec = importlib.util.spec_from_file_location("pilot_script_status_gate", ROOT / "scripts/pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _isolated_config(tmp_path: Path) -> Path:
    """A copy of the frozen full configuration with its run directory and
    approvals file redirected into ``tmp_path``. Every other path — the
    allocation, the seed pilot corpus, the topic bank — still names the real
    committed files, opened read-only."""
    text = FULL.read_text(encoding="utf-8")
    text = text.replace("  run: data/full/run_v2\n", f"  run: {tmp_path / 'run_v2'}\n", 1)
    text = text.replace(
        "  approvals: data/full/scenario_approvals_full_v2.yaml\n",
        f"  approvals: {tmp_path / 'scenario_approvals_full_v2.yaml'}\n", 1)
    assert str(tmp_path) in text and "data/full/run_v2" not in text.split("paths:", 1)[1].split(
        "\n\n", 1)[0]
    # ``load_config`` refuses a ``status: frozen`` file whose parent directory
    # is not literally named ``frozen``, or whose filename is not its version
    # (see ``config._check_location``); both are satisfied here, in a location
    # distinct from the real ``configs/frozen/``.
    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir(exist_ok=True)
    path = frozen_dir / "v2_full.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def scenario_text(scenario_id: str) -> str:
    """Distinct per scenario; long enough and free of any forbidden or leakage
    pattern, so every one of the 96 is genuinely machine-valid — matching what
    the live run actually produced."""
    return (f"Record {scenario_id} sets out a regional capacity decision for the coming "
            f"winters. The operator can extend the existing baseload plant or accelerate "
            f"the storage build already under tender. Both routes are funded and feasible, "
            f"and nothing states how reliability should be weighed against emissions. The "
            f"extended plant can deliver full output through any cold spell. Retiring it on "
            f"schedule would cut power-sector emissions. One route must be chosen before "
            f"the tender closes.")


class ScenarioOnlyResponder:
    """Answers every scenario call with valid synthetic text; refuses anything else."""

    def __call__(self, request):
        assert request.kind == "scenario", f"unexpected {request.kind} call"
        scenario_id = f"{request.decision_id}_v{request.variant_id}"
        return {"scenario_text": scenario_text(scenario_id)}


@pytest.fixture
def cfg_path(tmp_path):
    return _isolated_config(tmp_path)


@pytest.fixture
def cfg(cfg_path):
    return load_config(cfg_path)


@pytest.fixture
def seed(cfg):
    return load_seed_corpus(cfg, bank=BANK, root=ROOT)


@pytest.fixture
def plan(cfg, seed):
    return plan_full_corpus(cfg, BANK, seed)


@pytest.fixture
def bank_hash():
    return content_hash(BANK.model_dump(mode="json"))


@pytest.fixture
def allocation():
    return load_allocation(ROOT / "data/full/marker_allocation_full_v2.yaml")


def _new_topics(plan):
    return [t for t in BANK.topics if t.status == "curated" and t.decision_id in set(plan.new_decisions)]


# --- no full scenario run present ---------------------------------------------------


def test_status_with_no_run_output(cfg_path, tmp_path, capsys):
    """A fresh checkout: no run directory, no approvals file. ``status`` must
    say so plainly and point at the scenario stage, not at review or redrafts."""
    pilot = _load_pilot_script()
    assert not (tmp_path / "run_v2").exists()
    assert pilot.main(["status", "--config", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "NEW MATERIAL (not generated)" in out
    assert "scenario review" not in out
    assert "scenarios            0 of 96 generated" in out
    assert "next gate            run the authorised scenario stage: 0 of 96 scenario call(s) made" in out
    # still unauthorised, exactly as when a run exists
    assert "group drafting, redrafts and repairs are NOT authorised" in out


# --- a completed 96-scenario run, reviewed as 94 approved / 2 redraft --------------


@pytest.fixture
def completed_run(tmp_path, cfg, plan, bank_hash, allocation):
    """Every one of the 96 new scenarios drafted (via FakeBackend, offline) and
    then reviewed: 94 approved, 2 (``technology_08_v1``, ``technology_13_v1``)
    sent to redraft — the same shape as the real, live 2026-09-23 run."""
    store = CallStore(tmp_path / "run_v2", cfg, topic_bank_content_hash=bank_hash,
                      allocation_content_hash=allocation.content_hash)
    segmenter = segmenter_from_config(cfg)
    run_scenario_stage(_new_topics(plan), cfg, segmenter, FakeBackend(ScenarioOnlyResponder()),
                       store)
    scenarios = recorded_scenarios(store)
    assert len(scenarios) == 96
    assert all(not (record.get("error_codes")) for record in scenarios.values()), (
        "the synthetic scenario text must be machine-valid, exactly like the live run")

    approvals: dict[str, ScenarioApproval] = {}
    for scenario_id, record in scenarios.items():
        if scenario_id in REDRAFT_IDS:
            judgements = {name: True for name in REQUIRED_JUDGEMENTS}
            judgements["no_added_facts_or_quantities"] = False
            approvals[scenario_id] = ScenarioApproval(
                scenario_id=scenario_id, scenario_text_sha256=sha256_of(record["scenario_text"]),
                call_id=record["call_id"], config_content_hash=cfg.content_hash,
                topic_bank_content_hash=bank_hash, decision="redraft", judgements=judgements,
                decided_by="Vidhi Bhutani", decided_at=date(2026, 9, 23),
                reason="synthetic redraft reason")
        else:
            approvals[scenario_id] = ScenarioApproval(
                scenario_id=scenario_id, scenario_text_sha256=sha256_of(record["scenario_text"]),
                call_id=record["call_id"], config_content_hash=cfg.content_hash,
                topic_bank_content_hash=bank_hash, decision="approved",
                judgements={name: True for name in REQUIRED_JUDGEMENTS},
                decided_by="Vidhi Bhutani", decided_at=date(2026, 9, 23), reason=None)
    save_approvals(approvals, tmp_path / "scenario_approvals_full_v2.yaml")
    return store


def test_status_with_completed_run_and_review(cfg_path, completed_run, capsys):
    pilot = _load_pilot_script()
    assert pilot.main(["status", "--config", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "NEW MATERIAL (not generated)" not in out and "NEW MATERIAL" in out
    assert "scenarios            96 of 96 generated" in out
    assert "scenario review      94 approved, 2 redraft" in out
    assert ("next gate            decide whether to authorise exactly the 2 redraft call(s): "
           "technology_08_v1, technology_13_v1") in out
    assert "group drafting, redrafts and repairs are NOT authorised" in out
    assert "groups               0 of 192 generated" in out


def test_approvals_command_agrees_with_status(cfg_path, plan, completed_run, capsys):
    """The two read-only reports must not contradict each other: this is the
    defect that prompted the fix, and it must not recur.

    ``approvals`` without ``--only`` reports over every curated topic in the
    bank, including the 24 imported-seed scenarios it never drafted itself
    (those show as ``not_generated`` — a separate, known export-filtering
    concern, not this gate). Filtered to the 48 new decisions, exactly as the
    real review was run, it must count the same 94/2 as ``status``.
    """
    pilot = _load_pilot_script()

    assert pilot.main(["status", "--config", str(cfg_path)]) == 0
    status_out = capsys.readouterr().out

    assert pilot.main(["approvals", "--config", str(cfg_path),
                       "--only", *sorted(plan.new_decisions)]) == 0
    approvals_out = capsys.readouterr().out

    per_scenario_approved = re.findall(r"^\s+\S+\s+approved$", approvals_out, re.MULTILINE)
    per_scenario_redraft = re.findall(r"^\s+\S+\s+redraft\s", approvals_out, re.MULTILINE)
    assert len(per_scenario_approved) == 94
    assert len(per_scenario_redraft) == 2
    for scenario_id in REDRAFT_IDS:
        assert scenario_id in approvals_out
    assert "94 approved" in status_out
    assert "2 redraft" in status_out
    assert "2 of 96 expected scenario(s) not approved" in approvals_out
    for state in ("pending", "not_generated", "stale", "blocked_by_machine_errors"):
        assert state not in status_out
        assert not re.search(rf"^\s+\S+\s+{state}\b", approvals_out, re.MULTILINE)


# --- _next_gate: every approval_status() state, and their priority ----------------
#
# Unit-level and fully isolated: `_next_gate` is a pure function over counts and
# a redraft-id list, so these need no config, no run directory and no backend at
# all — just the real state-constant strings from `approvals.py`, the module
# this function's counts are actually built from.


NG_TOTAL = 96
NG_GROUP_TOTAL = 192


def _gate_for(**counts_and_ids):
    counts = {k: v for k, v in counts_and_ids.items() if k != "redraft_ids"}
    redraft_ids = counts_and_ids.get("redraft_ids", [])
    pilot = _load_pilot_script()
    return pilot._next_gate(NG_TOTAL, NG_TOTAL, 0, NG_GROUP_TOTAL, counts, redraft_ids)


def test_next_gate_nothing_drafted_yet():
    pilot = _load_pilot_script()
    assert pilot._next_gate(0, NG_TOTAL, 0, NG_GROUP_TOTAL, {}, []) == (
        f"run the authorised scenario stage: 0 of {NG_TOTAL} scenario call(s) made")


def test_next_gate_partially_drafted():
    pilot = _load_pilot_script()
    assert pilot._next_gate(40, NG_TOTAL, 0, NG_GROUP_TOTAL, {APPROVED: 0}, []) == (
        f"finish the scenario stage: 40 of {NG_TOTAL} scenario call(s) made")


def test_next_gate_needs_manual_review_leads_to_human_review():
    gate = _gate_for(**{APPROVED: 90, NEEDS_MANUAL_REVIEW: 6})
    assert gate == (f"scenario review of the {NG_TOTAL} drafted scenarios "
                    f"(6 still need a decision)")


def test_next_gate_pending_and_stale_also_lead_to_human_review():
    gate = _gate_for(**{APPROVED: 90, PENDING: 4, STALE: 2})
    assert "6 still need a decision" in gate
    assert gate.startswith("scenario review")


def test_next_gate_machine_blocked_scenario_prevents_group_authorisation():
    gate = _gate_for(**{APPROVED: 94, BLOCKED: 2})
    assert "2 scenario(s) blocked by machine errors" in gate
    assert "group drafting authorisation" not in gate
    assert "decide whether to authorise" not in gate


def test_next_gate_blocked_outranks_redraft():
    """A machine error can never be approved past (`approvals.py`); a blocked
    scenario must never be silently outvoted by a pile of ready redrafts."""
    gate = _gate_for(redraft_ids=["a_v1", "b_v1"], **{APPROVED: 92, REDRAFT: 2, BLOCKED: 2})
    assert gate.startswith("2 scenario(s) blocked by machine errors")
    assert "decide whether to authorise" not in gate


def test_next_gate_blocked_outranks_unresolved_human_review_too():
    gate = _gate_for(**{APPROVED: 90, BLOCKED: 1, NEEDS_MANUAL_REVIEW: 5})
    assert gate.startswith("1 scenario(s) blocked by machine errors")


def test_next_gate_mixed_blocked_and_redraft_reports_only_the_blocker():
    gate = _gate_for(redraft_ids=["climate_06_v1"], **{APPROVED: 93, REDRAFT: 1, BLOCKED: 2,
                                                         PENDING: 0})
    assert gate.startswith("2 scenario(s) blocked by machine errors")


def test_next_gate_redraft_is_the_gate_only_once_nothing_outranks_it():
    gate = _gate_for(redraft_ids=sorted(REDRAFT_IDS), **{APPROVED: 94, REDRAFT: 2})
    assert gate == (f"decide whether to authorise exactly the 2 redraft call(s): "
                    f"{', '.join(sorted(REDRAFT_IDS))}")


def test_next_gate_all_approved_offers_group_authorisation():
    gate = _gate_for(**{APPROVED: NG_TOTAL})
    assert gate == "group drafting authorisation (every scenario is approved)"


def test_next_gate_all_approved_and_groups_complete_moves_to_assembly():
    pilot = _load_pilot_script()
    gate = pilot._next_gate(NG_TOTAL, NG_TOTAL, NG_GROUP_TOTAL, NG_GROUP_TOTAL,
                            {APPROVED: NG_TOTAL}, [])
    assert gate == "corpus assembly"


def test_next_gate_never_claims_all_approved_on_an_incomplete_count():
    """Every scenario is drafted, nothing is blocked, unreviewed or sent to
    redraft — but the approved count still falls short of the total. This
    should never happen (the states are meant to be exhaustive and additive),
    but if it does, `_next_gate` must not paper over it by offering group
    authorisation anyway."""
    gate = _gate_for(**{APPROVED: 90})
    assert "group drafting authorisation" not in gate
    assert "do not add up" in gate
    assert "90" in gate and str(NG_TOTAL) in gate


def test_next_gate_never_claims_all_approved_on_an_unrecognised_state():
    """A future ``approval_status`` return value this function has never seen
    must not be silently folded into "approved"."""
    gate = _gate_for(**{APPROVED: 90, "some_future_state": 6})
    assert "group drafting authorisation" not in gate
    assert "do not add up" in gate
