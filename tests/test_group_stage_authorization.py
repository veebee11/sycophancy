"""Tests for the initial group-stage authorisation mechanism — the group
counterpart of ``tests/test_stage_authorization.py``.

``group_stage_targets``, ``run_initial_group_stage``,
``transport_blocked_group_targets`` and ``group_stage_authorization_problems``
are generic — not tied to any one configuration's scale — so proving them
correct needs only a small, fast, self-contained scenario-and-group setup,
not a synthetic replay of the full 96-scenario/192-group corpus. These tests
use the lightweight pilot-scale fixtures already shared by the test suite
(``cfg``, ``pilot_bank``, ``segmenter`` from ``tests/conftest.py``) and a
real ``allocate_markers`` allocation over two of the pilot's curated
decisions, entirely inside ``tmp_path``.

The real, committed ``data/full/authorizations/group_stage_v1.yaml`` and its
binding to the real 96-scenario/192-group full-v2 data is checked separately,
read-only, at the end of this file. Nothing here makes a request or reads a
credential, and no frozen or tracked file is ever written to.
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pytest
import yaml

from reasonstyle.config import load_config
from reasonstyle.generation import FakeBackend
from reasonstyle.generation.allocation import allocate_markers, load_allocation
from reasonstyle.generation.approvals import (
    REQUIRED_JUDGEMENTS,
    ScenarioApproval,
    load_approvals,
    save_approvals,
)
from reasonstyle.generation.corpus_source import load_seed_corpus, plan_full_corpus
from reasonstyle.generation.pipeline import (
    ACCEPTED,
    GROUP_STAGE_AUTHORIZATION_RECORD_VERSION,
    GROUP_STAGE_KIND,
    CallStore,
    GroupStageTarget,
    PipelineAbort,
    group_stage_targets,
    recorded_scenarios,
    run_initial_group_stage,
    run_scenario_stage,
    transport_blocked_group_targets,
)
from reasonstyle.generation.requests import RequestError
from reasonstyle.generation.stage_authorization import (
    StageAuthorizationError,
    group_stage_authorization_problems,
    load_stage_authorization,
)
from reasonstyle.hashing import content_hash, file_sha256, sha256_of

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "configs/frozen/v2_full.yaml"
FROZEN_CONTENT_HASH = "7548650b42e7cb7407f521d10d9e8c2d4f9f25b5074d98cc523b4a5e35a99c94"
REAL_APPROVALS = ROOT / "data/full/scenario_approvals_full_v2.yaml"
REAL_ALLOCATION = ROOT / "data/full/marker_allocation_full_v2.yaml"
REAL_ALLOCATION_CONTENT_HASH = "ba39dbcb24f53678fcaf1f4d43cc4a244ddd549adc94e547e8264391173a3658"
REAL_RUN_V2 = ROOT / "data/full/run_v2"
REAL_RECORD = ROOT / "data/full/authorizations/group_stage_v1.yaml"


def _load_pilot_script():
    spec = importlib.util.spec_from_file_location(
        "pilot_script_group_stage_auth", ROOT / "scripts/pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scenario_text(scenario_id: str) -> str:
    return (f"Record {scenario_id} sets out a regional capacity decision for the coming "
            f"winters. The operator can extend the existing baseload plant or accelerate "
            f"the storage build already under tender. Both routes are funded and feasible, "
            f"and nothing states how reliability should be weighed against emissions. The "
            f"extended plant can deliver full output through any cold spell. Retiring it on "
            f"schedule would cut power-sector emissions. One route must be chosen before "
            f"the tender closes.")


def bodies_for(marker: str, tag: str) -> dict[str, str]:
    """Four cells that pass sentence, ratio, marker and pair-content rules —
    reused verbatim from ``tests/test_pilot_orchestration.py``."""
    styled = marker[0].upper() + marker[1:]
    reason_opening = (f"The {tag} record sets out one relevant consideration for the route "
                      f"under discussion in this particular case")
    plain_opening = (f"I would choose the route under discussion in the {tag} case for no "
                     f"stated reason at all")
    return {
        "RS": (f"{reason_opening}. "
               f"{styled} the consideration holds, that route remains my preferred option."),
        "RP": (f"{reason_opening}. "
               f"The consideration holds, and that route remains my preferred option."),
        "NS": (f"{plain_opening}. "
               f"{styled} my view holds, that route remains my preferred option."),
        "NP": (f"{plain_opening}. "
               f"My view holds, and that route remains my preferred option."),
    }


class ScenarioOnlyResponder:
    def __call__(self, request):
        assert request.kind == "scenario"
        scenario_id = f"{request.decision_id}_v{request.variant_id}"
        return {"scenario_text": scenario_text(scenario_id)}


class GroupResponder:
    """Answers every group call with valid synthetic bodies."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def __call__(self, request):
        scenario_id = f"{request.decision_id}_v{request.variant_id}"
        self.calls.append((scenario_id, request.supported_option))
        tag = f"{scenario_id} {request.supported_option}".replace("_", " ")
        return bodies_for(request.context["marker_string"], tag)


# --- pilot-scale fixtures: two decisions, four scenarios, eight groups ------------


@pytest.fixture
def two_topics(pilot_bank):
    curated = sorted((t for t in pilot_bank.topics if t.status == "curated"),
                     key=lambda t: t.decision_id)
    return curated[:2]


@pytest.fixture
def allocation(pilot_bank, cfg):
    return allocate_markers(pilot_bank, cfg)


@pytest.fixture
def bank_hash(pilot_bank):
    return content_hash(pilot_bank.model_dump(mode="json"))


@pytest.fixture
def approvals_path(tmp_path):
    return tmp_path / "scenario_approvals.yaml"


@pytest.fixture
def store_with_scenarios(tmp_path, cfg, two_topics, bank_hash, allocation, segmenter):
    store = CallStore(tmp_path / "run", cfg, topic_bank_content_hash=bank_hash,
                      allocation_content_hash=allocation.content_hash)
    run_scenario_stage(two_topics, cfg, segmenter, FakeBackend(ScenarioOnlyResponder()), store)
    return store


@pytest.fixture
def approved_scenarios(store_with_scenarios, cfg, bank_hash, approvals_path):
    """94 approved... no — every one of the 4 scenarios (2 decisions x 2
    variants) approved, written to ``approvals_path``. Returns the dict of
    ``ScenarioApproval``."""
    scenarios = recorded_scenarios(store_with_scenarios)
    assert len(scenarios) == 4
    approvals = {}
    for scenario_id, record in scenarios.items():
        approvals[scenario_id] = ScenarioApproval(
            scenario_id=scenario_id, scenario_text_sha256=sha256_of(record["scenario_text"]),
            call_id=record["call_id"], config_content_hash=cfg.content_hash,
            topic_bank_content_hash=bank_hash, decision="approved",
            judgements={name: True for name in REQUIRED_JUDGEMENTS},
            decided_by="Vidhi Bhutani", decided_at=date(2026, 9, 24), reason=None)
    save_approvals(approvals, approvals_path)
    return approvals


@pytest.fixture
def new_groups(two_topics):
    return tuple((f"{t.decision_id}_v{v}", o) for t in two_topics for v in (1, 2)
                for o in ("opt_1", "opt_2"))


@pytest.fixture
def snapshot(store_with_scenarios, approved_scenarios):
    """The current-scenario snapshot every target is verified against."""
    return recorded_scenarios(store_with_scenarios)


@pytest.fixture
def targets(new_groups, approved_scenarios, allocation, cfg, bank_hash, snapshot):
    return group_stage_targets(new_groups, approved_scenarios, allocation,
                               config_content_hash=cfg.content_hash,
                               topic_bank_content_hash=bank_hash, scenarios=snapshot)


def _baseline_record(cfg, approvals_path, allocation, targets) -> dict:
    by_scenario: dict[str, dict] = {}
    for t in targets:
        by_scenario.setdefault(t.scenario_id, {})[t.supported_option] = {
            "scenario_call_id": t.scenario_call_id,
            "scenario_text_sha256": t.scenario_text_sha256,
            "marker_family": t.marker_family,
            "marker_string": t.marker_string,
            "marker_realization_id": t.marker_realization_id,
        }
    return {
        "record_version": GROUP_STAGE_AUTHORIZATION_RECORD_VERSION,
        "status": "authorized",
        "authorized_by": "Test Reviewer",
        "authorized_at": "2026-09-24",
        "config": {"config_version": cfg.config_version, "config_content_hash": cfg.content_hash},
        "kind": GROUP_STAGE_KIND,
        "max_paid_calls": len(targets),
        "calls_per_target": 1,
        "cells_per_call": ["RS", "RP", "NS", "NP"],
        "approvals_file": str(approvals_path),
        "approvals_file_sha256": file_sha256(approvals_path),
        "allocation_file": "allocation.yaml",
        "allocation_content_hash": allocation.content_hash,
        "targets": by_scenario,
        "excludes": ["repairs", "scenario calls"],
    }


def _write_record(tmp_path: Path, record: dict, name: str = "group_auth.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    return path


def _problems(record, cfg, targets, approvals_path, allocation):
    return group_stage_authorization_problems(
        record, cfg=cfg, kind=GROUP_STAGE_KIND,
        record_version=GROUP_STAGE_AUTHORIZATION_RECORD_VERSION, targets=targets,
        approvals_path=approvals_path, allocation_path="allocation.yaml",
        allocation_content_hash=allocation.content_hash)


# --- group_stage_targets: builder correctness and refusal -------------------------


def test_group_stage_targets_builds_two_per_scenario(targets, two_topics):
    assert len(targets) == 8
    from collections import Counter
    per_scenario = Counter(t.scenario_id for t in targets)
    assert len(per_scenario) == 4
    assert all(n == 2 for n in per_scenario.values())
    assert all({t.supported_option for t in targets if t.scenario_id == sid} == {"opt_1", "opt_2"}
              for sid in per_scenario)


def test_group_stage_targets_refuses_an_unapproved_scenario(new_groups, allocation, cfg,
                                                            bank_hash, snapshot):
    with pytest.raises(RequestError, match="not approved"):
        group_stage_targets(new_groups, {}, allocation, config_content_hash=cfg.content_hash,
                            topic_bank_content_hash=bank_hash, scenarios=snapshot)


def test_group_stage_targets_refuses_a_different_configuration_hash(
        new_groups, approved_scenarios, allocation, bank_hash, snapshot):
    with pytest.raises(RequestError, match="different configuration"):
        group_stage_targets(new_groups, approved_scenarios, allocation,
                            config_content_hash="0" * 64, topic_bank_content_hash=bank_hash,
                            scenarios=snapshot)


def test_targets_bind_the_snapshot_call_and_text(targets, snapshot):
    for t in targets:
        assert t.scenario_call_id == snapshot[t.scenario_id]["call_id"]
        assert t.scenario_text_sha256 == sha256_of(snapshot[t.scenario_id]["scenario_text"])


def _drift(snapshot: dict, scenario_id: str, how: str) -> dict:
    """A copy of ``snapshot`` with one scenario's current evidence drifted."""
    drifted = {k: dict(v) for k, v in snapshot.items()}
    record = drifted[scenario_id]
    if how == "altered_text":
        record["scenario_text"] = record["scenario_text"] + " One extra sentence."
    elif how == "wrong_call_id":
        record["call_id"] = "0" * 64
    elif how == "missing_text":
        record["scenario_text"] = ""
    elif how == "missing_record":
        del drifted[scenario_id]
    elif how == "not_accepted":
        record["outcome"] = "needs_manual_review"
    elif how == "machine_errors":
        record["error_codes"] = ["scenario_too_short"]
    else:                                                  # pragma: no cover
        raise AssertionError(how)
    return drifted


DRIFTS = ["altered_text", "wrong_call_id", "missing_text", "missing_record", "not_accepted",
          "machine_errors"]


@pytest.mark.parametrize("how", DRIFTS)
def test_group_stage_targets_refuses_drifted_current_evidence(
        new_groups, approved_scenarios, allocation, cfg, bank_hash, snapshot, how):
    scenario_id = sorted(snapshot)[0]
    with pytest.raises(RequestError, match=scenario_id):
        group_stage_targets(new_groups, approved_scenarios, allocation,
                            config_content_hash=cfg.content_hash,
                            topic_bank_content_hash=bank_hash,
                            scenarios=_drift(snapshot, scenario_id, how))


def test_group_stage_targets_refuses_a_stale_topic_bank_binding(
        new_groups, approved_scenarios, allocation, cfg, snapshot):
    with pytest.raises(RequestError, match="different topic bank"):
        group_stage_targets(new_groups, approved_scenarios, allocation,
                            config_content_hash=cfg.content_hash,
                            topic_bank_content_hash="0" * 64, scenarios=snapshot)


@pytest.mark.parametrize("how", DRIFTS)
def test_drafting_from_a_snapshot_other_than_the_verified_one_dispatches_nothing(
        tmp_path, cfg, approvals_path, allocation, targets, store_with_scenarios, snapshot,
        segmenter, two_topics, how):
    """Even with valid targets and a valid record, handing
    ``run_initial_group_stage`` drifted scenario evidence aborts before the
    first call: the prompt text must be the text the targets were bound to."""
    record = load_stage_authorization(
        _write_record(tmp_path, _baseline_record(cfg, approvals_path, allocation, targets)))
    store = store_with_scenarios
    store.allowed_kinds = frozenset({"group"})
    responder = GroupResponder()
    with pytest.raises(PipelineAbort, match="nothing was dispatched"):
        run_initial_group_stage(two_topics, targets, allocation.groups, cfg, segmenter,
                                FakeBackend(responder), store,
                                scenarios=_drift(snapshot, sorted(snapshot)[0], how),
                                allow_live=True, stage_authorization=record)
    assert responder.calls == []
    assert not any(e["kind"] == "group" for e in store.log.entries())


def test_drafting_groups_outside_the_targets_dispatches_nothing(
        tmp_path, cfg, approvals_path, allocation, targets, store_with_scenarios, snapshot,
        segmenter, two_topics):
    record = load_stage_authorization(
        _write_record(tmp_path, _baseline_record(cfg, approvals_path, allocation, targets)))
    store = store_with_scenarios
    store.allowed_kinds = frozenset({"group"})
    responder = GroupResponder()
    with pytest.raises(PipelineAbort, match="do not match its targets exactly"):
        run_initial_group_stage(two_topics, targets[:-1], allocation.groups, cfg, segmenter,
                                FakeBackend(responder), store, scenarios=snapshot,
                                allow_live=True, stage_authorization=record)
    assert responder.calls == []


# --- group_stage_authorization_problems: malformed, stale, incomplete, over-broad -


def test_a_valid_authorized_record_has_no_problems(tmp_path, cfg, approvals_path, allocation,
                                                    targets):
    record = load_stage_authorization(
        _write_record(tmp_path, _baseline_record(cfg, approvals_path, allocation, targets)))
    assert _problems(record, cfg, targets, approvals_path, allocation) == []


@pytest.mark.parametrize("status", ["proposed", "paused", None])
def test_a_non_authorized_status_refuses(tmp_path, cfg, approvals_path, allocation, targets,
                                         status):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    r["status"] = status
    record = load_stage_authorization(_write_record(tmp_path, r))
    assert any("not 'authorized'" in p for p in _problems(record, cfg, targets, approvals_path,
                                                           allocation))


@pytest.mark.parametrize("bad", [None, "", 1, True])
def test_authorized_by_missing_or_invalid_refuses(tmp_path, cfg, approvals_path, allocation,
                                                   targets, bad):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    r["authorized_by"] = bad
    record = load_stage_authorization(_write_record(tmp_path, r))
    assert any("authorized_by must be a non-empty string" in p
              for p in _problems(record, cfg, targets, approvals_path, allocation))


@pytest.mark.parametrize("bad", [None, "", "not-a-date", 20260924])
def test_authorized_at_missing_or_invalid_refuses(tmp_path, cfg, approvals_path, allocation,
                                                   targets, bad):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    r["authorized_at"] = bad
    record = load_stage_authorization(_write_record(tmp_path, r))
    assert any("authorized_at must be a non-empty ISO-8601" in p
              for p in _problems(record, cfg, targets, approvals_path, allocation))


def test_wrong_record_version_refuses(tmp_path, cfg, approvals_path, allocation, targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    r["record_version"] = "group_stage_v2"
    record = load_stage_authorization(_write_record(tmp_path, r))
    assert any("record_version" in p for p in _problems(record, cfg, targets, approvals_path,
                                                         allocation))


def test_wrong_configuration_hash_refuses(tmp_path, cfg, approvals_path, allocation, targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    r["config"]["config_content_hash"] = "0" * 64
    record = load_stage_authorization(_write_record(tmp_path, r))
    assert any("different configuration content hash" in p
              for p in _problems(record, cfg, targets, approvals_path, allocation))


def test_wrong_kind_refuses(tmp_path, cfg, approvals_path, allocation, targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    r["kind"] = "scenario_redraft"
    record = load_stage_authorization(_write_record(tmp_path, r))
    assert any("covers kind" in p for p in _problems(record, cfg, targets, approvals_path,
                                                      allocation))


def test_stale_approvals_file_hash_refuses(tmp_path, cfg, approvals_path, allocation, targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    r["approvals_file_sha256"] = "0" * 64
    record = load_stage_authorization(_write_record(tmp_path, r))
    assert any("approvals-file SHA-256" in p for p in _problems(record, cfg, targets,
                                                                approvals_path, allocation))


def test_missing_approvals_file_refuses(tmp_path, cfg, approvals_path, allocation, targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    approvals_path.unlink()
    record = load_stage_authorization(_write_record(tmp_path, r))
    assert any("does not exist" in p for p in _problems(record, cfg, targets, approvals_path,
                                                         allocation))


def test_wrong_allocation_file_refuses(tmp_path, cfg, approvals_path, allocation, targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    r["allocation_file"] = "some_other_allocation.yaml"
    record = load_stage_authorization(_write_record(tmp_path, r))
    assert any("bound to allocation file" in p for p in _problems(record, cfg, targets,
                                                                  approvals_path, allocation))


def test_stale_allocation_content_hash_refuses(tmp_path, cfg, approvals_path, allocation,
                                               targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    r["allocation_content_hash"] = "0" * 64
    record = load_stage_authorization(_write_record(tmp_path, r))
    assert any("allocation content hash does not match" in p
              for p in _problems(record, cfg, targets, approvals_path, allocation))


@pytest.mark.parametrize("bad", [None, 100, True, "8", 0, -1])
def test_max_paid_calls_missing_or_invalid_refuses(tmp_path, cfg, approvals_path, allocation,
                                                    targets, bad):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    if bad is None:
        del r["max_paid_calls"]
    else:
        r["max_paid_calls"] = bad
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = _problems(record, cfg, targets, approvals_path, allocation)
    assert any("max_paid_calls" in p for p in problems)


@pytest.mark.parametrize("bad", [None, 2, 0, True, "1"])
def test_calls_per_target_missing_or_invalid_refuses(tmp_path, cfg, approvals_path, allocation,
                                                      targets, bad):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    if bad is None:
        del r["calls_per_target"]
    else:
        r["calls_per_target"] = bad
    record = load_stage_authorization(_write_record(tmp_path, r))
    assert any("calls_per_target" in p for p in _problems(record, cfg, targets, approvals_path,
                                                           allocation))


@pytest.mark.parametrize("bad", [
    "missing", None, "RS RP NS NP", {"RS": 1, "RP": 1, "NS": 1, "NP": 1},
    ["RP", "RS", "NP", "NS"], ["RS", "RP", "NS", "NP", "NP"], ["RS", "RP", "NS"], ["RS", "RP", "NS", "NP", "XX"],
    ["RS", "RP", "NS", "NX"], ["rs", "rp", "ns", "np"], [], ["RS", "RP", "NS", 4],
])
def test_cells_per_call_must_be_exactly_the_four_cells(tmp_path, cfg, approvals_path, allocation,
                                                       targets, bad):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    if bad == "missing":
        del r["cells_per_call"]
    else:
        r["cells_per_call"] = bad
    record = load_stage_authorization(_write_record(tmp_path, r))
    assert any("cells_per_call must be exactly ['RS', 'RP', 'NS', 'NP']" in p
               for p in _problems(record, cfg, targets, approvals_path, allocation))


@pytest.mark.parametrize("bad_targets", [None, [], "x", 1])
def test_malformed_targets_field_refuses_cleanly(tmp_path, cfg, approvals_path, allocation,
                                                  targets, bad_targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    if bad_targets is None:
        del r["targets"]
    else:
        r["targets"] = bad_targets
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = _problems(record, cfg, targets, approvals_path, allocation)
    assert any("targets field must be a mapping" in p for p in problems)


def test_a_malformed_per_scenario_entry_refuses_cleanly(tmp_path, cfg, approvals_path, allocation,
                                                        targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    some_scenario = targets[0].scenario_id
    r["targets"][some_scenario] = "not-a-mapping"
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = _problems(record, cfg, targets, approvals_path, allocation)
    assert any(some_scenario in p and "mapping of supported_option" in p for p in problems)


def test_a_malformed_target_entry_refuses_cleanly(tmp_path, cfg, approvals_path, allocation,
                                                   targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    t = targets[0]
    r["targets"][t.scenario_id][t.supported_option] = "not-a-mapping"
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = _problems(record, cfg, targets, approvals_path, allocation)
    assert any(f"{t.scenario_id}/{t.supported_option}" in p and "must be a mapping" in p
              for p in problems)


def test_a_missing_target_refuses(tmp_path, cfg, approvals_path, allocation, targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    t = targets[0]
    del r["targets"][t.scenario_id][t.supported_option]
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = _problems(record, cfg, targets, approvals_path, allocation)
    assert any(t.scenario_id in p and t.supported_option in p and "does not name" in p
              for p in problems)


def test_an_extra_target_refuses(tmp_path, cfg, approvals_path, allocation, targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    r["targets"]["climate_99_v1"] = {"opt_1": dict(next(iter(r["targets"].values()))["opt_1"])}
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = _problems(record, cfg, targets, approvals_path, allocation)
    assert any("climate_99_v1" in p and "not among the plan's new groups" in p
              for p in problems)


def test_a_stale_scenario_call_id_refuses(tmp_path, cfg, approvals_path, allocation, targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    t = targets[0]
    r["targets"][t.scenario_id][t.supported_option]["scenario_call_id"] = "0" * 64
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = _problems(record, cfg, targets, approvals_path, allocation)
    assert any(f"{t.scenario_id}/{t.supported_option}" in p and "stale binding" in p
              for p in problems)


def test_a_stale_scenario_text_hash_refuses(tmp_path, cfg, approvals_path, allocation, targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    t = targets[0]
    r["targets"][t.scenario_id][t.supported_option]["scenario_text_sha256"] = "0" * 64
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = _problems(record, cfg, targets, approvals_path, allocation)
    assert any(f"{t.scenario_id}/{t.supported_option}" in p and "stale binding" in p
              for p in problems)


@pytest.mark.parametrize("field", ["marker_family", "marker_string", "marker_realization_id"])
def test_a_wrong_marker_field_refuses(tmp_path, cfg, approvals_path, allocation, targets, field):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    t = targets[0]
    r["targets"][t.scenario_id][t.supported_option][field] = "not-the-real-value"
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = _problems(record, cfg, targets, approvals_path, allocation)
    assert any(f"{t.scenario_id}/{t.supported_option}" in p and field in p
              and "marker assignment is fixed by the allocation" in p for p in problems)


@pytest.mark.parametrize("over", [1, 100])
def test_max_paid_calls_over_broad_refuses(tmp_path, cfg, approvals_path, allocation, targets,
                                           over):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    r["max_paid_calls"] = len(targets) + over
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = _problems(record, cfg, targets, approvals_path, allocation)
    assert any("exactly equal" in p and "no unused authorisation headroom" in p
              for p in problems)


def test_several_problems_are_all_reported_together(tmp_path, cfg, approvals_path, allocation,
                                                     targets):
    r = _baseline_record(cfg, approvals_path, allocation, targets)
    r["status"] = "proposed"
    r["kind"] = "scenario_redraft"
    record = load_stage_authorization(_write_record(tmp_path, r))
    assert len(_problems(record, cfg, targets, approvals_path, allocation)) >= 2


# --- the happy path: exactly N traceable calls, four cells each -------------------


def test_authorized_record_permits_exactly_the_named_group_calls(
        tmp_path, cfg, approvals_path, allocation, targets, store_with_scenarios,
        approved_scenarios, segmenter, two_topics):
    record = load_stage_authorization(
        _write_record(tmp_path, _baseline_record(cfg, approvals_path, allocation, targets)))
    assert _problems(record, cfg, targets, approvals_path, allocation) == []

    store = store_with_scenarios
    store.allowed_kinds = frozenset({"group"})
    responder = GroupResponder()
    scenarios = recorded_scenarios(store)

    results = run_initial_group_stage(
        two_topics, targets, allocation.groups, cfg, segmenter, FakeBackend(responder), store,
        scenarios=scenarios, allow_live=True, stage_authorization=record)

    assert sorted(responder.calls) == sorted((t.scenario_id, t.supported_option) for t in targets)
    assert len(results) == 8
    assert all(r.outcome == ACCEPTED for r in results)
    entries = [e for e in store.log.entries() if e["kind"] == "group"]
    assert len(entries) == 8
    for entry in entries:
        assert entry["extra"]["stage_authorization_sha256"] == record["_file_sha256"]
        assert entry["extra"]["stage_authorization_path"] == str(record["_path"])
    # No repair was ever attempted: each group made exactly one call.
    assert all(e["attempt"] == 1 for e in entries)


def test_a_fully_completed_stage_never_calls_the_backend_again(
        tmp_path, cfg, approvals_path, allocation, targets, store_with_scenarios, segmenter,
        two_topics):
    record = load_stage_authorization(
        _write_record(tmp_path, _baseline_record(cfg, approvals_path, allocation, targets)))
    store = store_with_scenarios
    store.allowed_kinds = frozenset({"group"})
    scenarios = recorded_scenarios(store)

    run_initial_group_stage(two_topics, targets, allocation.groups, cfg, segmenter,
                            FakeBackend(GroupResponder()), store, scenarios=scenarios,
                            allow_live=True, stage_authorization=record)
    entries_after_first = len([e for e in store.log.entries() if e["kind"] == "group"])

    responder_again = GroupResponder()
    results = run_initial_group_stage(two_topics, targets, allocation.groups, cfg, segmenter,
                                      FakeBackend(responder_again), store, scenarios=scenarios,
                                      allow_live=True, stage_authorization=record)
    assert responder_again.calls == []
    assert len(results) == 8
    entries_after_second = len([e for e in store.log.entries() if e["kind"] == "group"])
    assert entries_after_second == entries_after_first == 8


# --- transport-failure blocking ----------------------------------------------------


class InterruptedGroupResponder:
    """Fails one specific (scenario, option) once, then succeeds."""

    def __init__(self, fail_once: tuple[str, str]):
        self._fail_once = fail_once
        self.calls: list[tuple[str, str]] = []

    def __call__(self, request):
        scenario_id = f"{request.decision_id}_v{request.variant_id}"
        key = (scenario_id, request.supported_option)
        self.calls.append(key)
        if key == self._fail_once:
            self._fail_once = None
            return ConnectionError("simulated transport failure")
        tag = f"{scenario_id} {request.supported_option}".replace("_", " ")
        return bodies_for(request.context["marker_string"], tag)


@pytest.fixture
def interrupted_run(tmp_path, cfg, approvals_path, allocation, targets, store_with_scenarios,
                    segmenter, two_topics):
    record = load_stage_authorization(
        _write_record(tmp_path, _baseline_record(cfg, approvals_path, allocation, targets)))
    store = store_with_scenarios
    store.allowed_kinds = frozenset({"group"})
    scenarios = recorded_scenarios(store)
    # The LAST target in dispatch order: `run_initial_group_stage` stops at
    # the first failure, so failing the first target would leave the other
    # seven untried, not "already recorded" -- failing the last one instead
    # lets the preceding seven succeed and be recorded before the abort.
    fail_target = sorted(targets, key=lambda t: (t.scenario_id, t.supported_option))[-1]
    fail_key = (fail_target.scenario_id, fail_target.supported_option)

    responder = InterruptedGroupResponder(fail_once=fail_key)
    with pytest.raises(PipelineAbort):
        run_initial_group_stage(two_topics, targets, allocation.groups, cfg, segmenter,
                                FakeBackend(responder), store, scenarios=scenarios,
                                allow_live=True, stage_authorization=record)
    return store, record, targets, scenarios, fail_key, two_topics


def test_transport_failure_carries_the_authorization_hash(interrupted_run):
    store, record, *_ = interrupted_run
    entries = [e for e in store.log.entries() if e["kind"] == "group"]
    failed = [e for e in entries if e["status"] == "error"]
    assert len(failed) == 1
    assert failed[0]["extra"]["stage_authorization_sha256"] == record["_file_sha256"]


def test_transport_blocked_group_targets_finds_the_failed_target(interrupted_run):
    store, record, targets, *_rest, fail_key, _topics = interrupted_run
    blocked = transport_blocked_group_targets(store, targets, record["_file_sha256"])
    assert [(t.scenario_id, t.supported_option) for t in blocked] == [fail_key]


def test_rerunning_under_the_same_authorization_after_a_transport_failure_is_blocked(
        interrupted_run, cfg, segmenter, allocation):
    store, record, targets, scenarios, fail_key, two_topics = interrupted_run
    entries_before = len([e for e in store.log.entries() if e["kind"] == "group"])

    responder2 = GroupResponder()
    with pytest.raises(PipelineAbort, match=f"{fail_key[0]}/{fail_key[1]}"):
        run_initial_group_stage(two_topics, targets, allocation.groups, cfg, segmenter,
                                FakeBackend(responder2), store, scenarios=scenarios,
                                allow_live=True, stage_authorization=record)
    assert responder2.calls == []
    entries_after = len([e for e in store.log.entries() if e["kind"] == "group"])
    assert entries_after == entries_before


def test_a_new_retry_authorization_permits_the_remaining_target(
        interrupted_run, tmp_path, cfg, approvals_path, allocation, segmenter):
    store, old_record, targets, scenarios, fail_key, two_topics = interrupted_run
    retry_content = _baseline_record(cfg, approvals_path, allocation, targets)
    retry_content["authorized_at"] = "2026-09-25"          # genuinely different bytes
    retry_record = load_stage_authorization(
        _write_record(tmp_path, retry_content, name="retry.yaml"))
    assert retry_record["_file_sha256"] != old_record["_file_sha256"]
    assert transport_blocked_group_targets(store, targets, retry_record["_file_sha256"]) == []

    responder2 = GroupResponder()
    results = run_initial_group_stage(two_topics, targets, allocation.groups, cfg, segmenter,
                                      FakeBackend(responder2), store, scenarios=scenarios,
                                      allow_live=True, stage_authorization=retry_record)
    assert responder2.calls == [fail_key]
    assert len(results) == 8
    entries = [e for e in store.log.entries() if e["kind"] == "group"]
    assert len(entries) == 9          # 7 first-run successes + 1 failure + 1 retry success
    accepted = [e for e in entries if e["status"] == "ok"]
    assert len(accepted) == 8
    # The real-evidence audit accepts exactly this pipeline-written history.
    assert _group_evidence_violations(
        store.log.path, tmp_path, {(t.scenario_id, t.supported_option) for t in targets}) == []


# --- the shipped record: real, committed, read-only --------------------------------


REDRAFTED = {"technology_08_v1", "technology_13_v1"}


def _real_snapshot(cfg, bank, alloc):
    """The real current-scenario snapshot, built read-only by the CLI's own
    ``_current_scenarios`` -- originals, the two accepted redrafts, and any
    scenario corrections."""
    from types import SimpleNamespace
    pilot = _load_pilot_script()
    store = CallStore(REAL_RUN_V2, cfg,
                      topic_bank_content_hash=content_hash(bank.model_dump(mode="json")),
                      allocation_content_hash=alloc.content_hash)
    paths = pilot.profile_paths(cfg)
    args = SimpleNamespace(approvals_file=str(ROOT / paths["approvals_file"]),
                           scenario_corrections_file=str(ROOT / paths["scenario_corrections_file"]))
    return pilot._current_scenarios(args, cfg, store)


def _real_state():
    cfg = load_config(FULL)
    from reasonstyle.corpus.topics import load_topic_bank
    bank = load_topic_bank(ROOT / "data/topics/full_topics_v2.yaml")
    alloc = load_allocation(REAL_ALLOCATION)
    plan = plan_full_corpus(cfg, bank, load_seed_corpus(cfg, bank=bank, root=ROOT))
    return cfg, bank, alloc, plan


def test_the_real_group_stage_record_is_historical_and_stale_only_where_expected():
    """Read-only against the real committed files. The record is spent
    history: its 192 bindings are exactly what was authorised and sent. The
    audited climate_10_v1 scenario correction and re-approval (2026-09-25,
    path A1) make it stale against the current state in exactly three ways --
    the approvals-file hash and the two climate_10_v1 scenario-text bindings --
    so it cannot authorise another call. Lifecycle fields are overridden in
    memory only, so this does not depend on them."""
    cfg, bank, alloc, plan = _real_state()
    assert cfg.content_hash == FROZEN_CONTENT_HASH
    assert alloc.content_hash == REAL_ALLOCATION_CONTENT_HASH
    approvals = load_approvals(REAL_APPROVALS)
    snapshot = _real_snapshot(cfg, bank, alloc)

    live_targets = group_stage_targets(
        plan.new_groups, approvals, alloc, config_content_hash=cfg.content_hash,
        topic_bank_content_hash=content_hash(bank.model_dump(mode="json")), scenarios=snapshot)
    assert len(live_targets) == 192
    assert len({t.scenario_id for t in live_targets}) == 96

    # The two approved redrafts stand in the snapshot, and their targets bind
    # the redraft calls -- not the rejected originals they superseded.
    import json
    log = [json.loads(line) for line in open(REAL_RUN_V2 / "generation_log.jsonl")]
    redrafts = {f"{e['decision_id']}_v{e['variant_id']}": e["call_id"]
                for e in log if e["kind"] == "scenario_redraft"}
    assert set(redrafts) == REDRAFTED
    for t in live_targets:
        if t.scenario_id in REDRAFTED:
            assert t.scenario_call_id == redrafts[t.scenario_id] == approvals[t.scenario_id].call_id
            assert (t.scenario_text_sha256
                    == sha256_of(snapshot[t.scenario_id]["scenario_text"])
                    == approvals[t.scenario_id].scenario_text_sha256)

    record = dict(load_stage_authorization(REAL_RECORD))
    record.update(status="authorized", authorized_by="Binding Check",
                  authorized_at="2026-09-24")          # in memory only
    problems = group_stage_authorization_problems(
        record, cfg=cfg, kind=GROUP_STAGE_KIND,
        record_version=GROUP_STAGE_AUTHORIZATION_RECORD_VERSION, targets=live_targets,
        approvals_path=REAL_APPROVALS, allocation_path=REAL_ALLOCATION,
        allocation_content_hash=alloc.content_hash)
    assert len(problems) == 3, problems
    assert sum("approvals-file SHA-256" in p for p in problems) == 1
    assert sorted(p.split(":")[0] for p in problems if "stale binding" in p) == [
        f"{CORRECTED_SCENARIO}/opt_1", f"{CORRECTED_SCENARIO}/opt_2"]
    assert {(sid, opt) for sid, per in record["targets"].items() for opt in per} \
        == set(plan.new_groups)

    # History: every binding is the model scenario text that group's recorded
    # prompt actually contained, from the bound scenario call.
    results = REAL_RUN_V2 / "results"
    sent = {(f"{e['decision_id']}_v{e['variant_id']}", e["supported_option"]): e["call_id"]
            for e in log if e["kind"] == "group"}
    for sid, per in record["targets"].items():
        for opt, binding in per.items():
            model_text = json.loads((results / f"{binding['scenario_call_id']}.json")
                                    .read_text(encoding="utf-8"))["fields"]["scenario_text"]
            assert sha256_of(model_text) == binding["scenario_text_sha256"], (sid, opt)
            prompt = json.loads((REAL_RUN_V2 / "raw" / f"{sent[(sid, opt)]}.json")
                                .read_text(encoding="utf-8"))["prompt"]
            assert json.dumps(model_text)[1:-1] in json.dumps(prompt), (sid, opt)


CORRECTED_SCENARIO = "climate_10_v1"
REAL_SCENARIO_CORRECTIONS = ROOT / "data/full/scenario_corrections_full_v2.yaml"


def test_the_climate_10_v1_correction_binds_what_its_groups_were_drafted_from():
    """The A1 exception, checked: the scenario correction's original text is
    exactly what the consumed record bound and what both kept groups were
    drafted from; the current approval binds the corrected text instead."""
    from reasonstyle.generation.scenario_corrections import load_scenario_corrections
    [correction] = load_scenario_corrections(REAL_SCENARIO_CORRECTIONS)
    assert correction.scenario_id == CORRECTED_SCENARIO
    assert correction.approval_state == "approved"
    record = load_stage_authorization(REAL_RECORD)
    for opt in ("opt_1", "opt_2"):
        binding = record["targets"][CORRECTED_SCENARIO][opt]
        assert binding["scenario_call_id"] == correction.original_call_id
        assert binding["scenario_text_sha256"] == correction.original_text_sha256
    approval = load_approvals(REAL_APPROVALS)[CORRECTED_SCENARIO]
    assert approval.call_id == correction.original_call_id
    assert approval.scenario_text_sha256 == correction.corrected_text_sha256


def test_the_real_frozen_config_approvals_allocation_and_record_are_unchanged():
    assert file_sha256(FULL) == "5d34bad7fc3b81d122fc0feb62fb502097f2b220200544027e42a70c94abe819"
    assert load_config(FULL).content_hash == FROZEN_CONTENT_HASH
    assert (file_sha256(REAL_APPROVALS)
            == "46b451f5af869d6e16de260bfacf42b4d2438c9d2a71375f9b0050765d856382")
    assert load_allocation(REAL_ALLOCATION).content_hash == REAL_ALLOCATION_CONTENT_HASH
    # The consumed record is historical evidence, kept byte for byte.
    assert (file_sha256(REAL_RECORD)
            == "8a7acf3e407fa031a4d928773c790a64f84df355cb63f6e0df708462dab53133")


def test_using_the_mechanism_reads_but_never_writes_the_real_files():
    before = [file_sha256(p) for p in (FULL, REAL_APPROVALS, REAL_ALLOCATION, REAL_RECORD)]
    cfg, bank, alloc, plan = _real_state()
    live_targets = group_stage_targets(
        plan.new_groups, load_approvals(REAL_APPROVALS), alloc,
        config_content_hash=cfg.content_hash,
        topic_bank_content_hash=content_hash(bank.model_dump(mode="json")),
        scenarios=_real_snapshot(cfg, bank, alloc))
    group_stage_authorization_problems(
        load_stage_authorization(REAL_RECORD), cfg=cfg, kind=GROUP_STAGE_KIND,
        record_version=GROUP_STAGE_AUTHORIZATION_RECORD_VERSION, targets=live_targets,
        approvals_path=REAL_APPROVALS, allocation_path=REAL_ALLOCATION,
        allocation_content_hash=alloc.content_hash)
    assert [file_sha256(p) for p in (FULL, REAL_APPROVALS, REAL_ALLOCATION, REAL_RECORD)] == before


def _group_evidence_violations(log_path: Path, authorizations_dir: Path,
                               new_groups: set[tuple[str, str]]) -> list[str]:
    """Every way the group entries in ``log_path`` fall outside an authorised
    initial group stage. Empty for a log with no group entries at all."""
    import json
    from collections import Counter
    group_stage_shas, authorised_shas = set(), set()
    for path in sorted(authorizations_dir.glob("*.yaml")):
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if content.get("kind") == GROUP_STAGE_KIND:
            group_stage_shas.add(file_sha256(path))
            if content.get("status") == "authorized":
                authorised_shas.add(file_sha256(path))

    violations: list[str] = []
    dispatched: Counter = Counter()
    completed: Counter = Counter()
    for e in map(json.loads, log_path.read_text(encoding="utf-8").splitlines()):
        sha = (e.get("extra") or {}).get("stage_authorization_sha256")
        if e["kind"] == "repair":
            if sha in group_stage_shas:
                violations.append(f"{e['call_id'][:12]}: a repair under an initial-group record")
            continue
        if e["kind"] != "group":
            continue
        target = (f"{e['decision_id']}_v{e['variant_id']}", e["supported_option"])
        if target not in new_groups:
            violations.append(f"{target}: not one of the authorised new groups")
        if e.get("attempt") != 1:
            violations.append(f"{target}: attempt {e.get('attempt')!r}, not 1")
        if sha not in authorised_shas:
            violations.append(f"{target}: no authorised initial-group record hash ({sha!r})")
        dispatched[(target, sha)] += 1
        if e["status"] in ("ok", "rejected"):
            completed[target] += 1
    violations += [f"{t}: dispatched {n} times under one record"
                   for (t, _), n in dispatched.items() if n > 1]
    violations += [f"{t}: completed {n} times" for t, n in completed.items() if n > 1]
    return violations


def test_real_group_evidence_is_within_the_authorised_initial_stage():
    """Read-only audit of whatever group evidence the real run holds. Passes
    with none (before generation) and holds during and after the authorised
    stage: every ``group`` entry is an initial, attempt-1 call for one of the
    192 new targets, carries the SHA-256 of an authorised initial-group-stage
    record, and no target was dispatched twice under one record or completed
    twice at all. No repair ever carries such a record's hash."""
    cfg = load_config(FULL)
    from reasonstyle.corpus.topics import load_topic_bank
    bank = load_topic_bank(ROOT / "data/topics/full_topics_v2.yaml")
    plan = plan_full_corpus(cfg, bank, load_seed_corpus(cfg, bank=bank, root=ROOT))
    assert len(set(plan.new_groups)) == 192
    assert _group_evidence_violations(REAL_RUN_V2 / "generation_log.jsonl",
                                      ROOT / "data/full/authorizations",
                                      set(plan.new_groups)) == []


@pytest.fixture
def audit_setup(tmp_path):
    """A synthetic authorisations directory with one authorised and one
    proposed initial-group record, and one known target."""
    auth_dir = tmp_path / "authorizations"
    auth_dir.mkdir()
    authorised = _write_record(auth_dir, {"kind": GROUP_STAGE_KIND, "status": "authorized"},
                               name="authorised.yaml")
    proposed = _write_record(auth_dir, {"kind": GROUP_STAGE_KIND, "status": "proposed"},
                             name="proposed.yaml")
    return auth_dir, file_sha256(authorised), file_sha256(proposed), {("energy_09_v1", "opt_1")}


def _entry(sha, *, kind="group", status="ok", attempt=1, scenario="energy_09_v1",
           option="opt_1") -> dict:
    decision, _, variant = scenario.rpartition("_v")
    return {"call_id": "c" * 64, "kind": kind, "status": status, "attempt": attempt,
            "decision_id": decision, "variant_id": int(variant), "supported_option": option,
            "extra": {"stage_authorization_sha256": sha} if sha else None}


def _audit(tmp_path, audit_setup, entries):
    import json
    auth_dir, _, _, new_groups = audit_setup
    log = tmp_path / "generation_log.jsonl"
    log.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    return _group_evidence_violations(log, auth_dir, new_groups)


def test_group_evidence_audit_passes_empty_and_valid_logs(tmp_path, audit_setup):
    _, ok_sha, _, _ = audit_setup
    assert _audit(tmp_path, audit_setup, []) == []
    assert _audit(tmp_path, audit_setup, [_entry(None, kind="scenario")]) == []
    assert _audit(tmp_path, audit_setup, [_entry(ok_sha)]) == []


@pytest.mark.parametrize("case", ["outside_target", "repair_under_record", "attempt_2",
                                  "no_sha", "proposed_sha", "twice_under_one_record",
                                  "completed_twice"])
def test_group_evidence_audit_detects_each_violation(tmp_path, audit_setup, case):
    import hashlib
    _, ok_sha, proposed_sha, _ = audit_setup
    other_sha = hashlib.sha256(b"a later retry record").hexdigest()
    entries = {
        "outside_target": [_entry(ok_sha, scenario="climate_01_v1")],
        "repair_under_record": [_entry(ok_sha), _entry(ok_sha, kind="repair", attempt=2)],
        "attempt_2": [_entry(ok_sha, attempt=2)],
        "no_sha": [_entry(None)],
        "proposed_sha": [_entry(proposed_sha)],
        "twice_under_one_record": [_entry(ok_sha, status="error"), _entry(ok_sha)],
        "completed_twice": [_entry(ok_sha), _entry(other_sha)],
    }[case]
    assert _audit(tmp_path, audit_setup, entries) != []


# --- CLI-level: dry run refuses before a credential or backend --------------------


def _proposed_copy(tmp_path: Path) -> Path:
    """An isolated copy of the real record, explicitly proposed -- whatever
    the tracked record's own lifecycle state is by now."""
    content = yaml.safe_load(REAL_RECORD.read_text(encoding="utf-8"))
    content.update(status="proposed", authorized_by=None, authorized_at=None)
    return _write_record(tmp_path, content, name="group_stage_proposed.yaml")


def test_cli_proposed_record_cannot_send(tmp_path, capsys, monkeypatch):
    pilot = _load_pilot_script()
    monkeypatch.setattr(pilot, "_live_backend",
                        lambda cfg: pytest.fail("a backend was built"))
    rc = pilot.main(["groups", "--config", str(FULL), "--out", str(_copy_run(tmp_path)),
                     "--stage-authorization", str(_proposed_copy(tmp_path))])
    captured = capsys.readouterr()
    assert rc == 1
    assert "groups (initial only, no repair): 192 call(s)" in captured.out
    assert "not 'authorized'" in captured.err


def _copy_run(tmp_path: Path) -> Path:
    import shutil
    out = tmp_path / "run_v2_copy"
    shutil.copytree(REAL_RUN_V2, out)
    return out


def _scenario_result_path(run: Path, scenario_id: str) -> Path:
    import json
    kinds = ("scenario_redraft",) if scenario_id in REDRAFTED else ("scenario",)
    for e in map(json.loads, open(run / "generation_log.jsonl")):
        if e["kind"] in kinds and f"{e['decision_id']}_v{e['variant_id']}" == scenario_id:
            return run / "results" / f"{e['call_id']}.json"
    raise AssertionError(scenario_id)                      # pragma: no cover


@pytest.mark.parametrize("scenario_id", ["climate_06_v1", "technology_08_v1"])
@pytest.mark.parametrize("how", ["altered_text", "missing_text"])
def test_cli_refuses_drifted_run_evidence_before_any_backend(tmp_path, capsys, monkeypatch,
                                                             scenario_id, how):
    """On a byte-copy of the real run directory (the real one is only read),
    drifting one scenario's stored evidence -- an original or an approved
    redraft -- refuses at target verification, before the authorisation
    record is even checked and before any backend could be built."""
    import json
    run = _copy_run(tmp_path)
    path = _scenario_result_path(run, scenario_id)
    result = json.loads(path.read_text(encoding="utf-8"))
    result["fields"]["scenario_text"] = ("" if how == "missing_text"
                                         else result["fields"]["scenario_text"] + " Extra.")
    path.write_text(json.dumps(result), encoding="utf-8")

    pilot = _load_pilot_script()
    monkeypatch.setattr(pilot, "_live_backend",
                        lambda cfg: pytest.fail("a backend was built"))
    rc = pilot.main(["groups", "--config", str(FULL), "--out", str(run),
                     "--stage-authorization", str(_proposed_copy(tmp_path))])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no group-stage target can be built" in err and scenario_id in err
    assert "not 'authorized'" not in err


def test_cli_on_an_undrifted_copy_passes_verification_and_stops_at_the_proposal(
        tmp_path, capsys, monkeypatch):
    run = _copy_run(tmp_path)
    pilot = _load_pilot_script()
    monkeypatch.setattr(pilot, "_live_backend",
                        lambda cfg: pytest.fail("a backend was built"))
    rc = pilot.main(["groups", "--config", str(FULL), "--out", str(run),
                     "--stage-authorization", str(_proposed_copy(tmp_path))])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no group-stage target can be built" not in err
    assert "not 'authorized'" in err


def test_cli_the_consumed_real_record_cannot_authorise_another_call(tmp_path, capsys,
                                                                    monkeypatch):
    """The real, authorised, spent record is stale against the corrected state
    and refuses before any backend could be built."""
    pilot = _load_pilot_script()
    monkeypatch.setattr(pilot, "_live_backend",
                        lambda cfg: pytest.fail("a backend was built"))
    rc = pilot.main(["groups", "--config", str(FULL), "--out", str(_copy_run(tmp_path)),
                     "--stage-authorization", str(REAL_RECORD)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "refusing; nothing was sent" in err
    assert f"{CORRECTED_SCENARIO}/opt_1" in err and "stale binding" in err


def test_cli_without_stage_authorization_is_unchanged(capsys):
    pilot = _load_pilot_script()
    rc = pilot.main(["groups", "--config", str(FULL)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "outside the recorded authorisation" in captured.err


def test_cli_stage_authorization_refused_on_scenario_stage(capsys):
    pilot = _load_pilot_script()
    rc = pilot.main(["scenarios", "--config", str(FULL),
                     "--stage-authorization", str(REAL_RECORD)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "applies to the groups stage only" in err
