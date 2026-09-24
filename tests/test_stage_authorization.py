"""A later paid stage on the full v2 configuration — starting with
``redraft-scenarios`` — needs its own authorisation, separate from the frozen
configuration's own ``generation_authorization``: recording one inside
``configs/frozen/v2_full.yaml`` would edit a frozen file, change its content
hash, and stale every one of the 94 scenario approvals bound to that exact
hash. ``reasonstyle.generation.stage_authorization`` is that separate
mechanism; this file tests it.

Everything here runs against an isolated copy of the frozen configuration,
with its run directory and approvals file redirected under ``tmp_path``: the
real ``data/full/run_v2`` and ``data/full/scenario_approvals_full_v2.yaml``
are never opened for writing. No request is made and no credential is read.
The one exception is the last section, which reads (never writes) the real,
committed files to confirm the shipped authorisation record actually matches
them and that using the mechanism never touches either hash.
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pytest
import yaml

from reasonstyle.config import load_config
from reasonstyle.corpus import segmenter_from_config
from reasonstyle.corpus.topics import load_topic_bank
from reasonstyle.generation import FakeBackend
from reasonstyle.generation.allocation import load_allocation
from reasonstyle.generation.approvals import REQUIRED_JUDGEMENTS, ScenarioApproval, save_approvals
from reasonstyle.generation.corpus_source import load_seed_corpus, plan_full_corpus
from reasonstyle.generation.pipeline import CallStore, PipelineAbort, recorded_scenarios, run_scenario_stage
from reasonstyle.generation.redraft import (
    STAGE_AUTHORIZATION_RECORD_VERSION,
    redraft_targets,
    run_redraft_stage,
    transport_blocked_targets,
)
from reasonstyle.generation.stage_authorization import (
    StageAuthorizationError,
    load_stage_authorization,
    stage_authorization_problems,
)
from reasonstyle.hashing import file_sha256, sha256_of

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "configs/frozen/v2_full.yaml"
FROZEN_CONTENT_HASH = "7548650b42e7cb7407f521d10d9e8c2d4f9f25b5074d98cc523b4a5e35a99c94"
REAL_APPROVALS = ROOT / "data/full/scenario_approvals_full_v2.yaml"
REAL_RECORD = ROOT / "data/full/authorizations/scenario_redraft_v1.yaml"
BANK = load_topic_bank(ROOT / "data/topics/full_topics_v2.yaml")
REDRAFT_IDS = ("technology_08_v1", "technology_13_v1")
RECORD_VERSION = STAGE_AUTHORIZATION_RECORD_VERSION


def _load_pilot_script():
    spec = importlib.util.spec_from_file_location("pilot_script_stage_auth", ROOT / "scripts/pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _isolated_config(tmp_path: Path) -> Path:
    """A copy of the frozen full configuration with its run directory and
    approvals file redirected into ``tmp_path``. Identical in spirit to the
    fixture in ``test_full_status_gate.py``, kept local so this file stands
    alone."""
    text = FULL.read_text(encoding="utf-8")
    text = text.replace("  run: data/full/run_v2\n", f"  run: {tmp_path / 'run_v2'}\n", 1)
    text = text.replace(
        "  approvals: data/full/scenario_approvals_full_v2.yaml\n",
        f"  approvals: {tmp_path / 'scenario_approvals_full_v2.yaml'}\n", 1)
    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir(exist_ok=True)
    path = frozen_dir / "v2_full.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def scenario_text(scenario_id: str) -> str:
    return (f"Record {scenario_id} sets out a regional capacity decision for the coming "
            f"winters. The operator can extend the existing baseload plant or accelerate "
            f"the storage build already under tender. Both routes are funded and feasible, "
            f"and nothing states how reliability should be weighed against emissions. The "
            f"extended plant can deliver full output through any cold spell. Retiring it on "
            f"schedule would cut power-sector emissions. One route must be chosen before "
            f"the tender closes.")


class ScenarioOnlyResponder:
    def __call__(self, request):
        assert request.kind == "scenario"
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
    from reasonstyle.hashing import content_hash
    return content_hash(BANK.model_dump(mode="json"))


@pytest.fixture
def allocation():
    return load_allocation(ROOT / "data/full/marker_allocation_full_v2.yaml")


def _new_topics(plan):
    return [t for t in BANK.topics if t.status == "curated" and t.decision_id in set(plan.new_decisions)]


@pytest.fixture
def approvals_path(tmp_path):
    return tmp_path / "scenario_approvals_full_v2.yaml"


@pytest.fixture
def store_with_96_scenarios(tmp_path, cfg, plan, bank_hash, allocation):
    """Every one of the 96 new scenarios drafted offline, exactly like the
    ``completed_run`` fixture in ``test_full_status_gate.py``. Returns the
    store; the approvals file is built separately by ``redraft_pair``."""
    store = CallStore(tmp_path / "run_v2", cfg, topic_bank_content_hash=bank_hash,
                      allocation_content_hash=allocation.content_hash)
    segmenter = segmenter_from_config(cfg)
    run_scenario_stage(_new_topics(plan), cfg, segmenter, FakeBackend(ScenarioOnlyResponder()),
                       store)
    return store


@pytest.fixture
def redraft_pair(store_with_96_scenarios, cfg, bank_hash, approvals_path):
    """94 approved, 2 (``REDRAFT_IDS``) sent to redraft — written to
    ``approvals_path`` — and the live ``RedraftTarget`` list that follows
    from it. This is the fixture every authorisation test builds a record
    against."""
    scenarios = recorded_scenarios(store_with_96_scenarios)
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
    save_approvals(approvals, approvals_path)
    targets = redraft_targets(approvals, scenarios)
    assert sorted(t.scenario_id for t in targets) == sorted(REDRAFT_IDS)
    return targets


def _baseline_record(cfg, approvals_path: Path, targets) -> dict:
    return {
        "record_version": "scenario_redraft_v1",
        "status": "authorized",
        "authorized_by": "Test Reviewer",
        "authorized_at": "2026-09-24",
        "config": {"config_version": cfg.config_version, "config_content_hash": cfg.content_hash},
        "kind": "scenario_redraft",
        "max_paid_calls": 2,
        "calls_per_target": 1,
        "approvals_file": str(approvals_path),
        "approvals_file_sha256": file_sha256(approvals_path),
        "targets": {t.scenario_id: {"call_id": t.original_call_id,
                                    "scenario_text_sha256": t.original_text_sha256}
                   for t in targets},
        "excludes": ["group drafting", "repairs"],
    }


def _write_record(tmp_path: Path, record: dict, name: str = "auth.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(record, sort_keys=True), encoding="utf-8")
    return path


# --- load_stage_authorization: missing or malformed -------------------------------


def test_a_missing_file_refuses(tmp_path):
    with pytest.raises(StageAuthorizationError, match="does not exist"):
        load_stage_authorization(tmp_path / "nowhere.yaml")


def test_a_non_mapping_file_refuses(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(StageAuthorizationError, match="must be a mapping"):
        load_stage_authorization(path)


# --- stage_authorization_problems: every refusal category -------------------------


def test_a_valid_authorized_record_has_no_problems(tmp_path, cfg, approvals_path, redraft_pair):
    path = _write_record(tmp_path, _baseline_record(cfg, approvals_path, redraft_pair))
    record = load_stage_authorization(path)
    assert stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                        record_version=RECORD_VERSION, targets=redraft_pair,
                                        approvals_path=approvals_path) == []


@pytest.mark.parametrize("status", ["proposed", "paused", "revoked", None])
def test_a_non_authorized_status_refuses(tmp_path, cfg, approvals_path, redraft_pair, status):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    r["status"] = status
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("not 'authorized'" in p for p in problems)


@pytest.mark.parametrize("bad_value", [None, "", "   ", 42, True, ["Test Reviewer"]])
def test_authorized_by_missing_null_or_invalid_refuses(tmp_path, cfg, approvals_path, redraft_pair,
                                                        bad_value):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    if bad_value is None:
        del r["authorized_by"]
    else:
        r["authorized_by"] = bad_value
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("authorized_by must be a non-empty string" in p for p in problems)


@pytest.mark.parametrize("bad_value", [None, "", "not-a-date", "2026-13-40", 20260924, True])
def test_authorized_at_missing_null_or_invalid_refuses(tmp_path, cfg, approvals_path, redraft_pair,
                                                        bad_value):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    if bad_value is None:
        del r["authorized_at"]
    else:
        r["authorized_at"] = bad_value
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("authorized_at must be a non-empty ISO-8601" in p for p in problems)


def test_authorized_at_accepts_a_full_timestamp_not_only_a_bare_date(tmp_path, cfg, approvals_path,
                                                                     redraft_pair):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    r["authorized_at"] = "2026-09-24T10:15:00+09:00"
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert not any("authorized_at" in p for p in problems)


@pytest.mark.parametrize("bad_value", [None, "scenario_redraft_v2", "", 1, True])
def test_wrong_or_missing_record_version_refuses(tmp_path, cfg, approvals_path, redraft_pair,
                                                  bad_value):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    if bad_value is None:
        del r["record_version"]
    else:
        r["record_version"] = bad_value
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("record_version" in p and RECORD_VERSION in p for p in problems)


def test_a_different_configuration_version_refuses(tmp_path, cfg, approvals_path, redraft_pair):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    r["config"]["config_version"] = "v3_full"
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("v3_full" in p for p in problems)


def test_a_different_configuration_hash_refuses(tmp_path, cfg, approvals_path, redraft_pair):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    r["config"]["config_content_hash"] = "0" * 64
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("different configuration content hash" in p for p in problems)


def test_a_different_approvals_file_path_refuses(tmp_path, cfg, approvals_path, redraft_pair):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    r["approvals_file"] = "data/full/some_other_approvals.yaml"
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("approvals file" in p for p in problems)


def test_a_stale_approvals_file_hash_refuses(tmp_path, cfg, approvals_path, redraft_pair):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    r["approvals_file_sha256"] = "0" * 64
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("has changed since the authorisation was granted" in p for p in problems)


def test_a_missing_approvals_file_refuses(tmp_path, cfg, approvals_path, redraft_pair):
    """The record's own binding matches, but the file it names has been
    deleted since — a distinct failure mode from a path mismatch or a
    changed hash, and it must not raise trying to hash a file that is gone."""
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    approvals_path.unlink()
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("does not exist" in p for p in problems)


@pytest.mark.parametrize("bad_config", [None, "v2_full", ["v2_full"], 1, True])
def test_a_malformed_config_block_refuses_cleanly(tmp_path, cfg, approvals_path, redraft_pair,
                                                   bad_config):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    if bad_config is None:
        del r["config"]
    else:
        r["config"] = bad_config
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("config field must be a mapping" in p for p in problems)


def test_a_different_call_kind_refuses(tmp_path, cfg, approvals_path, redraft_pair):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    r["kind"] = "group"
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("covers kind 'group'" in p for p in problems)


def test_max_paid_calls_too_small_refuses_with_no_headroom_message(tmp_path, cfg, approvals_path,
                                                                   redraft_pair):
    """Not just "too low to cover the stage" — the record must equal the
    target count exactly, so a too-small ceiling is refused the same way a
    too-large one is: no unused headroom, in either direction."""
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    r["max_paid_calls"] = 1
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("exactly equal the 2 authorised target(s)" in p and "1" in p for p in problems)


def test_max_paid_calls_too_large_refuses_no_unused_headroom(tmp_path, cfg, approvals_path,
                                                              redraft_pair):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    r["max_paid_calls"] = 5
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("exactly equal the 2 authorised target(s)" in p and "no unused authorisation "
              "headroom" in p for p in problems)


@pytest.mark.parametrize("bad_value", [None, True, False, "2", 2.0, [2]])
def test_max_paid_calls_missing_null_or_wrongly_typed_refuses(tmp_path, cfg, approvals_path,
                                                               redraft_pair, bad_value):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    if bad_value is None:
        del r["max_paid_calls"]
    else:
        r["max_paid_calls"] = bad_value
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("max_paid_calls must be a required positive integer" in p for p in problems)


@pytest.mark.parametrize("bad_value", [None, 2, 0, -1, True, "1", 1.0])
def test_calls_per_target_missing_or_invalid_refuses(tmp_path, cfg, approvals_path, redraft_pair,
                                                      bad_value):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    if bad_value is None:
        del r["calls_per_target"]
    else:
        r["calls_per_target"] = bad_value
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("calls_per_target must be a required integer equal to exactly 1" in p
              for p in problems)


def test_a_missing_reviewed_target_refuses(tmp_path, cfg, approvals_path, redraft_pair):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    del r["targets"][REDRAFT_IDS[0]]
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any(REDRAFT_IDS[0] in p and "does not name" in p for p in problems)


def test_an_extra_target_refuses(tmp_path, cfg, approvals_path, redraft_pair):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    r["targets"]["climate_06_v1"] = {"call_id": "x" * 64, "scenario_text_sha256": "y" * 64}
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("climate_06_v1" in p and "does not currently mark redraft" in p for p in problems)


def test_a_stale_target_call_id_refuses(tmp_path, cfg, approvals_path, redraft_pair):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    r["targets"][REDRAFT_IDS[0]]["call_id"] = "0" * 64
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any(REDRAFT_IDS[0] in p and "stale binding" in p for p in problems)


def test_a_stale_target_text_hash_refuses(tmp_path, cfg, approvals_path, redraft_pair):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    r["targets"][REDRAFT_IDS[1]]["scenario_text_sha256"] = "0" * 64
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any(REDRAFT_IDS[1] in p and "stale binding" in p for p in problems)


@pytest.mark.parametrize("bad_targets", [None, [REDRAFT_IDS[0], REDRAFT_IDS[1]],
                                         "technology_08_v1", 1, True])
def test_a_malformed_targets_field_refuses_cleanly(tmp_path, cfg, approvals_path, redraft_pair,
                                                    bad_targets):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    if bad_targets is None:
        del r["targets"]
    else:
        r["targets"] = bad_targets
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert any("targets field must be a mapping" in p for p in problems)
    # Missing entirely, or malformed: either way, the live redraft targets
    # still end up reported as not covered, rather than crashing.
    assert any(REDRAFT_IDS[0] in p or REDRAFT_IDS[1] in p for p in problems)


@pytest.mark.parametrize("bad_entry", [None, "call-id-as-a-bare-string", ["call-id", "sha"], 1,
                                       True])
def test_a_malformed_target_entry_refuses_cleanly(tmp_path, cfg, approvals_path, redraft_pair,
                                                   bad_entry):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    if bad_entry is None:
        del r["targets"][REDRAFT_IDS[0]]
    else:
        r["targets"][REDRAFT_IDS[0]] = bad_entry
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    if bad_entry is None:
        assert any(REDRAFT_IDS[0] in p and "does not name" in p for p in problems)
    else:
        assert any(REDRAFT_IDS[0] in p and "must be a mapping" in p for p in problems)


def test_several_problems_are_all_reported_together(tmp_path, cfg, approvals_path, redraft_pair):
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    r["status"] = "proposed"
    r["kind"] = "group"
    record = load_stage_authorization(_write_record(tmp_path, r))
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=redraft_pair,
                                            approvals_path=approvals_path)
    assert len(problems) >= 2


# --- CLI-level: refuses before a credential or backend, --only stays powerless ----


def test_cli_status_proposed_cannot_send(cfg_path, approvals_path, redraft_pair, cfg, tmp_path,
                                         capsys):
    """The dry-run/refusal the researcher's review needs to see: a proposed
    record authorises nothing, and nothing is sent. A real before/after
    comparison of the generation log — not merely an existence check — is
    the proof: the refused invocation must add nothing to it at all."""
    pilot = _load_pilot_script()
    r = _baseline_record(cfg, approvals_path, redraft_pair)
    r["status"] = "proposed"
    r["authorized_by"] = None
    r["authorized_at"] = None
    path = _write_record(tmp_path, r)

    log_path = tmp_path / "run_v2" / "generation_log.jsonl"
    before = log_path.read_bytes()

    rc = pilot.main(["redraft-scenarios", "--config", str(cfg_path),
                     "--stage-authorization", str(path)])
    out = capsys.readouterr()
    assert rc == 1
    assert "refusing" in out.err
    assert "not 'authorized'" in out.err

    after = log_path.read_bytes()
    assert after == before, "a refused invocation must not append to the generation log"
    store = CallStore(tmp_path / "run_v2", cfg)
    assert all(e["kind"] != "scenario_redraft" for e in store.log.entries())


def test_cli_without_stage_authorization_is_unchanged(cfg_path, redraft_pair, capsys):
    """No ``--stage-authorization``: the frozen configuration's own authorisation
    governs, exactly as before this mechanism existed."""
    pilot = _load_pilot_script()
    rc = pilot.main(["redraft-scenarios", "--config", str(cfg_path)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "outside the recorded authorisation" in err


def test_cli_only_cannot_narrow_the_reviewed_set_even_with_authorization(
        cfg_path, cfg, approvals_path, redraft_pair, tmp_path, capsys):
    pilot = _load_pilot_script()
    path = _write_record(tmp_path, _baseline_record(cfg, approvals_path, redraft_pair))
    rc = pilot.main(["redraft-scenarios", "--config", str(cfg_path),
                     "--stage-authorization", str(path), "--only", "technology_08"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "cannot narrow a set that comes from the approvals file" in err


def test_cli_authorized_dry_run_reports_and_sends_nothing(cfg_path, cfg, approvals_path,
                                                          redraft_pair, tmp_path, capsys):
    """Even a fully ``authorized`` record sends nothing without ``--send`` and
    the live-generation environment keys — the ordinary dry-run gate, on top
    of the stage authorisation."""
    pilot = _load_pilot_script()
    path = _write_record(tmp_path, _baseline_record(cfg, approvals_path, redraft_pair))
    rc = pilot.main(["redraft-scenarios", "--config", str(cfg_path),
                     "--stage-authorization", str(path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "stage        redraft: 2 call(s)" in out
    assert "nothing was sent" in out
    store = CallStore(tmp_path / "run_v2", cfg)
    assert all(e["kind"] != "scenario_redraft" for e in store.log.entries())


# --- the happy path: exactly two calls, traced to the authorisation record --------


class RedraftResponder:
    """Answers a redraft call with new, still-valid synthetic text."""

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, request):
        scenario_id = f"{request.decision_id}_v{request.variant_id}"
        self.calls.append(scenario_id)
        return {"scenario_text": scenario_text(scenario_id).replace(
            "regional capacity decision", "regional capacity decision, revised")}


def test_authorized_record_permits_exactly_two_traceable_redraft_calls(
        tmp_path, cfg, approvals_path, redraft_pair, store_with_96_scenarios):
    path = _write_record(tmp_path, _baseline_record(cfg, approvals_path, redraft_pair))
    record = load_stage_authorization(path)
    assert stage_authorization_problems(
        record, cfg=cfg, kind="scenario_redraft", record_version=RECORD_VERSION,
        targets=redraft_pair, approvals_path=approvals_path) == []

    store = store_with_96_scenarios
    store.allowed_kinds = frozenset({"scenario_redraft"})
    responder = RedraftResponder()
    topics = _new_topics(plan_full_corpus(cfg, BANK, load_seed_corpus(cfg, bank=BANK, root=ROOT)))
    from reasonstyle.generation.approvals import load_approvals
    approvals = load_approvals(approvals_path)

    results = run_redraft_stage(topics, approvals, cfg, segmenter_from_config(cfg),
                                FakeBackend(responder), store, allow_live=True,
                                stage_authorization=record)

    assert sorted(responder.calls) == sorted(REDRAFT_IDS)
    assert len(results) == 2
    entries = [e for e in store.log.entries() if e["kind"] == "scenario_redraft"]
    assert len(entries) == 2
    for entry in entries:
        assert entry["extra"]["stage_authorization_sha256"] == record["_file_sha256"]
        assert entry["extra"]["stage_authorization_path"] == str(record["_path"])


# --- resumability: never more than one call per target, never more than the ceiling -


class InterruptedResponder:
    """Fails the second alphabetical target once, then succeeds."""

    def __init__(self, fail_once: str):
        self._fail_once = fail_once
        self.calls: list[str] = []

    def __call__(self, request):
        scenario_id = f"{request.decision_id}_v{request.variant_id}"
        self.calls.append(scenario_id)
        if scenario_id == self._fail_once:
            self._fail_once = None
            return ConnectionError("simulated transport failure")
        return {"scenario_text": scenario_text(scenario_id).replace(
            "regional capacity decision", "regional capacity decision, revised")}


@pytest.fixture
def interrupted_run(tmp_path, cfg, approvals_path, redraft_pair, store_with_96_scenarios):
    """``technology_08_v1`` sorts first and completes; ``technology_13_v1``
    hits a simulated transport failure. Returns ``(store, record, topics,
    approvals, segmenter)`` for the tests below."""
    store = store_with_96_scenarios
    store.allowed_kinds = frozenset({"scenario_redraft"})
    record = load_stage_authorization(
        _write_record(tmp_path, _baseline_record(cfg, approvals_path, redraft_pair)))
    topics = _new_topics(plan_full_corpus(cfg, BANK, load_seed_corpus(cfg, bank=BANK, root=ROOT)))
    from reasonstyle.generation.approvals import load_approvals
    approvals = load_approvals(approvals_path)
    segmenter = segmenter_from_config(cfg)

    responder = InterruptedResponder(fail_once="technology_13_v1")
    with pytest.raises(PipelineAbort):
        run_redraft_stage(topics, approvals, cfg, segmenter, FakeBackend(responder), store,
                          allow_live=True, stage_authorization=record)
    assert responder.calls == ["technology_08_v1", "technology_13_v1"]
    return store, record, topics, approvals, segmenter


def test_a_transport_failure_is_recorded_with_the_authorization_hash(interrupted_run, tmp_path):
    """Both attempts are logged — one completed (technology_08_v1), one a
    transport-failure audit line (technology_13_v1) that consumed no result
    but is still traceable to exactly the authorisation that permitted the
    dispatch, the same as a completed call would be."""
    store, record, *_ = interrupted_run
    entries = [e for e in store.log.entries() if e["kind"] == "scenario_redraft"]
    assert len(entries) == 2
    completed = [e for e in entries if e["status"] == "ok"]
    failed = [e for e in entries if e["status"] == "error"]
    assert len(completed) == 1 and completed[0]["decision_id"] == "technology_08"
    assert len(failed) == 1 and failed[0]["decision_id"] == "technology_13"
    for entry in entries:
        assert entry["extra"]["stage_authorization_sha256"] == record["_file_sha256"]
        assert entry["extra"]["stage_authorization_path"] == str(record["_path"])


def test_transport_blocked_targets_finds_the_failed_target(interrupted_run):
    store, record, _topics, approvals, _segmenter = interrupted_run
    targets = redraft_targets(approvals, recorded_scenarios(store))
    blocked = transport_blocked_targets(store, targets, record["_file_sha256"])
    assert [t.scenario_id for t in blocked] == ["technology_13_v1"]


def test_rerunning_under_the_same_authorization_after_a_transport_failure_is_blocked(
        interrupted_run, cfg):
    """The one call ``technology_13_v1`` was granted under this exact
    authorisation was already spent, even though it ended in a transport
    failure. Redispatching it automatically under the *same* record would
    risk a second paid attempt for a call this record only ever granted
    once — refused, and nothing new reaches the backend, not even for
    ``technology_08_v1``, which would merely have resumed."""
    store, record, topics, approvals, segmenter = interrupted_run
    entries_before = len([e for e in store.log.entries() if e["kind"] == "scenario_redraft"])

    responder2 = RedraftResponder()
    with pytest.raises(PipelineAbort, match="technology_13_v1"):
        run_redraft_stage(topics, approvals, cfg, segmenter, FakeBackend(responder2), store,
                          allow_live=True, stage_authorization=record)
    assert responder2.calls == []          # nothing was dispatched, not even the resumable one
    entries_after = len([e for e in store.log.entries() if e["kind"] == "scenario_redraft"])
    assert entries_after == entries_before


def test_a_new_retry_authorization_record_permits_dispatching_the_remaining_target(
        interrupted_run, tmp_path, cfg, approvals_path, redraft_pair):
    """A second, distinct authorisation record — same bindings, a different
    file and so a different SHA-256 — is what a genuine retry needs. Under
    it, ``technology_13_v1`` is no longer blocked (no transport-failure entry
    carries *this* record's hash) and dispatches; ``technology_08_v1``
    resumes from disk exactly as before, without being asked of the backend."""
    store, _old_record, topics, approvals, segmenter = interrupted_run
    # A genuinely new record, not merely a copy under a new filename: content
    # identical bytes would hash identically regardless of path, so the
    # retry is dated distinctly — exactly what a real, later, separate
    # authorisation decision would be.
    retry_content = _baseline_record(cfg, approvals_path, redraft_pair)
    retry_content["authorized_at"] = "2026-09-25"
    retry_record = load_stage_authorization(
        _write_record(tmp_path, retry_content, name="retry_auth.yaml"))
    assert retry_record["_file_sha256"] != _old_record["_file_sha256"]
    assert transport_blocked_targets(
        store, redraft_targets(approvals, recorded_scenarios(store)),
        retry_record["_file_sha256"]) == []

    responder2 = RedraftResponder()
    results = run_redraft_stage(topics, approvals, cfg, segmenter, FakeBackend(responder2), store,
                                allow_live=True, stage_authorization=retry_record)
    assert responder2.calls == ["technology_13_v1"]
    assert len(results) == 2
    entries = [e for e in store.log.entries() if e["kind"] == "scenario_redraft"]
    # The old transport-failure line for technology_13_v1 is never removed —
    # it stays as evidence — and the new successful attempt, traceable to the
    # retry record, is a further line beside it.
    assert len(entries) == 3
    completed = [e for e in entries if e["status"] == "ok"]
    assert {e["decision_id"] for e in completed} == {"technology_08", "technology_13"}
    new_completed = next(e for e in completed if e["decision_id"] == "technology_13")
    assert new_completed["extra"]["stage_authorization_sha256"] == retry_record["_file_sha256"]


def test_a_fully_completed_stage_never_calls_the_backend_again(
        tmp_path, cfg, approvals_path, redraft_pair, store_with_96_scenarios):
    store = store_with_96_scenarios
    store.allowed_kinds = frozenset({"scenario_redraft"})
    record = load_stage_authorization(
        _write_record(tmp_path, _baseline_record(cfg, approvals_path, redraft_pair)))
    topics = _new_topics(plan_full_corpus(cfg, BANK, load_seed_corpus(cfg, bank=BANK, root=ROOT)))
    from reasonstyle.generation.approvals import load_approvals
    approvals = load_approvals(approvals_path)
    segmenter = segmenter_from_config(cfg)

    run_redraft_stage(topics, approvals, cfg, segmenter, FakeBackend(RedraftResponder()), store,
                      allow_live=True, stage_authorization=record)
    entries_after_first = len([e for e in store.log.entries() if e["kind"] == "scenario_redraft"])

    responder_again = RedraftResponder()
    results = run_redraft_stage(topics, approvals, cfg, segmenter, FakeBackend(responder_again),
                                store, allow_live=True, stage_authorization=record)
    assert responder_again.calls == []            # nothing new was asked of the backend
    assert len(results) == 2
    entries_after_second = len([e for e in store.log.entries() if e["kind"] == "scenario_redraft"])
    assert entries_after_second == entries_after_first == 2


# --- the shipped record, and hash invariance on the real committed files ----------


def test_the_real_authorization_record_is_proposed_and_otherwise_valid():
    """Read-only against the real committed files. The shipped record is not
    operative for three separate, independent reasons — ``status`` is not
    ``'authorized'``, ``authorized_by`` names no reviewer, and
    ``authorized_at`` carries no date — while every scientific binding
    (configuration, approvals file, targets, call ids, text hashes,
    ``max_paid_calls`` against the target count, ``record_version``) already
    matches reality and produces no separate refusal."""
    cfg = load_config(FULL)
    assert cfg.content_hash == FROZEN_CONTENT_HASH
    from reasonstyle.generation.approvals import load_approvals
    approvals = load_approvals(REAL_APPROVALS)
    store = CallStore(ROOT / "data/full/run_v2", cfg)
    targets = redraft_targets(approvals, recorded_scenarios(store))
    assert sorted(t.scenario_id for t in targets) == sorted(REDRAFT_IDS)

    record = load_stage_authorization(REAL_RECORD)
    assert record["status"] == "proposed"
    assert record["authorized_by"] is None and record["authorized_at"] is None
    problems = stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                            record_version=RECORD_VERSION, targets=targets,
                                            approvals_path=REAL_APPROVALS)
    assert problems == [
        "the stage authorisation is 'proposed', not 'authorized'",
        "the stage authorisation names no reviewer: authorized_by must be a non-empty "
        "string, naming who authorised it",
        "the stage authorisation carries no valid authorisation date: authorized_at "
        "must be a non-empty ISO-8601 date or timestamp",
    ]


def test_the_real_frozen_config_and_approvals_hashes_are_unchanged():
    assert file_sha256(FULL) == "5d34bad7fc3b81d122fc0feb62fb502097f2b220200544027e42a70c94abe819"
    assert load_config(FULL).content_hash == FROZEN_CONTENT_HASH
    assert (file_sha256(REAL_APPROVALS)
           == "fd80575cf762513df5f2070cc181a2dc0e46112dfa7cf308fb82dd7ea2659346")


def test_using_the_mechanism_reads_but_never_writes_the_real_files():
    """Loading and checking the record is exercised end to end here, entirely
    read-only, and the real frozen config and approvals file are hashed
    before and after to prove it."""
    before_cfg, before_approvals = file_sha256(FULL), file_sha256(REAL_APPROVALS)
    cfg = load_config(FULL)
    from reasonstyle.generation.approvals import load_approvals
    approvals = load_approvals(REAL_APPROVALS)
    store = CallStore(ROOT / "data/full/run_v2", cfg)
    targets = redraft_targets(approvals, recorded_scenarios(store))
    record = load_stage_authorization(REAL_RECORD)
    stage_authorization_problems(record, cfg=cfg, kind="scenario_redraft",
                                 record_version=RECORD_VERSION, targets=targets,
                                 approvals_path=REAL_APPROVALS)
    assert file_sha256(FULL) == before_cfg
    assert file_sha256(REAL_APPROVALS) == before_approvals
