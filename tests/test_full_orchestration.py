"""Seed-aware orchestration of the full corpus: planning, the seed guard, the
offline preflight and the status view. Nothing here sends anything."""

from __future__ import annotations

import importlib.util
import os
import socket
from pathlib import Path

import pytest

from reasonstyle.config import load_config
from reasonstyle.corpus import segmenter_from_config
from reasonstyle.corpus.topics import load_topic_bank
from reasonstyle.generation import FakeBackend
from reasonstyle.generation.allocation import load_allocation
from reasonstyle.generation.corpus_source import load_seed_corpus, plan_full_corpus
from reasonstyle.generation.pipeline import (
    CallStore,
    PipelineAbort,
    _send_or_resume,
    run_group_stage,
    run_scenario_stage,
)
from reasonstyle.generation.requests import scenario_request

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "configs/experiment_v2_full.draft.yaml"
BANK = load_topic_bank(ROOT / "data/topics/full_topics_v2.yaml")


def _load_pilot_script():
    spec = importlib.util.spec_from_file_location("pilot_script", ROOT / "scripts/pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cfg():
    return load_config(FULL)


@pytest.fixture(scope="module")
def plan(cfg):
    return plan_full_corpus(cfg, BANK, load_seed_corpus(cfg, bank=BANK, root=ROOT))


def topics(ids):
    return [t for t in BANK.topics if t.decision_id in set(ids)]


def never(request):                                     # pragma: no cover - must not run
    raise AssertionError(f"a request was made for {request.decision_id}")


# --- the seed guard -----------------------------------------------------------------


def test_the_full_configuration_keeps_generation_blocked(cfg):
    assert cfg.raw["corpus"]["generation_block"]["blocked"] is True


def test_live_problems_refuse_even_with_every_key(cfg):
    pilot = _load_pilot_script()
    env = {"REASONSTYLE_ALLOW_OPENAI_GENERATION": "1",
           "REASONSTYLE_ALLOW_PILOT_GENERATION": "1"}
    problems = pilot.live_problems(True, cfg, env=env)
    assert problems and "generation is blocked" in problems[0]


def test_a_scenario_request_for_a_seed_decision_is_impossible(cfg, plan, tmp_path):
    store = CallStore(tmp_path / "run", cfg, refused_decisions=frozenset(plan.seed_decisions))
    backend = FakeBackend(never)
    with pytest.raises(PipelineAbort, match="seed decision"):
        run_scenario_stage(topics(["climate_01"]), cfg, segmenter_from_config(cfg), backend, store)
    assert backend.calls == [] and not (tmp_path / "run").exists()


def test_the_send_path_itself_refuses_a_seed_decision(cfg, plan, tmp_path):
    store = CallStore(tmp_path / "run", cfg, refused_decisions=frozenset(plan.seed_decisions))
    request = scenario_request(topics(["energy_02"])[0], 1, cfg)
    backend = FakeBackend(never)
    with pytest.raises(PipelineAbort, match="imported seed corpus"):
        _send_or_resume(store, request, backend, cfg, allow_live=False)
    assert backend.calls == [] and not (tmp_path / "run").exists()


def test_a_group_stage_naming_a_seed_decision_is_refused_before_any_call(cfg, plan, tmp_path):
    store = CallStore(tmp_path / "run", cfg, refused_decisions=frozenset(plan.seed_decisions))
    allocation = load_allocation(ROOT / "data/full/marker_allocation_full_v2.yaml")
    backend = FakeBackend(never)
    with pytest.raises(PipelineAbort, match="seed decision"):
        run_group_stage(topics(["technology_01", "technology_06"]), allocation.groups, cfg,
                        segmenter_from_config(cfg), backend, store, approvals={},
                        topic_bank_content_hash=None, scenarios={},
                        gate_verified_elsewhere=True)
    assert backend.calls == []


def test_pilot_behaviour_is_unchanged_without_a_seed(tmp_path):
    pilot_cfg = load_config(ROOT / "configs/experiment_v2_pilot.yaml")
    store = CallStore(tmp_path / "run", pilot_cfg)
    assert store.refused_decisions == frozenset()


# --- planning through the runner ------------------------------------------------------


def test_planning_skips_the_seed(plan):
    assert len(plan.seed_decisions) == 12
    assert len(plan.new_decisions) == 48
    assert len(plan.new_scenario_ids) == 96 and len(plan.new_groups) == 192
    assert not {s.rsplit("_v", 1)[0] for s in plan.new_scenario_ids} & set(plan.seed_decisions)


def test_the_call_budget(cfg, plan):
    budget = _load_pilot_script().call_budget(cfg, plan)
    assert budget == {"scenarios": 96, "groups": 192, "expected": 304,
                      "primary_ceiling": 672, "absolute_ceiling": 768}


def test_a_dry_group_stage_plans_only_new_groups_and_sends_nothing(capsys):
    pilot = _load_pilot_script()
    assert pilot.main(["groups", "--config", str(FULL)]) == 0
    out = capsys.readouterr().out
    assert "192 drafts" in out and "nothing was sent" in out
    assert not (ROOT / "data/full/run_v2").exists()


def test_the_full_corpus_is_never_assembled_by_the_runner(capsys):
    pilot = _load_pilot_script()
    assert pilot.main(["assemble", "--config", str(FULL)]) == 1
    assert "nothing is assembled" in capsys.readouterr().err


def test_outputs_under_data_pilot_are_refused(tmp_path):
    pilot = _load_pilot_script()
    text = FULL.read_text(encoding="utf-8").replace("  run: data/full/run_v2\n",
                                                    "  run: data/pilot/run_full\n", 1)
    assert "data/pilot/run_full" in text
    path = tmp_path / "full.yaml"
    path.write_text(text, encoding="utf-8")
    problems = pilot.seed_output_problems(load_config(path))
    assert any("under data/pilot/" in p for p in problems)


def test_the_offline_dry_run_renders_every_new_request_and_writes_nothing(cfg, plan, tmp_path):
    pilot = _load_pilot_script()
    allocation = load_allocation(ROOT / "data/full/marker_allocation_full_v2.yaml")
    new = [t for t in BANK.topics
           if t.status == "curated" and t.decision_id in set(plan.new_decisions)]
    counts, problems = pilot.offline_dry_run(cfg, BANK, allocation, new, plan)
    assert problems == []
    assert counts == {"scenarios": 96, "groups": 192}
    assert not (ROOT / "data/full/run_v2").exists()
    assert list(tmp_path.iterdir()) == []


def test_the_offline_dry_run_refuses_a_seed_decision(cfg, plan):
    pilot = _load_pilot_script()
    allocation = load_allocation(ROOT / "data/full/marker_allocation_full_v2.yaml")
    seeded = [t for t in BANK.topics if t.decision_id in {"climate_01"}]
    counts, problems = pilot.offline_dry_run(cfg, BANK, allocation, seeded, plan)
    assert counts == {"scenarios": 0, "groups": 0}
    assert any("seed decision" in p for p in problems)


def test_a_scenario_stage_dry_run_plans_96_and_sends_nothing(capsys):
    pilot = _load_pilot_script()
    assert pilot.main(["scenarios", "--config", str(FULL)]) == 0
    out = capsys.readouterr().out
    assert "scenarios: 96 calls at most" in out and "nothing was sent" in out
    assert "generation is blocked" in out
    assert not (ROOT / "data/full/run_v2").exists()


# --- the preflight: read-only, no credential, no connection ---------------------------------


class _Watched(dict):
    """An environment that records every key read from it."""

    def __init__(self, base):
        super().__init__(base)
        self.read: set[str] = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)


def test_the_preflight_reads_no_credential_and_opens_no_connection(monkeypatch, capsys):
    pilot = _load_pilot_script()
    watched = _Watched(os.environ)
    watched["OPENAI_API_KEY"] = "sk-test-must-never-be-read"
    monkeypatch.setattr(os, "environ", watched)

    def no_network(*args, **kwargs):
        raise AssertionError("the preflight opened a socket")
    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    for name in ("OpenAIResponsesBackend", "VLLMOpenAIBackend"):
        monkeypatch.setattr(pilot, name, lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("a backend was built")))

    assert pilot.main(["preflight", "--config", str(FULL)]) == 0
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY" not in watched.read
    assert "sk-test" not in out
    for expected in ("seed integrity    verified", "12 decisions, 24 scenarios, 48 groups, 192 texts",
                     "skipped (seed)    12", "to generate       48", "96 would be requested",
                     "192 would be requested", "new texts         768", "about 304",
                     "96 rendered in this dry run", "192 rendered in this dry run",
                     "ceiling, primary  672", "ceiling, absolute 768", "generation block  ACTIVE",
                     "no credential was read"):
        assert expected in out, expected
    assert not (ROOT / "data/full/run_v2").exists()


def test_the_preflight_fails_before_networking_on_a_mismatch(monkeypatch, tmp_path, capsys):
    pilot = _load_pilot_script()
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("socket")))
    text = FULL.read_text(encoding="utf-8").replace(
        "corpus_sha256: 7e0dae8415abe0499321bffb19742b031dea8eede08fca1fca20340373e625d8",
        "corpus_sha256: '" + "0" * 64 + "'", 1)
    path = tmp_path / "full.yaml"
    path.write_text(text, encoding="utf-8")
    assert pilot.main(["preflight", "--config", str(path)]) == 1
    assert "not the pinned" in capsys.readouterr().err


def test_the_preflight_fails_on_a_hand_edited_full_allocation(monkeypatch, tmp_path, capsys):
    pilot = _load_pilot_script()
    stored = ROOT / "data/full/marker_allocation_full_v2.yaml"
    copy = tmp_path / "alloc.yaml"
    copy.write_text(stored.read_text(encoding="utf-8").replace("seed: 13\n", "seed: 14\n", 1),
                    encoding="utf-8")
    assert pilot.main(["preflight", "--config", str(FULL), "--allocation", str(copy)]) == 1
    assert "not byte-identical" in capsys.readouterr().err


def test_the_preflight_refuses_a_configuration_without_a_seed(capsys):
    pilot = _load_pilot_script()
    assert pilot.main(["preflight", "--config", str(ROOT / "configs/experiment_v2_pilot.yaml")]) == 1


# --- status ---------------------------------------------------------------------------


def test_status_separates_seed_new_total_and_blockers(capsys):
    pilot = _load_pilot_script()
    assert pilot.main(["status", "--config", str(FULL)]) == 0
    out = capsys.readouterr().out
    for section in ("IMPORTED SEED", "NEW MATERIAL (not generated)", "TOTAL EVENTUAL CORPUS",
                    "CURRENT BLOCKERS"):
        assert section in out
    assert "0 of 96 generated" in out and "0 of 192 generated" in out
    assert "groups               240 allocated (48 imported + 192 new)" in out
    assert "texts                960" in out
    # Derived live, in this run: the allocation is rebuilt and compared, the
    # seed verified, and the dry run actually performed. Only the paid-call
    # authorisation is outstanding.
    assert "[met: rebuilt and compared byte for byte]" in out
    assert "[met: verified read-only in this run]" in out
    assert "96 scenario and 192 group requests rendered offline in this run" in out
    assert "explicit authorisation of paid external-generation calls  [OUTSTANDING" in out


def test_pilot_status_is_unchanged(capsys):
    pilot = _load_pilot_script()
    assert pilot.main(["status", "--config", str(ROOT / "configs/experiment_v2_pilot.yaml")]) == 0
    out = capsys.readouterr().out
    assert "IMPORTED SEED" not in out and "scenarios expected" in out
