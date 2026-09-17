"""The final five scenario edits are separate, bound and revalidated records."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from reasonstyle.corpus import segmenter_from_config
from reasonstyle.generation.approvals import approval_status, load_approvals
from reasonstyle.generation.pipeline import CallStore
from reasonstyle.generation.redraft import current_scenarios
from reasonstyle.generation.scenario_corrections import (
    ScenarioCorrectionError,
    apply_scenario_corrections,
    load_scenario_corrections,
)
from reasonstyle.hashing import content_hash

ROOT = Path(__file__).resolve().parents[1]
CORRECTIONS = ROOT / "data" / "pilot" / "scenario_corrections.yaml"
APPROVALS = ROOT / "data" / "pilot" / "scenario_approvals.yaml"
LIVE_RUN = ROOT / "data" / "pilot" / "run" / "pilot_redraft_snapshot_2026-09-17"

CORRECTED = {
    "climate_01_v1", "energy_01_v1", "energy_03_v2",
    "technology_03_v1", "technology_03_v2",
}


def test_the_ledger_has_exactly_five_approved_hash_bound_corrections(cfg):
    corrections = load_scenario_corrections(CORRECTIONS)
    assert {c.scenario_id for c in corrections} == CORRECTED
    assert all(c.approval_state == "approved" for c in corrections)
    assert all(c.editor == "Vidhi Bhutani" for c in corrections)
    assert all(c.original_text_sha256 and c.corrected_text_sha256 for c in corrections)
    assert all(not c.validation_error_codes and not c.validation_warning_codes
               for c in corrections)

    scenarios = {c.scenario_id: {
        "scenario_text": c.original_text,
        "call_id": c.original_call_id,
        "outcome": "accepted",
        "error_codes": [],
    } for c in corrections}
    applied = apply_scenario_corrections(
        scenarios, corrections, cfg, segmenter_from_config(cfg))
    assert all(applied[sid]["scenario_text"] != scenarios[sid]["scenario_text"]
               for sid in CORRECTED)
    assert all(not applied[sid]["error_codes"] for sid in CORRECTED)


@pytest.mark.parametrize("change,match", [
    ("call", "correction names call"),
    ("text", "original_text is not what the model returned"),
    ("state", "not approved"),
])
def test_a_correction_must_match_its_source_and_be_approved(cfg, change, match):
    correction = load_scenario_corrections(CORRECTIONS)[0]
    scenario = {correction.scenario_id: {
        "scenario_text": correction.original_text,
        "call_id": correction.original_call_id,
        "outcome": "accepted", "error_codes": [],
    }}
    if change == "call":
        scenario[correction.scenario_id]["call_id"] = "wrong-call"
    elif change == "text":
        scenario[correction.scenario_id]["scenario_text"] = "different text"
    else:
        correction = replace(correction, approval_state="pending")
    with pytest.raises(ScenarioCorrectionError, match=match):
        apply_scenario_corrections(
            scenario, [correction], cfg, segmenter_from_config(cfg))


def test_a_human_edit_cannot_override_a_machine_error(cfg):
    correction = load_scenario_corrections(CORRECTIONS)[0]
    bad = replace(correction, corrected_text="The authority should choose option A.")
    scenario = {bad.scenario_id: {
        "scenario_text": bad.original_text, "call_id": bad.original_call_id,
        "outcome": "accepted", "error_codes": [],
    }}
    with pytest.raises(ScenarioCorrectionError, match="machine errors"):
        apply_scenario_corrections(scenario, [bad], cfg, segmenter_from_config(cfg))


def test_all_twenty_four_current_texts_satisfy_the_final_gate(cfg, pilot_bank):
    if not LIVE_RUN.is_dir():
        pytest.skip(f"{LIVE_RUN} is local read-only evidence")
    bank_hash = content_hash(pilot_bank.model_dump(mode="json"))
    store = CallStore(LIVE_RUN, cfg, topic_bank_content_hash=bank_hash)
    scenarios = apply_scenario_corrections(
        current_scenarios(store), load_scenario_corrections(CORRECTIONS), cfg,
        segmenter_from_config(cfg))
    approvals = load_approvals(APPROVALS)
    assert len(scenarios) == len(approvals) == 24
    for scenario_id, record in scenarios.items():
        state, reasons = approval_status(
            scenario_id, record["scenario_text"], record["call_id"], approvals,
            config_content_hash=cfg.content_hash,
            topic_bank_content_hash=bank_hash,
            machine_errors=len(record.get("error_codes") or []))
        assert state == "approved", (scenario_id, reasons)


def test_the_group_stage_drafts_from_the_corrected_text(cfg, pilot_bank):
    """The point of the ledger: a corrected scenario is what the groups are
    built on. A group drafted from the superseded model text would be building
    on a scenario nobody approved."""
    if not LIVE_RUN.is_dir():
        pytest.skip(f"{LIVE_RUN} is local read-only evidence")
    import shutil
    import tempfile

    from reasonstyle.generation import FakeBackend, allocate_markers
    from reasonstyle.generation.pipeline import run_group_stage

    corrections = load_scenario_corrections(CORRECTIONS)
    bank_hash = content_hash(pilot_bank.model_dump(mode="json"))
    segmenter = segmenter_from_config(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        shutil.copytree(LIVE_RUN, run)                 # never write to the evidence
        store = CallStore(run, cfg, topic_bank_content_hash=bank_hash)
        scenarios = apply_scenario_corrections(
            current_scenarios(store), corrections, cfg, segmenter)

        seen = []

        def responder(request):
            seen.append(request)
            return {"RS": "x", "RP": "x", "NS": "x", "NP": "x"}   # rejected; no call needed

        topic = next(t for t in pilot_bank.topics if t.decision_id == "climate_01")
        allocation = allocate_markers(pilot_bank, cfg)
        run_group_stage([topic], allocation.groups, cfg, segmenter, FakeBackend(responder),
                        store, approvals=load_approvals(APPROVALS),
                        topic_bank_content_hash=bank_hash, variants=(1,),
                        scenarios=scenarios)

        assert seen, "the gate let the groups through"
        corrected = next(c for c in corrections if c.scenario_id == "climate_01_v1")
        for request in seen:
            assert corrected.corrected_text in request.prompt
            assert corrected.original_text not in request.prompt
