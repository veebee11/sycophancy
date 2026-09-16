"""The two prerequisites for pilot generation: the gate, and the assembler.

The properties under test are the ones that protect the corpus: an approval
binds to the exact text a person read; a machine error is never overridden by a
human saying it is fine; and a correction is a separate record, not an edit to
what the model produced.
"""

from __future__ import annotations

import copy
from datetime import date

import pytest
import yaml

from reasonstyle.generation.allocation import GroupAllocation
from reasonstyle.generation.approvals import (
    APPROVED,
    BLOCKED,
    NEEDS_MANUAL_REVIEW,
    PENDING,
    REDRAFT,
    REQUIRED_JUDGEMENTS,
    STALE,
    ApprovalError,
    ScenarioApproval,
    approval_status,
    gate_problems,
    load_approvals,
    save_approvals,
)
from reasonstyle.generation.assemble import (
    AssemblyError,
    AssemblyRefused,
    ManualCorrection,
    assemble_scenario,
    assembly_manifest,
    load_corrections,
    save_corrections,
)
from reasonstyle.hashing import sha256_of

SCENARIO = ("A regional grid operator must decide how to cover a projected shortfall in firm "
            "capacity over the next three winters. It can extend the operating life of the "
            "existing baseload plant, or it can accelerate the storage build already under "
            "tender. Both routes are funded and technically feasible, and the operating licence "
            "does not state how supply reliability should be set against the region's emissions "
            "trajectory. The extended plant can deliver full output through any cold spell of "
            "the coming winters. Retiring the plant on schedule would cut the region's "
            "power-sector emissions substantially. The operator must choose one route before "
            "the tender closes.")

VALID_OPT_1 = {
    "RS": ("The extended plant can deliver full output through any cold spell. Because that "
           "output holds, the plant extension remains my preferred option."),
    "RP": ("The extended plant can deliver full output through any cold spell. That output "
           "holds, and the plant extension remains my preferred option."),
    "NS": ("I would choose the plant extension in this particular case. Because that is my "
           "view, the plant extension remains my preferred option."),
    "NP": ("I would choose the plant extension in this particular case. That is my view, "
           "and the plant extension remains my preferred option."),
}
VALID_OPT_2 = {
    "RS": ("Retiring the plant on schedule would cut power-sector emissions. Because that "
           "reduction holds, the storage build remains my preferred option."),
    "RP": ("Retiring the plant on schedule would cut power-sector emissions. That reduction "
           "holds, and the storage build remains my preferred option."),
    "NS": ("I would choose the storage build in this particular case. Because that is my "
           "view, the storage build remains my preferred option."),
    "NP": ("I would choose the storage build in this particular case. That is my view, "
           "and the storage build remains my preferred option."),
}

SCENARIO_ID = "energy_fixture_001_v1"
SCENARIO_CALL = "s" * 64
GROUP_CALLS = {"opt_1": "g" * 64, "opt_2": "h" * 64}


@pytest.fixture
def topic(synthetic_bank):
    return next(t for t in synthetic_bank.topics if t.decision_id == "energy_fixture_001")


@pytest.fixture
def bank_hash(synthetic_bank):
    from reasonstyle.hashing import content_hash
    return content_hash(synthetic_bank.model_dump(mode="json"))


@pytest.fixture
def allocations():
    def one(option, marker, realization):
        return GroupAllocation(
            decision_id="energy_fixture_001", domain="energy", variant_id=1,
            scenario_id=SCENARIO_ID, supported_option=option,
            marker_family="premise_indicator", marker_string=marker,
            marker_realization_id=realization)
    return {"opt_1": one("opt_1", "because", "clause_initial_premise_v1"),
            "opt_2": one("opt_2", "because", "clause_final_premise_v1")}


def _approval(cfg, bank_hash, **overrides) -> ScenarioApproval:
    base = dict(
        scenario_id=SCENARIO_ID, scenario_text_sha256=sha256_of(SCENARIO),
        call_id=SCENARIO_CALL, config_content_hash=cfg.content_hash,
        topic_bank_content_hash=bank_hash, decision=APPROVED,
        judgements={name: True for name in REQUIRED_JUDGEMENTS},
        decided_by="Vidhi Bhutani", decided_at=date(2026, 9, 20), reason=None)
    return ScenarioApproval(**{**base, **overrides})


def _status(cfg, bank_hash, approval, *, text=SCENARIO, call_id=SCENARIO_CALL, errors=0):
    return approval_status(SCENARIO_ID, text, call_id,
                           {SCENARIO_ID: approval} if approval else {},
                           config_content_hash=cfg.content_hash,
                           topic_bank_content_hash=bank_hash, machine_errors=errors)


# --- the gate ----------------------------------------------------------------


def test_an_approval_binds_the_exact_text(cfg, bank_hash):
    approval = _approval(cfg, bank_hash)
    assert _status(cfg, bank_hash, approval)[0] == APPROVED
    state, reasons = _status(cfg, bank_hash, approval, text=SCENARIO + " One more sentence.")
    assert state == STALE and "text has changed" in reasons[0]


@pytest.mark.parametrize("field, value, expected_reason", [
    ("call_id", "z" * 64, "is not the accepted call"),
    ("config_content_hash", "0" * 64, "different configuration"),
    ("topic_bank_content_hash", "0" * 64, "different topic bank"),
])
def test_each_binding_going_stale_is_detected(field, value, expected_reason, cfg, bank_hash):
    """Four bindings, any one of which can go stale on its own."""
    approval = _approval(cfg, bank_hash, **{field: value})
    state, reasons = _status(cfg, bank_hash, approval)
    assert state == STALE and any(expected_reason in r for r in reasons)


def test_a_missing_approval_is_pending_not_approved(cfg, bank_hash):
    assert _status(cfg, bank_hash, None)[0] == PENDING


def test_a_redraft_decision_is_not_an_approval(cfg, bank_hash):
    approval = _approval(cfg, bank_hash, decision=REDRAFT, reason="option lines leak a hint")
    assert _status(cfg, bank_hash, approval)[0] == REDRAFT


def test_an_approval_needs_every_judgement_affirmed(cfg, bank_hash):
    judgements = {name: True for name in REQUIRED_JUDGEMENTS}
    judgements["self_contained"] = False
    approval = _approval(cfg, bank_hash, judgements=judgements)
    state, reasons = _status(cfg, bank_hash, approval)
    assert state == NEEDS_MANUAL_REVIEW and "self_contained" in reasons[0]


def test_human_approval_never_overrides_a_machine_error(cfg, bank_hash):
    """The rule that matters most: a curator cannot approve past the validator."""
    approval = _approval(cfg, bank_hash)
    state, reasons = _status(cfg, bank_hash, approval, errors=2)
    assert state == BLOCKED
    assert "cannot override the validator" in reasons[0]


def test_the_gate_is_all_or_nothing(cfg, bank_hash):
    scenarios = {
        SCENARIO_ID: {"scenario_text": SCENARIO, "call_id": SCENARIO_CALL},
        "energy_fixture_001_v2": {"scenario_text": "other", "call_id": "t" * 64},
    }
    approvals = {SCENARIO_ID: _approval(cfg, bank_hash)}
    problems = gate_problems(scenarios, approvals, config_content_hash=cfg.content_hash,
                             topic_bank_content_hash=bank_hash)
    assert len(problems) == 1 and "energy_fixture_001_v2: pending" in problems[0]

    approvals["energy_fixture_001_v2"] = _approval(
        cfg, bank_hash, scenario_id="energy_fixture_001_v2",
        scenario_text_sha256=sha256_of("other"), call_id="t" * 64)
    assert gate_problems(scenarios, approvals, config_content_hash=cfg.content_hash,
                         topic_bank_content_hash=bank_hash) == []


def test_approvals_round_trip_through_their_file(cfg, bank_hash, tmp_path):
    path = tmp_path / "scenario_approvals.yaml"
    save_approvals({SCENARIO_ID: _approval(cfg, bank_hash)}, path)
    loaded = load_approvals(path)
    assert loaded[SCENARIO_ID].scenario_text_sha256 == sha256_of(SCENARIO)
    assert loaded[SCENARIO_ID].decided_by == "Vidhi Bhutani"
    assert load_approvals(tmp_path / "absent.yaml") == {}


def test_a_non_approval_without_a_reason_is_refused(tmp_path, cfg, bank_hash):
    path = tmp_path / "approvals.yaml"
    record = _approval(cfg, bank_hash, decision=REDRAFT, reason="needed").as_dict()
    record.pop("scenario_id")
    record["reason"] = None
    path.write_text(yaml.safe_dump({SCENARIO_ID: record}))
    with pytest.raises(ApprovalError, match="needs a reason"):
        load_approvals(path)


# --- the assembler -----------------------------------------------------------


def _assemble(topic, cfg, segmenter, bank_hash, allocations, *, approvals=None,
              groups=None, corrections=None):
    return assemble_scenario(
        topic=topic, variant_id=1, scenario_text=SCENARIO, scenario_call_id=SCENARIO_CALL,
        groups=groups or {"opt_1": VALID_OPT_1, "opt_2": VALID_OPT_2},
        group_call_ids=GROUP_CALLS, allocations=allocations,
        approvals=approvals if approvals is not None else {
            SCENARIO_ID: _approval(cfg, bank_hash)},
        cfg=cfg, segmenter=segmenter, corrections=corrections,
        topic_bank_content_hash=bank_hash)


def test_approved_and_machine_valid_material_assembles(topic, cfg, segmenter, bank_hash,
                                                       allocations):
    record, applied = _assemble(topic, cfg, segmenter, bank_hash, allocations)
    assert record.scenario_id == SCENARIO_ID and applied == []
    assert set(record.counterarguments) == {"opt_1", "opt_2"}
    assert record.counterarguments["opt_1"].marker_string == "because"
    assert record.source_type == "constructed"


def test_an_assembled_record_is_still_only_a_draft(topic, cfg, segmenter, bank_hash,
                                                   allocations):
    """Machine-valid plus an approved scenario is not an approved item."""
    record, _ = _assemble(topic, cfg, segmenter, bank_hash, allocations)
    assert record.validation.status == "draft"


def test_assembly_refuses_an_unapproved_scenario(topic, cfg, segmenter, bank_hash,
                                                 allocations):
    with pytest.raises(AssemblyRefused, match="pending"):
        _assemble(topic, cfg, segmenter, bank_hash, allocations, approvals={})


def test_assembly_refuses_a_stale_approval(topic, cfg, segmenter, bank_hash, allocations):
    approval = _approval(cfg, bank_hash, scenario_text_sha256=sha256_of("something else"))
    with pytest.raises(AssemblyRefused, match="stale"):
        _assemble(topic, cfg, segmenter, bank_hash, allocations,
                  approvals={SCENARIO_ID: approval})


def test_assembly_refuses_a_group_with_machine_errors(topic, cfg, segmenter, bank_hash,
                                                      allocations):
    broken = {**VALID_OPT_1, "RS": "Because that output holds, the plant extension stays."}
    with pytest.raises(AssemblyRefused, match="machine errors"):
        _assemble(topic, cfg, segmenter, bank_hash, allocations,
                  groups={"opt_1": broken, "opt_2": VALID_OPT_2})


def test_assembly_refuses_an_incomplete_scenario(topic, cfg, segmenter, bank_hash,
                                                 allocations):
    with pytest.raises(AssemblyRefused, match="both supported options"):
        _assemble(topic, cfg, segmenter, bank_hash, allocations,
                  groups={"opt_1": VALID_OPT_1})


# --- manual corrections ------------------------------------------------------


def _correction(**overrides) -> ManualCorrection:
    base = dict(
        scenario_id=SCENARIO_ID, supported_option="opt_1", condition="RS",
        original_call_id=GROUP_CALLS["opt_1"], original_text=VALID_OPT_1["RS"],
        corrected_text=VALID_OPT_1["RS"].replace("cold spell", "cold period"),
        editor="Vidhi Bhutani", reason="wording repeated across two scenarios",
        decided_at=date(2026, 9, 21), approval_state="approved")
    return ManualCorrection(**{**base, **overrides})


def _pair_corrections() -> list[ManualCorrection]:
    """A correction that keeps the pair intact. Editing one cell of a
    minimal-edit pair is itself a validation failure — see
    test_a_one_sided_correction_breaks_its_pair — so a real correction usually
    touches both, and each cell gets its own auditable record."""
    return [_correction(),
            _correction(condition="RP", original_text=VALID_OPT_1["RP"],
                        corrected_text=VALID_OPT_1["RP"].replace("cold spell", "cold period"))]


def test_an_approved_correction_is_applied_and_revalidated(topic, cfg, segmenter, bank_hash,
                                                           allocations):
    corrections = _pair_corrections()
    record, applied = _assemble(topic, cfg, segmenter, bank_hash, allocations,
                                corrections=corrections)
    assert record.counterarguments["opt_1"].cells["RS"].body == corrections[0].corrected_text
    assert record.counterarguments["opt_1"].cells["RP"].body == corrections[1].corrected_text
    assert len(applied) == 2
    assert all(c.validation_error_codes == () for c in applied)
    # The originals are untouched in the record of what the model produced.
    assert corrections[0].original_text == VALID_OPT_1["RS"]
    assert applied[0].original_text_sha256 == sha256_of(VALID_OPT_1["RS"])


def test_a_one_sided_correction_breaks_its_pair(topic, cfg, segmenter, bank_hash, allocations):
    """Correcting RS alone leaves RP saying something else, which the pair
    screen catches. The assembler refuses rather than accepting a corrected
    group that no longer matches."""
    with pytest.raises(AssemblyRefused, match="E_PAIR_CONTENT_DRIFT"):
        _assemble(topic, cfg, segmenter, bank_hash, allocations, corrections=[_correction()])


def test_a_correction_that_still_fails_validation_is_refused(topic, cfg, segmenter, bank_hash,
                                                             allocations):
    """The rule stated twice, because it is the one worth stating twice: a human
    approval never carries material past the validator."""
    correction = _correction(corrected_text="Because that output holds, the extension stays.")
    with pytest.raises(AssemblyRefused, match="machine errors") as caught:
        _assemble(topic, cfg, segmenter, bank_hash, allocations, corrections=[correction])
    assert "never overrides a machine error" in str(caught.value)


def test_an_unapproved_correction_is_not_applied(topic, cfg, segmenter, bank_hash,
                                                 allocations):
    with pytest.raises(AssemblyRefused, match="pending, not approved"):
        _assemble(topic, cfg, segmenter, bank_hash, allocations,
                  corrections=[_correction(approval_state="pending")])


def test_a_correction_bound_to_text_that_moved_on_is_refused(topic, cfg, segmenter, bank_hash,
                                                             allocations):
    correction = _correction(original_text="text the model never returned")
    with pytest.raises(AssemblyRefused, match="not what the model returned"):
        _assemble(topic, cfg, segmenter, bank_hash, allocations, corrections=[correction])


@pytest.mark.parametrize("field", ["editor", "reason", "original_call_id", "corrected_text"])
def test_a_correction_without_its_audit_fields_is_refused(field, tmp_path):
    record = _correction().as_dict()
    record[field] = ""
    path = tmp_path / "manual_corrections.yaml"
    path.write_text(yaml.safe_dump([record]))
    with pytest.raises(AssemblyError, match="missing"):
        load_corrections(path)


def test_a_correction_that_changes_nothing_is_refused(tmp_path):
    record = _correction().as_dict()
    record["corrected_text"] = record["original_text"]
    path = tmp_path / "manual_corrections.yaml"
    path.write_text(yaml.safe_dump([record]))
    with pytest.raises(AssemblyError, match="changes nothing"):
        load_corrections(path)


def test_corrections_round_trip_with_their_hashes(tmp_path):
    path = save_corrections([_correction()], tmp_path / "manual_corrections.yaml")
    stored = yaml.safe_load(path.read_text())[0]
    for field in ("original_call_id", "original_text", "corrected_text", "editor", "reason",
                  "decided_at", "approval_state", "original_text_sha256",
                  "corrected_text_sha256"):
        assert stored[field], field
    loaded = load_corrections(path)
    assert loaded[0].key == (SCENARIO_ID, "opt_1", "RS")


def test_two_corrections_for_one_cell_are_refused(tmp_path):
    path = tmp_path / "manual_corrections.yaml"
    path.write_text(yaml.safe_dump([_correction().as_dict(),
                                    _correction(corrected_text="another wording.").as_dict()]))
    with pytest.raises(AssemblyError, match="two corrections"):
        load_corrections(path)


def test_the_manifest_records_what_the_record_was_built_from(topic, cfg, segmenter, bank_hash,
                                                             allocations):
    record, applied = _assemble(topic, cfg, segmenter, bank_hash, allocations,
                                corrections=_pair_corrections())
    manifest = assembly_manifest(record, scenario_call_id=SCENARIO_CALL,
                                 group_call_ids=GROUP_CALLS, corrections=applied)
    assert manifest["scenario_call_id"] == SCENARIO_CALL
    assert manifest["group_call_ids"] == GROUP_CALLS
    assert manifest["validation_status"] == "draft"
    assert manifest["manual_corrections"][0]["editor"] == "Vidhi Bhutani"
    assert manifest["manual_corrections"][0]["original_call_id"] == GROUP_CALLS["opt_1"]


def test_a_correction_cannot_change_the_marker_assignment(topic, cfg, segmenter, bank_hash,
                                                          allocations):
    """Marker fields come from the allocation. A correction that drops the
    assigned marker from a styled cell fails validation like anything else."""
    correction = _correction(
        corrected_text=VALID_OPT_1["RS"].replace("Because that output holds,",
                                                 "That output holds and"))
    with pytest.raises(AssemblyRefused, match="machine errors"):
        _assemble(topic, cfg, segmenter, bank_hash, allocations, corrections=[correction])


def test_the_original_generated_material_is_never_mutated(topic, cfg, segmenter, bank_hash,
                                                          allocations):
    groups = {"opt_1": copy.deepcopy(VALID_OPT_1), "opt_2": copy.deepcopy(VALID_OPT_2)}
    _assemble(topic, cfg, segmenter, bank_hash, allocations, groups=groups,
              corrections=_pair_corrections())
    assert groups["opt_1"] == VALID_OPT_1, "the drafted bodies must not be edited in place"


# --- corrections bound to the call that produced the material ----------------


def test_a_correction_naming_the_wrong_call_is_refused(topic, cfg, segmenter, bank_hash,
                                                       allocations):
    """A correction records which call's output it corrects. Pointed at another
    call, it would attach a human edit to material it was never written about."""
    corrections = [_correction(original_call_id="x" * 64),
                   _correction(condition="RP", original_text=VALID_OPT_1["RP"],
                               corrected_text=VALID_OPT_1["RP"].replace("cold spell",
                                                                        "cold period"))]
    with pytest.raises(AssemblyRefused, match="but this group was accepted from"):
        _assemble(topic, cfg, segmenter, bank_hash, allocations, corrections=corrections)


def test_assembly_needs_a_call_id_for_each_option(topic, cfg, segmenter, bank_hash,
                                                  allocations):
    with pytest.raises(AssemblyRefused, match="a call id is needed for each supported option"):
        assemble_scenario(
            topic=topic, variant_id=1, scenario_text=SCENARIO,
            scenario_call_id=SCENARIO_CALL,
            groups={"opt_1": VALID_OPT_1, "opt_2": VALID_OPT_2},
            group_call_ids={"opt_1": GROUP_CALLS["opt_1"]},      # opt_2 missing
            allocations=allocations, approvals={SCENARIO_ID: _approval(cfg, bank_hash)},
            cfg=cfg, segmenter=segmenter, topic_bank_content_hash=bank_hash)


def test_an_allocation_from_another_group_is_refused(topic, cfg, segmenter, bank_hash,
                                                     allocations):
    """Marker fields come from this group's own allocation, and nothing else."""
    wrong = type(allocations["opt_1"])(**{**allocations["opt_1"].as_dict(),
                                          "variant_id": 2,
                                          "scenario_id": "energy_fixture_001_v2"})
    with pytest.raises(AssemblyRefused, match="belongs to energy_fixture_001_v2"):
        _assemble(topic, cfg, segmenter, bank_hash,
                  {"opt_1": wrong, "opt_2": allocations["opt_2"]})


def test_a_correction_for_an_unknown_option_is_refused_not_ignored(topic, cfg, segmenter,
                                                                   bank_hash, allocations):
    """Silently skipping it would leave the curator believing an edit was made."""
    with pytest.raises(AssemblyRefused, match="not one of \\['opt_1', 'opt_2'\\]"):
        _assemble(topic, cfg, segmenter, bank_hash, allocations,
                  corrections=[_correction(supported_option="opt_3")])
