"""The two live stages and the corpus-wide assembly, driven offline.

Everything here runs through ``FakeBackend``: no socket, no GPU, no server. The
properties under test are the ones that keep a corpus honest — that a stage
cannot cross the curator's approval boundary by itself, that a completed call is
never paid for twice, that a partial pilot assembles nothing, and that no code
path affirms a human judgement.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from reasonstyle.corpus import load_corpus, validate_corpus
from reasonstyle.generation import FakeBackend
from reasonstyle.generation.approvals import REQUIRED_JUDGEMENTS, ScenarioApproval
from reasonstyle.generation.assemble import (
    AssemblyRefused,
    ManualCorrection,
    assemble_pilot,
    write_pilot_corpus,
)
from reasonstyle.generation.pipeline import (
    ACCEPTED,
    NEEDS_MANUAL_REVIEW,
    CallStore,
    PipelineAbort,
    recorded_groups,
    recorded_scenarios,
    run_group_stage,
    run_scenario_stage,
)
from reasonstyle.hashing import content_hash, sha256_of


# --- synthetic material that satisfies every machine rule --------------------


def scenario_text(scenario_id: str) -> str:
    """Distinct per scenario, so no two rendered texts collide."""
    return (f"Record {scenario_id} sets out a regional capacity decision for the coming "
            f"winters. The operator can extend the existing baseload plant or accelerate "
            f"the storage build already under tender. Both routes are funded and feasible, "
            f"and nothing states how reliability should be weighed against emissions. The "
            f"extended plant can deliver full output through any cold spell. Retiring it on "
            f"schedule would cut power-sector emissions. One route must be chosen before "
            f"the tender closes.")


def bodies_for(marker: str, tag: str) -> dict[str, str]:
    """Four cells that pass sentence, ratio, marker and pair-content rules.

    The plain member of each pair is its styled member with the connective
    removed and the clauses joined by "and" — the one permitted difference.
    """
    styled = marker[0].upper() + marker[1:]
    # The opening sentence is long enough that a three-word connective cannot by
    # itself push the group past the word-ratio limit.
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


class PilotResponder:
    """Answers scenario and group calls with valid synthetic material."""

    def __init__(self, *, fail: set[str] | None = None):
        self.requests = []
        self.fail = fail or set()

    def __call__(self, request):
        self.requests.append(request)
        scenario_id = f"{request.decision_id}_v{request.variant_id}"
        if request.kind == "scenario":
            if scenario_id in self.fail:
                return {"scenario_text": "Pick option A."}       # leaks a display label
            return {"scenario_text": scenario_text(scenario_id)}
        tag = f"{scenario_id} {request.supported_option}".replace("_", " ")
        return bodies_for(request.context["marker_string"], tag)

    @property
    def kinds(self):
        return [r.kind for r in self.requests]


@pytest.fixture
def topics(pilot_bank):
    return [t for t in pilot_bank.topics if t.status == "curated"]


@pytest.fixture
def bank_hash(pilot_bank):
    return content_hash(pilot_bank.model_dump(mode="json"))


@pytest.fixture
def allocation(pilot_bank, cfg):
    from reasonstyle.generation import allocate_markers
    return allocate_markers(pilot_bank, cfg)


@pytest.fixture
def store(tmp_path, cfg, bank_hash, allocation):
    return CallStore(tmp_path / "run", cfg, topic_bank_content_hash=bank_hash,
                     allocation_content_hash=allocation.content_hash)


def approvals_for(store, cfg, bank_hash, *, scenarios=None, skip=(), stale=()):
    """Approve every recorded scenario, except the ones named."""
    out = {}
    for scenario_id, record in (scenarios or recorded_scenarios(store)).items():
        if scenario_id in skip or not record.get("scenario_text"):
            continue
        text = ("some other text" if scenario_id in stale else record["scenario_text"])
        out[scenario_id] = ScenarioApproval(
            scenario_id=scenario_id, scenario_text_sha256=sha256_of(text),
            call_id=record["call_id"], config_content_hash=cfg.content_hash,
            topic_bank_content_hash=bank_hash, decision="approved",
            judgements={name: True for name in REQUIRED_JUDGEMENTS},
            decided_by="Vidhi Bhutani", decided_at=date(2026, 9, 20))
    return out


def _run_stage_one(topics, cfg, segmenter, store, responder=None):
    responder = responder or PilotResponder()
    results = run_scenario_stage(topics, cfg, segmenter, FakeBackend(responder), store)
    return responder, results


# --- stage separation --------------------------------------------------------


def test_the_scenario_stage_never_calls_a_group(topics, cfg, segmenter, store):
    responder, results = _run_stage_one(topics, cfg, segmenter, store)
    assert set(responder.kinds) == {"scenario"}
    assert len(results) == len(topics) * 2 == 24
    assert all(r.kind == "scenario" and r.outcome == ACCEPTED for r in results)


def test_the_scenario_stage_ceiling_is_one_call_per_scenario(topics, cfg, segmenter, store):
    responder, _ = _run_stage_one(topics, cfg, segmenter, store)
    assert len(responder.requests) == 24, "24 scenarios, one call each, no repair path"


def test_the_group_stage_never_calls_a_scenario(topics, cfg, segmenter, store, bank_hash,
                                                allocation):
    _run_stage_one(topics, cfg, segmenter, store)
    responder = PilotResponder()
    results = run_group_stage(topics, allocation.groups, cfg, segmenter,
                              FakeBackend(responder), store,
                              approvals=approvals_for(store, cfg, bank_hash),
                              topic_bank_content_hash=bank_hash)
    assert set(responder.kinds) <= {"group", "repair"}
    assert "scenario" not in responder.kinds
    assert len(results) == 48 and all(r.outcome == ACCEPTED for r in results)


def test_the_group_stage_ceiling_is_three_calls_per_group(topics, cfg, segmenter, store,
                                                          bank_hash, allocation):
    _run_stage_one(topics, cfg, segmenter, store)
    responder = PilotResponder()
    run_group_stage(topics, allocation.groups, cfg, segmenter, FakeBackend(responder), store,
                    approvals=approvals_for(store, cfg, bank_hash),
                    topic_bank_content_hash=bank_hash)
    budget = cfg.raw["corpus"]["repair"]["max_calls_per_group"]
    assert len(responder.requests) <= 48 * budget == 144
    assert len(responder.requests) == 48, "clean drafts need no repair"


def test_the_group_stage_will_not_generate_a_missing_scenario(topics, cfg, segmenter, store,
                                                              bank_hash, allocation):
    """Convenience here would mean drafting text nobody approved, then building
    on it."""
    responder, _ = _run_stage_one(topics, cfg, segmenter, store)
    approvals = approvals_for(store, cfg, bank_hash)
    # forget one scenario entirely
    scenarios = recorded_scenarios(store)
    missing = sorted(scenarios)[0]
    store.result_path(scenarios[missing]["call_id"]).unlink()
    store.raw_path(scenarios[missing]["call_id"]).unlink()
    lines = [json.loads(line) for line in store.log.path.read_text().splitlines()
             if json.loads(line)["call_id"] != scenarios[missing]["call_id"]]
    store.log.path.write_text("".join(json.dumps(e) + "\n" for e in lines))

    group_responder = PilotResponder()
    with pytest.raises(PipelineAbort, match="no usable scenario draft"):
        run_group_stage(topics, allocation.groups, cfg, segmenter,
                        FakeBackend(group_responder), store, approvals=approvals,
                        topic_bank_content_hash=bank_hash)
    assert group_responder.requests == []


@pytest.mark.parametrize("gate", [
    {"skip": ("climate_01_v1",)},            # one pending approval
    {"stale": ("energy_02_v2",)},            # one approval bound to other text
])
def test_one_unapproved_scenario_stops_every_group_call(gate, topics, cfg, segmenter, store,
                                                        bank_hash, allocation):
    _run_stage_one(topics, cfg, segmenter, store)
    approvals = approvals_for(store, cfg, bank_hash, **gate)
    responder = PilotResponder()
    with pytest.raises(PipelineAbort, match="curator gate is not satisfied"):
        run_group_stage(topics, allocation.groups, cfg, segmenter, FakeBackend(responder),
                        store, approvals=approvals, topic_bank_content_hash=bank_hash)
    assert responder.requests == [], "not one group call, anywhere"


def test_no_approvals_at_all_stops_every_group_call(topics, cfg, segmenter, store, bank_hash,
                                                    allocation):
    _run_stage_one(topics, cfg, segmenter, store)
    responder = PilotResponder()
    with pytest.raises(PipelineAbort):
        run_group_stage(topics, allocation.groups, cfg, segmenter, FakeBackend(responder),
                        store, approvals=None, topic_bank_content_hash=bank_hash)
    assert responder.requests == []


def test_a_machine_invalid_scenario_blocks_the_whole_gate(topics, cfg, segmenter, store,
                                                          bank_hash, allocation):
    responder, _ = _run_stage_one(topics, cfg, segmenter, store,
                                  PilotResponder(fail={"climate_01_v1"}))
    approvals = approvals_for(store, cfg, bank_hash)
    groups = PilotResponder()
    with pytest.raises(PipelineAbort, match="climate_01_v1"):
        run_group_stage(topics, allocation.groups, cfg, segmenter, FakeBackend(groups), store,
                        approvals=approvals, topic_bank_content_hash=bank_hash)
    assert groups.requests == []


# --- recovery ----------------------------------------------------------------


def test_a_resumed_run_sends_no_completed_call_again(topics, cfg, segmenter, store, bank_hash,
                                                     allocation):
    _run_stage_one(topics, cfg, segmenter, store)
    approvals = approvals_for(store, cfg, bank_hash)
    run_group_stage(topics, allocation.groups, cfg, segmenter, FakeBackend(PilotResponder()),
                    store, approvals=approvals, topic_bank_content_hash=bank_hash)
    before = len(store.log.entries())

    silent = PilotResponder()
    run_scenario_stage(topics, cfg, segmenter, FakeBackend(silent), store)
    run_group_stage(topics, allocation.groups, cfg, segmenter, FakeBackend(silent), store,
                    approvals=approvals, topic_bank_content_hash=bank_hash)
    assert silent.requests == [], "every call was recovered from disk"
    assert len(store.log.entries()) == before, "and none was logged twice"


# --- assembly ----------------------------------------------------------------


def _assembled(topics, cfg, segmenter, store, bank_hash, allocation, corrections=None):
    return assemble_pilot(
        topics=topics, allocation_groups=allocation.groups,
        approvals=approvals_for(store, cfg, bank_hash), corrections=corrections or [],
        scenarios=recorded_scenarios(store), groups=recorded_groups(store),
        cfg=cfg, segmenter=segmenter, topic_bank_content_hash=bank_hash)


def _complete_pilot(topics, cfg, segmenter, store, bank_hash, allocation):
    _run_stage_one(topics, cfg, segmenter, store)
    run_group_stage(topics, allocation.groups, cfg, segmenter, FakeBackend(PilotResponder()),
                    store, approvals=approvals_for(store, cfg, bank_hash),
                    topic_bank_content_hash=bank_hash)


def test_a_complete_pilot_assembles_to_24_scenarios_and_192_texts(
        topics, cfg, segmenter, store, bank_hash, allocation, tmp_path):
    _complete_pilot(topics, cfg, segmenter, store, bank_hash, allocation)
    records, manifest = _assembled(topics, cfg, segmenter, store, bank_hash, allocation)
    assert len(records) == 24
    assert sum(len(r.counterarguments) for r in records) == 48
    assert sum(len(b.cells) for r in records for b in r.counterarguments.values()) == 192
    assert manifest["n_texts"] == 192 and manifest["n_scenarios"] == 24
    assert manifest["machine_errors"] == 0
    assert all(r.validation.status == "draft" for r in records)

    corpus, manifest_path = write_pilot_corpus(
        records, manifest, corpus_path=tmp_path / "corpus.jsonl")
    reloaded = load_corpus(corpus)
    assert len(reloaded) == 24
    saved = json.loads(manifest_path.read_text())
    assert len(saved["scenarios"]) == 24
    assert all(entry["scenario_call_id"] for entry in saved["scenarios"])
    assert all(set(entry["group_call_ids"]) == {"opt_1", "opt_2"} for entry in saved["scenarios"])
    report = validate_corpus(reloaded, cfg, segmenter, corpus_scope="pilot")
    assert report.ok and report.human_review, "machine-valid, and still needing review"


def test_a_partial_pilot_assembles_nothing(topics, cfg, segmenter, store, bank_hash,
                                           allocation):
    _run_stage_one(topics, cfg, segmenter, store)      # scenarios only, no groups
    with pytest.raises(AssemblyRefused, match="the pilot is incomplete"):
        _assembled(topics, cfg, segmenter, store, bank_hash, allocation)


def test_a_failed_assembly_leaves_no_partial_corpus(topics, cfg, segmenter, store, bank_hash,
                                                    allocation, tmp_path):
    _run_stage_one(topics, cfg, segmenter, store)
    corpus = tmp_path / "corpus.jsonl"
    with pytest.raises(AssemblyRefused):
        records, manifest = _assembled(topics, cfg, segmenter, store, bank_hash, allocation)
        write_pilot_corpus(records, manifest, corpus_path=corpus)
    assert not corpus.exists()
    assert not corpus.with_suffix(".manifest.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_an_existing_corpus_is_not_replaced_without_saying_so(topics, cfg, segmenter, store,
                                                              bank_hash, allocation,
                                                              tmp_path):
    _complete_pilot(topics, cfg, segmenter, store, bank_hash, allocation)
    records, manifest = _assembled(topics, cfg, segmenter, store, bank_hash, allocation)
    corpus = tmp_path / "corpus.jsonl"
    write_pilot_corpus(records, manifest, corpus_path=corpus)
    first = corpus.read_text()

    with pytest.raises(AssemblyRefused, match="already exists"):
        write_pilot_corpus(records, manifest, corpus_path=corpus)
    assert corpus.read_text() == first
    write_pilot_corpus(records, manifest, corpus_path=corpus, overwrite=True)
    assert corpus.read_text() == first, "deterministic: the same inputs give the same file"


def test_a_correction_that_fails_validation_stops_the_whole_assembly(
        topics, cfg, segmenter, store, bank_hash, allocation):
    _complete_pilot(topics, cfg, segmenter, store, bank_hash, allocation)
    groups = recorded_groups(store)
    (scenario_id, option), record = sorted(groups.items())[0]
    bad = ManualCorrection(
        scenario_id=scenario_id, supported_option=option, condition="RS",
        original_call_id=record["call_id"], original_text=record["bodies"]["RS"],
        corrected_text="One sentence only, with no marker in it.",
        editor="Vidhi Bhutani", reason="tightening the wording",
        decided_at=date(2026, 9, 21), approval_state="approved")
    with pytest.raises(AssemblyRefused, match="machine errors"):
        _assembled(topics, cfg, segmenter, store, bank_hash, allocation, corrections=[bad])


def test_an_unapproved_correction_stops_the_assembly(topics, cfg, segmenter, store, bank_hash,
                                                     allocation):
    _complete_pilot(topics, cfg, segmenter, store, bank_hash, allocation)
    groups = recorded_groups(store)
    (scenario_id, option), record = sorted(groups.items())[0]
    pending = ManualCorrection(
        scenario_id=scenario_id, supported_option=option, condition="RS",
        original_call_id=record["call_id"], original_text=record["bodies"]["RS"],
        corrected_text=record["bodies"]["RS"].replace("relevant", "pertinent"),
        editor="Vidhi Bhutani", reason="wording", decided_at=date(2026, 9, 21),
        approval_state="pending")
    with pytest.raises(AssemblyRefused, match="not approved"):
        _assembled(topics, cfg, segmenter, store, bank_hash, allocation,
                   corrections=[pending])


def test_assembly_needs_the_curator_gate_too(topics, cfg, segmenter, store, bank_hash,
                                             allocation):
    """Machine-valid material with no approval assembles nothing."""
    _complete_pilot(topics, cfg, segmenter, store, bank_hash, allocation)
    with pytest.raises(AssemblyRefused, match="pending"):
        assemble_pilot(topics=topics, allocation_groups=allocation.groups, approvals={},
                       corrections=[], scenarios=recorded_scenarios(store),
                       groups=recorded_groups(store), cfg=cfg, segmenter=segmenter,
                       topic_bank_content_hash=bank_hash)


# --- the complete-pilot boundary ---------------------------------------------


def test_a_live_stage_cannot_be_pointed_at_a_subset(tmp_path):
    """A subset of the pilot is not the pilot: the marker allocation is
    balanced over all 12 decisions and both variants."""
    import importlib.util
    import pathlib as _pathlib
    from reasonstyle.config import load_config
    from reasonstyle.corpus.topics import load_topic_bank
    root = _pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("pilot_cli", root / "scripts" / "pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = load_config(root / "configs" / "experiment.yaml")
    bank = load_topic_bank(root / "data" / "topics" / "pilot_topics.yaml")

    class Args:
        only = None
        variants = [1, 2]

    assert module.whole_pilot_problems(Args(), bank, cfg) == []
    for only, variants, expected in [
        (["climate_01"], [1, 2], "subset"),
        (None, [1], "not both variants"),
        (None, [1, 1], "not both variants"),
        (None, [1, 2, 3], "not both variants"),
        (None, [2, 1], "not both variants"),
    ]:
        args = Args()
        args.only, args.variants = only, variants
        problems = module.whole_pilot_problems(args, bank, cfg)
        assert problems and any(expected in p for p in problems), (only, variants)
    assert module.WHOLE_PILOT_COMMANDS == ("scenarios", "groups", "assemble")


# --- correcting a group the model never got right ----------------------------


class StubbornResponder(PilotResponder):
    """Returns an unrepairable group for one option of one scenario."""

    def __init__(self, stuck: tuple[str, str]):
        super().__init__()
        self.stuck = stuck

    def __call__(self, request):
        reply = super().__call__(request)
        if request.kind in ("group", "repair"):
            key = (f"{request.decision_id}_v{request.variant_id}", request.supported_option)
            if key == self.stuck:
                # One sentence in RS where two are required: never repaired.
                return {**reply, "RS": "One sentence only, with no connective at all."}
        return reply


def _pair_corrections(record, scenario_id, option, marker_bodies):
    """Corrections for RS and RP, bound to the call that produced them."""
    return [
        ManualCorrection(
            scenario_id=scenario_id, supported_option=option, condition=condition,
            original_call_id=record["call_id"], original_text=record["bodies"][condition],
            corrected_text=marker_bodies[condition], editor="Vidhi Bhutani",
            reason="the model never produced two body sentences here",
            decided_at=date(2026, 9, 21), approval_state="approved")
        for condition in ("RS", "RP")
        if record["bodies"][condition] != marker_bodies[condition]]


def _pilot_with_one_stuck_group(topics, cfg, segmenter, store, bank_hash, allocation):
    stuck = ("climate_01_v1", "opt_1")
    _run_stage_one(topics, cfg, segmenter, store)
    run_group_stage(topics, allocation.groups, cfg, segmenter,
                    FakeBackend(StubbornResponder(stuck)), store,
                    approvals=approvals_for(store, cfg, bank_hash),
                    topic_bank_content_hash=bank_hash)
    return stuck


def test_a_needs_manual_review_group_keeps_its_bodies_and_call(topics, cfg, segmenter, store,
                                                               bank_hash, allocation):
    """Without them there would be nothing to correct, and nothing to bind a
    correction to."""
    stuck = _pilot_with_one_stuck_group(topics, cfg, segmenter, store, bank_hash, allocation)
    record = recorded_groups(store)[stuck]
    assert record["outcome"] == NEEDS_MANUAL_REVIEW
    assert record["accepted"] is False
    assert record["bodies"] and record["call_id"]
    assert record["calls"] == cfg.raw["corpus"]["repair"]["max_calls_per_group"]


def test_an_uncorrected_failed_group_still_blocks_assembly(topics, cfg, segmenter, store,
                                                           bank_hash, allocation):
    _pilot_with_one_stuck_group(topics, cfg, segmenter, store, bank_hash, allocation)
    with pytest.raises(AssemblyRefused, match="has no recorded correction"):
        _assembled(topics, cfg, segmenter, store, bank_hash, allocation)


def test_a_corrected_failed_group_assembles_after_revalidation(topics, cfg, segmenter, store,
                                                               bank_hash, allocation,
                                                               tmp_path):
    """The reason manual corrections exist, end to end."""
    stuck = _pilot_with_one_stuck_group(topics, cfg, segmenter, store, bank_hash, allocation)
    groups = recorded_groups(store)
    record = groups[stuck]
    marker = next(g.marker_string for g in allocation.groups
                  if (f"{g.decision_id}_v{g.variant_id}", g.supported_option) == stuck)
    fixed = bodies_for(marker, f"{stuck[0]} {stuck[1]}".replace("_", " "))
    corrections = _pair_corrections(record, stuck[0], stuck[1], fixed)
    assert corrections, "the stuck cell needs correcting"

    log_before = store.log.path.read_text()
    raw_before = sorted(p.name for p in store.log.raw_dir.glob("*.json"))

    records, manifest = _assembled(topics, cfg, segmenter, store, bank_hash, allocation,
                                   corrections=corrections)
    assert len(records) == 24
    corrected = next(r for r in records if r.scenario_id == stuck[0])
    assert corrected.counterarguments[stuck[1]].cells["RS"].body == fixed["RS"]
    entry = next(m for m in manifest["scenarios"] if m["scenario_id"] == stuck[0])
    assert entry["manual_corrections"], "the correction is recorded in the manifest"
    assert all(c["original_call_id"] == record["call_id"] for c in entry["manual_corrections"])
    assert all(c["validation_error_codes"] == [] for c in entry["manual_corrections"])

    # The record of what the model produced is untouched.
    assert store.log.path.read_text() == log_before
    assert sorted(p.name for p in store.log.raw_dir.glob("*.json")) == raw_before


@pytest.mark.parametrize("break_it", ["one_sided", "stale_call", "still_invalid"])
def test_a_bad_correction_of_a_failed_group_is_refused(break_it, topics, cfg, segmenter,
                                                       store, bank_hash, allocation):
    stuck = _pilot_with_one_stuck_group(topics, cfg, segmenter, store, bank_hash, allocation)
    record = recorded_groups(store)[stuck]
    marker = next(g.marker_string for g in allocation.groups
                  if (f"{g.decision_id}_v{g.variant_id}", g.supported_option) == stuck)
    fixed = bodies_for(marker, f"{stuck[0]} {stuck[1]}".replace("_", " "))
    corrections = _pair_corrections(record, stuck[0], stuck[1], fixed)

    if break_it == "one_sided":
        # RS corrected, RP left as it was: the pair no longer shares its
        # content words, which the screen catches.
        corrections = [type(c)(**{**{f.name: getattr(c, f.name)
                                     for f in c.__dataclass_fields__.values()},
                                  "corrected_text": c.corrected_text.replace(
                                      "consideration", "premise")})
                       for c in corrections]
        expected = "machine errors"
    elif break_it == "stale_call":
        corrections = [type(c)(**{**{f.name: getattr(c, f.name)
                                     for f in c.__dataclass_fields__.values()},
                                  "original_call_id": "z" * 64}) for c in corrections]
        expected = "but this group was accepted from"
    else:
        corrections = [type(c)(**{**{f.name: getattr(c, f.name)
                                     for f in c.__dataclass_fields__.values()},
                                  "corrected_text": "Still one sentence, still wrong."})
                       for c in corrections]
        expected = "machine errors"

    with pytest.raises(AssemblyRefused, match=expected):
        _assembled(topics, cfg, segmenter, store, bank_hash, allocation,
                   corrections=corrections)


# --- a partly reviewed approval template --------------------------------------


def test_a_partly_reviewed_template_loads_and_still_blocks_every_group(
        topics, cfg, segmenter, store, bank_hash, allocation, tmp_path):
    """The template the tool writes must be loadable while the curator works
    through it — and must approve nothing until they say so."""
    import yaml
    from reasonstyle.generation.approvals import approval_status, load_approvals, save_approvals
    _run_stage_one(topics, cfg, segmenter, store)
    scenarios = recorded_scenarios(store)
    approved = approvals_for(store, cfg, bank_hash)

    body = {}
    for index, (scenario_id, record) in enumerate(sorted(scenarios.items())):
        if index < 3:                                  # three reviewed so far
            body[scenario_id] = {k: v for k, v in approved[scenario_id].as_dict().items()
                                 if k != "scenario_id"}
        else:
            body[scenario_id] = {
                "scenario_text_sha256": sha256_of(record["scenario_text"]),
                "call_id": record["call_id"], "config_content_hash": cfg.content_hash,
                "topic_bank_content_hash": bank_hash, "decision": "pending",
                "judgements": {name: None for name in REQUIRED_JUDGEMENTS},
                "reason": "not yet reviewed", "decided_by": None, "decided_at": None}
    path = tmp_path / "scenario_approvals.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=True))

    loaded = load_approvals(path)                       # loads: no exception
    assert len(loaded) == 24
    states = {scenario_id: approval_status(
        scenario_id, scenarios[scenario_id]["scenario_text"],
        scenarios[scenario_id]["call_id"], loaded, config_content_hash=cfg.content_hash,
        topic_bank_content_hash=bank_hash)[0] for scenario_id in scenarios}
    assert sum(1 for s in states.values() if s == "approved") == 3
    assert sum(1 for s in states.values() if s == "pending") == 21

    responder = PilotResponder()
    with pytest.raises(PipelineAbort, match="curator gate is not satisfied"):
        run_group_stage(topics, allocation.groups, cfg, segmenter, FakeBackend(responder),
                        store, approvals=loaded, topic_bank_content_hash=bank_hash)
    assert responder.requests == [], "not one group call while anything is pending"


def test_an_approval_without_a_reviewer_does_not_pass_the_gate(cfg, bank_hash, tmp_path):
    import yaml
    from reasonstyle.generation.approvals import ApprovalError, load_approvals
    record = {"scenario_text_sha256": "a" * 64, "call_id": "b" * 64,
              "config_content_hash": cfg.content_hash, "topic_bank_content_hash": bank_hash,
              "decision": "approved",
              "judgements": {name: True for name in REQUIRED_JUDGEMENTS},
              "decided_by": None, "decided_at": None}
    path = tmp_path / "approvals.yaml"
    path.write_text(yaml.safe_dump({"climate_01_v1": record}))
    with pytest.raises(ApprovalError, match="reviewer and date"):
        load_approvals(path)


# --- the two-file write, described accurately ---------------------------------


def test_a_corpus_and_manifest_written_together_agree(topics, cfg, segmenter, store,
                                                      bank_hash, allocation, tmp_path):
    from reasonstyle.generation.assemble import corpus_matches_manifest
    _complete_pilot(topics, cfg, segmenter, store, bank_hash, allocation)
    records, manifest = _assembled(topics, cfg, segmenter, store, bank_hash, allocation)
    corpus, manifest_path = write_pilot_corpus(records, manifest,
                                               corpus_path=tmp_path / "corpus.jsonl")
    consistent, why = corpus_matches_manifest(corpus, manifest_path)
    assert consistent, why
    assert json.loads(manifest_path.read_text())["corpus_sha256"]


def test_an_interruption_between_the_two_renames_is_detectable(topics, cfg, segmenter, store,
                                                               bank_hash, allocation,
                                                               tmp_path):
    """The honest limit of the write: each file is replaced atomically, the pair
    is not one transaction, and a mismatch is detected rather than silent."""
    from reasonstyle.generation.assemble import corpus_matches_manifest
    _complete_pilot(topics, cfg, segmenter, store, bank_hash, allocation)
    records, manifest = _assembled(topics, cfg, segmenter, store, bank_hash, allocation)
    corpus, manifest_path = write_pilot_corpus(records, manifest,
                                               corpus_path=tmp_path / "corpus.jsonl")

    manifest_path.unlink()                              # crash after the first rename
    consistent, why = corpus_matches_manifest(corpus, manifest_path)
    assert not consistent and "manifest" in why

    stale = json.loads(json.dumps({**manifest, "corpus_sha256": "0" * 64}))
    manifest_path.write_text(json.dumps(stale))
    consistent, why = corpus_matches_manifest(corpus, manifest_path)
    assert not consistent and "hashes to" in why
