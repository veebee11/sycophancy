"""The v2 design through the real machinery, not only the validator directly.

A rule that holds when a test calls ``validate_group`` by hand, and is skipped
by the code that actually drafts, assembles or reviews a corpus, is not a rule.
These tests drive the production paths — the pipeline, the assembler, corpus
validation, the reading view and the command line — and check that each one
applies the v2 checks, isolates v2's outputs from v1's, and records both
provenance layers.

Nothing here reaches a network or spends anything: the generator is
``FakeBackend`` throughout, and the command-line tests run with sockets
poisoned.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from datetime import date

import pytest
import yaml

from reasonstyle.config import load_config
from reasonstyle.corpus import segmenter_from_config, validate_corpus
from reasonstyle.corpus.validate import endorsement_text
from reasonstyle.generation import FakeBackend, load_allocation
from reasonstyle.generation.allocation import GroupAllocation, MarkerAllocation
from reasonstyle.generation.approvals import REQUIRED_JUDGEMENTS, ScenarioApproval
from reasonstyle.generation.assemble import AssemblyRefused, assemble_pilot
from reasonstyle.generation.pipeline import (
    ACCEPTED,
    CallStore,
    recorded_groups,
    recorded_scenarios,
    run_group_stage,
    run_scenario_stage,
)
from reasonstyle.generation.scenario_source import (
    ScenarioSourceError,
    load_scenario_source,
    scenario_source_spec,
)
from reasonstyle.hashing import content_hash, sha256_of

ROOT = pathlib.Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "configs" / "experiment.yaml"
V1_OPENAI_CONFIG = ROOT / "configs" / "experiment_openai_pilot.yaml"
V2_CONFIG = ROOT / "configs" / "experiment_v2_pilot.yaml"
PILOT = ROOT / "scripts" / "pilot.py"
TOPICS = ROOT / "data" / "topics" / "pilot_topics.yaml"

NO_NETWORK = """
import socket, urllib.request
def _blocked(*args, **kwargs):
    raise AssertionError("the script attempted to contact a backend")
socket.socket = _blocked
socket.create_connection = _blocked
urllib.request.urlopen = _blocked
"""


@pytest.fixture(scope="module")
def v2():
    return load_config(V2_CONFIG)


@pytest.fixture
def topics(pilot_bank):
    return [t for t in pilot_bank.topics if t.status == "curated"]


@pytest.fixture
def bank_hash(pilot_bank):
    return content_hash(pilot_bank.model_dump(mode="json"))


@pytest.fixture
def no_network(tmp_path_factory):
    path = tmp_path_factory.mktemp("no_network_v2")
    (path / "sitecustomize.py").write_text(NO_NETWORK)
    return path


def _run_cli(config, *args, env=None, sitecustomize):
    environment = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    environment.update(env or {})
    environment["PYTHONPATH"] = f"{sitecustomize}:{os.environ.get('PYTHONPATH', '')}"
    return subprocess.run([sys.executable, str(PILOT), "--config", str(config), *args],
                          cwd=ROOT, capture_output=True, text=True, env=environment)


# --- a v2 allocation, built in memory; none is written to disk yet -----------


@pytest.fixture
def v2_allocation(pilot_bank, v2):
    """The four core markers over the 48 groups. In memory only.

    Decision 3 says the real `data/pilot/marker_allocation_v2.yaml` is created
    only once these fixes pass, so nothing here writes one.
    """
    families = {"therefore": "conclusion_indicator",
                "consequently": "conclusion_indicator",
                "it follows that": "metadiscursive_inference",
                "this implies": "metadiscursive_inference"}
    realizations = {"conclusion_indicator": "sentence_initial_conclusion_v1",
                    "metadiscursive_inference": "sentence_initial_metadiscursive_v1"}
    markers = list(families)
    groups = []
    decisions = sorted(t.decision_id for t in pilot_bank.topics if t.status == "curated")
    domains = {t.decision_id: t.domain for t in pilot_bank.topics}
    for index, decision in enumerate(decisions):
        for variant in (1, 2):
            marker = markers[(index * 2 + variant - 1) % len(markers)]
            family = families[marker]
            for option in ("opt_1", "opt_2"):
                groups.append(GroupAllocation(
                    decision_id=decision, domain=domains[decision], variant_id=variant,
                    scenario_id=f"{decision}_v{variant}", supported_option=option,
                    marker_family=family, marker_string=marker,
                    marker_realization_id=realizations[family]))
    return MarkerAllocation(groups=tuple(groups), seed=13,
                            config_content_hash=v2.content_hash,
                            topic_bank_content_hash=content_hash(
                                pilot_bank.model_dump(mode="json")))


class V2Responder:
    """Answers a v2 group call the way the design says the cells are built."""

    def __init__(self, *, break_cell: tuple[str, str] | None = None, mode: str = "premise"):
        self.requests = []
        self.break_cell, self.mode = break_cell, mode

    def __call__(self, request):
        self.requests.append(request)
        if request.kind == "scenario":
            return {"scenario_text": scenario_text(
                f"{request.decision_id}_v{request.variant_id}")}
        endorsement = request.context["endorsement"]
        marker = request.context["marker_string"]
        styled = marker[0].upper() + marker[1:]
        # Variant-specific, as a real premise is: it comes from the variant's own
        # scenario fact, so two variants of one decision never share an RS or RP.
        premise = (f"The {request.decision_id} record for variant {request.variant_id} "
                   f"states one relevant consideration for the route under discussion")
        capital = endorsement[0].upper() + endorsement[1:]
        bodies = {
            "RS": f"{premise}. {styled}, {endorsement}.",
            "RP": f"{premise}. {capital}.",
            "NS": f"{styled}, {endorsement}.",
            "NP": f"{capital}.",
        }
        key = (f"{request.decision_id}_v{request.variant_id}", request.supported_option)
        if self.break_cell and key == self.break_cell:
            if self.mode == "premise":          # a premise smuggled into NS
                bodies["NS"] = f"{styled}, {endorsement}, and the queue has grown."
            else:                               # the endorsement said twice
                bodies["NP"] = f"{capital}. {capital}."
        return bodies


def scenario_text(scenario_id: str) -> str:
    return (f"Record {scenario_id} sets out a regional capacity decision for the coming "
            f"winters. The operator can extend the existing baseload plant or accelerate "
            f"the storage build already under tender. Both routes are funded and feasible, "
            f"and nothing states how reliability should be weighed against emissions. The "
            f"extended plant can deliver full output through any cold spell. Retiring it on "
            f"schedule would cut power-sector emissions. One route must be chosen before "
            f"the tender closes.")


def _v2_run(tmp_path, v2, segmenter, topics, bank_hash, v2_allocation, responder=None):
    """A complete v2 group run against synthetic scenarios, offline."""
    store = CallStore(tmp_path / "run_v2", v2, topic_bank_content_hash=bank_hash,
                      allocation_content_hash=v2_allocation.content_hash)
    run_scenario_stage(topics, v2, segmenter, FakeBackend(V2Responder()), store)
    approvals = {
        scenario_id: ScenarioApproval(
            scenario_id=scenario_id, scenario_text_sha256=sha256_of(record["scenario_text"]),
            call_id=record["call_id"], config_content_hash=v2.content_hash,
            topic_bank_content_hash=bank_hash, decision="approved",
            judgements={name: True for name in REQUIRED_JUDGEMENTS},
            decided_by="Vidhi Bhutani", decided_at=date(2026, 9, 18))
        for scenario_id, record in recorded_scenarios(store).items()}
    responder = responder or V2Responder()
    results = run_group_stage(topics, v2_allocation.groups, v2, segmenter,
                              FakeBackend(responder), store, approvals=approvals,
                              topic_bank_content_hash=bank_hash)
    return store, approvals, results, responder


# --- the pipeline runs the v2 checks ----------------------------------------


def test_a_v2_group_drafted_through_the_pipeline_passes_every_v2_check(
        tmp_path, v2, segmenter, topics, bank_hash, v2_allocation):
    store, _, results, responder = _v2_run(tmp_path, v2, segmenter, topics, bank_hash,
                                           v2_allocation)
    assert len(results) == 48
    assert all(r.outcome == ACCEPTED for r in results), \
        sorted({c for r in results for a in r.attempts for c in a.error_codes})
    assert len([r for r in responder.requests if r.kind == "group"]) == 48
    assert all(r.template_name == "group_draft_v2"
               for r in responder.requests if r.kind == "group")


@pytest.mark.parametrize("mode,expected", [
    ("premise", "E_NO_PREMISE_CELL_HAS_EXTRA_CONTENT"),
    ("endorsement", "E_ENDORSEMENT_NOT_EXACTLY_ONCE"),
])
def test_the_pipeline_catches_a_v2_violation_rather_than_accepting_it(
        mode, expected, tmp_path, v2, segmenter, topics, bank_hash, v2_allocation):
    broken = ("climate_01_v1", "opt_1")
    _, _, results, _ = _v2_run(tmp_path, v2, segmenter, topics, bank_hash, v2_allocation,
                               responder=V2Responder(break_cell=broken, mode=mode))
    failed = [r for r in results if r.outcome != ACCEPTED]
    assert len(failed) == 1 and (failed[0].decision_id, failed[0].supported_option) == \
        ("climate_01", "opt_1")
    assert expected in {c for a in failed[0].attempts for c in a.error_codes}


def test_assembly_and_corpus_validation_run_the_same_v2_checks(
        tmp_path, v2, segmenter, topics, bank_hash, v2_allocation):
    store, approvals, _, _ = _v2_run(tmp_path, v2, segmenter, topics, bank_hash,
                                     v2_allocation)
    records, manifest = assemble_pilot(
        topics=topics, allocation_groups=v2_allocation.groups, approvals=approvals,
        corrections=[], scenarios=recorded_scenarios(store), groups=recorded_groups(store),
        cfg=v2, segmenter=segmenter, topic_bank_content_hash=bank_hash)
    assert len(records) == 24 and manifest["machine_errors"] == 0

    report = validate_corpus(records, v2, segmenter, corpus_scope="pilot")
    assert report.ok, [str(f) for f in report.errors]
    # The v2 codes ran: the info findings only exist in pairwise mode.
    assert any(f.code == "I_CROSS_PAIR_WORD_RATIO" for f in report.info)
    assert any(f.code == "I_PAIR_WORD_DELTA_BODY" for f in report.info)
    assert report.requires_human_review


def test_assembly_refuses_a_v2_group_that_breaks_a_v2_rule(
        tmp_path, v2, segmenter, topics, bank_hash, v2_allocation):
    store, approvals, _, _ = _v2_run(tmp_path, v2, segmenter, topics, bank_hash,
                                     v2_allocation,
                                     responder=V2Responder(break_cell=("climate_01_v1",
                                                                       "opt_1")))
    with pytest.raises(AssemblyRefused):
        assemble_pilot(
            topics=topics, allocation_groups=v2_allocation.groups, approvals=approvals,
            corrections=[], scenarios=recorded_scenarios(store),
            groups=recorded_groups(store), cfg=v2, segmenter=segmenter,
            topic_bank_content_hash=bank_hash)


def test_omitting_the_endorsement_blocks_a_v2_group(v2, segmenter):
    """A pairwise design without its endorsement is unchecked, not clean."""
    from reasonstyle.corpus.validate import validate_group
    from reasonstyle.generation.pipeline import _block_from
    allocation = GroupAllocation(
        decision_id="climate_01", domain="climate", variant_id=1,
        scenario_id="climate_01_v1", supported_option="opt_1",
        marker_family="conclusion_indicator", marker_string="therefore",
        marker_realization_id="sentence_initial_conclusion_v1")
    bodies = {c: "Therefore, I support the option to do the thing." for c in
              ("RS", "RP", "NS", "NP")}
    findings = validate_group("A scenario.", v2.raw["corpus"]["counterargument_opening"],
                              _block_from(bodies, allocation), v2, segmenter,
                              loc={"scenario_id": "climate_01_v1"}, endorsement=None)
    codes = {f.code for f in findings if f.severity == "error"}
    assert "E_ENDORSEMENT_NOT_SUPPLIED" in codes
    assert "I_ENDORSEMENT_NOT_SUPPLIED" not in {f.code for f in findings}


def test_the_group_review_holds_a_v2_run_to_the_v2_rules(
        tmp_path, v2, segmenter, topics, bank_hash, v2_allocation):
    from reasonstyle.generation.group_review import build_group_review
    store, approvals, _, _ = _v2_run(tmp_path, v2, segmenter, topics, bank_hash,
                                     v2_allocation)
    export = build_group_review(
        topics=topics, allocation_groups=v2_allocation.groups,
        scenarios=recorded_scenarios(store), store=store, cfg=v2, segmenter=segmenter,
        approvals=approvals, topic_bank_content_hash=bank_hash,
        allocation_content_hash=v2_allocation.content_hash)
    assert len(export.machine_valid) == 48 and len(export.needing_correction) == 0
    assert all(g["recorded_codes_agree"] for g in export.manifest["groups"]), \
        "the review re-derives the same codes the run recorded, under v2 rules"


# --- path isolation -----------------------------------------------------------


def test_v2_defaults_can_never_write_to_a_v1_path(v2):
    import importlib.util
    spec = importlib.util.spec_from_file_location("pilot_v2", PILOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v2_paths = module.profile_paths(v2)
    v1_paths = module.profile_paths(load_config(V1_CONFIG))
    hosted_paths = module.profile_paths(load_config(V1_OPENAI_CONFIG))
    for name, value in v2_paths.items():
        assert value not in (v1_paths[name], hosted_paths[name]), name
        assert "_v2" in value, f"{name}={value} does not name itself as v2"
    assert v2_paths["out"] == "data/pilot/run_v2"
    assert v2_paths["allocation"] == "data/pilot/marker_allocation_v2.yaml"
    assert v2_paths["corrections_file"] == "data/pilot/manual_corrections_v2.yaml"


@pytest.mark.parametrize("foreign", ["data/pilot/run", "data/pilot/run_openai"])
def test_a_v2_stage_refuses_a_v1_run_directory(foreign, v2):
    import importlib.util
    spec = importlib.util.spec_from_file_location("pilot_v2b", PILOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    store = CallStore(ROOT / foreign, v2)
    problems = module.run_directory_problems(store, v2)
    assert problems and "data/pilot/run_v2" in problems[0]
    assert "no other design writes into it" in problems[0]


def test_the_v2_command_line_reports_its_own_paths(tmp_path, no_network):
    result = _run_cli(V2_CONFIG, "groups", "--allocation",
                      "data/pilot/marker_allocation.yaml", env={},
                      sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "out          data/pilot/run_v2" in result.stdout
    assert "data/pilot/run_openai" not in result.stdout
    assert "attempted to contact a backend" not in result.stderr


def test_the_two_allocations_are_separate_files():
    """v2 has its own allocation and never writes over v1's."""
    assert (ROOT / "data" / "pilot" / "marker_allocation.yaml").is_file()
    v2_path = ROOT / "data" / "pilot" / "marker_allocation_v2.yaml"
    if v2_path.is_file():
        assert v2_path.read_bytes() != (
            ROOT / "data" / "pilot" / "marker_allocation.yaml").read_bytes()


# --- the scenario source ------------------------------------------------------


def test_the_v1_approvals_are_verified_against_the_v1_source_config(v2):
    source = load_scenario_source(v2, topics_path=TOPICS, root=ROOT)
    assert source.count == 24
    v1_hosted = load_config(V1_OPENAI_CONFIG)
    assert source.provenance["config_content_hash"] == v1_hosted.content_hash
    assert source.provenance["config_content_hash"] != v2.content_hash, \
        "the approvals keep the hash they were granted under"
    assert source.provenance["config_version"] == "openai_pilot_v1"
    assert set(source.provenance["verified"]) == {
        "config_content_hash", "call_id", "scenario_text_sha256", "topic_bank_content_hash"}
    assert source.provenance["access"] == "read_only"


def test_the_four_redrafted_scenarios_come_through_with_what_they_superseded(v2):
    source = load_scenario_source(v2, topics_path=TOPICS, root=ROOT)
    superseded = {k for k, v in source.provenance["scenarios"].items()
                  if v["superseded_call_id"]}
    assert superseded == {"climate_04_v2", "energy_01_v2", "energy_03_v2",
                          "technology_01_v1"}


def test_a_source_whose_hash_no_longer_verifies_is_refused(v2, tmp_path):
    """The check that makes reuse safe rather than merely convenient."""
    import copy
    from reasonstyle.config import ExperimentConfig, RawConfig
    raw = copy.deepcopy(v2.raw)
    raw["scenario_source"]["approvals"] = str(tmp_path / "stale.yaml")
    stale = yaml.safe_load((ROOT / "data" / "pilot"
                            / "scenario_approvals_openai.yaml").read_text())
    for record in stale.values():
        record["config_content_hash"] = "0" * 64
    (tmp_path / "stale.yaml").write_text(yaml.safe_dump(stale))
    cfg = ExperimentConfig(raw=raw, parsed=RawConfig.model_validate(raw), path=None)
    with pytest.raises(ScenarioSourceError, match="approved under configuration"):
        load_scenario_source(cfg, topics_path=TOPICS, root=ROOT)


def test_a_source_marked_writable_is_refused(v2):
    import copy
    from reasonstyle.config import ExperimentConfig, RawConfig
    raw = copy.deepcopy(v2.raw)
    raw["scenario_source"]["access"] = "read_write"
    cfg = ExperimentConfig(raw=raw, parsed=RawConfig.model_validate(raw), path=None)
    with pytest.raises(ScenarioSourceError, match="read_only"):
        load_scenario_source(cfg, topics_path=TOPICS, root=ROOT)


def test_the_v1_source_run_is_never_copied_or_written(v2, tmp_path):
    import hashlib
    v2_before = _run_v2_snapshot()
    run = ROOT / "data" / "pilot" / "run_openai"
    if not run.is_dir():
        pytest.skip("the hosted run is not on this machine (gitignored run artefacts)")
    before = {p.relative_to(run): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(run.rglob("*")) if p.is_file()}
    load_scenario_source(v2, topics_path=TOPICS, root=ROOT)
    after = {p.relative_to(run): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(run.rglob("*")) if p.is_file()}
    assert before == after and before
    assert _run_v2_snapshot() == v2_before, "nothing was written into the v2 run"


def test_a_v2_call_records_both_provenance_layers(tmp_path, v2, segmenter, topics,
                                                  bank_hash, v2_allocation,
                                                  real_source_run):
    store, source, _, _ = real_source_run
    entry = next(e for e in store.log.entries() if e["kind"] == "group")
    assert entry["config_content_hash"] == v2.content_hash, "the v2 group design"
    recorded = entry["extra"]["scenario_source"]
    assert recorded["config_content_hash"] == load_config(V1_OPENAI_CONFIG).content_hash
    assert recorded["config_content_hash"] != v2.content_hash
    assert recorded["run"] == "data/pilot/run_openai"
    assert recorded["access"] == "read_only"
    assert recorded["scenario_count"] == 24
    assert entry["template_name"] == "group_draft_v2"


def test_v2_logs_record_the_v2_template_names(tmp_path, v2, segmenter, topics, bank_hash,
                                              v2_allocation):
    store, _, _, _ = _v2_run(tmp_path, v2, segmenter, topics, bank_hash, v2_allocation)
    names = {e["template_name"] for e in store.log.entries() if e["kind"] == "group"}
    assert names == {"group_draft_v2"}
    v1_names = {e["template_name"] for e in store.log.entries() if e["kind"] == "scenario"}
    assert v1_names == {"scenario_draft_v1"}, "the scenario template did not change"


# --- duplicates ----------------------------------------------------------------


def test_keying_on_the_whole_input_removes_the_baseline_collision_entirely(
        tmp_path, v2, segmenter, topics, bank_hash, v2_allocation):
    """The preferable half of decision 2, and it turns out to be sufficient.

    Two variants of a decision share their options, so their NS and NP bodies
    are identical — but they do NOT share a scenario, and the model is given
    both. Keyed on the complete rendered input, they are different inputs and
    no duplicate arises at all. The narrow exception below is the fallback for
    a design that keys on the counterargument alone.
    """
    store, approvals, _, _ = _v2_run(tmp_path, v2, segmenter, topics, bank_hash,
                                     v2_allocation)
    records, _ = assemble_pilot(
        topics=topics, allocation_groups=v2_allocation.groups, approvals=approvals,
        corrections=[], scenarios=recorded_scenarios(store), groups=recorded_groups(store),
        cfg=v2, segmenter=segmenter, topic_bank_content_hash=bank_hash)
    report = validate_corpus(records, v2, segmenter, corpus_scope="pilot")
    assert report.ok, [str(f) for f in report.errors]
    assert not [f for f in report.errors if f.code == "E_DUPLICATE_TEXT"]
    bodies = [block.cells[c].body for r in records
              for block in r.counterarguments.values() for c in ("NS", "NP")]
    assert len(bodies) != len(set(bodies)), \
        "the baselines really do repeat; it is the scenario that distinguishes them"


def test_expected_cross_variant_baselines_are_informational(
        tmp_path, v2, segmenter, topics, bank_hash, v2_allocation):
    """The fallback: keyed on the counterargument alone, the repetition appears
    and is recorded as expected rather than failed."""
    import copy
    from reasonstyle.config import ExperimentConfig, RawConfig
    raw = copy.deepcopy(v2.raw)
    raw["corpus"]["duplicate_text"]["compare"] = "rendered_counterargument"
    narrow = ExperimentConfig(raw=raw, parsed=RawConfig.model_validate(raw), path=None)

    store, approvals, _, _ = _v2_run(tmp_path, v2, segmenter, topics, bank_hash,
                                     v2_allocation)
    records, _ = assemble_pilot(
        topics=topics, allocation_groups=v2_allocation.groups, approvals=approvals,
        corrections=[], scenarios=recorded_scenarios(store), groups=recorded_groups(store),
        cfg=v2, segmenter=segmenter, topic_bank_content_hash=bank_hash)
    report = validate_corpus(records, narrow, segmenter, corpus_scope="pilot")
    expected = [f for f in report.info if f.code == "I_EXPECTED_BASELINE_REPETITION"]
    assert expected, "the baselines repeat, and the repetition is recorded"
    assert all(f.condition in ("NS", "NP") for f in expected)
    assert not [f for f in report.errors if f.code == "E_DUPLICATE_TEXT"], \
        "an expected baseline is never an error"


def test_an_unrelated_duplicate_is_still_an_error(v2, segmenter):
    from reasonstyle.corpus.validate import validate_group
    from reasonstyle.generation.pipeline import _block_from
    allocation = GroupAllocation(
        decision_id="climate_01", domain="climate", variant_id=1,
        scenario_id="climate_01_v1", supported_option="opt_1",
        marker_family="conclusion_indicator", marker_string="therefore",
        marker_realization_id="sentence_initial_conclusion_v1")
    endorsement = endorsement_text("Keep the limit on its current schedule.", v2)
    capital = endorsement[0].upper() + endorsement[1:]
    bodies = {"RS": f"A premise holds. Therefore, {endorsement}.",
              "RP": f"A premise holds. {capital}.",
              "NS": f"Therefore, {endorsement}.",
              "NP": f"{capital}."}
    seen = {}
    scenario = "A scenario sentence. Another one."
    opening = v2.raw["corpus"]["counterargument_opening"]
    # A DIFFERENT decision produced the identical RS text: a real collision.
    first = validate_group(scenario, opening, _block_from(bodies, allocation), v2,
                           segmenter, loc={"scenario_id": "climate_01_v1"},
                           seen_texts=seen, endorsement=endorsement)
    assert not [f for f in first if f.code == "E_DUPLICATE_TEXT"]
    other = GroupAllocation(**{**allocation.as_dict(), "decision_id": "energy_02",
                               "scenario_id": "energy_02_v1", "domain": "energy"})
    second = validate_group(scenario, opening, _block_from(bodies, other), v2, segmenter,
                            loc={"scenario_id": "energy_02_v1"}, seen_texts=seen,
                            endorsement=endorsement)
    duplicates = {f.condition for f in second if f.code == "E_DUPLICATE_TEXT"}
    assert duplicates == {"RS", "RP", "NS", "NP"}, \
        "a different decision repeating a whole group is a collision, not a baseline"


def test_the_duplicate_key_is_the_whole_experimental_input(v2, segmenter):
    """Two cells that read the same after different scenarios are different
    inputs to the model, and the design keys on what the model actually sees."""
    assert v2.raw["corpus"]["duplicate_text"]["compare"] == "rendered_experimental_input"
    from reasonstyle.corpus.validate import _duplicate_key
    opening = v2.raw["corpus"]["counterargument_opening"]
    a = _duplicate_key(v2, "Scenario one.", opening, "Same body.")
    b = _duplicate_key(v2, "Scenario two.", opening, "Same body.")
    assert a != b
    assert _duplicate_key(load_config(V1_CONFIG), "Scenario one.", opening, "Same body.") == \
        _duplicate_key(load_config(V1_CONFIG), "Scenario two.", opening, "Same body."), \
        "v1 keyed on the rendered counterargument alone, and still does"


# --- premise indicators are unreachable from the v2 core ----------------------


def test_premise_indicators_cannot_be_selected_by_a_v2_allocation(v2):
    alloc = v2.raw["markers"]["allocation"]
    assert "premise_indicator" not in alloc["pilot_families"]
    assert "premise_indicator" not in alloc["selectable_families"]
    assert "premise_indicator" in alloc["refused_families"]
    assert "premise_indicator" not in v2.raw["markers"]["primary_families"]
    assert "premise_indicator" in v2.raw["markers"]["deferred_families"]
    note = v2.raw["markers"]["deferred_families"]["premise_indicator"]
    assert "NOT usable in the v2 four-cell corpus" in note
    assert "its own template, validator and analysis" in note


def test_held_out_family_generalisation_cannot_reach_a_deferred_family(v2):
    confirmatory = set(v2.raw["markers"]["roles"]["confirmatory"])
    assert confirmatory == {"conclusion_indicator", "metadiscursive_inference"}
    assert "premise_indicator" not in confirmatory
    assert set(v2.raw["markers"]["allocation"]["pilot_strings"]) == confirmatory
    strings = {s for v in v2.raw["markers"]["allocation"]["pilot_strings"].values()
               for s in v}
    assert strings == {"therefore", "consequently", "it follows that", "this implies"}


# --- the real v1 source, end to end, offline ---------------------------------


@pytest.fixture
def real_source_run(tmp_path, v2, segmenter, topics, bank_hash, v2_allocation):
    """v2 groups drafted from the ACTUAL approved v1 hosted scenarios, offline.

    No scenario is drafted here and none could be: the configuration declares a
    source, the source is verified, and the scenario stage refuses outright.
    The generator is ``FakeBackend``; nothing is sent and nothing is charged.
    """
    if not (ROOT / "data" / "pilot" / "run_openai").is_dir():
        pytest.skip("the hosted v1 run is not on this machine (gitignored run artefacts)")
    source = load_scenario_source(v2, topics_path=TOPICS, root=ROOT)
    store = CallStore(tmp_path / "run_v2", v2, topic_bank_content_hash=bank_hash,
                      allocation_content_hash=v2_allocation.content_hash,
                      scenario_source=source.provenance)
    responder = V2Responder()
    results = run_group_stage(
        topics, v2_allocation.groups, v2, segmenter, FakeBackend(responder), store,
        approvals=source.approvals, topic_bank_content_hash=bank_hash,
        scenarios=source.scenarios, gate_verified_elsewhere=True)
    return store, source, results, responder


def test_the_real_v1_scenarios_drive_a_complete_offline_v2_group_run(real_source_run, v2,
                                                                     segmenter, topics,
                                                                     bank_hash,
                                                                     v2_allocation):
    store, source, results, responder = real_source_run
    assert len(results) == 48 and all(r.outcome == ACCEPTED for r in results), \
        sorted({c for r in results for a in r.attempts for c in a.error_codes})
    assert source.count == 24

    # No v2 scenario call exists, and none can: the run holds groups only.
    kinds = {e["kind"] for e in store.log.entries()}
    assert kinds == {"group"}, kinds
    assert not any(r.kind in ("scenario", "scenario_redraft") for r in responder.requests)

    # status, over the same gate the generation used
    from reasonstyle.generation.approvals import approval_status
    approved = sum(
        1 for scenario_id, record in source.scenarios.items()
        if approval_status(scenario_id, record["scenario_text"], record["call_id"],
                           source.approvals,
                           config_content_hash=source.config_content_hash,
                           topic_bank_content_hash=bank_hash)[0] == "approved")
    assert approved == 24, "the source gate is satisfied under the SOURCE's hash"

    # group review
    from reasonstyle.generation.group_review import build_group_review
    export = build_group_review(
        topics=topics, allocation_groups=v2_allocation.groups, scenarios=source.scenarios,
        store=store, cfg=v2, segmenter=segmenter, approvals=source.approvals,
        topic_bank_content_hash=bank_hash,
        allocation_content_hash=v2_allocation.content_hash,
        approval_config_content_hash=source.config_content_hash,
        scenario_source=source.provenance)
    assert len(export.machine_valid) == 48 and len(export.needing_correction) == 0
    assert all(g["scenario_approval_state"] == "approved" for g in export.manifest["groups"])
    assert export.manifest["approval_config_content_hash"] == source.config_content_hash
    assert export.manifest["scenario_source"]["run"] == "data/pilot/run_openai"

    # assembly, on the verified source approvals
    records, manifest = assemble_pilot(
        topics=topics, allocation_groups=v2_allocation.groups,
        approvals=source.approvals,
        approval_config_content_hash=source.config_content_hash,
        scenario_source=source.provenance, corrections=[], scenarios=source.scenarios,
        groups=recorded_groups(store), cfg=v2, segmenter=segmenter,
        topic_bank_content_hash=bank_hash)
    assert len(records) == 24
    assert manifest["n_texts"] == 192 and manifest["machine_errors"] == 0

    # both provenance layers, in the manifest
    assert manifest["config_content_hash"] == v2.content_hash, "the v2 group design"
    assert manifest["approval_config_content_hash"] == source.config_content_hash
    assert manifest["scenario_source"]["config"] == "configs/experiment_openai_pilot.yaml"
    assert manifest["scenario_source"]["access"] == "read_only"
    assert manifest["scenario_source"]["expected_scenario_count"] == 24

    # and the assembled corpus validates under the v2 rules
    report = validate_corpus(records, v2, segmenter, corpus_scope="pilot")
    assert report.ok, [str(f) for f in report.errors]
    assert any(f.code == "I_CROSS_PAIR_WORD_RATIO" for f in report.info)
    assert report.requires_human_review


def test_the_real_source_run_directories_are_never_written(real_source_run):
    import hashlib
    # The fixture drove a complete group run into tmp_path. The repository's own
    # run directory belongs to the live v2 stage and this test never touches it,
    # which is what the store path below establishes.
    assert real_source_run[0].directory != ROOT / "data" / "pilot" / "run_v2"
    for name in ("run_openai", "run"):
        run = ROOT / "data" / "pilot" / name
        if not run.is_dir():
            continue
        digest = hashlib.sha256()
        for path in sorted(run.rglob("*")):
            if path.is_file():
                digest.update(path.read_bytes())
        assert digest.hexdigest(), f"{name} was read"


# --- source completeness -------------------------------------------------------


def _source_with(tmp_path, v2, mutate):
    """A v2 config whose source approvals have been mutated for the test."""
    import copy
    from reasonstyle.config import ExperimentConfig, RawConfig
    approvals = yaml.safe_load(
        (ROOT / "data" / "pilot" / "scenario_approvals_openai.yaml").read_text())
    mutate(approvals)
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(approvals))
    raw = copy.deepcopy(v2.raw)
    raw["scenario_source"]["approvals"] = str(path)
    return ExperimentConfig(raw=raw, parsed=RawConfig.model_validate(raw), path=None)


def test_a_source_missing_one_expected_scenario_is_refused(tmp_path, v2):
    cfg = _source_with(tmp_path, v2, lambda a: a.pop("energy_03_v1"))
    with pytest.raises(ScenarioSourceError, match="missing 1 expected scenario"):
        load_scenario_source(cfg, topics_path=TOPICS, root=ROOT)


def test_a_source_carrying_one_extra_scenario_is_refused(tmp_path, v2):
    def add_extra(approvals):
        approvals["energy_99_v1"] = dict(approvals["energy_03_v1"])
    cfg = _source_with(tmp_path, v2, add_extra)
    with pytest.raises(ScenarioSourceError, match="does not expect"):
        load_scenario_source(cfg, topics_path=TOPICS, root=ROOT)


def test_a_source_with_one_unapproved_scenario_is_refused(tmp_path, v2):
    def unapprove(approvals):
        approvals["climate_02_v1"]["decision"] = "redraft"
        approvals["climate_02_v1"]["reason"] = "not approved after all"
    cfg = _source_with(tmp_path, v2, unapprove)
    with pytest.raises(ScenarioSourceError, match="not 'approved'"):
        load_scenario_source(cfg, topics_path=TOPICS, root=ROOT)


def test_a_source_with_one_broken_text_binding_is_refused(tmp_path, v2):
    def rebind(approvals):
        approvals["technology_02_v2"]["scenario_text_sha256"] = "0" * 64
    cfg = _source_with(tmp_path, v2, rebind)
    with pytest.raises(ScenarioSourceError, match="text has changed"):
        load_scenario_source(cfg, topics_path=TOPICS, root=ROOT)


def test_the_expected_set_is_derived_from_the_topic_bank(v2, pilot_bank):
    from reasonstyle.generation.scenario_source import expected_scenario_ids
    expected = expected_scenario_ids(pilot_bank, v2)
    assert len(expected) == 24
    curated = {t.decision_id for t in pilot_bank.topics if t.status == "curated"}
    assert expected == {f"{d}_v{v}" for d in curated for v in (1, 2)}


# --- a source-declaring design drafts no scenario ------------------------------


@pytest.mark.parametrize("command", ["scenarios", "redraft-scenarios"])
def test_a_source_declaring_design_refuses_to_draft_scenarios(command, tmp_path,
                                                              no_network):
    before = _run_v2_snapshot()
    result = _run_cli(V2_CONFIG, command, "--allocation",
                      "data/pilot/marker_allocation.yaml", "--send",
                      env={"REASONSTYLE_ALLOW_OPENAI_GENERATION": "1",
                           "REASONSTYLE_ALLOW_PILOT_GENERATION": "1",
                           "OPENAI_API_KEY": "sk-" + "t" * 40},
                      sitecustomize=no_network)
    assert result.returncode == 1
    assert "reuses the approved scenarios" in result.stderr
    assert "no credential was read and no backend was built" in result.stderr
    assert "attempted to contact a backend" not in result.stderr
    assert _run_v2_snapshot() == before, "the refusal wrote nothing into the v2 run"


def test_scenario_review_writes_no_v2_approval_template(tmp_path, no_network):
    out = tmp_path / "review"
    result = _run_cli(V2_CONFIG, "scenario-review", "--allocation",
                      "data/pilot/marker_allocation.yaml", "--write-template",
                      "--review-out", str(out), env={}, sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert not (out / "scenario_approvals.template.yaml").exists()
    assert "no approval template was written" in result.stderr
    page = (out / "scenarios.md").read_text()
    assert "24 approved" in result.stdout
    assert "gate state: **approved**" in page


# --- the prompts say the right thing -------------------------------------------


@pytest.mark.parametrize("prompt", ["group_draft_v2.txt", "repair_v2.txt"])
def test_the_v2_prompts_carry_no_percentage_target(prompt):
    text = (ROOT / "prompts" / prompt).read_text()
    for banned in ("10 percent", "10%", "ten percent", "within 10"):
        assert banned not in text.lower(), f"{prompt} still sets a percentage target"
    assert "percentage target" in text, "and says so explicitly"


@pytest.mark.parametrize("prompt", ["group_draft_v2.txt", "repair_v2.txt"])
def test_the_v2_prompts_state_the_exact_marker_word_budget(prompt):
    text = " ".join((ROOT / "prompts" / prompt).read_text().split())
    assert "exactly the number of words in \"${marker_string}\"" in text
    assert "recorded descriptively" in text
    assert "NEVER" in text, "and is never something to repair"


def test_the_group_prompt_drops_the_reconstruction_requirement():
    text = (ROOT / "prompts" / "group_draft_v2.txt").read_text()
    assert "reconstruct any consideration" not in text
    assert "must not be able to reconstruct" not in text
    assert "state no explicit task-relevant premise" in text
    assert "part of what this experiment manipulates and\nmeasures" in text


def test_the_repair_prompt_uses_the_same_rule_as_the_draft_prompt():
    repair = (ROOT / "prompts" / "repair_v2.txt").read_text()
    assert "no explicit task-relevant premise" in repair
    assert "exactly the number of words in" in repair
    assert "NEVER something to repair" in repair
    assert "do not trim RS or RP" in repair


# --- the one-call smoke, for both designs --------------------------------------

SMOKE = ROOT / "scripts" / "openai_smoke_test.py"


def _run_v2_snapshot() -> dict:
    """What the v2 run directory holds right now, if it holds anything.

    These tests used to assert the directory did not exist, which was true only
    until the v2 group stage was run. The property that actually matters is
    unchanged either way: the operation under test writes nothing into it.
    """
    import hashlib
    run = ROOT / "data" / "pilot" / "run_v2"
    return {p.relative_to(run): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(run.rglob("*")) if p.is_file()} if run.is_dir() else {}


def _run_smoke(config, *args, env=None, sitecustomize):
    environment = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    environment.update(env or {})
    environment["PYTHONPATH"] = f"{sitecustomize}:{os.environ.get('PYTHONPATH', '')}"
    return subprocess.run([sys.executable, str(SMOKE), "--config", str(config), *args],
                          cwd=ROOT, capture_output=True, text=True, env=environment)


def test_the_v1_smoke_is_unchanged(tmp_path, no_network):
    """Same material, same marker, same prompt, same call id as it always had."""
    result = _run_smoke(V1_OPENAI_CONFIG, "--out", str(tmp_path / "smoke"), env={},
                        sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "SYNTHETIC energy_fixture_001 v1 / opt_1 (premise_indicator, 'because')" \
        in result.stdout
    assert "call_id      05a54889c0b4f92ae28a6740972e19b9e4f925ed99177fcaf23d8374f3e4b366" \
        in result.stdout
    assert "prompt hash  14df854e5228c780" in result.stdout
    assert "design       pairwise" not in result.stdout
    assert "endorsement" not in result.stdout
    assert "DRY RUN: no key was read" in result.stdout
    assert "attempted to contact a backend" not in result.stderr


def test_the_v2_smoke_uses_a_permitted_marker_not_the_fixture_because(tmp_path, no_network):
    result = _run_smoke(V2_CONFIG, "--out", str(tmp_path / "smoke"), env={},
                        sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "(conclusion_indicator, 'therefore')" in result.stdout
    assert "premise_indicator" not in result.stdout
    assert "'because'" not in result.stdout


def test_the_v2_smoke_uses_the_v2_prompt_and_endorsement(tmp_path, no_network):
    out = tmp_path / "smoke"
    result = _run_smoke(V2_CONFIG, "--out", str(out), env={}, sitecustomize=no_network)
    assert "design       pairwise (v2) · template group_draft_v2" in result.stdout
    assert "I support the option to extend the operating life" in result.stdout
    request = json.loads(next((out / "requests").glob("*_group.json")).read_text())
    assert request["template_name"] == "group_draft_v2"
    assert request["context"]["marker_string"] == "therefore"
    assert request["context"]["endorsement"].startswith("I support the option to ")
    assert "THE FIXED ENDORSEMENT" in request["prompt"]
    assert "10 percent" not in request["prompt"]
    assert "exactly the number of words in" in " ".join(request["prompt"].split())


def test_the_v2_smoke_defaults_to_its_own_directory(tmp_path, no_network):
    import importlib.util
    spec = importlib.util.spec_from_file_location("smoke_mod", SMOKE)
    source = SMOKE.read_text()
    assert '"data/pilot/smoke_v2" if matching_mode(cfg) == PAIRWISE' in source
    assert '"data/pilot/smoke_openai"' in source
    del spec


@pytest.mark.parametrize("config", [V1_OPENAI_CONFIG, V2_CONFIG])
@pytest.mark.parametrize("reserved", ["data/pilot/run", "data/pilot/run_openai",
                                      "data/pilot/run_v2"])
def test_the_smoke_refuses_every_pilot_run_directory(config, reserved, no_network):
    result = _run_smoke(config, "--out", reserved, env={}, sitecustomize=no_network)
    assert result.returncode == 1
    assert "is a pilot run directory" in result.stderr
    assert "not corpus evidence" in result.stderr


@pytest.mark.parametrize("config", [V1_OPENAI_CONFIG, V2_CONFIG])
@pytest.mark.parametrize("env", [{}, {"REASONSTYLE_ALLOW_OPENAI_GENERATION": "1"},
                                 {"OPENAI_API_KEY": "sk-" + "t" * 40}])
def test_the_smoke_sends_nothing_without_send_and_the_authorisation(config, env, tmp_path,
                                                                    no_network):
    result = _run_smoke(config, "--out", str(tmp_path / "smoke"), *( ["--send"] if env.get(
        "REASONSTYLE_ALLOW_OPENAI_GENERATION") is None else []), env=env,
        sitecustomize=no_network)
    assert "attempted to contact a backend" not in result.stderr
    assert not (tmp_path / "smoke" / "generation_log.jsonl").exists()


def test_the_smoke_keeps_its_one_call_ceiling_for_both_designs(tmp_path, no_network):
    for config in (V1_OPENAI_CONFIG, V2_CONFIG):
        result = _run_smoke(config, "--out", str(tmp_path / config.stem), env={},
                            sitecustomize=no_network)
        assert "ceiling      1 call. No repair, no retry, no continuation into the pilot." \
            in result.stdout, config.name
    source = SMOKE.read_text()
    assert "REASONSTYLE_ALLOW_PILOT_GENERATION" not in source
    assert "run_group_stage" not in source and "repair_request" not in source


def test_the_v2_smoke_validates_only_its_own_synthetic_group():
    """The rest of the fixture corpus was written to the v1 rules; validating it
    under v2 would report failures that say nothing about this call."""
    source = SMOKE.read_text()
    assert "Only the synthetic group this call produced" in source
    assert "validate_group(" in source


# --- the v2 allocation ----------------------------------------------------------


V2_ALLOCATION = ROOT / "data" / "pilot" / "marker_allocation_v2.yaml"


@pytest.mark.skipif(not V2_ALLOCATION.is_file(), reason="the v2 allocation is not built")
def test_the_v2_allocation_is_balanced_and_uses_only_permitted_markers(v2, pilot_bank):
    import collections
    from reasonstyle.generation import allocate_markers, allocation_problems
    alloc = load_allocation(V2_ALLOCATION)
    assert len(alloc.groups) == 48
    assert allocation_problems(alloc, v2) == []
    assert allocate_markers(pilot_bank, v2).content_hash == alloc.content_hash

    markers = collections.Counter(g.marker_string for g in alloc.groups)
    assert dict(markers) == {"therefore": 12, "consequently": 12,
                             "it follows that": 12, "this implies": 12}
    families = collections.Counter(g.marker_family for g in alloc.groups)
    assert dict(families) == {"conclusion_indicator": 24, "metadiscursive_inference": 24}
    realizations = {g.marker_realization_id for g in alloc.groups}
    assert realizations == {"sentence_initial_conclusion_v1",
                            "sentence_initial_metadiscursive_v1"}
    banned = {"because", "given that", "considering that", "however", "even so",
              "nevertheless", "despite this"}
    assert not (set(markers) & banned)

    for marker in markers:
        here = [g for g in alloc.groups if g.marker_string == marker]
        assert collections.Counter(g.domain for g in here) == \
            collections.Counter({"climate": 4, "energy": 4, "technology": 4})
        assert collections.Counter(g.supported_option for g in here) == \
            collections.Counter({"opt_1": 6, "opt_2": 6})
    for family in families:
        by_variant = collections.Counter(g.variant_id for g in alloc.groups
                                         if g.marker_family == family)
        assert by_variant == collections.Counter({1: 12, 2: 12}), \
            f"{family} must not be confounded with variant"


@pytest.mark.skipif(not V2_ALLOCATION.is_file(), reason="the v2 allocation is not built")
def test_the_v1_allocation_is_untouched_by_the_v2_one():
    v1 = load_allocation(ROOT / "data" / "pilot" / "marker_allocation.yaml")
    assert v1.content_hash == "60621c17186e25cc37d4ced304086ac58599c5a720fc3c922cda6d553f693ecd"
    assert v1.config_content_hash == \
        "9da99ff12674cc91f0bbf76e137bf1cb347c30eb49691cfcfe3e4914c2fa148f"
    v2_alloc = load_allocation(V2_ALLOCATION)
    assert v2_alloc.content_hash != v1.content_hash
    assert {g.marker_string for g in v1.groups} != {g.marker_string for g in v2_alloc.groups}


# --- choosing the smoke marker explicitly ---------------------------------------


def test_omitting_marker_leaves_both_designs_exactly_as_they_were(tmp_path, no_network):
    """The default is what it always was: the fixture's marker for v1, the first
    selectable one for v2."""
    v1 = _run_smoke(V1_OPENAI_CONFIG, "--out", str(tmp_path / "v1"), env={},
                    sitecustomize=no_network)
    assert "call_id      05a54889c0b4f92ae28a6740972e19b9e4f925ed99177fcaf23d8374f3e4b366" \
        in v1.stdout
    assert "prompt hash  14df854e5228c780" in v1.stdout
    assert "(premise_indicator, 'because')" in v1.stdout

    v2 = _run_smoke(V2_CONFIG, "--out", str(tmp_path / "v2"), env={},
                    sitecustomize=no_network)
    assert "(conclusion_indicator, 'therefore')" in v2.stdout
    explicit = _run_smoke(V2_CONFIG, "--out", str(tmp_path / "v2b"), "--marker", "therefore",
                          env={}, sitecustomize=no_network)
    assert _call_id(v2.stdout) == _call_id(explicit.stdout), \
        "naming the default marker changes nothing"


def _call_id(stdout: str) -> str:
    return next(line.split()[1] for line in stdout.splitlines()
                if line.startswith("call_id "))


def test_selecting_it_follows_that_switches_family_and_realization(tmp_path, no_network):
    out = tmp_path / "smoke"
    result = _run_smoke(V2_CONFIG, "--out", str(out), "--marker", "it follows that",
                        env={}, sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "(metadiscursive_inference, 'it follows that')" in result.stdout
    request = json.loads(next((out / "requests").glob("*_group.json")).read_text())
    assert request["context"]["marker_string"] == "it follows that"
    assert request["context"]["marker_family"] == "metadiscursive_inference"
    assert request["context"]["marker_realization_id"] == \
        "sentence_initial_metadiscursive_v1"
    assert request["template_name"] == "group_draft_v2"


def test_the_two_v2_markers_share_everything_but_the_marker(tmp_path, no_network):
    """The endorsement, the template and the schema are one; only the marker and
    the realization description move."""
    requests = {}
    for marker in ("therefore", "it follows that"):
        out = tmp_path / marker.replace(" ", "_")
        _run_smoke(V2_CONFIG, "--out", str(out), "--marker", marker, env={},
                   sitecustomize=no_network)
        requests[marker] = json.loads(
            next((out / "requests").glob("*_group.json")).read_text())
    a, b = requests["therefore"], requests["it follows that"]
    assert a["context"]["endorsement"] == b["context"]["endorsement"]
    assert a["template_name"] == b["template_name"] == "group_draft_v2"
    assert a["template_sha256"] == b["template_sha256"]
    assert a["response_schema"] == b["response_schema"]
    assert a["prompt"] != b["prompt"]

    import difflib
    differing = [line for line in difflib.unified_diff(
        a["prompt"].splitlines(), b["prompt"].splitlines(), lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("---", "+++"))]
    assert differing
    for line in differing:
        carries = ("therefore" in line or "it follows that" in line
                   or "marker opens the concluding sentence" in line
                   or "metadiscursive frame opens the concluding sentence" in line)
        assert carries, f"a line differs for no marker-related reason: {line!r}"


@pytest.mark.parametrize("marker", ["because", "given that", "however", "thus", "nonsense"])
def test_a_non_selectable_marker_is_refused_before_any_credential(marker, tmp_path,
                                                                  no_network):
    result = _run_smoke(V2_CONFIG, "--out", str(tmp_path / "smoke"), "--marker", marker,
                        "--send",
                        env={"REASONSTYLE_ALLOW_OPENAI_GENERATION": "1",
                             "OPENAI_API_KEY": "sk-" + "t" * 40},
                        sitecustomize=no_network)
    assert result.returncode == 1
    assert "is not selectable under this configuration and its allocation" in result.stderr
    assert "no credential was read and no connection was made" in result.stderr
    assert "attempted to contact a backend" not in result.stderr
    assert not (tmp_path / "smoke" / "generation_log.jsonl").exists()


def test_marker_is_refused_for_the_non_pairwise_design(tmp_path, no_network):
    result = _run_smoke(V1_OPENAI_CONFIG, "--out", str(tmp_path / "smoke"),
                        "--marker", "therefore", env={}, sitecustomize=no_network)
    assert result.returncode == 1
    assert "applies only to a pairwise design" in result.stderr


def test_the_selectable_set_is_the_configuration_and_the_allocation_agreeing(v2):
    import importlib.util
    spec = importlib.util.spec_from_file_location("smoke_sel", SMOKE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    available = module.selectable_markers(v2)
    assert set(available) == {"therefore", "consequently", "it follows that",
                              "this implies"}
    assert available["therefore"] == "conclusion_indicator"
    assert available["it follows that"] == "metadiscursive_inference"
    if V2_ALLOCATION.is_file():
        allocated = {g.marker_string for g in load_allocation(V2_ALLOCATION).groups}
        assert set(available) == allocated, "both sources agree on the same four"


def test_the_pipeline_catches_a_missing_terminal_period(tmp_path, v2, segmenter, topics,
                                                        bank_hash, v2_allocation):
    """Through the real controller, not the validator alone: a body that lost
    its full stop must fail and enter the repair path, never be accepted."""
    class DropsThePeriod(V2Responder):
        def __call__(self, request):
            bodies = super().__call__(request)
            key = (f"{request.decision_id}_v{request.variant_id}",
                   request.supported_option)
            if key == ("climate_01_v1", "opt_1"):
                bodies["NP"] = bodies["NP"].rstrip(".")
            return bodies

    _, _, results, _ = _v2_run(tmp_path, v2, segmenter, topics, bank_hash, v2_allocation,
                               responder=DropsThePeriod())
    failed = [r for r in results if r.outcome != ACCEPTED]
    assert len(failed) == 1
    codes = {c for a in failed[0].attempts for c in a.error_codes}
    assert "E_BODY_TERMINAL_PUNCTUATION" in codes
    assert "E_PAIR_TERMINAL_PUNCTUATION_MISMATCH" in codes
    assert len(failed[0].attempts) == 3, "it went down the existing repair path"


@pytest.mark.parametrize("prompt", ["group_draft_v2.txt", "repair_v2.txt"])
def test_the_v2_prompts_require_one_terminal_full_stop(prompt):
    text = " ".join((ROOT / "prompts" / prompt).read_text().split())
    assert "ends with exactly one full stop" in text
    assert "two bodies of a pair end identically" in text
