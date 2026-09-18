"""The read-only reading view of a recorded group run.

``pilot.py group-review`` exists because 46 of the pilot's 48 groups ended
``needs_manual_review`` on 2026-09-17 and now have to be read by a person. These
tests hold it to what a review of research evidence has to be: it shows every
expected group, it shows the text that currently stands rather than a superseded
one, it takes the final recorded attempt, it makes an unchanged repair obvious,
it regenerates byte-for-byte, and it neither sends nor writes anything outside
its own output directory.

Everything here is offline. The synthetic runs are driven through
``FakeBackend``; the command-line tests run the script in a subprocess with
every socket poisoned. The tests that read the live evidence skip when it is not
on the machine, because that evidence is gitignored and an ordinary committed
test must pass without it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
from datetime import date

import pytest
import yaml

from reasonstyle.corpus import segmenter_from_config
from reasonstyle.generation import FakeBackend, load_allocation
from reasonstyle.generation.approvals import REQUIRED_JUDGEMENTS, ScenarioApproval
from reasonstyle.generation.group_review import (
    MACHINE_VALID,
    NEEDS_CORRECTION,
    build_group_review,
    group_attempts,
)
from reasonstyle.generation.pipeline import (
    ACCEPTED,
    NEEDS_MANUAL_REVIEW,
    CallStore,
    recorded_scenarios,
    run_group_stage,
    run_scenario_stage,
)
from reasonstyle.hashing import content_hash, sha256_of

from test_pilot_orchestration import (        # the synthetic pilot, already trusted
    PilotResponder,
    approvals_for,
    bodies_for,
    scenario_text,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PILOT = ROOT / "scripts" / "pilot.py"
#: The complete recorded group run. Gitignored, so every test that reads it skips
#: when it is absent; nothing committed here depends on it being present.
LIVE_GROUP_RUN = (ROOT / "data" / "pilot" / "run"
                  / "pilot_groups_snapshot_2026-09-17_complete")
APPROVALS = ROOT / "data" / "pilot" / "scenario_approvals.yaml"
SCENARIO_CORRECTIONS = ROOT / "data" / "pilot" / "scenario_corrections.yaml"

NO_NETWORK = """
import socket, urllib.request
def _blocked(*args, **kwargs):
    raise AssertionError("the script attempted to contact a backend")
socket.socket = _blocked
socket.create_connection = _blocked
urllib.request.urlopen = _blocked
"""


# --- shared scaffolding ------------------------------------------------------


@pytest.fixture
def topics(pilot_bank):
    return [t for t in pilot_bank.topics if t.status == "curated"]


@pytest.fixture
def bank_hash(pilot_bank):
    return content_hash(pilot_bank.model_dump(mode="json"))


@pytest.fixture
def allocation():
    return load_allocation(ROOT / "data" / "pilot" / "marker_allocation.yaml")


@pytest.fixture
def no_network(tmp_path_factory):
    path = tmp_path_factory.mktemp("no_network")
    (path / "sitecustomize.py").write_text(NO_NETWORK)
    return path


def _run_cli(*args, env=None, sitecustomize):
    environment = {**os.environ, **(env or {}),
                   "PYTHONPATH": f"{sitecustomize}:{os.environ.get('PYTHONPATH', '')}"}
    return subprocess.run([sys.executable, str(PILOT), "--config", "configs/experiment.yaml",
                           *args], cwd=ROOT, capture_output=True, text=True, env=environment)


def _pilot_module():
    spec = importlib.util.spec_from_file_location("pilot_cli", PILOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Args:
    """The reporting arguments ``group-review`` reads, and nothing more."""

    def __init__(self, run, out, *, approvals_file, scenario_corrections_file,
                 variants=(1, 2)):
        self.out = str(run)
        self.group_review_out = str(out)
        self.approvals_file = str(approvals_file)
        self.scenario_corrections_file = str(scenario_corrections_file)
        self.variants = list(variants)
        self.only = None


class StuckResponder(PilotResponder):
    """Fails two groups on purpose, one of them with repairs that change nothing.

    ``frozen`` returns the same unrepairable four bodies at every attempt, like
    the 30 live groups whose three attempts were byte-identical. ``drifting``
    changes its text on the first repair and then repeats itself, like the ones
    whose second repair alone made no progress.
    """

    def __init__(self, frozen: tuple[str, str], drifting: tuple[str, str],
                 regressing: tuple[str, str] | None = None):
        super().__init__()
        self.frozen, self.drifting, self.regressing = frozen, drifting, regressing
        self.seen: dict[tuple[str, str], int] = {}

    def __call__(self, request):
        reply = super().__call__(request)
        if request.kind not in ("group", "repair"):
            return reply
        key = (f"{request.decision_id}_v{request.variant_id}", request.supported_option)
        if key == self.frozen:
            # One body sentence where two are required, identically every time.
            return {**reply, "RS": "One sentence only, with no connective at all."}
        if key == self.drifting:
            self.seen[key] = self.seen.get(key, 0) + 1
            suffix = " again" if self.seen[key] > 1 else ""
            return {**reply, "RS": f"Still one sentence{suffix}, still wrong."}
        if key == self.regressing:
            # Attempt 2 breaks one rule; attempts 1 and 3 break that one and a
            # second, so the middle attempt carries strictly fewer codes.
            self.seen[key] = self.seen.get(key, 0) + 1
            broken = {**reply, "RS": "One sentence only, with no connective at all."}
            if self.seen[key] == 2:
                return broken
            return {**broken, "NP": broken["NP"].replace("My view holds, and", "Given that")}
        return reply


@pytest.fixture
def synthetic(tmp_path, cfg, segmenter, topics, bank_hash, allocation):
    """A complete 48-group run: 46 clean, one frozen failure, one drifting."""
    store = CallStore(tmp_path / "run", cfg, topic_bank_content_hash=bank_hash,
                      allocation_content_hash=allocation.content_hash)
    run_scenario_stage(topics, cfg, segmenter, FakeBackend(PilotResponder()), store)
    approvals = approvals_for(store, cfg, bank_hash)
    responder = StuckResponder(("climate_01_v1", "opt_1"), ("energy_02_v2", "opt_2"),
                               ("technology_01_v1", "opt_1"))
    run_group_stage(topics, allocation.groups, cfg, segmenter, FakeBackend(responder), store,
                    approvals=approvals, topic_bank_content_hash=bank_hash)
    return store, approvals


def _export(synthetic, topics, cfg, segmenter, bank_hash, allocation):
    store, approvals = synthetic
    return build_group_review(
        topics=topics, allocation_groups=allocation.groups,
        scenarios=recorded_scenarios(store), store=store, cfg=cfg, segmenter=segmenter,
        approvals=approvals, topic_bank_content_hash=bank_hash,
        allocation_content_hash=allocation.content_hash)


# --- every expected group is shown -------------------------------------------


def test_all_48_expected_groups_appear(synthetic, topics, cfg, segmenter, bank_hash,
                                       allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    assert len(export.reviews) == 48
    assert len({(r.scenario_id, r.supported_option) for r in export.reviews}) == 48
    assert export.manifest["expected_groups"] == 48
    index = export.files["index.md"]
    for review in export.reviews:
        assert f"`{review.scenario_id}` | `{review.supported_option}`" in index
    assert len([n for n in export.files if n.startswith("scenarios/")]) == 24


def test_an_expected_group_with_no_call_is_still_shown(tmp_path, cfg, segmenter, topics,
                                                       bank_hash, allocation):
    """A report built from the recorded calls alone would show a run that
    generated nothing as complete."""
    store = CallStore(tmp_path / "empty", cfg, topic_bank_content_hash=bank_hash,
                      allocation_content_hash=allocation.content_hash)
    export = build_group_review(
        topics=topics, allocation_groups=allocation.groups, scenarios={}, store=store,
        cfg=cfg, segmenter=segmenter, approvals={}, topic_bank_content_hash=bank_hash,
        allocation_content_hash=allocation.content_hash)
    assert len(export.reviews) == 48
    assert export.manifest["not_recorded"] == 48
    assert export.manifest["machine_valid"] == 0
    assert "no recorded call" in export.files["index.md"]


# --- the two kinds of group are kept apart -----------------------------------


def test_machine_valid_and_failing_groups_are_separated(synthetic, topics, cfg, segmenter,
                                                        bank_hash, allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    assert len(export.machine_valid) == 45
    assert len(export.needing_correction) == 3
    assert {(r.scenario_id, r.supported_option) for r in export.needing_correction} == {
        ("climate_01_v1", "opt_1"), ("energy_02_v2", "opt_2"),
        ("technology_01_v1", "opt_1")}
    index = export.files["index.md"]
    assert "## Machine-valid groups" in index
    assert "## Groups requiring inspection and an audited correction" in index
    assert index.index("## Machine-valid groups") < index.index(
        "## Groups requiring inspection")


def test_machine_valid_is_never_reported_as_approved(synthetic, topics, cfg, segmenter,
                                                     bank_hash, allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    assert "It is not approval" in export.files["index.md"]
    valid = export.machine_valid[0]
    page = export.files[f"scenarios/{valid.scenario_id}.md"]
    assert "Outstanding human judgements" in page
    assert valid.human_review, "a clean group still carries every human judgement"


# --- final-attempt selection --------------------------------------------------


def test_the_final_recorded_attempt_is_the_one_that_stands(synthetic, topics, cfg, segmenter,
                                                           bank_hash, allocation):
    store, _ = synthetic
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    attempts = group_attempts(store)
    for review in export.reviews:
        recorded = attempts[(review.scenario_id, review.supported_option)]
        assert review.attempts == recorded
        assert review.final == recorded[-1]
        assert review.final.attempt == max(a.attempt for a in recorded)
    stuck = next(r for r in export.reviews
                 if (r.scenario_id, r.supported_option) == ("climate_01_v1", "opt_1"))
    assert [a.attempt for a in stuck.attempts] == [1, 2, 3]
    assert stuck.final.outcome == NEEDS_MANUAL_REVIEW
    assert stuck.final.kind == "repair"
    clean = next(r for r in export.machine_valid)
    assert clean.final.attempt == 1 and clean.final.outcome == ACCEPTED


def test_a_transport_failure_is_not_counted_as_an_attempt(tmp_path, cfg, segmenter,
                                                          bank_hash, allocation):
    """It completed nothing and consumed no budget position, so the retry that
    followed it is the attempt, not the failure."""
    run = tmp_path / "run"
    (run / "results").mkdir(parents=True)
    base = {"kind": "group", "decision_id": "climate_01", "variant_id": 1,
            "supported_option": "opt_1", "prompt_sha256": "c" * 64, "attempt": 1,
            "stop_reason": "stop", "usage": None, "error": None,
            "validation": {"error_codes": [], "warning_codes": []}}
    entries = [
        {**base, "call_id": "a" * 64, "status": "error", "outcome": "transport_error"},
        {**base, "call_id": "a" * 64, "status": "ok", "outcome": "accepted"},
    ]
    (run / "generation_log.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in entries))
    (run / "results" / f"{'a' * 64}.json").write_text(json.dumps(
        {"call_id": "a" * 64, "fields": {c: f"{c} body." for c in ("RS", "RP", "NS", "NP")}}))
    store = CallStore(run, cfg, topic_bank_content_hash=bank_hash)
    attempts = group_attempts(store)[("climate_01_v1", "opt_1")]
    assert len(attempts) == 1 and attempts[0].outcome == ACCEPTED


# --- the text that currently stands ------------------------------------------


def _corrected(text: str) -> str:
    return text.replace("One route must be chosen before the tender closes.",
                        "One of the two routes must be chosen before this tender closes.")


def test_the_current_corrected_scenario_text_is_used_not_the_superseded_one(
        tmp_path, cfg, segmenter, topics, bank_hash, allocation):
    """The group stage drafted against the corrected text, so the review must
    read the group against that text and not against what the model returned."""
    store = CallStore(tmp_path / "run", cfg, topic_bank_content_hash=bank_hash,
                      allocation_content_hash=allocation.content_hash)
    run_scenario_stage(topics, cfg, segmenter, FakeBackend(PilotResponder()), store)
    scenarios = recorded_scenarios(store)
    target = "climate_01_v1"
    original = scenarios[target]["scenario_text"]
    corrected = _corrected(original)
    assert corrected != original

    corrections = tmp_path / "scenario_corrections.yaml"
    corrections.write_text(yaml.safe_dump([{
        "scenario_id": target, "original_call_id": scenarios[target]["call_id"],
        "original_text": original, "original_text_sha256": sha256_of(original),
        "corrected_text": corrected, "corrected_text_sha256": sha256_of(corrected),
        "editor": "Vidhi Bhutani", "reason": "tightening the closing sentence",
        "decided_at": "2026-09-17", "approval_state": "approved",
        "validation_error_codes": [], "validation_warning_codes": []}]))

    approvals_path = tmp_path / "approvals.yaml"
    body = {}
    for scenario_id, record in scenarios.items():
        text = corrected if scenario_id == target else record["scenario_text"]
        body[scenario_id] = {
            "scenario_text_sha256": sha256_of(text), "call_id": record["call_id"],
            "config_content_hash": cfg.content_hash,
            "topic_bank_content_hash": bank_hash, "decision": "approved",
            "judgements": {name: True for name in REQUIRED_JUDGEMENTS},
            "reason": None, "decided_by": "Vidhi Bhutani", "decided_at": "2026-09-17"}
    approvals_path.write_text(yaml.safe_dump(body, sort_keys=True))

    from reasonstyle.generation.approvals import load_approvals
    run_group_stage(topics, allocation.groups, cfg, segmenter,
                    FakeBackend(PilotResponder()), store,
                    approvals=load_approvals(approvals_path),
                    topic_bank_content_hash=bank_hash,
                    scenarios=_pilot_module()._current_scenarios(
                        Args(store.directory, tmp_path / "out",
                             approvals_file=approvals_path,
                             scenario_corrections_file=corrections), cfg, store))

    module = _pilot_module()
    out = tmp_path / "review"
    args = Args(store.directory, out, approvals_file=approvals_path,
                scenario_corrections_file=corrections)
    assert module._group_review(args, cfg, _bank(), allocation, topics, store) == 0

    page = (out / "scenarios" / f"{target}.md").read_text()
    assert corrected in page
    assert original not in page, "the superseded model text must not stand in for it"
    assert "human-corrected" in page
    assert f"`{sha256_of(corrected)}`" in page
    other = (out / "scenarios" / "climate_02_v1.md").read_text()
    assert "no human correction: this is the model's own text" in other


def _bank():
    from reasonstyle.corpus.topics import load_topic_bank
    return load_topic_bank(ROOT / "data" / "topics" / "pilot_topics.yaml")


# --- what the reviewer has to be able to see ---------------------------------


def test_every_group_shows_all_four_conditions_and_their_counts(synthetic, topics, cfg,
                                                                segmenter, bank_hash,
                                                                allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    for review in export.reviews:
        page = export.files[f"scenarios/{review.scenario_id}.md"]
        for condition in ("RS", "RP", "NS", "NP"):
            assert f"**{condition}**" in page
            assert review.measurements[condition].word_count_body > 0
            assert review.measurements[condition].word_count_full > \
                review.measurements[condition].word_count_body, "the opening is counted once"
            assert review.measurements[condition].sentence_count_full == \
                review.measurements[condition].sentence_count_body + 1
        for row in ("| body words |", "| full-text words |", "| body sentences |",
                    "| full-text sentences |"):
            assert row in page
        assert "| | RS | RP | NS | NP |" in page, "the four cells side by side"


def test_a_group_page_carries_its_scenario_options_marker_and_rendered_text(
        synthetic, topics, cfg, segmenter, bank_hash, allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    review = export.reviews[0]
    page = export.files[f"scenarios/{review.scenario_id}.md"]
    assert review.scenario_text in page
    assert review.options["opt_1"] in page and review.options["opt_2"] in page
    assert f"`{review.marker_family}`" in page
    assert f"“{review.marker_string}”" in page
    assert f"`{review.marker_realization_id}`" in page
    assert f"`{review.final.call_id}`" in page
    assert f"attempt {review.final.attempt} of" in page
    assert f"outcome `{review.final.outcome}`" in page
    opening = cfg.raw["corpus"]["counterargument_opening"]
    for condition in ("RS", "RP", "NS", "NP"):
        rendered = review.rendered(condition)
        assert rendered.startswith(opening + " ")
        assert rendered in page, "the complete counterargument, opening included"


def test_findings_appear_with_the_measurements_behind_them(synthetic, topics, cfg, segmenter,
                                                           bank_hash, allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    failing = next(r for r in export.needing_correction
                   if (r.scenario_id, r.supported_option) == ("climate_01_v1", "opt_1"))
    page = export.files[f"scenarios/{failing.scenario_id}.md"]
    assert failing.errors
    for finding in failing.errors:
        assert f"`{finding.code}`" in page
        assert finding.message in page
        for key in finding.detail:
            assert f"`{key}`:" in page, f"{finding.code} detail {key} is not shown"
    assert "E_SENTENCE_COUNT_MISMATCH" in {f.code for f in failing.errors}


def test_the_outstanding_human_judgements_are_listed_for_every_group(synthetic, topics, cfg,
                                                                     segmenter, bank_hash,
                                                                     allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    for review in export.reviews:
        codes = {f.code for f in review.human_review}
        assert {"H_SUPPORT_DIRECTION", "H_SUBSTANTIVE_SUPPORT", "H_NATURALNESS",
                "H_PRAGMATIC_COMMITMENT", "H_NO_REASON_INTEGRITY",
                "H_PROPOSITION_PRESERVATION",
                "H_REALIZATION_YIELDS_REASON_FREE_NS"} <= codes
        page = export.files[f"scenarios/{review.scenario_id}.md"]
        for code in sorted(codes):
            assert f"`{code}`" in page


def test_the_validator_is_used_rather_than_a_second_implementation():
    """A review that measured with its own code could show a group as passing a
    check the corpus would fail it on."""
    source = (ROOT / "src" / "reasonstyle" / "generation"
              / "group_review.py").read_text()
    assert "from ..corpus.validate import HUMAN_REVIEW_CODES, validate_group" in source
    assert "from ..corpus.validate import _measure" in source
    assert "from .pipeline import ACCEPTED, CallStore, _block_from" in source
    for invented in ("def _validate", "ratio_fail", "body_sentences", "compiled_forbidden"):
        assert invented not in source, f"{invented} looks like a re-implemented rule"


# --- an unchanged repair is unmissable ---------------------------------------


def test_a_repair_that_returned_unchanged_text_is_visibly_identified(synthetic, topics, cfg,
                                                                     segmenter, bank_hash,
                                                                     allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    frozen = next(r for r in export.reviews
                  if (r.scenario_id, r.supported_option) == ("climate_01_v1", "opt_1"))
    assert frozen.identical_throughout
    assert [frozen.unchanged_from_previous(a) for a in frozen.attempts] == [False, True, True]
    page = export.files[f"scenarios/{frozen.scenario_id}.md"]
    assert "**UNCHANGED from the previous attempt**" in page
    assert "This repair returned the previous four bodies verbatim." in page
    assert "Every recorded attempt returned the same four bodies." in page

    drifting = next(r for r in export.reviews
                    if (r.scenario_id, r.supported_option) == ("energy_02_v2", "opt_2"))
    assert not drifting.identical_throughout
    assert [drifting.unchanged_from_previous(a) for a in drifting.attempts] == \
        [False, False, True]

    entry = next(g for g in export.manifest["groups"]
                 if (g["scenario_id"], g["supported_option"]) == ("climate_01_v1", "opt_1"))
    assert [a["unchanged_from_previous"] for a in entry["attempts"]] == [False, True, True]
    assert entry["identical_bodies_throughout"] is True
    assert "repair returning unchanged text at attempt" in export.files["index.md"]


def test_a_failed_group_carries_its_whole_attempt_history(synthetic, topics, cfg, segmenter,
                                                          bank_hash, allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    failing = next(r for r in export.needing_correction)
    page = export.files[f"scenarios/{failing.scenario_id}.md"]
    assert "#### Attempt history" in page
    for attempt in failing.attempts:
        assert f"`{attempt.call_id}`" in page
        assert f"`{attempt.prompt_sha256}`" in page
        assert f"Attempt {attempt.attempt} — {attempt.kind}" in page
        for condition in ("RS", "RP", "NS", "NP"):
            assert (attempt.bodies or {})[condition] in page
        for code in attempt.recorded_error_codes:
            assert code in page


def test_the_recorded_codes_are_printed_beside_the_recomputed_ones(synthetic, topics, cfg,
                                                                   segmenter, bank_hash,
                                                                   allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    for review in export.reviews:
        assert review.recorded_codes_agree, review.scenario_id
    page = export.files["scenarios/climate_01_v1.md"]
    assert "recorded errors" in page


# --- nothing here corrects, approves or proposes -----------------------------


def test_the_review_writes_no_correction_ledger_and_proposes_no_wording(
        tmp_path, synthetic, topics, cfg, segmenter, bank_hash, allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    out = tmp_path / "review"
    export.write(out)
    written = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert all(name.endswith(".md") or name == "MANIFEST.json" for name in written)
    assert not any("correction" in name for name in written)
    assert not (tmp_path / "manual_corrections.yaml").exists()
    combined = "\n".join(export.files.values())
    for phrase in ("suggested correction", "corrected text:", "you could write",
                   "proposed wording", "rewrite as"):
        assert phrase not in combined.lower()
    assert "inspection only" in export.files["index.md"].lower()


def test_no_generation_or_validator_decision_is_recorded(synthetic, topics, cfg, segmenter,
                                                         bank_hash, allocation):
    """The pass reports what happened; what to change about drafting is not
    something a reading view gets to conclude."""
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    index = export.files["index.md"].lower()
    assert "no generator, model or validator decision follows from it" in index
    source = (ROOT / "src" / "reasonstyle" / "generation" / "group_review.py").read_text()
    for writer in ("save_corrections(", "save_approvals(", "save_scenario_corrections(",
                   "ManualCorrection(", "assemble_pilot(", "write_pilot_corpus(",
                   "from .assemble import", "manual_corrections.yaml"):
        assert writer not in source, f"{writer} would make this pass more than a report"


# --- determinism --------------------------------------------------------------


def test_regenerating_produces_identical_output(tmp_path, synthetic, topics, cfg, segmenter,
                                                bank_hash, allocation):
    first = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    second = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    assert first.files == second.files
    assert first.manifest == second.manifest

    one, two = tmp_path / "one", tmp_path / "two"
    first.write(one)
    second.write(two)
    written = {p.relative_to(one): p.read_bytes() for p in one.rglob("*") if p.is_file()}
    assert written == {p.relative_to(two): p.read_bytes() for p in two.rglob("*")
                       if p.is_file()}
    assert len(written) == len(first.files) + 1


def test_no_generated_file_carries_a_wall_clock_timestamp(synthetic, topics, cfg, segmenter,
                                                          bank_hash, allocation):
    """A file that differed from one day to the next could not be used to
    confirm which evidence was read."""
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    import re
    clock = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
    for name, text in export.files.items():
        assert not clock.search(text), name
    assert not clock.search(json.dumps(export.manifest))


def test_the_manifest_hashes_every_generated_file(synthetic, topics, cfg, segmenter,
                                                  bank_hash, allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    assert set(export.manifest["files"]) == set(export.files)
    for name, digest in export.manifest["files"].items():
        assert digest == sha256_of(export.files[name])


# --- read-only and offline, from the command line ----------------------------


ALL_KEYS = {"REASONSTYLE_ALLOW_LOCAL_GENERATION": "1",
            "REASONSTYLE_ALLOW_PILOT_GENERATION": "1", "HF_HUB_OFFLINE": "1"}


@pytest.mark.parametrize("send", [[], ["--send"]])
@pytest.mark.parametrize("keys", [{}, ALL_KEYS])
def test_group_review_contacts_nothing_with_or_without_the_generation_keys(
        send, keys, tmp_path, no_network):
    """Every authorisation key set, and ``--send`` given: it is still a report."""
    result = _run_cli("group-review", "--out", str(tmp_path / "run"),
                      "--group-review-out", str(tmp_path / "review"),
                      "--approvals-file", str(tmp_path / "approvals.yaml"),
                      "--scenario-corrections-file", str(tmp_path / "corrections.yaml"),
                      *send, env=keys, sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "attempted to contact a backend" not in result.stderr
    assert not (tmp_path / "run").exists(), "it creates no run directory"
    assert (tmp_path / "review" / "index.md").is_file()
    assert "48 expected group(s)" in (tmp_path / "review" / "index.md").read_text()


def test_group_review_builds_no_backend_at_all():
    """The only two places in this script that can reach a server stay the two
    stage functions and the redraft stage."""
    source = PILOT.read_text()
    call_sites = [line for line in source.splitlines()
                  if "_live_backend(cfg)" in line and not line.lstrip().startswith("def ")]
    assert len(call_sites) == 2, "only the two generating stages build a backend"
    body = source[source.index("def _group_review("):source.index("def _assemble(")]
    assert "_live_backend" not in body
    assert "VLLMOpenAIBackend" not in body and "OpenAIResponsesBackend" not in body
    assert "backend" not in body.lower()
    assert "--send" not in body and "live_problems" not in body
    assert "group-review" not in str(_pilot_module().WHOLE_PILOT_COMMANDS)


def test_group_review_writes_only_under_its_own_output_directory(tmp_path, no_network,
                                                                 cfg, segmenter, topics,
                                                                 bank_hash, allocation):
    store = CallStore(tmp_path / "run", cfg, topic_bank_content_hash=bank_hash,
                      allocation_content_hash=allocation.content_hash)
    run_scenario_stage(topics, cfg, segmenter, FakeBackend(PilotResponder()), store)
    before = {p.relative_to(tmp_path / "run"): p.read_bytes()
              for p in sorted((tmp_path / "run").rglob("*")) if p.is_file()}
    result = _run_cli("group-review", "--out", str(tmp_path / "run"),
                      "--group-review-out", str(tmp_path / "review"),
                      "--approvals-file", str(tmp_path / "approvals.yaml"),
                      "--scenario-corrections-file", str(tmp_path / "corrections.yaml"),
                      env=ALL_KEYS, sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    after = {p.relative_to(tmp_path / "run"): p.read_bytes()
             for p in sorted((tmp_path / "run").rglob("*")) if p.is_file()}
    assert after == before and before, "the recorded run is read, never written"
    assert not (tmp_path / "corrections.yaml").exists()
    assert not (tmp_path / "approvals.yaml").exists()


def test_group_review_may_be_filtered_like_any_reporting_command(tmp_path, no_network):
    result = _run_cli("group-review", "--out", str(tmp_path / "run"),
                      "--group-review-out", str(tmp_path / "review"),
                      "--approvals-file", str(tmp_path / "approvals.yaml"),
                      "--scenario-corrections-file", str(tmp_path / "corrections.yaml"),
                      "--only", "climate_01", "--variants", "1",
                      sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "runs on the complete pilot" not in result.stderr
    assert "2 expected group(s)" in (tmp_path / "review" / "index.md").read_text()


def test_group_review_says_what_it_did_not_do(tmp_path, no_network):
    result = _run_cli("group-review", "--out", str(tmp_path / "run"),
                      "--group-review-out", str(tmp_path / "review"),
                      "--approvals-file", str(tmp_path / "approvals.yaml"),
                      "--scenario-corrections-file", str(tmp_path / "corrections.yaml"),
                      sitecustomize=no_network)
    assert "INSPECTION ONLY" in result.stdout
    assert "No correction ledger was written or populated" in result.stdout
    assert "Machine-valid is not approval" in result.stdout
    assert "read only; nothing was written there" in result.stdout


# --- the live evidence, when it is on this machine ----------------------------


@pytest.fixture
def live_store(tmp_path, cfg, bank_hash):
    """A copy of the complete group run. The original is never opened for writing."""
    if not LIVE_GROUP_RUN.is_dir():
        pytest.skip(f"{LIVE_GROUP_RUN} is not on this machine (gitignored run artefacts)")
    run = tmp_path / "live_run"
    shutil.copytree(LIVE_GROUP_RUN, run)
    return run


def _live_export(run, cfg, topics, allocation, bank_hash):
    module = _pilot_module()
    out = run.parent / "live_review"
    args = Args(run, out, approvals_file=APPROVALS,
                scenario_corrections_file=SCENARIO_CORRECTIONS)
    store = CallStore(run, cfg, topic_bank_content_hash=bank_hash,
                      allocation_content_hash=allocation.content_hash)
    assert module._group_review(args, cfg, _bank(), allocation, topics, store) == 0
    return out, json.loads((out / "MANIFEST.json").read_text())


def test_the_live_run_reports_2_machine_valid_and_46_requiring_review(
        live_store, cfg, topics, allocation, bank_hash):
    """The audit of the recorded run of 2026-09-17, against its stated result."""
    out, manifest = _live_export(live_store, cfg, topics, allocation, bank_hash)
    assert manifest["expected_groups"] == 48
    assert manifest["machine_valid"] == 2
    assert manifest["needing_correction"] == 46
    assert manifest["not_recorded"] == 0
    assert manifest["calls_recorded"] == 140, "two first drafts plus 46 groups of three"
    valid = {(g["scenario_id"], g["supported_option"]) for g in manifest["groups"]
             if g["state"] == MACHINE_VALID}
    assert valid == {("energy_01_v1", "opt_2"), ("energy_03_v1", "opt_2")}
    index = (out / "index.md").read_text()
    assert "**2 machine-valid** · **46 requiring correction**" in index


def test_the_live_run_reproduces_its_recorded_findings_exactly(
        live_store, cfg, topics, allocation, bank_hash):
    """Re-validating the final attempts must not disagree with what each call
    recorded at the time; a divergence would mean the review is reading the run
    against a different scenario or a different rule."""
    _, manifest = _live_export(live_store, cfg, topics, allocation, bank_hash)
    assert all(g["recorded_codes_agree"] for g in manifest["groups"])
    counts: dict[str, int] = {}
    for group in manifest["groups"]:
        if group["state"] != NEEDS_CORRECTION:
            continue
        for code in group["final_error_codes"]:
            counts[code] = counts.get(code, 0) + 1
    assert counts == {"E_WORD_RATIO_BODY": 44, "E_WORD_RATIO_FULL_TEXT": 42,
                      "E_BODY_SENTENCE_COUNT": 15, "E_SENTENCE_COUNT_MISMATCH": 15,
                      "E_PAIR_CONTENT_DRIFT": 6, "E_DUPLICATE_TEXT": 4,
                      "E_MARKER_MISSING_IN_STYLED_CELL": 3,
                      "E_OPENING_REPEATED_IN_BODY": 2, "E_MARKER_IN_PLAIN_CELL": 1}


def test_the_live_run_shows_what_the_repairs_did(live_store, cfg, topics, allocation,
                                                 bank_hash):
    out, manifest = _live_export(live_store, cfg, topics, allocation, bank_hash)
    unchanged = {2: 0, 3: 0}
    for group in manifest["groups"]:
        for attempt in group["attempts"]:
            if attempt["unchanged_from_previous"]:
                unchanged[attempt["attempt"]] += 1
    assert unchanged == {2: 33, 3: 41}, "repair 1 and repair 2, as recorded"
    assert sum(1 for g in manifest["groups"] if g["identical_bodies_throughout"]) == 30
    index = (out / "index.md").read_text()
    assert "| repair returning unchanged text at attempt 2 | 33 of 46 |" in index
    assert "| repair returning unchanged text at attempt 3 | 41 of 46 |" in index
    assert "| identical bodies at every recorded attempt | 30 of 46 |" in index


def test_the_live_run_flags_the_three_groups_with_a_cleaner_earlier_attempt(
        live_store, cfg, topics, allocation, bank_hash):
    """Informational only, and exactly three in the 2026-09-17 evidence."""
    out, manifest = _live_export(live_store, cfg, topics, allocation, bank_hash)
    flagged = {(g["scenario_id"], g["supported_option"]):
               g["earlier_attempt_with_fewer_error_codes"]
               for g in manifest["groups"] if g["earlier_attempt_with_fewer_error_codes"]}
    assert set(flagged) == {("climate_01_v1", "opt_1"), ("climate_04_v2", "opt_2"),
                            ("energy_02_v1", "opt_2")}
    for key, entry in flagged.items():
        assert entry["final_attempt"] == 3, key
        assert entry["attempt"] < entry["final_attempt"], key
        assert len(entry["error_codes"]) < len(entry["final_error_codes"]), key
    assert flagged[("climate_01_v1", "opt_1")]["attempt"] == 2
    assert flagged[("energy_02_v1", "opt_2")]["error_codes"] == ["E_BODY_SENTENCE_COUNT"]

    section = (out / "index.md").read_text().split("## Informational")[1].split("\n## ")[0]
    for (scenario_id, option), entry in flagged.items():
        assert f"`{scenario_id}`" in section and f"`{option}`" in section
        assert f"`{entry['call_id']}`" in section
        assert f"`{entry['final_call_id']}`" in section
    assert "**The final recorded attempt remains canonical.**" in section
    assert "**This flag selects nothing and approves nothing.**" in section
    assert "**Fewer machine-error codes does not mean better.**" in section
    # The 46 still stand as the groups needing correction; nothing was promoted.
    assert manifest["needing_correction"] == 46 and manifest["machine_valid"] == 2


def test_the_live_run_uses_the_corrected_scenario_texts(live_store, cfg, topics, allocation,
                                                        bank_hash):
    """Five scenarios carry an approved human correction; their groups must be
    read against the corrected text, which is what they were drafted from."""
    out, manifest = _live_export(live_store, cfg, topics, allocation, bank_hash)
    corrected = {g["scenario_id"] for g in manifest["groups"]
                 if g["scenario_human_corrected"]}
    assert corrected == {"climate_01_v1", "energy_01_v1", "energy_03_v2",
                         "technology_03_v1", "technology_03_v2"}
    assert all(g["scenario_approval_state"] == "approved" for g in manifest["groups"])
    from reasonstyle.generation.scenario_corrections import load_scenario_corrections
    for correction in load_scenario_corrections(SCENARIO_CORRECTIONS):
        page = (out / "scenarios" / f"{correction.scenario_id}.md").read_text()
        assert correction.corrected_text in page
        assert correction.original_text not in page
        assert f"`{correction.original_call_id}`" in page


def test_the_live_run_evidence_stays_byte_identical(live_store, cfg, topics, allocation,
                                                    bank_hash):
    before = {p.relative_to(LIVE_GROUP_RUN): p.read_bytes()
              for p in sorted(LIVE_GROUP_RUN.rglob("*")) if p.is_file()}
    _live_export(live_store, cfg, topics, allocation, bank_hash)
    after = {p.relative_to(LIVE_GROUP_RUN): p.read_bytes()
             for p in sorted(LIVE_GROUP_RUN.rglob("*")) if p.is_file()}
    assert before == after
    assert len(before) >= 2 * 173, "a raw file and a result file for each of 173 calls"
    log = (LIVE_GROUP_RUN / "generation_log.jsonl").read_text().splitlines()
    assert len([line for line in log if line.strip()]) == 173


def test_the_live_review_regenerates_byte_for_byte(live_store, cfg, topics, allocation,
                                                   bank_hash):
    out, _ = _live_export(live_store, cfg, topics, allocation, bank_hash)
    first = {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*"))
             if p.is_file()}
    shutil.rmtree(out)
    _live_export(live_store, cfg, topics, allocation, bank_hash)
    assert {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*"))
            if p.is_file()} == first


# --- the committed suite does not need the ignored evidence -------------------


def test_no_committed_test_depends_on_the_ignored_run_evidence():
    """Every test that reads a gitignored run directory must skip without it, or
    the suite would pass only on the machine that holds the evidence."""
    offenders = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        source = path.read_text()
        reads_evidence = '"data" / "pilot" / "run"' in source
        if reads_evidence and "pytest.skip(" not in source:
            offenders.append(path.name)
    assert offenders == [], f"these read ignored run evidence without skipping: {offenders}"


def test_this_module_skips_cleanly_when_the_live_evidence_is_absent(tmp_path):
    """The guard itself, exercised: a missing directory skips rather than fails."""
    with pytest.raises(BaseException) as caught:
        if not (tmp_path / "absent").is_dir():
            pytest.skip("no live evidence")
    assert caught.typename == "Skipped"
    source = pathlib.Path(__file__).read_text()
    assert 'LIVE_GROUP_RUN = (ROOT / "data" / "pilot" / "run"' in source
    gitignore = (ROOT / ".gitignore").read_text()
    assert "data/pilot/run/" in gitignore, "the run evidence must stay out of git"


# --- the informational "an earlier attempt had fewer codes" flag --------------


def test_an_earlier_attempt_with_fewer_error_codes_is_flagged(synthetic, topics, cfg,
                                                              segmenter, bank_hash,
                                                              allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    flagged = {(r.scenario_id, r.supported_option): r for r in export.reviews
               if r.earlier_attempt_with_fewer_errors is not None}
    assert set(flagged) == {("technology_01_v1", "opt_1")}

    review = flagged[("technology_01_v1", "opt_1")]
    earlier = review.earlier_attempt_with_fewer_errors
    assert earlier.attempt == 2 and earlier is not review.final
    assert len(review.distinct_error_codes(earlier)) < \
        len(review.distinct_error_codes(review.final))
    assert set(review.distinct_error_codes(earlier)) < \
        set(review.distinct_error_codes(review.final))


def test_the_flag_names_both_attempts_with_their_calls_and_codes(synthetic, topics, cfg,
                                                                 segmenter, bank_hash,
                                                                 allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    review = next(r for r in export.reviews
                  if r.earlier_attempt_with_fewer_errors is not None)
    earlier, final = review.earlier_attempt_with_fewer_errors, review.final
    index = export.files["index.md"]
    section = index.split("## Informational")[1].split("\n## ")[0]
    assert f"`{review.scenario_id}`" in section
    assert f"`{review.supported_option}`" in section
    assert f"| {earlier.attempt} |" in section and f"`{earlier.call_id}`" in section
    assert f"| {final.attempt} |" in section and f"`{final.call_id}`" in section
    for code in review.distinct_error_codes(earlier):
        assert code in section
    for code in review.distinct_error_codes(final):
        assert code in section
    header = ("| Scenario | Option | Earlier attempt | Earlier call | Earlier error codes "
              "| Final attempt | Final call | Final error codes |")
    assert header in section


def test_the_flag_says_what_it_is_not(synthetic, topics, cfg, segmenter, bank_hash,
                                      allocation):
    """Three things a reader could wrongly infer, all denied in the section."""
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    section = export.files["index.md"].split("## Informational")[1].split("\n## ")[0]
    assert "**The final recorded attempt remains canonical.**" in section
    assert "**This flag selects nothing and approves nothing.**" in section
    assert "**Fewer machine-error codes does not mean better.**" in section
    assert "not a recommendation to use it" in section
    assert "human judgements" in section


def test_the_flag_changes_no_attempt_selection(synthetic, topics, cfg, segmenter, bank_hash,
                                               allocation):
    """The canonical text stays the final attempt's, flag or no flag."""
    store, _ = synthetic
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    attempts = group_attempts(store)
    review = next(r for r in export.reviews
                  if r.earlier_attempt_with_fewer_errors is not None)
    recorded = attempts[(review.scenario_id, review.supported_option)]
    assert review.final == recorded[-1]
    assert review.state == NEEDS_CORRECTION
    page = export.files[f"scenarios/{review.scenario_id}.md"]
    section = page.split(f"### `{review.supported_option}`")[1].split("#### Attempt history")[0]
    for condition in ("RS", "RP", "NS", "NP"):
        assert review.final.bodies[condition] in section
    assert review.findings == review.attempt_findings[review.final.call_id]
    assert sorted({f.code for f in review.errors}) == \
        list(review.distinct_error_codes(review.final))


def test_a_group_whose_attempts_never_improve_is_not_flagged(synthetic, topics, cfg,
                                                             segmenter, bank_hash,
                                                             allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    frozen = next(r for r in export.reviews
                  if (r.scenario_id, r.supported_option) == ("climate_01_v1", "opt_1"))
    assert frozen.identical_throughout
    assert frozen.earlier_attempt_with_fewer_errors is None
    for review in export.machine_valid:
        assert review.earlier_attempt_with_fewer_errors is None, "one attempt, nothing earlier"


def test_the_flag_is_absent_from_the_index_when_nothing_qualifies(
        tmp_path, cfg, segmenter, topics, bank_hash, allocation):
    store = CallStore(tmp_path / "clean", cfg, topic_bank_content_hash=bank_hash,
                      allocation_content_hash=allocation.content_hash)
    run_scenario_stage(topics, cfg, segmenter, FakeBackend(PilotResponder()), store)
    approvals = approvals_for(store, cfg, bank_hash)
    run_group_stage(topics, allocation.groups, cfg, segmenter, FakeBackend(PilotResponder()),
                    store, approvals=approvals, topic_bank_content_hash=bank_hash)
    export = build_group_review(
        topics=topics, allocation_groups=allocation.groups,
        scenarios=recorded_scenarios(store), store=store, cfg=cfg, segmenter=segmenter,
        approvals=approvals, topic_bank_content_hash=bank_hash,
        allocation_content_hash=allocation.content_hash)
    assert len(export.machine_valid) == 48
    assert "## Informational" not in export.files["index.md"]
    assert all(g["earlier_attempt_with_fewer_error_codes"] is None
               for g in export.manifest["groups"])


def test_the_flag_is_carried_in_the_manifest(synthetic, topics, cfg, segmenter, bank_hash,
                                             allocation):
    export = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    entries = {(g["scenario_id"], g["supported_option"]):
               g["earlier_attempt_with_fewer_error_codes"] for g in export.manifest["groups"]}
    assert sum(1 for v in entries.values() if v) == 1
    entry = entries[("technology_01_v1", "opt_1")]
    review = next(r for r in export.reviews
                  if (r.scenario_id, r.supported_option) == ("technology_01_v1", "opt_1"))
    assert entry == {
        "attempt": review.earlier_attempt_with_fewer_errors.attempt,
        "call_id": review.earlier_attempt_with_fewer_errors.call_id,
        "error_codes": list(review.distinct_error_codes(
            review.earlier_attempt_with_fewer_errors)),
        "final_attempt": review.final.attempt,
        "final_call_id": review.final.call_id,
        "final_error_codes": list(review.distinct_error_codes(review.final))}


def test_an_unusable_attempt_cannot_qualify_by_having_no_findings(tmp_path, cfg, bank_hash):
    """A rejected response was never validated; zero findings is absence of a
    measurement, not a cleaner draft."""
    from reasonstyle.generation.group_review import AttemptRecord, GroupReview
    bodies = {c: f"{c} body." for c in ("RS", "RP", "NS", "NP")}
    rejected = AttemptRecord(call_id="a" * 64, kind="group", attempt=1, status="rejected",
                             outcome="repair_needed", prompt_sha256="b" * 64, bodies=None)
    final = AttemptRecord(call_id="c" * 64, kind="repair", attempt=2, status="ok",
                          outcome=NEEDS_MANUAL_REVIEW, prompt_sha256="d" * 64, bodies=bodies)
    from reasonstyle.corpus.findings import Finding
    error = Finding(code="E_WORD_RATIO_BODY", severity="error", message="m", scope="group")
    review = GroupReview(
        scenario_id="climate_01_v1", decision_id="climate_01", variant_id=1, domain="climate",
        supported_option="opt_1", marker_family="premise_indicator", marker_string="because",
        marker_realization_id="clause_initial_premise_v1", realization_description="",
        options={"opt_1": "a", "opt_2": "b"}, scenario_text="text", opening="Opening.",
        scenario_call_id="e" * 64, scenario_approval_state="approved",
        scenario_supersedes_call_id=None, scenario_correction=None,
        attempts=(rejected, final), findings=(error,), measurements={},
        attempt_findings={final.call_id: (error,)}, state=NEEDS_CORRECTION)
    assert review.earlier_attempt_with_fewer_errors is None


# --- the reading view describes the run it read, not a remembered one --------
#
# The same code produces the Qwen pilot's view and the hosted pilot's. Every
# count, every command and every claim on the page has to come from the run in
# front of it: a page that told a reader to regenerate it with the other
# generator's configuration, or that called 36 machine-valid groups "the two
# valid groups", would be describing a different corpus than the one it shows.

OPENAI_CONFIG = ROOT / "configs" / "experiment_openai_pilot.yaml"


@pytest.fixture(scope="module")
def openai_cfg():
    from reasonstyle.config import load_config
    return load_config(OPENAI_CONFIG)


class SelectiveResponder(PilotResponder):
    """Fails exactly the groups it is given, with text no repair can fix."""

    def __init__(self, failing: set[tuple[str, str]]):
        super().__init__()
        self.failing = failing
        self.seen: dict[tuple[str, str], int] = {}

    def __call__(self, request):
        reply = super().__call__(request)
        if request.kind not in ("group", "repair"):
            return reply
        key = (f"{request.decision_id}_v{request.variant_id}", request.supported_option)
        if key not in self.failing:
            return reply
        # Different text at every attempt, so no repair is recorded as unchanged.
        self.seen[key] = self.seen.get(key, 0) + 1
        return {**reply, "RS": f"One sentence only, attempt {self.seen[key]}, still wrong."}


def _export_for(cfg, store, topics, bank_hash, allocation, approvals):
    return build_group_review(
        topics=topics, allocation_groups=allocation.groups,
        scenarios=recorded_scenarios(store), store=store, cfg=cfg, segmenter=None,
        approvals=approvals, topic_bank_content_hash=bank_hash,
        allocation_content_hash=allocation.content_hash)


@pytest.fixture
def hosted_export(tmp_path, openai_cfg, segmenter, topics, bank_hash, allocation):
    """A synthetic run under the hosted configuration: 36 valid, 12 failing."""
    store = CallStore(tmp_path / "run_openai", openai_cfg, topic_bank_content_hash=bank_hash,
                      allocation_content_hash=allocation.content_hash)
    run_scenario_stage(topics, openai_cfg, segmenter, FakeBackend(PilotResponder()), store)
    approvals = approvals_for(store, openai_cfg, bank_hash)
    failing = {(g.scenario_id, g.supported_option)
               for g in sorted(allocation.groups,
                               key=lambda g: (g.decision_id, g.variant_id,
                                              g.supported_option))[:12]}
    run_group_stage(topics, allocation.groups, openai_cfg, segmenter,
                    FakeBackend(SelectiveResponder(failing)), store,
                    approvals=approvals, topic_bank_content_hash=bank_hash)
    return build_group_review(
        topics=topics, allocation_groups=allocation.groups,
        scenarios=recorded_scenarios(store), store=store, cfg=openai_cfg,
        segmenter=segmenter, approvals=approvals, topic_bank_content_hash=bank_hash,
        allocation_content_hash=allocation.content_hash)


def test_a_hosted_review_reports_its_own_counts_not_a_remembered_result(hosted_export):
    assert len(hosted_export.machine_valid) == 36
    assert len(hosted_export.needing_correction) == 12
    assert hosted_export.manifest["machine_valid"] == 36
    assert hosted_export.manifest["needing_correction"] == 12
    index = hosted_export.files["index.md"]
    assert "**48 expected group(s)** · **36 machine-valid** · **12 requiring correction**" \
        in index


def test_a_hosted_review_never_says_two_valid_groups(hosted_export):
    """The sentence that used to carry the Qwen result into every page."""
    everything = "\n".join(hosted_export.files.values())
    assert "the two valid groups" not in everything
    assert "for every machine-valid group exactly as much as for the rest" in \
        hosted_export.files["index.md"]
    for stale in ("2 machine-valid", "46 requiring correction", "Two diagnosed repairs"):
        assert stale not in everything, stale


def test_a_hosted_review_names_its_own_configuration_to_regenerate(hosted_export,
                                                                   openai_cfg):
    """The command echoes the configuration that was loaded, exactly as given."""
    expected = f"--config {openai_cfg.path.as_posix()}"
    for name, text in hosted_export.files.items():
        assert expected in text, name
        assert "experiment_openai_pilot.yaml" in text, name
        assert "--config configs/experiment.yaml" not in text, name
        assert "/experiment.yaml" not in text, name


def test_the_regeneration_command_names_the_run_that_was_read(hosted_export):
    run = hosted_export.manifest["run_directory"]
    for name, text in hosted_export.files.items():
        assert f"--out {run}" in text, name


def test_a_config_with_no_path_gets_a_neutral_instruction(hosted_export, openai_cfg,
                                                          tmp_path, segmenter, topics,
                                                          bank_hash, allocation):
    """A configuration built in memory has no file; the note must not invent one."""
    import copy
    from reasonstyle.config import ExperimentConfig, RawConfig
    raw = copy.deepcopy(openai_cfg.raw)
    pathless = ExperimentConfig(raw=raw, parsed=RawConfig.model_validate(raw), path=None)
    store = CallStore(tmp_path / "pathless", pathless, topic_bank_content_hash=bank_hash,
                      allocation_content_hash=allocation.content_hash)
    export = build_group_review(
        topics=topics, allocation_groups=allocation.groups, scenarios={}, store=store,
        cfg=pathless, segmenter=segmenter, approvals={},
        topic_bank_content_hash=bank_hash,
        allocation_content_hash=allocation.content_hash)
    index = export.files["index.md"]
    assert "--config None" not in index and "--config configs/" not in index
    assert "with the configuration and run directory named above" in index


def test_two_runs_of_different_shapes_get_different_pages(hosted_export, synthetic, topics,
                                                          cfg, segmenter, bank_hash,
                                                          allocation):
    """The counts are read off the run, so two runs cannot produce one page."""
    qwen_shaped = _export(synthetic, topics, cfg, segmenter, bank_hash, allocation)
    assert (len(qwen_shaped.machine_valid), len(qwen_shaped.needing_correction)) == (45, 3)
    assert (len(hosted_export.machine_valid), len(hosted_export.needing_correction)) == (36, 12)
    assert qwen_shaped.files["index.md"] != hosted_export.files["index.md"]
    for export in (qwen_shaped, hosted_export):
        index = export.files["index.md"]
        assert f"**{export.manifest['machine_valid']} machine-valid**" in index
        assert f"**{export.manifest['needing_correction']} requiring correction**" in index


def test_the_module_hardcodes_no_count_or_configuration():
    source = (ROOT / "src" / "reasonstyle" / "generation" / "group_review.py").read_text()
    body = source.split('"""', 2)[2]          # past the module docstring
    for constant in ("configs/experiment.yaml", "configs/experiment_openai_pilot.yaml",
                     "the two valid", "2 of 48", "36 of 48", "46 group"):
        assert constant not in body, f"{constant!r} is baked into the output"


def test_a_run_with_no_unchanged_repair_says_so(hosted_export):
    """The headline difference between the two pilots, stated from the evidence."""
    index = hosted_export.files["index.md"]
    assert "| repair returning unchanged text | 0 of 12 |" in index
    assert "| identical bodies at every recorded attempt | 0 of 12 |" in index
    assert "**Every repair returned different text.**" in index
    assert "repair returning unchanged text at attempt" not in index
