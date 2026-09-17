"""The two command-line entry points: what they can and cannot do.

``pilot.py`` sends only from its two stage commands, only with both
authorisation keys, and only over the complete pilot; everything else in it
reads and reports. ``pipeline_smoke.py`` may send exactly one scenario and one
group, with the local-generation key alone.

Every test here runs the scripts as subprocesses with a poisoned network: any
socket use fails loudly rather than reaching a real endpoint.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PILOT = ROOT / "scripts" / "pilot.py"
PIPELINE_SMOKE = ROOT / "scripts" / "pipeline_smoke.py"

#: Imported into the child process before the script runs: every socket, and
#: urllib's opener, raise. A script that tries to contact anything crashes.
NO_NETWORK = """
import socket, urllib.request
def _blocked(*args, **kwargs):
    raise AssertionError("the script attempted to contact a backend")
socket.socket = _blocked
socket.create_connection = _blocked
urllib.request.urlopen = _blocked
"""


@pytest.fixture
def no_network(tmp_path_factory):
    """A directory on PYTHONPATH whose sitecustomize poisons every socket."""
    path = tmp_path_factory.mktemp("no_network")
    (path / "sitecustomize.py").write_text(NO_NETWORK)
    return path


def _run(script: pathlib.Path, *args: str, env: dict[str, str] | None = None,
         sitecustomize: pathlib.Path):
    environment = {**os.environ, **(env or {}),
                   "PYTHONPATH": f"{sitecustomize}:{os.environ.get('PYTHONPATH', '')}"}
    return subprocess.run([sys.executable, str(script), "--config", "configs/experiment.yaml",
                           *args],
                          cwd=ROOT, capture_output=True, text=True, env=environment)


def _load_pilot_module():
    spec = importlib.util.spec_from_file_location("pilot_cli", PILOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- what pilot.py may and may not do ----------------------------------------


ALL_KEYS = {"REASONSTYLE_ALLOW_LOCAL_GENERATION": "1",
            "REASONSTYLE_ALLOW_PILOT_GENERATION": "1",
            "HF_HUB_OFFLINE": "1"}


@pytest.mark.parametrize("command", ["plan", "status", "approvals"])
@pytest.mark.parametrize("send", [[], ["--send"]])
def test_no_pilot_invocation_contacts_a_backend(command, send, tmp_path, no_network):
    """Every command, with and without --send, with every key set."""
    result = _run(PILOT, command, "--out", str(tmp_path / "run"), *send,
                  env=ALL_KEYS, sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "attempted to contact a backend" not in result.stderr
    assert not (tmp_path / "run" / "generation_log.jsonl").exists()
    assert not (tmp_path / "run" / "raw").exists()


@pytest.mark.parametrize("keys", [
    dict(zip(ALL_KEYS, combination))
    for combination in itertools.product(["1", ""], repeat=len(ALL_KEYS))
])
def test_no_environment_combination_makes_plan_send(keys, tmp_path, no_network):
    """`plan` is a report. No key turns it into a run."""
    result = _run(PILOT, "plan", "--out", str(tmp_path / "run"), "--send",
                  env=keys, sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "DRY RUN" in result.stdout
    assert not (tmp_path / "run" / "generation_log.jsonl").exists()
    assert not (tmp_path / "run" / "raw").exists()


def test_the_refusal_names_every_missing_key(tmp_path, no_network):
    result = _run(PILOT, "scenarios", "--out", str(tmp_path / "run"), "--send",
                  env={k: "" for k in ALL_KEYS}, sitecustomize=no_network)
    assert "REASONSTYLE_ALLOW_LOCAL_GENERATION" in result.stdout
    assert "REASONSTYLE_ALLOW_PILOT_GENERATION" in result.stdout
    assert "HF_HUB_OFFLINE" in result.stdout
    assert "separate authorisation from a smoke call" in result.stdout


def test_only_the_two_stage_commands_can_reach_a_backend(tmp_path, no_network):
    """Everything else in this script reads files and reports."""
    source = PILOT.read_text()
    # The backend is constructed only in the two stage functions and the
    # redraft stage; every other command reads files and prints.
    generating = (source[source.index("def _stage("):source.index("def _scenario_review(")])
    assert source.count("VLLMOpenAIBackend()") == 2
    assert generating.count("VLLMOpenAIBackend()") == 2
    for command in ("plan", "status", "approvals", "scenario-review", "assemble"):
        result = _run(PILOT, command, "--out", str(tmp_path / "run"), "--send",
                      "--approvals-file", str(tmp_path / "approvals.yaml"),
                      "--corrections-file", str(tmp_path / "corrections.yaml"),
                      "--corpus", str(tmp_path / "corpus.jsonl"),
                      "--review-out", str(tmp_path / "review"),
                      env=ALL_KEYS, sitecustomize=no_network)
        assert "attempted to contact a backend" not in result.stderr, command
        assert not (tmp_path / "run" / "raw").exists(), command


def test_the_two_stages_are_separate_commands(tmp_path, no_network):
    """No command drafts scenarios and then their groups: the curator's
    approval sits between them, and crossing it automatically is the thing the
    gate exists to prevent."""
    plan = _run(PILOT, "plan", "--out", str(tmp_path / "run"), env=ALL_KEYS,
                sitecustomize=no_network)
    assert "scenarios    24 calls" in plan.stdout and "groups       48 drafts" in plan.stdout
    scenarios = _run(PILOT, "scenarios", "--out", str(tmp_path / "run"),
                     sitecustomize=no_network)
    groups = _run(PILOT, "groups", "--out", str(tmp_path / "run"),
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  sitecustomize=no_network)
    assert "stage        scenarios" in scenarios.stdout
    assert "stage        groups" in groups.stdout
    unknown = _run(PILOT, "everything", "--out", str(tmp_path / "run2"),
                   sitecustomize=no_network)
    assert unknown.returncode != 0 and "invalid choice" in unknown.stderr


# --- pipeline_smoke.py: one scenario, one group, four calls ------------------


def test_the_pipeline_smoke_sends_nothing_without_authorisation(tmp_path, no_network):
    result = _run(PIPELINE_SMOKE, "--out", str(tmp_path / "smoke"), "--send",
                  env={"REASONSTYLE_ALLOW_LOCAL_GENERATION": "", "HF_HUB_OFFLINE": ""},
                  sitecustomize=no_network)
    assert result.returncode == 0
    assert "nothing was sent" in result.stdout
    assert "REASONSTYLE_ALLOW_LOCAL_GENERATION" in result.stdout
    assert "HF_HUB_OFFLINE" in result.stdout
    assert not (tmp_path / "smoke" / "generation_log.jsonl").exists()


def test_the_pipeline_smoke_refuses_without_a_resolvable_revision(tmp_path, no_network):
    result = _run(PIPELINE_SMOKE, "--out", str(tmp_path / "smoke"), "--send",
                  "--hf-home", str(tmp_path / "empty"),
                  env={"REASONSTYLE_ALLOW_LOCAL_GENERATION": "1", "HF_HUB_OFFLINE": "1"},
                  sitecustomize=no_network)
    assert result.returncode == 1
    assert "not in the cache" in result.stderr
    assert not (tmp_path / "smoke" / "generation_log.jsonl").exists()


def test_the_pipeline_smoke_refuses_without_a_server_runtime_record(tmp_path, cfg, no_network):
    revision = cfg.raw["models"]["generator"]["model"]["revision"]
    snapshot = tmp_path / "hub" / "models--Qwen--Qwen3-14B" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "model-00001-of-00001.safetensors"):
        (snapshot / name).write_text("{}")
    result = _run(PIPELINE_SMOKE, "--out", str(tmp_path / "smoke"), "--send",
                  "--hf-home", str(tmp_path), "--server-runtime", str(tmp_path / "none.json"),
                  env={"REASONSTYLE_ALLOW_LOCAL_GENERATION": "1", "HF_HUB_OFFLINE": "1"},
                  sitecustomize=no_network)
    assert result.returncode == 1
    assert "serve_vllm.sh" in result.stderr


def test_the_pipeline_smoke_announces_its_ceiling_and_its_synthetic_material(
        tmp_path, no_network):
    result = _run(PIPELINE_SMOKE, "--out", str(tmp_path / "smoke"), sitecustomize=no_network)
    assert "ceiling      4 calls" in result.stdout
    assert "SYNTHETIC energy_fixture_001" in result.stdout
    assert "opt_1" in result.stdout


def test_the_pipeline_smoke_needs_no_pilot_authorisation():
    source = PIPELINE_SMOKE.read_text()
    assert "REASONSTYLE_ALLOW_PILOT_GENERATION" not in source
    assert "run_pilot" not in source, "one group only: never the pilot controller"
    # The repair logic is the controller's, not a copy living in the script.
    assert "draft_group" in source and "repair_request" not in source


def test_the_pipeline_smoke_names_only_the_fixture_decision():
    source = PIPELINE_SMOKE.read_text()
    assert "energy_fixture_001" in source
    bank = json.loads(subprocess.check_output(
        [sys.executable, "-c",
         "import json;from reasonstyle.corpus.topics import load_topic_bank;"
         "print(json.dumps([t.decision_id for t in "
         "load_topic_bank('data/topics/pilot_topics.yaml').topics]))"],
        cwd=ROOT, text=True))
    for pilot_decision in bank:
        assert pilot_decision not in source, f"a pilot brief is named: {pilot_decision}"


# --- the approvals report distinguishes why a scenario is not usable ---------


def _log_line(**overrides):
    entry = {"call_id": "a" * 64, "kind": "scenario", "attempt": 1,
             "decision_id": "climate_01", "variant_id": 1, "supported_option": None,
             "template_name": "scenario_draft_v1", "template_sha256": "b" * 64,
             "prompt_sha256": "c" * 64, "model": "Qwen/Qwen3-14B", "model_returned": None,
             "request_fields": {}, "config_content_hash": "d" * 64,
             "topic_bank_content_hash": "e" * 64, "allocation_content_hash": None,
             "response_sha256": None, "stop_reason": "stop", "usage": None,
             "status": "ok", "error": None, "generated_at": "2026-09-16T10:00:00+00:00",
             "outcome": "accepted", "validation": {"error_codes": [], "warning_codes": []}}
    return {**entry, **overrides}


def _recorded(tmp_path, entries, results=None):
    """A run directory with hand-written log lines and result files."""
    run = tmp_path / "run"
    (run / "results").mkdir(parents=True)
    (run / "generation_log.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in entries))
    for call_id, fields in (results or {}).items():
        (run / "results" / f"{call_id}.json").write_text(
            json.dumps({"call_id": call_id, "kind": "scenario", "fields": fields}))
    return run


def _report(tmp_path, entries, no_network, results=None):
    run = _recorded(tmp_path, entries, results)
    result = _run(PILOT, "approvals", "--out", str(run), "--only", "climate_01",
                  "--variants", "1", "--approvals-file", str(tmp_path / "approvals.yaml"),
                  sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_a_transport_failure_leaves_the_scenario_not_generated(tmp_path, no_network):
    """No response arrived, so there is no text to approve — whatever the log
    shows about the attempt."""
    stdout = _report(tmp_path, [_log_line(status="error", outcome="transport_error",
                                          stop_reason=None)], no_network)
    assert "climate_01_v1" in stdout and "not_generated" in stdout
    assert "1 of 1 expected scenario(s) not approved" in stdout


def test_a_rejected_response_leaves_the_scenario_not_generated(tmp_path, no_network):
    stdout = _report(tmp_path, [_log_line(status="rejected", outcome="needs_manual_review",
                                          stop_reason="length")], no_network)
    assert "not_generated" in stdout
    assert "transport failure or rejected response" in stdout


def test_a_scenario_with_machine_errors_is_reported_as_blocked(tmp_path, no_network):
    """A machine error is not something approval can be recorded against."""
    entry = _log_line(outcome="needs_manual_review",
                      validation={"error_codes": ["E_LABEL_LEAKAGE"], "warning_codes": []})
    stdout = _report(tmp_path, [entry], no_network,
                     results={"a" * 64: {"scenario_text": "text with option A in it"}})
    assert "blocked_by_machine_errors" in stdout
    assert "cannot override the validator" in stdout


def test_an_unaccepted_but_error_free_scenario_needs_manual_review(tmp_path, no_network):
    entry = _log_line(outcome="needs_manual_review")
    stdout = _report(tmp_path, [entry], no_network,
                     results={"a" * 64: {"scenario_text": "a drafted scenario"}})
    assert "needs_manual_review" in stdout
    assert "not accepted" in stdout


def test_an_accepted_scenario_is_judged_on_its_approval(tmp_path, no_network):
    stdout = _report(tmp_path, [_log_line()], no_network,
                     results={"a" * 64: {"scenario_text": "a drafted scenario"}})
    assert "pending" in stdout and "no approval recorded" in stdout
    assert "not_generated" not in stdout and "blocked_by_machine_errors" not in stdout


# --- the two live stages, from the command line ------------------------------


STAGE_KEYS = {"REASONSTYLE_ALLOW_LOCAL_GENERATION": "1",
              "REASONSTYLE_ALLOW_PILOT_GENERATION": "1", "HF_HUB_OFFLINE": "1"}


@pytest.mark.parametrize("command", ["scenarios", "groups"])
@pytest.mark.parametrize("missing", ["REASONSTYLE_ALLOW_LOCAL_GENERATION",
                                     "REASONSTYLE_ALLOW_PILOT_GENERATION", "HF_HUB_OFFLINE"])
def test_a_stage_needs_all_three_keys(command, missing, tmp_path, no_network):
    """Each key is a separate decision, and any one missing stops the stage
    before a call artefact exists."""
    env = {**STAGE_KEYS, missing: ""}
    result = _run(PILOT, command, "--out", str(tmp_path / "run"), "--send",
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  env=env, sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "nothing was sent" in result.stdout and missing in result.stdout
    assert not (tmp_path / "run" / "generation_log.jsonl").exists()
    assert not (tmp_path / "run" / "raw").exists()


@pytest.mark.parametrize("command", ["scenarios", "groups"])
def test_a_stage_without_send_reports_and_sends_nothing(command, tmp_path, no_network):
    result = _run(PILOT, command, "--out", str(tmp_path / "run"),
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  env=STAGE_KEYS, sitecustomize=no_network)
    assert result.returncode == 0
    assert "nothing was sent" in result.stdout
    assert not (tmp_path / "run").exists() or not (tmp_path / "run" / "raw").exists()


@pytest.mark.parametrize("command", ["scenarios", "groups"])
def test_a_stage_refuses_before_the_first_call_without_the_server_checks(command, tmp_path,
                                                                        no_network):
    """Authorised, but no cached model: it stops before any call artefact."""
    result = _run(PILOT, command, "--out", str(tmp_path / "run"), "--send",
                  "--hf-home", str(tmp_path / "empty"),
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  env=STAGE_KEYS, sitecustomize=no_network)
    assert result.returncode == 1
    assert "no call was made" in result.stderr
    assert not (tmp_path / "run" / "generation_log.jsonl").exists()


def test_the_stage_commands_announce_their_own_ceilings(tmp_path, no_network):
    scenarios = _run(PILOT, "scenarios", "--out", str(tmp_path / "run"),
                     sitecustomize=no_network)
    groups = _run(PILOT, "groups", "--out", str(tmp_path / "run"),
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  sitecustomize=no_network)
    assert "24 calls at most" in scenarios.stdout and "no repair path" in scenarios.stdout
    assert "48 drafts, at most 144 calls" in groups.stdout


def test_the_groups_command_reports_the_gate_that_would_block_it(tmp_path, no_network):
    result = _run(PILOT, "groups", "--out", str(tmp_path / "run"),
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  sitecustomize=no_network)
    assert "24 scenario(s) would block group drafting" in result.stdout


def test_no_command_crosses_the_approval_boundary(tmp_path, no_network):
    """There is no command that drafts scenarios and then their groups."""
    source = PILOT.read_text()
    assert "run_scenario_stage" in source and "run_group_stage" in source
    body = source[source.index("def _stage("):]
    assert 'kind == "scenarios"' in body and 'kind == "groups"' in body
    # one stage per invocation: the two calls are in mutually exclusive branches
    assert body.count("run_scenario_stage(") == 1 and body.count("run_group_stage(") == 1


def test_the_scenario_review_template_approves_nothing(tmp_path, no_network):
    import yaml
    result = _run(PILOT, "scenario-review", "--out", str(tmp_path / "run"),
                  "--review-out", str(tmp_path / "review"), "--write-template",
                  "--only", "climate_01", "--variants", "1", sitecustomize=no_network)
    assert result.returncode == 0
    template = yaml.safe_load((tmp_path / "review" /
                               "scenario_approvals.template.yaml").read_text())
    entry = template["climate_01_v1"]
    assert entry["decision"] == "pending"
    assert set(entry["judgements"].values()) == {None}
    assert entry["decided_by"] is None
    assert "Approving is" in result.stdout
    review = (tmp_path / "review" / "scenarios.md").read_text()
    assert "climate_01_v1" in review and "Judgements to record" in review


def test_assemble_refuses_a_partial_pilot_and_writes_nothing(tmp_path, no_network):
    corpus = tmp_path / "corpus.jsonl"
    result = _run(PILOT, "assemble", "--out", str(tmp_path / "run"),
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  "--corrections-file", str(tmp_path / "corrections.yaml"),
                  "--corpus", str(corpus), sitecustomize=no_network)
    assert result.returncode == 1
    assert "nothing was assembled" in result.stderr
    assert not corpus.exists() and not corpus.with_suffix(".manifest.json").exists()


def test_the_status_command_counts_without_contacting_anything(tmp_path, no_network):
    result = _run(PILOT, "status", "--out", str(tmp_path / "run"),
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  "--corrections-file", str(tmp_path / "corrections.yaml"),
                  sitecustomize=no_network)
    assert result.returncode == 0
    for label in ("scenarios expected", "scenarios generated", "scenarios approved",
                  "groups expected", "accepted, no repair", "accepted after repair",
                  "needs_manual_review", "groups with an approved correction",
                  "groups ready to assemble", "blocking assembly"):
        assert label in result.stdout, label
    assert "attempted to contact a backend" not in result.stderr


# --- the whole-pilot boundary, from the command line -------------------------


@pytest.mark.parametrize("command", ["scenarios", "groups", "redraft-scenarios", "assemble"])
@pytest.mark.parametrize("selection", [
    ["--only", "climate_01"],
    ["--variants", "1"],
    ["--variants", "1", "1"],
    ["--variants", "1", "2", "3"],
])
def test_a_whole_pilot_command_refuses_a_subset(command, selection, tmp_path, no_network):
    """No live generation or assembly can quietly produce part of a pilot."""
    corpus = tmp_path / "corpus.jsonl"
    result = _run(PILOT, command, "--out", str(tmp_path / "run"), "--send", *selection,
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  "--corrections-file", str(tmp_path / "corrections.yaml"),
                  "--corpus", str(corpus), env=STAGE_KEYS, sitecustomize=no_network)
    assert result.returncode == 1
    assert "runs on the complete pilot" in result.stderr
    assert not (tmp_path / "run").exists() or not (tmp_path / "run" / "raw").exists()
    assert not corpus.exists()


@pytest.mark.parametrize("command", ["scenario-review", "approvals", "status", "log", "plan"])
def test_a_reporting_command_may_be_filtered(command, tmp_path, no_network):
    """Targeted reporting is useful and harmless: it reads and prints."""
    result = _run(PILOT, command, "--out", str(tmp_path / "run"), "--only", "climate_01",
                  "--variants", "1", "--approvals-file", str(tmp_path / "approvals.yaml"),
                  "--corrections-file", str(tmp_path / "corrections.yaml"),
                  "--review-out", str(tmp_path / "review"), sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "runs on the complete pilot" not in result.stderr


def test_the_review_page_carries_the_brief_behind_the_judgements(tmp_path, no_network):
    """A curator should not have to open the topic YAML to judge whether both
    facts are stated or the options are balanced."""
    _run(PILOT, "scenario-review", "--out", str(tmp_path / "run"),
         "--review-out", str(tmp_path / "review"), "--only", "climate_01", "--variants", "1",
         sitecustomize=no_network)
    page = (tmp_path / "review" / "scenarios.md").read_text()
    for heading in ("The brief this scenario was drafted from", "**Decision.**",
                    "**Option opt_1.**", "**Option opt_2.**", "*competing goal:*",
                    "**Why underdetermined.**", "**Variant 1 context.**",
                    "Variant facts", "The generated scenario"):
        assert heading in page, heading
    assert "`opt_1`" in page and "`opt_2`" in page


def test_status_counts_a_corrected_group_as_ready_not_blocked(tmp_path, no_network):
    """Counts come from each scenario's own state, and a corrected group is not
    reported as permanently blocking."""
    result = _run(PILOT, "status", "--out", str(tmp_path / "run"),
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  "--corrections-file", str(tmp_path / "corrections.yaml"),
                  sitecustomize=no_network)
    assert "groups ready to assemble" in result.stdout
    assert "groups with an approved correction" in result.stdout
    assert "correction record(s), one per cell" in result.stdout
    assert "blocking assembly" in result.stdout
    assert "scenario(s) not approved" in result.stdout
