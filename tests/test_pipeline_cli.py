"""The two command-line entry points: what they can and cannot do.

``pilot.py`` must not be able to send at all, at any argument or environment
combination, until the curator-approval gate and the corpus assembler exist.
``pipeline_smoke.py`` may send exactly one scenario and one group, and only
with the same authorisations and server checks the one-call smoke test makes.

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


# --- pilot.py cannot send ----------------------------------------------------


ALL_KEYS = {"REASONSTYLE_ALLOW_LOCAL_GENERATION": "1",
            "REASONSTYLE_ALLOW_PILOT_GENERATION": "1",
            "HF_HUB_OFFLINE": "1"}


@pytest.mark.parametrize("command", ["plan", "status"])
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
def test_no_environment_combination_unlocks_sending(keys, tmp_path, no_network):
    result = _run(PILOT, "plan", "--out", str(tmp_path / "run"), "--send",
                  env=keys, sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "this script cannot send" in result.stdout
    assert "DRY RUN" in result.stdout
    assert not (tmp_path / "run" / "generation_log.jsonl").exists()


def test_the_refusal_names_the_work_that_is_missing(tmp_path, no_network):
    result = _run(PILOT, "plan", "--out", str(tmp_path / "run"), "--send",
                  env=ALL_KEYS, sitecustomize=no_network)
    assert "curator-approval gate" in result.stdout
    assert "corpus assembler" in result.stdout
    assert "pipeline_smoke.py" in result.stdout


def test_pilot_has_no_live_code_path_at_all():
    """Not a runtime check: the module imports no backend and no run_pilot, so
    there is nothing for a future edit to switch on by accident."""
    source = PILOT.read_text()
    for forbidden in ("VLLMOpenAIBackend", "run_pilot", "allow_live", "FakeBackend"):
        assert forbidden not in source, forbidden
    module = _load_pilot_module()
    assert module.live_problems(False) and module.live_problems(True)
    assert module.live_problems(True, env=ALL_KEYS), "no environment empties the refusal"


def test_one_plan_command_covers_scenarios_and_their_groups(tmp_path, no_network):
    """No command may suggest a narrower live scope than the pipeline has: the
    groups of a scenario are drafted from that scenario's own accepted text."""
    result = _run(PILOT, "plan", "--out", str(tmp_path / "run"),
                  env=ALL_KEYS, sitecustomize=no_network)
    assert "scenarios    24 calls" in result.stdout
    assert "groups       48 drafts" in result.stdout
    bad = _run(PILOT, "groups", "--out", str(tmp_path / "run2"),
               sitecustomize=no_network)
    assert bad.returncode != 0 and "invalid choice" in bad.stderr


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
