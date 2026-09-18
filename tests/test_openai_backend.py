"""The hosted generator: OpenAI's Responses API, exercised entirely offline.

Not one test here reaches a network, reads a real credential or costs anything.
Every response is a fake one fed to an injected ``_urlopen``, and every
environment is an injected mapping, so the whole surface — payload, parsing,
authorisation, redaction, error handling, provenance — is tested without an
account existing.

What these tests are for is the part that cannot be checked by reading the
diff: that the payload is exactly what the configuration describes and nothing
more, that a secret cannot escape through a record or an error message, that a
paid call that came back is never silently made twice, and that the Qwen
configuration, its run directory and its rules are untouched by any of it.
"""

from __future__ import annotations

import io
import json
import pathlib
import urllib.error

import pytest

from reasonstyle.config import ConfigError, load_config
from reasonstyle.corpus.topics import load_topic_bank
from reasonstyle.generation import (
    API_KEY_ENV,
    OPENAI_AUTHORIZATION_ENV,
    OPENAI_BACKEND,
    BackendUnavailable,
    LiveCallRefused,
    OpenAIAuthError,
    OpenAIModelUnavailable,
    OpenAIRateLimited,
    OpenAIResponsesBackend,
    openai_authorization_problems,
    openai_preflight_problems,
    openai_responses_payload,
    redact_secrets,
    request_payload,
)
from reasonstyle.generation.backends import BackendError
from reasonstyle.generation.requests import group_request, scenario_request

ROOT = pathlib.Path(__file__).resolve().parents[1]
QWEN_CONFIG = ROOT / "configs" / "experiment.yaml"
OPENAI_CONFIG = ROOT / "configs" / "experiment_openai_pilot.yaml"

#: A stand-in that is shaped like a real key and is not one. It exists so the
#: redaction tests have something to fail on; it authenticates nothing.
FAKE_KEY = "sk-" + "t" * 40


@pytest.fixture(scope="module")
def oai_cfg():
    return load_config(OPENAI_CONFIG)


@pytest.fixture(scope="module")
def topic():
    bank = load_topic_bank(ROOT / "data" / "topics" / "pilot_topics.yaml")
    return sorted((t for t in bank.topics if t.status == "curated"),
                  key=lambda t: t.decision_id)[0]


@pytest.fixture
def live_env():
    return {OPENAI_AUTHORIZATION_ENV: "1", API_KEY_ENV: FAKE_KEY}


def _request(oai_cfg, topic):
    return scenario_request(topic, 1, oai_cfg)


def _reply(content: dict | str, *, status: str = "completed", model: str = "gpt-5.6-sol",
           usage: dict | None = None, incomplete: dict | None = None,
           output_text: bool = True) -> dict:
    text = content if isinstance(content, str) else json.dumps(content)
    raw: dict = {
        "id": "resp_abc123", "object": "response", "status": status, "model": model,
        "created_at": 1789000000, "store": False, "background": False,
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": text}]}],
        "usage": usage or {"input_tokens": 1200, "output_tokens": 180,
                           "input_tokens_details": {"cached_tokens": 1024},
                           "output_tokens_details": {"reasoning_tokens": 0},
                           "total_tokens": 1380},
    }
    if output_text:
        raw["output_text"] = text
    if incomplete:
        raw["incomplete_details"] = incomplete
    return raw


class _Recorder:
    """A stand-in transport. Records what was asked; reaches nothing."""

    def __init__(self, reply: dict | Exception, *, headers: dict | None = None):
        self.reply, self.headers = reply, headers or {"x-request-id": "req_xyz789"}
        self.calls: list[dict] = []

    def __call__(self, request, timeout=None):
        self.calls.append({"url": request.full_url, "headers": dict(request.headers),
                           "body": json.loads(request.data.decode("utf-8")),
                           "method": request.get_method(), "timeout": timeout})
        if isinstance(self.reply, Exception):
            raise self.reply
        body = json.dumps(self.reply).encode("utf-8")
        headers = self.headers

        class _Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False
        response = _Response(body)
        response.headers = headers
        return response


def _send(oai_cfg, topic, reply, *, env=None, **kw):
    transport = _Recorder(reply, **kw)
    backend = OpenAIResponsesBackend(_urlopen=transport, _env=env)
    return backend.send(_request(oai_cfg, topic), oai_cfg, allow_live=True), transport


# --- the payload is exactly what the configuration describes -----------------


def test_the_payload_is_the_configured_request_and_nothing_more(oai_cfg, topic):
    payload = openai_responses_payload(_request(oai_cfg, topic), oai_cfg)
    assert set(payload) == {"model", "input", "temperature", "max_output_tokens",
                            "reasoning", "text", "store", "background"}
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["temperature"] == 0.3
    assert payload["max_output_tokens"] == 700
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["store"] is False and payload["background"] is False


def test_the_model_is_the_exact_id_never_the_moving_alias(oai_cfg, topic):
    model = oai_cfg.raw["models"]["generator"]["model"]
    assert model["id"] == "gpt-5.6-sol"
    assert "gpt-5.6" in model["refused_aliases"]
    assert openai_responses_payload(_request(oai_cfg, topic), oai_cfg)["model"] == model["id"]


def test_a_pinned_snapshot_is_used_only_when_it_is_configured(oai_cfg, topic):
    """A more specific pin is a deliberate, approved edit — never a silent one."""
    assert oai_cfg.raw["models"]["generator"]["model"]["pinned_snapshot"] is None
    import copy
    raw = copy.deepcopy(oai_cfg.raw)
    raw["models"]["generator"]["model"]["pinned_snapshot"] = "gpt-5.6-sol-2026-08-01"
    from reasonstyle.config import ExperimentConfig, RawConfig
    pinned = ExperimentConfig(raw=raw, parsed=RawConfig.model_validate(raw), path=None)
    assert openai_responses_payload(_request(pinned, topic), pinned)["model"] == \
        "gpt-5.6-sol-2026-08-01"


@pytest.mark.parametrize("absent", ["top_p", "tools", "tool_choice", "previous_response_id",
                                    "conversation", "seed", "attachments", "file_ids",
                                    "web_search_options", "n", "stream"])
def test_the_payload_carries_no_tool_state_or_second_sampling_knob(oai_cfg, topic, absent):
    payload = openai_responses_payload(_request(oai_cfg, topic), oai_cfg)
    assert absent not in _keys_in(payload), (
        f"{absent!r} must not appear anywhere in the request body")


def _keys_in(value) -> set[str]:
    """Every mapping key at every depth, so an absence test cannot be fooled."""
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys_in(v)}
    if isinstance(value, list):
        return {k for item in value for k in _keys_in(item)}
    return set()


def test_structured_output_is_the_templates_own_strict_schema(oai_cfg, topic):
    request = group_request_for(oai_cfg, topic)
    payload = openai_responses_payload(request, oai_cfg)
    fmt = payload["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert fmt["schema"] == request.response_schema
    assert fmt["schema"]["additionalProperties"] is False
    assert set(fmt["schema"]["properties"]) == {"RS", "RP", "NS", "NP"}


def group_request_for(cfg, topic):
    from reasonstyle.generation import load_allocation
    allocation = load_allocation(ROOT / "data" / "pilot" / "marker_allocation.yaml")
    group = next(g for g in allocation.groups
                 if g.decision_id == topic.decision_id and g.variant_id == 1
                 and g.supported_option == "opt_1")
    return group_request(topic, 1, "A scenario sentence. And another one.", group, cfg)


def test_the_payload_dispatcher_follows_the_configured_backend(oai_cfg, topic):
    qwen = load_config(QWEN_CONFIG)
    assert "input" in request_payload(_request(oai_cfg, topic), oai_cfg)
    assert "messages" in request_payload(scenario_request(topic, 1, qwen), qwen)
    assert "max_tokens" in request_payload(scenario_request(topic, 1, qwen), qwen)


# --- structured-output parsing -----------------------------------------------


def test_a_completed_reply_is_parsed_into_the_structured_object(oai_cfg, topic, live_env):
    response, transport = _send(oai_cfg, topic, _reply({"scenario_text": "A scenario."}),
                                env=live_env)
    assert response.content == {"scenario_text": "A scenario."}
    assert response.stop_reason == "stop"
    assert response.model_returned == "gpt-5.6-sol"
    assert len(transport.calls) == 1, "one draft, one call"


def test_the_output_array_is_read_when_output_text_is_absent(oai_cfg, topic, live_env):
    reply = _reply({"scenario_text": "A scenario."}, output_text=False)
    response, _ = _send(oai_cfg, topic, reply, env=live_env)
    assert response.content == {"scenario_text": "A scenario."}


def test_several_messages_are_not_concatenated_into_one_draft(oai_cfg, topic, live_env):
    """One call must produce exactly one draft; stitching two together would
    invent a draft the model did not return."""
    reply = _reply({"scenario_text": "A."}, output_text=False)
    reply["output"] = reply["output"] * 2
    response, _ = _send(oai_cfg, topic, reply, env=live_env)
    assert response.content is None


def test_malformed_output_is_a_spent_attempt_not_a_transport_failure(oai_cfg, topic,
                                                                     live_env):
    """It was paid for. Recording it as a transport failure would let the
    controller make the same paid call again."""
    response, _ = _send(oai_cfg, topic, _reply("not json at all"), env=live_env)
    assert response.stop_reason == "stop"
    assert response.content == "not json at all"
    from reasonstyle.generation import ResponseRejected, parse_response
    with pytest.raises(ResponseRejected):
        parse_response(_request(oai_cfg, topic), response.content)


def test_an_incomplete_response_is_recorded_as_incomplete(oai_cfg, topic, live_env):
    reply = _reply({"scenario_text": "cut off"}, status="incomplete",
                   incomplete={"reason": "max_output_tokens"})
    response, _ = _send(oai_cfg, topic, reply, env=live_env)
    assert response.stop_reason == "incomplete:max_output_tokens"
    assert response.stop_reason != "stop", "the controller rejects it, spending the attempt"


# --- what gets recorded -------------------------------------------------------


def test_the_provider_metadata_records_the_whole_non_secret_account(oai_cfg, topic, live_env):
    response, _ = _send(oai_cfg, topic, _reply({"scenario_text": "A scenario."}),
                        env=live_env)
    meta = response.provider_meta
    assert meta["provider"] == "openai" and meta["api"] == "responses"
    assert meta["endpoint"] == "https://api.openai.com/v1/responses"
    assert meta["response_id"] == "resp_abc123"
    assert meta["request_id"] == "req_xyz789"
    assert meta["requested_model"] == "gpt-5.6-sol"
    assert meta["returned_model"] == "gpt-5.6-sol"
    assert meta["status"] == "completed"
    assert meta["store"] is False and meta["background"] is False
    assert meta["created_at"] == 1789000000


def test_usage_is_recorded_including_cached_and_reasoning_token_details(oai_cfg, topic,
                                                                       live_env):
    response, _ = _send(oai_cfg, topic, _reply({"scenario_text": "A scenario."}),
                        env=live_env)
    assert response.usage["input_tokens"] == 1200
    assert response.usage["output_tokens"] == 180
    assert response.usage["input_tokens_details"]["cached_tokens"] == 1024
    assert response.usage["output_tokens_details"]["reasoning_tokens"] == 0


def test_a_more_specific_snapshot_is_flagged_for_approval_not_followed_silently(
        oai_cfg, topic, live_env):
    reply = _reply({"scenario_text": "A."}, model="gpt-5.6-sol-2026-08-01")
    response, _ = _send(oai_cfg, topic, reply, env=live_env)
    assert response.model_returned == "gpt-5.6-sol-2026-08-01"
    assert response.provider_meta["snapshot_more_specific_than_requested"] is True
    assert oai_cfg.raw["models"]["generator"]["model"]["pinned_snapshot"] is None, \
        "the configuration is not edited by a response"


def test_a_different_model_is_refused_rather_than_accepted(oai_cfg, topic, live_env):
    reply = _reply({"scenario_text": "A."}, model="gpt-4o-mini")
    with pytest.raises(OpenAIModelUnavailable, match="gpt-4o-mini"):
        _send(oai_cfg, topic, reply, env=live_env)


# --- the credential never escapes ---------------------------------------------


def test_the_key_is_sent_in_a_header_and_recorded_nowhere(oai_cfg, topic, live_env):
    response, transport = _send(oai_cfg, topic, _reply({"scenario_text": "A."}),
                                env=live_env)
    sent = transport.calls[0]
    assert sent["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in json.dumps(sent["body"]), "never in the request body"
    recorded = json.dumps({"payload": sent["body"], "meta": response.provider_meta,
                           "usage": response.usage, "raw": response.raw})
    assert FAKE_KEY not in recorded and "Bearer" not in recorded
    assert "authorization" not in json.dumps(sent["body"]).lower()


def test_the_recorded_payload_is_the_complete_body_minus_no_non_secret_field(
        oai_cfg, topic, live_env):
    _, transport = _send(oai_cfg, topic, _reply({"scenario_text": "A."}), env=live_env)
    assert transport.calls[0]["body"] == openai_responses_payload(
        _request(oai_cfg, topic), oai_cfg)


@pytest.mark.parametrize("code,kind", [(401, OpenAIAuthError), (403, OpenAIAuthError),
                                       (429, OpenAIRateLimited),
                                       (404, OpenAIModelUnavailable),
                                       (500, BackendError)])
def test_an_http_failure_is_typed_and_carries_no_credential(oai_cfg, topic, live_env,
                                                            code, kind):
    detail = json.dumps({"error": {"message": f"echoing your key {FAKE_KEY} back at you",
                                   "code": "model_not_found" if code == 404 else "other"}})
    error = urllib.error.HTTPError("https://api.openai.com/v1/responses", code, "err", {},
                                   io.BytesIO(detail.encode()))
    with pytest.raises(kind) as caught:
        _send(oai_cfg, topic, error, env=live_env)
    message = str(caught.value)
    assert FAKE_KEY not in message
    assert "[redacted]" in message
    assert str(code) in message


def test_a_transport_failure_says_nothing_was_retried(oai_cfg, topic, live_env):
    with pytest.raises(BackendUnavailable, match="no automatic retry"):
        _send(oai_cfg, topic, urllib.error.URLError("connection refused"), env=live_env)


def test_a_timeout_says_the_repeat_is_a_human_decision(oai_cfg, topic, live_env):
    with pytest.raises(BackendUnavailable, match="not retried"):
        _send(oai_cfg, topic, TimeoutError("timed out"), env=live_env)


@pytest.mark.parametrize("text,expected_absent", [
    (f"leaked {FAKE_KEY} here", FAKE_KEY),
    ("Authorization: Bearer abcdef0123456789", "abcdef0123456789"),
    ('{"api_key": "sk-livekeymaterial0000"}', "sk-livekeymaterial0000"),
])
def test_redaction_removes_anything_credential_shaped(text, expected_absent):
    cleaned = redact_secrets(text, secret=FAKE_KEY)
    assert expected_absent not in cleaned
    assert "[redacted]" in cleaned


def test_the_configuration_holds_only_the_name_of_the_variable(oai_cfg):
    api = oai_cfg.raw["models"]["generator"]["openai"]
    assert api["api_key_env_name"] == "OPENAI_API_KEY"
    body = OPENAI_CONFIG.read_text()
    assert "sk-" not in body and "Bearer " not in body


def test_a_credential_in_the_configuration_is_refused(tmp_path):
    import copy
    import yaml
    raw = copy.deepcopy(yaml.safe_load(OPENAI_CONFIG.read_text()))
    raw["models"]["generator"]["openai"]["note"] = "sk-averyrealsecretvalue"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="looks like a credential"):
        load_config(path)


# --- authorisation gates ------------------------------------------------------


def test_a_live_call_needs_send_and_the_openai_key(oai_cfg, topic):
    assert openai_authorization_problems(True, env={OPENAI_AUTHORIZATION_ENV: "1"}) == []
    assert "did not pass --send" in " ".join(
        openai_authorization_problems(False, env={OPENAI_AUTHORIZATION_ENV: "1"}))
    assert OPENAI_AUTHORIZATION_ENV in " ".join(
        openai_authorization_problems(True, env={}))


def test_the_local_key_does_not_authorise_a_paid_external_call(oai_cfg, topic):
    env = {"REASONSTYLE_ALLOW_LOCAL_GENERATION": "1", API_KEY_ENV: FAKE_KEY}
    transport = _Recorder(_reply({"scenario_text": "A."}))
    with pytest.raises(LiveCallRefused, match=OPENAI_AUTHORIZATION_ENV):
        OpenAIResponsesBackend(_urlopen=transport, _env=env).send(
            _request(oai_cfg, topic), oai_cfg, allow_live=True)
    assert transport.calls == [], "nothing was sent"


def test_without_send_nothing_is_sent_however_the_environment_reads(oai_cfg, topic, live_env):
    transport = _Recorder(_reply({"scenario_text": "A."}))
    with pytest.raises(LiveCallRefused, match="did not pass --send"):
        OpenAIResponsesBackend(_urlopen=transport, _env=live_env).send(
            _request(oai_cfg, topic), oai_cfg, allow_live=False)
    assert transport.calls == []


def test_an_absent_key_stops_the_call_without_naming_a_value(oai_cfg, topic):
    transport = _Recorder(_reply({"scenario_text": "A."}))
    with pytest.raises(BackendUnavailable) as caught:
        OpenAIResponsesBackend(_urlopen=transport,
                               _env={OPENAI_AUTHORIZATION_ENV: "1"}).send(
            _request(oai_cfg, topic), oai_cfg, allow_live=True)
    assert transport.calls == []
    assert API_KEY_ENV in str(caught.value)
    assert "never read, printed or stored" in str(caught.value)


def test_the_preflight_checks_presence_only(oai_cfg):
    assert openai_preflight_problems(oai_cfg, env={API_KEY_ENV: FAKE_KEY}) == []
    problems = openai_preflight_problems(oai_cfg, env={})
    assert problems and API_KEY_ENV in problems[0]
    assert FAKE_KEY not in " ".join(problems)
    assert openai_preflight_problems(oai_cfg, env={API_KEY_ENV: "   "}), "blank is not set"


def test_the_preflight_refuses_a_configuration_for_the_other_backend():
    qwen = load_config(QWEN_CONFIG)
    problems = openai_preflight_problems(qwen, env={API_KEY_ENV: FAKE_KEY})
    assert problems and "local_vllm_openai" in problems[0]


# --- the pilot CLI, in a subprocess with every socket poisoned ----------------

import importlib.util
import os
import subprocess
import sys

PILOT = ROOT / "scripts" / "pilot.py"
SMOKE = ROOT / "scripts" / "openai_smoke_test.py"

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
    path = tmp_path_factory.mktemp("no_network_openai")
    (path / "sitecustomize.py").write_text(NO_NETWORK)
    return path


def _run(script, *args, env=None, sitecustomize):
    environment = {k: v for k, v in os.environ.items() if k != API_KEY_ENV}
    environment.update(env or {})
    environment["PYTHONPATH"] = f"{sitecustomize}:{os.environ.get('PYTHONPATH', '')}"
    return subprocess.run([sys.executable, str(script), "--config",
                           "configs/experiment_openai_pilot.yaml", *args],
                          cwd=ROOT, capture_output=True, text=True, env=environment)


def _pilot_module():
    spec = importlib.util.spec_from_file_location("pilot_cli_openai", PILOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALL_PILOT_KEYS = {OPENAI_AUTHORIZATION_ENV: "1",
                  "REASONSTYLE_ALLOW_PILOT_GENERATION": "1"}


@pytest.mark.parametrize("command", ["plan", "status", "approvals"])
def test_no_openai_pilot_report_contacts_anything(command, tmp_path, no_network):
    result = _run(PILOT, command, "--out", str(tmp_path / "run"), "--send",
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  "--corrections-file", str(tmp_path / "corrections.yaml"),
                  "--scenario-corrections-file", str(tmp_path / "sc.yaml"),
                  env=ALL_PILOT_KEYS, sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "attempted to contact a backend" not in result.stderr
    assert not (tmp_path / "run" / "generation_log.jsonl").exists()


def test_a_dry_run_needs_no_key_and_contacts_nothing(tmp_path, no_network):
    result = _run(PILOT, "plan", "--out", str(tmp_path / "run"),
                  env={}, sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "DRY RUN: nothing was sent" in result.stdout
    assert "168 calls" in result.stdout, "24 scenarios + 48 groups x 3"
    assert "attempted to contact a backend" not in result.stderr


@pytest.mark.parametrize("command", ["scenarios", "groups"])
@pytest.mark.parametrize("missing", [OPENAI_AUTHORIZATION_ENV,
                                     "REASONSTYLE_ALLOW_PILOT_GENERATION"])
def test_an_openai_pilot_stage_needs_send_and_both_keys(command, missing, tmp_path,
                                                        no_network):
    env = {**ALL_PILOT_KEYS, missing: "", API_KEY_ENV: FAKE_KEY}
    result = _run(PILOT, command, "--out", str(tmp_path / "run"), "--send",
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  env=env, sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "nothing was sent" in result.stdout and missing in result.stdout
    assert not (tmp_path / "run" / "generation_log.jsonl").exists()
    assert "attempted to contact a backend" not in result.stderr


@pytest.mark.parametrize("command", ["scenarios", "groups"])
def test_an_openai_pilot_stage_without_send_sends_nothing(command, tmp_path, no_network):
    result = _run(PILOT, command, "--out", str(tmp_path / "run"),
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  env={**ALL_PILOT_KEYS, API_KEY_ENV: FAKE_KEY}, sitecustomize=no_network)
    assert result.returncode == 0
    assert "nothing was sent" in result.stdout
    assert not (tmp_path / "run").exists() or not (tmp_path / "run" / "raw").exists()


def test_the_local_generation_key_does_not_authorise_the_openai_pilot(tmp_path, no_network):
    result = _run(PILOT, "scenarios", "--out", str(tmp_path / "run"), "--send",
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  env={"REASONSTYLE_ALLOW_LOCAL_GENERATION": "1",
                       "REASONSTYLE_ALLOW_PILOT_GENERATION": "1",
                       "HF_HUB_OFFLINE": "1", API_KEY_ENV: FAKE_KEY},
                  sitecustomize=no_network)
    assert "nothing was sent" in result.stdout
    assert OPENAI_AUTHORIZATION_ENV in result.stdout
    assert not (tmp_path / "run" / "generation_log.jsonl").exists()


def test_an_authorised_stage_stops_at_the_preflight_without_a_key(tmp_path, no_network):
    """Authorised to spend money, but nothing to authenticate with: it stops
    before the first call, and no call artefact exists."""
    result = _run(PILOT, "scenarios", "--out", str(tmp_path / "run"), "--send",
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  env=ALL_PILOT_KEYS, sitecustomize=no_network)
    assert result.returncode == 1
    assert "no call was made" in result.stderr
    assert API_KEY_ENV in result.stderr
    assert "nothing here reads, prints or stores its value" in result.stderr
    assert not (tmp_path / "run" / "generation_log.jsonl").exists()
    assert "attempted to contact a backend" not in result.stderr


def test_the_two_stages_stay_separate_under_the_openai_config(tmp_path, no_network):
    """Scenario drafting is still followed by human approval before groups."""
    groups = _run(PILOT, "groups", "--out", str(tmp_path / "run"),
                  "--approvals-file", str(tmp_path / "approvals.yaml"),
                  env=ALL_PILOT_KEYS, sitecustomize=no_network)
    assert "24 scenario(s) would block group drafting" in groups.stdout
    module = _pilot_module()
    source = PILOT.read_text()
    body = source[source.index("def _stage("):]
    assert body.count("run_scenario_stage(") == 1 and body.count("run_group_stage(") == 1
    assert "group-review" not in str(module.WHOLE_PILOT_COMMANDS)


# --- the two generators' run directories never mix ---------------------------


def test_the_two_backends_have_separate_default_run_directories():
    module = _pilot_module()
    assert module.RUN_DIRECTORIES == {"local_vllm_openai": "data/pilot/run",
                                      OPENAI_BACKEND: "data/pilot/run_openai"}
    assert len(set(module.RUN_DIRECTORIES.values())) == 2


def test_the_openai_config_defaults_to_its_own_run_directory(tmp_path, no_network):
    result = _run(PILOT, "groups", "--approvals-file", str(tmp_path / "approvals.yaml"),
                  env={}, sitecustomize=no_network)
    assert "data/pilot/run_openai" in result.stdout
    assert "out          data/pilot/run\n" not in result.stdout


def test_an_openai_stage_refuses_the_qwen_run_directory(oai_cfg, tmp_path):
    from reasonstyle.generation import CallStore
    module = _pilot_module()
    store = CallStore(ROOT / module.RUN_DIRECTORIES["local_vllm_openai"], oai_cfg)
    problems = module.run_directory_problems(store, oai_cfg)
    assert problems and "local_vllm_openai" in problems[0]
    assert "data/pilot/run_openai" in problems[0]


def test_a_run_directory_holding_another_generators_calls_is_refused(oai_cfg, tmp_path):
    from reasonstyle.generation import CallStore
    module = _pilot_module()
    run = tmp_path / "mixed"
    run.mkdir()
    (run / "generation_log.jsonl").write_text(json.dumps({
        "call_id": "a" * 64, "kind": "scenario", "attempt": 1, "decision_id": "climate_01",
        "variant_id": 1, "supported_option": None, "status": "ok", "outcome": "accepted",
        "prompt_sha256": "b" * 64,
        "runtime": {"backend": "local_vllm_openai"}}) + "\n")
    problems = module.run_directory_problems(CallStore(run, oai_cfg), oai_cfg)
    assert problems and "local_vllm_openai" in problems[-1]
    assert "never appends to it" in problems[-1]


def test_a_clean_openai_run_directory_is_accepted(oai_cfg, tmp_path):
    from reasonstyle.generation import CallStore
    module = _pilot_module()
    assert module.run_directory_problems(CallStore(tmp_path / "fresh", oai_cfg), oai_cfg) == []


def test_the_qwen_run_and_snapshot_are_not_reachable_from_the_openai_defaults():
    module = _pilot_module()
    qwen = pathlib.Path(module.RUN_DIRECTORIES["local_vllm_openai"])
    hosted = pathlib.Path(module.RUN_DIRECTORIES[OPENAI_BACKEND])
    assert qwen != hosted and hosted not in qwen.parents and qwen not in hosted.parents
    gitignore = (ROOT / ".gitignore").read_text()
    assert "data/pilot/run_openai/" in gitignore
    assert "data/pilot/smoke_openai/" in gitignore


# --- the one-call smoke -------------------------------------------------------


def test_the_openai_smoke_dry_run_needs_no_key_and_charges_nothing(tmp_path, no_network):
    result = _run(SMOKE, "--out", str(tmp_path / "smoke"), env={}, sitecustomize=no_network)
    assert result.returncode == 0, result.stderr
    assert "DRY RUN: no key was read, no connection was made and nothing was charged." \
        in result.stdout
    assert "ceiling      1 call" in result.stdout
    assert "SYNTHETIC energy_fixture_001" in result.stdout
    assert "attempted to contact a backend" not in result.stderr


def test_the_openai_smoke_refuses_a_pilot_run_directory(tmp_path, no_network):
    result = _run(SMOKE, "--out", "data/pilot/run_openai", env={}, sitecustomize=no_network)
    assert result.returncode == 1
    assert "not corpus evidence" in result.stderr


def test_the_openai_smoke_cannot_become_a_pilot_stage():
    source = SMOKE.read_text()
    assert "REASONSTYLE_ALLOW_PILOT_GENERATION" not in source
    assert "run_pilot" not in source and "run_scenario_stage" not in source
    assert "run_group_stage" not in source and "repair_request" not in source
    assert 'default="data/pilot/smoke_openai"' in source


def test_the_openai_smoke_names_only_the_fixture_decision():
    source = SMOKE.read_text()
    assert "energy_fixture_001" in source
    bank = load_topic_bank(ROOT / "data" / "topics" / "pilot_topics.yaml")
    for pilot_decision in (t.decision_id for t in bank.topics):
        assert pilot_decision not in source, f"a pilot brief is named: {pilot_decision}"


# --- nothing about the Qwen configuration or its rules moved -----------------


def test_the_qwen_configuration_hash_has_not_moved():
    """Every one of the 24 scenario approvals binds it."""
    assert load_config(QWEN_CONFIG).content_hash == \
        "9da99ff12674cc91f0bbf76e137bf1cb347c30eb49691cfcfe3e4914c2fa148f"


def test_the_two_configurations_differ_only_in_the_generator(oai_cfg):
    """A second generator is one block. If anything else drifts, the two pilots
    stop being comparable and this test is where that shows up."""
    import copy
    qwen = load_config(QWEN_CONFIG)
    a, b = copy.deepcopy(qwen.raw), copy.deepcopy(oai_cfg.raw)
    for raw in (a, b):
        del raw["models"]["generator"]
        del raw["config_version"]
    assert a == b
    assert qwen.content_hash != oai_cfg.content_hash, "a different generator, a different hash"


def test_the_openai_pilot_keeps_every_corpus_rule(oai_cfg):
    qwen = load_config(QWEN_CONFIG)
    assert oai_cfg.raw["corpus"] == qwen.raw["corpus"]
    assert oai_cfg.raw["matching"] == qwen.raw["matching"]
    assert oai_cfg.raw["markers"] == qwen.raw["markers"]
    assert oai_cfg.raw["forbidden"] == qwen.raw["forbidden"]
    assert oai_cfg.raw["prompts"] == qwen.raw["prompts"]
    assert oai_cfg.raw["corpus"]["repair"]["max_calls_per_group"] == 3
    assert oai_cfg.parsed.matching.words.ratio_fail == qwen.parsed.matching.words.ratio_fail
    assert oai_cfg.parsed.matching.words.ratio_warn == qwen.parsed.matching.words.ratio_warn


def test_the_same_marker_allocation_serves_both_configurations(oai_cfg):
    """The allocation hash excludes the config hash by design, so one fixed
    allocation covers both pilots and the 48 groups keep their markers."""
    from reasonstyle.generation import allocate_markers, allocation_problems, load_allocation
    stored = load_allocation(ROOT / "data" / "pilot" / "marker_allocation.yaml")
    bank = load_topic_bank(ROOT / "data" / "topics" / "pilot_topics.yaml")
    rebuilt = allocate_markers(bank, oai_cfg)
    assert rebuilt.content_hash == stored.content_hash
    assert allocation_problems(rebuilt, oai_cfg) == []


def test_the_hosted_run_record_claims_no_reproducibility(oai_cfg):
    from reasonstyle.generation import describe_run
    env = describe_run(oai_cfg, cached=None,
                       endpoint="https://api.openai.com/v1/responses").as_dict()
    assert env["backend"] == OPENAI_BACKEND
    assert env["repo_id"] == "gpt-5.6-sol"
    assert env["revision"] is None and env["snapshot_path"] is None
    assert env["seed"] is None and env["gpu"] is None
    assert env["provider"]["store"] is False and env["provider"]["background"] is False
    assert env["provider"]["api_key_env_name"] == "OPENAI_API_KEY"
    assert "nothing here claims reproducibility" in env["reproducibility_note"]
    assert FAKE_KEY not in json.dumps(env)


def test_each_generator_gets_its_own_approvals_and_correction_ledgers():
    """A Qwen approval binds the Qwen config hash and a Qwen call id, so it can
    never apply to a run drafted by another generator. Separate files make the
    attempt impossible rather than merely wrong."""
    module = _pilot_module()
    qwen = module.PROFILE_PATHS["local_vllm_openai"]
    hosted = module.PROFILE_PATHS[OPENAI_BACKEND]
    assert set(qwen) == set(hosted)
    for name in qwen:
        assert qwen[name] != hosted[name], name
    assert qwen["approvals_file"] == "data/pilot/scenario_approvals.yaml"
    assert hosted["approvals_file"] == "data/pilot/scenario_approvals_openai.yaml"
    assert hosted["corrections_file"] == "data/pilot/manual_corrections_openai.yaml"


def test_no_counterargument_correction_ledger_exists_for_either_generator():
    """Nothing in this codebase writes a correction; a correction is a separate,
    separately approved human act, and neither pilot has reached one.

    A scenario *approvals* file is a different artefact and is deliberately not
    listed here: recording a curator's review is exactly what that file is for,
    and one exists for each generator whose scenarios have been read.
    """
    for ledger in ("data/pilot/manual_corrections.yaml",
                   "data/pilot/manual_corrections_openai.yaml"):
        assert not (ROOT / ledger).exists(), f"{ledger} must not be populated here"


def test_the_openai_pilot_reports_run_against_an_empty_run_directory(tmp_path, no_network):
    """Every reporting command works before a single call exists."""
    for command in ("status", "approvals", "log", "group-review"):
        result = _run(PILOT, command, "--out", str(tmp_path / "run"),
                      "--approvals-file", str(tmp_path / "approvals.yaml"),
                      "--corrections-file", str(tmp_path / "corrections.yaml"),
                      "--scenario-corrections-file", str(tmp_path / "sc.yaml"),
                      "--group-review-out", str(tmp_path / "review"),
                      env={}, sitecustomize=no_network)
        assert result.returncode == 0, (command, result.stderr)
        assert "attempted to contact a backend" not in result.stderr
