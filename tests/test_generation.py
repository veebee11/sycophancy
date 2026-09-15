"""Drafting requests, responses, logging, provenance and the backends.

No test reaches a network or a GPU. The vLLM backend is exercised through its
payload builder and a stub opener, and the refusal paths are tested directly:
"it refuses unless authorised", "a missing model is an error, not a download"
and "a failure is never retried here" are the properties that matter most.
"""

from __future__ import annotations

import copy
import io
import json
import os
import pathlib
import urllib.error

import pytest

from reasonstyle.config import ConfigError, load_config
from reasonstyle.corpus.topics import TopicBank
from reasonstyle.generation import (
    AUTHORIZATION_ENV,
    BackendError,
    FakeBackend,
    GenerationLog,
    LiveCallRefused,
    ModelNotCached,
    RequestError,
    ResponseRejected,
    VLLMOpenAIBackend,
    allocate_markers,
    authorization_problems,
    check_request_provenance,
    describe_run,
    group_request,
    load_server_runtime,
    offline_problems,
    parse_response,
    repair_request,
    resolve_cached_model,
    revision_agreement,
    scenario_request,
    vllm_payload,
    write_request,
)
from reasonstyle.generation.log import LogEntry
from reasonstyle.hashing import content_hash

SCENARIO_TEXT = ("A regional grid operator must decide how to cover a shortfall in firm "
                 "capacity. The extended plant can deliver full output through any cold "
                 "spell. Retiring the plant on schedule would cut emissions.")


@pytest.fixture
def topic(synthetic_bank):
    return next(t for t in synthetic_bank.topics if t.decision_id == "energy_fixture_001")


@pytest.fixture
def allocation(cfg, pilot_bank):
    return allocate_markers(pilot_bank, cfg)


# --- templates --------------------------------------------------------------


def test_scenario_request_carries_only_the_brief(topic, cfg):
    request = scenario_request(topic, 1, cfg)
    assert topic.decision_framing in request.prompt
    assert topic.variants["v1"].scenario_facts.opt_1[0] in request.prompt
    assert topic.variants["v1"].scenario_facts.opt_2[0] in request.prompt
    # the band the curator approved, not a number invented in code
    band = cfg.raw["corpus"]["scenario_words"]
    assert f"{band['min']} to {band['max']} words" in request.prompt
    assert "${" not in request.prompt


def test_an_uncurated_brief_is_never_drafted(synthetic_bank, cfg):
    proposed = next(t for t in synthetic_bank.topics if t.status == "proposed")
    with pytest.raises(RequestError, match="not curated"):
        scenario_request(proposed, 1, cfg)


def test_group_request_states_the_marker_and_its_realization(topic, cfg, allocation):
    group = allocation.groups[0]
    group = type(group)(**{**group.as_dict(), "decision_id": topic.decision_id,
                           "variant_id": 1, "scenario_id": f"{topic.decision_id}_v1"})
    request = group_request(topic, 1, SCENARIO_TEXT, group, cfg)
    description = cfg.raw["markers"]["realization"]["registry"][
        group.marker_realization_id]["description"]
    assert f'"{group.marker_string}"' in request.prompt
    assert description in request.prompt
    assert cfg.raw["corpus"]["counterargument_opening"] in request.prompt
    assert str(cfg.raw["corpus"]["body_sentences"]) in request.prompt


def test_group_prompt_defines_no_reason_as_no_task_relevant_reason(topic, cfg, allocation):
    """A literal ban on every clause would forbid the clause that hosts the
    marker; what is banned is task-relevant support."""
    group = allocation.groups[0]
    group = type(group)(**{**group.as_dict(), "decision_id": topic.decision_id,
                           "variant_id": 1, "scenario_id": f"{topic.decision_id}_v1"})
    prompt = group_request(topic, 1, SCENARIO_TEXT, group, cfg).prompt
    assert "no task-relevant reason" in prompt
    assert "self-referential clause is allowed" in prompt
    for banned in ("value, goal, priority or trade-off", "new factual claim",
                   "consequence, effect or outcome"):
        assert banned in prompt


def test_a_changed_template_file_fails_to_load(cfg, tmp_path):
    """The hash is what fixes the prompt; the file is only where it lives."""
    raw = copy.deepcopy(cfg.raw)
    raw["prompts"]["drafting"]["repair_v1"]["template_sha256"] = "0" * 64
    path = tmp_path / "configs" / "experiment.yaml"
    path.parent.mkdir(parents=True)
    import yaml
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    (tmp_path / "prompts").symlink_to(cfg.path.parent.parent / "prompts")
    with pytest.raises(ConfigError, match="recorded hash"):
        load_config(path)


# --- responses --------------------------------------------------------------


def test_a_well_formed_response_is_accepted(topic, cfg):
    request = scenario_request(topic, 1, cfg)
    assert parse_response(request, {"scenario_text": " text "}) == {"scenario_text": "text"}
    assert parse_response(request, '{"scenario_text": "text"}') == {"scenario_text": "text"}


@pytest.mark.parametrize("payload, message", [
    ({}, "missing"),
    ({"scenario_text": "t", "note": "extra"}, "unexpected"),
    ({"scenario_text": ""}, "empty"),
    ({"scenario_text": 3}, "must be a string"),
    ("not json at all", "not JSON"),
    ('["a list"]', "must be a JSON object"),
])
def test_a_malformed_response_is_rejected_not_repaired(topic, cfg, payload, message):
    request = scenario_request(topic, 1, cfg)
    with pytest.raises(ResponseRejected, match=message):
        parse_response(request, payload)


def test_group_response_must_carry_all_four_conditions(topic, cfg, allocation):
    group = allocation.groups[0]
    group = type(group)(**{**group.as_dict(), "decision_id": topic.decision_id,
                           "variant_id": 1, "scenario_id": f"{topic.decision_id}_v1"})
    request = group_request(topic, 1, SCENARIO_TEXT, group, cfg)
    with pytest.raises(ResponseRejected, match="missing"):
        parse_response(request, {"RS": "a", "RP": "b", "NS": "c"})


# --- repair budget ----------------------------------------------------------


def _group_for(topic, allocation, variant_id=1):
    g = allocation.groups[0]
    return type(g)(**{**g.as_dict(), "decision_id": topic.decision_id,
                      "variant_id": variant_id,
                      "scenario_id": f"{topic.decision_id}_v{variant_id}"})


BODIES = {"RS": "a b.", "RP": "c d.", "NS": "e f.", "NP": "g h."}


def test_a_repair_quotes_the_validator_findings(topic, cfg, allocation):
    request = repair_request(topic, 1, SCENARIO_TEXT, _group_for(topic, allocation),
                             BODIES, ["E_WORD_RATIO_BODY: ratio 1.31"], 2, cfg)
    assert "E_WORD_RATIO_BODY: ratio 1.31" in request.prompt
    assert request.attempt == 2


def test_a_repair_without_findings_is_refused(topic, cfg, allocation):
    with pytest.raises(RequestError, match="findings"):
        repair_request(topic, 1, SCENARIO_TEXT, _group_for(topic, allocation),
                       BODIES, [], 2, cfg)


def test_the_repair_budget_is_enforced(topic, cfg, allocation):
    """Three calls per group: one draft and two repairs. A fourth is refused,
    and the group is marked for manual review instead."""
    budget = cfg.raw["corpus"]["repair"]
    assert budget["max_calls_per_group"] == budget["max_repair_calls"] + 1
    repair_request(topic, 1, SCENARIO_TEXT, _group_for(topic, allocation), BODIES,
                   ["E_X"], budget["max_calls_per_group"], cfg)
    with pytest.raises(RequestError, match="budget"):
        repair_request(topic, 1, SCENARIO_TEXT, _group_for(topic, allocation), BODIES,
                       ["E_X"], budget["max_calls_per_group"] + 1, cfg)


def test_attempts_are_counted_from_the_log(tmp_path, cfg):
    log = GenerationLog(tmp_path / "log.jsonl")
    for attempt in (1, 2):
        log.append(LogEntry(
            call_id=f"c{attempt}", kind="group" if attempt == 1 else "repair", attempt=attempt,
            decision_id="energy_fixture_001", variant_id=1, supported_option="opt_1",
            template_name="t", template_sha256="d", prompt_sha256="p", model="m",
            model_returned=None, request_fields={}, config_content_hash="c",
            topic_bank_content_hash="b", allocation_content_hash="a", response_sha256=None,
            stop_reason=None, usage=None, status="ok", error=None, generated_at="now"))
    assert log.attempts_for("energy_fixture_001", 1, "opt_1") == 2
    assert log.attempts_for("energy_fixture_001", 2, "opt_1") == 0


# --- logging ----------------------------------------------------------------


def test_the_log_records_a_refusal_like_any_other_call(tmp_path):
    log = GenerationLog(tmp_path / "log.jsonl")
    log.append(LogEntry(
        call_id="c", kind="scenario", attempt=1, decision_id="energy_fixture_001", variant_id=1,
        supported_option=None, template_name="t", template_sha256="d", prompt_sha256="p",
        model="m", model_returned=None, request_fields={}, config_content_hash="c",
        topic_bank_content_hash="b", allocation_content_hash=None, response_sha256=None,
        stop_reason=None, usage=None, status="refused", error="no approval",
        generated_at="now"))
    entry, = log.entries()
    assert entry["status"] == "refused" and entry["error"] == "no approval"


def test_raw_traffic_is_stored_verbatim(tmp_path):
    log = GenerationLog(tmp_path / "log.jsonl")
    path = log.store_raw("abc", {"model": "m"}, {"content": [{"text": "hi"}]}, prompt="P")
    stored = json.loads(path.read_text())
    assert stored["prompt"] == "P" and stored["response"] == {"content": [{"text": "hi"}]}


def test_a_request_file_can_be_read_before_anything_is_sent(topic, cfg, tmp_path):
    request = scenario_request(topic, 1, cfg)
    path = write_request(request, tmp_path)
    stored = json.loads(path.read_text())
    assert stored["prompt"] == request.prompt
    assert stored["prompt_sha256"] == request.prompt_sha256
    assert stored["response_schema"]["required"] == ["scenario_text"]


# --- backends ---------------------------------------------------------------


def test_the_payload_is_built_from_the_configuration(topic, cfg):
    request = scenario_request(topic, 1, cfg)
    payload = vllm_payload(request, cfg)
    dec = cfg.raw["models"]["generator"]["decoding"]
    assert payload["model"] == cfg.raw["models"]["generator"]["model"]["repo_id"]
    assert payload["messages"] == [{"role": "user", "content": request.prompt}]
    for field in ("temperature", "top_p", "max_tokens", "seed", "n"):
        assert payload[field] == dec[field]
    # non-thinking mode is a chat-template argument, not a decoding parameter
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_every_sampling_field_is_sent_explicitly_at_its_neutral_value(topic, cfg):
    """Nothing that shapes sampling is left to a server default."""
    payload = vllm_payload(scenario_request(topic, 1, cfg), cfg)
    assert {k: payload[k] for k in ("top_k", "min_p", "repetition_penalty",
                                    "presence_penalty", "frequency_penalty")} == {
        "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
        "presence_penalty": 0.0, "frequency_penalty": 0.0}
    dec = cfg.raw["models"]["generator"]["decoding"]
    sampling = {k for k in payload if k not in
                ("model", "messages", "response_format", "chat_template_kwargs")}
    assert sampling == set(dec) - {"thinking"}, "every decoding field is sent, and only those"


def test_the_payload_constrains_the_output_to_the_response_schema(topic, cfg):
    request = scenario_request(topic, 1, cfg)
    payload = vllm_payload(request, cfg)
    assert payload["response_format"]["json_schema"]["schema"] == request.response_schema
    assert payload["response_format"]["json_schema"]["strict"] is True


def test_no_credential_is_ever_sent_or_configured(topic, cfg):
    """The generator is a local model: there is nothing to authenticate to."""
    payload = vllm_payload(scenario_request(topic, 1, cfg), cfg)
    serialized = json.dumps(payload).casefold()
    for word in ("api_key", "authorization", "x-api-key", "bearer", "credential"):
        assert word not in serialized
    # "max_tokens" is a decoding field, not a credential — so check the keys
    assert not [k for k in payload if "auth" in k or k.endswith("_key")]
    assert not (set(cfg.raw["models"]["generator"]) & {"api_key", "api_key_env", "token", "auth"})


def test_the_endpoint_is_local_only(cfg):
    url = cfg.raw["models"]["generator"]["vllm"]["base_url"]
    assert url.startswith("http://127.0.0.1")


def test_a_run_needs_both_the_argument_and_the_environment_variable(topic, cfg, monkeypatch):
    """Authorisation deliberately does not live in the experiment config, so
    reading or sharing the config cannot by itself enable a run."""
    monkeypatch.setenv(AUTHORIZATION_ENV, "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert authorization_problems(True) == []
    assert "did not pass --send" in " ".join(authorization_problems(False))

    monkeypatch.delenv(AUTHORIZATION_ENV)
    assert AUTHORIZATION_ENV in " ".join(authorization_problems(True))
    with pytest.raises(LiveCallRefused, match=AUTHORIZATION_ENV):
        VLLMOpenAIBackend().send(scenario_request(topic, 1, cfg), cfg, allow_live=True)


def test_a_run_without_offline_mode_is_refused(topic, cfg, monkeypatch):
    """Otherwise a missing model becomes an unattended multi-gigabyte download."""
    monkeypatch.setenv(AUTHORIZATION_ENV, "1")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert offline_problems(cfg)
    with pytest.raises(LiveCallRefused, match="HF_HUB_OFFLINE"):
        VLLMOpenAIBackend().send(scenario_request(topic, 1, cfg), cfg, allow_live=True)


class _Stub:
    """Stands in for urlopen: records what was sent, returns a canned body."""

    def __init__(self, body):
        self._body = json.dumps(body).encode()
        self.sent = None
        self.headers = None

    def __call__(self, request, timeout=None):
        self.sent = json.loads(request.data.decode()) if request.data else None
        self.headers = request.headers
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _completion(content):
    return {"model": "Qwen/Qwen3-14B",
            "usage": {"prompt_tokens": 800, "completion_tokens": 120},
            "choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(content)}}]}


def test_a_structured_response_is_read_back(topic, cfg, monkeypatch):
    monkeypatch.setenv(AUTHORIZATION_ENV, "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    stub = _Stub(_completion({"scenario_text": "a drafted scenario"}))
    request = scenario_request(topic, 1, cfg)
    response = VLLMOpenAIBackend(_urlopen=stub).send(request, cfg, allow_live=True)

    assert parse_response(request, response.content) == {"scenario_text": "a drafted scenario"}
    assert response.model_returned == "Qwen/Qwen3-14B"
    assert response.usage["completion_tokens"] == 120
    assert stub.sent["seed"] == cfg.raw["models"]["generator"]["decoding"]["seed"]
    assert "authorization" not in {k.casefold() for k in stub.headers}


def test_prose_instead_of_the_object_is_an_error_not_a_retry(topic, cfg, monkeypatch):
    monkeypatch.setenv(AUTHORIZATION_ENV, "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    stub = _Stub({"model": "Qwen/Qwen3-14B",
                  "choices": [{"finish_reason": "stop",
                               "message": {"content": "Here is the scenario you asked for."}}]})
    backend = VLLMOpenAIBackend(_urlopen=stub)
    with pytest.raises(BackendError, match="not the requested JSON object"):
        backend.send(scenario_request(topic, 1, cfg), cfg, allow_live=True)


def test_an_empty_response_is_an_error(topic, cfg, monkeypatch):
    monkeypatch.setenv(AUTHORIZATION_ENV, "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    stub = _Stub({"model": "m", "choices": [{"finish_reason": "length",
                                             "message": {"content": ""}}]})
    with pytest.raises(BackendError, match="no content"):
        VLLMOpenAIBackend(_urlopen=stub).send(scenario_request(topic, 1, cfg), cfg,
                                              allow_live=True)


def test_a_failed_call_is_never_retried_by_the_backend(topic, cfg, monkeypatch):
    """Whether to try again is a decision made above the backend, under the
    repair budget — never inside it."""
    monkeypatch.setenv(AUTHORIZATION_ENV, "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    calls = []

    def counting(request, timeout=None):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 500, "boom", {}, io.BytesIO(b"{}"))

    with pytest.raises(BackendError):
        VLLMOpenAIBackend(_urlopen=counting).send(scenario_request(topic, 1, cfg), cfg,
                                                  allow_live=True)
    assert len(calls) == 1


def test_the_fake_backend_records_what_it_was_asked(topic, cfg):
    backend = FakeBackend(lambda r: {"scenario_text": "a synthetic scenario"})
    request = scenario_request(topic, 1, cfg)
    response = backend.send(request, cfg)
    assert backend.calls == [request]
    assert parse_response(request, response.content)["scenario_text"] == "a synthetic scenario"


# --- the environment record --------------------------------------------------


COMPLETE = ("config.json", "tokenizer_config.json", "model.safetensors")


def _cache(tmp_path, repo_id="Qwen/Qwen3-14B", sha="a" * 40, files=COMPLETE, ref="main",
           index=None):
    repo = tmp_path / "hub" / ("models--" + repo_id.replace("/", "--"))
    snapshot = repo / "snapshots" / sha
    snapshot.mkdir(parents=True)
    for name in files:
        (snapshot / name).write_text("{}")
    if index is not None:
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {f"layer.{i}": name for i, name in enumerate(index)}}))
    if ref:
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / ref).write_text(sha)
    return tmp_path


def test_a_cached_model_resolves_to_a_commit(tmp_path):
    home = _cache(tmp_path)
    cached = resolve_cached_model("Qwen/Qwen3-14B", hf_home=home)
    assert cached.revision == "a" * 40
    assert cached.refs == ("main",)
    assert cached.snapshot_path.endswith("a" * 40)


def test_an_absent_model_is_reported_not_downloaded(tmp_path):
    (tmp_path / "hub").mkdir()
    with pytest.raises(ModelNotCached, match="not in the cache"):
        resolve_cached_model("Qwen/Qwen3-14B", hf_home=tmp_path)


def test_a_snapshot_without_its_tokenizer_is_refused(tmp_path):
    home = _cache(tmp_path, files=("config.json",))
    with pytest.raises(ModelNotCached, match="incomplete"):
        resolve_cached_model("Qwen/Qwen3-14B", hf_home=home)


def test_metadata_without_weights_is_refused(tmp_path):
    """The shape of an interrupted download: config and tokenizer present, no
    weights at all."""
    home = _cache(tmp_path, files=("config.json", "tokenizer_config.json"))
    with pytest.raises(ModelNotCached, match="not its weights"):
        resolve_cached_model("Qwen/Qwen3-14B", hf_home=home)


def test_every_shard_named_by_the_index_must_be_present(tmp_path):
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    home = _cache(tmp_path, files=("config.json", "tokenizer_config.json", shards[0]),
                  index=shards)
    with pytest.raises(ModelNotCached, match="not its weights"):
        resolve_cached_model("Qwen/Qwen3-14B", hf_home=home)

    complete = _cache(tmp_path / "complete",
                      files=("config.json", "tokenizer_config.json", *shards), index=shards)
    cached = resolve_cached_model("Qwen/Qwen3-14B", hf_home=complete)
    assert set(cached.weight_files) == set(shards)


def test_an_empty_weight_file_does_not_count_as_present(tmp_path):
    home = _cache(tmp_path, files=("config.json", "tokenizer_config.json"))
    snapshot = next((home / "hub").glob("**/snapshots/*"))
    (snapshot / "model.safetensors").write_text("")      # a zero-byte stub
    with pytest.raises(ModelNotCached, match="not its weights"):
        resolve_cached_model("Qwen/Qwen3-14B", hf_home=home)


def test_the_three_revisions_must_agree(tmp_path):
    cached = resolve_cached_model("Qwen/Qwen3-14B", hf_home=_cache(tmp_path))
    assert revision_agreement("a" * 40, cached, "a" * 40) == []
    assert "not recorded" in " ".join(revision_agreement(None, cached))
    assert "configuration pins" in " ".join(revision_agreement("b" * 40, cached))
    assert "server reports" in " ".join(revision_agreement("a" * 40, cached, "c" * 40))


SERVER_RUNTIME = {
    "host": "chomusuke02", "repo_id": "Qwen/Qwen3-14B", "revision": "a" * 40,
    "gpu_index": "2", "gpu_name": "NVIDIA RTX A6000", "dtype": "bfloat16",
    "max_model_len": 8192, "seed": 20260914, "generation_config": "vllm",
    "libraries": {"vllm": "0.8.5", "transformers": "4.51.0", "torch": "2.6.0"},
}


def test_the_environment_record_carries_what_a_thesis_must_report(cfg, tmp_path, monkeypatch):
    """GPU and dtype come from the SERVER's record, never from this shell: the
    client may be a laptop on another host."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")      # a client-side red herring
    cached = resolve_cached_model("Qwen/Qwen3-14B", hf_home=_cache(tmp_path))
    env = describe_run(cfg, cached=cached, endpoint="http://127.0.0.1:8011/v1",
                       server=SERVER_RUNTIME,
                       prompt_sha256="p" * 64, response_sha256="r" * 64,
                       input_tokens=800, output_tokens=120)
    record = env.as_dict()
    for key in ("repo_id", "revision", "backend", "decoding", "seed", "gpu", "dtype",
                "libraries", "server", "prompt_sha256", "response_sha256",
                "input_tokens", "output_tokens"):
        assert record[key] is not None, key
    assert record["gpu"] == "NVIDIA RTX A6000"
    assert record["server"]["gpu_index"] == "2"
    assert record["server"]["libraries"]["vllm"] == "0.8.5"
    assert "cuda_visible_devices" not in record
    assert record["decoding"]["temperature"] == 0.3
    # the seed is recorded, and explicitly not claimed to guarantee reproduction
    assert "does not guarantee bit-for-bit" in record["reproducibility_note"]


def test_a_missing_server_record_is_an_error_not_an_inference(tmp_path):
    with pytest.raises(FileNotFoundError, match="serve_vllm.sh"):
        load_server_runtime(tmp_path / "server_runtime.json")


def test_a_generator_from_an_evaluated_family_is_refused(cfg, tmp_path):
    """Llama is one of the families under evaluation; it cannot also draft the
    corpus."""
    import yaml
    raw = copy.deepcopy(cfg.raw)
    raw["models"]["generator"]["model"]["repo_id"] = "meta-llama/Llama-3.1-8B-Instruct"
    path = tmp_path / "configs" / "experiment.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="evaluated family"):
        load_config(path)


def test_run_authorisation_is_not_a_config_key(cfg, tmp_path):
    import yaml
    raw = copy.deepcopy(cfg.raw)
    raw["models"]["generator"]["live_calls_enabled"] = True
    path = tmp_path / "configs" / "experiment.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="authorisation"):
        load_config(path)


# --- provenance -------------------------------------------------------------


def _entry(request, cfg, bank, allocation=None, **over):
    base = dict(
        call_id=request.call_id, kind=request.kind, attempt=request.attempt,
        decision_id=request.decision_id, variant_id=request.variant_id,
        supported_option=request.supported_option, template_name=request.template_name,
        template_sha256=request.template_sha256, prompt_sha256=request.prompt_sha256,
        config_content_hash=cfg.content_hash,
        topic_bank_content_hash=content_hash(bank.model_dump(mode="json")),
        allocation_content_hash=allocation.content_hash if allocation else None)
    return {**base, **over}


def test_provenance_matches_when_the_inputs_are_unchanged(cfg, pilot_bank, allocation):
    topic = next(t for t in pilot_bank.topics if t.status == "curated")
    request = scenario_request(topic, 1, cfg)
    result = check_request_provenance(_entry(request, cfg, pilot_bank), pilot_bank,
                                      allocation, cfg)
    assert result.matched and result.problems == ()


def test_provenance_catches_a_brief_revised_after_drafting(cfg, pilot_bank, allocation):
    topic = next(t for t in pilot_bank.topics if t.status == "curated")
    request = scenario_request(topic, 1, cfg)
    entry = _entry(request, cfg, pilot_bank)

    revised = pilot_bank.model_dump(mode="json")
    for t in revised["topics"]:
        if t["decision_id"] == topic.decision_id:
            t["decision_framing"] = "A different decision entirely."
    revised_bank = TopicBank.model_validate(revised)

    result = check_request_provenance(entry, revised_bank, allocation, cfg)
    assert not result.matched
    assert any("topic bank hash" in p for p in result.problems)


def test_provenance_catches_a_prompt_that_was_not_built_from_the_brief(
        cfg, pilot_bank, allocation):
    topic = next(t for t in pilot_bank.topics if t.status == "curated")
    request = scenario_request(topic, 1, cfg)
    entry = _entry(request, cfg, pilot_bank, prompt_sha256="0" * 64)
    result = check_request_provenance(entry, pilot_bank, allocation, cfg)
    assert not result.matched
    assert any("rebuilt prompt differs" in p for p in result.problems)


def test_group_provenance_needs_the_scenario_it_was_given(cfg, pilot_bank, allocation):
    topic = next(t for t in pilot_bank.topics if t.status == "curated")
    group = allocation.for_group(f"{topic.decision_id}_v1", "opt_1")
    request = group_request(topic, 1, SCENARIO_TEXT, group, cfg)
    entry = _entry(request, cfg, pilot_bank, allocation)

    good = check_request_provenance(entry, pilot_bank, allocation, cfg,
                                    scenario_text=SCENARIO_TEXT)
    assert good.matched
    bad = check_request_provenance(entry, pilot_bank, allocation, cfg,
                                   scenario_text=SCENARIO_TEXT + " And one more claim.")
    assert not bad.matched


# --- the smoke-test script ---------------------------------------------------


def _run_smoke(tmp_path, *args, env=None):
    import subprocess
    import sys as _sys
    root = pathlib.Path(__file__).resolve().parents[1]
    environment = {**os.environ, "HF_HOME": str(tmp_path), **(env or {})}
    return subprocess.run(
        [_sys.executable, "scripts/smoke_test.py", "--config", "configs/experiment.yaml",
         "--out", str(tmp_path / "smoke"), *args],
        cwd=root, env=environment, capture_output=True, text=True)


def test_the_smoke_test_sends_nothing_without_both_authorisations(tmp_path):
    result = _run_smoke(tmp_path, "--send", env={AUTHORIZATION_ENV: ""})
    assert result.returncode == 0
    assert "nothing was sent" in result.stdout
    assert AUTHORIZATION_ENV in result.stdout
    assert not (tmp_path / "smoke" / "generation_log.jsonl").exists()


def test_the_smoke_test_refuses_when_the_revision_cannot_be_resolved(tmp_path):
    """Authorised, offline, but the weights are not cached: it stops rather
    than running a model whose revision could not be recorded."""
    (tmp_path / "hub").mkdir()
    result = _run_smoke(tmp_path, "--send",
                        env={AUTHORIZATION_ENV: "1", "HF_HUB_OFFLINE": "1"})
    assert result.returncode == 1
    assert "not resolvable in the local cache" in result.stderr
    assert not (tmp_path / "smoke" / "generation_log.jsonl").exists()


def test_the_smoke_test_uses_synthetic_material_only(tmp_path):
    """It must never draft a pilot brief: the id it names is the fixture's."""
    result = _run_smoke(tmp_path)
    assert "SYNTHETIC energy_fixture_001" in result.stdout
    for pilot_id in ("climate_01", "energy_02", "technology_04"):
        assert pilot_id not in result.stdout


def test_the_smoke_test_refuses_without_a_server_runtime_record(tmp_path):
    """A live run must record the GPU that loaded the weights, which only the
    launcher knows."""
    home = _cache(tmp_path)
    result = _run_smoke(tmp_path, "--send", "--server-runtime",
                        str(tmp_path / "absent.json"),
                        env={AUTHORIZATION_ENV: "1", "HF_HUB_OFFLINE": "1",
                             "HF_HOME": str(home)})
    assert result.returncode == 1
    assert "serve_vllm.sh" in result.stderr


def test_the_smoke_test_refuses_when_the_revisions_disagree(tmp_path):
    home = _cache(tmp_path)
    runtime = tmp_path / "server_runtime.json"
    runtime.write_text(json.dumps({**SERVER_RUNTIME, "revision": "b" * 40}))
    result = _run_smoke(tmp_path, "--send", "--server-runtime", str(runtime),
                        env={AUTHORIZATION_ENV: "1", "HF_HUB_OFFLINE": "1",
                             "HF_HOME": str(home)})
    assert result.returncode == 1
    assert "same commit" in result.stderr


# --- the server launcher (read as text; never executed here) -----------------


LAUNCHER = pathlib.Path(__file__).resolve().parents[1] / "scripts/server/serve_vllm.sh"


def test_the_launcher_uses_the_project_virtualenv():
    """Not whatever "vllm" happens to be first on PATH, and no activation step."""
    text = LAUNCHER.read_text()
    assert 'VLLM="$VENV/bin/vllm"' in text and 'PYTHON="$VENV/bin/python"' in text
    assert 'exec "$VLLM" serve' in text


def test_the_launcher_pins_the_revision_it_was_given():
    text = LAUNCHER.read_text()
    assert "--require-config-match --print-revision" in text
    assert '--revision "$REVISION"' in text


def test_the_launcher_reads_the_same_cache_layout_as_the_preflight():
    """--download-dir would point vLLM at a flat directory instead of the hub
    cache, and the offline check would then prove nothing."""
    commands = [line for line in LAUNCHER.read_text().splitlines()
                if not line.lstrip().startswith("#")]
    text = "\n".join(commands)
    assert "--download-dir" not in text
    assert 'export HF_HOME=' in text and "HF_HUB_OFFLINE=1" in text
    assert 'SNAPSHOT="$HF_HOME/hub/models--' in text


def test_the_launcher_disables_the_models_own_generation_defaults():
    """--generation-config auto (vLLM's default) would let generation_config.json
    fill in top_k, temperature and top_p for any field a request omits."""
    commands = "\n".join(line for line in LAUNCHER.read_text().splitlines()
                         if not line.lstrip().startswith("#"))
    assert "GENERATION_CONFIG=vllm" in commands
    assert "GENERATION_CONFIG:-" not in commands, "a fixed choice, not an override"
    assert '--generation-config "$GENERATION_CONFIG"' in commands
    assert '"generation_config": "$GENERATION_CONFIG"' in commands     # recorded
    serve = commands[commands.index('exec "$VLLM" serve'):]
    assert "--generation-config" in serve


def test_a_server_without_the_neutral_generation_config_is_refused(cfg):
    from reasonstyle.generation import server_settings_problems
    assert server_settings_problems(cfg, SERVER_RUNTIME) == []
    for bad in ({**SERVER_RUNTIME, "generation_config": "auto"},
                {k: v for k, v in SERVER_RUNTIME.items() if k != "generation_config"}):
        problems = server_settings_problems(cfg, bad)
        assert problems and "--generation-config vllm" in problems[0]


def test_the_smoke_test_refuses_a_server_that_loaded_model_generation_defaults(tmp_path):
    home = _cache(tmp_path)
    runtime = tmp_path / "server_runtime.json"
    runtime.write_text(json.dumps({**SERVER_RUNTIME, "generation_config": "auto"}))
    result = _run_smoke(tmp_path, "--send", "--server-runtime", str(runtime),
                        env={AUTHORIZATION_ENV: "1", "HF_HUB_OFFLINE": "1",
                             "HF_HOME": str(home)})
    assert result.returncode == 1
    assert "generation_config='auto'" in result.stderr
    assert not (tmp_path / "smoke" / "generation_log.jsonl").exists()


def test_the_launcher_writes_a_runtime_record_and_binds_localhost():
    text = LAUNCHER.read_text()
    for field in ("gpu_index", "gpu_name", "dtype", "revision", "max_model_len",
                  "\"vllm\"", "\"transformers\"", "seed"):
        assert field in text
    assert "--host 127.0.0.1" in text
