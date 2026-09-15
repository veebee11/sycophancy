"""One four-condition group from the SYNTHETIC fixture, against the local server.

    # 1. read-only cache check (no download, no server)
    HF_HUB_OFFLINE=1 uv run python scripts/preflight_model.py --config configs/experiment.yaml

    # 2. the dry run: what would be sent. Omitting --send IS the dry run;
    #    no server is contacted and nothing is generated.
    uv run python scripts/smoke_test.py --config configs/experiment.yaml

    # 3. the one call, once the server is up
    REASONSTYLE_ALLOW_LOCAL_GENERATION=1 HF_HUB_OFFLINE=1 \
    uv run python scripts/smoke_test.py --config configs/experiment.yaml --send

Exactly one call. The material is the synthetic fixture decision
``energy_fixture_001`` — never a pilot brief — and a failure is reported, not
retried:
no repair request is built, no second attempt is made, and the script never
continues into pilot generation.

What it verifies: the four bodies parse against the group schema; the validator
runs over a corpus record assembled from them; the model id and resolved
revision are recorded and agree with the configuration and with the server's own
runtime record; the decoding settings and seed are recorded; the prompt hash is
reproducible from the template and the brief, and the response hash from the
parsed object. GPU, dtype and library versions come from the server's runtime
record, not from this terminal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reasonstyle.config import load_config
from reasonstyle.corpus import load_corpus, segmenter_from_config, validate_corpus
from reasonstyle.corpus.schemas import Cell
from reasonstyle.corpus.topics import load_topic_bank
from reasonstyle.generation import (
    AUTHORIZATION_ENV,
    BackendError,
    GenerationLog,
    ModelNotCached,
    ResponseRejected,
    VLLMOpenAIBackend,
    authorization_problems,
    describe_run,
    group_request,
    load_server_runtime,
    offline_problems,
    parse_response,
    resolve_cached_model,
    revision_agreement,
    server_settings_problems,
    vllm_payload,
    write_request,
)
from reasonstyle.generation.allocation import GroupAllocation
from reasonstyle.generation.log import LogEntry, utc_now
from reasonstyle.hashing import content_hash

# Synthetic throughout. The fixture brief and the fixture corpus share one id,
# and "fixture" is part of it, so a smoke-test record can never be mistaken for
# a pilot item. (The bare "fixture_001" cannot be used as a brief id: a topic id
# must begin with its domain.)
FIXTURE_DECISION = "energy_fixture_001"
FIXTURE_CORPUS = "data/fixtures/corpus.jsonl"
FIXTURE_TOPICS = "data/fixtures/topics.yaml"


def _allocation_from_fixture(record, option: str) -> GroupAllocation:
    """The marker this fixture group already uses — the smoke test changes
    nothing about the design, it only exercises the path."""
    block = record.counterarguments[option]
    return GroupAllocation(
        decision_id=record.decision_id, domain=record.domain, variant_id=record.variant_id,
        scenario_id=record.scenario_id, supported_option=option,
        marker_family=block.marker_family, marker_string=block.marker_string,
        marker_realization_id=block.marker_realization_id)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus", default=FIXTURE_CORPUS)
    ap.add_argument("--topics", default=FIXTURE_TOPICS)
    ap.add_argument("--option", default="opt_1", choices=["opt_1", "opt_2"])
    ap.add_argument("--model", help="override the configured repository id")
    ap.add_argument("--base-url", help="override the configured local endpoint")
    ap.add_argument("--hf-home", help="cache location for the read-only revision check")
    ap.add_argument("--out", default="data/pilot/smoke")
    ap.add_argument("--server-runtime", default="data/pilot/server_runtime.json",
                    help="the record scripts/server/serve_vllm.sh wrote when it started")
    ap.add_argument("--send", action="store_true",
                    help=f"make the one call; also needs {AUTHORIZATION_ENV}=1")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    gen = cfg.raw["models"]["generator"]
    repo_id = args.model or gen["model"]["repo_id"]

    records = load_corpus(args.corpus)
    record = next((r for r in records
                   if r.decision_id == FIXTURE_DECISION and r.variant_id == 1), None)
    if record is None:
        print(f"{args.corpus} has no {FIXTURE_DECISION} variant 1", file=sys.stderr)
        return 1
    bank = load_topic_bank(args.topics)
    topic = next((t for t in bank.topics if t.decision_id == FIXTURE_DECISION), None)
    if topic is None:
        print(f"{args.topics} has no brief for {FIXTURE_DECISION}", file=sys.stderr)
        return 1
    allocation = _allocation_from_fixture(record, args.option)

    request = group_request(topic, 1, record.scenario_text, allocation, cfg)
    payload = vllm_payload(request, cfg)
    if args.base_url:
        gen["vllm"]["base_url"] = args.base_url

    print(f"material     SYNTHETIC {FIXTURE_DECISION} v1 / {args.option} "
          f"({allocation.marker_family}, {allocation.marker_string!r})")
    print(f"call_id      {request.call_id}")
    print(f"prompt hash  {request.prompt_sha256[:16]}  ({len(request.prompt.split())} words)")
    print(f"endpoint     {gen['vllm']['base_url']}")
    print(f"model        {repo_id}")
    print(f"decoding     {json.dumps(gen['decoding'])}")

    # -- the resolved revision, read from the local cache only ---------------
    cached = None
    try:
        cached = resolve_cached_model(repo_id, hf_home=args.hf_home)
        print(f"revision     {cached.revision}  ({cached.snapshot_path})")
        print(f"weights      {len(cached.weight_files)} file(s) verified present")
    except ModelNotCached as exc:
        print(f"revision     NOT RESOLVED: {exc}", file=sys.stderr)

    request_dir = Path(args.out)
    request_path = write_request(request, request_dir)
    print(f"request      {request_path}")

    blocking = authorization_problems(args.send) + offline_problems(cfg)
    if blocking:
        print("\nnothing was sent:")
        for problem in blocking:
            print(f"  - {problem}")
        print("\n--- payload that would be sent ---")
        print(json.dumps(payload, indent=2, ensure_ascii=False)[:2000])
        return 0
    if cached is None:
        print("\nrefusing to run: the weights are not resolvable in the local cache, so the "
              "revision could not be recorded.", file=sys.stderr)
        return 1

    # The server's own record of what it loaded and on which GPU. Required for a
    # live run: GPU and dtype must come from the machine that holds the weights.
    try:
        server = load_server_runtime(args.server_runtime)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"\nrefusing to run: {exc}", file=sys.stderr)
        return 1
    disagreements = (revision_agreement(gen["model"]["revision"], cached,
                                        server.get("revision"))
                     + server_settings_problems(cfg, server))
    if disagreements:
        print("\nrefusing to run: the configuration, the cache and the server must name "
              "the same commit, and the server must run with the configured settings:",
              file=sys.stderr)
        for problem in disagreements:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"server       gpu {server.get('gpu_index')} ({server.get('gpu_name')}), "
          f"{server.get('dtype')}, vllm {server.get('libraries', {}).get('vllm')}, "
          f"generation_config {server.get('generation_config')}")

    log = GenerationLog(Path(args.out) / "generation_log.jsonl",
                        raw_dir=Path(args.out) / "raw")
    backend = VLLMOpenAIBackend(base_url=args.base_url)

    def record_call(status, error, response=None, fields=None):
        env = describe_run(
            cfg, cached=cached, endpoint=gen["vllm"]["base_url"], server=server,
            prompt_sha256=request.prompt_sha256,
            response_sha256=GenerationLog.response_digest(fields) if fields else None,
            input_tokens=(response.usage or {}).get("prompt_tokens") if response else None,
            output_tokens=(response.usage or {}).get("completion_tokens") if response else None)
        log.append(LogEntry(
            call_id=request.call_id, kind=request.kind, attempt=request.attempt,
            decision_id=request.decision_id, variant_id=request.variant_id,
            supported_option=request.supported_option, template_name=request.template_name,
            template_sha256=request.template_sha256, prompt_sha256=request.prompt_sha256,
            model=repo_id, model_returned=response.model_returned if response else None,
            request_fields={k: v for k, v in payload.items() if k != "messages"},
            config_content_hash=cfg.content_hash,
            topic_bank_content_hash=content_hash(bank.model_dump(mode="json")),
            allocation_content_hash=None,
            response_sha256=GenerationLog.response_digest(fields) if fields else None,
            stop_reason=response.stop_reason if response else None,
            usage=response.usage if response else None,
            status=status, error=error, generated_at=utc_now(),
            model_revision=cached.revision, seed=gen["decoding"]["seed"],
            gpu=server.get("gpu_name"), runtime=env.as_dict(),
            outcome="smoke_test_only"))
        return env

    # -- the one call --------------------------------------------------------
    try:
        response = backend.send(request, cfg, allow_live=True)
    except BackendError as exc:
        record_call("error", str(exc))
        print(f"\nthe call failed: {exc}", file=sys.stderr)
        print("No retry was attempted and no repair request was built.", file=sys.stderr)
        return 1

    raw_path = log.store_raw(request.call_id,
                             {k: v for k, v in payload.items() if k != "messages"},
                             response.raw, prompt=request.prompt)

    try:
        fields = parse_response(request, response.content)
    except ResponseRejected as exc:
        record_call("rejected", str(exc), response)
        print(f"\nthe response was rejected: {exc}", file=sys.stderr)
        print(f"raw response kept at {raw_path}", file=sys.stderr)
        print("No retry was attempted and no repair request was built.", file=sys.stderr)
        return 1

    env = record_call("ok", None, response, fields)

    # -- validate, by assembling a record from what came back ----------------
    cells = {c: Cell(condition=c, body=fields[c],
                     markers_present=c in ("RS", "NS"),
                     marker_family=allocation.marker_family if c in ("RS", "NS") else None)
             for c in ("RS", "RP", "NS", "NP")}
    block = record.counterarguments[args.option].model_copy(update={"cells": cells})
    drafted = record.model_copy(update={
        "counterarguments": {**record.counterarguments, args.option: block}})
    others = [r for r in records if r.scenario_id != record.scenario_id]
    report = validate_corpus([drafted, *others], cfg, segmenter_from_config(cfg),
                             corpus_scope="fixture")
    mine = [f for f in report.findings
            if f.scenario_id in (None, drafted.scenario_id)
            and f.supported_option in (None, args.option)]

    print("\n--- the four bodies ---")
    for condition in ("RS", "RP", "NS", "NP"):
        print(f"\n{condition}: {fields[condition]}")

    print("\n--- validator ---")
    print(f"errors {len(report.errors)}, warnings {len(report.warnings)}, "
          f"human judgements outstanding {len(report.human_review)}")
    for finding in mine:
        if finding.severity in ("error", "warning"):
            print(f"  {finding.severity}: {finding.code} — {finding.message[:160]}")

    print("\n--- record ---")
    print(f"model            {repo_id}")
    print(f"revision         {cached.revision}")
    print(f"served as        {response.model_returned}")
    print(f"decoding         {json.dumps(env.decoding)}")
    print(f"seed             {env.seed}")
    print(f"gpu              {server.get('gpu_index')} ({env.gpu}), dtype {env.dtype}")
    print(f"server libraries {server.get('libraries')}")
    print(f"client libraries {env.libraries}")
    print(f"server record    {args.server_runtime}")
    print(f"tokens           in {env.input_tokens}, out {env.output_tokens}")
    print(f"prompt sha256    {env.prompt_sha256}")
    print(f"response sha256  {env.response_sha256}")
    print(f"request file     {request_path}")
    print(f"raw response     {raw_path}")
    print(f"log              {log.path}")
    print(f"\n{env.reproducibility_note}")
    print("\nThis was a smoke test: one call, no retry, no pilot generation.")
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
