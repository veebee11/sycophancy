"""The repair path, once, on synthetic material: one scenario and one group.

    # dry run: builds the scenario request, contacts nothing
    uv run python scripts/pipeline_smoke.py --config configs/experiment.yaml

    # the live gate, once the server is up
    REASONSTYLE_ALLOW_LOCAL_GENERATION=1 HF_HUB_OFFLINE=1 HF_HOME=/data/$USER/hf_cache \
    uv run python scripts/pipeline_smoke.py --config configs/experiment.yaml --send

What the two earlier smoke tests could not show: whether a failing group can be
*repaired*. This runs the real controller — no repair logic is duplicated here —
over the synthetic fixture decision ``energy_fixture_001``, and nothing else.

**Ceiling: four calls.** One scenario draft, then at most one group draft plus
two repairs. The scenario has no repair path: if it fails its machine checks the
run stops there, with one call made. The run never continues into the other
supported option, the other variant, or any other decision, and it is not pilot
generation: it needs no pilot authorisation and writes to its own directory.

Before the first call it makes the same checks as ``smoke_test.py``: both
authorisations, offline mode, the weights resolved from the local cache, the
server's runtime record, agreement between the configured, cached and served
revision, and the launch settings the record must show.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reasonstyle.config import load_config
from reasonstyle.corpus import segmenter_from_config
from reasonstyle.corpus.topics import load_topic_bank
from reasonstyle.generation import (
    AUTHORIZATION_ENV,
    CallStore,
    ModelNotCached,
    PipelineAbort,
    VLLMOpenAIBackend,
    authorization_problems,
    check_request_provenance,
    draft_group,
    draft_scenario,
    load_server_runtime,
    offline_problems,
    resolve_cached_model,
    revision_agreement,
    server_settings_problems,
)
from reasonstyle.generation.allocation import GroupAllocation
from reasonstyle.generation.pipeline import ACCEPTED
from reasonstyle.generation.requests import scenario_request
from reasonstyle.hashing import content_hash

#: Synthetic throughout, and never a pilot brief. "fixture" is in the id, so a
#: record from this run can never be mistaken for pilot material.
FIXTURE_DECISION = "energy_fixture_001"
FIXTURE_TOPICS = "data/fixtures/topics.yaml"
FIXTURE_CORPUS = "data/fixtures/corpus.jsonl"


def _allocation_for(cfg, option: str) -> GroupAllocation:
    """The marker this fixture group already uses. The smoke test changes
    nothing about the design; it only exercises the path."""
    from reasonstyle.corpus import load_corpus
    record = next(r for r in load_corpus(FIXTURE_CORPUS)
                  if r.decision_id == FIXTURE_DECISION and r.variant_id == 1)
    block = record.counterarguments[option]
    return GroupAllocation(
        decision_id=record.decision_id, domain=record.domain, variant_id=1,
        scenario_id=record.scenario_id, supported_option=option,
        marker_family=block.marker_family, marker_string=block.marker_string,
        marker_realization_id=block.marker_realization_id)


def _print_attempts(result) -> None:
    for attempt in result.attempts:
        line = (f"  attempt {attempt.attempt} {attempt.kind:<8} {attempt.status:<9} "
                f"-> {attempt.outcome}")
        if attempt.reused:
            line += "  (recovered from disk, not re-sent)"
        print(line)
        if attempt.error_codes:
            print(f"      errors   {', '.join(attempt.error_codes)}")
        if attempt.warning_codes:
            print(f"      warnings {', '.join(attempt.warning_codes)}")
        if attempt.error:
            print(f"      note     {attempt.error[:160]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--topics", default=FIXTURE_TOPICS)
    ap.add_argument("--option", default="opt_1", choices=["opt_1", "opt_2"],
                    help="the ONE supported option this run drafts")
    ap.add_argument("--out", default="data/pilot/smoke_pipeline")
    ap.add_argument("--base-url", help="override the configured local endpoint")
    ap.add_argument("--hf-home", help="cache location for the read-only revision check")
    ap.add_argument("--server-runtime", default="data/pilot/server_runtime.json")
    ap.add_argument("--send", action="store_true",
                    help=f"make the calls; also needs {AUTHORIZATION_ENV}=1")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    gen = cfg.raw["models"]["generator"]
    if args.base_url:
        gen["vllm"]["base_url"] = args.base_url
    bank = load_topic_bank(args.topics)
    topic = next((t for t in bank.topics if t.decision_id == FIXTURE_DECISION), None)
    if topic is None:
        print(f"{args.topics} has no brief for {FIXTURE_DECISION}", file=sys.stderr)
        return 1
    allocation = _allocation_for(cfg, args.option)
    budget = cfg.raw["corpus"]["repair"]["max_calls_per_group"]

    print(f"material     SYNTHETIC {FIXTURE_DECISION} v1 / {args.option} "
          f"({allocation.marker_family}, {allocation.marker_string!r})")
    print(f"ceiling      {1 + budget} calls: 1 scenario, then at most {budget} for one group "
          f"(1 draft + {cfg.raw['corpus']['repair']['max_repair_calls']} repairs)")
    print(f"endpoint     {gen['vllm']['base_url']}")
    print(f"model        {gen['model']['repo_id']}")
    print(f"decoding     {json.dumps(gen['decoding'])}")
    print(f"out          {args.out}")

    blocking = authorization_problems(args.send) + offline_problems(cfg)
    if blocking:
        request = scenario_request(topic, 1, cfg)
        print("\nnothing was sent:")
        for problem in blocking:
            print(f"  - {problem}")
        print(f"\nthe first request would be the scenario draft {request.call_id[:16]} "
              f"({len(request.prompt.split())} words).")
        return 0

    # -- the same pre-flight the one-call smoke test makes --------------------
    try:
        cached = resolve_cached_model(gen["model"]["repo_id"], hf_home=args.hf_home)
    except ModelNotCached as exc:
        print(f"\nrefusing to run: {exc}", file=sys.stderr)
        return 1
    try:
        server = load_server_runtime(args.server_runtime)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"\nrefusing to run: {exc}", file=sys.stderr)
        return 1
    problems = (revision_agreement(gen["model"]["revision"], cached, server.get("revision"))
                + server_settings_problems(cfg, server))
    if problems:
        print("\nrefusing to run: the configuration, the cache and the server must name the "
              "same commit, and the server must run with the configured settings:",
              file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"revision     {cached.revision}  ({cached.snapshot_path})")
    print(f"server       gpu {server.get('gpu_index')} ({server.get('gpu_name')}), "
          f"{server.get('dtype')}, vllm {server.get('libraries', {}).get('vllm')}, "
          f"generation_config {server.get('generation_config')}, seed {server.get('seed')}")

    store = CallStore(Path(args.out), cfg, cached=cached, server=server,
                      endpoint=gen["vllm"]["base_url"],
                      topic_bank_content_hash=content_hash(bank.model_dump(mode="json")),
                      allocation_content_hash=None)
    backend = VLLMOpenAIBackend(base_url=args.base_url)
    segmenter = segmenter_from_config(cfg)

    # -- one scenario ---------------------------------------------------------
    try:
        scenario = draft_scenario(topic, 1, cfg, segmenter, backend, store, allow_live=True)
    except PipelineAbort as exc:
        print(f"\nthe run stopped: {exc}", file=sys.stderr)
        return 1
    print("\n--- scenario ---")
    _print_attempts(scenario)
    if not scenario.accepted:
        print(f"\nthe scenario is {scenario.outcome}: no group was drafted on top of it.",
              file=sys.stderr)
        _report(store, cfg, bank, allocation)
        return 1
    print(f"\n{scenario.payload['scenario_text']}")

    # -- one group, with its repair path --------------------------------------
    try:
        group = draft_group(topic, 1, scenario.payload["scenario_text"], allocation, cfg,
                            segmenter, backend, store, allow_live=True)
    except PipelineAbort as exc:
        print(f"\nthe run stopped: {exc}", file=sys.stderr)
        return 1
    print(f"\n--- group {args.option} ---")
    _print_attempts(group)
    if group.payload:
        for condition in ("RS", "RP", "NS", "NP"):
            print(f"\n{condition}: {group.payload[condition]}")

    outstanding = sorted({f.code for f in group.findings if f.severity == "human_review"})
    print(f"\noutcome      scenario {scenario.outcome}, group {group.outcome}")
    print(f"calls        {scenario.calls_made + group.calls_made} of {1 + budget} permitted")
    print(f"human review {len(outstanding)} codes outstanding for this group: "
          f"{', '.join(outstanding)}")
    _report(store, cfg, bank, allocation, scenario_text=scenario.payload["scenario_text"])
    print("\nThis was a smoke test on synthetic material: one scenario, one option, no other "
          "variant or decision, and no pilot generation.")
    if group.outcome != ACCEPTED:
        print(f"The group is {group.outcome}; nothing was hand-corrected.", file=sys.stderr)
        return 1
    print("The group is machine-valid. That is not approval: the judgements above remain.")
    return 0


def _report(store: CallStore, cfg, bank, allocation, scenario_text: str | None = None) -> None:
    """Provenance, and the post-call check that each drafted request really was
    built from the curated brief.

    A scenario request is rebuilt from the brief alone. A group request is
    rebuilt from the brief plus **the exact accepted scenario text it was
    given** — rebuilding it from anything else would compare against a prompt
    that was never sent. A repair cannot be rebuilt this way at all: its inputs
    are the previous attempt's bodies and the validator's findings, which this
    checker does not reconstruct, so it says so rather than leaving a blank.
    """
    entries = store.log.entries()
    print(f"\nlog          {store.log.path} ({len(entries)} line(s))")
    for entry in entries:
        if entry["kind"] == "scenario":
            result = check_request_provenance(entry, bank, _alloc_stub(allocation), cfg)
        elif entry["kind"] == "group" and scenario_text is not None:
            result = check_request_provenance(entry, bank, _alloc_stub(allocation), cfg,
                                              scenario_text=scenario_text)
        elif entry["kind"] == "group":
            result = None
            checked = "provenance n/a: the accepted scenario text is not available here"
        else:
            result = None
            checked = ("provenance n/a: a repair is built from the previous attempt's bodies "
                       "and the validator's findings, which this checker does not rebuild")
        if result is not None:
            checked = ("provenance ok" if result.matched
                       else f"PROVENANCE: {'; '.join(result.problems)}")
        print(f"  {entry['call_id'][:16]} {entry['kind']:<8} attempt {entry['attempt']} "
              f"{entry['status']:<9} {entry['outcome']:<20} {checked}")


def _alloc_stub(allocation: GroupAllocation):
    """The provenance checker takes a whole allocation; this run has one group.

    The smoke test drafts one fixture group rather than reading the pilot
    allocation, so the logged allocation hash is null and the checker skips
    that comparison; the marker fields themselves still come from this group.
    """
    from reasonstyle.generation.allocation import MarkerAllocation
    return MarkerAllocation(groups=(allocation,), seed=0, config_content_hash="",
                            topic_bank_content_hash="")


if __name__ == "__main__":
    raise SystemExit(main())
