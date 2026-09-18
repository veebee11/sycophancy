"""One call to the OpenAI Responses API, on the SYNTHETIC fixture. Never the pilot.

    # 1. the dry run: exactly what would be sent. No key, no network, no charge.
    uv run python scripts/openai_smoke_test.py --config configs/experiment_openai_pilot.yaml

    # 2. the one paid call
    REASONSTYLE_ALLOW_OPENAI_GENERATION=1 \
    uv run python scripts/openai_smoke_test.py --config configs/experiment_openai_pilot.yaml --send

**Exactly one call**, and the material is the synthetic fixture decision
``energy_fixture_001`` — never a pilot brief, so no pilot topic is spent
proving that the transport works. A failure is reported, not retried: no repair
request is built, no second attempt is made, and this script never continues
into pilot generation. It does not carry the pilot authorisation key and cannot
be turned into a pilot stage by setting one.

Its output goes to a **disposable smoke directory**, `data/pilot/smoke_openai/`
by default, and never to a pilot run directory: a transport check is not
evidence and must not sit among the calls a corpus is built from.

**The credential** is read from ``OPENAI_API_KEY`` by the backend at the moment
of the call and by nothing else. This script never reads it, never prints it,
never stores it and never puts it in an error message; what it prints is
whether the variable is set.

What it verifies: that the payload is the one the configuration describes — the
exact pinned model id, temperature, no ``top_p``, the reasoning effort, the
strict JSON schema, ``store: false``, ``background: false`` and no tools; that
the reply parses against the group schema; that the validator runs over a
record assembled from it; and that the requested and returned model ids, the
response id, the request id, the usage and the stop reason are all recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reasonstyle.config import load_config
from reasonstyle.corpus import load_corpus, segmenter_from_config, validate_corpus
from reasonstyle.corpus.schemas import Cell, DirectionBlock
from reasonstyle.corpus.topics import load_topic_bank
from reasonstyle.corpus.validate import (
    PAIRWISE,
    endorsement_text,
    matching_mode,
    validate_group,
    validate_scenario_text,
)
from reasonstyle.generation import (
    OPENAI_AUTHORIZATION_ENV,
    OPENAI_BACKEND,
    BackendError,
    GenerationLog,
    OpenAIResponsesBackend,
    ResponseRejected,
    describe_run,
    group_request,
    openai_authorization_problems,
    openai_preflight_problems,
    openai_responses_payload,
    parse_response,
    redact_secrets,
    scenario_request,
    write_request,
)
from reasonstyle.generation.allocation import (
    AllocationError,
    GroupAllocation,
    load_allocation,
)
from reasonstyle.generation.log import LogEntry, utc_now

#: The synthetic decision. A pilot brief is never used here: a transport check
#: must not consume, or appear to consume, corpus material.
FIXTURE_DECISION = "energy_fixture_001"


class SmokeRefused(RuntimeError):
    """The call may not be built. Raised before any credential or connection."""


def selectable_markers(cfg) -> dict[str, str]:
    """``marker -> family`` for the markers this design may smoke-test.

    Both sources have to agree. The configuration says which families are
    selectable and which strings each contributes; the allocation says which
    markers the corpus is actually built from. A marker in one and not the
    other is not something to smoke-test, and the intersection is what this
    returns. An absent allocation file leaves the configuration's answer
    standing, so a dry run still works before one is built.
    """
    alloc_cfg = cfg.raw["markers"]["allocation"]
    families = alloc_cfg.get("selectable_families") or alloc_cfg["pilot_families"]
    configured = {marker: family for family in families
                  for marker in alloc_cfg["pilot_strings"][family]}
    path = (cfg.parsed.paths or {}).get("allocation")
    if not path or not Path(path).is_file():
        return configured
    try:
        allocated = {g.marker_string for g in load_allocation(path).groups}
    except AllocationError:                                 # pragma: no cover - defensive
        return configured
    return {marker: family for marker, family in configured.items() if marker in allocated}


def _allocation_from_fixture(record, option: str, cfg,
                             marker: str | None = None) -> GroupAllocation:
    """The marker this smoke call uses, for whichever design is configured.

    v1 inherits the fixture corpus's own allocation, unchanged. v2 cannot: the
    fixture's ``opt_1`` carries ``because``, a premise indicator, which v2
    defers precisely because it cannot realize a no-premise cell — a smoke call
    on it would be testing a marker the design refuses. So v2 takes a marker its
    own configuration and allocation both make selectable, with that family's
    permitted sentence-initial realization: ``--marker`` when one is named, and
    otherwise the first, which keeps the default exactly what it was.
    """
    block = record.counterarguments[option]
    if matching_mode(cfg) != PAIRWISE:
        if marker is not None:
            raise SmokeRefused(
                f"--marker {marker!r} applies only to a pairwise design. This "
                f"configuration takes its marker from the fixture corpus "
                f"({block.marker_string!r}), and changing it would make the smoke call "
                f"test something the fixture is not.")
        return GroupAllocation(
            decision_id=record.decision_id, domain=record.domain,
            variant_id=record.variant_id, scenario_id=record.scenario_id,
            supported_option=option, marker_family=block.marker_family,
            marker_string=block.marker_string,
            marker_realization_id=block.marker_realization_id)

    available = selectable_markers(cfg)
    if marker is None:
        marker = next(iter(available))
    elif marker not in available:
        raise SmokeRefused(
            f"{marker!r} is not selectable under this configuration and its allocation. "
            f"Selectable: {sorted(available)}. Nothing was sent, no credential was read "
            f"and no connection was made.")
    family = available[marker]
    registry = cfg.raw["markers"]["realization"]["registry"]
    realizations = sorted(rid for rid, spec in registry.items()
                          if spec["family"] == family)
    if len(realizations) != 1:                              # pragma: no cover - config-checked
        raise SmokeRefused(f"{family}: expected one permitted realization, got "
                           f"{realizations}")
    return GroupAllocation(
        decision_id=record.decision_id, domain=record.domain,
        variant_id=record.variant_id, scenario_id=record.scenario_id,
        supported_option=option, marker_family=family, marker_string=marker,
        marker_realization_id=realizations[0])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--corpus", default="data/fixtures/corpus.jsonl")
    ap.add_argument("--topics", default="data/fixtures/topics.yaml")
    ap.add_argument("--kind", default="group", choices=["group", "scenario"])
    ap.add_argument("--option", default="opt_1", choices=["opt_1", "opt_2"])
    ap.add_argument("--marker",
                    help="the marker to smoke-test, for a pairwise design. Must be "
                         "selectable under both the configuration and its allocation. "
                         "Omitted, the first selectable marker is used, which is what "
                         "the default has always been")
    ap.add_argument("--out", default=None,
                    help="a disposable smoke directory; never a pilot run directory. "
                         "Defaults to data/pilot/smoke_openai for the v1 design and "
                         "data/pilot/smoke_v2 for the pairwise v2 design")
    ap.add_argument("--send", action="store_true",
                    help=f"make the one paid call; also needs {OPENAI_AUTHORIZATION_ENV}=1")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    gen = cfg.raw["models"]["generator"]
    if gen["backend"] != OPENAI_BACKEND:
        print(f"{args.config} names backend {gen['backend']!r}, not {OPENAI_BACKEND!r}. "
              f"Use configs/experiment_openai_pilot.yaml.", file=sys.stderr)
        return 1

    out = Path(args.out if args.out is not None else
               ("data/pilot/smoke_v2" if matching_mode(cfg) == PAIRWISE
                else "data/pilot/smoke_openai"))
    for reserved in ("data/pilot/run", "data/pilot/run_openai", "data/pilot/run_v2"):
        if out.resolve() == Path(reserved).resolve():
            print(f"refusing: {reserved} is a pilot run directory. A smoke call is a "
                  f"transport check, not corpus evidence, and never goes there.",
                  file=sys.stderr)
            return 1

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
    try:
        allocation = _allocation_from_fixture(record, args.option, cfg, args.marker)
    except SmokeRefused as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1

    if args.kind == "scenario":
        request = scenario_request(topic, 1, cfg)
        print(f"material     SYNTHETIC {FIXTURE_DECISION} v1 scenario brief")
    else:
        request = group_request(topic, 1, record.scenario_text, allocation, cfg)
        print(f"material     SYNTHETIC {FIXTURE_DECISION} v1 / {args.option} "
              f"({allocation.marker_family}, {allocation.marker_string!r})")
    payload = openai_responses_payload(request, cfg)

    model = gen["model"]["pinned_snapshot"] or gen["model"]["id"]
    print(f"kind         {request.kind}")
    print(f"call_id      {request.call_id}")
    print(f"prompt hash  {request.prompt_sha256[:16]}  ({len(request.prompt.split())} words)")
    print(f"endpoint     {gen['openai']['endpoint']}")
    print(f"model        {model}   (exact id; aliases {gen['model']['refused_aliases']} "
          f"are refused)")
    print(f"ceiling      1 call. No repair, no retry, no continuation into the pilot.")
    endorsement = endorsement_text(record.options[args.option], cfg)
    if matching_mode(cfg) == PAIRWISE:
        print(f"design       pairwise (v2) · template {request.template_name}")
        print(f"endorsement  {endorsement!r}")
        print(f"             checked in every cell, not only asked for in the prompt")
    print("\npayload (the complete request body; the Authorization header is built at "
          "call time and is not part of it):")
    print(json.dumps({k: ("<the rendered prompt>" if k == "input" else v)
                      for k, v in payload.items()}, indent=2))
    for absent in ("top_p", "tools", "previous_response_id", "conversation", "seed"):
        assert absent not in payload, absent
    print("\nabsent on purpose: top_p (provider default, not tuned alongside temperature), "
          "tools, conversation, previous_response_id, seed.")

    path = write_request(request, out / "requests")
    print(f"request      {path}")
    print(f"             (the complete rendered prompt is in that file, under \"prompt\")")

    blocking = openai_authorization_problems(args.send)
    if blocking:
        print("\nnothing was sent:")
        for problem in blocking:
            print(f"  - {problem}")
        print("\nDRY RUN: no key was read, no connection was made and nothing was charged.")
        return 0

    preflight = openai_preflight_problems(cfg)
    if preflight:
        print("\nrefusing to run; no call was made:", file=sys.stderr)
        for problem in preflight:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    log = GenerationLog(out / "generation_log.jsonl", raw_dir=out / "raw")
    backend = OpenAIResponsesBackend()
    try:
        response = backend.send(request, cfg, allow_live=True)
    except BackendError as exc:
        # Every message from the backend is already scrubbed; scrubbed again
        # here so that no path can print something credential-shaped.
        print(f"\nthe call failed: {redact_secrets(str(exc))}", file=sys.stderr)
        print("Nothing was retried. One failure is one failure.", file=sys.stderr)
        return 1

    meta = response.provider_meta or {}
    log.store_raw(request.call_id, payload, response.raw, prompt=request.prompt,
                  meta={"stop_reason": response.stop_reason,
                        "model_returned": response.model_returned,
                        "usage": response.usage, "provider": meta})
    print(f"\nresponse id  {meta.get('response_id')}")
    print(f"request id   {meta.get('request_id')}")
    print(f"model asked  {meta.get('requested_model')}")
    print(f"model served {meta.get('returned_model')}")
    print(f"status       {meta.get('status')}   stop_reason {response.stop_reason}")
    print(f"store        {meta.get('store')}   background {meta.get('background')}")
    print(f"usage        {json.dumps(response.usage)}")

    if meta.get("snapshot_more_specific_than_requested"):
        print(f"\nREPORT FOR APPROVAL: the provider served {meta.get('returned_model')!r}, a "
              f"more specifically pinned snapshot of {meta.get('requested_model')!r}. "
              f"Nothing was changed. If you approve it, record it as "
              f"models.generator.model.pinned_snapshot in {args.config}.", file=sys.stderr)

    status = "ok"
    fields = None
    error = None
    if response.stop_reason != "stop":
        status, error = "rejected", f"stop_reason={response.stop_reason!r}, not 'stop'"
        print(f"\nthe response is not a complete draft: {error}", file=sys.stderr)
    else:
        try:
            fields = parse_response(request, response.content)
        except (ResponseRejected, TypeError) as exc:
            status, error = "rejected", str(exc)
            print(f"\nthe response did not fit the schema: {exc}", file=sys.stderr)

    codes: list[str] = []
    warnings: list[str] = []
    if fields is not None:
        segmenter = segmenter_from_config(cfg)
        if request.kind == "scenario":
            findings = validate_scenario_text(fields["scenario_text"], cfg, segmenter,
                                              loc={"decision_id": topic.decision_id})
        else:
            block = DirectionBlock(
                supported_option=allocation.supported_option,
                marker_family=allocation.marker_family,
                marker_string=allocation.marker_string,
                marker_realization_id=allocation.marker_realization_id,
                cells={c: Cell(condition=c, body=fields[c], markers_present=c in ("RS", "NS"),
                               marker_family=allocation.marker_family
                               if c in ("RS", "NS") else None)
                       for c in ("RS", "RP", "NS", "NP")})
            if matching_mode(cfg) == PAIRWISE:
                # Only the synthetic group this call produced. The rest of the
                # fixture corpus was written to the v1 rules and would report
                # v2 failures that say nothing about this call.
                findings = validate_group(
                    record.scenario_text, cfg.raw["corpus"]["counterargument_opening"],
                    block, cfg, segmenter,
                    loc={"decision_id": record.decision_id,
                         "scenario_id": record.scenario_id},
                    endorsement=endorsement)
            else:
                updated = record.model_copy(update={
                    "counterarguments": {**record.counterarguments,
                                         allocation.supported_option: block}})
                report = validate_corpus([updated], cfg, segmenter, corpus_scope="fixture")
                findings = [f for f in report.findings
                            if f.supported_option == allocation.supported_option]
        codes = sorted({f.code for f in findings if f.severity == "error"})
        warnings = sorted({f.code for f in findings if f.severity == "warning"})
        human = sorted({f.code for f in findings if f.severity == "human_review"})
        print(f"\nvalidator    {len(codes)} error(s) {codes}")
        print(f"             {len(warnings)} warning(s) {warnings}")
        print(f"             {len(human)} outstanding human judgement code(s) {human}")
        if codes:
            status = "validation_failed"

    log.append(LogEntry(
        call_id=request.call_id, kind=request.kind, attempt=1,
        decision_id=request.decision_id, variant_id=request.variant_id,
        supported_option=request.supported_option,
        template_name=request.template_name, template_sha256=request.template_sha256,
        prompt_sha256=request.prompt_sha256, model=model,
        model_returned=response.model_returned,
        request_fields={k: v for k, v in payload.items() if k != "input"},
        config_content_hash=cfg.content_hash, topic_bank_content_hash="(fixture)",
        allocation_content_hash=None,
        response_sha256=GenerationLog.response_digest(fields) if fields else None,
        stop_reason=response.stop_reason, usage=response.usage,
        status=status, error=error, generated_at=utc_now(),
        outcome="accepted" if status == "ok" else "needs_manual_review",
        validation={"error_codes": codes, "warning_codes": warnings,
                    "machine_valid": status == "ok"},
        model_revision=gen["model"].get("pinned_snapshot"), seed=None, gpu=None,
        runtime=describe_run(cfg, cached=None, endpoint=gen["openai"]["endpoint"],
                             prompt_sha256=request.prompt_sha256,
                             response_sha256=GenerationLog.response_digest(fields)
                             if fields else None,
                             input_tokens=(response.usage or {}).get("input_tokens"),
                             output_tokens=(response.usage or {}).get("output_tokens"),
                             ).as_dict(),
        extra={"provider": meta}))

    print(f"\nwrote        {out / 'generation_log.jsonl'}")
    print(f"             {out / 'raw' / (request.call_id + '.json')}")
    print("\nOne call was made and nothing was retried. This is a transport and format "
          "check on synthetic material: it establishes nothing about the corpus, and "
          "machine-valid would not be approval even if every code were clean.")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
