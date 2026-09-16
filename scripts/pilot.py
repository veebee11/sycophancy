"""The pilot drafting controller: scenarios, then groups with bounded repair.

    uv run python scripts/pilot.py plan --config configs/experiment.yaml
    uv run python scripts/pilot.py status --config configs/experiment.yaml
    uv run python scripts/pilot.py approvals --config configs/experiment.yaml

Generation is **two resumable stages separated by curator approval**, and they
are two commands on purpose. ``scenarios`` drafts the pilot's scenarios and
stops; a person then reads them and records approvals; ``groups`` refuses until
every one of them is approved against its exact text, and only then drafts the
groups with their bounded repair. No command crosses that boundary on its own.

Commands::

    plan               what a full run would send; writes request files, sends nothing
    scenarios          stage one: draft or recover the scenarios (--send to run)
    scenario-review    the recorded scenarios, for reading, plus a blank approval template
    approvals          where the curator gate stands, per expected scenario
    groups             stage two: draft or recover the groups (--send to run)
    assemble           build data/pilot/corpus.jsonl and its manifest from what is recorded
    status             counts: generated, approved, accepted, repaired, blocking

A live stage needs all three keys — ``--send``,
``REASONSTYLE_ALLOW_LOCAL_GENERATION=1``, ``REASONSTYLE_ALLOW_PILOT_GENERATION=1``
— plus ``HF_HUB_OFFLINE=1``, and it runs the same cache, revision, runtime-record
and server-setting checks as the smoke tests before its first call. Without
``--send`` a stage reports its plan and sends nothing.

Ceilings: the scenario stage makes one call per scenario and has no repair
path — a failing scenario is a curator decision, and there is no automatic
redraft. The group stage makes at most three calls per group, one draft and two
repairs. Completed calls are recovered from disk and never sent again.

**No pilot call has been authorised.** The two synthetic repair smokes are
finished and no further one is planned; nothing in this file has been run
against a server.

The controller itself is ``src/reasonstyle/generation/pipeline.py``; it was
exercised by ``scripts/pipeline_smoke.py`` on one synthetic group, twice, and
those runs are finished. Machine-valid is not approved: the human judgements the
validator lists stay outstanding either way, and an assembled record remains a
draft until they are recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from reasonstyle.config import load_config
from reasonstyle.corpus import segmenter_from_config
from reasonstyle.corpus.topics import load_topic_bank
from reasonstyle.generation import (
    CallStore,
    ModelNotCached,
    VLLMOpenAIBackend,
    authorization_problems,
    load_allocation,
    load_server_runtime,
    offline_problems,
    resolve_cached_model,
    revision_agreement,
    server_settings_problems,
    write_request,
)
from reasonstyle.generation.approvals import approval_status, load_approvals
from reasonstyle.generation.assemble import (
    AssemblyError,
    assemble_pilot,
    load_corrections,
    write_pilot_corpus,
)
from reasonstyle.generation.pipeline import (
    ACCEPTED,
    NEEDS_MANUAL_REVIEW,
    PipelineAbort,
    gate_problems_for,
    recorded_groups,
    recorded_scenarios,
    run_group_stage,
    run_scenario_stage,
)
from reasonstyle.generation.requests import group_request, scenario_request
from reasonstyle.hashing import content_hash, sha256_of

#: The second key a live pilot stage requires, over and above the key a smoke
#: call needs. Drafting a corpus is not the same decision as one smoke call.
PILOT_AUTHORIZATION_ENV = "REASONSTYLE_ALLOW_PILOT_GENERATION"

#: The seven scenario judgements a curator records. A generated template
#: carries them all as `null`: nothing here ever affirms one.
from reasonstyle.generation.approvals import REQUIRED_JUDGEMENTS


def live_problems(send: bool, cfg=None, env=None) -> list[str]:
    """Why a live stage may not send. Empty means every requirement is met.

    Three keys, each a separate decision: ``--send`` names the intent,
    ``REASONSTYLE_ALLOW_LOCAL_GENERATION=1`` permits a local model run at all,
    and ``REASONSTYLE_ALLOW_PILOT_GENERATION=1`` permits *pilot* generation
    specifically — drafting a corpus is not the same act as one smoke call.
    ``HF_HUB_OFFLINE=1`` keeps a missing model an error rather than a download.
    """
    env = os.environ if env is None else env
    problems = authorization_problems(send, env=env)
    if cfg is not None:
        problems += offline_problems(cfg, env=env)
    elif env.get("HF_HUB_OFFLINE") != "1":
        problems.append("HF_HUB_OFFLINE=1 is required")
    if env.get(PILOT_AUTHORIZATION_ENV) != "1":
        problems.append(
            f"{PILOT_AUTHORIZATION_ENV}=1 is not set: pilot generation is a separate "
            f"authorisation from a smoke call")
    return problems


def server_problems(args, cfg) -> tuple[list[str], Any, dict]:
    """The same pre-flight the smoke tests make, before any pilot call.

    ``(problems, cached, server)``. Nothing is sent while this returns
    problems, and no call artefact is created either.
    """
    try:
        cached = resolve_cached_model(cfg.raw["models"]["generator"]["model"]["repo_id"],
                                      hf_home=args.hf_home)
    except ModelNotCached as exc:
        return ([str(exc)], None, {})
    try:
        server = load_server_runtime(args.server_runtime)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return ([str(exc)], cached, {})
    problems = (revision_agreement(cfg.raw["models"]["generator"]["model"]["revision"],
                                   cached, server.get("revision"))
                + server_settings_problems(cfg, server))
    return (problems, cached, server)


#: Commands that act on the pilot itself. A subset of them is not a pilot: the
#: marker allocation is balanced over all 12 decisions and both variants, and a
#: corpus built from part of it would be a different corpus quietly.
WHOLE_PILOT_COMMANDS = ("scenarios", "groups", "assemble")


def whole_pilot_problems(args, bank, cfg) -> list[str]:
    """Why a whole-pilot command may not run on this selection."""
    curated = sorted(t.decision_id for t in bank.topics if t.status == "curated")
    expected = cfg.raw["corpus"]["decisions_pilot"]
    problems = []
    if args.only:
        problems.append(f"--only {list(args.only)} selects a subset; this command runs the "
                        f"whole pilot or nothing")
    if list(args.variants) != [1, 2]:
        problems.append(f"--variants {list(args.variants)} is not both variants [1, 2]")
    if len(curated) != expected:
        problems.append(f"{len(curated)} curated decisions, not {expected}: the pilot is "
                        f"incomplete, and a partial corpus is not the pilot")
    return problems


def _inputs(args):
    cfg = load_config(args.config)
    bank = load_topic_bank(args.topics)
    allocation = load_allocation(args.allocation)
    topics = [t for t in bank.topics if t.status == "curated"]
    if args.only:
        topics = [t for t in topics if t.decision_id in set(args.only)]
    return cfg, bank, allocation, topics


def _plan(cfg, bank, allocation, topics, out: Path, variants) -> int:
    """Write every request that a live run would send, and print the plan."""
    scenarios = groups = 0
    for topic in sorted(topics, key=lambda t: t.decision_id):
        for variant_id in variants:
            write_request(scenario_request(topic, variant_id, cfg), out / "requests")
            scenarios += 1
            for group in allocation.groups:
                if (group.decision_id, group.variant_id) != (topic.decision_id, variant_id):
                    continue
                # A group request needs scenario text, which does not exist in a
                # dry run: the brief's own framing stands in, purely so the
                # request can be rendered and read.
                write_request(group_request(topic, variant_id, topic.decision_framing,
                                            group, cfg), out / "requests")
                groups += 1
    budget = cfg.raw["corpus"]["repair"]
    print(f"config       {cfg.config_version} {cfg.content_hash[:12]}")
    print(f"topics       {len(topics)} curated decisions, variants {list(variants)}")
    print(f"scenarios    {scenarios} calls (one each; no scenario repair template)")
    print(f"groups       {groups} drafts, up to {budget['max_repair_calls']} repairs each "
          f"({budget['max_calls_per_group']} calls per group at most)")
    print(f"ceiling      {scenarios + groups * budget['max_calls_per_group']} calls")
    print(f"requests     written to {out / 'requests'}")
    print("\nDRY RUN: nothing was sent. The group requests above use the brief's framing "
          "as a stand-in for scenario text, which a live run takes from the accepted "
          "scenario draft instead.")
    return 0


def _status(store: CallStore) -> int:
    entries = store.log.entries()
    if not entries:
        print("no calls recorded yet.")
        return 0
    by_outcome: dict[str, int] = {}
    for entry in entries:
        by_outcome[entry.get("outcome") or "?"] = by_outcome.get(entry.get("outcome") or "?", 0) + 1
    print(f"calls recorded {len(entries)} in {store.log.path}")
    for outcome, count in sorted(by_outcome.items()):
        print(f"  {outcome:<32} {count}")
    print("\nA recorded 'accepted' means the validator found no machine errors. It is not "
          "human approval: the item, pair and scenario judgements are still outstanding.")
    return 0


def _approvals(args, cfg, bank, topics, store: CallStore) -> int:
    """Report the gate over the EXPECTED pilot, not only over what exists.

    Every requested topic and variant is listed, including the ones that have
    not been drafted at all — a report that counted only accepted drafts would
    say "0 blocking" for a pilot that has generated nothing. It reads the
    approvals file and the recorded calls, writes nothing, decides nothing and
    contacts nothing: approving is the curator's act, made in the file itself.
    """
    from reasonstyle.generation.approvals import approval_status, load_approvals
    approvals = load_approvals(args.approvals_file)
    bank_hash = content_hash(bank.model_dump(mode="json"))

    # A scenario counts as drafted only when a call produced usable text. A
    # transport failure or a schema rejection produced none, so the scenario is
    # "not generated" however many times it was attempted.
    drafted: dict[str, dict[str, Any]] = {}
    for entry in store.log.entries():
        if entry["kind"] != "scenario":
            continue
        scenario_id = f"{entry['decision_id']}_v{entry['variant_id']}"
        drafted.setdefault(scenario_id, {})
        if entry["status"] != "ok":
            continue
        result = store.result_path(entry["call_id"])
        fields = {}
        if result.is_file():
            fields = json.loads(result.read_text(encoding="utf-8")).get("fields") or {}
        text = fields.get("scenario_text") or ""
        if not text:
            continue
        candidate = {"text": text, "call_id": entry["call_id"],
                     "errors": len((entry.get("validation") or {}).get("error_codes") or []),
                     "outcome": entry.get("outcome")}
        # An accepted call wins; otherwise the latest usable one stands.
        if drafted[scenario_id].get("outcome") != "accepted":
            drafted[scenario_id] = candidate

    expected = [f"{topic.decision_id}_v{variant_id}"
                for topic in sorted(topics, key=lambda t: t.decision_id)
                for variant_id in args.variants]

    print(f"approvals file {args.approvals_file} ({len(approvals)} record(s))")
    print(f"expected scenarios {len(expected)}; scenario calls recorded {len(drafted)}")
    blocking = 0
    for scenario_id in expected:
        record = drafted.get(scenario_id) or {}
        if not record.get("text"):
            state = "not_generated"
            detail = ("no scenario call recorded" if scenario_id not in drafted
                      else "no call produced usable text (transport failure or rejected "
                           "response)")
        elif record["errors"]:
            state, reasons = approval_status(
                scenario_id, record["text"], record["call_id"], approvals,
                config_content_hash=cfg.content_hash, topic_bank_content_hash=bank_hash,
                machine_errors=record["errors"])
            detail = "; ".join(r for r in reasons if r)
        elif record.get("outcome") != "accepted":
            state = "needs_manual_review"
            detail = f"the scenario call ended {record.get('outcome')!r}, not accepted"
        else:
            state, reasons = approval_status(
                scenario_id, record["text"], record["call_id"], approvals,
                config_content_hash=cfg.content_hash, topic_bank_content_hash=bank_hash)
            detail = "; ".join(r for r in reasons if r)
        print(f"  {scenario_id:<28} {state}" + (f"  ({detail})" if detail else ""))
        blocking += state != "approved"

    print(f"\n{blocking} of {len(expected)} expected scenario(s) not approved. Group drafting "
          f"needs EVERY expected scenario drafted, machine-valid and approved against its "
          f"exact text, call, configuration and topic bank; an edit makes an approval stale.")
    print("A machine error can never be approved past: fix or redraft instead.")
    return 0


def _stage(args, cfg, bank, allocation, topics, store: CallStore, *, kind: str) -> int:
    """One live stage: scenarios, or groups. Never both."""
    variants = tuple(args.variants)
    expected_scenarios = len(topics) * len(variants)
    budget = cfg.raw["corpus"]["repair"]["max_calls_per_group"]
    expected_groups = expected_scenarios * 2
    if kind == "scenarios":
        print(f"stage        scenarios: {expected_scenarios} calls at most, one per scenario, "
              f"no repair path")
    else:
        print(f"stage        groups: {expected_groups} drafts, at most "
              f"{expected_groups * budget} calls including repairs")
    print(f"config       {cfg.config_version} {cfg.content_hash[:12]}")
    print(f"out          {store.directory}")

    problems = live_problems(args.send, cfg)
    if problems:
        print("\nnothing was sent:")
        for problem in problems:
            print(f"  - {problem}")
        if kind == "groups":
            gate = gate_problems_for(topics, store, cfg,
                                     approvals=load_approvals(args.approvals_file),
                                     topic_bank_content_hash=content_hash(
                                         bank.model_dump(mode="json")),
                                     variants=variants)
            print(f"\ngate: {len(gate)} scenario(s) would block group drafting"
                  + (f"; first: {gate[0]}" if gate else ""))
        return 0

    checks, cached, server = server_problems(args, cfg)
    if checks:
        print("\nrefusing to run; no call was made:", file=sys.stderr)
        for problem in checks:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    store.cached, store.server = cached, server
    store.endpoint = cfg.raw["models"]["generator"]["vllm"]["base_url"]
    print(f"revision     {cached.revision}")
    print(f"server       gpu {server.get('gpu_index')} ({server.get('gpu_name')}), "
          f"{server.get('dtype')}, generation_config {server.get('generation_config')}")

    backend = VLLMOpenAIBackend()
    segmenter = segmenter_from_config(cfg)
    before = len(store.log.entries())
    try:
        if kind == "scenarios":
            results = run_scenario_stage(topics, cfg, segmenter, backend, store,
                                         allow_live=True, variants=variants)
            ceiling = expected_scenarios
        else:
            results = run_group_stage(
                topics, allocation.groups, cfg, segmenter, backend, store,
                approvals=load_approvals(args.approvals_file),
                topic_bank_content_hash=content_hash(bank.model_dump(mode="json")),
                allow_live=True, variants=variants)
            ceiling = expected_groups * budget
    except PipelineAbort as exc:
        print(f"\nthe stage stopped: {exc}", file=sys.stderr)
        return 1

    made = len(store.log.entries()) - before
    accepted = sum(1 for r in results if r.outcome == ACCEPTED)
    print(f"\n{accepted}/{len(results)} {kind} accepted by the machine checks")
    print(f"calls made   {made} (ceiling {ceiling})")
    if made > ceiling:                       # pragma: no cover - defensive
        print("the stage exceeded its ceiling", file=sys.stderr)
        return 1
    if kind == "scenarios":
        print("\nStage one is finished. Read the scenarios (pilot.py scenario-review), "
              "record approvals, then run the groups stage.")
    print("Machine-valid is not approval: the human judgements remain outstanding.")
    return 0 if accepted == len(results) else 1


def _scenario_review(args, cfg, bank, topics, store: CallStore) -> int:
    """A deterministic, read-only view of the recorded scenarios, for reading.

    Plus, optionally, an approval template — every judgement ``null`` and every
    decision ``pending``. Nothing here affirms a judgement or approves anything;
    that is the curator's act, made by editing the file.
    """
    from reasonstyle.corpus.validate import validate_scenario_text
    segmenter = segmenter_from_config(cfg)
    bank_hash = content_hash(bank.model_dump(mode="json"))
    scenarios = recorded_scenarios(store)
    out = Path(args.review_out)
    out.mkdir(parents=True, exist_ok=True)

    lines = ["# Pilot scenarios, for review", "",
             f"config `{cfg.config_version}` `{cfg.content_hash[:12]}` · topic bank "
             f"`{bank_hash[:12]}`", "",
             "Read each scenario against its brief, then record a decision in the approvals "
             "file. Approving binds the exact text below: any edit makes the approval stale.",
             ""]
    template: dict[str, Any] = {}
    for topic in sorted(topics, key=lambda t: t.decision_id):
        for variant_id in args.variants:
            scenario_id = f"{topic.decision_id}_v{variant_id}"
            record = scenarios.get(scenario_id) or {}
            text = record.get("scenario_text") or ""
            findings = (validate_scenario_text(text, cfg, segmenter,
                                               loc={"decision_id": topic.decision_id,
                                                    "scenario_id": scenario_id})
                        if text else [])
            machine = [f"{f.severity}: {f.code}" for f in findings
                       if f.severity in ("error", "warning")]
            variant = topic.variants[f"v{variant_id}"]
            lines += [
                f"## {scenario_id}", "",
                f"- decision `{topic.decision_id}` · domain `{topic.domain}` · variant "
                f"{variant_id}",
                f"- call `{record.get('call_id', '(not generated)')}`",
                f"- text sha256 `{sha256_of(text) if text else '(none)'}`",
                f"- config `{cfg.content_hash}`",
                f"- topic bank `{bank_hash}`",
                f"- machine findings: {', '.join(machine) if machine else 'none'}", "",
                "### The brief this scenario was drafted from", "",
                f"**Decision.** {topic.decision_framing}", "",
                f"**Option opt_1.** {topic.options['opt_1']}",
                f"  *competing goal:* {topic.competing_goals['opt_1']}", "",
                f"**Option opt_2.** {topic.options['opt_2']}",
                f"  *competing goal:* {topic.competing_goals['opt_2']}", "",
                f"**Why underdetermined.** {topic.why_underdetermined}", "",
                f"**Variant {variant_id} context.** {variant.context}", "",
                "**Variant facts, which the scenario must state and must not exceed:**", ""]
            for option in ("opt_1", "opt_2"):
                for fact in getattr(variant.scenario_facts, option):
                    lines.append(f"- `{option}` {fact}")
            lines += ["",
                      "### The generated scenario", "",
                      "```", text or "(no scenario recorded)", "```", "",
                      "Judgements to record:", ""]
            lines += [f"- [ ] {name}" for name in REQUIRED_JUDGEMENTS]
            lines.append("")
            template[scenario_id] = {
                "scenario_text_sha256": sha256_of(text) if text else None,
                "call_id": record.get("call_id"),
                "config_content_hash": cfg.content_hash,
                "topic_bank_content_hash": bank_hash,
                "decision": "pending",                   # never approved by a tool
                "judgements": {name: None for name in REQUIRED_JUDGEMENTS},
                "reason": "not yet reviewed",
                "decided_by": None,
                "decided_at": None,
            }
    (out / "scenarios.md").write_text("\n".join(lines), encoding="utf-8")
    written = [out / "scenarios.md"]
    if args.write_template:
        import yaml
        path = out / "scenario_approvals.template.yaml"
        path.write_text(yaml.safe_dump(template, sort_keys=True, allow_unicode=True),
                        encoding="utf-8")
        written.append(path)
    for path in written:
        print(f"wrote {path}")
    print(f"{sum(1 for s in scenarios.values() if s.get('scenario_text'))} of "
          f"{len(template)} expected scenarios have recorded text.")
    print("Every template decision is 'pending' and every judgement null: approving is "
          "yours, and the file is where you do it.")
    return 0


def _assemble(args, cfg, bank, allocation, topics, store: CallStore) -> int:
    """Build the corpus from what is recorded, or build nothing."""
    segmenter = segmenter_from_config(cfg)
    try:
        records, manifest = assemble_pilot(
            topics=topics, allocation_groups=allocation.groups,
            approvals=load_approvals(args.approvals_file),
            corrections=load_corrections(args.corrections_file),
            scenarios=recorded_scenarios(store), groups=recorded_groups(store),
            cfg=cfg, segmenter=segmenter,
            topic_bank_content_hash=content_hash(bank.model_dump(mode="json")),
            variants=tuple(args.variants))
    except AssemblyError as exc:
        print(f"\nnothing was assembled:\n{exc}", file=sys.stderr)
        return 1

    try:
        corpus, manifest_path = write_pilot_corpus(
            records, manifest, corpus_path=args.corpus, overwrite=args.overwrite)
    except AssemblyError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    print(f"wrote {corpus} ({len(records)} scenarios, {manifest['n_texts']} texts)")
    print(f"wrote {manifest_path}")
    print(f"validation   {manifest['machine_errors']} errors, "
          f"{manifest['machine_warnings']} warnings, "
          f"{manifest['outstanding_human_review']} human judgements outstanding")
    print(f"status       every record is '{manifest['validation_status']}'. Machine-valid "
          f"and scenario-approved is not an approved corpus.")
    print("\nNext: the review export, then the item, pair and scenario human review:")
    print(f"  uv run python scripts/export_for_review.py --config {args.config} "
          f"--corpus {args.corpus} --scope pilot --out review/pilot")
    return 0


def _counts(args, cfg, bank, topics, store: CallStore) -> int:
    """Read-only counts across both stages."""
    variants = tuple(args.variants)
    expected_scenarios = len(topics) * len(variants)
    scenarios = recorded_scenarios(store)
    groups = recorded_groups(store)
    approvals = load_approvals(args.approvals_file)
    bank_hash = content_hash(bank.model_dump(mode="json"))
    corrections = load_corrections(args.corrections_file)

    generated = sum(1 for s in scenarios.values() if s.get("scenario_text"))
    # Counted per scenario, from its own state — never inferred by subtracting
    # however many messages the gate happened to produce.
    approved = 0
    for topic in sorted(topics, key=lambda t: t.decision_id):
        for variant_id in variants:
            scenario_id = f"{topic.decision_id}_v{variant_id}"
            record = scenarios.get(scenario_id) or {}
            if not record.get("scenario_text"):
                continue
            state, _ = approval_status(
                scenario_id, record["scenario_text"], record["call_id"], approvals,
                config_content_hash=cfg.content_hash, topic_bank_content_hash=bank_hash,
                machine_errors=len(record.get("error_codes") or []))
            approved += state == "approved"

    expected_groups = expected_scenarios * 2
    accepted_first = sum(1 for g in groups.values()
                         if g["outcome"] == ACCEPTED and g["calls"] == 1)
    accepted_repaired = sum(1 for g in groups.values()
                            if g["outcome"] == ACCEPTED and g["calls"] > 1)
    manual = sum(1 for g in groups.values() if g["outcome"] == NEEDS_MANUAL_REVIEW)
    correction_records = [c for c in corrections if c.approval_state == "approved"]
    corrected_groups = {(c.scenario_id, c.supported_option) for c in correction_records}
    # A group is ready when the model got it right, or when an approved
    # correction is recorded for it. Whether a corrected group is *valid* is
    # decided by the validator at assembly, not counted here.
    ready = {key for key, g in groups.items() if g["outcome"] == ACCEPTED} | {
        key for key in corrected_groups if key in groups}
    blocking_groups = expected_groups - len(ready)
    blocking_scenarios = expected_scenarios - approved

    print(f"scenarios expected      {expected_scenarios}")
    print(f"scenarios generated     {generated}")
    print(f"scenarios approved      {approved}")
    print(f"groups expected         {expected_groups}")
    print(f"accepted, no repair     {accepted_first}")
    print(f"accepted after repair   {accepted_repaired}")
    print(f"needs_manual_review     {manual}")
    print(f"groups with an approved correction {len(corrected_groups & set(groups))} "
          f"(from {len(correction_records)} correction record(s), one per cell)")
    print(f"groups ready to assemble{len(ready):>4}")
    print(f"blocking assembly       {blocking_scenarios} scenario(s) not approved, "
          f"{blocking_groups} group(s) neither accepted nor corrected")
    print("\nA corrected group is counted as ready, not as blocked: whether its corrected "
          "cells pass is decided by the validator when you assemble.")
    print("A group counted as accepted is machine-valid only; the item, pair and "
          "scenario judgements are outstanding until recorded.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "scenarios", "scenario-review", "approvals",
                                       "groups", "assemble", "status", "log"],
                    help="plan; draft the scenarios; export them for review; report the "
                         "gate; draft the groups; assemble the corpus; or report counts")
    ap.add_argument("--approvals-file", default="data/pilot/scenario_approvals.yaml")
    ap.add_argument("--config", required=True)
    ap.add_argument("--topics", default="data/topics/pilot_topics.yaml")
    ap.add_argument("--allocation", default="data/pilot/marker_allocation.yaml")
    ap.add_argument("--out", default="data/pilot/run")
    ap.add_argument("--only", nargs="*", help="limit to these decision ids")
    ap.add_argument("--variants", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--corrections-file", default="data/pilot/manual_corrections.yaml")
    ap.add_argument("--corpus", default="data/pilot/corpus.jsonl")
    ap.add_argument("--review-out", default="review/pilot")
    ap.add_argument("--write-template", action="store_true",
                    help="also write a blank approval template (every decision pending)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing corpus deliberately")
    ap.add_argument("--hf-home", help="cache location for the read-only revision check")
    ap.add_argument("--server-runtime", default="data/pilot/server_runtime.json")
    ap.add_argument("--send", action="store_true",
                    help="make the calls of a stage; also needs both authorisation keys")
    args = ap.parse_args(argv)

    cfg, bank, allocation, topics = _inputs(args)
    if args.command in WHOLE_PILOT_COMMANDS:
        problems = whole_pilot_problems(args, bank, cfg)
        if problems:
            print(f"refusing: {args.command} runs on the complete pilot — all 12 curated "
                  f"decisions, variants 1 and 2:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            print("Targeted filtering is available on the reporting commands: "
                  "scenario-review, approvals, status, log and plan.", file=sys.stderr)
            return 1
    out = Path(args.out)
    store = CallStore(out, cfg,
                      topic_bank_content_hash=content_hash(bank.model_dump(mode="json")),
                      allocation_content_hash=allocation.content_hash)

    if args.command == "log":
        return _status(store)
    if args.command == "status":
        return _counts(args, cfg, bank, topics, store)
    if args.command == "approvals":
        return _approvals(args, cfg, bank, topics, store)
    if args.command == "scenario-review":
        return _scenario_review(args, cfg, bank, topics, store)
    if args.command == "assemble":
        return _assemble(args, cfg, bank, allocation, topics, store)
    if args.command in ("scenarios", "groups"):
        return _stage(args, cfg, bank, allocation, topics, store, kind=args.command)
    return _plan(cfg, bank, allocation, topics, out, tuple(args.variants))


if __name__ == "__main__":
    raise SystemExit(main())
