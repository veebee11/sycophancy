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
    group-review       the recorded groups, for reading; writes Markdown only
    assemble           build data/pilot/corpus.jsonl and its manifest from what is recorded
    status             counts: generated, approved, accepted, repaired, blocking
    preflight          full corpus only: verify the seed, the allocation and the plan;
                       read-only, no credential, no connection

A live stage needs all three keys — ``--send``,
``REASONSTYLE_ALLOW_LOCAL_GENERATION=1``, ``REASONSTYLE_ALLOW_PILOT_GENERATION=1``
— plus ``HF_HUB_OFFLINE=1``, and it runs the same cache, revision, runtime-record
and server-setting checks as the smoke tests before its first call. Without
``--send`` a stage reports its plan and sends nothing.

Ceilings: the scenario stage makes one call per scenario and has no repair
path — a failing scenario is a curator decision, and there is no automatic
redraft. The group stage makes at most three calls per group, one draft and two
repairs. Completed calls are recovered from disk and never sent again.

The 24-call scenario stage and the bounded nine-call redraft stage have run.
All 24 final scenario texts are approved, including five exact-source-bound
human corrections that leave model evidence untouched. The 140-call group stage
ran on 2026-09-17: 2 of the 48 groups are machine-valid and 46 ended
``needs_manual_review``. ``group-review`` is how those 46 are read; it sends
nothing, corrects nothing and proposes nothing.

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
    AllocationError,
    CallStore,
    generator_endpoint,
    ModelNotCached,
    OPENAI_AUTHORIZATION_ENV,
    OPENAI_BACKEND,
    OpenAIResponsesBackend,
    VLLMOpenAIBackend,
    authorization_problems,
    openai_authorization_problems,
    openai_preflight_problems,
    load_allocation,
    load_server_runtime,
    offline_problems,
    resolve_cached_model,
    revision_agreement,
    server_settings_problems,
    write_request,
)
from reasonstyle.generation.approvals import (
    APPROVED,
    BLOCKED,
    PENDING,
    REDRAFT,
    STALE,
    approval_status,
    load_approvals,
)
# The approval gate's own "needs manual review" state, named explicitly and
# distinctly from ``pipeline.NEEDS_MANUAL_REVIEW`` (the *group*-stage outcome,
# imported below): the two happen to share a string today, but nothing here
# should rely on that being true tomorrow.
from reasonstyle.generation.approvals import NEEDS_MANUAL_REVIEW as SCENARIO_NEEDS_MANUAL_REVIEW
from reasonstyle.generation.assemble import (
    AssemblyError,
    assemble_pilot,
    load_corrections,
    write_pilot_corpus,
)
from reasonstyle.generation.pipeline import (
    ACCEPTED,
    GROUP_STAGE_AUTHORIZATION_RECORD_VERSION,
    GROUP_STAGE_KIND,
    NEEDS_MANUAL_REVIEW,
    PipelineAbort,
    gate_problems_for,
    group_stage_targets,
    recorded_groups,
    recorded_scenarios,
    run_group_stage,
    run_initial_group_stage,
    run_scenario_stage,
    transport_blocked_group_targets,
)
from reasonstyle.generation.group_review import (
    MACHINE_VALID,
    NEEDS_CORRECTION,
    build_group_review,
)
from reasonstyle.generation.redraft import (
    STAGE_AUTHORIZATION_RECORD_VERSION,
    current_scenarios,
    redraft_targets,
    redraft_template_sha256,
    run_redraft_stage,
    transport_blocked_targets,
)
from reasonstyle.generation.stage_authorization import (
    StageAuthorizationError,
    group_stage_authorization_problems,
    load_stage_authorization,
    stage_authorization_problems,
)
from reasonstyle.generation.scenario_source import (
    ScenarioSourceError,
    load_scenario_source,
    scenario_source_spec,
)
from reasonstyle.generation.corpus_source import (
    SeedCorpusError,
    load_seed_corpus,
    plan_full_corpus,
    seed_corpus_spec,
)
from reasonstyle.generation.scenario_corrections import (
    ScenarioCorrectionError,
    apply_scenario_corrections,
    load_scenario_corrections,
)
from reasonstyle.generation.requests import RequestError, group_request, scenario_request
from reasonstyle.hashing import content_hash, sha256_of

#: The second key a live pilot stage requires, over and above the key a smoke
#: call needs. Drafting a corpus is not the same decision as one smoke call.
PILOT_AUTHORIZATION_ENV = "REASONSTYLE_ALLOW_PILOT_GENERATION"

#: The seven scenario judgements a curator records. A generated template
#: carries them all as `null`: nothing here ever affirms one.
from reasonstyle.generation.approvals import REQUIRED_JUDGEMENTS


def live_problems(send: bool, cfg=None, env=None) -> list[str]:
    """Why a live stage may not send. Empty means every requirement is met.

    Three keys, each a separate decision. ``--send`` names the intent;
    ``REASONSTYLE_ALLOW_PILOT_GENERATION=1`` permits *pilot* generation
    specifically, because drafting a corpus is not the same act as one smoke
    call; and the third depends on which generator the configuration names.
    A local run needs ``REASONSTYLE_ALLOW_LOCAL_GENERATION=1`` plus
    ``HF_HUB_OFFLINE=1``, which keeps a missing model an error rather than a
    download. A hosted run needs ``REASONSTYLE_ALLOW_OPENAI_GENERATION=1``
    instead: permitting a run on our own GPU and permitting a paid call to a
    third party are different decisions, and neither key stands in for the
    other.
    """
    env = os.environ if env is None else env
    block = (cfg.raw["corpus"].get("generation_block") or {}) if cfg is not None else {}
    if block.get("blocked"):
        # A configuration can refuse its own generation. The full-corpus draft
        # does, until its allocation, seed import, tested orchestration and
        # paid-call authorisation are all in place; the configuration's own
        # reason says which. Lifting it is an edit to the configuration, made
        # deliberately.
        return [f"{cfg.config_version}: generation is blocked by this configuration. "
                f"{block.get('reason', '')}".strip()]
    if cfg is not None and cfg.raw["models"]["generator"]["backend"] == OPENAI_BACKEND:
        # A paid external call. The local-generation key does not authorise one,
        # and offline mode is meaningless: there are no weights to fetch.
        problems = openai_authorization_problems(send, env=env)
    else:
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

    A hosted generator has no local cache, no weights and no server of ours to
    check, so its pre-flight is a different one: the credential is confirmed
    **present** without its value ever being read, the model id is confirmed
    pinned, and the endpoint is confirmed to be the HTTPS Responses API.
    """
    if cfg.raw["models"]["generator"]["backend"] == OPENAI_BACKEND:
        return (openai_preflight_problems(cfg), None, {})
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


#: Where each generator's run evidence lives. They are separate directories on
#: purpose: the completed Qwen run is finished historical evidence, and a second
#: generator's calls must never be appended to it or written over it.
RUN_DIRECTORIES = {
    "local_vllm_openai": "data/pilot/run",
    OPENAI_BACKEND: "data/pilot/run_openai",
}

#: Every per-generator path, defaulted from the configuration rather than typed
#: on the command line. An approval binds the configuration hash and the exact
#: call it was granted against, so a Qwen approval can never apply to a run
#: drafted by another generator — and a shared file would be an invitation to
#: try. The same goes for the correction ledgers, the corpus and the review
#: exports: one generator, one set of records.
PROFILE_PATHS = {
    "local_vllm_openai": {
        "allocation": "data/pilot/marker_allocation.yaml",
        "out": "data/pilot/run",
        "approvals_file": "data/pilot/scenario_approvals.yaml",
        "scenario_corrections_file": "data/pilot/scenario_corrections.yaml",
        "corrections_file": "data/pilot/manual_corrections.yaml",
        "corpus": "data/pilot/corpus.jsonl",
        "review_out": "review/pilot",
        "group_review_out": "review/pilot_groups",
    },
    OPENAI_BACKEND: {
        "allocation": "data/pilot/marker_allocation.yaml",
        "out": "data/pilot/run_openai",
        "approvals_file": "data/pilot/scenario_approvals_openai.yaml",
        "scenario_corrections_file": "data/pilot/scenario_corrections_openai.yaml",
        "corrections_file": "data/pilot/manual_corrections_openai.yaml",
        "corpus": "data/pilot/corpus_openai.jsonl",
        "review_out": "review/pilot_openai",
        "group_review_out": "review/pilot_groups_openai",
    },
}


def _print_generator(cfg, cached, server: dict) -> None:
    """What is about to draft, printed before the first call. No secret."""
    gen = cfg.raw["models"]["generator"]
    if gen["backend"] == OPENAI_BACKEND:
        model = gen["model"]["pinned_snapshot"] or gen["model"]["id"]
        dec = gen["decoding"]
        print(f"generator    {gen['backend']} · model {model} (exact id, not an alias)")
        print(f"endpoint     {gen['openai']['endpoint']}")
        print(f"decoding     profile {dec['profile']} · temperature {dec['temperature']} · "
              f"top_p provider default (not sent) · max_output_tokens "
              f"{dec['max_output_tokens']} · reasoning {dec['reasoning']}")
        print(f"statelessness store={gen['openai']['store']}, "
              f"background={gen['openai']['background']}, tools {gen['openai']['tools']}, "
              f"no conversation or previous-response state")
        print(f"credential   read from {gen['openai']['api_key_env_name']} at call time; "
              f"never printed, stored, hashed or recorded")
        return
    print(f"revision     {cached.revision}")
    print(f"server       gpu {server.get('gpu_index')} ({server.get('gpu_name')}), "
          f"{server.get('dtype')}, generation_config {server.get('generation_config')}")


#: How a configuration's own ``paths`` block maps onto the command-line names.
_DECLARED_PATHS = {"run": "out", "approvals": "approvals_file",
                   "scenario_corrections": "scenario_corrections_file",
                   "corrections": "corrections_file", "corpus": "corpus",
                   "review": "review_out", "group_review": "group_review_out",
                   "allocation": "allocation"}


def profile_paths(cfg) -> dict[str, str]:
    """Every default path for this configuration.

    A configuration that declares ``paths`` owns its outputs outright: that is
    what lets a second *design* share a backend with the first without sharing
    a run directory, an approvals file, a correction ledger, a corpus or a
    review export. One that declares none keeps the per-generator defaults,
    which is what every configuration before v2 means.
    """
    declared = cfg.parsed.paths or {}
    paths = dict(PROFILE_PATHS[cfg.raw["models"]["generator"]["backend"]])
    for key, value in declared.items():
        if key not in _DECLARED_PATHS:
            raise KeyError(f"paths.{key} is not a path this tool sets")
        paths[_DECLARED_PATHS[key]] = value
    return paths


def _live_backend(cfg):
    """The one place a live backend is constructed, chosen by the config.

    Everything else in this script reads files and prints. Which generator runs
    is a property of the configuration that was named on the command line, not
    of a flag someone can flip at the call site.
    """
    if cfg.raw["models"]["generator"]["backend"] == OPENAI_BACKEND:
        return OpenAIResponsesBackend()
    return VLLMOpenAIBackend()


def run_directory_problems(store: CallStore, cfg) -> list[str]:
    """Why this run directory does not belong to this generator.

    A run directory is one generator's evidence. Mixing two into it would make
    the corpus that came out of it unattributable, and appending to the
    completed Qwen run would alter finished evidence. Both are refused before
    any call is made.
    """
    backend = cfg.raw["models"]["generator"]["backend"]
    expected = profile_paths(cfg)["out"]
    problems = []
    here = Path(store.directory).resolve()
    # Every run directory any design owns, including the ones a configuration
    # declares for itself. A design may write only to its own.
    owned = dict(RUN_DIRECTORIES)
    owned.update({"v2_pilot": "data/pilot/run_v2", "v2_full": "data/full/run_v2"})
    for other, path in owned.items():
        if Path(path).resolve() == here and Path(expected).resolve() != here:
            problems.append(f"{store.directory} belongs to {other!r}; this configuration "
                            f"writes to {expected}. A run directory is one design's "
                            f"evidence, and no other design writes into it.")
    seen = set()
    for entry in store.log.entries():
        recorded = ((entry.get("runtime") or {}).get("backend")
                    or (entry.get("request_fields") or {}).get("backend"))
        if recorded:
            seen.add(recorded)
    foreign = sorted(seen - {backend})
    if foreign:
        problems.append(
            f"{store.directory} already holds calls made by {foreign}; this configuration "
            f"names {backend!r}. A run directory is one generator's evidence, and a second "
            f"generator never appends to it.")
    return problems


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


PILOT_TOPICS = "data/topics/pilot_topics.yaml"

#: Commands that would draft, or write requests for, material. For a design that
#: grows from a seed they run on the new decisions only.
DRAFTING_COMMANDS = ("plan", "scenarios", "groups", "redraft-scenarios")


def seed_output_problems(cfg) -> list[str]:
    """Why a seed-aware design's outputs are not safely its own. A design that
    imports the pilot read-only never writes anywhere under data/pilot/."""
    problems = []
    pilot = Path("data/pilot").resolve()
    for name, path in sorted(profile_paths(cfg).items()):
        resolved = Path(path).resolve()
        if resolved == pilot or pilot in resolved.parents:
            problems.append(f"{name} {path} is under data/pilot/, which a seed-aware design "
                            f"only ever reads")
    return problems


def _inputs(args):
    """Load the configuration, resolve this design's paths, then everything else.

    The paths come first because the allocation is one of them: a design that
    declares its own allocation must not have the previous design's loaded out
    from under it before the declaration is read.
    """
    cfg = load_config(args.config)
    for name, default in profile_paths(cfg).items():
        if getattr(args, name, None) is None:
            setattr(args, name, default)
    if args.topics is None:
        # A configuration that names its own topic bank is read against it; one
        # that names none is a pilot configuration, read against the pilot bank.
        args.topics = (cfg.raw.get("topics") or {}).get("bank") or PILOT_TOPICS
    bank = load_topic_bank(args.topics)
    allocation = load_allocation(args.allocation)
    topics = [t for t in bank.topics if t.status == "curated"]
    if args.only:
        topics = [t for t in topics if t.decision_id in set(args.only)]
    return cfg, bank, allocation, topics


def _current_scenarios(args, cfg, store: CallStore) -> dict[str, dict[str, Any]]:
    """The scenario text that currently stands, from wherever this design gets it.

    A design that declares a ``scenario_source`` reads another's approved
    scenarios instead of drafting its own: they are verified against the source
    configuration's hash, the accepted call ids, the text hashes and the topic
    bank, and the source run directory is only ever opened for reading.
    Otherwise this is the design's own drafts, redrafts and approved human
    corrections, exactly as before.
    """
    if scenario_source_spec(cfg):
        return _verified_source(args, cfg).scenarios
    scenarios = current_scenarios(store, load_approvals(args.approvals_file))
    return apply_scenario_corrections(
        scenarios, load_scenario_corrections(args.scenario_corrections_file), cfg,
        segmenter_from_config(cfg))


def _scenario_provenance(args, cfg) -> dict[str, Any] | None:
    """What every v2 call records about where its scenario came from."""
    if not scenario_source_spec(cfg):
        return None
    return _verified_source(args, cfg).provenance


#: One verified source per invocation. Verification is a pure read of files that
#: do not change mid-command, and doing it three times per command was
#: duplicated work rather than extra assurance.
_SOURCE_CACHE: dict[tuple[str, str], Any] = {}


def _verified_source(args, cfg):
    """The declared scenario source, verified once and reused."""
    key = (cfg.content_hash, str(args.topics))
    if key not in _SOURCE_CACHE:
        _SOURCE_CACHE[key] = load_scenario_source(cfg, topics_path=args.topics)
    return _SOURCE_CACHE[key]


def _gate(args, cfg, store: CallStore):
    """The approvals and the configuration hash they were granted under.

    ONE representation, used by generation, ``status``, ``scenario-review``,
    ``group-review`` and ``assemble`` alike. A design that reuses another's
    scenarios uses the *source's* approvals and the *source's* configuration
    hash, because that is what the curator read the text under; a design that
    drafts its own uses its own file and its own hash. Nothing anywhere
    fabricates an approval or restamps one with a hash it was not granted under.
    """
    source = scenario_source_spec(cfg)
    if not source:
        return load_approvals(args.approvals_file), cfg.content_hash, None
    verified = _verified_source(args, cfg)
    return verified.approvals, verified.config_content_hash, verified


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
    approvals, gate_hash, source = _gate(args, cfg, store)
    bank_hash = content_hash(bank.model_dump(mode="json"))

    # A scenario counts as drafted only when a call produced usable text. A
    # transport failure or a schema rejection produced none, so the scenario is
    # "not generated" however many times it was attempted.
    drafted: dict[str, dict[str, Any]] = {}
    attempted: set[str] = set()
    for scenario_id, record in _current_scenarios(args, cfg, store).items():
        attempted.add(scenario_id)
        if record.get("scenario_text"):
            drafted[scenario_id] = {"text": record["scenario_text"],
                                    "call_id": record["call_id"],
                                    "errors": len(record.get("error_codes") or []),
                                    "outcome": record.get("outcome")}
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
            detail = ("no call produced usable text (transport failure or rejected "
                      "response)" if scenario_id in attempted
                      else "no scenario call recorded")
        elif record["errors"]:
            state, reasons = approval_status(
                scenario_id, record["text"], record["call_id"], approvals,
                config_content_hash=gate_hash, topic_bank_content_hash=bank_hash,
                machine_errors=record["errors"])
            detail = "; ".join(r for r in reasons if r)
        elif record.get("outcome") != "accepted":
            state = "needs_manual_review"
            detail = f"the scenario call ended {record.get('outcome')!r}, not accepted"
        else:
            state, reasons = approval_status(
                scenario_id, record["text"], record["call_id"], approvals,
                config_content_hash=gate_hash, topic_bank_content_hash=bank_hash)
            detail = "; ".join(r for r in reasons if r)
        print(f"  {scenario_id:<28} {state}" + (f"  ({detail})" if detail else ""))
        blocking += state != "approved"

    print(f"\n{blocking} of {len(expected)} expected scenario(s) not approved. Group drafting "
          f"needs EVERY expected scenario drafted, machine-valid and approved against its "
          f"exact text, call, configuration and topic bank; an edit makes an approval stale.")
    print("A machine error can never be approved past: fix or redraft instead.")
    return 0


def reused_scenarios_problem(cfg, what: str) -> str | None:
    """Why a design that reuses scenarios may not draft them.

    Checked before any authorisation, any credential and any backend: a
    configuration that declares a ``scenario_source`` has its scenarios
    already, read-only and approved under another design's hash. Drafting or
    redrafting here would produce text nobody approved, in a run directory that
    is supposed to hold groups only — and would spend money to do it.
    """
    source = scenario_source_spec(cfg)
    if not source:
        return None
    return (f"{what} is refused: this configuration reuses the approved scenarios of "
            f"{source['config']} ({source['run']}, read-only) and drafts none of its own. "
            f"Nothing was sent, no credential was read and no backend was built.")


def _stage(args, cfg, bank, allocation, topics, store: CallStore, *, kind: str) -> int:
    """One live stage: scenarios, or groups. Never both.

    A group stage can be covered by two mechanisms, exactly as a redraft can
    (``_redraft``'s docstring): no ``--stage-authorization`` leaves it to the
    frozen configuration's own ``generation_authorization`` (refused, for
    ``configs/frozen/v2_full.yaml``, since it excludes ``group``); with one, a
    separate, tracked, fail-closed record is checked instead
    (``group_stage_authorization_problems``), and only a record whose every
    check passes lets the send path accept a ``group`` call at all — for
    exactly the initial draft of its named targets, never a repair.
    ``--stage-authorization`` on the scenario stage is refused outright: that
    stage is what the configuration's own authorisation already covers.
    """
    stage_group_auth_record = None
    if kind == "groups" and args.stage_authorization:
        try:
            stage_group_auth_record = load_stage_authorization(args.stage_authorization)
        except StageAuthorizationError as exc:
            print(f"refusing; nothing was sent: {exc}", file=sys.stderr)
            return 1
    elif kind == "scenarios" and args.stage_authorization:
        print("refusing; nothing was sent: --stage-authorization applies to the groups "
              "stage only; the scenario stage is governed by the frozen configuration's own "
              "generation_authorization", file=sys.stderr)
        return 1

    if kind == "scenarios":
        refusal = reused_scenarios_problem(cfg, "the scenario stage")
        if refusal:
            print(f"refusing: {refusal}", file=sys.stderr)
            return 1
    variants = tuple(args.variants)
    expected_scenarios = len(topics) * len(variants)
    budget = cfg.raw["corpus"]["repair"]["max_calls_per_group"]
    expected_groups = expected_scenarios * 2
    if kind == "scenarios":
        print(f"stage        scenarios: {expected_scenarios} calls at most, one per scenario, "
              f"no repair path")
    elif stage_group_auth_record is not None:
        print(f"stage        groups (initial only, no repair): {expected_groups} call(s) at "
              f"most, one per group")
    else:
        print(f"stage        groups: {expected_groups} drafts, at most "
              f"{expected_groups * budget} calls including repairs")
    print(f"config       {cfg.config_version} {cfg.content_hash[:12]}")
    print(f"out          {store.directory}")

    group_targets = group_scenarios = None
    if kind == "groups" and stage_group_auth_record is not None:
        # ONE snapshot of the current scenario evidence: the targets are
        # verified against it here, and the group prompts are built from this
        # same snapshot below -- never re-read in between.
        group_scenarios = _current_scenarios(args, cfg, store)
        try:
            group_targets = group_stage_targets(
                tuple((f"{topic.decision_id}_v{variant_id}", option)
                      for topic in topics for variant_id in variants
                      for option in ("opt_1", "opt_2")),
                _gate(args, cfg, store)[0], allocation, config_content_hash=cfg.content_hash,
                topic_bank_content_hash=content_hash(bank.model_dump(mode="json")),
                scenarios=group_scenarios)
        except RequestError as exc:
            print(f"refusing: {exc}", file=sys.stderr)
            return 1
        problems = group_stage_authorization_problems(
            stage_group_auth_record, cfg=cfg, kind=GROUP_STAGE_KIND,
            record_version=GROUP_STAGE_AUTHORIZATION_RECORD_VERSION, targets=group_targets,
            approvals_path=args.approvals_file, allocation_path=args.allocation,
            allocation_content_hash=allocation.content_hash)
        if problems:
            print("refusing; nothing was sent:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        blocked = transport_blocked_group_targets(
            store, group_targets, stage_group_auth_record["_file_sha256"])
        if blocked:
            print("refusing; nothing was sent:", file=sys.stderr)
            for target in blocked:
                print(f"  - {target.scenario_id}/{target.supported_option}: a transport "
                      f"failure already spent this target's one authorised call under this "
                      f"exact stage authorisation; a new, explicit retry authorisation record "
                      f"is required before it can be dispatched again", file=sys.stderr)
            return 1
        # Only a fully validated stage authorisation widens the send path,
        # and only to the one kind it covers.
        store.allowed_kinds = frozenset({"group"})
        print(f"authorised by stage authorisation {stage_group_auth_record['_path']} "
              f"{stage_group_auth_record['_file_sha256'][:12]}")
    else:
        planned = expected_scenarios if kind == "scenarios" else expected_groups * budget
        refusals = paid_call_problems(cfg, "scenario" if kind == "scenarios" else "group",
                                          planned)
        if refusals:
            print("\nrefusing; nothing was sent, no credential was read and no backend was "
                  "built:", file=sys.stderr)
            for refusal in refusals:
                print(f"  - {refusal}", file=sys.stderr)
            return 1
        auth = authorization(cfg)
        if auth:
            print(f"authorised   {auth['scope']}: at most {auth['max_paid_calls']} paid "
                  f"call(s), {auth['calls_per_scenario']} per scenario, recorded by "
                  f"{auth['authorized_by']} on {auth['authorized_at']}")

    problems = live_problems(args.send, cfg)
    if problems:
        print("\nnothing was sent:")
        for problem in problems:
            print(f"  - {problem}")
        if kind == "groups" and not scenario_source_spec(cfg):
            gate = gate_problems_for(topics, store, cfg,
                                     approvals=_gate(args, cfg, store)[0],
                                     topic_bank_content_hash=content_hash(
                                         bank.model_dump(mode="json")),
                                     variants=variants,
                                     scenarios=_current_scenarios(args, cfg, store))
            print(f"\ngate: {len(gate)} scenario(s) would block group drafting"
                  + (f"; first: {gate[0]}" if gate else ""))
        return 0

    checks = server_problems(args, cfg)[0] + run_directory_problems(store, cfg)
    if checks:
        print("\nrefusing to run; no call was made:", file=sys.stderr)
        for problem in checks:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    _, cached, server = server_problems(args, cfg)
    store.cached, store.server = cached, server
    store.endpoint = generator_endpoint(cfg)
    store.scenario_source = _scenario_provenance(args, cfg)
    _print_generator(cfg, cached, server)
    if store.scenario_source:
        source = store.scenario_source
        print(f"scenarios    reused read-only from {source['run']} "
              f"({source['scenario_count']} approved)")
        print(f"             verified against {source['config']} "
              f"{source['config_content_hash'][:12]} ({source['config_version']}): "
              f"{', '.join(source['verified'])}")
        print(f"             every call records that source AND this design's config "
              f"{cfg.content_hash[:12]}")

    backend = _live_backend(cfg)
    segmenter = segmenter_from_config(cfg)
    before = len(store.log.entries())
    try:
        if kind == "scenarios":
            results = run_scenario_stage(topics, cfg, segmenter, backend, store,
                                         allow_live=True, variants=variants)
            ceiling = expected_scenarios
        elif stage_group_auth_record is not None:
            results = run_initial_group_stage(
                topics, group_targets, allocation.groups, cfg, segmenter, backend, store,
                scenarios=group_scenarios, allow_live=True,
                variants=variants, stage_authorization=stage_group_auth_record)
            ceiling = len(group_targets)
        else:
            source = scenario_source_spec(cfg)
            results = run_group_stage(
                topics, allocation.groups, cfg, segmenter, backend, store,
                approvals=_gate(args, cfg, store)[0],
                topic_bank_content_hash=content_hash(bank.model_dump(mode="json")),
                allow_live=True, variants=variants,
                scenarios=_current_scenarios(args, cfg, store),
                gate_verified_elsewhere=bool(source))
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


def _redraft(args, cfg, bank, topics, store: CallStore) -> int:
    """Redraft exactly the scenarios the curator marked ``redraft``.

    The set comes from the approvals file, never from ``--only``: which
    scenarios need drafting again is the reviewer's finding. One call each, no
    group call, and no automatic second attempt.

    A redraft is a paid call like any other. Two mechanisms can cover it, and
    exactly one applies on a given invocation:

    - No ``--stage-authorization``: the frozen configuration's own
      ``generation_authorization`` is checked, exactly as for every other
      stage. For ``configs/frozen/v2_full.yaml`` this always refuses —
      ``scenario_redraft`` is excluded from it by name — precisely as before
      this flag existed.
    - ``--stage-authorization PATH``: a separate, tracked record is checked
      instead (:mod:`stage_authorization`). It deliberately *widens* what the
      configuration alone would allow — a validated record is exactly what
      lets ``scenario_redraft`` through at all, for two exact, named targets
      — but it never touches the configuration file or its content hash; see
      that module's docstring for why. Only once every one of its checks
      passes (fails closed: a missing, null or wrongly typed field refuses
      cleanly) does the send path (``CallStore.allowed_kinds``) accept a
      ``scenario_redraft`` call. A transport failure still spends the target's
      one authorised call under this exact record — retrying needs a new,
      explicit retry authorisation, not another invocation of this one
      (:func:`reasonstyle.generation.redraft.transport_blocked_targets`).
    """
    stage_auth_record = None
    if args.stage_authorization:
        try:
            stage_auth_record = load_stage_authorization(args.stage_authorization)
        except StageAuthorizationError as exc:
            print(f"refusing; nothing was sent: {exc}", file=sys.stderr)
            return 1
    else:
        refusals = paid_call_problems(cfg, "scenario_redraft")
        if refusals:
            print("refusing; nothing was sent: " + "; ".join(refusals), file=sys.stderr)
            return 1

    refusal = reused_scenarios_problem(cfg, "redraft-scenarios")
    if refusal:
        print(f"refusing: {refusal}", file=sys.stderr)
        return 1
    # The set is the curator's finding, so an operator narrowing it would be
    # overruling the review rather than filtering a report. This holds
    # whichever authorisation mechanism applies: a stage authorisation binds
    # to the exact reviewed targets and can never be used to widen or
    # redirect the set --only would have narrowed.
    if args.only or list(args.variants) != [1, 2]:
        print("refusing: redraft-scenarios runs on the complete pilot — the scenarios the "
              "curator marked redraft, and only those:", file=sys.stderr)
        if args.only:
            print(f"  - --only {list(args.only)} cannot narrow a set that comes from the "
                  f"approvals file", file=sys.stderr)
        if list(args.variants) != [1, 2]:
            print(f"  - --variants {list(args.variants)} is not both variants [1, 2]",
                  file=sys.stderr)
        print("Targeted filtering is available on the reporting commands: scenario-review, "
              "approvals, status, log and plan.", file=sys.stderr)
        return 1

    approvals = load_approvals(args.approvals_file)
    try:
        targets = redraft_targets(approvals, recorded_scenarios(store))
    except RequestError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
    if not targets:
        print("no scenario is marked redraft; nothing to do.")
        return 0

    if stage_auth_record is not None:
        problems = stage_authorization_problems(
            stage_auth_record, cfg=cfg, kind="scenario_redraft",
            record_version=STAGE_AUTHORIZATION_RECORD_VERSION, targets=targets,
            approvals_path=args.approvals_file)
        if problems:
            print("refusing; nothing was sent:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        blocked = transport_blocked_targets(store, targets, stage_auth_record["_file_sha256"])
        if blocked:
            print("refusing; nothing was sent:", file=sys.stderr)
            for target in blocked:
                print(f"  - {target.scenario_id}: a transport failure already spent this "
                      f"target's one authorised call under this exact stage authorisation; a "
                      f"new, explicit retry authorisation record is required before it can be "
                      f"dispatched again", file=sys.stderr)
            return 1
        # Only a fully validated stage authorisation widens the send path,
        # and only to the one kind it covers: a redraft call, and nothing
        # else this run could otherwise be asked to make.
        store.allowed_kinds = frozenset({"scenario_redraft"})

    print(f"stage        redraft: {len(targets)} call(s), one per rejected scenario, no "
          f"second attempt")
    print(f"template     prompts/scenario_redraft_v1.txt "
          f"{redraft_template_sha256()[:12]} (hashed here, not in the experiment config)")
    print(f"config       {cfg.config_version} {cfg.content_hash[:12]} (unchanged)")
    if stage_auth_record is not None:
        print(f"authorised by stage authorisation {stage_auth_record['_path']} "
              f"{stage_auth_record['_file_sha256'][:12]}")
    for target in targets:
        print(f"  {target.scenario_id:<20} supersedes {target.original_call_id[:12]}  "
              f"failed: {', '.join(target.failed_judgements)}")

    problems = live_problems(args.send, cfg)
    if problems:
        print("\nnothing was sent:")
        for problem in problems:
            print(f"  - {problem}")
        return 0

    checks, cached, server = server_problems(args, cfg)
    checks = checks + run_directory_problems(store, cfg)
    if checks:
        print("\nrefusing to run; no call was made:", file=sys.stderr)
        for problem in checks:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    store.cached, store.server = cached, server
    store.endpoint = generator_endpoint(cfg)

    before = len(store.log.entries())
    try:
        results = run_redraft_stage(topics, approvals, cfg, segmenter_from_config(cfg),
                                    _live_backend(cfg), store, allow_live=True,
                                    stage_authorization=stage_auth_record)
    except PipelineAbort as exc:
        print(f"\nthe stage stopped: {exc}", file=sys.stderr)
        return 1
    made = len(store.log.entries()) - before
    accepted = sum(1 for r in results if r.outcome == ACCEPTED)
    print(f"\n{accepted}/{len(results)} redrafts are machine-valid and changed")
    print(f"calls made   {made} (ceiling {len(targets)})")
    print("\nEvery redraft is unapproved until you read its new text: run "
          "'pilot.py scenario-review', record decisions, then the groups stage.")
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
    scenarios = _current_scenarios(args, cfg, store)
    out = Path(args.review_out)
    out.mkdir(parents=True, exist_ok=True)

    from reasonstyle.generation.approvals import approval_status
    approvals, gate_hash, source = _gate(args, cfg, store)
    states = {}
    for scenario_id, record in scenarios.items():
        if not record.get("scenario_text"):
            continue
        states[scenario_id] = approval_status(
            scenario_id, record["scenario_text"], record["call_id"], approvals,
            config_content_hash=gate_hash, topic_bank_content_hash=bank_hash,
            machine_errors=len(record.get("error_codes") or []))[0]
    counts: dict[str, int] = {}
    for state in states.values():
        counts[state] = counts.get(state, 0) + 1

    lines = ["# Pilot scenarios, for review", "",
             f"config `{cfg.config_version}` `{cfg.content_hash[:12]}` · topic bank "
             f"`{bank_hash[:12]}`", "",
             "Read each scenario against its brief, then record a decision in the approvals "
             "file. Approving binds the exact text below: any edit makes the approval stale.",
             "",
             "| state | scenarios |", "|---|---|"]
    lines += [f"| {state} | {count} |" for state, count in sorted(counts.items())]
    if counts == {"approved": len(states)}:
        lines += ["", "All current scenario texts are approved; no scenario judgement is "
                  "outstanding. Generated text and human corrections remain separately "
                  "identified below.", ""]
    else:
        lines += ["", "A scenario that already carries an approval needs nothing further. "
                  "One shown as **redrafted** carries new text that has not been reviewed: "
                  "its seven judgements below are unanswered.", ""]
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
            state = states.get(scenario_id, "not_generated")
            approval = approvals.get(scenario_id)
            superseded = record.get("supersedes_call_id")
            correction = record.get("scenario_correction")
            heading = {"approved": "approved, no action needed",
                       "pending": "awaiting your decision",
                       "redraft": "you asked for a redraft",
                       "stale": "the approval no longer matches this text",
                       }.get(state, state)
            if superseded:
                heading = ("approved redraft, no action needed" if state == "approved"
                           else "REDRAFTED — new text, seven judgements unanswered")
            if correction:
                heading = "HUMAN-CORRECTED — approved exact text"
            lines += [
                f"## {scenario_id} — {heading}", "",
                f"- decision `{topic.decision_id}` · domain `{topic.domain}` · variant "
                f"{variant_id}",
                f"- call `{record.get('call_id', '(not generated)')}`",
                f"- text sha256 `{sha256_of(text) if text else '(none)'}`",
                f"- config `{cfg.content_hash}`",
                f"- topic bank `{bank_hash}`",
                f"- machine findings: {', '.join(machine) if machine else 'none'}", "",
                f"- gate state: **{state}**", ""]
            if superseded:
                lines += [
                    f"- this text SUPERSEDES call `{superseded}`, which you rejected",
                    f"- your reason then: "
                    f"{record.get('redraft_failure_reason') or '(not recorded)'}",
                    f"- judgements that failed then: "
                    f"{', '.join(record.get('redraft_failed_judgements') or []) or 'none recorded'}",
                    ""]
            if correction:
                lines += [
                    f"- human correction by {correction['editor']} on "
                    f"{correction['decided_at']}, bound to call "
                    f"`{correction['original_call_id']}`",
                    f"- correction reason: {correction['reason']}",
                    f"- original text sha256 `{correction['original_text_sha256']}`",
                    f"- corrected text sha256 `{correction['corrected_text_sha256']}`", ""]
            elif state == "redraft" and approval is not None:
                lines += [f"- your reason: {approval.reason}",
                          f"- judgements that failed: "
                          f"{', '.join(sorted(n for n, v in approval.judgements.items() if v is False)) or 'none recorded'}"]
                if not superseded:
                    # Only while the redraft is still outstanding. Once one has
                    # been generated and accepted, the text printed above IS the
                    # redraft, and saying it had not been generated would point
                    # the curator's judgement at the wrong words.
                    lines.append("- **this is the original text; its redraft has not been "
                                 "generated yet**")
                else:
                    lines.append("- **the text above is the redraft; it is awaiting your "
                                 "decision, and the seven judgements below are for it**")
                lines.append("")
            elif state == "approved" and approval is not None:
                lines += [f"- approved by {approval.decided_by} on {approval.decided_at}", ""]
            lines += [
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
                      "```", text or "(no scenario recorded)", "```", ""]
            if state == "approved":
                lines += ["Already approved; no judgement is outstanding.", ""]
            else:
                lines += ["Judgements to record:", ""]
                lines += [f"- [ ] {name}" for name in REQUIRED_JUDGEMENTS]
                lines.append("")
            if state == "approved":
                continue                 # an approved scenario needs no new template entry
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
    if args.write_template and source is not None:
        print("no approval template was written: these scenarios are already approved "
              f"under {scenario_source_spec(cfg)['config']}, and a template here would "
              f"invite a second approval of text that already has one.", file=sys.stderr)
    elif args.write_template:
        import yaml
        path = out / "scenario_approvals.template.yaml"
        path.write_text(yaml.safe_dump(template, sort_keys=True, allow_unicode=True),
                        encoding="utf-8")
        written.append(path)
    for path in written:
        print(f"wrote {path}")
    recorded = sum(1 for s in scenarios.values() if s.get("scenario_text"))
    print(f"{recorded} scenario(s) with recorded text; "
          + ", ".join(f"{count} {state}" for state, count in sorted(counts.items())))
    if args.write_template and source is None:
        print(f"the template holds the {len(template)} scenario(s) still needing a "
              f"decision; every one is 'pending' with null judgements. Approving is "
              f"yours, and the file is where you do it.")
    return 0


def _group_review(args, cfg, bank, allocation, topics, store: CallStore) -> int:
    """A deterministic, read-only reading view of the recorded group run.

    It reads the log, the per-call results, the approvals and the scenario
    correction ledger, and writes Markdown under ``--group-review-out``. It
    makes no call, repairs nothing, approves nothing, corrects nothing, writes
    nothing into the run directory and touches no corpus. It proposes no
    corrected wording either: what to write instead is the curator's act,
    recorded in an approved ledger, and a tool drafting it here would be making
    that decision for them.
    """
    bank_hash = content_hash(bank.model_dump(mode="json"))
    approvals, gate_hash, _ = _gate(args, cfg, store)
    export = build_group_review(
        topics=topics, allocation_groups=allocation.groups,
        scenarios=_current_scenarios(args, cfg, store), store=store, cfg=cfg,
        segmenter=segmenter_from_config(cfg),
        approvals=approvals,
        topic_bank_content_hash=bank_hash,
        allocation_content_hash=allocation.content_hash,
        variants=tuple(args.variants),
        approval_config_content_hash=gate_hash,
        scenario_source=_scenario_provenance(args, cfg))
    out = Path(args.group_review_out)
    export.write(out)

    manifest = export.manifest
    print(f"wrote {len(export.files)} file(s) + MANIFEST.json to {out}/")
    print(f"config       {cfg.config_version} {cfg.content_hash[:12]}")
    print(f"run          {store.directory} (read only; nothing was written there)")
    print(f"groups expected         {manifest['expected_groups']}")
    print(f"machine-valid           {manifest['machine_valid']}")
    print(f"requiring correction    {manifest['needing_correction']}")
    if manifest["not_recorded"]:
        print(f"no recorded call        {manifest['not_recorded']}")
    print(f"calls recorded          {manifest['calls_recorded']}")
    unchanged = sum(1 for g in manifest["groups"] for a in g["attempts"]
                    if a["unchanged_from_previous"])
    identical = sum(1 for g in manifest["groups"] if g["identical_bodies_throughout"])
    print(f"repairs returning unchanged text  {unchanged}")
    print(f"groups identical at every attempt {identical}")
    flagged = [g for g in manifest["groups"]
               if g["earlier_attempt_with_fewer_error_codes"]]
    print(f"earlier attempt with fewer error codes {len(flagged)} (informational: the final "
          f"attempt stays canonical, and fewer codes is not better text)")
    diverged = [g["scenario_id"] for g in manifest["groups"] if not g["recorded_codes_agree"]]
    if diverged:
        print(f"NOTE the recomputed findings differ from the recorded codes for "
              f"{len(diverged)} group(s); the pages say so rather than resolving it",
              file=sys.stderr)
    print(f"\nindex        {out / 'index.md'}")
    print(f"combined     {out / 'all_groups.md'}")
    print("\nThis pass is INSPECTION ONLY. No correction ledger was written or populated, "
          "no wording was proposed, and no generator, model or validator decision follows "
          "from it.")
    print("Machine-valid is not approval: every human judgement listed under each group is "
          "outstanding, for the machine-valid groups exactly as much as for the rest.")
    return 0


def _assemble(args, cfg, bank, allocation, topics, store: CallStore) -> int:
    """Build the corpus from what is recorded, or build nothing."""
    segmenter = segmenter_from_config(cfg)
    approvals, gate_hash, _ = _gate(args, cfg, store)
    try:
        records, manifest = assemble_pilot(
            topics=topics, allocation_groups=allocation.groups,
            approvals=approvals,
            approval_config_content_hash=gate_hash,
            scenario_source=_scenario_provenance(args, cfg),
            corrections=load_corrections(args.corrections_file),
            scenarios=_current_scenarios(args, cfg, store),
            groups=recorded_groups(store),
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
    approvals, gate_hash, source = _gate(args, cfg, store)
    scenarios = _current_scenarios(args, cfg, store)
    groups = recorded_groups(store)
    approvals, gate_hash, source = _gate(args, cfg, store)
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
                config_content_hash=gate_hash, topic_bank_content_hash=bank_hash,
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


#: The pilot's measured rates, used only for the *expected* call estimate: 4 of
#: the 24 hosted scenarios were sent back for a redraft, and every one of the 48
#: v2 groups was accepted on its first call. Ceilings never use them.
PILOT_REDRAFT_RATE = (4, 24)


def call_budget(cfg, plan) -> dict[str, int]:
    """Expected calls and the two ceilings, for the new material only."""
    per_group = cfg.raw["corpus"]["repair"]["max_calls_per_group"]
    scenarios, groups = len(plan.new_scenario_ids), len(plan.new_groups)
    redrafts, of = PILOT_REDRAFT_RATE
    return {"scenarios": scenarios, "groups": groups,
            "expected": scenarios + scenarios * redrafts // of + groups,
            "primary_ceiling": scenarios + groups * per_group,
            "absolute_ceiling": scenarios + groups * per_group + scenarios}


def _allocation_check(cfg, bank, allocation, seed, plan, path) -> list[str]:
    """The stored full allocation against the seed, the plan and a fresh build."""
    from reasonstyle.generation.allocation import (
        allocate_full_markers,
        full_allocation_problems,
        render_full_allocation,
    )
    spec = seed_corpus_spec(cfg)
    problems = []
    rows = {(g.scenario_id, g.supported_option) for g in allocation.groups}
    seed_rows = {(g.scenario_id, g.supported_option) for g in seed.allocation.groups}
    if rows - seed_rows != set(plan.new_groups):
        problems.append("the full allocation's new rows are not the planned new groups")
    for g in seed.allocation.groups:
        if allocation.for_group(g.scenario_id, g.supported_option) != g:
            problems.append(f"seed row {g.scenario_id}/{g.supported_option} differs")
    seeded = allocate_full_markers(bank, cfg, seed.allocation,
                                   seed_allocation_path=spec["allocation"])
    problems += full_allocation_problems(seeded, bank, cfg)
    # The file actually loaded for this command, which is what a run would use.
    path = Path(path)
    if not path.is_file() or path.read_text(encoding="utf-8") != render_full_allocation(seeded):
        problems.append(f"{path} is not byte-identical to a fresh build")
    return problems


def authorization(cfg) -> dict[str, Any]:
    """The recorded authorisation for paid calls, or an empty mapping."""
    return (cfg.raw["corpus"].get("generation_authorization") or {}) if cfg is not None else {}


def paid_call_problems(cfg, kind: str, planned_calls: int | None = None) -> list[str]:
    """Why this stage is not covered by the researcher's recorded authorisation.

    Lifting the generation block says the design is ready. It does not say what
    may be paid for: that is recorded per stage in the configuration, and
    checked here before any backend is built or credential read. A stage the
    authorisation does not name is refused however many keys are set.
    """
    auth = authorization(cfg)
    if not auth:
        # A design that records no scoped authorisation is governed by its own
        # gates — the generation block, --send and the two environment keys —
        # exactly as before. This check narrows those; it never replaces them.
        return []
    problems = []
    if auth.get("status") != "authorized":
        problems.append(f"the recorded authorisation is {auth.get('status')!r}, not 'authorized'")
    allowed = list(auth.get("allowed_kinds") or ())
    if kind not in allowed:
        problems.append(
            f"a {kind!r} call is outside the recorded authorisation. It covers "
            f"{auth.get('scope')!r} ({allowed}); excluded: "
            f"{', '.join(auth.get('excludes') or [])}. A further stage needs a further "
            f"authorisation recorded in the configuration.")
    ceiling = auth.get("max_paid_calls")
    if planned_calls is not None and ceiling is not None and planned_calls > ceiling:
        problems.append(f"this stage would make {planned_calls} call(s); the authorisation "
                        f"covers at most {ceiling}")
    return problems


def offline_dry_run(cfg, bank, allocation, topics, plan) -> tuple[dict[str, int], list[str]]:
    """Render every request a full run would send, in memory. Writes nothing.

    This is the dry run itself, not a record that one happened: each scenario
    and group request is built and discarded, so a template, placeholder or
    allocation fault is found here rather than on a paid call. It is what lets
    ``status`` and ``preflight`` report the offline checks as satisfied without
    a state file anyone could forge.
    """
    problems: list[str] = []
    seed_ids = set(plan.seed_decisions)
    by_group = {(g.decision_id, g.variant_id, g.supported_option): g for g in allocation.groups}
    scenarios = groups = 0
    for topic in sorted(topics, key=lambda t: t.decision_id):
        if topic.decision_id in seed_ids:
            problems.append(f"{topic.decision_id} is a seed decision and must not be planned")
            continue
        for variant_id in plan.variants:
            try:
                scenario_request(topic, variant_id, cfg)
            except RequestError as exc:
                problems.append(f"{topic.decision_id}_v{variant_id}: {exc}")
                continue
            scenarios += 1
            for option in cfg.raw["corpus"]["supported_options"]:
                group = by_group.get((topic.decision_id, variant_id, option))
                if group is None:
                    problems.append(f"{topic.decision_id}_v{variant_id}/{option}: no allocation")
                    continue
                try:
                    # The brief's framing stands in for scenario text, exactly as
                    # the `plan` command does: no scenario exists in a dry run.
                    group_request(topic, variant_id, topic.decision_framing, group, cfg)
                except RequestError as exc:
                    problems.append(f"{topic.decision_id}_v{variant_id}/{option}: {exc}")
                    continue
                groups += 1
    counts = {"scenarios": scenarios, "groups": groups}
    if scenarios != len(plan.new_scenario_ids):
        problems.append(f"{scenarios} scenario requests planned; the plan needs "
                        f"{len(plan.new_scenario_ids)}")
    if groups != len(plan.new_groups):
        problems.append(f"{groups} group requests planned; the plan needs {len(plan.new_groups)}")
    return counts, problems


def _preflight(args, cfg, bank, allocation, seed, plan) -> int:
    """Read-only. Verifies everything a full run would rely on, and prints the
    plan. Builds no backend, reads no credential and opens no connection."""
    problems = _allocation_check(cfg, bank, allocation, seed, plan, args.allocation)
    new_topics = [t for t in bank.topics
                  if t.status == "curated" and t.decision_id in set(plan.new_decisions)]
    dry, dry_problems = offline_dry_run(cfg, bank, allocation, new_topics, plan)
    problems += dry_problems
    block = cfg.raw["corpus"].get("generation_block") or {}
    auth = authorization(cfg)
    scenario_refusals = paid_call_problems(cfg, "scenario", len(plan.new_scenario_ids))
    problems += scenario_refusals
    if not paid_call_problems(cfg, "group", len(plan.new_groups)):
        problems.append("group drafting is authorised; this preflight expects the scenario "
                        "stage only")
    if not seed.provenance["pinned"]:
        problems.append("the seed declaration carries no pins")
    budget = call_budget(cfg, plan)
    p = seed.provenance
    print(f"config            {cfg.config_version} {cfg.content_hash}")
    print(f"seed integrity    verified: {', '.join(p['verified'])}")
    print(f"                  pins: {'all six checked against the configuration' if p['pinned'] else 'NONE declared'}")
    print(f"                  run evidence: {p['run_evidence']}")
    print(f"seed counts       {p['counts']['decisions']} decisions, "
          f"{p['counts']['scenarios']} scenarios, {p['counts']['groups']} groups, "
          f"{p['counts']['texts']} texts ({p['manual_corrections']} approved corrections, "
          f"validation {p['validation_status']})")
    print(f"seed config       {p['config']} {p['config_version']} {p['config_content_hash']}")
    print(f"seed corpus       {p['corpus']} sha256 {p['corpus_sha256']}")
    print(f"seed manifest     {p['manifest']} sha256 {p['manifest_sha256']}")
    print(f"seed allocation   {p['allocation']} {p['allocation_content_hash']}")
    print(f"full topic bank   {args.topics} {content_hash(bank.model_dump(mode='json'))}")
    print(f"full allocation   {args.allocation} {allocation.content_hash} "
          f"({len(allocation.groups)} groups)")
    print(f"skipped (seed)    {len(plan.seed_decisions)}: {' '.join(plan.seed_decisions)}")
    print(f"to generate       {len(plan.new_decisions)}: {' '.join(plan.new_decisions)}")
    print(f"scenarios         {budget['scenarios']} would be requested "
          f"({dry['scenarios']} rendered in this dry run, nothing written)")
    print(f"groups            {budget['groups']} would be requested "
          f"({dry['groups']} rendered in this dry run, nothing written)")
    print(f"new texts         {plan.new_texts}")
    print(f"whole corpus      the figures below are for the eventual full corpus and are "
          f"NOT authorised; only the {len(plan.new_scenario_ids)} scenario calls above are")
    print(f"calls expected    about {budget['expected']} (at the pilot's rates: "
          f"{PILOT_REDRAFT_RATE[0]} of {PILOT_REDRAFT_RATE[1]} scenarios redrafted, every group "
          f"accepted first time)")
    print(f"ceiling, primary  {budget['primary_ceiling']} (scenarios + {budget['groups']} "
          f"groups x {cfg.raw['corpus']['repair']['max_calls_per_group']})")
    print(f"ceiling, absolute {budget['absolute_ceiling']} (every scenario also redrafted once)")
    generator = cfg.raw["models"]["generator"]
    decoding, openai = generator["decoding"], generator["openai"]
    print(f"endpoint          {openai['endpoint']}")
    print(f"model             {generator['model']['id']} (exact id; aliases refused: "
          f"{', '.join(generator['model']['refused_aliases'])})")
    print(f"decoding          temperature {decoding['temperature']}, top_p "
          f"{decoding['top_p'] if decoding['top_p'] is not None else 'provider default (not sent)'}, "
          f"max_output_tokens {decoding['max_output_tokens']}, reasoning effort "
          f"{decoding['reasoning']['effort']}, n {decoding['n']}, seed "
          f"{decoding['seed'] if decoding['seed'] is not None else 'not sent'}")
    print(f"request state     store {openai['store']}, background {openai['background']}, tools "
          f"{openai['tools']}, conversations {openai['conversations']}, previous_response_state "
          f"{openai['previous_response_state']}, automatic_retries "
          f"{openai['automatic_retries']}")
    print(f"authorised        {auth.get('scope')}: {len(plan.new_scenario_ids)} scenario "
          f"call(s), {auth.get('calls_per_scenario')} per scenario, ceiling "
          f"{auth.get('max_paid_calls')} (recorded by {auth.get('authorized_by')} on "
          f"{auth.get('authorized_at')})")
    print(f"group calls       0 authorised ({', '.join(auth.get('excludes') or [])} are "
          f"excluded and refused before a backend is built)")
    print(f"outputs           {profile_paths(cfg)['out']} (gitignored; nothing under data/pilot/)")
    print(f"generation block  {'ACTIVE' if block.get('blocked') else 'lifted ' + str(block.get('lifted_at')) + ' by ' + str(block.get('lifted_by'))}")
    print("network           no credential was read, no backend was built and no connection "
          "was made")
    if problems:
        print("\npreflight FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    auth_line = (f"{auth.get('max_paid_calls')} initial scenario call(s)" if auth
                 else "nothing")
    print(f"\npreflight passed. Nothing was sent. Authorised for {auth_line}; every other "
          f"stage is refused.")
    return 0


def new_scenario_approval_counts(args, cfg, bank, plan, scenarios) -> tuple[dict[str, int], list[str]]:
    """How the new scenarios' approval gate currently reads, one state per drafted scenario.

    Counted live from the approvals file and this run's recorded scenario text
    and machine findings — never from a stored claim. Only scenarios with
    recorded text are counted; an undrafted scenario contributes nothing here
    (``status`` already reports it separately as not yet generated).
    """
    approvals, gate_hash, _source = _gate(args, cfg, None)
    bank_hash = content_hash(bank.model_dump(mode="json"))
    counts: dict[str, int] = {}
    redraft_ids: list[str] = []
    for scenario_id in plan.new_scenario_ids:
        record = scenarios.get(scenario_id) or {}
        text = record.get("scenario_text")
        if not text:
            continue
        state, _reasons = approval_status(
            scenario_id, text, record.get("call_id"), approvals,
            config_content_hash=gate_hash, topic_bank_content_hash=bank_hash,
            machine_errors=len(record.get("error_codes") or []))
        counts[state] = counts.get(state, 0) + 1
        if state == REDRAFT:
            redraft_ids.append(scenario_id)
    return counts, sorted(redraft_ids)


def _next_gate(made_scenarios: int, total_scenarios: int, made_groups: int, total_groups: int,
              approval_counts: dict[str, int], redraft_ids: list[str]) -> str:
    """The single next thing to do, derived from what is actually recorded now.

    Priority, deliberately, is not the order ``approval_status`` happens to
    check states in:

    1. **Machine-blocked** scenarios outrank everything else. A human
       approval can never override a machine error (``approvals.py``), so a
       blocked scenario must never be silently outvoted by a pile of
       redrafts, and "group drafting authorisation" must never appear while
       one is outstanding.
    2. **Unresolved human review** — ``pending``, ``stale`` or
       ``needs_manual_review`` — comes next: these are simply not yet
       decided, and nothing past them (a redraft authorisation, a group
       authorisation) is meaningful until they are.
    3. **Redraft** is the narrow, already-scoped next decision only once
       nothing above it is outstanding.
    4. **Group drafting authorisation** is offered only when the approved
       count is *exactly* the complete expected total — never inferred by
       elimination, so a count that does not add up (a bug, a state this
       function does not yet know about) falls through to the last branch
       instead of falsely claiming every scenario is approved.
    """
    if made_scenarios == 0:
        return f"run the authorised scenario stage: 0 of {total_scenarios} scenario call(s) made"
    if made_scenarios < total_scenarios:
        return (f"finish the scenario stage: {made_scenarios} of {total_scenarios} scenario "
                f"call(s) made")

    blocked_n = approval_counts.get(BLOCKED, 0)
    if blocked_n:
        return (f"{blocked_n} scenario(s) blocked by machine errors need a redraft or "
                f"correction before group drafting can be authorised")

    needs_review_n = (approval_counts.get(PENDING, 0) + approval_counts.get(STALE, 0)
                      + approval_counts.get(SCENARIO_NEEDS_MANUAL_REVIEW, 0))
    if needs_review_n:
        return (f"scenario review of the {total_scenarios} drafted scenarios "
                f"({needs_review_n} still need a decision)")

    redraft_n = approval_counts.get(REDRAFT, 0)
    if redraft_n:
        return (f"decide whether to authorise exactly the {redraft_n} redraft call(s): "
                f"{', '.join(redraft_ids)}")

    approved_n = approval_counts.get(APPROVED, 0)
    if approved_n == total_scenarios:
        if made_groups < total_groups:
            return "group drafting authorisation (every scenario is approved)"
        return "corpus assembly"
    # Every scenario is drafted, and none is blocked, unreviewed or sent to
    # redraft, yet the approved count still is not the complete total: the
    # counts do not add up to a recognised, complete picture. Say so plainly
    # rather than assume the gate is clear.
    return (f"scenario approval counts do not add up to a clear gate: {approved_n} of "
            f"{total_scenarios} approved, recorded states {dict(sorted(approval_counts.items()))} "
            f"— read status by hand")


def _full_status(args, cfg, bank, allocation, seed, plan, store) -> int:
    """Read-only: the seed, what is new, the eventual corpus, and what blocks it."""
    p = seed.provenance
    # Supersession-resolved: a scenario a redraft replaced must be judged by
    # its current, accepted text and call, exactly as `approvals`/
    # `scenario-review` already judge it (`_current_scenarios`) — otherwise
    # this view and those disagree about a redraft the approvals file has
    # since caught up with.
    scenarios = current_scenarios(store) if store.log.path.is_file() else {}
    groups = recorded_groups(store) if store.log.path.is_file() else {}
    made_scenarios = sum(1 for sid in plan.new_scenario_ids
                         if (scenarios.get(sid) or {}).get("scenario_text"))
    made_groups = sum(1 for key in plan.new_groups if key in groups)
    block = cfg.raw["corpus"].get("generation_block") or {}
    total_decisions = len(plan.expected_decisions)
    approval_counts, redraft_ids = new_scenario_approval_counts(args, cfg, bank, plan, scenarios)
    print(f"config                 {cfg.config_version} {cfg.content_hash[:12]}")
    print("\nIMPORTED SEED (read-only, verified; never regenerated)")
    print(f"  decisions            {p['counts']['decisions']}")
    print(f"  scenarios            {p['counts']['scenarios']}")
    print(f"  groups               {p['counts']['groups']}")
    print(f"  texts                {p['counts']['texts']}")
    print(f"  source               {p['corpus']} ({p['config_version']} "
          f"{p['config_content_hash'][:12]}), validation {p['validation_status']}, "
          f"{seed.manifest.get('outstanding_human_review')} human judgements outstanding")
    print("\n" + ("NEW MATERIAL" if made_scenarios or made_groups else "NEW MATERIAL (not generated)"))
    print(f"  decisions            {len(plan.new_decisions)}")
    print(f"  scenarios            {made_scenarios} of {len(plan.new_scenario_ids)} generated")
    if made_scenarios:
        print(f"  scenario review      "
              + ", ".join(f"{n} {state}" for state, n in sorted(approval_counts.items())))
    print(f"  groups               {made_groups} of {len(plan.new_groups)} generated")
    print(f"  texts                0 of {plan.new_texts} assembled")
    print(f"  run directory        {store.directory} "
          f"({'exists' if store.directory.exists() else 'does not exist'})")
    print("\nTOTAL EVENTUAL CORPUS")
    print(f"  decisions            {total_decisions}")
    print(f"  scenarios            {p['counts']['scenarios'] + len(plan.new_scenario_ids)}")
    print(f"  groups               {len(allocation.groups)} allocated "
          f"({p['counts']['groups']} imported + {len(plan.new_groups)} new)")
    print(f"  texts                {p['counts']['texts'] + plan.new_texts}")
    print("\nCURRENT BLOCKERS")
    # Every state below is derived here, now: the allocation is rebuilt and
    # compared, the seed is verified, and the dry run is actually performed.
    # Nothing is read from a stored claim that a check once passed.
    allocation_ok = not _allocation_check(cfg, bank, allocation, seed, plan, args.allocation)
    new_topics = [t for t in bank.topics
                  if t.status == "curated" and t.decision_id in set(plan.new_decisions)]
    dry, dry_problems = offline_dry_run(cfg, bank, allocation, new_topics, plan)
    auth = authorization(cfg)
    if not block.get("blocked"):
        print(f"  generation block     lifted {block.get('lifted_at')} by "
              f"{block.get('lifted_by')}; every requirement was met and checked")
        print(f"  paid calls           {auth.get('scope')}: at most "
              f"{auth.get('max_paid_calls')} scenario call(s); group drafting, redrafts and "
              f"repairs are NOT authorised")
        print(f"  next gate            "
              + _next_gate(made_scenarios, len(plan.new_scenario_ids), made_groups,
                           len(plan.new_groups), approval_counts, redraft_ids))
        print(f"  allocation           {'checked byte for byte' if allocation_ok else 'NOT MET'}")
        print(f"  offline dry run      {dry['scenarios']} scenario and {dry['groups']} group "
              f"requests rendered in this run"
              + ("" if not dry_problems else f" — PROBLEM: {dry_problems[0]}"))
    elif block.get("blocked"):
        print("  generation block     ACTIVE — lifted only by a deliberate edit to the config")
        for item in block.get("requires") or []:
            if "allocation" in item:
                state = "met: rebuilt and compared byte for byte" if allocation_ok else "NOT MET"
            elif "pilot corpus" in item:
                state = "met: verified read-only in this run"
            elif "dry run" in item or "test" in item:
                state = (f"met: {dry['scenarios']} scenario and {dry['groups']} group requests "
                         f"rendered offline in this run" if not dry_problems
                         else f"NOT MET: {dry_problems[0]}")
            elif "authorisation" in item or "authorization" in item:
                state = "OUTSTANDING — the only remaining requirement"
            else:
                state = "outstanding"
            print(f"    - {item}  [{state}]")
    else:
        print("  generation block     not active")
    print("\nNothing here sends a request. Seed decisions are refused on every send path.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "scenarios", "scenario-review", "approvals",
                                       "redraft-scenarios", "groups", "group-review",
                                       "assemble", "status", "log", "preflight"],
                    help="plan; draft the scenarios; export them for review; report the "
                         "gate; redraft the scenarios the curator rejected; draft the "
                         "groups; export the recorded groups for reading; assemble the "
                         "corpus; or report counts")
    ap.add_argument("--approvals-file", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--topics", default=None,
                    help="the topic bank; defaults to the one the configuration names, "
                         "else the pilot bank")
    ap.add_argument("--allocation", default=None)
    ap.add_argument("--out", default=None,
                    help="the run directory; defaults to this generator's own "
                         "(data/pilot/run for the local model, data/pilot/run_openai for "
                         "the hosted one), so the two can never be mixed by accident. The "
                         "approvals file, correction ledgers, corpus and review exports "
                         "default per generator in the same way")
    ap.add_argument("--only", nargs="*", help="limit to these decision ids")
    ap.add_argument("--variants", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--corrections-file", default=None)
    ap.add_argument("--scenario-corrections-file", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--review-out", default=None)
    ap.add_argument("--group-review-out", default=None,
                    help="where group-review writes its Markdown; it writes nowhere else")
    ap.add_argument("--write-template", action="store_true",
                    help="also write a blank approval template (every decision pending)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing corpus deliberately")
    ap.add_argument("--hf-home", help="cache location for the read-only revision check")
    ap.add_argument("--server-runtime", default="data/pilot/server_runtime.json")
    ap.add_argument("--send", action="store_true",
                    help="make the calls of a stage; also needs both authorisation keys")
    ap.add_argument("--stage-authorization", default=None,
                    help="a separate, tracked authorisation record for a paid stage the "
                         "frozen configuration's own generation_authorization excludes by "
                         "name (e.g. redraft-scenarios). Never edits the configuration; a "
                         "stage this flag does not name stays governed by the configuration "
                         "alone, exactly as before.")
    args = ap.parse_args(argv)

    try:
        cfg, bank, allocation, topics = _inputs(args)
    except (AllocationError, FileNotFoundError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
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
    seed = plan = None
    if seed_corpus_spec(cfg):
        # A design that grows from a frozen seed verifies it first, every time,
        # before anything else reads a run or plans a request.
        problems = seed_output_problems(cfg)
        if problems:
            print("refusing: " + "; ".join(problems), file=sys.stderr)
            return 1
        try:
            seed = load_seed_corpus(cfg, bank=bank)
            plan = plan_full_corpus(cfg, bank, seed)
        except SeedCorpusError as exc:
            print(f"refusing: {exc}", file=sys.stderr)
            return 1
        if args.command == "assemble":
            print("refusing: the full corpus is assembled from the verified seed plus a "
                  "completed full run (corpus_source.combine_corpus); no full run exists, so "
                  "nothing is assembled", file=sys.stderr)
            return 1
        if args.command in DRAFTING_COMMANDS:
            topics = [t for t in topics if t.decision_id in set(plan.new_decisions)]
    elif args.command == "preflight":
        print("refusing: preflight checks a seed-aware full-corpus configuration, and "
              f"{args.config} declares no seed_corpus", file=sys.stderr)
        return 1
    out = Path(args.out)
    store = CallStore(out, cfg,
                      topic_bank_content_hash=content_hash(bank.model_dump(mode="json")),
                      allocation_content_hash=allocation.content_hash,
                      refused_decisions=frozenset(plan.seed_decisions) if plan else frozenset(),
                      allowed_kinds=(frozenset(authorization(cfg)["allowed_kinds"])
                                     if authorization(cfg).get("allowed_kinds") else None))
    if args.command == "preflight":
        return _preflight(args, cfg, bank, allocation, seed, plan)
    if args.command == "status" and seed is not None:
        return _full_status(args, cfg, bank, allocation, seed, plan, store)

    try:
        if args.command == "log":
            return _status(store)
        if args.command == "status":
            return _counts(args, cfg, bank, topics, store)
        if args.command == "approvals":
            return _approvals(args, cfg, bank, topics, store)
        if args.command == "scenario-review":
            return _scenario_review(args, cfg, bank, topics, store)
        if args.command == "group-review":
            return _group_review(args, cfg, bank, allocation, topics, store)
        if args.command == "assemble":
            return _assemble(args, cfg, bank, allocation, topics, store)
        if args.command == "redraft-scenarios":
            return _redraft(args, cfg, bank, topics, store)
        if args.command in ("scenarios", "groups"):
            return _stage(args, cfg, bank, allocation, topics, store, kind=args.command)
    except (ScenarioCorrectionError, ScenarioSourceError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
    return _plan(cfg, bank, allocation, topics, out, tuple(args.variants))


if __name__ == "__main__":
    raise SystemExit(main())
