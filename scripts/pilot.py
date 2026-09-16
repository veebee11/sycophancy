"""The pilot drafting controller: scenarios, then groups with bounded repair.

    uv run python scripts/pilot.py plan --config configs/experiment.yaml
    uv run python scripts/pilot.py status --config configs/experiment.yaml
    uv run python scripts/pilot.py approvals --config configs/experiment.yaml

**This script cannot send.** It plans and it reports, and that is all it can do:
there is no live code path in it at any argument or environment combination.
``plan`` covers the whole pilot — scenarios *and* their groups — because a
group is drafted from its scenario's approved text, so no command here can offer
a narrower live scope than the whole pilot. Generation itself is **two
resumable stages separated by curator approval**, not one uninterrupted run.

Two things had to exist before live pilot generation could be implemented at
all. Both are now built and offline-tested — the **curator-approval gate**
(``generation/approvals.py``) and the **corpus assembler**
(``generation/assemble.py``). What is still missing is the live pilot execution
path itself, which is separately authorised work and is not written here, so
``--send`` refuses whatever the environment says.

The two synthetic repair smokes are finished and no further one is planned. They
showed what they were for: the controller sends distinct, diagnosed repairs and
routes unchanged output to ``needs_manual_review``. Whether this model repairs
this fixture automatically is not a prerequisite for anything — an unrepaired
group is a reviewed group, which is what ``needs_manual_review`` means.

Pilot generation is **two resumable stages, not one run**: every scenario is
drafted first, the curator approves each one against its exact text, and only
then are groups drafted. ``approvals`` reports where that gate stands.

``approvals`` reports the gate: which scenarios are approved, pending, stale,
refused or blocked by machine errors. It reads files and contacts nothing. The controller
itself was exercised by ``scripts/pipeline_smoke.py``, which drafts ONE
synthetic group with its repair path and nothing else. It ran twice on
2026-09-16 and those runs are finished; no further synthetic repair smoke is
planned.

What the controller does, when it is eventually authorised, is in
``src/reasonstyle/generation/pipeline.py``: one call per scenario, one draft
plus at most two validator-driven repairs per group, every call recorded, and
``needs_manual_review`` when a budget runs out. Machine-valid is not approved:
the human judgements the validator lists stay outstanding either way.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from reasonstyle.config import load_config
from reasonstyle.corpus.topics import load_topic_bank
from reasonstyle.generation import CallStore, load_allocation, write_request
from reasonstyle.generation.requests import group_request, scenario_request
from reasonstyle.hashing import content_hash

#: The second key live pilot generation would need, once it exists. Named here
#: so the refusal can say what it is; setting it changes nothing today.
PILOT_AUTHORIZATION_ENV = "REASONSTYLE_ALLOW_PILOT_GENERATION"

#: What must be built before a live pilot path may be written at all. While this
#: is non-empty, `--send` refuses regardless of the environment.
MISSING_PREREQUISITES = (
    "the live two-stage pilot execution path, which is not written: the gate "
    "(generation/approvals.py) and the per-scenario assembly core "
    "(generation/assemble.py) exist and are offline-tested, but nothing drives them "
    "against a server, and no corpus-wide assembly driver writes data/pilot/corpus.jsonl",
)


def live_problems(send: bool, cfg=None, env=None) -> list[str]:
    """Why this script will not send. Never empty: it cannot send at all.

    Deliberately independent of the environment. Pilot generation is blocked on
    work that does not exist, so no combination of keys can unblock it, and
    there is no live branch below for one to reach.
    """
    env = os.environ if env is None else env
    problems = [f"not implemented: {item}" for item in MISSING_PREREQUISITES]
    if send:
        problems.append(
            f"--send is refused: even with {PILOT_AUTHORIZATION_ENV}=1 this script has no "
            f"live path, and the two-stage pilot execution it would need is not written.")
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "status", "approvals"],
                    help="plan the whole pilot (scenarios and their groups), report what has "
                         "been recorded so far, or report the curator-approval gate")
    ap.add_argument("--approvals-file", default="data/pilot/scenario_approvals.yaml")
    ap.add_argument("--config", required=True)
    ap.add_argument("--topics", default="data/topics/pilot_topics.yaml")
    ap.add_argument("--allocation", default="data/pilot/marker_allocation.yaml")
    ap.add_argument("--out", default="data/pilot/run")
    ap.add_argument("--only", nargs="*", help="limit to these decision ids")
    ap.add_argument("--variants", nargs="*", type=int, default=[1, 2])
    ap.add_argument("--send", action="store_true",
                    help="attempt a live run; refused until pilot generation is authorised")
    args = ap.parse_args(argv)

    cfg, bank, allocation, topics = _inputs(args)
    out = Path(args.out)
    store = CallStore(out, cfg,
                      topic_bank_content_hash=content_hash(bank.model_dump(mode="json")),
                      allocation_content_hash=allocation.content_hash)

    if args.command == "status":
        return _status(store)
    if args.command == "approvals":
        return _approvals(args, cfg, bank, topics, store)

    # There is no live branch in this file. `--send` reaches this refusal and
    # stops; nothing below it can contact a backend.
    print("this script cannot send:")
    for problem in live_problems(args.send, cfg):
        print(f"  - {problem}")
    print()
    return _plan(cfg, bank, allocation, topics, out, tuple(args.variants))


if __name__ == "__main__":
    raise SystemExit(main())
