"""The pilot drafting controller: scenarios, then groups with bounded repair.

    uv run python scripts/pilot.py plan --config configs/experiment.yaml
    uv run python scripts/pilot.py status --config configs/experiment.yaml

**This script cannot send.** It plans and it reports, and that is all it can do:
there is no live code path in it at any argument or environment combination.
``plan`` covers the whole pilot — scenarios *and* their groups — because they
are one run: the groups of a scenario are drafted from that scenario's accepted
text, so no command here can offer a narrower live scope than the whole pilot.

Two things must exist before live pilot generation is implemented at all, and
neither does yet:

1. the **curator-approval gate** — every scenario approved against its exact
   text hash before any group is drafted from it;
2. the **corpus assembler** — the step that turns accepted drafts into
   ``data/pilot/corpus.jsonl``.

Until then ``--send`` refuses, whatever the environment says. The controller
itself is exercised by ``scripts/pipeline_smoke.py``, which drafts ONE synthetic
group with its repair path and nothing else.

What the controller does, when it is eventually authorised, is in
``src/reasonstyle/generation/pipeline.py``: one call per scenario, one draft
plus at most two validator-driven repairs per group, every call recorded, and
``needs_manual_review`` when a budget runs out. Machine-valid is not approved:
the human judgements the validator lists stay outstanding either way.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
    "the curator-approval gate: every scenario approved against its exact text hash "
    "before any group is drafted from it",
    "the corpus assembler: accepted drafts written to data/pilot/corpus.jsonl",
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
            f"live path. One synthetic group with its repair path can be run through "
            f"scripts/pipeline_smoke.py instead.")
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "status"],
                    help="plan the whole pilot (scenarios and their groups), or report "
                         "what has been recorded so far")
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

    # There is no live branch in this file. `--send` reaches this refusal and
    # stops; nothing below it can contact a backend.
    print("this script cannot send:")
    for problem in live_problems(args.send, cfg):
        print(f"  - {problem}")
    print()
    return _plan(cfg, bank, allocation, topics, out, tuple(args.variants))


if __name__ == "__main__":
    raise SystemExit(main())
