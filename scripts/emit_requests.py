"""Write drafting requests to files, so they can be read before anything is sent.

    uv run python scripts/emit_requests.py --config configs/experiment.yaml \
        --topics data/topics/pilot_topics.yaml --kind scenario
    uv run python scripts/emit_requests.py --config configs/experiment.yaml \
        --topics data/topics/pilot_topics.yaml --kind group \
        --allocation data/pilot/marker_allocation.yaml --scenarios data/pilot/scenarios.yaml

Nothing is sent here. A request file holds the rendered prompt, the template
and its hash, the response schema, and the keys of what it drafts. Group
requests need the drafted scenario texts, so scenarios are emitted, answered
and imported first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from reasonstyle.config import load_config
from reasonstyle.corpus.schemas import SEMANTIC_OPTIONS
from reasonstyle.corpus.topics import load_topic_bank
from reasonstyle.generation import (
    AllocationError,
    RequestError,
    group_request,
    load_allocation,
    scenario_request,
    write_request,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--kind", required=True, choices=["scenario", "group"])
    ap.add_argument("--allocation", default="data/pilot/marker_allocation.yaml")
    ap.add_argument("--scenarios", default="data/pilot/scenarios.yaml",
                    help="drafted scenario texts; required for --kind group")
    ap.add_argument("--out", default="data/pilot/requests")
    ap.add_argument("--only", help="restrict to one decision_id")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    bank = load_topic_bank(args.topics)
    topics = [t for t in bank.topics if t.status == "curated"
              and (args.only is None or t.decision_id == args.only)]
    if not topics:
        print("no curated topics selected", file=sys.stderr)
        return 1
    variants = [int(v[1:]) for v in cfg.raw["topics"]["variants"]]

    requests = []
    try:
        if args.kind == "scenario":
            for topic in topics:
                for variant_id in variants:
                    requests.append(scenario_request(topic, variant_id, cfg))
        else:
            alloc = load_allocation(args.allocation)
            scenarios = yaml.safe_load(Path(args.scenarios).read_text(encoding="utf-8"))
            for topic in topics:
                for variant_id in variants:
                    scenario_id = f"{topic.decision_id}_v{variant_id}"
                    if scenario_id not in scenarios:
                        print(f"no drafted scenario for {scenario_id}", file=sys.stderr)
                        return 1
                    for option in SEMANTIC_OPTIONS:
                        requests.append(group_request(
                            topic, variant_id, scenarios[scenario_id],
                            alloc.for_group(scenario_id, option), cfg))
    except (RequestError, AllocationError, FileNotFoundError) as exc:
        print(f"cannot build requests:\n  {exc}", file=sys.stderr)
        return 1

    for request in requests:
        write_request(request, args.out)
    print(f"wrote {len(requests)} {args.kind} request(s) to {args.out}/")
    print(f"  config {cfg.config_version} {cfg.content_hash[:12]}")
    print("  nothing was sent: review the files, then run the approved backend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
