"""Check the topic bank against the configuration and the source registry.

    uv run python scripts/prepare_topic_bank.py --config configs/experiment.yaml \
        --registry data/sources/registry.yaml --topics data/topics/pilot_topics.yaml
    ... --require-ready    also fail unless drafting may begin

Fails on any machine error. With --require-ready it also fails unless there are
exactly the required number of curated decisions per domain; proposed and
rejected candidates are not counted.
"""

from __future__ import annotations

import argparse
import sys

from reasonstyle.config import load_config
from reasonstyle.corpus.sources import SourceRegistryError, load_registry
from reasonstyle.corpus.topics import TopicBankError, check_topics, load_topic_bank


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--require-ready", action="store_true",
                    help="fail unless there are exactly enough curated decisions per domain")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    try:
        registry = load_registry(args.registry)
        bank = load_topic_bank(args.topics)
    except (SourceRegistryError, TopicBankError) as exc:
        print(f"cannot load:\n{exc}", file=sys.stderr)
        return 1
    report = check_topics(bank, cfg, registry)

    for topic in sorted(bank.topics, key=lambda t: t.decision_id):
        own = report.for_decision(topic.decision_id)
        print(f"  {topic.decision_id:<16} {topic.domain:<11} {topic.status:<9} "
              f"{sum(f.severity == 'error' for f in own)} error(s), "
              f"{sum(f.severity == 'warning' for f in own)} warning(s)")
        for f in own:
            print(f"      {f.severity}: {f.code} — {f.message}")

    counts = ", ".join(f"{d} {n}" for d, n in sorted(report.curated_per_domain.items()))
    print(f"\ncurated per domain: {counts} (need {report.required_per_domain} each)")
    print(f"config {cfg.config_version} {cfg.content_hash[:12]}")

    if report.errors:
        print(f"{len(report.errors)} machine error(s)", file=sys.stderr)
        return 1
    if args.require_ready and not report.ready_for_drafting:
        print("not ready for drafting:\n  " + "\n  ".join(report.readiness_problems()),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
