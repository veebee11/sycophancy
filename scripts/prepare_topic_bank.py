"""Check the topic bank against the configuration and the source registry.

    uv run python scripts/prepare_topic_bank.py --config configs/experiment.yaml \
        --registry data/sources/registry.yaml --topics data/topics/pilot_topics.yaml \
        --source-texts data/sources/raw
    ... --require-ready    also fail unless drafting may begin

--source-texts is required: every brief is screened for runs of six or more
words shared with the downloaded source text. Pass `none` only for synthetic
fixtures; the output then says plainly that the screen was not run.

Fails on any machine error. With --require-ready it also fails unless there are
exactly the required number of curated decisions per domain; proposed and
rejected candidates are not counted.
"""

from __future__ import annotations

import argparse
import sys

from reasonstyle.config import load_config
from reasonstyle.corpus.source_texts import load_source_texts, screen_summary
from reasonstyle.corpus.sources import SourceRegistryError, load_registry
from reasonstyle.corpus.topics import TopicBankError, check_topics, load_topic_bank


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--source-texts", required=True, metavar="RAW_DIR|none",
                    help="downloaded source files to screen briefs against, or 'none'")
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
    texts = {} if args.source_texts == "none" else load_source_texts(args.source_texts)
    if args.source_texts != "none" and not texts:
        print(f"no source text found under {args.source_texts}", file=sys.stderr)
        return 1
    report = check_topics(bank, cfg, registry, source_texts=texts)

    for topic in sorted(bank.topics, key=lambda t: t.decision_id):
        own = report.for_decision(topic.decision_id)
        print(f"  {topic.decision_id:<16} {topic.domain:<11} {topic.status:<9} "
              f"{sum(f.severity == 'error' for f in own)} error(s), "
              f"{sum(f.severity == 'warning' for f in own)} warning(s)")
        for f in own:
            print(f"      {f.severity}: {f.code} — {f.message}")

    counts = ", ".join(f"{d} {n}" for d, n in sorted(report.curated_per_domain.items()))
    print(f"\ncurated per domain: {counts} (need {report.required_per_domain} each)")
    if texts:
        summary = screen_summary(texts)
        print(f"overlap screen: {summary['documents']} source documents, "
              f"{summary['words']:,} words")
    else:
        print("overlap screen: NOT RUN (--source-texts none)")
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
