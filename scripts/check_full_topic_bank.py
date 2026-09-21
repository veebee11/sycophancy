"""Check the full topic bank as a whole, against the pilot it grew from.

    uv run python scripts/check_full_topic_bank.py \
        --config configs/experiment_v2_full.draft.yaml \
        --pilot-topics data/topics/pilot_topics.yaml \
        --registry data/sources/registry.yaml --source-texts data/sources/raw

Counts, unique ids and propositions, byte-for-byte pilot preservation, registry
keys, locators checked against the pinned source files, a lexical near-duplicate
screen, and the absence of anything generated from the bank. Per-brief checks
(word limits, overlap, citability) are prepare_topic_bank.py's job.

Exits 1 on any error. Warnings are listed for the curator and do not fail.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reasonstyle.config import load_config
from reasonstyle.corpus.full_bank import check_full_bank, near_duplicates
from reasonstyle.corpus.sources import load_registry
from reasonstyle.corpus.topics import TopicBankError, load_topic_bank


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pilot-topics", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--source-texts", required=True, metavar="RAW_DIR|none")
    ap.add_argument("--root", default=".")
    ap.add_argument("--top", type=int, default=5, help="most similar pairs to list")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    bank_path = Path(cfg.raw["topics"]["bank"])
    try:
        bank = load_topic_bank(bank_path)
        pilot = load_topic_bank(args.pilot_topics)
    except TopicBankError as exc:
        print(f"cannot load:\n{exc}", file=sys.stderr)
        return 1
    registry = load_registry(args.registry)
    per_domain = cfg.raw["topics"]["curated_per_domain_for_drafting"]
    findings = check_full_bank(
        bank=bank,
        bank_text=bank_path.read_text(encoding="utf-8"),
        pilot=pilot,
        pilot_text=Path(args.pilot_topics).read_text(encoding="utf-8"),
        domains=list(cfg.raw["domains"]["ids"]),
        per_domain=per_domain,
        registry_keys=registry.datasets,
        raw_dir=None if args.source_texts == "none" else Path(args.source_texts),
        output_paths=cfg.raw["paths"],
        root=Path(args.root),
        generation_blocked=bool(cfg.raw["corpus"]["generation_block"]["blocked"]),
    )

    statuses = {s: sum(t.status == s for t in bank.topics) for s in ("curated", "proposed", "rejected")}
    domains = list(cfg.raw["domains"]["ids"])
    in_play = {d: sum(t.domain == d and t.status != "rejected" for t in bank.topics) for d in domains}
    print(f"{bank_path}: {len(bank.topics)} records — {statuses}")
    print("in play per domain: " + ", ".join(f"{d} {n}" for d, n in in_play.items())
          + f" (needs {per_domain} each)")
    print(f"pilot records checked: {len(pilot.topics)}")
    active = [t for t in bank.topics if t.status != "rejected"]
    print(f"most similar pairs (of {len(active) * (len(active) - 1) // 2}):")
    for score, a, b in near_duplicates(active)[:args.top]:
        print(f"  {score:.2f}  {a}  {b}")
    for f in findings:
        where = f" [{f.decision_id}]" if f.decision_id else ""
        print(f"  {f.severity}: {f.code}{where} — {f.message}")
    errors = sum(f.severity == "error" for f in findings)
    warnings = len(findings) - errors
    print(f"{errors} error(s), {warnings} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
