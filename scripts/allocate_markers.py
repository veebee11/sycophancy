"""Build the pilot marker allocation, deterministically, before any drafting.

    uv run python scripts/allocate_markers.py --config configs/experiment.yaml \
        --topics data/topics/pilot_topics.yaml --out data/pilot/marker_allocation.yaml
    ... --check   rebuild and fail if the file on disk differs
    ... --table   print the full allocation table

The drafting scripts read this file; they never choose a marker themselves. The
allocation is checked against every rule in the configuration before it is
written, and the file records its own content hash so a hand edit is detected.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from reasonstyle.config import load_config
from reasonstyle.corpus.topics import TopicBankError, load_topic_bank
from reasonstyle.generation.allocation import (
    AllocationError,
    allocate_markers,
    allocation_problems,
    load_allocation,
    save_allocation,
)


def _table(alloc) -> str:
    L = ["| decision | domain | variant | opt_1 realization | opt_2 realization | family | marker |",
         "|---|---|---|---|---|---|---|"]
    by_slot: dict[tuple[str, int], dict] = {}
    for g in alloc.groups:
        by_slot.setdefault((g.decision_id, g.variant_id), {})[g.supported_option] = g
    for (decision_id, variant_id), pair in sorted(by_slot.items()):
        one, two = pair["opt_1"], pair["opt_2"]
        L.append(f"| {decision_id} | {one.domain} | v{variant_id} | "
                 f"{one.marker_realization_id} | {two.marker_realization_id} | "
                 f"{one.marker_family} | {one.marker_string!r} |")
    counts = Counter(g.marker_string for g in alloc.groups)
    decisions = {m: len({g.decision_id for g in alloc.groups if g.marker_string == m})
                 for m in sorted(counts)}
    L += ["", "| marker | groups | decisions |", "|---|---|---|"]
    L += [f"| {m!r} | {counts[m]} | {decisions[m]} |" for m in sorted(counts)]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--out", default="data/pilot/marker_allocation.yaml")
    ap.add_argument("--check", action="store_true",
                    help="rebuild and fail if the stored allocation differs")
    ap.add_argument("--table", action="store_true", help="print the allocation table")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    try:
        bank = load_topic_bank(args.topics)
        alloc = allocate_markers(bank, cfg)
    except (TopicBankError, AllocationError) as exc:
        print(f"cannot allocate:\n  {exc}", file=sys.stderr)
        return 1

    problems = allocation_problems(alloc, cfg)
    if problems:
        print("the allocation violates its own rules:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    if args.check:
        try:
            stored = load_allocation(args.out)
        except (AllocationError, FileNotFoundError) as exc:
            print(f"cannot read the stored allocation: {exc}", file=sys.stderr)
            return 1
        if stored.content_hash != alloc.content_hash:
            print(f"the stored allocation differs from a fresh build\n"
                  f"  stored {stored.content_hash[:12]}, rebuilt {alloc.content_hash[:12]}",
                  file=sys.stderr)
            return 1
        print(f"allocation matches its inputs ({len(alloc.groups)} groups, "
              f"{alloc.content_hash[:12]}).")
        return 0

    save_allocation(alloc, args.out)
    print(f"wrote {len(alloc.groups)} groups to {args.out}")
    print(f"  seed {alloc.seed}, allocation {alloc.content_hash[:12]}, "
          f"config {cfg.config_version} {cfg.content_hash[:12]}")
    for family, n in sorted(Counter(g.marker_family for g in alloc.groups).items()):
        markers = sorted({g.marker_string for g in alloc.groups if g.marker_family == family})
        print(f"  {family:<26} {n} groups, markers {markers}")
    if args.table:
        print("\n" + _table(alloc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
