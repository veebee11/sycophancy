"""Build the pilot marker allocation, deterministically, before any drafting.

    uv run python scripts/allocate_markers.py --config configs/experiment.yaml \
        --topics data/topics/pilot_topics.yaml --out data/pilot/marker_allocation.yaml
    ... --check   rebuild and fail if the file on disk differs
    ... --table   print the full allocation table

Full-corpus mode imports a frozen allocation's rows exactly and allocates only
the rest, so the whole corpus is balanced:

    uv run python scripts/allocate_markers.py --config configs/frozen/v2_full.yaml \
        --topics data/topics/full_topics_v2.yaml \
        --seed-allocation data/pilot/marker_allocation_v2.yaml \
        --out data/full/marker_allocation_full_v2.yaml
    ... --check   rebuild and fail unless the stored file is byte-identical

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
    allocate_full_markers,
    allocate_markers,
    allocation_problems,
    balance_table,
    full_allocation_problems,
    load_allocation,
    render_full_allocation,
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
    ap.add_argument("--seed-allocation", default=None,
                    help="full-corpus mode: import this allocation's rows exactly and "
                         "allocate only the rest")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed_allocation:
        return _full(args, cfg)
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


def _full(args, cfg) -> int:
    """Seed-aware mode. The stored file is compared byte for byte, so an edit to
    any row or to the recorded provenance is caught, not only an edit to a row."""
    try:
        bank = load_topic_bank(args.topics)
        seed = load_allocation(args.seed_allocation)
        seeded = allocate_full_markers(bank, cfg, seed,
                                       seed_allocation_path=args.seed_allocation)
    except (TopicBankError, AllocationError, FileNotFoundError) as exc:
        print(f"cannot allocate:\n  {exc}", file=sys.stderr)
        return 1
    problems = allocation_problems(seeded.allocation, cfg) + \
        full_allocation_problems(seeded, bank, cfg)
    if problems:
        print("the full allocation violates its own rules:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    text = render_full_allocation(seeded)
    alloc = seeded.allocation
    out = Path(args.out)
    if args.check:
        if not out.is_file():
            print(f"cannot read the stored allocation: {out} does not exist", file=sys.stderr)
            return 1
        stored = out.read_text(encoding="utf-8")
        if stored != text:
            print(f"the stored full allocation is not byte-identical to a fresh build\n"
                  f"  {out} was edited, or its inputs changed", file=sys.stderr)
            return 1
        try:
            load_allocation(out)
        except AllocationError as exc:                      # pragma: no cover - byte-equal
            print(f"cannot read the stored allocation: {exc}", file=sys.stderr)
            return 1
        print(f"full allocation matches its inputs byte for byte ({len(alloc.groups)} groups: "
              f"{len(seed.groups)} imported, {len(alloc.groups) - len(seed.groups)} new; "
              f"{alloc.content_hash[:12]}).")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {len(alloc.groups)} groups to {out}: {len(seed.groups)} imported from "
          f"{args.seed_allocation} ({seed.content_hash[:12]}), "
          f"{len(alloc.groups) - len(seed.groups)} newly allocated")
    print(f"  seed {alloc.seed}, allocation {alloc.content_hash[:12]}, "
          f"config {cfg.config_version} {cfg.content_hash[:12]}")
    for level, rows in balance_table(alloc.groups).items():
        for key, row in rows.items():
            print(f"  {level:<13} {key:<26} " + " ".join(f"{k}={row[k]}" for k in sorted(row)))
    if args.table:
        print("\n" + _table(alloc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
