#!/usr/bin/env python3
"""Plan and check the v3 Llama behavioural experiment without loading a model.

Examples:
  uv run python scripts/behavioral_v3.py check \
      --config configs/behavioral_v3_llama31_8b.draft.yaml
  uv run python scripts/behavioral_v3.py emit-plan \
      --config configs/behavioral_v3_llama31_8b.draft.yaml --out /tmp/v3-plan

Neither command imports transformers, contacts Hugging Face, downloads weights,
or performs a behavioural evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reasonstyle.behavioral.plan import BehavioralPlanError, build_plan, load_spec, plan_texts


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "emit-plan"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        spec = load_spec(args.config)
        plan = build_plan(spec)
    except (OSError, KeyError, TypeError, BehavioralPlanError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(plan.manifest, indent=2, sort_keys=True))
    if args.command == "check":
        if args.out:
            parser.error("--out is only valid with emit-plan")
        return 0
    if not args.out:
        parser.error("emit-plan requires --out")
    out = Path(args.out)
    files = {"initials": out / "initial_prompts.jsonl",
             "candidates": out / "branch_candidates.jsonl",
             "manifest": out / "MANIFEST.json"}
    existing = [str(p) for p in files.values() if p.exists()]
    if existing and not args.overwrite:
        print("refusing to overwrite: " + ", ".join(existing), file=sys.stderr)
        return 1
    texts = plan_texts(plan)
    for name, path in files.items():
        _write_atomic(path, texts[name])
    print(f"wrote deterministic plan to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
