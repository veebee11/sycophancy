#!/usr/bin/env python3
"""Pin compatibility evidence into a new run-ready behavioural config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from reasonstyle.behavioral.pinning import pin_compatibility
from reasonstyle.behavioral.plan import BehavioralPlanError, load_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--instruct-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    destination = Path(args.out)
    if destination.exists():
        print(f"refusing to overwrite {destination}", file=sys.stderr)
        return 1
    try:
        spec = load_spec(args.config)
        reports = {}
        for variant, raw_path in (("base", args.base_report),
                                  ("instruct", args.instruct_report)):
            path = Path(raw_path).resolve()
            reports[variant] = (path, json.loads(path.read_text(encoding="utf-8")))
        pinned = pin_compatibility(spec, reports)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(yaml.safe_dump(pinned, sort_keys=False), encoding="utf-8")
        load_spec(temporary)
        temporary.replace(destination)
    except (OSError, json.JSONDecodeError, BehavioralPlanError, KeyError, TypeError,
            ValueError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
    print(f"wrote run-ready behavioural config to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
