"""Check the reference-source registry. Offline: nothing is downloaded.

    uv run python scripts/verify_sources.py --registry data/sources/registry.yaml
    uv run python scripts/verify_sources.py --registry data/sources/registry.yaml \
        --cite polianna genai4pa_jrc_2025

The script fails if the registry is malformed or an outcome is not supported by
its evidence (for example, a citable entry with no licence, or an excluded entry
with no reason).

With --cite it also fails unless every listed key is citable. That is the gate
that matters: every source a topic actually cites must be citable. A candidate
that nothing cites may stay unverified without blocking anything.
"""

from __future__ import annotations

import argparse
import sys

from reasonstyle.corpus.sources import SourceRegistryError, load_registry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--cite", nargs="+", metavar="KEY", default=[],
                    help="fail unless every listed registry key is citable")
    args = ap.parse_args(argv)

    try:
        registry = load_registry(args.registry)
    except SourceRegistryError as exc:
        print(f"registry is invalid:\n{exc}", file=sys.stderr)
        return 1

    width = max(len(k) for k in registry.datasets)
    for key, dataset in sorted(registry.datasets.items()):
        line = f"  {key:<{width}}  {dataset.outcome}"
        if dataset.outcome == "excluded":
            line += f"  — {dataset.exclusion_reason}"
        print(line)
    print(f"\n{len(registry.datasets)} candidates: {len(registry.citable())} citable, "
          f"{len(registry.excluded())} excluded, {len(registry.unverified())} unverified")

    problems = registry.citation_problems(args.cite)
    if problems:
        print("\nmay not be cited:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    if args.cite:
        print(f"all {len(args.cite)} cited source(s) are citable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
