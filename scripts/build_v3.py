"""Build, check and review the draft full-v3 stimulus set. Offline; no model call.

    uv run python scripts/build_v3.py build            # write data/full_v3/ (refuses to overwrite)
    uv run python scripts/build_v3.py build --overwrite
    uv run python scripts/build_v3.py check            # rebuild in memory; fail on any byte drift
    uv run python scripts/build_v3.py review           # write review/full_v3_multimarker_openings/

The source is the pinned full-v2 corpus, verified by hash before anything is
read further; v2 is never written. The allocation is solved from the recorded
seed and must reproduce the committed allocation file byte for byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reasonstyle.config import load_config
from reasonstyle.corpus import segmenter_from_config
from reasonstyle.v3.annotation import annotation_units, unit_counts
from reasonstyle.v3.build import (
    ARTIFACTS, allocation_yaml, artifact_texts, build, decisions_of, make_manifest,
    scenario_texts_of, source_records, write_outputs,
)
from reasonstyle.v3.reliability import propose_reliability
from reasonstyle.v3.review import build_review, write_review
from reasonstyle.v3.spec import SPEC_PATH, V3Error, load_spec
from reasonstyle.v3.validate import balance_tables, unit_problems, validate_build


def assemble(spec, *, replace_allocation: bool = False):
    """Build and validate everything in memory.

    An existing allocation file must be exactly what the recorded seed produces;
    only a deliberate ``build --overwrite`` (``replace_allocation``) may replace
    one that differs. Returns ``(built, manifest, findings, texts)``; ``texts``
    holds every artifact exactly as it would be written."""
    records, _, source = source_records(spec)
    allocation_path = Path(spec.outputs["allocation"])
    solved = allocation_yaml(spec, decisions_of(records), source, scenario_texts_of(records))
    if (allocation_path.is_file() and not replace_allocation
            and allocation_path.read_text(encoding="utf-8") != solved):
        raise V3Error(f"{allocation_path} is not what the recorded seed produces; refusing")
    built = build(spec, allocation_text=solved)
    segmenter = segmenter_from_config(load_config(spec.source["config"]))
    units = annotation_units(spec, built)
    summary = unit_counts(units)
    findings = validate_build(built, spec, records, segmenter) + unit_problems(summary)
    reliability = propose_reliability(spec, built, units)
    texts = artifact_texts(built, units, reliability)
    manifest = make_manifest(spec, built, findings, summary, reliability, balance_tables(built),
                             texts)
    texts = artifact_texts(built, units, reliability, manifest)
    return built, manifest, findings, texts


def outputs(spec) -> dict[str, Path]:
    return {name: Path(spec.outputs[name]) for name in ARTIFACTS}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "check", "review"])
    ap.add_argument("--spec", default=str(SPEC_PATH))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--review-out", default=None)
    args = ap.parse_args(argv)
    try:
        spec = load_spec(args.spec)
        built, manifest, findings, texts = assemble(
            spec, replace_allocation=args.command == "build" and args.overwrite)
    except V3Error as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
    errors = [f for f in findings if f.severity == "error"]
    if errors:
        print(f"refusing: {len(errors)} machine error(s); nothing was written:", file=sys.stderr)
        for f in errors[:20]:
            print(f"  - {f.code} {f.where}: {f.message}", file=sys.stderr)
        return 1
    out = outputs(spec)
    if args.command == "build":
        try:
            write_outputs(spec, texts, out, overwrite=args.overwrite)
        except V3Error as exc:
            print(f"refusing: {exc}", file=sys.stderr)
            return 1
        for path in out.values():
            print(f"wrote {path}")
    elif args.command == "check":
        drift = [str(p) for name, p in out.items()
                 if not p.is_file() or p.read_text(encoding="utf-8") != texts[name]]
        if drift:
            print("v3 artifacts are missing or differ from a fresh build:", file=sys.stderr)
            for p in drift:
                print(f"  {p}", file=sys.stderr)
            return 1
        print(f"v3 artifacts match a fresh deterministic build ({len(out)} files).")
    else:
        review = build_review(spec, built, manifest)
        target = Path(args.review_out or spec.outputs["review"])
        write_review(review, target)
        print(f"wrote {len(review)} review file(s) to {target}/")
    c = manifest["counts"]
    print(f"  {c['decisions']} decisions, {c['scenarios']} scenarios, {c['groups']} groups, "
          f"{c['unique_bodies']} bodies, {c['stimuli']} stimuli; status {manifest['status']}")
    print(f"  machine validation: 0 errors, {manifest['machine_validation']['warnings']} warning(s)")
    units = manifest["annotation_units"]
    print(f"  annotation units (outstanding): stimulus {units['stimulus']['units']}, "
          f"pair {units['pair']['units']}, scenario {units['scenario']['units']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
