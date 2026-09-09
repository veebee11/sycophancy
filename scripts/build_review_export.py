"""Generate the human-review export from a canonical JSONL corpus.

    uv run python scripts/build_review_export.py --config configs/experiment_v3.yaml
    uv run python scripts/build_review_export.py --config configs/experiment_v3.yaml \
        --corpus data/pilot/corpus.jsonl --scope pilot
    uv run python scripts/build_review_export.py --config configs/experiment_v3.yaml --check

--config is REQUIRED. An artefact-generating command must never silently pick up
whichever configuration happens to be newest: the export records the config hash,
and that hash has to correspond to a version someone chose deliberately.

Output is a read-only view. Judgements belong in data/annotations/, never in the
generated Markdown. Nothing here is specific to any one corpus.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import sys
import tempfile
from pathlib import Path

from reasonstyle.config import load_config
from reasonstyle.corpus import load_corpus
from reasonstyle.review import build_review_export
from reasonstyle.segmentation import segmenter_from_config
from reasonstyle.validate import validate_corpus, with_measurements


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/fixtures/tiny_corpus.jsonl")
    ap.add_argument("--config", required=True,
                    help="explicit path to the frozen experiment config; never defaulted, "
                         "because the generated artefacts record its hash")
    ap.add_argument("--scope", default="fixture", choices=["fixture", "pilot", "full"])
    ap.add_argument("--out", default="review")
    ap.add_argument("--annotations", default="data/annotations")
    ap.add_argument("--check", action="store_true",
                    help="regenerate into a temp directory and fail on any drift")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    segmenter = segmenter_from_config(cfg)
    records = load_corpus(args.corpus)
    report = validate_corpus(records, cfg, segmenter, corpus_scope=args.scope)
    measured = with_measurements(records, cfg, segmenter)
    export = build_review_export(measured, report, cfg, segmenter,
                                 source=args.corpus, annotations_dir=args.annotations)

    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            export.write(tmp)
            differing = []
            for name in sorted(export.files):
                a, b = Path(args.out) / name, Path(tmp) / name
                if not a.exists() or not filecmp.cmp(a, b, shallow=False):
                    differing.append(name)
            manifest = Path(args.out) / "MANIFEST.json"
            if not manifest.exists() or json.loads(manifest.read_text()) != export.manifest:
                differing.append("MANIFEST.json")
        if differing:
            print("review export is out of date or was edited by hand:", file=sys.stderr)
            for name in differing:
                print(f"  {name}", file=sys.stderr)
            print("\nregenerate with: uv run python scripts/build_review_export.py",
                  file=sys.stderr)
            return 1
        print(f"review export matches the corpus ({len(export.files)} files).")
        return 0

    export.write(args.out)
    print(f"wrote {len(export.files)} files + MANIFEST.json to {args.out}/")
    print(f"  config {cfg.config_version} {cfg.content_hash[:12]} · "
          f"corpus {report.corpus_content_hash[:12]}")
    print(f"  {report.n_scenarios} scenarios, {report.n_texts} texts, "
          f"{len(report.errors)} errors, {len(report.warnings)} warnings, "
          f"{len(report.human_review)} outstanding human judgements")
    for name, sep in sorted(export.separation.items()):
        if not sep.satisfied:
            print(f"  NOTE {name}: {sep.note()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
