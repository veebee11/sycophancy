"""Generate a read-only human-review export — of the corpus, or of the topic bank.

    uv run python scripts/export_for_review.py --config configs/experiment.yaml
    uv run python scripts/export_for_review.py --config configs/experiment.yaml \
        --corpus data/pilot/corpus.jsonl --scope pilot
    uv run python scripts/export_for_review.py --config configs/experiment.yaml \
        --topics data/topics/pilot_topics.yaml --registry data/sources/registry.yaml
    ... --source-texts data/sources/raw     required with --topics ('none' for fixtures)
    ... --check    regenerate into a temporary directory and fail on any drift

--config is REQUIRED. An artefact-generating command must never silently pick up
whichever configuration happens to be newest: the export records the config hash,
and that hash has to correspond to a version someone chose deliberately.

The corpus export is written to <out>/, the topic export to <out>/topics/.
Output is a read-only view. Judgements belong in the annotation files or the
topic YAML's curation block, never in the generated Markdown.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import sys
import tempfile
from pathlib import Path

from reasonstyle.config import load_config
from reasonstyle.corpus import load_corpus, segmenter_from_config, validate_corpus, with_measurements
from reasonstyle.corpus.review import build_review_export
from reasonstyle.corpus.source_texts import load_source_texts
from reasonstyle.corpus.sources import load_registry
from reasonstyle.corpus.topic_review import build_topic_export
from reasonstyle.corpus.topics import check_topics, load_topic_bank


def _matches(export, out: Path) -> list[str]:
    """Files under ``out`` that differ from a fresh regeneration."""
    differing = []
    with tempfile.TemporaryDirectory() as tmp:
        export.write(tmp)
        for name in sorted(export.files):
            if not (out / name).exists() or not filecmp.cmp(out / name, Path(tmp) / name,
                                                            shallow=False):
                differing.append(name)
    manifest = out / "MANIFEST.json"
    if not manifest.exists() or json.loads(manifest.read_text()) != export.manifest:
        differing.append("MANIFEST.json")
    return differing


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="explicit path to the experiment config; never defaulted, "
                         "because the generated artefacts record its hash")
    ap.add_argument("--corpus", default="data/fixtures/corpus.jsonl")
    ap.add_argument("--scope", default="fixture", choices=["fixture", "pilot", "full"])
    ap.add_argument("--topics", help="export the topic bank instead of the corpus")
    ap.add_argument("--registry", help="source registry; required with --topics")
    ap.add_argument("--source-texts", metavar="RAW_DIR|none",
                    help="downloaded source files for the overlap screen; required with --topics")
    ap.add_argument("--out", default="review")
    ap.add_argument("--annotations", default="data/annotations")
    ap.add_argument("--check", action="store_true",
                    help="regenerate into a temp directory and fail on any drift")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    if args.topics:
        if not args.registry or not args.source_texts:
            ap.error("--topics requires --registry and --source-texts")
        registry = load_registry(args.registry)
        bank = load_topic_bank(args.topics)
        texts = {} if args.source_texts == "none" else load_source_texts(args.source_texts)
        report = check_topics(bank, cfg, registry, source_texts=texts)
        export = build_topic_export(bank, report, cfg, registry, source=args.topics)
        out = Path(args.out) / "topics"
        summary = (f"  {len(bank.topics)} topics, {len(report.errors)} errors, "
                   f"{len(report.warnings)} warnings, curated per domain "
                   f"{report.curated_per_domain}")
    else:
        segmenter = segmenter_from_config(cfg)
        records = load_corpus(args.corpus)
        report = validate_corpus(records, cfg, segmenter, corpus_scope=args.scope)
        export = build_review_export(with_measurements(records, cfg, segmenter), report, cfg,
                                     segmenter, source=args.corpus,
                                     annotations_dir=args.annotations)
        out = Path(args.out)
        summary = (f"  {report.n_scenarios} scenarios, {report.n_texts} texts, "
                   f"{len(report.errors)} errors, {len(report.warnings)} warnings, "
                   f"{len(report.human_review)} outstanding human judgements")

    if args.check:
        differing = _matches(export, out)
        if differing:
            print("review export is out of date or was edited by hand:", file=sys.stderr)
            for name in differing:
                print(f"  {name}", file=sys.stderr)
            return 1
        print(f"review export matches its source ({len(export.files)} files).")
        return 0

    export.write(out)
    print(f"wrote {len(export.files)} files + MANIFEST.json to {out}/")
    print(f"  config {cfg.config_version} {cfg.content_hash[:12]}")
    print(summary)
    if not args.topics:
        for name, sep in sorted(export.separation.items()):
            if not sep.satisfied:
                print(f"  NOTE {name}: {sep.note()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
