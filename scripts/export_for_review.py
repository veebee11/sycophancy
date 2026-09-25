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
from dataclasses import replace
from pathlib import Path

from reasonstyle.config import load_config
from reasonstyle.corpus import load_corpus, segmenter_from_config, validate_corpus, with_measurements
from reasonstyle.corpus.findings import ValidationReport
from reasonstyle.corpus.review import build_review_export
from reasonstyle.corpus.source_texts import load_source_texts
from reasonstyle.corpus.sources import load_registry
from reasonstyle.corpus.topic_review import build_topic_export
from reasonstyle.corpus.topics import check_topics, load_topic_bank
from reasonstyle.hashing import file_sha256


_PINNED_SEED_INFO = "I_PINNED_SEED_CONFIG_HASH"


def _verified_seed_scenarios(corpus: Path, records, cfg) -> set[str]:
    """Return the full corpus's verified, byte-preserved seed scenarios.

    A full corpus deliberately carries the frozen pilot records without
    restamping their configuration hash. The ordinary validator quite
    correctly reports those hashes as different from the full configuration;
    the assembly gate then accepts *exactly* the declared seed records after
    checking their provenance. A review export must make the same distinction
    without turning arbitrary config mismatches into exceptions.

    This is fail-closed: no sidecar, stale corpus bytes, incomplete provenance,
    an unexpected source role or any record/hash disagreement returns an empty
    set, leaving every validator error untouched.
    """
    manifest_path = corpus.with_suffix(".manifest.json")
    if not corpus.is_file() or not manifest_path.is_file():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    if (manifest.get("corpus_scope") != "full"
            or manifest.get("corpus_file") != corpus.name
            or manifest.get("corpus_sha256") != file_sha256(corpus)
            or manifest.get("config_version") != cfg.config_version
            or manifest.get("config_content_hash") != cfg.content_hash):
        return set()

    by_id = {record.scenario_id: record for record in records}
    if len(by_id) != len(records):
        return set()
    scenario_sources = manifest.get("scenario_sources")
    if not isinstance(scenario_sources, dict) or set(scenario_sources) != set(by_id):
        return set()
    if set(scenario_sources.values()) != {"seed", "full_run"}:
        return set()

    sources = manifest.get("sources")
    if not isinstance(sources, list):
        return set()
    seed_sources = [source for source in sources
                    if isinstance(source, dict) and source.get("role") == "seed"]
    if len(seed_sources) != 1:
        return set()
    seed = seed_sources[0]
    seed_hash = seed.get("config_content_hash")
    seed_records = seed.get("scenarios")
    if (seed.get("pinned") is not True or seed.get("access") != "read_only"
            or not isinstance(seed_hash, str) or len(seed_hash) != 64
            or not isinstance(seed_records, dict)):
        return set()

    declared = set(seed_records)
    assigned = {scenario_id for scenario_id, source in scenario_sources.items()
                if source == "seed"}
    if not declared or declared != assigned:
        return set()
    if any(by_id[scenario_id].config_content_hash != seed_hash for scenario_id in declared):
        return set()
    if any(record.config_content_hash != cfg.content_hash
           for scenario_id, record in by_id.items() if scenario_id not in declared):
        return set()
    return declared


def _review_report(records, cfg, segmenter, corpus: Path, scope: str) -> ValidationReport:
    """Validate for review, reclassifying only verified seed hash findings."""
    report = validate_corpus(records, cfg, segmenter, corpus_scope=scope)
    if scope != "full":
        return report
    seed_scenarios = _verified_seed_scenarios(corpus, records, cfg)
    if not seed_scenarios:
        return report

    findings = []
    for finding in report.findings:
        if finding.code == "E_CONFIG_HASH_MISMATCH" and finding.scenario_id in seed_scenarios:
            findings.append(replace(
                finding,
                code=_PINNED_SEED_INFO,
                severity="info",
                message=("record retains the verified pinned pilot configuration hash "
                         "by design; it was imported byte for byte"),
            ))
        else:
            findings.append(finding)
    return replace(report, findings=tuple(findings))


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
    ap.add_argument("--reliability-sample", metavar="PIN_JSON",
                    help="reuse a pinned reliability sample (blind ids, labels, orders) instead of "
                         "drawing one from this corpus; refuses unless it reproduces exactly")
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
        corpus = Path(args.corpus)
        records = load_corpus(corpus)
        report = _review_report(records, cfg, segmenter, corpus, args.scope)
        pin = (json.loads(Path(args.reliability_sample).read_text(encoding="utf-8"))
               if args.reliability_sample else None)
        try:
            export = build_review_export(with_measurements(records, cfg, segmenter), report, cfg,
                                         segmenter, source=args.corpus,
                                         annotations_dir=args.annotations, sampling_pin=pin)
        except ValueError as exc:
            print(f"refusing; nothing was written: {exc}", file=sys.stderr)
            return 1
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
        seed_infos = report.codes("info").get(_PINNED_SEED_INFO, 0)
        if seed_infos:
            print(f"  NOTE {seed_infos} verified pinned-seed configuration hashes are "
                  "expected provenance, not machine errors")
        for name, sep in sorted(export.separation.items()):
            if not sep.satisfied:
                print(f"  NOTE {name}: {sep.note()}")
        for name, result in sorted(export.balance.items()):
            print(f"  {'' if result.satisfied else 'NOTE '}{name} sample: {result.note()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
