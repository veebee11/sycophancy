"""The full review export understands the byte-preserved pilot seed.

The generic validator must continue to reject arbitrary configuration-hash
mismatches. Only the exact seed described by the hash-matched assembly
manifest is informational in the human-review view.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from reasonstyle.config import load_config
from reasonstyle.corpus import load_corpus, segmenter_from_config
from reasonstyle.corpus.store import save_corpus
from reasonstyle.hashing import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/frozen/v2_full.yaml"
CORPUS = ROOT / "data/full/corpus_full_v2.jsonl"
MANIFEST = ROOT / "data/full/corpus_full_v2.manifest.json"


def _script():
    spec = importlib.util.spec_from_file_location(
        "export_for_review_full_seed", ROOT / "scripts/export_for_review.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(corpus: Path):
    cfg = load_config(CONFIG)
    records = load_corpus(corpus)
    return _script()._review_report(
        records, cfg, segmenter_from_config(cfg), corpus, "full")


def test_real_full_review_has_no_false_seed_hash_errors():
    report = _report(CORPUS)
    assert report.errors == ()
    assert len(report.warnings) == 20
    assert report.codes("info")["I_PINNED_SEED_CONFIG_HASH"] == 24


def test_missing_sidecar_never_suppresses_hash_errors(tmp_path):
    corpus = tmp_path / CORPUS.name
    corpus.write_bytes(CORPUS.read_bytes())
    report = _report(corpus)
    assert report.codes("error")["E_CONFIG_HASH_MISMATCH"] == 24
    assert report.codes("info")["I_PINNED_SEED_CONFIG_HASH"] == 0


def test_stale_sidecar_never_suppresses_hash_errors(tmp_path):
    corpus = tmp_path / CORPUS.name
    corpus.write_bytes(CORPUS.read_bytes())
    manifest = json.loads(MANIFEST.read_text())
    manifest["corpus_sha256"] = "0" * 64
    corpus.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    report = _report(corpus)
    assert report.codes("error")["E_CONFIG_HASH_MISMATCH"] == 24


def test_unexpected_nonseed_hash_mismatch_is_never_suppressed(tmp_path):
    corpus = tmp_path / CORPUS.name
    records = load_corpus(CORPUS)
    new_index = next(i for i, record in enumerate(records)
                     if record.scenario_id == "climate_06_v1")
    records[new_index] = records[new_index].model_copy(
        update={"config_content_hash": "1" * 64})
    save_corpus(records, corpus)

    manifest = json.loads(MANIFEST.read_text())
    manifest["corpus_sha256"] = file_sha256(corpus)
    corpus.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    report = _report(corpus)

    # The unexpected full-run mismatch invalidates the seed exception as a
    # whole, so neither it nor the 24 seed mismatches can disappear silently.
    assert report.codes("error")["E_CONFIG_HASH_MISMATCH"] == 25
    assert report.codes("info")["I_PINNED_SEED_CONFIG_HASH"] == 0
