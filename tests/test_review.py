"""Review exporter: deterministic, read-only, blinded where it must be.

Three properties matter. The export must be byte-reproducible from the corpus,
so it can never become a second source of truth. The curator view must show
every experimental cell plainly. The blinded packets must leak nothing that
would guide an annotator toward the intended answer.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from reasonstyle.config import load_config
from reasonstyle.corpus.store import corpus_content_hash, load_corpus
from reasonstyle.corpus.review import (
    transcript_appendix,
    build_review_export,
    highlight_markers,
    item_sampling_units,
    order_with_separation,
    strip_highlighting,
    word_diff,
)
from reasonstyle.corpus.segmentation import segmenter_from_config
from reasonstyle.corpus.validate import validate_corpus, with_measurements

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "corpus.jsonl"


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs" / "experiment.yaml")


@pytest.fixture(scope="module")
def segmenter(cfg):
    return segmenter_from_config(cfg)


@pytest.fixture(scope="module")
def records():
    return load_corpus(FIXTURE)


@pytest.fixture(scope="module")
def export(records, cfg, segmenter):
    report = validate_corpus(records, cfg, segmenter, corpus_scope="fixture")
    measured = with_measurements(records, cfg, segmenter)
    return build_review_export(measured, report, cfg, segmenter, source=FIXTURE,
                               annotations_dir=ROOT / "data" / "annotations")


def _build(records, cfg, segmenter, **kw):
    report = validate_corpus(records, cfg, segmenter, corpus_scope="fixture")
    return build_review_export(with_measurements(records, cfg, segmenter), report,
                               cfg, segmenter, source=FIXTURE, **kw)


# --- determinism ------------------------------------------------------------


def test_the_export_is_byte_reproducible(records, cfg, segmenter):
    a, b = _build(records, cfg, segmenter), _build(records, cfg, segmenter)
    assert a.files == b.files
    assert a.manifest == b.manifest


def test_no_generated_file_carries_a_wall_clock_timestamp(export):
    """A timestamp would make the export differ from one day to the next."""
    import datetime
    today = datetime.date.today().isoformat()
    for name, text in export.files.items():
        assert today not in text, name
    assert "generated_at" not in export.manifest
    assert json.dumps(export.manifest).count(today) == 0


def test_the_manifest_is_deterministic_and_hashes_every_file(export):
    assert set(export.manifest["files"]) == set(export.files)
    assert all(len(h) == 64 for h in export.manifest["files"].values())
    assert export.manifest["corpus_freeze_timestamp"] is None   # not frozen yet


def test_every_file_records_the_corpus_and_config_hash(export, records, cfg):
    corpus_hash = corpus_content_hash(records)
    for name, text in export.files.items():
        if name.endswith(".md"):
            assert corpus_hash[:12] in text, name
            assert cfg.content_hash[:12] in text, name


def test_a_changed_corpus_changes_the_export(records, cfg, segmenter):
    edited = list(records)
    edited[0] = edited[0].model_copy(
        update={"scenario_text": records[0].scenario_text + " One more clause."})
    changed = _build(edited, cfg, segmenter)
    assert changed.manifest["corpus_content_hash"] != corpus_content_hash(records)
    assert changed.files != _build(records, cfg, segmenter).files


def test_derived_measurements_do_not_change_the_corpus_identity(records, cfg, segmenter):
    """The hash printed in the review must be the hash of the JSONL as stored,
    not of the measured copies the exporter works with."""
    export = _build(records, cfg, segmenter)
    assert export.manifest["corpus_content_hash"] == corpus_content_hash(records)
    assert export.manifest["corpus_file_sha256"] is not None


# --- required outputs -------------------------------------------------------


def test_the_export_contains_every_required_file(export):
    names = set(export.files)
    assert "index.md" in names
    assert "all_decisions.md" in names
    assert "decisions/energy_fixture_001.md" in names
    for annotator in ("annotator_1", "annotator_2"):
        for level in ("item", "pair", "scenario"):
            assert f"blind/{annotator}/{level}_packet.md" in names
    for level in ("item", "pair", "scenario"):
        assert f"blind_key/{level}_key.jsonl" in names


def test_the_index_covers_every_decision(export, records):
    index = export.files["index.md"]
    for decision_id in {r.decision_id for r in records}:
        assert f"`{decision_id}`" in index
        assert f"decisions/{decision_id}.md" in index
    assert "16 counterargument texts" in index


def test_the_decision_file_holds_both_variants_and_all_sixteen_texts(export, records):
    text = export.files["decisions/energy_fixture_001.md"]
    assert "Variant 1 — `energy_fixture_001_v1`" in text
    assert "Variant 2 — `energy_fixture_001_v2`" in text
    for record in records:
        for option in ("opt_1", "opt_2"):
            for condition in ("RS", "RP", "NS", "NP"):
                assert record.counterarguments[option].cells[condition].body in text
    body = text.split("## Appendix — canonical transcripts")[0]
    assert body.count("```text") == 16


def test_the_curator_view_labels_every_experimental_cell(export):
    text = export.files["decisions/energy_fixture_001.md"]
    for condition in ("RS", "RP", "NS", "NP"):
        assert f"**{condition}** — " in text
    assert "Marker family" in text and "realization" in text
    assert "Words body/full" in text and "Sentences body/full" in text


def test_the_decision_file_shows_findings_and_outstanding_review(export):
    text = export.files["decisions/energy_fixture_001.md"]
    assert "#### Machine findings" in text
    assert "*No machine errors or warnings.*" in text
    assert "#### Human review still required" in text
    for code in ("H_SUPPORT_DIRECTION", "H_NO_REASON_INTEGRITY",
                 "H_PROPOSITION_PRESERVATION", "H_REALIZATION_YIELDS_REASON_FREE_NS"):
        assert code in text


def test_all_four_comparison_views_are_labelled(export):
    text = export.files["decisions/energy_fixture_001.md"]
    for a, b, name in (("RS", "RP", "style_with_reason"), ("NS", "NP", "style_without_reason"),
                       ("RS", "NS", "content_with_style"), ("RP", "NP", "content_plain")):
        assert f"**{a} − {b}** · `{name}`" in text


def test_the_combined_file_contains_every_decision_file(export):
    combined = export.files["all_decisions.md"]
    assert "## Contents" in combined
    body = export.files["decisions/energy_fixture_001.md"].split("\n", 1)[1]
    assert body in combined


# --- highlighting must not touch the experimental text ----------------------


def test_highlighting_is_reversible_and_never_alters_the_text(export, records):
    for record in records:
        for option in ("opt_1", "opt_2"):
            block = record.counterarguments[option]
            for condition in ("RS", "RP", "NS", "NP"):
                body = block.cells[condition].body
                assert strip_highlighting(highlight_markers(body, block.marker_string)) == body


def test_the_canonical_text_appears_verbatim_in_a_fenced_block(export, records):
    text = export.files["decisions/energy_fixture_001.md"]
    for record in records:
        for option in ("opt_1", "opt_2"):
            for condition in ("RS", "RP", "NS", "NP"):
                body = record.counterarguments[option].cells[condition].body
                assert f"```text\n{body}\n```" in text


def test_highlighting_marks_the_marker_only_in_styled_cells(records):
    block = records[0].counterarguments["opt_1"]
    assert "**Because**" in highlight_markers(block.cells["RS"].body, block.marker_string)
    assert "**" not in highlight_markers(block.cells["RP"].body, block.marker_string)


def test_word_diff_is_deterministic_and_shows_both_sides():
    a, b = "the margin holds and it stands", "because the margin holds it stands"
    assert word_diff(a, b, "X", "Y") == word_diff(a, b, "X", "Y")
    assert "X:" in word_diff(a, b, "X", "Y") and "Y:" in word_diff(a, b, "X", "Y")


# --- blinding ---------------------------------------------------------------


def _packets(export):
    return {n: t for n, t in export.files.items() if n.startswith("blind/")}


def test_blinded_packets_hide_condition_and_marker_metadata(export):
    for name, text in _packets(export).items():
        for leak in ("RS", "RP", "NS", "NP", "marker_family", "marker_realization_id",
                     "premise_indicator", "conclusion_indicator", "concession_contrast",
                     "metadiscursive_inference", "clause_initial_premise_v1"):
            assert leak not in text, f"{name} leaks {leak!r}"


def test_blinded_packets_never_reveal_the_intended_option(export):
    for name, text in _packets(export).items():
        assert "opt_1" not in text and "opt_2" not in text, name
        assert "supported_option" not in text, name


def test_blinded_packets_hide_measurements_and_findings(export):
    for name, text in _packets(export).items():
        for leak in ("Words body/full", "Sentences body/full", "Machine findings",
                     "H_SUPPORT_DIRECTION", "sibling"):
            if leak == "sibling" and "item_packet" in name or "pair_packet" in name:
                continue        # the separation note legitimately says "sibling"
            assert leak not in text, f"{name} leaks {leak!r}"


def test_blinded_option_labels_are_p_and_q_not_a_and_b(export):
    for name, text in _packets(export).items():
        if name.endswith("pair_packet.md"):
            continue
        assert "**Option P**" in text and "**Option Q**" in text
        assert "Option A" not in text and "Option B" not in text


def test_the_item_packet_asks_for_direction_with_an_unclear_option(export):
    text = export.files["blind/annotator_1/item_packet.md"]
    assert "Which option does this reply support?  P / Q / unclear" in text
    assert "Does it supply a premise" in text


def test_the_pair_packet_shows_two_unlabelled_texts(export):
    text = export.files["blind/annotator_1/pair_packet.md"]
    assert "**Text 1**" in text and "**Text 2**" in text
    assert "same substantive claims" in text


def test_the_scenario_packet_shows_no_counterarguments(export, records):
    text = export.files["blind/annotator_1/scenario_packet.md"]
    assert "without any" in text
    for record in records:
        for option in ("opt_1", "opt_2"):
            for condition in ("RS", "RP", "NS", "NP"):
                assert record.counterarguments[option].cells[condition].body not in text


def test_both_annotators_get_the_same_items_in_different_orders(export):
    def ids(name, prefix):
        return [line.split("`")[1] for line in export.files[name].splitlines()
                if line.startswith(f"### ") and prefix in line]
    a = ids("blind/annotator_1/item_packet.md", "Item")
    b = ids("blind/annotator_2/item_packet.md", "Item")
    assert set(a) == set(b) and len(a) == len(b)      # kappa needs identical items
    assert len(a) >= 2


def test_the_unblinding_key_is_separate_and_complete(export):
    keys = [json.loads(line) for line in
            export.files["blind_key/item_key.jsonl"].splitlines() if line.strip()]
    assert keys
    for key in keys:
        assert set(key) >= {"blind_id", "scenario_id", "supported_option",
                            "condition", "option_labels"}
        assert set(key["option_labels"]) == {"P", "Q"}
    assert "never share" in export.files["blind_key/README.md"].lower()
    # the key lives in its own directory, not beside the packets
    assert not any(n.startswith("blind/") and "key" in n for n in export.files)


# --- sibling separation -----------------------------------------------------


def test_separation_is_reported_for_whatever_the_sample_contains(export, cfg):
    """Whatever the draw, the outcome is stated in the packet and the manifest."""
    requested = cfg.raw["annotation"]["reliability_subsample"]["min_sibling_separation"]
    for name, result in export.separation.items():
        assert result.requested == requested == 20
        recorded = export.manifest["sibling_separation"][name]
        assert recorded["satisfied"] is result.satisfied
        assert recorded["sibling_pairs"] == result.n_sibling_pairs
        if result.n_sibling_pairs == 0:
            assert result.satisfied is True and result.achieved is None


def test_the_packet_states_the_separation_outcome(export):
    text = export.files["blind/annotator_1/item_packet.md"]
    assert "sibling separation" in text.lower()


def test_an_infeasible_separation_is_reported_not_silently_relaxed():
    """Four items from two groups cannot be 20 apart. The exporter must say so
    and return its best arrangement, never shrink the constraint."""
    items = [("g1", 1), ("g1", 2), ("g2", 1), ("g2", 2)]
    order, result = order_with_separation(
        items, lambda t: t[0], minimum=20, rng=random.Random(0))
    assert sorted(order) == sorted(items)          # nothing dropped
    assert result.requested == 20
    assert result.satisfied is False
    assert result.n_sibling_pairs == 2
    assert result.achieved is not None and 0 < result.achieved < 20
    assert "INFEASIBLE" in result.note()
    assert "not relaxed" in result.note()


def test_a_feasible_separation_is_honoured():
    items = [("g1", 1), ("g1", 2), *[(f"x{i}", 0) for i in range(8)]]
    order, result = order_with_separation(
        items, lambda t: t[0], minimum=3, rng=random.Random(1))
    assert result.satisfied is True
    assert result.achieved is not None and result.achieved >= 3
    positions = [i for i, t in enumerate(order) if t[0] == "g1"]
    assert abs(positions[1] - positions[0]) >= 3


def test_separation_is_trivially_met_when_no_siblings_exist():
    items = [(f"g{i}", 0) for i in range(5)]
    _, result = order_with_separation(items, lambda t: t[0], minimum=20, rng=random.Random(2))
    assert result.n_sibling_pairs == 0 and result.satisfied is True
    assert "trivially met" in result.note()


# --- corpus independence ----------------------------------------------------


def test_the_exporter_is_not_specific_to_the_current_fixture(records, cfg, segmenter):
    """A second decision must appear without any change to the exporter."""
    second = records[0].model_copy(update={
        "decision_id": "fixture_002", "scenario_id": "fixture_002_v1",
        "domain": "climate"})
    export = _build([*records, second], cfg, segmenter)
    assert "decisions/fixture_002.md" in export.files
    assert "`fixture_002`" in export.files["index.md"]
    assert "climate" in export.files["index.md"]
    assert export.manifest["n_decisions"] == 2


# --- canonical transcript appendix ------------------------------------------


@pytest.fixture(scope="module")
def appendix(export):
    return export.files["decisions/energy_fixture_001.md"].split(
        "## Appendix — canonical transcripts")[1]


def test_the_appendix_exists_and_is_labelled_canonical_not_model_input(appendix):
    assert "canonical transcript, not the exact model input" in appendix
    assert "tokenizer chat template" in appendix
    assert "model-compatibility stage" in appendix
    assert "No model or tokenizer has been loaded." in appendix


def test_the_appendix_shows_all_four_branches_for_each_initial_choice(appendix):
    for initial in ("opt_1", "opt_2"):
        assert f"If the model initially chooses `{initial}`" in appendix
    for condition in ("RS", "RP", "NS", "NP"):
        assert f"**{condition}** — " in appendix


def test_the_appendix_never_appears_in_a_blinded_packet(export):
    for name, text in export.files.items():
        if name.startswith("blind/"):
            assert "canonical transcript" not in text.lower(), name




# --- stratification uses the group's marker family ---------------------------


def test_plain_cells_are_stratified_by_their_groups_marker_family(records):
    """RP and NP carry no marker, so their cell-level marker_family is null.
    The sampling stratum must use the family assigned to the whole group."""
    by_id = {r.scenario_id: r for r in records}
    units = item_sampling_units(records)
    assert len(units) == 16
    for (scenario_id, option, condition), stratum, balance in units:
        block = by_id[scenario_id].counterarguments[option]
        assert stratum[-1] == block.marker_family
        assert stratum[-1] is not None
        # the supported option is balanced marginally, not crossed into the
        # stratum: 3 x 4 x 3 = 36 strata is what a 48-item sample can cover.
        assert option not in stratum and balance == (option,)
    # the distinction is real: the plain cells' own field is null
    plain = [by_id[s].counterarguments[o].cells[c].marker_family
             for (s, o, c), _, _ in units if c in ("RP", "NP")]
    assert plain and all(f is None for f in plain)


def test_the_supported_option_is_balanced_marginally(records, cfg):
    """It is not a crossed stratum, so the sampler must keep the running tally
    even: an unbalanced sample would silently weight one direction."""
    from collections import Counter
    from reasonstyle.corpus.review import _rng, _stratified_sample, item_sampling_units
    from reasonstyle.corpus.store import corpus_content_hash

    sub = cfg.raw["annotation"]["reliability_subsample"]
    units = item_sampling_units(records, sub["stratify_by"], sub["balance_marginally"])
    chosen = _stratified_sample(units, 0.5,
                                _rng(cfg, corpus_content_hash(records), "item-sample"))
    by_option = Counter(option for _, option, _ in chosen)
    assert abs(by_option["opt_1"] - by_option["opt_2"]) <= 1
