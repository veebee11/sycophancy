"""Sentence segmentation: pinned, deterministic, and honest about its limits.

D1 makes exact sentence-count equality a hard rejection criterion, so these
tests protect two things: the behaviour the design depends on (a semicolon does
not end a sentence), and the honesty of the failure mode (constructions pysbd
handles badly are *flagged*, not silently trusted).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reasonstyle.config import load_config
from reasonstyle.segmentation import (
    PysbdSegmenter,
    Segmentation,
    SegmentationError,
    Segmenter,
    SegmenterVersionMismatch,
    build_ambiguity_patterns,
    segmenter_from_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIGS / "experiment_v2.yaml")


@pytest.fixture(scope="module")
def seg(cfg):
    return segmenter_from_config(cfg)


# --- identity and pinning ---------------------------------------------------


def test_segmenter_satisfies_the_protocol(seg):
    assert isinstance(seg, Segmenter)


def test_version_is_pinned_and_recorded(seg, cfg):
    assert seg.info.library == "pysbd"
    assert seg.info.version == cfg.raw["segmentation"]["version"]
    assert seg.info.language == "en"
    # every result carries it, so it can be written into artefacts
    assert seg.segment("Costs fall.").segmenter.as_dict() == {
        "library": "pysbd", "version": seg.info.version, "language": "en"}


def test_a_version_mismatch_refuses_to_run(cfg):
    """A segmenter upgrade can change which items are admissible under D1."""
    with pytest.raises(SegmenterVersionMismatch, match="bump the config"):
        PysbdSegmenter(expected_version="0.0.1-not-installed")


def test_a_config_without_a_segmentation_block_is_refused():
    v1 = load_config(CONFIGS / "experiment_v1.yaml")
    with pytest.raises(SegmentationError, match="v2 or later"):
        segmenter_from_config(v1)


# --- the D1 guarantee -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The tender closes in March; therefore the schedule is tight.",
        "Storage procurement is funded; consequently the extension saves nothing.",
        "The margin narrows; however the emissions path improves.",
        "Reserve cover is thin; this implies a longer outage window.",
    ],
)
def test_a_semicolon_does_not_terminate_a_sentence(seg, text):
    """REGRESSION GUARD. D1 permits a semicolon as the device that achieves
    explicit framing inside one sentence. A segmenter that split here would
    make the styled and plain cells impossible to sentence-match."""
    assert seg.segment(text).count == 1


def test_the_config_declares_that_guarantee(cfg):
    assert cfg.raw["segmentation"]["guarantees"]["semicolon_does_not_terminate_sentence"] is True


def test_a_styled_and_plain_pair_can_match_exactly(seg):
    """The construction D1 requires: explicit framing at the same count."""
    plain = "Storage costs are falling. The plant should close."
    explicit_two = "Storage costs are falling. Therefore the plant should close."
    explicit_one = "Because storage costs are falling, the plant should close."
    assert seg.segment(plain).count == seg.segment(explicit_two).count == 2
    assert seg.segment(explicit_one).count == 1


# --- counting behaviour we rely on ------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Storage costs are falling. The plant should close.", 2),
        ("Costs fall. Emissions rise. The board must choose.", 3),
        ("The plant runs at 40.5 GW. Storage is cheaper.", 2),   # decimal
        ("Demand grew 1.5 percent.", 1),
        ("Several options exist, e.g. storage or demand response. Costs differ.", 2),
        ("Two options exist: extend the plant or build storage.", 1),  # colon
        ("Should it close? Yes! The margin is thin.", 3),
        ("", 0),
        ("   ", 0),
    ],
)
def test_sentence_counts(seg, text, expected):
    assert seg.segment(text).count == expected


def test_segmentation_is_deterministic(seg):
    text = "Costs fall. Emissions rise; therefore the board must choose."
    first, second = seg.segment(text), seg.segment(text)
    assert first.sentences == second.sentences
    assert first.ambiguities == second.ambiguities


def test_the_segmenter_never_rewrites_the_text(seg, cfg):
    """`clean: False` — the thing being measured must not be altered."""
    assert cfg.raw["segmentation"]["clean"] is False
    text = "Costs fall. Emissions rise."
    result = seg.segment(text)
    assert " ".join(result.sentences) == text
    assert result.text == text


# --- honesty about the limits -----------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("The plant runs at 40.5 GW.", "decimal_number"),
        ("In the U.S. the margin is thin.", "initialism"),
        ("Costs rose by approx. ten percent.", "abbreviation"),
        ("Options:\n- extend the plant\n- build storage", "bullet_list"),
        ("Options:\n1. extend the plant\n2. build storage", "numbered_list"),
        ("Costs fall.\nEmissions rise.", "line_break"),
    ],
)
def test_unreliable_constructions_are_flagged(seg, text, kind):
    result = seg.segment(text)
    assert result.is_ambiguous
    assert kind in result.ambiguity_kinds()


def test_known_mis_segmentations_are_flagged_rather_than_trusted(seg):
    """pysbd gets both of these wrong. The contract is not that the count is
    right — it is that the text is flagged for human confirmation and, upstream,
    that the generator was told to avoid the construction."""
    # truth is 2 sentences; pysbd reads "U.S." as an abbreviation and returns 1
    us = seg.segment("In the U.S. Storage costs fell sharply.")
    assert us.is_ambiguous and "initialism" in us.ambiguity_kinds()

    # truth is 1 sentence; pysbd splits after "approx." and returns 2
    approx = seg.segment("Costs rose by approx. Ten percent was absorbed.")
    assert approx.is_ambiguous and "abbreviation" in approx.ambiguity_kinds()


def test_clean_prose_raises_no_flags(seg):
    result = seg.segment(
        "Storage procurement is already funded; therefore the extension delays "
        "the emissions trajectory without saving capital."
    )
    assert not result.is_ambiguous
    assert result.ambiguity_kinds() == ()


def test_ambiguity_spans_locate_the_construction(seg):
    text = "The plant runs at 40.5 GW."
    (flag,) = [a for a in seg.segment(text).ambiguities if a.kind == "decimal_number"]
    start, end = flag.span
    assert text[start:end] == flag.text
    assert "0.5" in flag.text


def test_the_abbreviation_detector_is_generated_from_the_list(cfg):
    """The list stays the single source of truth; no second regex to drift."""
    spec = cfg.raw["segmentation"]
    assert "abbreviation" not in spec["ambiguity_patterns"]
    patterns = build_ambiguity_patterns(spec)
    assert "abbreviation" in patterns
    for abbreviation in spec["text_restrictions"]["known_abbreviations"]:
        assert patterns["abbreviation"].search(f"costs rose by {abbreviation} ten percent")


def test_result_is_immutable(seg):
    result = seg.segment("Costs fall.")
    assert isinstance(result, Segmentation)
    with pytest.raises(Exception):
        result.sentences = ()          # type: ignore[misc]
