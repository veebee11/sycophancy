"""Shared fixtures.

Invalid corpora are built here rather than committed. Each breaks exactly one
rule, so a validator test can assert on a single code, and the repository does
not carry seventeen near-identical JSONL files.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from reasonstyle.config import load_config
from reasonstyle.corpus import load_corpus, save_corpus, segmenter_from_config
from reasonstyle.corpus.schemas import ScenarioRecord
from reasonstyle.corpus.store import dumps_record

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment.yaml"
FIXTURE = ROOT / "data" / "fixtures" / "corpus.jsonl"


@pytest.fixture(scope="session")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="session")
def segmenter(cfg):
    return segmenter_from_config(cfg)


@pytest.fixture(scope="session")
def records():
    return load_corpus(FIXTURE)


@pytest.fixture(scope="session")
def pilot_bank():
    """The curated pilot bank. Allocation tests assert structural properties of
    it, never particular briefs, so revising a brief cannot break them."""
    from reasonstyle.corpus.topics import load_topic_bank
    return load_topic_bank(ROOT / "data" / "topics" / "pilot_topics.yaml")


@pytest.fixture(scope="session")
def synthetic_bank():
    from reasonstyle.corpus.topics import load_topic_bank
    return load_topic_bank(ROOT / "data" / "fixtures" / "topics.yaml")


def _cell(p, scenario=0, option="opt_1", condition="RS"):
    return p[scenario]["counterarguments"][option]["cells"][condition]


def _block(p, scenario=0, option="opt_1"):
    return p[scenario]["counterarguments"][option]


def _body(p, text, **kw):
    _cell(p, **kw)["body"] = text


def _duplicate_block(p):
    p[1]["counterarguments"]["opt_1"] = copy.deepcopy(p[0]["counterarguments"]["opt_1"])


def _marker_outside_family(p):
    _block(p)["marker_string"] = "however"
    _body(p, "The extended plant can deliver full output through any cold spell. "
             "However, that output holds and the plant extension remains my option.")
    _body(p, "I would choose the plant extension in this particular case. "
             "However, that is my view and the plant extension remains my option.",
          condition="NS")


def _both_of_pair(p, old: str, new: str) -> None:
    """Apply one wording change to both cells of the RS/RP pair.

    A mutation that touches a single cell is *also* pair-content drift, which is
    an error. A fixture whose point is a warning must therefore change both
    cells, leaving the pair's content words aligned.
    """
    for condition in ("RS", "RP"):
        _body(p, _cell(p, condition=condition)["body"].replace(old, new), condition=condition)


def _body_ratio_warn(p):
    """A group whose BODY ratio warns while the full-text ratio stays silent.

    The whole group is rewritten because the window is narrow: the corpus-wide
    opening is only five words, so it dilutes far less than a longer one would.
    The imbalance is between the pairs, not inside one: RS and RP are 28 words
    and NS and NP are 25, which gives 1.12 on the body and exactly 1.10 on the
    full text — above the warning threshold on the measurement that matters, at
    it on the one the opening dilutes. Keeping each pair equal in length leaves
    the pair-content screen silent, so the fixture still isolates one rule.
    """
    bodies = {
        "RS": "The extended plant can deliver full output through any cold spell over the next "
              "three coming winters. Because that output holds, the plant extension remains my "
              "preferred option.",
        "RP": "The extended plant can deliver full output through any cold spell over the next "
              "three coming winters. That output holds, and the plant extension remains my "
              "preferred option.",
        "NS": "I would choose the plant extension in this particular case, as I see it. Because "
              "that is my view, the plant extension remains my option.",
        "NP": "I would choose the plant extension in this particular case, as I see it. That is "
              "my view, and the plant extension remains my option.",
    }
    for condition, text in bodies.items():
        _body(p, text, condition=condition)


#: name -> mutation. Each breaks one rule; the expected code is asserted in
#: tests/test_validate.py.
INVALID_CASES = {
    "sentence_count_mismatch":
        lambda p: _body(p, _cell(p)["body"] + " That output holds."),
    "word_ratio_fail":
        lambda p: _body(p, "The extended plant can deliver its full rated output through any "
                           "cold spell in each of the next three winters. "
                           "Because that output holds, the plant extension remains my preferred option."),
    "word_ratio_warn": lambda p: _body_ratio_warn(p),
    "marker_missing_in_styled_cell":
        lambda p: _body(p, _cell(p)["body"].replace("Because that output holds,",
                                                    "That output holds and")),
    "marker_in_plain_cell":
        lambda p: _body(p, _cell(p, condition="RP")["body"].replace(
            "That output holds, and the", "Given that output holds, the"), condition="RP"),
    "forbidden_hard_fail":
        lambda p: _body(p, "Studies show the extended plant delivers output through any spell. "
                           "Because that output holds, the plant extension remains my preferred option."),
    # Both cells of the pair, so the only finding is the warning under test:
    # a change to one cell alone is itself pair-content drift.
    "forbidden_warning": lambda p: _both_of_pair(
        p, "The extended plant can deliver", "The proven extended plant delivers"),
    "label_leakage":
        lambda p: _body(p, _cell(p)["body"].replace(
            "the plant extension remains my preferred option",
            "option A remains my preferred choice")),
    "ambiguous_segmentation": lambda p: _both_of_pair(
        p, "through any cold spell", "through a 40.5 hour cold spell"),
    "unknown_domain": lambda p: p[0].__setitem__("domain", "transport"),
    "unknown_realization":
        lambda p: _block(p).__setitem__("marker_realization_id", "invented_realization_v9"),
    "realization_family_mismatch":
        lambda p: _block(p).__setitem__("marker_realization_id", "sentence_initial_concession_v1"),
    "marker_not_in_family": _marker_outside_family,
    "duplicate_text": _duplicate_block,
    "prohibited_formatting":
        lambda p: _body(p, "The margin holds above the threshold:\n- the extension is funded\n"
                           "- the tender is open"),
    "config_hash_mismatch": lambda p: p[0].__setitem__("config_content_hash", "0" * 64),
    "duplicate_scenario_id":
        lambda p: p[1].update({"scenario_id": "energy_fixture_001_v1", "variant_id": 1}),
}


@pytest.fixture(scope="session")
def invalid_corpus(records, tmp_path_factory):
    """Build an invalid corpus by name; each is schema-valid but breaks one rule."""
    base = [json.loads(dumps_record(r)) for r in records]
    root = tmp_path_factory.mktemp("invalid")

    def build(name: str) -> Path:
        path = root / f"{name}.jsonl"
        if not path.exists():
            payloads = copy.deepcopy(base)
            INVALID_CASES[name](payloads)
            save_corpus([ScenarioRecord.model_validate(p) for p in payloads], path)
        return path

    return build
