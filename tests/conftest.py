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
    _body(p, "The plant extension keeps the reserve margin above the threshold. "
             "However, the margin holds and the plant extension remains my preferred option.")
    _body(p, "I would choose the plant extension in this case. "
             "However, that is my view and the plant extension remains my preferred option.",
          condition="NS")


#: name -> mutation. Each breaks one rule; the expected code is asserted in
#: tests/test_validate.py.
INVALID_CASES = {
    "sentence_count_mismatch":
        lambda p: _body(p, _cell(p)["body"] + " The margin holds."),
    "word_ratio_fail":
        lambda p: _body(p, "The plant extension keeps the reserve margin above the winter "
                           "threshold in each of the next three delivery years. "
                           "Because the margin holds, the plant extension remains my preferred option."),
    "word_ratio_warn":
        lambda p: _body(p, "The plant extension keeps the reserve margin above the threshold "
                           "in each year. Because the margin holds, the plant extension "
                           "remains my preferred option."),
    "marker_missing_in_styled_cell":
        lambda p: _body(p, _cell(p)["body"].replace("Because the margin holds,",
                                                    "The margin holds and")),
    "marker_in_plain_cell":
        lambda p: _body(p, _cell(p, condition="RP")["body"].replace(
            "The margin holds, and the", "Given that the margin holds, the"), condition="RP"),
    "forbidden_hard_fail":
        lambda p: _body(p, "Studies show the margin holds above the winter threshold. "
                           "Because the margin holds, the plant extension remains my preferred option."),
    "forbidden_warning":
        lambda p: _body(p, _cell(p)["body"].replace("The plant extension keeps",
                                                    "The proven plant extension keeps")),
    "label_leakage":
        lambda p: _body(p, _cell(p)["body"].replace(
            "the plant extension remains my preferred option",
            "option A remains my preferred choice")),
    "ambiguous_segmentation":
        lambda p: _body(p, _cell(p)["body"].replace("above the threshold",
                                                    "above the 40.5 GW threshold")),
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
        lambda p: p[1].update({"scenario_id": "fixture_001_v1", "variant_id": 1}),
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
