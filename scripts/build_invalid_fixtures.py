"""Build data/fixtures/invalid/*.jsonl — one corpus per failure mode.

Each file is *schema-valid* (it loads) but breaks exactly one corpus rule, so a
validator test can assert on a single expected code. Failures the schema itself
rejects (missing cells, mis-filed blocks, a plain cell claiming markers) are
covered in tests/test_schemas.py instead; they can never reach a JSONL file.

All texts are synthetic. Nothing here is research material.
"""
import copy
import json
from pathlib import Path

from reasonstyle.corpus import dumps_record, load_corpus
from reasonstyle.schemas import ScenarioRecord

SRC = Path("data/fixtures/tiny_corpus.jsonl")
OUT = Path("data/fixtures/invalid")
BASE = [json.loads(dumps_record(r)) for r in load_corpus(SRC)]


def write(name: str, mutate) -> None:
    payloads = copy.deepcopy(BASE)
    mutate(payloads)
    records = [ScenarioRecord.model_validate(p) for p in payloads]   # must still load
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.jsonl").write_text("".join(f"{dumps_record(r)}\n" for r in records))


def cell(p, scenario=0, option="opt_1", condition="RS"):
    return p[scenario]["counterarguments"][option]["cells"][condition]


def block(p, scenario=0, option="opt_1"):
    return p[scenario]["counterarguments"][option]


def set_body(p, text, scenario=0, option="opt_1", condition="RS"):
    cell(p, scenario, option, condition)["body"] = text


def _duplicate_block(p):
    """Copy a whole valid block, so only E_DUPLICATE_TEXT fires."""
    p[1]["counterarguments"]["opt_1"] = copy.deepcopy(p[0]["counterarguments"]["opt_1"])


def _marker_outside_family(p):
    """A marker that is real but belongs to a different family. Both styled
    cells are rewritten so only E_MARKER_NOT_IN_FAMILY fires."""
    block(p)["marker_string"] = "however"
    set_body(p, "The plant extension keeps the reserve margin above the threshold. "
                "However, the margin holds and the plant extension remains my preferred option.")
    set_body(p, "I would choose the plant extension in this case. "
                "However, that is my view and the plant extension remains my preferred option.",
             condition="NS")


CASES = {
    # D1: the four cells no longer share a sentence count
    "sentence_count_mismatch":
        lambda p: set_body(p, cell(p)["body"] + " The margin holds."),
    # D2: body ratio past the hard cap (1.43)
    "word_ratio_fail":
        lambda p: set_body(
            p, "The plant extension keeps the reserve margin above the winter threshold "
               "in each of the next three delivery years. "
               "Because the margin holds, the plant extension remains my preferred option."),
    # D2: body ratio 1.143 -> warning. The FULL-text ratio is only 1.094, so
    # this case exists to show why the ratio is measured on both.
    "word_ratio_warn":
        lambda p: set_body(
            p, "The plant extension keeps the reserve margin above the threshold in each year. "
               "Because the margin holds, the plant extension remains my preferred option."),
    # a styled cell that lost its marker
    "marker_missing_in_styled_cell":
        lambda p: set_body(p, cell(p)["body"].replace(
            "Because the margin holds,", "The margin holds and")),
    # marker absence is what defines RP and NP
    "marker_in_plain_cell":
        lambda p: set_body(p, cell(p, condition="RP")["body"].replace(
            "The margin holds, and the", "Given that the margin holds, the"), condition="RP"),
    # hard-fail forbidden phrase
    "forbidden_hard_fail":
        lambda p: set_body(
            p, "Studies show the margin holds above the winter threshold. "
               "Because the margin holds, the plant extension remains my preferred option."),
    # warning-severity forbidden phrase: scenario-groundable, routes to review
    "forbidden_warning":
        lambda p: set_body(p, cell(p)["body"].replace(
            "The plant extension keeps", "The proven plant extension keeps")),
    # D5: display-label leakage
    "label_leakage":
        lambda p: set_body(p, cell(p)["body"].replace(
            "the plant extension remains my preferred option",
            "option A remains my preferred choice")),
    # segmenter is unreliable on decimals -> warning, not error
    "ambiguous_segmentation":
        lambda p: set_body(p, cell(p)["body"].replace(
            "above the threshold", "above the 40.5 GW threshold")),
    "unknown_domain":
        lambda p: p[0].__setitem__("domain", "transport"),
    "unknown_realization":
        lambda p: block(p).__setitem__("marker_realization_id", "invented_realization_v9"),
    "realization_family_mismatch":
        lambda p: block(p).__setitem__("marker_realization_id", "sentence_initial_concession_v1"),
    "marker_not_in_family": _marker_outside_family,
    "duplicate_text": _duplicate_block,
    "prohibited_formatting":
        lambda p: set_body(
            p, "The margin holds above the threshold:\n- the extension is funded\n"
               "- the tender is open"),
    "config_hash_mismatch":
        lambda p: p[0].__setitem__("config_content_hash", "0" * 64),
    "duplicate_scenario_id":
        lambda p: p[1].update({"scenario_id": "fixture_001_v1", "variant_id": 1}),
}

for name, mutate in CASES.items():
    write(name, mutate)
print(f"wrote {len(CASES)} invalid fixtures to {OUT}")
