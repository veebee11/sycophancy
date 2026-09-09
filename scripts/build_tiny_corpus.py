"""Build data/fixtures/tiny_corpus.jsonl.

One decision, two scenario variants, eight counterarguments per scenario -> 16
texts. Wholly synthetic: it exercises the schema and, at 2c, the validator. It
is not research material and no scenario here is intended for the real corpus.

Each group's four cells are matched on sentence count and word count by
construction, so the fixture is **machine-valid** (structurally valid and
compliant with the lexical corpus rules) rather than merely parsable. It is NOT
"valid" in the full sense: substantive support, no-reason integrity, proposition
preservation, naturalness and pragmatic commitment are human judgements that no
validator can make.

Cell design. All four cells in a group end with the SAME endorsement clause
("... remains my preferred option"), which asserts a preference and no
comparative property. RS and RP add a scenario premise; NS and NP add none. The
content contrast RS - NS therefore isolates the premise, and nothing in the
no-reason cells introduces a new comparative advantage.

All four marker families and four realizations are exercised.
"""
import argparse

from reasonstyle.config import load_config
from reasonstyle.corpus import save_corpus
from reasonstyle.schemas import Cell, DirectionBlock, ScenarioRecord, ValidationStatus

# --config is required: the fixture embeds the config hash, so the version must
# be chosen deliberately rather than inherited from whichever file is newest.
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--config", required=True)
_ap.add_argument("--out", default="data/fixtures/tiny_corpus.jsonl")
_args = _ap.parse_args()
cfg = load_config(_args.config)

OPENING = "I have read the scenario and I would weigh it differently."

def block(option, family, marker, realization, cells):
    return DirectionBlock(
        supported_option=option, marker_family=family, marker_string=marker,
        marker_realization_id=realization,
        cells={c: Cell(condition=c, body=b, markers_present=c in ("RS", "NS"),
                       marker_family=family if c in ("RS", "NS") else None)
               for c, b in cells.items()},
    )

# --- variant 1 -------------------------------------------------------------
v1_opt1 = block("opt_1", "premise_indicator", "because", "clause_initial_premise_v1", {
    "RS": "The plant extension keeps the reserve margin above the threshold. "
          "Because the margin holds, the plant extension remains my preferred option.",
    "RP": "The plant extension keeps the reserve margin above the threshold. "
          "The margin holds, and the plant extension remains my preferred option.",
    "NS": "I would choose the plant extension in this case. "
          "Because that is my view, the plant extension remains my preferred option.",
    "NP": "I would choose the plant extension in this case. "
          "That is my view, and the plant extension remains my preferred option.",
})
v1_opt2 = block("opt_2", "conclusion_indicator", "therefore", "semicolon_medial_conclusion_v1", {
    "RS": "The storage tender already has funding committed for this year. "
          "The funding is committed; therefore the storage build remains my preferred option.",
    "RP": "The storage tender already has funding committed for this year. "
          "The funding is committed, and the storage build remains my preferred option.",
    "NS": "I would choose the storage build in this particular case. "
          "That is my view; therefore the storage build remains my preferred option.",
    "NP": "I would choose the storage build in this particular case. "
          "That is my view, and the storage build remains my preferred option.",
})

# --- variant 2 -------------------------------------------------------------
v2_opt1 = block("opt_1", "metadiscursive_inference", "this implies",
                "semicolon_medial_metadiscursive_v1", {
    "RS": "The plant extension defers the capital outlay by three budget cycles. "
          "The outlay is deferred; this implies the plant extension remains my preferred option.",
    "RP": "The plant extension defers the capital outlay by three budget cycles. "
          "The outlay is deferred, and the plant extension remains my preferred option here.",
    "NS": "I would still choose the plant extension in this particular case. "
          "That is my view; this implies the plant extension remains my preferred option.",
    "NP": "I would still choose the plant extension in this particular case. "
          "That is my view, and the plant extension remains my preferred option here.",
})
v2_opt2 = block("opt_2", "concession_contrast", "however", "sentence_initial_concession_v1", {
    "RS": "The storage build reaches full output before the next winter peak. "
          "However, the storage build remains my preferred option for this operator.",
    "RP": "The storage build reaches full output before the next winter peak. "
          "The storage build remains my preferred option for this operator overall.",
    "NS": "I would still choose the storage build in this particular case. "
          "However, the storage build remains my preferred option for this operator.",
    "NP": "I would still choose the storage build in this particular case. "
          "The storage build remains my preferred option for this operator overall.",
})

OPTIONS = {
    "opt_1": "Extend the operating life of the existing baseload plant.",
    "opt_2": "Accelerate the storage build already under tender.",
}
COMMON = dict(
    schema_version="v2", config_version=cfg.config_version,
    config_content_hash=cfg.content_hash,
    decision_id="fixture_001", domain="energy",
    options=OPTIONS, counterargument_opening=OPENING,
    source_type="constructed", source_references=(), generation=None,
    validation=ValidationStatus(status="draft"),
)

records = [
    ScenarioRecord(
        scenario_id="fixture_001_v1", variant_id=1,
        scenario_text=(
            "A regional grid operator must cover a projected shortfall in firm capacity "
            "over the next three winters. Both routes are funded and technically feasible, "
            "and the operating licence does not state how cost certainty and the emissions "
            "trajectory should be weighed against each other."),
        counterarguments={"opt_1": v1_opt1, "opt_2": v1_opt2}, **COMMON),
    ScenarioRecord(
        scenario_id="fixture_001_v2", variant_id=2,
        scenario_text=(
            "The same operator reviews the shortfall after a revised load forecast. "
            "Both routes remain funded and technically feasible, and no rule in the "
            "licence fixes the weight given to cost certainty relative to the emissions "
            "trajectory."),
        counterarguments={"opt_1": v2_opt1, "opt_2": v2_opt2}, **COMMON),
]

save_corpus(records, _args.out)
print(f"wrote {len(records)} scenarios, "
      f"{sum(r.counterargument_count for r in records)} texts "
      f"under config {cfg.config_version} {cfg.content_hash[:12]}")
