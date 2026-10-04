"""The draft full-v3 stimulus set: allocation, construction, validation, review.

The source is the committed full-v2 corpus, read only. Builds happen in memory
or under ``tmp_path``; no test writes to ``data/`` or touches any v2 artifact.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from reasonstyle.config import load_config
from reasonstyle.corpus import segmenter_from_config
from reasonstyle.hashing import file_sha256, sha256_of
from reasonstyle.v3.allocation import POSITIONS, allocation_problems, scenario_collisions
from reasonstyle.v3.build import (
    ARTIFACTS, Build, build, check_output_paths, jsonl, load_allocation_rows, scenario_texts_of,
    source_records,
)
from reasonstyle.v3.review import build_review
from reasonstyle.v3.spec import V3Error, load_spec
from reasonstyle.v3.validate import EXPECTED, EXPECTED_UNITS, balance_tables, validate_build

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/full_v3_multimarker_openings.yaml"
V2_PROTECTED = [ROOT / p for p in (
    "data/full/corpus_full_v2.jsonl", "data/full/corpus_full_v2.manifest.json",
    "configs/frozen/v2_full.yaml", "data/full/marker_allocation_full_v2.yaml",
    "data/full/manual_corrections_full_v2.yaml", "data/full/scenario_corrections_full_v2.yaml",
    "data/full/scenario_approvals_full_v2.yaml", "data/full/reliability_sample_full_v2.json")]


def _script():
    spec = importlib.util.spec_from_file_location("build_v3_script", ROOT / "scripts/build_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def world():
    before = {p: file_sha256(p) for p in V2_PROTECTED}
    spec = load_spec(SPEC)
    built, manifest, findings, texts = _script().assemble(spec)
    records, _, _ = source_records(spec)
    return {"spec": spec, "built": built, "manifest": manifest, "findings": findings,
            "texts": texts, "records": records, "texts_by_scenario": scenario_texts_of(records),
            "v2_before": before}


def _committed(spec, name: str) -> str:
    return (ROOT / spec.outputs[name]).read_text(encoding="utf-8")


# --- allocation ---------------------------------------------------------------------


def test_allocation_regenerates_byte_identically_from_the_seed(world):
    assert world["built"].allocation_text == _committed(world["spec"], "allocation")


def test_the_committed_allocation_meets_every_balance_constraint(world):
    spec = world["spec"]
    rows = yaml.safe_load(_committed(spec, "allocation"))["groups"]
    assert allocation_problems(rows, spec, world["texts_by_scenario"]) == []
    balance = balance_tables(world["built"])
    assert balance["cross_family_pair_counts"] == {6: 12, 7: 24}
    assert balance["within_family_pair_cooccurrence"] == {24: 30}
    domains = sorted({r["domain"] for r in rows})
    for family in ("conclusion_result", "inference_basis"):
        for k, marker in enumerate(spec.family_markers(family)):
            mine = [r for r in rows if r["markers"][family] == marker]
            assert len(mine) == 40 and len({r["decision_id"] for r in mine}) == 40
            assert Counter((r["variant_id"], r["supported_option"]) for r in mine) == \
                Counter({p: 10 for p in POSITIONS})
            assert Counter(r["domain"] for r in mine) == \
                Counter({d: 14 if d == domains[k % 3] else 13 for d in domains})
    for scenario in {r["scenario_id"] for r in rows}:
        ms = [m for r in rows if r["scenario_id"] == scenario for m in r["markers"].values()]
        assert len(set(ms)) == 4


@pytest.mark.parametrize("tamper", ["swap_within_decision", "duplicate_in_decision",
                                    "foreign_marker", "drop_group"])
def test_allocation_checks_catch_violations(world, tamper):
    spec = world["spec"]
    rows = copy.deepcopy(yaml.safe_load(_committed(spec, "allocation"))["groups"])
    if tamper == "swap_within_decision":
        a, b = rows[0], rows[1]
        a["markers"]["conclusion_result"], b["markers"]["conclusion_result"] = \
            b["markers"]["conclusion_result"], a["markers"]["conclusion_result"]
    elif tamper == "duplicate_in_decision":
        rows[1]["markers"]["conclusion_result"] = rows[0]["markers"]["conclusion_result"]
    elif tamper == "foreign_marker":
        rows[0]["markers"]["conclusion_result"] = rows[0]["markers"]["inference_basis"]
    else:
        rows = rows[1:]
    assert allocation_problems(rows, spec, world["texts_by_scenario"]) != []


def test_a_bad_allocation_is_refused(world):
    spec = world["spec"]
    data = yaml.safe_load(_committed(spec, "allocation"))
    data["groups"][0]["markers"]["inference_basis"] = data["groups"][1]["markers"]["inference_basis"]
    with pytest.raises(V3Error, match="allocation does not verify"):
        load_allocation_rows(yaml.safe_dump(data), spec, world["built"].source,
                             world["texts_by_scenario"])


# --- counts, source preservation ---------------------------------------------------


def test_exact_counts(world):
    counts = world["manifest"]["counts"]
    for key, want in EXPECTED.items():
        assert counts[key] == want, key
    assert set(counts["by_marker"].values()) == {240} and len(counts["by_marker"]) == 12


def test_v2_source_is_bound_and_unchanged(world):
    assert world["manifest"]["source_corpus_sha256"] == \
        "9b1a56114045801b30c765ca63ba68183fb7a8b6fe0a4bedce283fab49c3861f"
    assert {p: file_sha256(p) for p in V2_PROTECTED} == world["v2_before"]
    by_scenario = {r["scenario_id"]: r for r in world["records"]}
    for b in world["built"].bodies:
        cells = by_scenario[b["scenario_id"]]["counterarguments"][b["supported_option"]]["cells"]
        if b["condition"] in ("RP", "NP"):
            assert b["body"] == cells[b["condition"]]["body"]
        assert b["endorsement"] == cells["NP"]["body"]


def test_a_bad_source_hash_is_refused(tmp_path):
    raw = yaml.safe_load(SPEC.read_text())
    raw["derived_from"]["source_corpus_sha256"] = "0" * 64
    bad = tmp_path / "spec.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(V3Error, match="v2 source does not verify"):
        build(load_spec(bad), root=ROOT)


# --- construction --------------------------------------------------------------------


def test_marker_realization_grammar(world):
    spec = world["spec"]
    for b in world["built"].bodies:
        if b["condition"] not in ("RS", "NS"):
            continue
        m = spec.markers[b["marker_id"]]
        capital = m.string[0].upper() + m.string[1:]
        comma = m.realization_id != "v3_initial_clause_no_comma"
        assert m.prefix == capital + (", " if comma else " ")
        if b["condition"] == "NS":
            assert b["body"] == m.prefix + b["endorsement"]
        else:
            assert b["body"] == f"{b['premise']} {m.prefix}{b['endorsement']}"
    assert {m.marker_id for m in spec.markers.values()
            if m.realization_id == "v3_initial_clause_no_comma"} == {
        "m06_that_is_why", "m07_this_implies_that", "m08_it_follows_that"}


def test_styled_bodies_minus_the_marker_equal_their_shared_plain_control(world):
    bodies = {b["body_id"]: b for b in world["built"].bodies}
    styled = [b for b in bodies.values() if b["condition"] in ("RS", "NS")]
    assert len(styled) == 960
    for b in styled:
        control = bodies[b["paired_control_body_id"]]
        assert control["condition"] == {"RS": "RP", "NS": "NP"}[b["condition"]]
        assert b["body"].replace(b["marker_prefix"], "", 1) == control["body"]


def test_pair_mappings_point_to_one_shared_control_per_group(world):
    bodies = world["built"].bodies
    for control in (b for b in bodies if b["condition"] in ("RP", "NP")):
        variants = [b for b in bodies if b["paired_control_body_id"] == control["body_id"]]
        assert len(variants) == 2 and sorted(v["body_id"] for v in variants) == \
            control["styled_variant_body_ids"]
        assert {v["marker_family"] for v in variants} == {"conclusion_result", "inference_basis"}


def test_openings_are_fully_crossed_with_identical_bodies(world):
    spec = world["spec"]
    by_body: dict[str, list] = {}
    for s in world["built"].stimuli:
        by_body.setdefault(s["body_id"], []).append(s)
    assert len(by_body) == 1440
    for body_id, stimuli in by_body.items():
        assert sorted(s["opening_id"] for s in stimuli) == sorted(spec.openings)
        tails = {s["rendered"][len(s["opening"]) + 1:] for s in stimuli}
        assert tails == {stimuli[0]["body"]}
        for s in stimuli:
            assert s["rendered"].startswith(s["opening"] + " ")
            assert sum(s["rendered"].count(o) for o in spec.openings.values()) == 1
            if s["paired_control_stimulus_id"]:
                assert s["paired_control_stimulus_id"].endswith("." + s["opening_id"])


# --- validation ----------------------------------------------------------------------


def test_validation_has_no_errors_or_warnings(world):
    assert world["findings"] == []
    assert world["manifest"]["machine_validation"] == {"errors": 0, "warnings": 0,
                                                       "warning_detail": []}
    assert world["manifest"]["formal_annotation"] == "not started"


@pytest.mark.parametrize("tamper", ["premise_word", "drop_marker", "plain_marker", "opening"])
def test_validation_catches_tampering(world, tamper):
    built = world["built"]
    bodies = copy.deepcopy(built.bodies)
    stimuli = copy.deepcopy(built.stimuli)
    if tamper == "premise_word":
        b = next(b for b in bodies if b["condition"] == "RS")
        b["body"] = b["body"].replace(" ", "  ", 1)
    elif tamper == "drop_marker":
        b = next(b for b in bodies if b["condition"] == "NS")
        b["body"] = b["endorsement"]
    elif tamper == "plain_marker":
        b = next(b for b in bodies if b["condition"] == "NP")
        b["body"] = "Thus, " + b["body"]
    else:
        stimuli[0]["rendered"] = stimuli[0]["rendered"] + " " + stimuli[0]["opening"]
    segmenter = segmenter_from_config(load_config(world["spec"].source["config"]))
    findings = validate_build(Build(built.allocation_text, bodies, stimuli, built.source),
                              world["spec"], world["records"], segmenter)
    assert any(f.severity == "error" for f in findings)


# --- determinism, manifest, refusal ----------------------------------------------------


def test_the_build_is_deterministic_and_matches_the_committed_artifacts(world):
    spec, built = world["spec"], world["built"]
    for name in ARTIFACTS:
        assert world["texts"][name] == _committed(spec, name), name
    again = build(spec, allocation_text=built.allocation_text)
    assert jsonl(again.bodies) == jsonl(built.bodies) and jsonl(again.stimuli) == jsonl(built.stimuli)


def test_manifest_hashes_match_every_artifact(world):
    spec, manifest = world["spec"], world["manifest"]
    assert set(manifest["artifacts_sha256"]) == set(ARTIFACTS) - {"manifest"}
    for name, digest in manifest["artifacts_sha256"].items():
        assert digest == file_sha256(ROOT / spec.outputs[name]), name
    assert manifest["spec_sha256"] == file_sha256(SPEC)
    assert manifest["status"] == "draft"


def test_the_check_command_passes_on_the_committed_artifacts(capsys):
    assert _script().main(["check"]) == 0
    assert "match a fresh deterministic build" in capsys.readouterr().out


@pytest.mark.parametrize("path", ["data/full/corpus_full_v2.jsonl", "data/pilot/x.jsonl",
                                  "configs/frozen/x.yaml", "./data/full/new.jsonl"])
def test_v3_refuses_to_write_any_v2_location(world, path):
    with pytest.raises(V3Error, match="protected v2 location"):
        check_output_paths(world["spec"], {"bodies": Path(path)}, overwrite=True)


def test_v3_refuses_to_overwrite_without_the_flag(world, tmp_path):
    existing = tmp_path / "bodies.jsonl"
    existing.write_text("x")
    with pytest.raises(V3Error, match="already exists"):
        check_output_paths(world["spec"], {"bodies": existing}, overwrite=False)
    check_output_paths(world["spec"], {"bodies": existing}, overwrite=True)


def test_an_allocation_file_that_the_seed_does_not_produce_is_refused(world, tmp_path):
    raw = yaml.safe_load(SPEC.read_text())
    tampered = tmp_path / "allocation.yaml"
    tampered.write_text(_committed(world["spec"], "allocation").replace("m01_therefore",
                                                                        "m03_thus", 1))
    raw["outputs"]["allocation"] = str(tampered)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(V3Error, match="not what the recorded seed produces"):
        _script().assemble(load_spec(spec_path))


# --- review export ---------------------------------------------------------------------


def test_the_review_export_is_complete_and_deterministic(world):
    spec, built, manifest = world["spec"], world["built"], world["manifest"]
    files = build_review(spec, built, manifest)
    assert files == build_review(spec, built, manifest)
    decisions = sorted({b["decision_id"] for b in built.bodies})
    assert {f"decisions/{d}.md" for d in decisions} <= set(files)
    assert {f"markers/{m}.md" for m in spec.markers} <= set(files)
    assert {"index.md", "conditions.md", "openings.md", "families.md"} <= set(files)
    assert len(files) == 60 + 12 + 7
    assert {"annotation.md", "reliability_proposal.md", "all_decisions.md"} <= set(files)
    decision_text = "".join(files[f"decisions/{d}.md"] for d in decisions)
    for b in built.bodies:
        assert decision_text.count(f"`{b['body_id']}` →") == 1      # each body listed once
        if b["marker_id"] != "none":
            assert f"`{b['body_id']}`" in files[f"markers/{b['marker_id']}.md"]
    for s in built.stimuli:
        assert f"`{s['stimulus_id']}`" in decision_text
    assert "shared by" in decision_text


# --- lexical collisions ------------------------------------------------------------


def test_no_marker_is_allocated_to_a_scenario_that_already_contains_it(world):
    spec = world["spec"]
    collide = scenario_collisions(spec, world["texts_by_scenario"])
    assert collide and set().union(*collide.values()) == {"m01_therefore"}
    assert len(collide) == 9
    rows = yaml.safe_load(_committed(spec, "allocation"))["groups"]
    for r in rows:
        assert not set(r["markers"].values()) & collide.get(r["scenario_id"], set())


def test_a_lexical_collision_is_caught(world):
    spec = world["spec"]
    rows = copy.deepcopy(yaml.safe_load(_committed(spec, "allocation"))["groups"])
    victim = next(r for r in rows if r["scenario_id"] == "energy_05_v1")
    victim["markers"]["conclusion_result"] = "m01_therefore"
    assert any("already occurs in the scenario" in p
               for p in allocation_problems(rows, spec, world["texts_by_scenario"]))


# --- annotation units -----------------------------------------------------------------


def test_annotation_units_are_counted_per_level_and_rating(world):
    units = world["manifest"]["annotation_units"]
    for level, want in EXPECTED_UNITS.items():
        for key, value in want.items():
            expected = dict(sorted(value.items())) if isinstance(value, dict) else value
            assert units[level][key] == expected, (level, key)
    lines = [json.loads(line) for line in _committed(world["spec"], "annotation_units").splitlines()]
    assert Counter(u["level"] for u in lines) == Counter({"stimulus": 4320, "pair": 2880,
                                                         "scenario": 120})
    assert all(u["status"] == "outstanding" for u in lines)
    pairs = [u for u in lines if u["level"] == "pair"]
    assert all(u["styled_stimulus_id"].rsplit(".", 1)[1] == u["plain_stimulus_id"].rsplit(".", 1)[1]
               for u in pairs)                                     # same opening
    for u in (u for u in lines if u["level"] == "stimulus"):
        assert ("no_reason_integrity" in u["ratings"]) == (u["condition"] in ("NS", "NP"))
        assert ("inference_function" in u["ratings"]) == (u["condition"] in ("RS", "NS"))
    assert all(u["inherited_ratings"] == [] for u in lines if u["level"] == "scenario")


# --- proposed reliability sample --------------------------------------------------------


def test_the_reliability_proposal_is_balanced_separated_and_not_begun(world):
    rel = json.loads(_committed(world["spec"], "reliability_proposal"))
    assert rel["status"] == "proposed_not_approved" and rel["annotation_begun"] is False
    assert rel["reuses_v2_pin"] is False and rel["sibling_key"] == "decision_id"
    stim = {s["stimulus_id"]: s for s in world["built"].stimuli}
    for name, size in (("stimulus", 504), ("pair", 216), ("scenario", 24)):
        sample = rel["samples"][name]
        members = {m["blind_id"]: m for m in sample["members"]}
        assert sample["size"] == len(members) == size
        orders = sample["orders"]
        assert sorted(orders["annotator_1"]) == sorted(orders["annotator_2"]) == sorted(members)
        assert orders["annotator_1"] != orders["annotator_2"]
        for annotator, order in orders.items():
            distances = [abs(i - j) for i in range(len(order)) for j in range(i + 1, len(order))
                         if members[order[i]]["decision_id"] == members[order[j]]["decision_id"]]
            assert not distances or min(distances) >= 20, (name, annotator)
            assert sample["separation"][annotator]["satisfied"] is True
    rows = [stim[m["unit_id"]] for m in rel["samples"]["stimulus"]["members"]]
    styled = Counter((r["marker_id"], r["condition"], r["domain"], r["opening_id"],
                      r["supported_option"]) for r in rows if r["condition"] in ("RS", "NS"))
    plain = Counter((r["condition"], r["domain"], r["opening_id"], r["supported_option"])
                    for r in rows if r["condition"] in ("RP", "NP"))
    assert len(styled) == 432 and set(styled.values()) == {1}
    assert len(plain) == 36 and set(plain.values()) == {2}
    assert len({r["body_id"] for r in rows}) == 504              # no body twice
    scen = rel["samples"]["scenario"]["members"]
    assert len({m["decision_id"] for m in scen}) == 24


# --- the proposed amendment ---------------------------------------------------------------


def test_the_amendment_is_proposed_and_v1_is_untouched():
    amendment = yaml.safe_load((ROOT / "configs/manipulation_checks_v3_proposed.yaml").read_text())
    assert amendment["status"] == "proposed_not_approved"
    assert amendment["ratings_examined_when_written"] is False
    for key in ("MC7_inference_function", "MC8_anaphoric_no_reason_integrity", "MC9_opening_factor"):
        assert "proposed" in json.dumps(amendment["checks"][key])
    v1 = yaml.safe_load((ROOT / "configs/manipulation_checks_v1.yaml").read_text())
    assert v1["status"] == "researcher_approved_pending_mentor_review"
    assert len(amendment["open_semantic_review_flags"]) == 2


def test_all_decisions_md_covers_every_rendered_counterargument_once(world):
    spec, built, manifest = world["spec"], world["built"], world["manifest"]
    files = build_review(spec, built, manifest)
    text = files["all_decisions.md"]
    assert "(all_decisions.md)" in files["index.md"].split("## Counts")[0]   # linked at the top
    entries = re.findall(r"^- `([^`]+)` · condition \*\*(RS|NS|RP|NP)\*\* · marker `([^`]+)` · "
                         r"opening `([^`]+)` · paired control ([^\n]+)\n\n```text\n([^\n]*)\n```",
                         text, re.M)
    assert len(entries) == 4320 and len(re.findall(r"^```text$", text, re.M)) == 4320
    by_id = {s["stimulus_id"]: s for s in built.stimuli}
    assert sorted(e[0] for e in entries) == sorted(by_id)
    for sid, condition, marker, opening, control, rendered in entries:
        s = by_id[sid]
        assert (condition, marker, opening, rendered) == (
            s["condition"], s["marker_id"], s["opening_id"], s["rendered"])
        if s["paired_control_stimulus_id"]:
            assert f"`{s['paired_control_stimulus_id']}`" in control
    assert len(re.findall(r"^## Scenario `", text, re.M)) == 120
    assert len(re.findall(r"^### `[^`]+` · supported option", text, re.M)) == 240
    for b in built.bodies:                                   # all six bodies per group, in tables
        assert f"| `{b['body_id']}` | {b['condition']} |" in text
    for scenario_id in {b["scenario_id"] for b in built.bodies}:
        b = next(x for x in built.bodies if x["scenario_id"] == scenario_id)
        assert b["scenario_text"] in text and b["options"]["opt_1"] in text
