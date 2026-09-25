"""Seed-aware full-corpus assembly: ``pilot.py assemble`` on a design that
imports the frozen v2 pilot.

Everything is isolated in ``tmp_path``. The six declared seed files are byte
copies (so every pin still matches) and the seed's run directory is pointed at
a path that does not exist, so no run evidence is read. The only real files
opened are committed, tracked inputs, read-only: the frozen configuration
(copied, with its paths redirected), the topic bank, the full allocation, and
the seed manifest's own scenario-source approvals ledger and configuration.
The new material is a synthetic, offline full run: 96 scenarios and 192 groups
drafted through ``FakeBackend``, one group made machine-invalid and repaired by
an approved cell correction, and one scenario corrected after its groups were
drafted. No request leaves the process and nothing under ``data/`` is written.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from datetime import date
from pathlib import Path

import pytest
import yaml

from reasonstyle.config import load_config
from reasonstyle.corpus import segmenter_from_config
from reasonstyle.corpus.topics import load_topic_bank
from reasonstyle.corpus.validate import endorsement_text
from reasonstyle.generation import FakeBackend
from reasonstyle.generation.allocation import load_allocation
from reasonstyle.generation.approvals import REQUIRED_JUDGEMENTS, ScenarioApproval, save_approvals
from reasonstyle.generation.assemble import ManualCorrection, save_corrections
from reasonstyle.generation.corpus_source import (
    combined_corpus_problems,
    load_seed_corpus,
    plan_full_corpus,
)
from reasonstyle.generation.pipeline import (
    CallStore,
    draft_group,
    recorded_groups,
    recorded_scenarios,
    run_scenario_stage,
)
from reasonstyle.generation.scenario_corrections import (
    ScenarioCorrection,
    save_scenario_corrections,
)
from reasonstyle.hashing import content_hash, sha256_of

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "configs/frozen/v2_full.yaml"
BANK = load_topic_bank(ROOT / "data/topics/full_topics_v2.yaml")
ALLOCATION = load_allocation(ROOT / "data/full/marker_allocation_full_v2.yaml")
CFG = load_config(FULL)
SEED_KEYS = ("config", "corpus", "manifest", "allocation", "corrections", "topics")
BROKEN_GROUP = ("energy_18_v1", "opt_1")
CORRECTED_SCENARIO = "climate_10_v1"


def _load_pilot_script():
    spec = importlib.util.spec_from_file_location("pilot_script_full_assembly",
                                                  ROOT / "scripts/pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scenario_text(scenario_id: str) -> str:
    return (f"Record {scenario_id} sets out a regional capacity decision for the coming "
            f"winters. The operator can extend the existing baseload plant or accelerate "
            f"the storage build already under tender. Both routes are funded and feasible, "
            f"and nothing states how reliability should be weighed against emissions. The "
            f"extended plant can deliver full output through any cold spell. Retiring it on "
            f"schedule would cut power-sector emissions. One route must be chosen before "
            f"the tender closes.")


def bodies_for(scenario_id: str, option: str) -> dict[str, str]:
    """Four v2 cells in the live run's own form: a premise taken from the
    scenario, then the allocated marker realization before the fixed
    endorsement (a comma only after a conclusion indicator)."""
    row = next(g for g in ALLOCATION.groups
               if (g.scenario_id, g.supported_option) == (scenario_id, option))
    topic = next(t for t in BANK.topics if t.decision_id == row.decision_id)
    endorsement = endorsement_text(topic.options[option], CFG) + "."
    styled = row.marker_string[0].upper() + row.marker_string[1:]
    marker = f"{styled}, " if row.marker_family == "conclusion_indicator" else f"{styled} "
    premise = f"Record {scenario_id} sets out a regional capacity decision for the coming winters."
    return {"RS": f"{premise} {marker}{endorsement}", "RP": f"{premise} {endorsement}",
            "NS": f"{marker}{endorsement}", "NP": endorsement}


class Responder:
    def __call__(self, request):
        scenario_id = f"{request.decision_id}_v{request.variant_id}"
        if request.kind == "scenario":
            return {"scenario_text": scenario_text(scenario_id)}
        bodies = bodies_for(scenario_id, request.supported_option)
        if (scenario_id, request.supported_option) == BROKEN_GROUP:
            bodies["RP"] = bodies["RP"].rstrip(".")          # a machine error to correct
        return bodies


def _isolated_config(tmp: Path, seed_dir: Path, corpus: Path) -> Path:
    """The frozen full configuration, with every run, ledger, corpus and seed
    path redirected into ``tmp``; pins unchanged."""
    text = FULL.read_text(encoding="utf-8")
    replace = {
        "  run: data/full/run_v2\n": f"  run: {tmp / 'run_v2'}\n",
        "  approvals: data/full/scenario_approvals_full_v2.yaml\n":
            f"  approvals: {tmp / 'approvals.yaml'}\n",
        "  scenario_corrections: data/full/scenario_corrections_full_v2.yaml\n":
            f"  scenario_corrections: {tmp / 'scenario_corrections.yaml'}\n",
        "  corrections: data/full/manual_corrections_full_v2.yaml\n":
            f"  corrections: {tmp / 'manual_corrections.yaml'}\n",
        "  corpus: data/full/corpus_full_v2.jsonl\n": f"  corpus: {corpus}\n",
        "  run: data/pilot/run_v2\n": f"  run: {tmp / 'no_seed_run_evidence'}\n",
    }
    spec = yaml.safe_load(text)["seed_corpus"]
    for key in SEED_KEYS:
        replace[f"  {key}: {spec[key]}\n"] = f"  {key}: {seed_dir / Path(spec[key]).name}\n"
    for old, new in replace.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    frozen = tmp / "frozen"
    frozen.mkdir(exist_ok=True)
    (frozen / "v2_full.yaml").write_text(text, encoding="utf-8")
    return frozen / "v2_full.yaml"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A complete synthetic full run beside a byte-copied seed. Built once."""
    tmp = tmp_path_factory.mktemp("full_assembly")
    seed_dir = tmp / "seed"
    seed_dir.mkdir()
    spec = yaml.safe_load(FULL.read_text(encoding="utf-8"))["seed_corpus"]
    for key in SEED_KEYS:
        shutil.copyfile(ROOT / spec[key], seed_dir / Path(spec[key]).name)
    cfg_path = _isolated_config(tmp, seed_dir, tmp / "unused" / "corpus_full_v2.jsonl")
    cfg = load_config(cfg_path)
    seed = load_seed_corpus(cfg, bank=BANK)
    plan = plan_full_corpus(cfg, BANK, seed)
    bank_hash = content_hash(BANK.model_dump(mode="json"))
    segmenter = segmenter_from_config(cfg)
    topics = sorted((t for t in BANK.topics if t.decision_id in set(plan.new_decisions)),
                    key=lambda t: t.decision_id)

    store = CallStore(tmp / "run_v2", cfg, topic_bank_content_hash=bank_hash,
                      allocation_content_hash=ALLOCATION.content_hash)
    backend = FakeBackend(Responder())
    run_scenario_stage(topics, cfg, segmenter, backend, store)
    scenarios = recorded_scenarios(store)
    by_key = {(g.scenario_id, g.supported_option): g for g in ALLOCATION.groups}
    for topic in topics:
        for variant in (1, 2):
            sid = f"{topic.decision_id}_v{variant}"
            for option in ("opt_1", "opt_2"):
                draft_group(topic, variant, scenarios[sid]["scenario_text"],
                            by_key[(sid, option)], cfg, segmenter, backend, store,
                            allow_live=True, max_calls_per_group=1)
    groups = recorded_groups(store)
    assert len(scenarios) == 96 and len(groups) == 192
    assert not groups[BROKEN_GROUP]["accepted"]

    # One scenario corrected AFTER its groups were drafted, then re-approved.
    original = scenarios[CORRECTED_SCENARIO]
    corrected_text = original["scenario_text"] + " The operator must choose one route."
    save_scenario_corrections([ScenarioCorrection(
        scenario_id=CORRECTED_SCENARIO, original_call_id=original["call_id"],
        original_text=original["scenario_text"], corrected_text=corrected_text,
        editor="Test Curator", reason="synthetic scenario correction",
        decided_at=date(2026, 9, 25), approval_state="approved")],
        tmp / "scenario_corrections.yaml")
    approvals = {}
    for sid, record in scenarios.items():
        text = corrected_text if sid == CORRECTED_SCENARIO else record["scenario_text"]
        approvals[sid] = ScenarioApproval(
            scenario_id=sid, scenario_text_sha256=sha256_of(text), call_id=record["call_id"],
            config_content_hash=cfg.content_hash, topic_bank_content_hash=bank_hash,
            decision="approved", judgements={n: True for n in REQUIRED_JUDGEMENTS},
            decided_by="Test Curator", decided_at=date(2026, 9, 25), reason=None)
    save_approvals(approvals, tmp / "approvals.yaml")

    broken = groups[BROKEN_GROUP]
    save_corrections([ManualCorrection(
        scenario_id=BROKEN_GROUP[0], supported_option=BROKEN_GROUP[1], condition="RP",
        original_call_id=broken["call_id"], original_text=broken["bodies"]["RP"],
        corrected_text=broken["bodies"]["RP"] + ".", editor="Test Curator",
        reason="synthetic terminal punctuation", decided_at=date(2026, 9, 25),
        approval_state="approved")], tmp / "manual_corrections.yaml")
    return {"tmp": tmp, "cfg_path": cfg_path, "seed_dir": seed_dir, "seed": seed, "plan": plan,
            "cfg": cfg,
            "groups": groups, "segmenter": segmenter}


def _assemble(world, out_dir: Path, *extra: str) -> tuple[int, Path, str]:
    """``pilot.py assemble`` on the isolated configuration, writing to
    ``out_dir`` through the existing ``--corpus`` flag. One configuration per
    world: its content hash is what the approvals bind."""
    corpus = out_dir / "corpus_full_v2.jsonl"
    rc = _load_pilot_script().main(["assemble", "--config", str(world["cfg_path"]),
                                    "--corpus", str(corpus), *extra])
    return rc, corpus, str(world["cfg_path"])


def test_assembles_the_seed_and_the_new_material_into_one_draft_corpus(world, tmp_path, capsys):
    rc, corpus, _ = _assemble(world, tmp_path)
    out = capsys.readouterr()
    assert rc == 0, out.err
    manifest = json.loads(corpus.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    body = corpus.read_text(encoding="utf-8")
    lines = body.splitlines()
    records = [json.loads(line) for line in lines]

    assert len(lines) == 120
    assert (manifest["n_decisions"], manifest["n_scenarios"], manifest["n_groups"],
            manifest["n_texts"]) == (60, 120, 240, 960)
    assert manifest["counts_by_source"] == {"seed": {"scenarios": 24, "texts": 192},
                                            "full_run": {"scenarios": 96, "texts": 768}}
    assert manifest["machine_errors"] == 0
    assert manifest["seed_config_hash_findings"]["count"] == 24
    assert manifest["validation_status"] == "draft"
    assert {r["validation"]["status"] for r in records} == {"draft"}
    assert manifest["corpus_sha256"] == sha256_of(body)
    # The seed passes through byte for byte, under its own configuration hash.
    seed_lines = (world["seed_dir"] / "corpus_v2.jsonl").read_text(encoding="utf-8").splitlines()
    assert set(seed_lines) <= set(lines)
    assert sorted(k for k, v in manifest["scenario_sources"].items() if v == "seed") \
        == sorted(world["seed"].lines)

    # The approved cell correction is applied and recorded.
    rp = next(r for r in records if r["scenario_id"] == BROKEN_GROUP[0]) \
        ["counterarguments"][BROKEN_GROUP[1]]["cells"]["RP"]["body"]
    assert rp == world["groups"][BROKEN_GROUP]["bodies"]["RP"] + "."
    run_source = manifest["sources"][1]
    applied = [c for s in run_source["scenarios"] for c in s["manual_corrections"]]
    assert [(c["scenario_id"], c["condition"], c["approval_state"]) for c in applied] \
        == [(BROKEN_GROUP[0], "RP", "approved")]

    # The scenario correction is applied, and its groups are recorded as drafted
    # from the original text.
    record = next(r for r in records if r["scenario_id"] == CORRECTED_SCENARIO)
    assert record["scenario_text"].endswith("The operator must choose one route.")
    [scenario_correction] = manifest["scenario_corrections"]
    assert {o: g["drafted_from"] for o, g in scenario_correction["groups"].items()} \
        == {"opt_1": "original", "opt_2": "original"}
    assert set(manifest["inputs"]) == {"config", "topic_bank", "allocation", "scenario_approvals",
                                       "scenario_corrections", "manual_corrections"}
    assert all(v["sha256"] for v in manifest["inputs"].values())


def test_an_existing_corpus_is_never_replaced_without_overwrite(world, tmp_path, capsys):
    rc, corpus, _ = _assemble(world, tmp_path)
    assert rc == 0
    before = corpus.read_bytes(), corpus.with_suffix(".manifest.json").read_bytes()
    capsys.readouterr()
    rc, _, _ = _assemble(world, tmp_path)
    assert rc == 1 and "already exists" in capsys.readouterr().err
    assert (corpus.read_bytes(), corpus.with_suffix(".manifest.json").read_bytes()) == before
    rc, _, _ = _assemble(world, tmp_path, "--overwrite")
    assert rc == 0
    assert (corpus.read_bytes(), corpus.with_suffix(".manifest.json").read_bytes()) == before


def test_a_group_with_an_unresolved_machine_error_refuses_everything(world, tmp_path, capsys):
    ledger = world["tmp"] / "manual_corrections.yaml"
    kept = ledger.read_bytes()
    ledger.write_text("[]\n", encoding="utf-8")
    try:
        rc, corpus, _ = _assemble(world, tmp_path)
    finally:
        ledger.write_bytes(kept)
    assert rc == 1
    assert f"{BROKEN_GROUP[0]}/{BROKEN_GROUP[1]}" in capsys.readouterr().err
    assert not corpus.exists() and not corpus.with_suffix(".manifest.json").exists()


def test_a_seed_that_does_not_match_its_pins_refuses_everything(world, tmp_path, capsys):
    seed_corpus = world["seed_dir"] / "corpus_v2.jsonl"
    kept = seed_corpus.read_bytes()
    seed_corpus.write_bytes(kept + b"\n")
    try:
        rc, corpus, _ = _assemble(world, tmp_path)
    finally:
        seed_corpus.write_bytes(kept)
    assert rc == 1 and "the seed corpus does not verify" in capsys.readouterr().err
    assert not corpus.exists()


def test_combined_corpus_problems_catches_coverage_count_and_marker_faults(world, tmp_path):
    rc, corpus, _ = _assemble(world, tmp_path)
    assert rc == 0
    body = corpus.read_text(encoding="utf-8")
    args = (world["seed"], world["plan"])
    rest = (ALLOCATION, world["cfg"], world["segmenter"])
    assert combined_corpus_problems(*args, body, *rest)[0] == []

    lines = body.splitlines()
    dropped = "".join(line + "\n" for line in lines[:-1])
    problems = combined_corpus_problems(*args, dropped, *rest)[0]
    assert any("allocation coverage is not exact" in p for p in problems)
    assert any("the plan requires" in p for p in problems)

    record = json.loads(lines[-1])
    block = next(iter(record["counterarguments"].values()))
    block["marker_realization_id"] = "not_the_allocated_realization"
    tampered = "".join(line + "\n" for line in lines[:-1]) + json.dumps(record) + "\n"
    problems = combined_corpus_problems(*args, tampered, *rest)[0]
    assert any("marker fields differ from the allocation" in p or "E_REALIZATION" in p
               for p in problems)


# --- a seed cell corrected as a full-design overlay -----------------------------------

SEED_TARGET = ("energy_03_v1", "opt_1")
SEED_OLD = "Several completed wind-farm designs"
SEED_NEW = "Completed wind-farm designs"


def _seed_corrections(world, *, original_suffix: str = "") -> list[ManualCorrection]:
    manifest = json.loads((world["seed_dir"] / "corpus_v2.manifest.json").read_text())
    call = next(s for s in manifest["scenarios"]
                if s["scenario_id"] == SEED_TARGET[0])["group_call_ids"][SEED_TARGET[1]]
    record = next(r for r in world["seed"].records if r.scenario_id == SEED_TARGET[0])
    out = []
    for condition in ("RS", "RP"):
        body = record.counterarguments[SEED_TARGET[1]].cells[condition].body
        out.append(ManualCorrection(
            scenario_id=SEED_TARGET[0], supported_option=SEED_TARGET[1], condition=condition,
            original_call_id=call, original_text=body + original_suffix,
            corrected_text=body.replace(SEED_OLD, SEED_NEW), editor="Test Curator",
            reason="synthetic seed overlay", decided_at=date(2026, 9, 25),
            approval_state="approved"))
    return out


def _with_ledger(world, extra):
    from reasonstyle.generation.assemble import load_corrections
    ledger = world["tmp"] / "manual_corrections.yaml"
    kept = ledger.read_bytes()
    save_corrections(load_corrections(ledger) + extra, ledger)
    return ledger, kept


def test_a_seed_cell_is_corrected_only_as_a_full_design_overlay(world, tmp_path, capsys):
    seed_before = {p.name: p.read_bytes() for p in world["seed_dir"].iterdir()}
    ledger, kept = _with_ledger(world, _seed_corrections(world))
    try:
        rc, corpus, _ = _assemble(world, tmp_path)
    finally:
        ledger.write_bytes(kept)
    assert rc == 0, capsys.readouterr().err
    # The frozen seed files are untouched.
    assert {p.name: p.read_bytes() for p in world["seed_dir"].iterdir()} == seed_before

    lines = corpus.read_text(encoding="utf-8").splitlines()
    by_id = {json.loads(line)["scenario_id"]: line for line in lines}
    seed_lines = {json.loads(line)["scenario_id"]: line for line in
                  seed_before["corpus_v2.jsonl"].decode().splitlines()}
    assert [s for s in seed_lines if by_id[s] != seed_lines[s]] == [SEED_TARGET[0]]
    record = json.loads(by_id[SEED_TARGET[0]])
    cells = record["counterarguments"][SEED_TARGET[1]]["cells"]
    assert all(cells[c]["body"].startswith(SEED_NEW) for c in ("RS", "RP"))
    original = json.loads(seed_lines[SEED_TARGET[0]])["counterarguments"][SEED_TARGET[1]]["cells"]
    assert all(cells[c]["body"] == original[c]["body"] for c in ("NS", "NP"))
    assert record["config_content_hash"] == world["seed"].provenance["config_content_hash"]

    manifest = json.loads(corpus.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed_integrity"]["byte_identical_to_pinned_seed"] == 23
    assert manifest["seed_integrity"]["corrected_by_full_design_overlay"] == [SEED_TARGET[0]]
    overlay = manifest["seed_overlays"][SEED_TARGET[0]]
    assert overlay["original_line_sha256"] == sha256_of(seed_lines[SEED_TARGET[0]])
    assert overlay["corrected_line_sha256"] == sha256_of(by_id[SEED_TARGET[0]])
    assert [c["condition"] for c in overlay["corrections"]] == ["RS", "RP"]
    assert manifest["machine_errors"] == 0 and manifest["n_texts"] == 960


def test_a_seed_correction_bound_to_other_text_refuses(world, tmp_path, capsys):
    ledger, kept = _with_ledger(world, _seed_corrections(world, original_suffix=" stale"))
    try:
        rc, corpus, _ = _assemble(world, tmp_path)
    finally:
        ledger.write_bytes(kept)
    assert rc == 1 and "no longer there" in capsys.readouterr().err
    assert not corpus.exists()


def test_a_correction_that_applies_nowhere_refuses(world, tmp_path, capsys):
    [stray, _] = _seed_corrections(world)
    stray = ManualCorrection(**{**{f: getattr(stray, f) for f in stray.__dataclass_fields__},
                                "scenario_id": "climate_99_v1"})
    ledger, kept = _with_ledger(world, [stray])
    try:
        rc, corpus, _ = _assemble(world, tmp_path)
    finally:
        ledger.write_bytes(kept)
    assert rc == 1 and "must apply exactly once" in capsys.readouterr().err
    assert not corpus.exists()
