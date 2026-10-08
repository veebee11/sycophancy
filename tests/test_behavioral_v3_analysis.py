"""Behavioural analysis pipeline (behavioral_v3_detailed_v1).

Needs the isolated analysis environment (``requirements/analysis.txt``); the
module is skipped where NumPy is absent, e.g. in the GPU inference venv. The
integration tests also need the run evidence under ``runs/`` (gitignored) and
skip without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

from reasonstyle.behavioral.analysis.design import (  # noqa: E402
    Block,
    common_semantic_choice_initials,
    common_valid_initials,
    make_block,
)
from reasonstyle.behavioral.analysis.estimate import (  # noqa: E402
    Bootstrap,
    decision_estimate,
    holm,
    mean_of,
    summarise,
)
from reasonstyle.behavioral.analysis.tables import (  # noqa: E402
    flip_cell,
    flip_contrast,
    marker_movement,
    movement_cell,
    movement_contrast,
)
from reasonstyle.behavioral.analysis.validate import (  # noqa: E402
    AnalysisError,
    load_analysis_spec,
    validate_evidence,
)
from reasonstyle.behavioral.analysis.write import (  # noqa: E402
    OutputDir,
    csv_text,
    gzip_bytes,
    verify_directory,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/analysis/behavioral_v3_detailed_v1.yaml"
HAVE_EVIDENCE = (ROOT / "runs/behavioral_v3/base-full-v3/COMPLETE").is_file() and \
    (ROOT / "configs/behavioral_v3_llama31_8b.pinned.yaml").is_file() and \
    (ROOT / "runs/behavioral_v3_robustness/base-minimal-ab-v1/COMPLETE").is_file()
needs_evidence = pytest.mark.skipif(not HAVE_EVIDENCE, reason="run evidence not present")


def _rows(movement=None, flips=None, *, decision="d1", initial="i1", opening="op",
          markers=("m_cr", "m_ib"), pair_ok=True):
    """Six score rows of one block; movement/flip by (cond, marker)."""
    movement = movement or {}
    flips = flips or {}
    rows = []
    for cond, mk, fam in (("RS", markers[0], "conclusion_result"),
                          ("RS", markers[1], "inference_basis"),
                          ("NS", markers[0], "conclusion_result"),
                          ("NS", markers[1], "inference_basis"),
                          ("RP", "none", "none"), ("NP", "none", "none")):
        ctrl = {"RS": "RP", "NS": "NP"}.get(cond)
        rows.append({
            "branch_id": f"{initial}.{cond}.{mk}.{opening}", "condition": cond, "marker_id": mk,
            "marker_family": fam, "stimulus_id": f"{cond}.{mk}.{opening}",
            "paired_control_stimulus_id": (f"{ctrl}.none.{opening}" if pair_ok else "x")
            if ctrl else None,
            "decision_id": decision, "domain": "climate", "scenario_id": "s1", "order_id": "o1",
            "supported_option": "opt_2", "m_before": -0.5,
            "movement_toward_counter": movement.get((cond, mk), 1.0),
            "flip": flips.get((cond, mk), True), "input_tokens": 10})
    return rows


def _block(**kw) -> Block:
    decision = kw.pop("decision", "d1")
    initial = kw.pop("initial", "i1")
    run = kw.pop("run", "base")
    return make_block(run, "base", initial, "op", _rows(decision=decision, initial=initial, **kw),
                      "opt_1")


# --- cell collapse, shared controls, ties --------------------------------------------

def test_styled_cells_average_two_markers_and_controls_enter_once():
    b = _block(movement={("RS", "m_cr"): 2.0, ("RS", "m_ib"): 4.0, ("NS", "m_cr"): 1.0,
                         ("NS", "m_ib"): 0.0, ("RP", "none"): 5.0, ("NP", "none"): -1.0})
    assert b.movement == {"RS": 3.0, "NS": 0.5, "RP": 5.0, "NP": -1.0}
    assert movement_cell("RP")(b) == (5.0, 1, 0)        # one shared control row
    assert movement_cell("RS")(b) == (3.0, 2, 0)        # two marker rows, collapsed


def test_a_block_must_hold_one_marker_per_family_and_its_own_controls():
    rows = _rows()
    with pytest.raises(AnalysisError, match="2 RS"):
        make_block("base", "base", "i1", "op", rows[:-1] + [rows[4]], "opt_1")
    with pytest.raises(AnalysisError, match="not paired"):
        make_block("base", "base", "i1", "op", _rows(pair_ok=False), "opt_1")
    with pytest.raises(AnalysisError, match="2 RS"):
        make_block("base", "base", "i1", "op", rows[:5], "opt_1")


def test_marker_contrasts_pair_with_the_shared_control():
    b = _block(movement={("NS", "m_cr"): 2.0, ("NS", "m_ib"): 0.5, ("NP", "none"): 0.25,
                         ("RS", "m_cr"): 3.0, ("RP", "none"): 1.0})
    assert marker_movement("NS_minus_NP", "m_cr")(b)[0] == 1.75
    assert marker_movement("NS_minus_NP", "m_ib")(b)[0] == 0.25
    assert marker_movement("RS_minus_RP", "m_cr")(b)[0] == 2.0
    assert marker_movement("NS_minus_NP", "m_other")(b) is None


def test_post_ties_stay_in_movement_and_leave_flip_only():
    b = _block(movement={("NP", "none"): 0.5}, flips={("NP", "none"): None,
                                                       ("NS", "m_cr"): None})
    assert movement_cell("NP")(b)[0] == 0.5               # retained for movement
    assert flip_cell("NP")(b) is None                      # undefined for flip
    assert b.flip_cell("NS") == 1.0                        # the defined NS row only
    assert flip_contrast("NS_minus_NP")(b) is None         # needs a defined NP
    assert flip_contrast("RS_minus_RP")(b) == (0.0, 3, 3)


EXPECTED = {"NS_minus_NP": 0.5 - 2.0, "RS_minus_RP": 4.0 - 3.0, "RS_minus_NS": 4.0 - 0.5,
            "RP_minus_NP": 3.0 - 2.0, "interaction": (4.0 - 3.0) - (0.5 - 2.0)}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_contrast_has_the_frozen_sign(name):
    b = _block(movement={("RS", "m_cr"): 4.0, ("RS", "m_ib"): 4.0, ("NS", "m_cr"): 0.0,
                         ("NS", "m_ib"): 1.0, ("RP", "none"): 3.0, ("NP", "none"): 2.0})
    assert movement_contrast(name)(b)[0] == EXPECTED[name]


# --- weighting and bootstrap ---------------------------------------------------------

def _boot(n=60, reps=400, seed=20261005):
    decisions = [f"d{i:02d}" for i in range(n)]
    domain = {d: ("climate", "energy", "technology")[i % 3] for i, d in enumerate(decisions)}
    return Bootstrap(decisions, domain, reps, seed), decisions, domain


def test_blocks_average_within_decision_then_decisions_equally():
    boot, decisions, _ = _boot()
    blocks = [_block(decision=decisions[0], initial=f"a{k}",
                     movement={("NP", "none"): 0.0, ("NS", "m_cr"): 1.0, ("NS", "m_ib"): 1.0})
              for k in range(3)]
    blocks.append(_block(decision=decisions[1], initial="b0",
                         movement={("NP", "none"): 0.0, ("NS", "m_cr"): 4.0, ("NS", "m_ib"): 4.0}))
    est = decision_estimate(boot, blocks, movement_contrast("NS_minus_NP"))
    assert est.point == 2.5                     # (1 + 4) / 2, not (1+1+1+4) / 4
    assert est.denominators["n_decisions"] == 2 and est.denominators["n_blocks"] == 4


def test_bootstrap_is_stratified_seeded_and_saves_its_draws():
    boot, decisions, domain = _boot()
    again, _, _ = _boot()
    assert (boot.draws == again.draws).all()
    assert not (boot.draws == _boot(seed=1)[0].draws).all()
    assert boot.counts.sum(axis=1).tolist() == [60.0] * 400
    for k, dom in enumerate(boot.domains):
        cols = boot.draws[:, 20 * k:20 * (k + 1)]
        assert {domain[boot.decisions[i]] for i in np.unique(cols)} == {dom}
    assert len(boot.decision_ids()[0]) == 60


def test_whole_decisions_are_resampled_jointly_across_runs():
    boot, decisions, _ = _boot()
    def blocks(run):
        return [_block(run=run, decision=d, initial=f"{run}{d}",
                       movement={("NP", "none"): 0.0, ("NS", "m_cr"): float(i % 7),
                                 ("NS", "m_ib"): float(i % 7)})
                for i, d in enumerate(decisions)]
    a = decision_estimate(boot, blocks("base"), movement_contrast("NS_minus_NP"))
    b = decision_estimate(boot, blocks("instruct"), movement_contrast("NS_minus_NP"))
    diff = summarise(b - a, 0.95)
    assert diff["ci_low"] == diff["ci_high"] == 0.0         # identical draws for both runs
    level = summarise(a, 0.95)
    assert level["ci_low"] < level["estimate"] < level["ci_high"]
    fam = mean_of([a, b])
    assert fam.point == a.point and fam.denominators["n_decisions"] == 60


def test_holm():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx(
        {"a": 0.03, "b": 0.06, "c": 0.06})
    assert holm({"a": 0.5, "b": 0.9}) == {"a": 1.0, "b": 1.0}


# --- subsets ---------------------------------------------------------------------------

def _ev(choices: dict[str, dict[str, str | None]]):
    runs = {name: SimpleNamespace(initials=[{"initial_id": i, "excluded": c is None,
                                             "initial_option": c} for i, c in ch.items()])
            for name, ch in choices.items()}
    return SimpleNamespace(runs=runs)


def test_common_valid_and_common_semantic_subsets():
    ev = _ev({"base": {"i1": "opt_1", "i2": None, "i3": "opt_2", "i4": "opt_1"},
              "instruct": {"i1": "opt_2", "i2": "opt_1", "i3": "opt_2", "i4": None},
              "ab_bfirst": {"i1": "opt_1", "i2": "opt_1", "i3": "opt_1", "i4": "opt_1"}})
    assert common_valid_initials(ev) == {"i1", "i3"}
    assert common_semantic_choice_initials(ev, ["base", "ab_bfirst"]) == {"i1", "i4"}


# --- serialisation ---------------------------------------------------------------------

def test_serialisation_is_deterministic_and_overwrite_is_guarded(tmp_path):
    rows = [{"a": 0.1 + 0.2, "b": None, "c": float("nan")}]
    assert csv_text(rows) == "a,b,c\n0.30000000000000004,,\n"
    assert gzip_bytes("x") == gzip_bytes("x")
    od = OutputDir(tmp_path / "out", "v1", overwrite=False)
    od.write("t.csv", "a\n1\n", rows=1)
    od.finish({"analysis_version": "v1"})
    assert verify_directory(tmp_path / "out") == []
    with pytest.raises(FileExistsError, match="--overwrite"):
        OutputDir(tmp_path / "out", "v1", overwrite=False)
    with pytest.raises(FileExistsError, match="does not hold"):
        OutputDir(tmp_path / "out", "other", overwrite=True)
    (tmp_path / "foreign").mkdir()
    with pytest.raises(FileExistsError, match="does not hold"):
        OutputDir(tmp_path / "foreign", "v1", overwrite=True)
    (tmp_path / "out/t.csv").write_text("tampered")
    assert verify_directory(tmp_path / "out") == ["hash mismatch t.csv"]


# --- real evidence -----------------------------------------------------------------------

def _copy_world(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel in ("configs/analysis", "configs/robustness", "data/full_v3", "runs/behavioral_v3",
                "runs/behavioral_v3_robustness"):
        shutil.copytree(ROOT / rel, root / rel)
    for rel in ("configs/behavioral_v3_llama31_8b.pinned.yaml",
                "configs/full_v3_multimarker_openings.yaml"):
        shutil.copyfile(ROOT / rel, root / rel)
    return root


@needs_evidence
def test_real_evidence_validates():
    ev = validate_evidence(load_analysis_spec(SPEC))
    assert {k: len(r.scores) for k, r in ev.runs.items()} == {
        "base": 4266, "instruct": 4140, "ab_bfirst": 4212, "minimal_ab": 3744}
    assert len(ev.runs["base"].exclusions) == 3 and len(ev.runs["instruct"].exclusions) == 10


@needs_evidence
@pytest.mark.parametrize("edit,match", [
    (lambda s: s.replace("e958822a78a853f9a28a6c3f3bcce2238b1cd9d49963588468ab3ab0467d9d25",
                         "f" * 64), "behavioral_scores.jsonl hashes"),
    (lambda s: s.replace("revision: 0e9e39f249a16976918f6564b8830bc894c89659",
                         "revision: other"), "revision"),
    (lambda s: s.replace("excluded_exact_initial_ties: 3, movement_rows: 4266,\n               "
                         "post_exact_ties: 101, flip_defined: 4165",
                         "excluded_exact_initial_ties: 4, movement_rows: 4248,\n               "
                         "post_exact_ties: 101, flip_defined: 4147"), "counts"),
    (lambda s: s.replace("f682fcad2749fb298c196997e01729a09b0e40db589075d9e99a1cf8a18c1d25",
                         "0" * 64, 1), "behavioral_config"),
])
def test_altered_hashes_counts_or_revisions_are_refused(tmp_path, edit, match):
    root = _copy_world(tmp_path)
    spec = root / "configs/analysis/behavioral_v3_detailed_v1.yaml"
    text = spec.read_text()
    changed = edit(text)
    assert changed != text
    spec.write_text(changed)
    with pytest.raises(AnalysisError, match=match):
        validate_evidence(load_analysis_spec(spec))


@needs_evidence
def test_a_tampered_run_file_is_refused(tmp_path):
    root = _copy_world(tmp_path)
    path = root / "runs/behavioral_v3/instruct-full-v3/initial_scores.jsonl"
    path.chmod(0o644)
    path.write_text(path.read_text().replace('"logit_a":', '"logit_a": ', 1))
    with pytest.raises(AnalysisError, match="run_status refused"):
        validate_evidence(load_analysis_spec(root / "configs/analysis/behavioral_v3_detailed_v1.yaml"))


@needs_evidence
def test_repeated_runs_give_byte_identical_outputs(tmp_path):
    outs = []
    for name in ("one", "two"):
        out = tmp_path / name
        subprocess.run([sys.executable, str(ROOT / "scripts/analyze_behavioral_v3.py"),
                        str(SPEC), "--out", str(out)], check=True, cwd=ROOT,
                       capture_output=True)
        outs.append(json.loads((out / "analysis_manifest.json").read_text())["outputs"])
    volatile = {"execution_environment.json"}
    assert {k: v for k, v in outs[0].items() if k not in volatile} == \
        {k: v for k, v in outs[1].items() if k not in volatile}
    assert verify_directory(tmp_path / "one") == []
