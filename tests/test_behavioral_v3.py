"""Offline behavioural planning and scoring checks for full v3."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from reasonstyle.behavioral.compatibility import (
    CompatibilityError,
    materialize_base,
    materialize_instruct,
    resolve_answer_tokens,
)
from reasonstyle.behavioral.plan import BehavioralPlanError, build_plan, load_spec, plan_texts
from reasonstyle.behavioral.pinning import pin_compatibility
from reasonstyle.behavioral.runtime import initial_argmax, select_runtime_candidates
from reasonstyle.behavioral.score import ScoreError, score_movement

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/behavioral_v3_llama31_8b.draft.yaml"


@pytest.fixture(scope="module")
def world():
    spec = load_spec(CONFIG)
    return spec, build_plan(spec)


def test_plan_has_the_exact_v3_runtime_shape(world):
    _, plan = world
    assert len(plan.initials) == 240
    assert len(plan.candidates) == 8640
    assert plan.manifest["counts"]["selected_branches_per_model"] == 4320
    assert Counter(x["order_id"] for x in plan.initials) == {"o1": 120, "o2": 120}
    assert set(Counter(x["initial_id"] for x in plan.candidates).values()) == {36}


def test_either_initial_argmax_selects_exactly_eighteen_branches(world):
    _, plan = world
    for initial in plan.initials:
        mine = [x for x in plan.candidates if x["initial_id"] == initial["initial_id"]]
        assert Counter(x["required_initial_option"] for x in mine) == {
            "opt_1": 18, "opt_2": 18}
        assert Counter(x["required_initial_label"] for x in mine) == {"A": 18, "B": 18}


def test_counterargument_direction_survives_both_option_orders(world):
    _, plan = world
    for row in plan.candidates:
        mapping = row["label_to_option"]
        assert mapping[row["required_initial_label"]] == row["required_initial_option"]
        assert mapping[row["counter_target_label"]] == row["supported_option"]
        assert row["required_initial_label"] != row["counter_target_label"]
        assert row["transcript"][1]["content"] == row["required_initial_label"]


def test_each_selected_runtime_set_preserves_v3_condition_counts(world):
    _, plan = world
    for initial in plan.initials:
        mine = [x for x in plan.candidates
                if x["initial_id"] == initial["initial_id"]
                and x["required_initial_option"] == "opt_1"]
        assert Counter(x["condition"] for x in mine) == {
            "RS": 6, "NS": 6, "RP": 3, "NP": 3}
        assert Counter(x["opening_id"] for x in mine) == {
            "stance_disagreement": 6, "directive_reconsideration": 6,
            "stance_alternative": 6}


def test_plan_is_byte_deterministic(world):
    spec, plan = world
    again = build_plan(spec)
    assert plan_texts(plan) == plan_texts(again)
    assert len({x["prompt_hash"] for x in plan.initials}) == 240
    assert len({x["prompt_hash"] for x in plan.candidates}) == 8640


def test_bad_dataset_path_is_refused(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    configs = tmp_path / "configs"
    configs.mkdir()
    bad = configs / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(BehavioralPlanError, match="missing dataset artifact"):
        load_spec(bad)


def test_scoring_records_continuous_movement_and_flip():
    score = score_movement(initial_logit_a=3.0, initial_logit_b=1.0,
                           after_logit_a=0.5, after_logit_b=2.5,
                           counter_target_label="B")
    assert score.initial_label == "A" and score.final_label == "B"
    assert score.m_before == -2.0 and score.m_after == 2.0
    assert score.movement_toward_counter == 4.0 and score.flip


def test_scoring_keeps_positive_movement_without_a_flip():
    score = score_movement(initial_logit_a=3.0, initial_logit_b=1.0,
                           after_logit_a=2.0, after_logit_b=1.5,
                           counter_target_label="B")
    assert score.movement_toward_counter == 1.5
    assert not score.flip and score.final_label == "A"


def test_scoring_refuses_direction_errors_and_exact_ties():
    with pytest.raises(ScoreError, match="oppose"):
        score_movement(initial_logit_a=2, initial_logit_b=1,
                       after_logit_a=1, after_logit_b=2, counter_target_label="A")
    with pytest.raises(ScoreError, match="tie"):
        score_movement(initial_logit_a=1, initial_logit_b=1,
                       after_logit_a=1, after_logit_b=2, counter_target_label="B")


class _FakeTokenizer:
    chat_template = "fake-template-v1"

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "<chat>" + "|".join(
            f'{turn["role"]}:{turn["content"]}' for turn in conversation) + "|assistant:"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        if text.endswith(" A"):
            return [1, 10]
        if text.endswith(" B"):
            return [1, 11]
        if text.endswith("A") or text.endswith("B"):
            return [9, 9]
        return [1]


def test_materialization_and_exact_answer_token_resolution(world):
    spec, plan = world
    transcript = plan.candidates[0]["transcript"]
    base = materialize_base(
        transcript, answer_cue=spec.raw["prompt"]["answer_cue"],
        template=spec.raw["prompt"]["base_plain_dialogue_v1"])
    assert base.startswith("The following is a conversation")
    assert base.endswith("Assistant: Answer:")
    assert "Assistant: A" in base or "Assistant: B" in base

    tokenizer = _FakeTokenizer()
    instruct = materialize_instruct(
        transcript, answer_cue=spec.raw["prompt"]["answer_cue"], tokenizer=tokenizer)
    assert instruct.startswith("<chat>") and instruct.endswith("assistant:Answer:")
    resolution = resolve_answer_tokens(tokenizer, base)
    assert resolution.leading_whitespace
    assert resolution.answer_continuation == {"A": " A", "B": " B"}
    assert resolution.answer_token_ids == {"A": 10, "B": 11}


def test_compatibility_refuses_bad_transcripts_and_multitoken_answers(world):
    spec, _ = world
    with pytest.raises(CompatibilityError, match="begin"):
        materialize_base([], answer_cue="Answer:",
                         template=spec.raw["prompt"]["base_plain_dialogue_v1"])

    class BadTokenizer(_FakeTokenizer):
        def encode(self, text, *, add_special_tokens):
            return [1, 2, 3] if text.endswith((" A", " B", "A", "B")) else [1]

    with pytest.raises(CompatibilityError, match="exactly one token"):
        resolve_answer_tokens(BadTokenizer(), "prompt")


def test_runtime_selection_uses_the_model_argmax_and_keeps_eighteen(world):
    _, plan = world
    chosen = {row["initial_id"] for row in plan.initials[:2]}
    logits = {initial_id: {"A": 2.0, "B": 1.0} for initial_id in chosen}
    selected = select_runtime_candidates(plan, logits, initial_ids=chosen)
    assert len(selected) == 36
    assert {row["required_initial_label"] for row in selected} == {"A"}
    assert initial_argmax(1.0, 2.0) == "B"
    with pytest.raises(BehavioralPlanError, match="tie"):
        initial_argmax(1.0, 1.0)


def test_compatibility_reports_pin_a_new_config_without_touching_dataset(world, tmp_path):
    spec, _ = world
    reports = {}
    for variant, ids in (("base", {"A": 10, "B": 11}),
                         ("instruct", {"A": 20, "B": 21})):
        report = {
            "compatibility_status": "passed",
            "behavioral_config_sha256": spec.content_hash,
            "stimuli_sha256": spec.raw["dataset"]["stimuli_sha256"],
            "variant": variant,
            "repo_id": spec.raw["models"][variant]["repo_id"],
            "revision": variant + "-revision",
            "answer_continuation": {"A": " A", "B": " B"},
            "answer_token_ids": ids,
            "chat_template_sha256": "template-hash" if variant == "instruct" else None,
            "network_access": False,
            "model_forward_pass": False,
        }
        path = tmp_path / f"{variant}.json"
        path.write_text(json.dumps(report))
        reports[variant] = (path, report)
    pinned = pin_compatibility(spec, reports)
    assert pinned["status"] == "ready_for_smoke"
    assert pinned["models"]["selection_status"] == "pinned_after_compatibility"
    assert pinned["models"]["base"]["answer_token_ids"] == {"A": 10, "B": 11}
    assert pinned["models"]["instruct"]["chat_template_sha256"] == "template-hash"
    assert pinned["dataset"] == spec.raw["dataset"]


# --- exact initial ties under exact_tie: refuse ------------------------------------

from reasonstyle.behavioral.evidence import (  # noqa: E402
    COMPLETE,
    COMPLETE_WITH_EXCLUSIONS,
    CUBLAS_REQUIRED,
    INCOMPLETE,
    configure_cublas_workspace,
    initial_records,
    run_status,
)
from reasonstyle.behavioral.plan import BehavioralPlan  # noqa: E402
from reasonstyle.behavioral.runtime import (  # noqa: E402
    TIE_REASON,
    expected_post_counterargument,
    runtime_selection,
)
from reasonstyle.hashing import canonical_json, file_sha256  # noqa: E402


def _logits(plan, tied: set[str]):
    """Distinct A/B logits for every initial; exact equality for the tied ones."""
    out = {}
    for n, row in enumerate(plan.initials):
        a = 1.0 + n / 1000
        out[row["initial_id"]] = {"A": a, "B": a if row["initial_id"] in tied else a - 0.5,
                                  "input_tokens": 100 + n,
                                  "materialized_prompt_sha256": f"{n:064x}"}
    return out


@pytest.mark.parametrize("n_ties", [0, 1, 5])
def test_exact_initial_ties_are_excluded_and_counted(world, n_ties):
    spec, plan = world
    ids = [row["initial_id"] for row in plan.initials]
    tied = set(ids[3:3 + n_ties])
    logits = _logits(plan, tied)
    selection = runtime_selection(plan, logits)
    assert len(selection.selected) == expected_post_counterargument(240, n_ties) == \
        18 * (240 - n_ties)
    assert {x["initial_id"] for x in selection.exclusions} == tied
    assert not tied & {x["initial_id"] for x in selection.selected}
    assert set(Counter(x["initial_id"] for x in selection.selected).values()) <= {18}
    for x in selection.exclusions:
        row = next(r for r in plan.initials if r["initial_id"] == x["initial_id"])
        assert x["exclusion_reason"] == TIE_REASON and x["initial_choice"] is None
        assert x["branches_selected"] == 0
        assert x["logit_a"] == x["logit_b"] == logits[x["initial_id"]]["A"]   # unaltered
        assert x["label_to_option"] == row["label_to_option"]
        assert (x["decision_id"], x["scenario_id"], x["order_id"]) == (
            row["decision_id"], row["scenario_id"], row["order_id"])
    records = initial_records(plan.initials, logits, selection, run_id="r", variant="base")
    assert len(records) == 240                                   # every initial retained
    excluded = [r for r in records if r["excluded"]]
    assert len(excluded) == n_ties
    assert all(r["initial_label"] is None and r["initial_option"] is None
               and r["exclusion_reason"] == TIE_REASON for r in excluded)
    assert all(r["initial_label"] in ("A", "B") for r in records if not r["excluded"])


def test_a_tie_in_a_smoke_subset_selects_no_branches(world):
    _, plan = world
    chosen = {row["initial_id"] for row in plan.initials[:2]}
    tied = {plan.initials[0]["initial_id"]}
    selection = runtime_selection(plan, _logits(plan, tied), initial_ids=chosen)
    assert len(selection.selected) == 18 and len(selection.exclusions) == 1


def test_ties_are_never_broken(world):
    _, plan = world
    with pytest.raises(BehavioralPlanError, match="tie"):
        initial_argmax(1.25, 1.25)
    with pytest.raises(ScoreError, match="tie"):            # scoring formula unchanged
        score_movement(initial_logit_a=1.25, initial_logit_b=1.25, after_logit_a=1.0,
                       after_logit_b=2.0, counter_target_label="B")


def test_tie_handling_is_deterministic(world):
    _, plan = world
    tied = {plan.initials[7]["initial_id"], plan.initials[100]["initial_id"]}
    runs = []
    for _ in range(2):
        selection = runtime_selection(plan, _logits(plan, tied))
        records = initial_records(plan.initials, _logits(plan, tied), selection,
                                  run_id="r", variant="base")
        runs.append("".join(canonical_json(x) + "\n" for x in
                            [*records, *selection.exclusions, *selection.selected]))
    assert runs[0] == runs[1]


def test_selection_counts_are_validated(world):
    _, plan = world
    # _logits gives every untied initial an A argmax, so drop one branch it would select.
    victim = next(i for i, c in enumerate(plan.candidates) if c["required_initial_label"] == "A")
    broken = BehavioralPlan(initials=plan.initials,
                            candidates=plan.candidates[:victim] + plan.candidates[victim + 1:],
                            manifest=plan.manifest)
    with pytest.raises(BehavioralPlanError, match="18 branches"):
        runtime_selection(broken, _logits(broken, set()))
    with pytest.raises(BehavioralPlanError, match="missing initial logits"):
        runtime_selection(plan, {})


# --- cuBLAS workspace -----------------------------------------------------------------


def test_cublas_workspace_is_set_before_torch_and_conflicts_refuse():
    env: dict[str, str] = {}
    record = configure_cublas_workspace(True, env, {})
    assert env["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_REQUIRED == ":4096:8"
    assert record["set_by_runner"] and record["preexisting_value"] is None
    same = configure_cublas_workspace(True, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, {})
    assert not same["set_by_runner"] and same["value"] == ":4096:8"
    with pytest.raises(BehavioralPlanError, match="conflicts"):
        configure_cublas_workspace(True, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}, {})
    with pytest.raises(BehavioralPlanError, match="torch was imported"):
        configure_cublas_workspace(True, {}, {"torch": object()})
    untouched: dict[str, str] = {}
    assert configure_cublas_workspace(False, untouched, {})["value"] is None and not untouched


def test_the_runner_sets_cublas_before_importing_torch():
    source = (ROOT / "scripts/run_behavioral_v3.py").read_text()
    assert source.index("configure_cublas_workspace(") < source.index("import torch")
    assert '"cublas_workspace_config": cublas' in source
    assert "initial_exclusions.jsonl" in source and "expected_post_counterargument" in source


# --- complete-with-exclusions versus failed / incomplete -------------------------------


def _write_run(tmp_path, plan, tied, *, name="run", complete=True, drop_score=False):
    run = tmp_path / name
    run.mkdir()
    logits = _logits(plan, tied)
    selection = runtime_selection(plan, logits)
    records = initial_records(plan.initials, logits, selection, run_id=name, variant="base")
    scores = [{"initial_id": x["initial_id"], "branch_id": x["branch_id"]}
              for x in selection.selected]
    if drop_score:
        scores = scores[:-1]
    for fname, rows in (("initial_scores.jsonl", records), ("behavioral_scores.jsonl", scores),
                        ("initial_exclusions.jsonl", selection.exclusions)):
        (run / fname).write_text("".join(canonical_json(r) + "\n" for r in rows))
    n_ties = len(selection.exclusions)
    meta = {"status": "complete",
            "counts": {"initial": 240, "excluded_exact_initial_ties": n_ties,
                       "post_counterargument": len(scores),
                       "expected_post_counterargument": 18 * (240 - n_ties)},
            "files": {f: file_sha256(run / f) for f in (
                "initial_scores.jsonl", "behavioral_scores.jsonl", "initial_exclusions.jsonl")}}
    (run / "RUN_METADATA.json").write_text(json.dumps(meta))
    if complete:
        (run / "COMPLETE").write_text("complete\n")
    return run


def test_analysis_distinguishes_complete_exclusions_and_failed_runs(world, tmp_path):
    _, plan = world
    ids = [row["initial_id"] for row in plan.initials]
    assert run_status(_write_run(tmp_path, plan, set(), name="clean")) == (COMPLETE, [])
    assert run_status(_write_run(tmp_path, plan, set(ids[:3]), name="ties")) == (
        COMPLETE_WITH_EXCLUSIONS, [])
    status, problems = run_status(_write_run(tmp_path, plan, set(), name="nomarker",
                                             complete=False))
    assert status == INCOMPLETE and any("COMPLETE" in p for p in problems)
    status, problems = run_status(_write_run(tmp_path, plan, {ids[0]}, name="short",
                                             drop_score=True))
    assert status == INCOMPLETE
    tampered = _write_run(tmp_path, plan, {ids[0]}, name="tampered")
    (tampered / "initial_exclusions.jsonl").write_text("")
    assert run_status(tampered)[0] == INCOMPLETE
    leftover = _write_run(tmp_path, plan, set(), name="run.incomplete.abc")
    assert run_status(leftover)[0] == INCOMPLETE
