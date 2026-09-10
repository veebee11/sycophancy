"""Prompt renderer: semantic direction, branch isolation, and determinism.

Nothing here loads a model or a tokenizer. The canonical object under test is
the model-independent transcript; materialization into a base-model string is
tested separately, and materialization for an instruct model is asserted to be
deferred rather than invented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from reasonstyle.config import load_config
from reasonstyle.corpus import load_corpus
from reasonstyle.rendering import (
    AnswerContinuationUnresolved,
    MaterializationDeferred,
    RenderingError,
    Transcript,
    branches_share_prefix,
    materialize,
    option_orders,
    render_branches,
    render_initial,
)
from reasonstyle.schemas import CORE_CONDITIONS

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixtures" / "tiny_corpus.jsonl"
BASE = "base_scaffold_v1"
INSTRUCT = "instruct_chat_v1"
TEST_CONTINUATION = {"A": " A", "B": " B"}


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs" / "experiment_v4.yaml")


@pytest.fixture(scope="module")
def record():
    return load_corpus(FIXTURE)[0]


@pytest.fixture(scope="module")
def orders(cfg):
    return option_orders(cfg)


# --- option orders ----------------------------------------------------------


def test_both_option_orders_exist_and_are_mirror_images(orders):
    o1, o2 = orders
    assert (o1.order_id, o2.order_id) == ("o1", "o2")
    assert o1.label_to_option == {"A": "opt_1", "B": "opt_2"}
    assert o2.label_to_option == {"A": "opt_2", "B": "opt_1"}
    for order in orders:
        assert order.label_for(order.option_for("A")) == "A"
        assert set(order.label_to_option.values()) == {"opt_1", "opt_2"}


def test_swapping_the_order_swaps_the_displayed_text(record, orders, cfg):
    o1, o2 = orders
    first = render_initial(record, o1, cfg, BASE).transcript.turns[0].content
    second = render_initial(record, o2, cfg, BASE).transcript.turns[0].content
    assert f"A. {record.options['opt_1']}" in first
    assert f"A. {record.options['opt_2']}" in second
    assert first != second


# --- semantic direction is invariant under display order --------------------


@pytest.mark.parametrize("initial", ["opt_1", "opt_2"])
def test_counterarguments_always_oppose_the_initial_choice(record, orders, cfg, initial):
    opposite = record.opposing_option(initial)
    for order in orders:
        for branch in render_branches(record, order, cfg, BASE, initial):
            assert branch.initial_option == initial
            assert branch.counter_target_option == opposite
            body = record.counterarguments[opposite].cells[branch.condition].body
            assert body in branch.transcript.turns[2].content


@pytest.mark.parametrize("initial", ["opt_1", "opt_2"])
def test_option_order_never_changes_semantic_direction(record, orders, cfg, initial):
    """The letters differ between orders; the semantics must not."""
    o1, o2 = orders
    a = render_branches(record, o1, cfg, BASE, initial)
    b = render_branches(record, o2, cfg, BASE, initial)
    for first, second in zip(a, b):
        assert first.condition == second.condition
        assert first.counter_target_option == second.counter_target_option
        assert first.initial_option == second.initial_option
        # the counterargument text itself is identical, only the letters move
        assert first.transcript.turns[2] == second.transcript.turns[2]
    assert a[0].initial_label != b[0].initial_label
    assert a[0].counter_target_label != b[0].counter_target_label


def test_display_letters_never_become_the_stored_identity(record, orders, cfg):
    for order in orders:
        for branch in render_branches(record, order, cfg, BASE, "opt_1"):
            assert branch.counter_target_option in {"opt_1", "opt_2"}
            assert branch.initial_option in {"opt_1", "opt_2"}
            assert branch.initial_label in {"A", "B"}
            # the letter is only ever a lookup away from the identity
            assert branch.label_to_option[branch.initial_label] == branch.initial_option
            assert branch.label_to_option[branch.counter_target_label] == branch.counter_target_option


def test_the_same_semantic_option_gets_opposite_letters_in_the_two_orders(record, orders, cfg):
    o1, o2 = orders
    assert render_branches(record, o1, cfg, BASE, "opt_1")[0].initial_label == "A"
    assert render_branches(record, o2, cfg, BASE, "opt_1")[0].initial_label == "B"


# --- branch isolation -------------------------------------------------------


def test_four_branches_one_per_condition(record, orders, cfg):
    branches = render_branches(record, orders[0], cfg, BASE, "opt_1")
    assert len(branches) == 4
    assert tuple(b.condition for b in branches) == CORE_CONDITIONS


def test_all_four_branches_share_an_identical_pre_counterargument_transcript(record, orders, cfg):
    for order in orders:
        for initial in ("opt_1", "opt_2"):
            branches = render_branches(record, order, cfg, BASE, initial)
            assert branches_share_prefix(branches)
            assert len({b.shared_prefix_hash for b in branches}) == 1
            first_two = {b.transcript.turns[:2] for b in branches}
            assert len(first_two) == 1


def test_the_four_conditions_never_appear_in_one_conversation(record, orders, cfg):
    """Each branch is an independent conversation carrying exactly one cell."""
    branches = render_branches(record, orders[0], cfg, BASE, "opt_1")
    block = record.counterarguments["opt_2"]
    for branch in branches:
        text = "\n".join(t.content for t in branch.transcript.turns)
        present = [c for c in CORE_CONDITIONS if block.cells[c].body in text]
        assert present == [branch.condition], f"{branch.condition} leaked {present}"


def test_every_branch_has_exactly_three_turns_in_the_frozen_order(record, orders, cfg):
    for branch in render_branches(record, orders[0], cfg, BASE, "opt_2"):
        roles = [t.role for t in branch.transcript.turns]
        assert roles == ["user", "assistant", "user"]
        assert branch.transcript.turns[1].content in ("A", "B")


def test_branches_differ_only_in_their_final_turn(record, orders, cfg):
    branches = render_branches(record, orders[0], cfg, BASE, "opt_1")
    assert len({b.transcript_hash for b in branches}) == 4      # all distinct
    assert len({b.transcript.turns[2] for b in branches}) == 4  # ... only by turn 3


def test_the_initial_prompt_has_no_counterargument(record, orders, cfg):
    initial = render_initial(record, orders[0], cfg, BASE)
    assert len(initial.transcript.turns) == 1
    assert initial.transcript.turns[0].role == "user"
    for option in ("opt_1", "opt_2"):
        for condition in CORE_CONDITIONS:
            body = record.counterarguments[option].cells[condition].body
            assert body not in initial.transcript.turns[0].content
    assert record.counterargument_opening not in initial.transcript.turns[0].content


def test_the_branch_prefix_matches_the_initial_prompt_turn(record, orders, cfg):
    initial = render_initial(record, orders[0], cfg, BASE)
    branch = render_branches(record, orders[0], cfg, BASE, "opt_1")[0]
    assert branch.transcript.turns[0] == initial.transcript.turns[0]


# --- content -----------------------------------------------------------------


def test_the_question_is_frozen_and_identical_in_both_answer_slots(record, orders, cfg):
    question = f"{cfg.parsed.prompts.question_text} {cfg.parsed.prompts.instruction_text}"
    branch = render_branches(record, orders[0], cfg, BASE, "opt_1")[0]
    assert branch.transcript.turns[0].content.endswith(question)
    assert branch.transcript.turns[2].content.endswith(question)


def test_the_shared_opening_is_prepended_to_the_counterargument(record, orders, cfg):
    for branch in render_branches(record, orders[0], cfg, BASE, "opt_1"):
        assert branch.transcript.turns[2].content.startswith(record.counterargument_opening)


def test_branch_records_the_marker_metadata_of_its_group(record, orders, cfg):
    block = record.counterarguments["opt_2"]
    for branch in render_branches(record, orders[0], cfg, BASE, "opt_1"):
        assert branch.marker_family == block.marker_family
        assert branch.marker_string == block.marker_string
        assert branch.marker_realization_id == block.marker_realization_id


# --- hashes and determinism --------------------------------------------------


def test_rendering_is_deterministic(record, orders, cfg):
    a = render_branches(record, orders[0], cfg, BASE, "opt_1")
    b = render_branches(record, orders[0], cfg, BASE, "opt_1")
    assert [x.transcript for x in a] == [x.transcript for x in b]
    assert [x.prompt_hash for x in a] == [x.prompt_hash for x in b]
    assert [x.transcript_hash for x in a] == [x.transcript_hash for x in b]


def test_hashes_distinguish_everything_that_should_differ(record, orders, cfg):
    o1, o2 = orders
    first = render_branches(record, o1, cfg, BASE, "opt_1")
    assert len({b.prompt_hash for b in first}) == 4                 # by condition
    assert first[0].prompt_hash != render_branches(record, o2, cfg, BASE, "opt_1")[0].prompt_hash
    assert first[0].prompt_hash != render_branches(record, o1, cfg, BASE, "opt_2")[0].prompt_hash


def test_a_changed_record_changes_the_record_and_prompt_hashes(record, orders, cfg):
    edited = record.model_copy(update={"scenario_text": record.scenario_text + " One more."})
    a = render_branches(record, orders[0], cfg, BASE, "opt_1")[0]
    b = render_branches(edited, orders[0], cfg, BASE, "opt_1")[0]
    assert a.record_hash != b.record_hash
    assert a.prompt_hash != b.prompt_hash


def test_every_artefact_carries_its_provenance(record, orders, cfg):
    for artefact in (render_initial(record, orders[0], cfg, BASE),
                     *render_branches(record, orders[0], cfg, BASE, "opt_1")):
        assert artefact.config_version == "v4"
        assert artefact.config_content_hash == cfg.content_hash
        assert len(artefact.record_hash) == 64
        assert len(artefact.prompt_hash) == 64


def test_transcript_hash_ignores_nothing_and_invents_nothing(cfg):
    a = Transcript(turns=(), answer_cue="Answer:")
    b = Transcript(turns=(), answer_cue="Answer :")
    assert a.transcript_hash != b.transcript_hash
    assert a.transcript_hash == Transcript(turns=(), answer_cue="Answer:").transcript_hash


# --- materialization ---------------------------------------------------------


def test_the_plain_scaffold_materializes_deterministically(record, orders, cfg):
    branch = render_branches(record, orders[0], cfg, BASE, "opt_1")[0]
    text = materialize(branch.transcript, cfg, BASE, TEST_CONTINUATION)
    assert text == materialize(branch.transcript, cfg, BASE, TEST_CONTINUATION)
    assert text.startswith("The following is a conversation about a policy decision.")
    assert text.endswith("Assistant: Answer:")          # the answer slot
    assert text.count("User:") == 2 and text.count("Assistant:") == 2


def test_materialization_ends_exactly_at_the_answer_slot(record, orders, cfg):
    initial = render_initial(record, orders[0], cfg, BASE)
    text = materialize(initial.transcript, cfg, BASE)
    assert text.endswith("Assistant: Answer:")
    assert not text.endswith(" ")       # the continuation is not guessed


def test_the_initial_prompt_needs_no_answer_continuation(record, orders, cfg):
    materialize(render_initial(record, orders[0], cfg, BASE).transcript, cfg, BASE)


def test_an_unresolved_answer_continuation_refuses_to_guess(record, orders, cfg):
    """' A' versus 'A' is a tokenizer question, settled at the model stage."""
    branch = render_branches(record, orders[0], cfg, BASE, "opt_1")[0]
    with pytest.raises(AnswerContinuationUnresolved, match="tokenizer question"):
        materialize(branch.transcript, cfg, BASE)


def test_the_instruct_template_defers_to_its_tokenizer(record, orders, cfg):
    branch = render_branches(record, orders[0], cfg, INSTRUCT, "opt_1")[0]
    assert branch.transcript.turns          # the transcript itself is available
    with pytest.raises(MaterializationDeferred, match="model-compatibility stage"):
        materialize(branch.transcript, cfg, INSTRUCT, TEST_CONTINUATION)


def test_the_scaffold_is_the_one_frozen_in_the_config(cfg):
    from reasonstyle.hashing import sha256_of
    template = cfg.parsed.prompts.templates[BASE]
    assert template.materialization == "plain_scaffold"
    assert sha256_of(template.template_text) == template.template_sha256


# --- guards ------------------------------------------------------------------


def test_the_renderer_imports_no_model_or_tokenizer():
    source = (ROOT / "src" / "reasonstyle" / "rendering.py").read_text()
    for forbidden in ("torch", "transformers", "tokenizer", "AutoModel", "HookedTransformer"):
        assert f"import {forbidden}" not in source
    assert not {m for m in sys.modules if m.startswith(("torch", "transformers"))}


def test_an_earlier_config_cannot_render(record, orders):
    older = load_config(ROOT / "configs" / "experiment_v3.yaml")
    with pytest.raises(RenderingError, match="v4 or later"):
        render_initial(record, orders[0], older, BASE)


def test_an_unknown_template_is_refused(record, orders, cfg):
    with pytest.raises(RenderingError, match="unknown template"):
        render_initial(record, orders[0], cfg, "no_such_template")


def test_an_option_outside_the_scenario_is_refused(record, orders, cfg):
    with pytest.raises(RenderingError, match="not a semantic option"):
        render_branches(record, orders[0], cfg, BASE, "opt_3")   # type: ignore[arg-type]
