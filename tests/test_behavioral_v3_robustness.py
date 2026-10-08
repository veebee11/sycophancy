"""Offline checks of the Base prompt-format robustness variants (no model)."""

from __future__ import annotations

import pytest

from reasonstyle.behavioral.compatibility import CompatibilityError, materialize_base
from reasonstyle.behavioral.plan import build_plan, load_spec
from reasonstyle.behavioral.robustness import (
    build_variant_plan,
    compatibility_gate,
    load_variants,
    resolve_label_tokens,
)
from reasonstyle.hashing import sha256_of

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ROOT / "configs/robustness/base_prompt_variants_v1.yaml"


@pytest.fixture(scope="module")
def world():
    spec = load_spec(ROOT / "configs/behavioral_v3_llama31_8b.draft.yaml")
    plan = build_plan(spec)
    raw, variants = load_variants(VARIANTS)
    plans = {name: build_variant_plan(spec, plan, v) for name, v in variants.items()}
    return spec, plan, raw, variants, plans


def _reference_prompt(spec, row):
    return materialize_base(row["transcript"], answer_cue=spec.raw["prompt"]["answer_cue"],
                            template=spec.raw["prompt"]["base_plain_dialogue_v1"])


def test_reference_rendering_is_byte_identical_to_the_original_prompts(world):
    spec, plan, _, _, plans = world
    ref = plans["reference"]
    for row, vrow in zip(plan.initials + plan.candidates, ref.initials + ref.candidates,
                         strict=True):
        assert vrow["prompt"] == _reference_prompt(spec, row)
        assert vrow["prompt_sha256"] == sha256_of(vrow["prompt"])


def test_every_variant_keeps_ids_mappings_and_slots(world):
    _, plan, _, _, plans = world
    for vplan in plans.values():
        assert [r["initial_id"] for r in vplan.initials] == [r["initial_id"] for r in plan.initials]
        assert [r["branch_id"] for r in vplan.candidates] == [r["branch_id"] for r in plan.candidates]
        for a, b in zip(plan.candidates, vplan.candidates, strict=True):
            assert a["label_to_option"] == b["label_to_option"]
            assert a["required_initial_label"] == b["required_initial_label"]
            assert a["counter_target_label"] == b["counter_target_label"]
    digests = {name: p.prompt_digest for name, p in plans.items()}
    assert len(set(digests.values())) == 4


def test_ab_bfirst_only_swaps_the_option_lines(world):
    spec, plan, _, _, plans = world
    for row, vrow in zip(plan.initials, plans["ab_bfirst"].initials, strict=True):
        ref = _reference_prompt(spec, row)
        scenario_opts = row["label_to_option"]
        a_line = next(l for l in ref.splitlines() if l.startswith("A. "))
        b_line = next(l for l in ref.splitlines() if l.startswith("B. "))
        assert vrow["prompt"] == ref.replace(f"{a_line}\n{b_line}", f"{b_line}\n{a_line}")
        assert vrow["prompt"].index("B. ") < vrow["prompt"].index("A. ")
        assert scenario_opts == vrow["label_to_option"]


def test_numeric_labels_replace_only_the_labels(world):
    spec, plan, _, _, plans = world
    for row, vrow in zip(plan.candidates[:200], plans["numeric_12"].candidates[:200]):
        prompt = vrow["prompt"]
        assert "Answer with 1 or 2 only." in prompt and "Answer with A or B only." not in prompt
        assert "\n1. " in prompt and "\n2. " in prompt
        answer = {"A": "1", "B": "2"}[row["required_initial_label"]]
        assert f"\n\nAssistant: {answer}\n\nUser: " in prompt
        assert prompt.index("\n1. ") < prompt.index("\n2. ")


def test_minimal_template_has_no_dialogue_and_extends_the_initial(world):
    spec, plan, _, _, plans = world
    vplan = plans["minimal_ab"]
    initial = {r["initial_id"]: r["prompt"] for r in vplan.initials}
    for row in vplan.initials:
        assert row["prompt"].startswith("The following is a policy decision.\n\n")
        assert row["prompt"].endswith("Which option should be chosen? Answer with A or B only.\nAnswer:")
        assert "User:" not in row["prompt"] and "Assistant:" not in row["prompt"]
    for row in vplan.candidates[:300]:
        head = initial[row["initial_id"]] + f" {row['required_initial_label']}\n\nCounterargument: "
        assert row["prompt"].startswith(head)
        assert row["prompt"].endswith("\n\nWhich option should be chosen? Answer with A or B only.\nAnswer:")


class CharTokenizer:
    """One token per character, except that the given strings are single tokens."""

    def __init__(self, units=(" A", " B", " 1", " 2")):
        self.units = sorted(units, key=len, reverse=True)

    def encode(self, text, *, add_special_tokens):
        out, i = [], 0
        while i < len(text):
            for unit in self.units:
                if text.startswith(unit, i) and i + len(unit) == len(text):
                    out.append(1000 + self.units.index(unit))
                    i += len(unit)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out


def test_gate_pins_one_token_map_and_refuses_multi_token_labels(world):
    _, _, _, _, plans = world
    vplan = plans["numeric_12"]
    small = type(vplan)(variant=vplan.variant, initials=vplan.initials[:3],
                        candidates=vplan.candidates[:5])
    report = compatibility_gate(CharTokenizer(), small)
    assert report["answer_continuation"] == {"A": " 1", "B": " 2"}
    assert report["prompts_checked"] == 8
    with pytest.raises(CompatibilityError, match="single-token"):
        compatibility_gate(CharTokenizer(units=(" A", " B")), small)
    with pytest.raises(CompatibilityError, match="single-token"):
        resolve_label_tokens(CharTokenizer(), "x", {"A": "10", "B": "2"})
