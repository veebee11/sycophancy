"""Pure runtime selection helpers for the v3 behavioural experiment."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .plan import BehavioralPlan, BehavioralPlanError


def initial_argmax(logit_a: float, logit_b: float) -> str:
    if float(logit_a) == float(logit_b):
        raise BehavioralPlanError("exact A/B tie in the initial reading")
    return "A" if logit_a > logit_b else "B"


def select_runtime_candidates(plan: BehavioralPlan,
                              initial_logits: dict[str, dict[str, float]],
                              *, initial_ids: set[str] | None = None,
                              ) -> list[dict[str, Any]]:
    """Select counterarguments that oppose each model's initial A/B argmax."""
    chosen_ids = initial_ids or {row["initial_id"] for row in plan.initials}
    missing = chosen_ids - set(initial_logits)
    if missing:
        raise BehavioralPlanError(f"missing initial logits for {len(missing)} prompt(s)")
    labels = {
        initial_id: initial_argmax(initial_logits[initial_id]["A"],
                                   initial_logits[initial_id]["B"])
        for initial_id in chosen_ids
    }
    selected = [row for row in plan.candidates
                if row["initial_id"] in chosen_ids and
                row["required_initial_label"] == labels[row["initial_id"]]]
    counts = Counter(row["initial_id"] for row in selected)
    if set(counts) != chosen_ids or set(counts.values()) != {18}:
        raise BehavioralPlanError("runtime selection must yield 18 branches per initial")
    return selected
