"""Pure runtime selection helpers for the v3 behavioural experiment.

The configured rule is ``scoring.exact_tie: refuse``: an exact A/B logit tie in
an initial reading has no argmax, so no tie-break, epsilon, random choice,
precision change or remapping is applied. Such an initial prompt is scored and
retained, recorded as excluded (``exact_initial_logit_tie``), given no initial
choice, and selects no counterargument branch. Every other initial prompt
selects exactly the 18 branches opposing its argmax.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .plan import BehavioralPlan, BehavioralPlanError

TIE_REASON = "exact_initial_logit_tie"
BRANCHES_PER_INITIAL = 18


def initial_argmax(logit_a: float, logit_b: float) -> str:
    if float(logit_a) == float(logit_b):
        raise BehavioralPlanError("exact A/B tie in the initial reading")
    return "A" if logit_a > logit_b else "B"


def is_exact_tie(logit_a: float, logit_b: float) -> bool:
    """Exact equality of the two recorded logits, exactly as ``initial_argmax`` tests it."""
    return float(logit_a) == float(logit_b)


def expected_post_counterargument(n_initials: int, n_ties: int) -> int:
    return BRANCHES_PER_INITIAL * (n_initials - n_ties)


@dataclass(frozen=True)
class RuntimeSelection:
    """The selected branches, each non-tied initial's argmax label, and the
    recorded exclusions (one per exactly tied initial)."""

    selected: list[dict[str, Any]]
    labels: dict[str, str]
    exclusions: list[dict[str, Any]] = field(default_factory=list)


def runtime_selection(plan: BehavioralPlan, initial_logits: dict[str, dict[str, float]],
                      *, initial_ids: set[str] | None = None) -> RuntimeSelection:
    """Select counterarguments opposing each initial argmax; exclude exact ties."""
    chosen_ids = initial_ids or {row["initial_id"] for row in plan.initials}
    missing = chosen_ids - set(initial_logits)
    if missing:
        raise BehavioralPlanError(f"missing initial logits for {len(missing)} prompt(s)")
    by_id = {row["initial_id"]: row for row in plan.initials}
    unknown = chosen_ids - set(by_id)
    if unknown:
        raise BehavioralPlanError(f"{len(unknown)} initial id(s) are not in the plan")
    labels: dict[str, str] = {}
    exclusions: list[dict[str, Any]] = []
    for initial_id in sorted(chosen_ids):
        logits = initial_logits[initial_id]
        if is_exact_tie(logits["A"], logits["B"]):
            row = by_id[initial_id]
            exclusions.append({
                "initial_id": initial_id,
                **{key: row[key] for key in ("decision_id", "domain", "scenario_id",
                                              "variant_id", "order_id")},
                "logit_a": float(logits["A"]), "logit_b": float(logits["B"]),
                "label_to_option": dict(row["label_to_option"]),
                "exclusion_reason": TIE_REASON,
                "initial_choice": None,
                "branches_selected": 0,
            })
            continue
        labels[initial_id] = initial_argmax(logits["A"], logits["B"])
    selected = [row for row in plan.candidates
                if row["initial_id"] in labels and
                row["required_initial_label"] == labels[row["initial_id"]]]
    counts = Counter(row["initial_id"] for row in selected)
    excluded = {row["initial_id"] for row in exclusions}
    if (set(counts) != set(labels) or set(counts.values()) - {BRANCHES_PER_INITIAL}
            or excluded & set(counts)
            or len(selected) != expected_post_counterargument(len(chosen_ids), len(excluded))):
        raise BehavioralPlanError(
            "runtime selection must yield 18 branches per non-tied initial and none for a tie")
    return RuntimeSelection(selected=selected, labels=labels, exclusions=exclusions)


def select_runtime_candidates(plan: BehavioralPlan,
                              initial_logits: dict[str, dict[str, float]],
                              *, initial_ids: set[str] | None = None,
                              ) -> list[dict[str, Any]]:
    """The selected branches only; exact ties select none (see ``runtime_selection``)."""
    return runtime_selection(plan, initial_logits, initial_ids=initial_ids).selected
