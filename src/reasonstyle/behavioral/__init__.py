"""Behavioural planning and exact A/B logit scoring for the v3 corpus."""

from .plan import BehavioralPlan, BehavioralSpec, build_plan, load_spec
from .runtime import initial_argmax, select_runtime_candidates
from .score import BehavioralScore, ScoreError, score_movement

__all__ = [
    "BehavioralPlan", "BehavioralScore", "BehavioralSpec", "ScoreError",
    "build_plan", "initial_argmax", "load_spec", "score_movement",
    "select_runtime_candidates",
]
