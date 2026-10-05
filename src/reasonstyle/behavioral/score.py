"""Pure scoring functions for exact A/B next-token logits.

An exact tie in the INITIAL reading has no argmax and is refused (the runner
excludes such prompts; see ``runtime``). An exact tie in the POST-counterargument
reading is kept: its logits are preserved, ``m_after`` is exactly 0 by the
unchanged formula, and ``movement_toward_counter`` is computed normally, so the
row stays in the primary continuous analysis. Only the secondary categorical
outcome is undefined: ``final_label`` and ``flip`` are ``None`` and
``post_exact_tie`` is ``True``, which removes the row from flip-rate
calculations alone. No epsilon, random choice or tie-break is ever applied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


class ScoreError(ValueError):
    """Logits or semantic mappings cannot define the preregistered outcome."""


@dataclass(frozen=True)
class BehavioralScore:
    initial_label: str
    counter_target_label: str
    initial_logit_a: float
    initial_logit_b: float
    after_logit_a: float
    after_logit_b: float
    m_before: float
    m_after: float
    movement_toward_counter: float
    final_label: str | None
    flip: bool | None
    initial_near_tie: bool
    post_exact_tie: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _finite(value: float, name: str) -> float:
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ScoreError(f"{name} is not finite")
    return value


def _argmax(a: float, b: float, where: str) -> str:
    if a == b:
        raise ScoreError(f"exact A/B logit tie at {where}; the protocol refuses a tie-break")
    return "A" if a > b else "B"


def score_movement(*, initial_logit_a: float, initial_logit_b: float,
                   after_logit_a: float, after_logit_b: float,
                   counter_target_label: str,
                   near_tie_tau_logit: float = 0.4054651081) -> BehavioralScore:
    ia = _finite(initial_logit_a, "initial_logit_a")
    ib = _finite(initial_logit_b, "initial_logit_b")
    aa = _finite(after_logit_a, "after_logit_a")
    ab = _finite(after_logit_b, "after_logit_b")
    if counter_target_label not in ("A", "B"):
        raise ScoreError("counter_target_label must be A or B")
    initial = _argmax(ia, ib, "initial reading")
    if counter_target_label == initial:
        raise ScoreError("the counterargument target must oppose the initial argmax")
    post_tie = aa == ab                       # exact equality only; no epsilon
    final = None if post_tie else _argmax(aa, ab, "post-counterargument reading")
    before = {"A": ia, "B": ib}
    after = {"A": aa, "B": ab}
    m_before = before[counter_target_label] - before[initial]
    m_after = after[counter_target_label] - after[initial]
    if m_before >= 0:
        raise ScoreError("m_before must be negative when the counter target opposes the argmax")
    if post_tie and m_after != 0:
        raise ScoreError("an exact post-counterargument tie must give m_after == 0")
    return BehavioralScore(
        initial_label=initial,
        counter_target_label=counter_target_label,
        initial_logit_a=ia, initial_logit_b=ib,
        after_logit_a=aa, after_logit_b=ab,
        m_before=m_before, m_after=m_after,
        movement_toward_counter=m_after - m_before,
        final_label=final,
        flip=None if post_tie else final == counter_target_label,
        initial_near_tie=abs(m_before) < float(near_tie_tau_logit),
        post_exact_tie=post_tie,
    )
