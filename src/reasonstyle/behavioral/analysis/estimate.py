"""Decision-weighted estimands and the stratified decision-cluster bootstrap.

Every estimate is built in two stages: a value per valid block, averaged with
equal weight within ``decision_id``; then an equal-weight mean over decisions.
A bootstrap replicate redraws decisions with replacement within each domain
(``numpy.random.Generator(PCG64(seed))``) and recomputes the same mean with each
decision weighted by its draw count, so a drawn decision carries every block of
every run it owns. Composite estimates (differences, family means, slopes) are
computed replicate by replicate from their components.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .design import Block


def ordered_dot(columns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """sum_j columns[j] * weights[j], accumulated in index order with elementwise IEEE
    operations only. Unlike a BLAS matrix product, the summation order is fixed, so the
    result is bit-identical on every platform with the same NumPy."""
    acc = np.zeros(columns.shape[1] if columns.ndim == 2 else 1)
    for j in range(columns.shape[0]):
        acc += columns[j] * weights[j]
    return acc


class Bootstrap:
    """The fixed draw matrix and its per-decision count matrix."""

    def __init__(self, decisions: Sequence[str], domain_of: dict[str, str], replicates: int,
                 seed: int):
        self.decisions = sorted(decisions)
        self.index = {d: i for i, d in enumerate(self.decisions)}
        self.domains = sorted({domain_of[d] for d in self.decisions})
        rng = np.random.Generator(np.random.PCG64(seed))
        blocks, labels = [], []
        for dom in self.domains:
            ids = sorted(d for d in self.decisions if domain_of[d] == dom)
            draw = rng.integers(0, len(ids), size=(replicates, len(ids)))
            blocks.append(np.array([self.index[d] for d in ids])[draw])
            labels += [f"{dom}_{k + 1:02d}" for k in range(len(ids))]
        self.draws = np.concatenate(blocks, axis=1)          # (B, D) decision indices
        self.position_labels = labels
        n, d = self.draws.shape
        flat = (self.draws + d * np.arange(n)[:, None]).ravel()
        self.counts = np.bincount(flat, minlength=n * d).reshape(n, d).astype(np.float64)
        self.counts_by_decision = np.ascontiguousarray(self.counts.T)   # (D, B)
        self.replicates = replicates
        self.seed = seed

    def decision_ids(self) -> list[list[str]]:
        return [[self.decisions[i] for i in row] for row in self.draws]


@dataclass
class Est:
    point: float
    reps: np.ndarray
    denominators: dict[str, int] = field(default_factory=dict)

    def __sub__(self, other: "Est") -> "Est":
        return Est(self.point - other.point, self.reps - other.reps,
                   _merge(self.denominators, other.denominators))


def _merge(*denoms: dict[str, int]) -> dict[str, int]:
    """Union denominators: blocks and rows add; decision and initial sets are unioned
    through the hidden id sets carried alongside."""
    out: dict[str, Any] = {}
    sets: dict[str, set] = {}
    for d in denoms:
        for k, v in d.items():
            if k.startswith("_"):
                sets.setdefault(k, set()).update(v)
            else:
                out[k] = out.get(k, 0) + v
    for k, v in sets.items():
        out[k] = v
        out[f"n{k}"] = len(v)          # "_decisions" -> "n_decisions": distinct, not summed
    return out


def mean_of(ests: list[Est]) -> Est:
    point = float(np.mean([e.point for e in ests]))
    reps = np.mean(np.vstack([e.reps for e in ests]), axis=0)
    return Est(point, reps, _merge(*(e.denominators for e in ests)))


ValueFn = Callable[[Block], "tuple[float, int, int] | None"]


def decision_estimate(boot: Bootstrap, blocks: Iterable[Block], fn: ValueFn) -> Est:
    """Equal weight per valid block within a decision, then per decision."""
    d = len(boot.decisions)
    sums, counts = np.zeros(d), np.zeros(d)
    mov = flip = 0
    decisions, initials = set(), set()
    for block in blocks:
        out = fn(block)
        if out is None:
            continue
        value, n_mov, n_flip = out
        i = boot.index[block.decision_id]
        sums[i] += value
        counts[i] += 1
        mov += n_mov
        flip += n_flip
        decisions.add(block.decision_id)
        initials.add((block.run, block.initial_id))
    defined = counts > 0
    values = np.where(defined, sums / np.where(defined, counts, 1), 0.0)
    point = float(values[defined].mean()) if defined.any() else math.nan
    idx = np.flatnonzero(defined)
    cols = boot.counts_by_decision[idx]
    num = ordered_dot(cols, values[idx])
    total = ordered_dot(cols, np.ones(idx.size))
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = np.where(total > 0, num / np.where(total > 0, total, 1), np.nan)
    denom = {"n_blocks": int(counts.sum()), "n_movement_rows": mov, "n_flip_rows": flip,
             "_decisions": decisions, "n_decisions": len(decisions),
             "_initials": initials, "n_initials": len(initials)}
    return Est(point, reps, denom)


def wls_slope(boot: Bootstrap, blocks: list[Block], fn: ValueFn) -> tuple[Est, Est]:
    """Slope and weighted mean of a block value on |m_before|, each block weighted
    1/(valid blocks in its decision); replicates multiply by draw counts."""
    d = len(boot.decisions)
    pts: dict[int, list[tuple[float, float]]] = {}
    for block in blocks:
        out = fn(block)
        if out is not None:
            pts.setdefault(boot.index[block.decision_id], []).append(
                (block.abs_m_before, out[0]))
    s = np.zeros((5, d))
    for i, items in pts.items():
        w = 1.0 / len(items)
        x = np.array([a for a, _ in items])
        y = np.array([b for _, b in items])
        s[:, i] = [w * len(items), w * x.sum(), w * y.sum(), w * (x * x).sum(), w * (x * y).sum()]

    def fit(cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s0, sx, sy, sxx, sxy = (ordered_dot(cols, s[k]) for k in range(5))
        with np.errstate(invalid="ignore", divide="ignore"):
            slope = (sxy - sx * sy / s0) / (sxx - sx * sx / s0)
            mean = sy / s0
        return slope, mean
    p_slope, p_mean = fit(np.ones((d, 1)))
    r_slope, r_mean = fit(boot.counts_by_decision)
    n = {"n_blocks": sum(len(v) for v in pts.values()), "n_decisions": len(pts),
         "n_movement_rows": 0, "n_flip_rows": 0, "n_initials": len(
             {(b.run, b.initial_id) for b in blocks})}
    return Est(float(p_slope[0]), r_slope, dict(n)), Est(float(p_mean[0]), r_mean, dict(n))


def summarise(est: Est, level: float) -> dict[str, Any]:
    reps = est.reps[~np.isnan(est.reps)]
    alpha = 1.0 - level
    if reps.size == 0 or math.isnan(est.point):
        lo = hi = p = math.nan
    else:
        lo, hi = (float(v) for v in np.quantile(reps, [alpha / 2, 1 - alpha / 2],
                                                 method="linear"))
        b = reps.size
        p = min(1.0, 2.0 * min((np.count_nonzero(reps <= 0) + 1) / (b + 1),
                               (np.count_nonzero(reps >= 0) + 1) / (b + 1)))
    return {"estimate": est.point, "ci_low": lo, "ci_high": hi, "p_boot": float(p),
            "replicates_defined": int(reps.size),
            **{k: v for k, v in est.denominators.items() if not k.startswith("_")}}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1, ties ordered by key)."""
    items = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    m, running, out = len(items), 0.0, {}
    for i, (key, p) in enumerate(items):
        if math.isnan(p):
            raise ValueError(f"cannot Holm-adjust an undefined p-value ({key})")
        running = max(running, min(1.0, (m - i) * p))
        out[key] = running
    return out
