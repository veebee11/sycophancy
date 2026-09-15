"""The reliability sampler: exact size, proportional strata, marginal balance.

Balance is tested against an independent oracle rather than against the
sampler's own dynamic programme. The oracle enumerates every admissible choice
of boundary strata and uses the fact that, for fixed quotas, the reachable
first-value counts form one contiguous interval.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

from reasonstyle.corpus.review import stratified_sample

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("RS", "RP", "NS", "NP")
SEEDS = range(100)


# --- helpers ------------------------------------------------------------------

def make_units(spec):
    """``spec``: {stratum: (n_opt_1, n_opt_2)} -> sampler units."""
    units = []
    for key, (n1, n2) in spec.items():
        for i in range(n1):
            units.append(((key, "opt_1", i), key, ("opt_1",)))
        for i in range(n2):
            units.append(((key, "opt_2", i), key, ("opt_2",)))
    return units


def target_of(units, fraction):
    return max(1, round(len(units) * fraction))


def oracle(units, fraction):
    """Smallest reachable |opt_1 - opt_2| under largest-remainder quotas, and
    the admissible (floor, remainder, boundary) description used to check them."""
    strata = defaultdict(lambda: [0, 0])
    for _, key, (opt,) in units:
        strata[key][0 if opt == "opt_1" else 1] += 1
    target = target_of(units, fraction)
    exact = Fraction(str(fraction))
    floors = {k: math.floor((n1 + n2) * exact) for k, (n1, n2) in strata.items()}
    rems = {k: (n1 + n2) * exact - floors[k] for k, (n1, n2) in strata.items()}
    extra = target - sum(floors.values())
    eligible = [k for k, (n1, n2) in strata.items() if floors[k] < n1 + n2]
    fixed, boundary, picks = set(), [], 0
    for value in sorted({rems[k] for k in eligible}, reverse=True):
        if extra == 0:
            break
        group = [k for k in eligible if rems[k] == value]
        if len(group) <= extra:
            fixed |= set(group)
            extra -= len(group)
        else:
            boundary, picks, extra = group, extra, 0
    best = None
    for combo in itertools.combinations(boundary, picks):
        lo = hi = 0
        for k, (n1, n2) in strata.items():
            q = floors[k] + (k in fixed) + (k in combo)
            lo += max(0, q - n2)
            hi += min(q, n1)
        a = min(max(Fraction(target, 2), lo), hi)          # nearest reachable point
        cands = {math.floor(a), math.ceil(a)} & set(range(lo, hi + 1))
        value = min(abs(2 * c - target) for c in cands)
        best = value if best is None else min(best, value)
    return best, floors, rems, fixed, set(boundary), picks


def counts(chosen):
    return Counter(unit[1] for unit in chosen)


def check_invariants(units, fraction, seed):
    result = stratified_sample(units, fraction, random.Random(seed))
    target = target_of(units, fraction)
    best, floors, rems, fixed, boundary, picks = oracle(units, fraction)

    # exact size, no duplicates, every unit genuine
    assert len(result.chosen) == target
    assert len(set(result.chosen)) == target
    assert set(result.chosen) <= {u for u, _, _ in units}

    # stratum quotas are exactly largest-remainder admissible
    stratum_of = {u: key for u, key, _ in units}
    per_stratum = Counter(stratum_of[u] for u in result.chosen)
    extras = set()
    for key in floors:
        got = per_stratum.get(key, 0)
        assert got == result.quotas[key]
        assert got in (floors[key], floors[key] + 1), key
        if got == floors[key] + 1:
            extras.add(key)
    assert fixed <= extras
    assert extras - fixed <= boundary
    assert len(extras - fixed) == picks

    # balance: optimal, and reported truthfully
    c = counts(result.chosen)
    imbalance = abs(c["opt_1"] - c["opt_2"])
    assert imbalance == best
    assert result.balance.imbalance == imbalance
    assert result.balance.best_under_stratification == best
    assert result.balance.ideal == target % 2
    assert result.balance.satisfied == (best == target % 2)
    return result


# --- the promised properties ---------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_fixture_shape_is_perfectly_balanced_for_every_seed(seed):
    """The shape that exposed the defect: 16 single-unit strata, 8 per option."""
    spec = {(f"g{g}", c): ((1, 0) if g % 2 == 0 else (0, 1))
            for g in range(4) for c in CONDITIONS}
    units = make_units(spec)
    even = check_invariants(units, 0.5, seed)          # 8 items
    assert counts(even.chosen) == {"opt_1": 4, "opt_2": 4}
    odd = check_invariants(units, 0.2, seed)           # 3 items
    assert abs(counts(odd.chosen)["opt_1"] - counts(odd.chosen)["opt_2"]) == 1


def _pilot_units():
    alloc = yaml.safe_load((ROOT / "data/pilot/marker_allocation.yaml").read_text())["groups"]
    items = [((g["scenario_id"], g["supported_option"], c),
              (g["domain"], c, g["marker_family"]), (g["supported_option"],))
             for g in alloc for c in CONDITIONS]
    pairs = [((g["scenario_id"], g["supported_option"], p),
              (g["domain"], p, g["marker_family"]), (g["supported_option"],))
             for g in alloc for p in ("RS_RP", "NS_NP")]
    return items, pairs


@pytest.mark.parametrize("seed", SEEDS)
def test_pilot_shape_is_balanced_and_covers_every_stratum(seed):
    """Pilot allocation: 48 groups -> 192 items in 36 strata, 96 pairs in 18."""
    items, pairs = _pilot_units()
    item_result = check_invariants(items, 0.2, seed)
    assert len(item_result.chosen) == 38
    assert len({key for u, key, _ in items if u in set(item_result.chosen)}) == 36
    assert counts(item_result.chosen) == {"opt_1": 19, "opt_2": 19}

    pair_result = check_invariants(pairs, 0.2, seed)
    assert len(pair_result.chosen) == 19                # odd
    c = counts(pair_result.chosen)
    assert abs(c["opt_1"] - c["opt_2"]) == 1


def _random_spec(rng):
    spec = {}
    for s in range(rng.randint(1, 12)):
        n = rng.randint(1, 7)
        n1 = rng.choice([0, n, rng.randint(0, n)])      # pure strata are common
        spec[(f"s{s}",)] = (n1, n - n1)
    return spec


@pytest.mark.parametrize("case", range(200))
def test_heterogeneous_strata_reach_the_oracle_optimum(case):
    rng = random.Random(case)
    units = make_units(_random_spec(rng))
    fraction = rng.choice([0.2, 0.25, 1 / 3, 0.5, 0.6, 1.0])
    check_invariants(units, fraction, seed=case)


def test_perfect_balance_is_reached_whenever_it_is_feasible():
    hits = 0
    for case in range(400):
        rng = random.Random(10_000 + case)
        units = make_units(_random_spec(rng))
        fraction = rng.choice([0.2, 0.25, 0.5])
        best = oracle(units, fraction)[0]
        target = target_of(units, fraction)
        if best == target % 2:
            hits += 1
            result = stratified_sample(units, fraction, random.Random(case))
            c = counts(result.chosen)
            assert abs(c["opt_1"] - c["opt_2"]) == target % 2
            assert result.balance.satisfied
    assert hits > 100                                    # the case is well exercised


def test_infeasible_because_of_available_units_is_reported():
    units = make_units({("a",): (10, 0), ("b",): (0, 2)})
    result = stratified_sample(units, 1.0, random.Random(0))   # all 12 units
    assert counts(result.chosen) == {"opt_1": 10, "opt_2": 2}
    b = result.balance
    assert (b.imbalance, b.ideal, b.best_from_units, b.best_under_stratification) == (8, 0, 8, 8)
    assert not b.satisfied
    assert "INFEASIBLE" in b.note() and "available units" in b.note()


def test_infeasible_because_of_stratum_quotas_is_reported():
    """Units would allow 3/3, but the quotas fix 2+2 from opt_1-only strata."""
    units = make_units({("a",): (4, 0), ("b",): (4, 0), ("c",): (0, 4)})
    result = stratified_sample(units, 0.5, random.Random(0))
    b = result.balance
    assert counts(result.chosen) == {"opt_1": 4, "opt_2": 2}
    assert (b.imbalance, b.ideal, b.best_from_units, b.best_under_stratification) == (2, 0, 0, 2)
    assert "stratum quotas" in b.note()


def test_boundary_ties_are_used_for_balance_not_left_to_chance():
    """Six tied singleton strata, 3 per option, sample 2: must be 1 + 1."""
    units = make_units({(f"s{i}",): ((1, 0) if i < 3 else (0, 1)) for i in range(6)})
    for seed in SEEDS:
        result = stratified_sample(units, 1 / 3, random.Random(seed))
        assert counts(result.chosen) == {"opt_1": 1, "opt_2": 1}


def test_a_strictly_larger_remainder_is_never_skipped_for_balance():
    """Balance may only choose among strata TIED at the boundary remainder.

    fraction 0.25, target 2. Stratum 'a' (2 x opt_2) has remainder 0.5 and must
    take the extra unit, although taking tied stratum 'c' (opt_1, remainder
    0.25) instead would give a perfect split. The imbalance is reported, with
    the quotas named as the cause.
    """
    units = make_units({("a",): (0, 2), ("b",): (0, 1), ("c",): (1, 0), ("d",): (0, 4)})
    for seed in SEEDS:
        result = stratified_sample(units, 0.25, random.Random(seed))
        per = Counter(u[0] for u in result.chosen)
        assert per == {("a",): 1, ("d",): 1}
        b = result.balance
        assert (b.imbalance, b.best_under_stratification, b.best_from_units) == (2, 2, 0)
        assert "stratum quotas" in b.note()


def test_same_units_and_seed_give_the_same_sample():
    items, _ = _pilot_units()
    first = stratified_sample(items, 0.2, random.Random(42))
    second = stratified_sample(list(items), 0.2, random.Random(42))
    assert first.chosen == second.chosen
    assert first.quotas == second.quotas
    assert first.balance == second.balance


def test_different_seeds_explore_different_optimal_samples():
    items, _ = _pilot_units()
    samples = {tuple(sorted(stratified_sample(items, 0.2, random.Random(s)).chosen))
               for s in range(20)}
    assert len(samples) > 1


def test_units_without_a_balance_key_keep_the_proportional_sample():
    units = [((f"u{i}",), (f"d{i % 3}",)) for i in range(12)]
    result = stratified_sample(units, 0.5, random.Random(3))
    assert result.balance is None
    assert len(result.chosen) == 6
    assert all(v == 2 for v in result.quotas.values())


def test_more_than_two_balance_values_is_refused():
    units = [((i,), ("s",), (v,)) for i, v in enumerate("xyz")]
    with pytest.raises(ValueError, match="two values"):
        stratified_sample(units, 1.0, random.Random(0))


def test_the_real_fixture_packet_is_balanced(records, cfg):
    """The committed fixture, through the exporter's own unit builder and seed."""
    from reasonstyle.corpus.review import _rng, item_sampling_units
    from reasonstyle.corpus.store import corpus_content_hash

    sub = cfg.raw["annotation"]["reliability_subsample"]
    units = item_sampling_units(records, sub["stratify_by"], sub["balance_marginally"])
    result = stratified_sample(units, sub["fraction"],
                               _rng(cfg, corpus_content_hash(records), "item-sample"))
    assert len(result.chosen) == max(1, round(len(units) * sub["fraction"]))
    assert result.balance.satisfied
    assert result.balance.imbalance == len(result.chosen) % 2
