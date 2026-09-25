"""Exact sibling separation for the blinded reliability packets.

Separation is the absolute distance ``|i - j|`` between the packet positions of
two items sharing a ``(scenario_id, supported_option)``; a requirement of ``d``
is met when every such distance is at least ``d`` — ``d`` positions apart,
meaning at least ``d - 1`` intervening items (20 apart = 19 intervening). ``solve_separation`` is a
complete backtracking search: it returns an ordering or proves none exists, and
its status is cross-checked here against brute-force enumeration.

The real-corpus tests read the committed full corpus read-only and build the
export in memory; nothing is written and no run evidence is read.
"""

from __future__ import annotations

import importlib.util
import json
import random
import re
from itertools import combinations, permutations
from pathlib import Path

import pytest

from reasonstyle.corpus.review import (
    PROVED_INFEASIBLE,
    SATISFIED,
    UNDETERMINED,
    min_sibling_gap,
    order_with_separation,
    solve_separation,
)

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "configs/frozen/v2_full.yaml"
CORPUS = ROOT / "data/full/corpus_full_v2.jsonl"
PIN = ROOT / "data/full/reliability_sample_full_v2.json"


def _all_pair_distances(order, key):
    """Every sibling pair, not just neighbours."""
    return [abs(i - j) for i, j in combinations(range(len(order)), 2)
            if key(order[i]) == key(order[j])]


def _items(sizes):
    return [(f"g{g}", m) for g, k in enumerate(sizes) for m in range(k)]


def _brute_force_feasible(sizes, d):
    labels = [g for g, k in enumerate(sizes) for _ in range(k)]
    return any(all(abs(i - j) >= d for i, j in combinations(range(len(p)), 2) if p[i] == p[j])
               for p in set(permutations(labels)))


# --- the distance check itself ------------------------------------------------------


def test_min_sibling_gap_equals_the_minimum_over_every_sibling_pair():
    rng = random.Random(3)
    for _ in range(200):
        order = [rng.randrange(5) for _ in range(rng.randrange(2, 12))]
        distances = _all_pair_distances(order, lambda x: x)
        assert min_sibling_gap(order, lambda x: x) == (min(distances) if distances else None)


# --- exactness ----------------------------------------------------------------------


def test_a_known_feasible_example_is_solved_and_every_pair_verified():
    sizes = [4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1]          # 21 items
    items = _items(sizes)
    order, result = order_with_separation(items, lambda t: t[0], 5, random.Random(0),
                                          attempts=0)      # force the exact path
    assert result.satisfied and result.status == SATISFIED
    assert result.method == "exact_backtracking"
    assert sorted(order) == sorted(items)
    assert min(_all_pair_distances(order, lambda t: t[0])) == result.achieved >= 5


def test_a_known_infeasible_example_is_proved_by_exhaustive_search():
    # Three pairs in six positions: distance 4 fits only (0,4), (0,5), (1,5),
    # which cannot hold three disjoint pairs. No counting bound catches it.
    status, order, detail = solve_separation([2, 2, 2], 4, random.Random(0))
    assert status == PROVED_INFEASIBLE and order is None
    assert "exhaustive" in detail
    assert not _brute_force_feasible([2, 2, 2], 4)
    assert solve_separation([2, 2, 2], 3, random.Random(0))[0] == SATISFIED


def test_a_span_bound_proves_infeasibility_immediately():
    status, _, detail = solve_separation([3, 1], 3, random.Random(0))
    assert status == PROVED_INFEASIBLE and "span" in detail


@pytest.mark.parametrize("seed", range(60))
def test_solver_status_matches_brute_force_on_small_instances(seed):
    rng = random.Random(seed)
    sizes = [rng.randint(1, 3) for _ in range(rng.randint(2, 4))]
    while sum(sizes) > 8:
        sizes.pop()
    d = rng.randint(1, sum(sizes))
    status, order, _ = solve_separation(sizes, d, random.Random(seed))
    assert status in (SATISFIED, PROVED_INFEASIBLE)
    assert (status == SATISFIED) == _brute_force_feasible(sizes, d)
    if order is not None:
        assert all(abs(i - j) >= d for i, j in combinations(range(len(order)), 2)
                   if order[i] == order[j])


# --- determinism and fail-closed behaviour --------------------------------------------


def test_the_same_seed_gives_the_same_order_and_different_seeds_differ():
    items = _items([4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1])
    first = order_with_separation(items, lambda t: t[0], 6, random.Random(7), attempts=0)[0]
    again = order_with_separation(items, lambda t: t[0], 6, random.Random(7), attempts=0)[0]
    other = order_with_separation(items, lambda t: t[0], 6, random.Random(8), attempts=0)[0]
    assert first == again and first != other
    assert min_sibling_gap(other, lambda t: t[0]) >= 6


def test_a_proved_infeasible_requirement_never_claims_compliance():
    items = _items([2, 2, 2])
    order, result = order_with_separation(items, lambda t: t[0], 4, random.Random(0))
    assert sorted(order) == sorted(items)                  # nothing dropped
    assert not result.satisfied and result.status == PROVED_INFEASIBLE
    assert result.achieved < 4
    assert "PROVED INFEASIBLE" in result.note() and "not relaxed" in result.note()
    assert " met " not in result.note()


def test_an_exhausted_search_limit_is_undetermined_not_infeasible():
    items = _items([2] * 30 + [1] * 40)
    order, result = order_with_separation(items, lambda t: t[0], 50, random.Random(0),
                                          attempts=0, node_limit=5)
    assert sorted(order) == sorted(items)
    assert not result.satisfied and result.status == UNDETERMINED
    assert "NOT established" in result.note() and "INFEASIBLE" not in result.note()


# --- the real, fixed reliability sample -----------------------------------------------


@pytest.fixture(scope="module")
def real_export():
    from reasonstyle.config import load_config
    from reasonstyle.corpus import load_corpus, segmenter_from_config, with_measurements
    from reasonstyle.corpus.review import build_review_export
    spec = importlib.util.spec_from_file_location("export_for_review_sep",
                                                  ROOT / "scripts/export_for_review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = load_config(FULL)
    segmenter = segmenter_from_config(cfg)
    records = load_corpus(CORPUS)
    report = module._review_report(records, cfg, segmenter, CORPUS, "full")
    pin = json.loads(PIN.read_text(encoding="utf-8"))
    return build_review_export(with_measurements(records, cfg, segmenter), report, cfg,
                               segmenter, source=str(CORPUS), sampling_pin=pin), records


def test_the_fixed_item_sample_is_unchanged(real_export):
    """The 192 sampled items are exactly those the export drew before the exact
    ordering existed, reproduced through the committed pin even though the
    corpus text has since been corrected."""
    from reasonstyle.hashing import sha256_of
    export, _ = real_export
    keys = export.files["blind_key/item_key.jsonl"]
    assert len(keys.splitlines()) == 192
    assert sha256_of(keys) == ITEM_KEY_SHA256
    assert json.loads(PIN.read_text(encoding="utf-8"))["item_key_sha256"] == ITEM_KEY_SHA256
    assert export.manifest["reliability_sample"]["pinned"] is True


def test_both_item_packets_meet_separation_20_and_match_the_key(real_export):
    export, records = real_export
    by_scenario = {r.scenario_id: r for r in records}
    keys = {k["blind_id"]: k for k in map(json.loads,
                                          export.files["blind_key/item_key.jsonl"].splitlines())}
    orders = {}
    for annotator in ("annotator_1", "annotator_2"):
        text = export.files[f"blind/{annotator}/item_packet.md"]
        ids = re.findall(r"^### Item `(i\d{4})`", text, re.M)
        assert sorted(ids) == sorted(keys)                 # same items, each once
        distances = _all_pair_distances(
            ids, lambda b: (keys[b]["scenario_id"], keys[b]["supported_option"]))
        assert len(distances) == 64 and min(distances) >= 20
        for block in re.split(r"^### Item `", text, flags=re.M)[1:]:
            key = keys[block[:5]]
            options = by_scenario[key["scenario_id"]].options
            assert f"**Option P** — {options[key['option_labels']['P']]}" in block
            assert f"**Option Q** — {options[key['option_labels']['Q']]}" in block
        separation = export.manifest["sibling_separation"][f"{annotator}/item"]
        assert separation["satisfied"] and separation["achieved"] == min(distances)
        assert ("Minimum sibling separation 20 met: siblings are at least 20 positions "
                "apart, meaning at least 19 intervening items") in text
        orders[annotator] = ids
    assert orders["annotator_1"] != orders["annotator_2"]  # independent orders


#: sha256 of ``blind_key/item_key.jsonl`` for the committed full corpus, as the
#: export produced it before exact ordering was added (review/full_corpus_v2).
ITEM_KEY_SHA256 = "105c96abdaadd13e993afa63ac5df95a3c5a7cf5caeb1ee0f686dac7e4f32973"


# --- the reliability-sample pin -------------------------------------------------------


@pytest.fixture(scope="module")
def pinned_inputs():
    from reasonstyle.config import load_config
    from reasonstyle.corpus import load_corpus, segmenter_from_config, validate_corpus
    cfg = load_config(FULL)
    segmenter = segmenter_from_config(cfg)
    records = load_corpus(CORPUS)
    return cfg, segmenter, records, validate_corpus, json.loads(PIN.read_text(encoding="utf-8"))


def _build(cfg, segmenter, records, validate_corpus, pin):
    from reasonstyle.corpus.review import build_review_export
    report = validate_corpus(records, cfg, segmenter, corpus_scope="full")
    return build_review_export(records, report, cfg, segmenter, source=str(CORPUS),
                               sampling_pin=pin)


def test_a_pinned_sample_survives_a_text_only_correction(pinned_inputs):
    """Changing a cell's text changes the corpus hash; with the pin, the
    sample, blind ids, labels and packet orders are all unchanged."""
    cfg, segmenter, records, validate_corpus, pin = pinned_inputs
    before = _build(cfg, segmenter, records, validate_corpus, pin)
    edited = list(records)
    record = edited[0]
    block = record.counterarguments["opt_1"]
    cells = dict(block.cells)
    cells["RP"] = cells["RP"].model_copy(update={"body": cells["RP"].body + " "})
    edited[0] = record.model_copy(update={"counterarguments": {
        **record.counterarguments, "opt_1": block.model_copy(update={"cells": cells})}})
    after = _build(cfg, segmenter, edited, validate_corpus, pin)
    assert before.manifest["corpus_content_hash"] != after.manifest["corpus_content_hash"]
    for name in ("item", "pair", "scenario"):
        assert after.files[f"blind_key/{name}_key.jsonl"] == \
            before.files[f"blind_key/{name}_key.jsonl"]
    for annotator in ("annotator_1", "annotator_2"):
        ids = [re.findall(r"^### Item `(i\d{4})`", export.files[
            f"blind/{annotator}/item_packet.md"], re.M) for export in (before, after)]
        assert ids[0] == ids[1]


@pytest.mark.parametrize("field", ["item_key_sha256", "pair_key_sha256", "scenario_key_sha256",
                                   "config_content_hash"])
def test_a_pin_that_does_not_reproduce_refuses(pinned_inputs, field):
    cfg, segmenter, records, validate_corpus, pin = pinned_inputs
    with pytest.raises(ValueError):
        _build(cfg, segmenter, records, validate_corpus, {**pin, field: "0" * 64})


def test_the_cli_refuses_a_bad_pin_and_writes_nothing(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("export_for_review_pin",
                                                  ROOT / "scripts/export_for_review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bad = tmp_path / "pin.json"
    bad.write_text(json.dumps({**json.loads(PIN.read_text()), "item_key_sha256": "0" * 64}))
    out = tmp_path / "review"
    rc = module.main(["--config", str(FULL), "--corpus", str(CORPUS), "--scope", "full",
                      "--out", str(out), "--reliability-sample", str(bad)])
    assert rc == 1 and "refusing; nothing was written" in capsys.readouterr().err
    assert not out.exists()
