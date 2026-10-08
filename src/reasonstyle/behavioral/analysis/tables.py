"""Every analysis table of ``behavioral_v3_detailed_v1``.

Rows are plain dicts; ``write`` serialises and sorts them. Each estimate row
carries its denominators: decisions, initial readings, blocks, raw movement
rows and defined flip rows used.
"""

from __future__ import annotations

import math
import statistics as st
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Callable

import numpy as np

from .design import (
    PLAIN,
    STYLED,
    Block,
    build_blocks,
    common_semantic_choice_initials,
    common_valid_initials,
)
from .estimate import Bootstrap, Est, decision_estimate, holm, mean_of, summarise, wls_slope
from .validate import (
    CONDITIONS,
    EXPECTED_CONTRASTS,
    FAMILIES,
    REFERENCE_MODELS,
    ValidatedEvidence,
    post_tie_breakdown,
    tie_counts,
)

CONTRASTS = tuple(EXPECTED_CONTRASTS)
PLANNED = ("NS_minus_NP", "RS_minus_RP", "RS_minus_NS", "RP_minus_NP")
MARKER_CONTRASTS = ("NS_minus_NP", "RS_minus_RP", "RS_minus_NS", "interaction")
STYLE_PAIR = ("NS_minus_NP", "RS_minus_RP")
LABEL = {"NS_minus_NP": "NS − NP", "RS_minus_RP": "RS − RP", "RS_minus_NS": "RS − NS",
         "RP_minus_NP": "RP − NP", "interaction": "(RS − RP) − (NS − NP)"}
ROWS_PER_CELL = {"RS": 2, "NS": 2, "RP": 1, "NP": 1}


# -- block value functions --------------------------------------------------------
def movement_contrast(name: str):
    coef = EXPECTED_CONTRASTS[name]

    def fn(b: Block):
        used = [c for c, k in coef.items() if k]
        return (sum(coef[c] * b.movement[c] for c in used),
                sum(ROWS_PER_CELL[c] for c in used), 0)
    return fn


def flip_contrast(name: str):
    coef = EXPECTED_CONTRASTS[name]

    def fn(b: Block):
        used = [c for c, k in coef.items() if k]
        cells = {c: b.flip_cell(c) for c in used}
        if any(v is None for v in cells.values()):
            return None
        rows = sum(1 for c in used for f in b.flips[c].values() if f is not None)
        return (sum(coef[c] * cells[c] for c in used), sum(ROWS_PER_CELL[c] for c in used), rows)
    return fn


def movement_cell(cond: str):
    return lambda b: (b.movement[cond], ROWS_PER_CELL[cond], 0)


def flip_cell(cond: str):
    def fn(b: Block):
        v = b.flip_cell(cond)
        if v is None:
            return None
        return (v, ROWS_PER_CELL[cond], sum(1 for f in b.flips[cond].values() if f is not None))
    return fn


def marker_movement(name: str, marker: str):
    def fn(b: Block):
        if marker not in b.markers.values():
            return None
        rs, ns = b.marker_movement["RS"][marker], b.marker_movement["NS"][marker]
        rp, np_ = b.movement["RP"], b.movement["NP"]
        value = {"NS_minus_NP": ns - np_, "RS_minus_RP": rs - rp, "RS_minus_NS": rs - ns,
                 "interaction": (rs - rp) - (ns - np_)}[name]   # movement is never undefined
        return value, {"NS_minus_NP": 2, "RS_minus_RP": 2, "RS_minus_NS": 2,
                       "interaction": 4}[name], 0
    return fn


def marker_flip(name: str, marker: str):
    def fn(b: Block):
        if marker not in b.markers.values():
            return None
        cells = {"RS": b.flip_cell("RS", marker), "NS": b.flip_cell("NS", marker),
                 "RP": b.flip_cell("RP"), "NP": b.flip_cell("NP")}
        used = {"NS_minus_NP": ("NS", "NP"), "RS_minus_RP": ("RS", "RP"),
                "RS_minus_NS": ("RS", "NS"), "interaction": ("RS", "RP", "NS", "NP")}[name]
        if any(cells[c] is None for c in used):
            return None
        coef = {"NS_minus_NP": {"NS": 1, "NP": -1}, "RS_minus_RP": {"RS": 1, "RP": -1},
                "RS_minus_NS": {"RS": 1, "NS": -1},
                "interaction": {"RS": 1, "RP": -1, "NS": -1, "NP": 1}}[name]
        return sum(k * cells[c] for c, k in coef.items()), len(used), len(used)
    return fn


OUTCOMES = {"movement": (movement_contrast, movement_cell),
            "flip": (flip_contrast, flip_cell)}


class Tables:
    def __init__(self, evidence: ValidatedEvidence):
        self.ev = evidence
        spec = evidence.spec.raw
        boot_cfg = spec["inference"]["bootstrap"]
        self.level = float(boot_cfg["confidence_level"])
        self.blocks = build_blocks(evidence)
        domain_of = {}
        for row in evidence.stimuli.values():
            domain_of[row["decision_id"]] = row["domain"]
        self.boot = Bootstrap(sorted(domain_of), domain_of, boot_cfg["replicates"],
                              boot_cfg["seed"])
        self.tau = 0.4054651081
        self.runs = list(evidence.runs)
        self.robust_runs = [r for r in self.runs if r not in REFERENCE_MODELS]
        self.common = common_valid_initials(evidence)
        self.semantic_runs = ["base"] + self.robust_runs
        self.common_semantic = common_semantic_choice_initials(evidence, self.semantic_runs)
        self.saved: dict[str, Est] = {}
        self.summary: list[dict[str, Any]] = []
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self._by_run = defaultdict(list)
        for b in self.blocks:
            self._by_run[b.run].append(b)
        self.quartile_cuts = {m: self._quartiles(m) for m in REFERENCE_MODELS}

    # -- helpers ---------------------------------------------------------------------
    def sel(self, run: str, keep: Callable[[Block], bool] = lambda b: True) -> list[Block]:
        return [b for b in self._by_run[run] if keep(b)]

    def est(self, blocks, fn) -> Est:
        return decision_estimate(self.boot, blocks, fn)

    def row(self, table: str, est: Est, *, save: bool = False, **keys: Any) -> dict[str, Any]:
        sid = "|".join([table] + [f"{k}={keys[k]}" for k in keys])
        out = {**keys, **summarise(est, self.level), "statistic_id": sid}
        if "quantity" in keys and keys["quantity"] in LABEL:
            out["quantity_label"] = LABEL[keys["quantity"]]
        self.summary.append(out)
        if save:
            self.saved[sid] = est
        return out

    def core_set(self, table: str, run: str, blocks: list[Block], outcomes=("movement", "flip"),
                 cells: bool = True, save: bool = False, **keys) -> dict[tuple, Est]:
        ests = {}
        for outcome in outcomes:
            cfn, cellfn = OUTCOMES[outcome]
            for name in CONTRASTS:
                e = self.est(blocks, cfn(name))
                ests[(outcome, name)] = e
                self.tables[table].append(self.row(table, e, save=save, run=run, **keys,
                                                   outcome=outcome, kind="contrast",
                                                   quantity=name))
            if cells:
                for cond in CONDITIONS:
                    e = self.est(blocks, cellfn(cond))
                    ests[(outcome, cond)] = e
                    self.tables[table].append(self.row(table, e, save=save, run=run, **keys,
                                                       outcome=outcome, kind="cell",
                                                       quantity=cond))
        return ests

    def _quartiles(self, model: str) -> list[float]:
        mb = {r["initial_id"]: abs(r["m_before"]) for r in self.ev.runs[model].scores}
        values = np.array(sorted(mb.values()))
        return [float(v) for v in np.quantile(values, [0.25, 0.5, 0.75], method="linear")]

    def quartile(self, model: str, value: float) -> str:
        q1, q2, q3 = self.quartile_cuts[model]
        return "Q1" if value <= q1 else "Q2" if value <= q2 else "Q3" if value <= q3 else "Q4"

    # -- build -----------------------------------------------------------------------
    def build(self) -> dict[str, list[dict[str, Any]]]:
        for name in TABLE_NAMES:
            self.tables[name] = []
        self.evidence_tables()
        self.descriptives()
        self.core()
        self.strata()
        self.markers()
        self.margin()
        self.model_comparison()
        self.robustness()
        self.multiplicity()
        return self.tables

    def evidence_tables(self) -> None:
        t = self.tables
        for c in self.ev.checks:
            t["evidence_validation"].append({k: (v if isinstance(v, (bool, int, float)) or v is None
                                                 else str(v)) for k, v in c.items()})
        for name, run in self.ev.runs.items():
            counts = tie_counts(run)
            for x in sorted(run.exclusions, key=lambda r: r["initial_id"]):
                t["initial_ties"].append({
                    "run": name, "initial_id": x["initial_id"], "decision_id": x["decision_id"],
                    "scenario_id": x["scenario_id"], "order_id": x["order_id"],
                    "logit_a": x["logit_a"], "logit_b": x["logit_b"],
                    "exclusion_reason": x["exclusion_reason"],
                    "run_initial_readings": counts["initial_readings"],
                    "run_exact_initial_ties": counts["exact_initial_ties"],
                    "run_exact_initial_ties_pct": counts["exact_initial_ties_pct"]})
            breakdown = post_tie_breakdown(run)
            rows = Counter((r["condition"], r["marker_id"], r["opening_id"]) for r in run.scores)
            for (cond, marker, opening), n in sorted(rows.items()):
                k = breakdown.get((cond, marker, opening), 0)
                t["post_ties_by_model_condition_marker_opening"].append({
                    "run": name, "condition": cond, "marker_id": marker, "opening_id": opening,
                    "movement_rows": n, "exact_post_ties": k, "flip_defined_rows": n - k,
                    "exact_post_ties_pct": 100.0 * k / n})
            for order in ("o1", "o2"):
                mine = [r for r in run.initials if r["order_id"] == order]
                labels = Counter(r["initial_label"] or "tie" for r in mine)
                for slot in ("A", "B", "tie"):
                    t["initial_choice_by_model_order_label"].append({
                        "run": name, "order_id": order, "slot": slot,
                        "response_label": run.response_labels.get(slot, "tie"),
                        "displayed_position": (run.display_order.index(slot) + 1
                                               if slot in run.display_order else None),
                        "count": labels.get(slot, 0), "initial_readings": len(mine),
                        "pct": 100.0 * labels.get(slot, 0) / len(mine)})
                options = Counter(r["initial_option"] or "tie" for r in mine)
                for opt in ("opt_1", "opt_2", "tie"):
                    t["initial_semantic_choice_by_model_order"].append({
                        "run": name, "order_id": order, "initial_option": opt,
                        "count": options.get(opt, 0), "initial_readings": len(mine),
                        "pct": 100.0 * options.get(opt, 0) / len(mine)})
            t["flip_denominators"] += self._flip_denominators(name, run)

    def _flip_denominators(self, name, run) -> list[dict[str, Any]]:
        out = []
        for cond in CONDITIONS:
            rows = [r for r in run.scores if r["condition"] == cond]
            post = sum(1 for r in rows if r["post_exact_tie"])
            out.append({"run": name, "level": "condition_rows", "quantity": cond,
                        "movement_rows": len(rows), "exact_post_ties": post,
                        "flip_defined_rows": len(rows) - post, "blocks": None,
                        "blocks_flip_defined": None})
        blocks = self._by_run[name]
        for cname in CONTRASTS:
            fn = flip_contrast(cname)
            ok = sum(1 for b in blocks if fn(b) is not None)
            out.append({"run": name, "level": "block_contrast", "quantity": cname,
                        "movement_rows": None, "exact_post_ties": None,
                        "flip_defined_rows": None, "blocks": len(blocks),
                        "blocks_flip_defined": ok})
        return out

    def descriptives(self) -> None:
        t = self.tables
        for name, run in self.ev.runs.items():
            valid = [r for r in run.initials if not r["excluded"]]
            mb = {r["initial_id"]: r["m_before"] for r in run.scores}
            values = sorted(abs(mb[r["initial_id"]]) for r in valid)
            q = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
            t["initial_margin_descriptives"].append({
                "run": name, "valid_initials": len(values), "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)), "min": values[0], "q25": float(q[0]),
                "median": float(q[1]), "q75": float(q[2]), "max": values[-1],
                "near_ties_below_tau": sum(1 for v in values if v < self.tau),
                "tau": self.tau})
            blocks = self._by_run[name]
            for cond in CONDITIONS:
                rows = [r for r in run.scores if r["condition"] == cond]
                mv = sorted(r["movement_toward_counter"] for r in rows)
                qq = np.quantile(mv, [0.25, 0.5, 0.75], method="linear")
                e = self.est(blocks, movement_cell(cond))
                t["condition_descriptives_movement"].append({
                    **self.row("condition_descriptives_movement", e, save=True, run=name,
                               quantity=cond),
                    "raw_rows": len(rows), "raw_mean": float(np.mean(mv)),
                    "raw_sd": float(np.std(mv, ddof=1)), "raw_min": mv[0],
                    "raw_q25": float(qq[0]), "raw_median": float(qq[1]),
                    "raw_q75": float(qq[2]), "raw_max": mv[-1],
                    "raw_positive": sum(1 for v in mv if v > 0)})
                defined = [r for r in rows if r["flip"] is not None]
                flips = sum(1 for r in defined if r["flip"])
                e = self.est(blocks, flip_cell(cond))
                t["condition_descriptives_flip"].append({
                    **self.row("condition_descriptives_flip", e, save=True, run=name,
                               quantity=cond),
                    "raw_rows": len(rows), "raw_flip_defined": len(defined), "raw_flips": flips,
                    "raw_flip_rate": flips / len(defined) if defined else None,
                    "raw_exact_post_ties": len(rows) - len(defined)})

    def core(self) -> None:
        for model in REFERENCE_MODELS:
            blocks = self._by_run[model]
            ests = self.core_set("contrasts_by_model", model, blocks)
            for outcome in ("movement", "flip"):
                table = f"core_contrasts_{outcome}"
                rows = []
                for name in CONTRASTS:
                    rows.append(self.row(table, ests[(outcome, name)], save=True, run=model,
                                         outcome=outcome, quantity=name))
                adjusted = holm({r["quantity"]: r["p_boot"] for r in rows
                                 if r["quantity"] in PLANNED})
                for r in rows:
                    r["holm_family"] = (f"planned|{outcome}|{model}" if r["quantity"] in PLANNED
                                        else "interaction_reported_separately")
                    r["p_holm"] = adjusted.get(r["quantity"])
                self.tables[table] += rows
            for outcome in ("movement", "flip"):
                cfn = OUTCOMES[outcome][0]
                full = {n: self.est(blocks, cfn(n)) for n in CONTRASTS}
                sens = self.sel(model, lambda b: b.abs_m_before >= self.tau)
                for name in CONTRASTS:
                    e = self.est(sens, cfn(name))
                    r = self.row("near_tie_sensitivity", e, save=True, run=model,
                                 outcome=outcome, quantity=name)
                    r["full_sample_estimate"] = full[name].point
                    r["analysis_role"] = "sensitivity (|m_before| >= tau); full sample primary"
                    self.tables["near_tie_sensitivity"].append(r)

    def strata(self) -> None:
        for model in REFERENCE_MODELS:
            for attr, table, het in (("domain", "contrasts_by_domain", None),
                                     ("order_id", "contrasts_by_option_order",
                                      "order_heterogeneity"),
                                     ("opening_id", "contrasts_by_opening",
                                      "opening_heterogeneity")):
                levels = sorted({getattr(b, attr) for b in self._by_run[model]})
                ests = {}
                for lv in levels:
                    ests[lv] = self.core_set(table, model,
                                             self.sel(model, lambda b, a=attr, v=lv:
                                                      getattr(b, a) == v),
                                             **{attr: lv})
                if het:
                    for a, b in combinations(levels, 2):
                        for outcome in ("movement", "flip"):
                            for name in CONTRASTS:
                                e = ests[a][(outcome, name)] - ests[b][(outcome, name)]
                                r = self.row(het, e, run=model, outcome=outcome,
                                             comparison=f"{a}_minus_{b}", quantity=name)
                                r["estimate_first"] = ests[a][(outcome, name)].point
                                r["estimate_second"] = ests[b][(outcome, name)].point
                                if attr == "order_id":
                                    s1 = summarise(ests[a][(outcome, name)], self.level)
                                    s2 = summarise(ests[b][(outcome, name)], self.level)
                                    side = [(s["ci_low"] > 0) - (s["ci_high"] < 0) for s in (s1, s2)]
                                    r["same_sign"] = (s1["estimate"] > 0) == (s2["estimate"] > 0)
                                    r["both_intervals_exclude_zero_same_side"] = \
                                        side[0] == side[1] != 0
                                self.tables[het].append(r)

    def marker_ests(self, run: str, blocks: list[Block], outcome: str = "movement"
                    ) -> dict[tuple[str, str], Est]:
        fnf = marker_movement if outcome == "movement" else marker_flip
        return {(m, n): self.est(blocks, fnf(n, m)) for m in sorted(self.ev.markers)
                for n in MARKER_CONTRASTS}

    def markers(self) -> None:
        holm_rows = []
        for model in REFERENCE_MODELS:
            blocks = self._by_run[model]
            for outcome in ("movement", "flip"):
                ests = self.marker_ests(model, blocks, outcome)
                for (m, n), e in ests.items():
                    info = self.ev.markers[m]
                    r = self.row("contrasts_by_individual_marker", e, run=model,
                                 outcome=outcome, marker_id=m, quantity=n)
                    r.update(marker_family=info["family"], marker_subtype=info["subtype"],
                             marker_string=info["string"])
                    self.tables["contrasts_by_individual_marker"].append(r)
                    if outcome == "movement" and n in STYLE_PAIR:
                        holm_rows.append(r)
                fam = {}
                for family in FAMILIES:
                    members = sorted(m for m, i in self.ev.markers.items() if i["family"] == family)
                    for n in MARKER_CONTRASTS:
                        e = mean_of([ests[(m, n)] for m in members])
                        fam[(family, n)] = e
                        r = self.row("contrasts_by_marker_family", e, run=model, outcome=outcome,
                                     marker_family=family, quantity=n)
                        r["weighting"] = "unweighted mean of six marker estimates"
                        self.tables["contrasts_by_marker_family"].append(r)
                for n in MARKER_CONTRASTS:
                    e = fam[(FAMILIES[0], n)] - fam[(FAMILIES[1], n)]
                    r = self.row("contrasts_by_marker_family", e, run=model, outcome=outcome,
                                 marker_family=f"{FAMILIES[0]}_minus_{FAMILIES[1]}", quantity=n)
                    r["weighting"] = "difference of family means"
                    self.tables["contrasts_by_marker_family"].append(r)
            for opening in sorted(self.ev.openings):
                sub = self.sel(model, lambda b, o=opening: b.opening_id == o)
                for m in sorted(self.ev.markers):
                    for n in STYLE_PAIR:
                        e = self.est(sub, marker_movement(n, m))
                        r = self.row("marker_by_opening", e, run=model, marker_id=m,
                                     opening_id=opening, quantity=n)
                        r["marker_family"] = self.ev.markers[m]["family"]
                        self.tables["marker_by_opening"].append(r)
        adjusted = holm({r["statistic_id"]: r["p_boot"] for r in holm_rows})
        for r in self.tables["contrasts_by_individual_marker"]:
            r["p_holm_48"] = adjusted.get(r["statistic_id"])
        self._marker_holm = (holm_rows, adjusted)

    def margin(self) -> None:
        for model in REFERENCE_MODELS:
            cuts = self.quartile_cuts[model]
            for qname in ("Q1", "Q2", "Q3", "Q4"):
                sub = self.sel(model, lambda b, q=qname: self.quartile(model, b.abs_m_before) == q)
                for name in CONTRASTS:
                    e = self.est(sub, movement_contrast(name))
                    r = self.row("margin_quartiles", e, run=model, quartile=qname, quantity=name)
                    r.update(q25=cuts[0], q50=cuts[1], q75=cuts[2])
                    self.tables["margin_quartiles"].append(r)
            for name in CONTRASTS:
                slope, mean = wls_slope(self.boot, self._by_run[model], movement_contrast(name))
                for term, e in (("slope_per_logit_abs_m_before", slope),
                                ("weighted_mean", mean)):
                    r = self.row("margin_moderation", e, save=True, run=model, term=term,
                                 quantity=name)
                    self.tables["margin_moderation"].append(r)

    def model_comparison(self) -> None:
        ests = {}
        for model in REFERENCE_MODELS:
            sub = self.sel(model, lambda b: b.initial_id in self.common)
            ests[model] = self.core_set("model_common_subset", model, sub, save=True)
        for key in ests["base"]:
            outcome, quantity = key
            e = ests["instruct"][key] - ests["base"][key]
            r = self.row("base_vs_instruct", e, save=True, outcome=outcome,
                         kind="contrast" if quantity in CONTRASTS else "cell",
                         quantity=quantity)
            r.update(base_estimate=ests["base"][key].point,
                     instruct_estimate=ests["instruct"][key].point,
                     difference="instruct_minus_base",
                     interpretation="association between checkpoints and prompt formats")
            self.tables["base_vs_instruct"].append(r)

    def robustness(self) -> None:
        ref = {r["initial_id"]: r for r in self.ev.runs["base"].initials}
        for name in ["base"] + self.robust_runs:
            run = self.ev.runs[name]
            valid = [r for r in run.initials if not r["excluded"]]
            both = [r for r in valid if not ref[r["initial_id"]]["excluded"]]
            agree = sum(1 for r in both if r["initial_option"] == ref[r["initial_id"]]["initial_option"])
            for order in ("o1", "o2", "all"):
                mine = [r for r in valid if order == "all" or r["order_id"] == order]
                total = [r for r in run.initials if order == "all" or r["order_id"] == order]
                first = run.display_order[0]
                self.tables["robustness_initial_diagnostics"].append({
                    "run": name, "status": run.status, "order_id": order,
                    "template": run.template, "display_order": "".join(run.display_order),
                    "response_labels": f"A={run.response_labels['A']},B={run.response_labels['B']}",
                    "initial_readings": len(total), "exact_initial_ties": len(total) - len(mine),
                    "valid_initials": len(mine),
                    "chose_slot_A_token": sum(1 for r in mine if r["initial_label"] == "A"),
                    "chose_slot_B_token": sum(1 for r in mine if r["initial_label"] == "B"),
                    "chose_first_listed": sum(1 for r in mine if r["initial_label"] == first),
                    "chose_second_listed": sum(1 for r in mine if r["initial_label"] != first),
                    "chose_opt_1": sum(1 for r in mine if r["initial_option"] == "opt_1"),
                    "chose_opt_2": sum(1 for r in mine if r["initial_option"] == "opt_2"),
                    "valid_in_reference_too": len(both) if order == "all" else None,
                    "same_semantic_choice_as_reference": agree if order == "all" else None,
                    "agreement_pct": (100.0 * agree / len(both) if both else None)
                    if order == "all" else None})
        for name, refusal in sorted(self.ev.refusals.items()):
            self.tables["robustness_initial_diagnostics"].append({
                "run": name, "status": "refused_at_compatibility_gate", "order_id": "all",
                "template": None, "display_order": None, "response_labels": None,
                "initial_readings": 0, "exact_initial_ties": None, "valid_initials": 0,
                "refusal_reason": refusal["reason"]})
        for name in ["base"] + self.robust_runs:
            blocks = self._by_run[name]
            full = self.core_set("robustness_contrasts", name, blocks, cells=True,
                                 save=name != "base", sample="full_valid", stratum="all")
            orders = {}
            for o in ("o1", "o2"):
                orders[o] = self.core_set("robustness_contrasts", name,
                                          self.sel(name, lambda b, o=o: b.order_id == o),
                                          cells=False, sample="full_valid", stratum=o)
            for outcome in ("movement", "flip"):
                for cn in CONTRASTS:
                    e = orders["o1"][(outcome, cn)] - orders["o2"][(outcome, cn)]
                    self.tables["robustness_contrasts"].append(self.row(
                        "robustness_contrasts", e, run=name, sample="full_valid",
                        stratum="o1_minus_o2", outcome=outcome, kind="order_difference",
                        quantity=cn))
            for opening in sorted(self.ev.openings):
                self.core_set("robustness_contrasts", name,
                              self.sel(name, lambda b, o=opening: b.opening_id == o),
                              outcomes=("movement",), cells=False, sample="full_valid",
                              stratum=opening)
            ests = self.marker_ests(name, blocks)
            for family in FAMILIES:
                members = sorted(m for m, i in self.ev.markers.items() if i["family"] == family)
                for n in STYLE_PAIR:
                    e = mean_of([ests[(m, n)] for m in members])
                    self.tables["robustness_contrasts"].append(self.row(
                        "robustness_contrasts", e, run=name, sample="full_valid",
                        stratum=family, outcome="movement", kind="family_mean", quantity=n))
            for m in sorted(self.ev.markers):
                for n in STYLE_PAIR:
                    self.tables["robustness_contrasts"].append(self.row(
                        "robustness_contrasts", ests[(m, n)], run=name, sample="full_valid",
                        stratum=m, outcome="movement", kind="marker", quantity=n))
        sub_ests = {}
        for name in self.semantic_runs:
            sub = self.sel(name, lambda b: b.initial_id in self.common_semantic)
            sub_ests[name] = self.core_set("robustness_common_semantic_choice_subset", name, sub,
                                           save=True, comparison="estimate")
        for name in self.robust_runs:
            for key, e in sub_ests[name].items():
                outcome, q = key
                d = e - sub_ests["base"][key]
                r = self.row("robustness_common_semantic_choice_subset", d, save=True, run=name,
                             comparison="variant_minus_reference", outcome=outcome,
                             kind="contrast" if q in CONTRASTS else "cell", quantity=q)
                self.tables["robustness_common_semantic_choice_subset"].append(r)
        for r in self.tables["robustness_common_semantic_choice_subset"]:
            r["subset_initials"] = len(self.common_semantic)
            r["subset_runs"] = ",".join(self.semantic_runs)

    def multiplicity(self) -> None:
        t = self.tables["multiplicity_adjustments"]
        for outcome in ("movement", "flip"):
            for r in self.tables[f"core_contrasts_{outcome}"]:
                if r["p_holm"] is not None:
                    t.append({"family": r["holm_family"], "family_size": 4,
                              "statistic_id": r["statistic_id"], "run": r["run"],
                              "outcome": outcome, "quantity": r["quantity"],
                              "estimate": r["estimate"], "p_boot": r["p_boot"],
                              "p_holm": r["p_holm"]})
        rows, adjusted = self._marker_holm
        for r in rows:
            t.append({"family": "individual_markers_48", "family_size": len(rows),
                      "statistic_id": r["statistic_id"], "run": r["run"], "outcome": "movement",
                      "quantity": f"{r['marker_id']}:{r['quantity']}", "estimate": r["estimate"],
                      "p_boot": r["p_boot"], "p_holm": adjusted[r["statistic_id"]]})


TABLE_NAMES = (
    "evidence_validation", "initial_ties", "post_ties_by_model_condition_marker_opening",
    "initial_choice_by_model_order_label", "initial_semantic_choice_by_model_order",
    "initial_margin_descriptives", "condition_descriptives_movement",
    "condition_descriptives_flip", "core_contrasts_movement", "core_contrasts_flip",
    "contrasts_by_model", "contrasts_by_domain", "contrasts_by_option_order",
    "order_heterogeneity", "contrasts_by_opening", "opening_heterogeneity",
    "contrasts_by_marker_family", "contrasts_by_individual_marker", "marker_by_opening",
    "margin_quartiles", "margin_moderation", "near_tie_sensitivity", "model_common_subset",
    "base_vs_instruct", "multiplicity_adjustments", "flip_denominators",
    "robustness_initial_diagnostics", "robustness_contrasts",
    "robustness_common_semantic_choice_subset")
