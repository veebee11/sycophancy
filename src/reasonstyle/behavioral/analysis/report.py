"""The Markdown report and README of ``behavioral_v3_detailed_v1``.

Every number is read from the computed tables. The wording describes observed
behavioural differences only; it never says a manipulation worked.
"""

from __future__ import annotations

import math
from typing import Any

from .tables import CONTRASTS, LABEL

RUNS = {"base": "Base", "instruct": "Instruct", "ab_bfirst": "Base R1 (B line first)",
        "minimal_ab": "Base R3 (minimal scaffold)", "numeric_12": "Base R2 (labels 1/2)"}


def f(x: Any, d: int = 3, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-0.000", "0.000").replace("+0.000", "0.000") if float(s) == 0 else s


def p(x: Any) -> str:
    if x is None:
        return "—"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def ci(r: dict[str, Any]) -> str:
    return f"{f(r['estimate'], sign=True)} [{f(r['ci_low'], sign=True)}, {f(r['ci_high'], sign=True)}]"


def side(r: dict[str, Any]) -> str:
    if r["ci_low"] is None:
        return "—"
    return "above 0" if r["ci_low"] > 0 else "below 0" if r["ci_high"] < 0 else "includes 0"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


class Report:
    def __init__(self, ev, t: dict[str, list[dict[str, Any]]], review: dict[str, Any],
                 meta: dict[str, Any]):
        self.ev, self.t, self.review, self.meta = ev, t, review, meta
        self.spec = ev.spec.raw

    def rows(self, name: str, **w) -> list[dict[str, Any]]:
        return [r for r in self.t[name] if all(r.get(k) == v for k, v in w.items())]

    def one(self, name: str, **w) -> dict[str, Any]:
        rows = self.rows(name, **w)
        assert len(rows) == 1, (name, w, len(rows))
        return rows[0]

    def contrast_table(self, name: str, runs, holm: bool = False, **w) -> str:
        header = ["run", "contrast", "estimate [95% CI]", "interval", "p (boot)"]
        header += ["Holm p (planned family)"] if holm else []
        header += ["decisions", "initials", "blocks", "movement rows", "flip rows"]
        body = []
        for run in runs:
            for c in CONTRASTS:
                r = self.one(name, run=run, quantity=c, **w)
                line = [RUNS.get(run, run), LABEL[c], ci(r), side(r), p(r["p_boot"])]
                if holm:
                    line.append(p(r.get("p_holm")) if r.get("p_holm") is not None
                                else "separate (unadjusted)")
                line += [str(r["n_decisions"]), str(r["n_initials"]), str(r["n_blocks"]),
                         str(r["n_movement_rows"]), str(r["n_flip_rows"])]
                body.append(line)
        return table(header, body)

    def render(self) -> str:
        s = self.spec
        out: list[str] = []
        w = out.append
        w("# Behavioural analysis of full v3 — Llama-3.1-8B Base and Instruct\n")
        w(f"Analysis `{s['analysis_version']}`; dataset `full_v3_multimarker_openings` "
          "(**draft**). Developmental, exploratory results on a researcher-reviewed draft. No "
          "human manipulation check or inter-annotator reliability study has been run, so "
          "nothing here shows that the reason or style manipulations operate as intended for "
          "human readers. This report describes observed model behaviour only. No mechanistic "
          "analysis has been run.\n")
        self.validation(w)
        self.estimand(w)
        self.core(w)
        self.robustness(w)
        self.exploratory(w)
        self.outstanding(w)
        self.implications(w)
        self.provenance(w)
        return "\n".join(out)

    # -- sections ---------------------------------------------------------------------
    def validation(self, w) -> None:
        ev, s = self.ev, self.spec
        w("## 1. Evidence validation\n")
        w(f"All {len(ev.checks)} recorded checks passed before any estimate was computed. For "
          "every run this means: `run_status`, every file SHA-256 recomputed against "
          "`RUN_METADATA.json` and against this analysis's pinned hashes, the expected counts, "
          "model revision, config hashes and answer tokens, and every row re-scored from its "
          "stored logits with the project's scoring function and matched exactly to the plan "
          "branch its initial argmax selects.\n")
        body = []
        for name, run in ev.runs.items():
            c = run.metadata["counts"]
            body.append([RUNS[name], f"`{run.metadata['run_id']}`", run.status,
                         str(c["initial"]), str(c["excluded_exact_initial_ties"]),
                         str(c["post_counterargument"]), str(c["post_exact_ties"]),
                         str(c["flip_defined"]), f"`{run.file_hashes['behavioral_scores.jsonl'][:16]}…`"])
        w(table(["run", "run ID", "status", "initials", "exact initial ties", "movement rows",
                 "post ties", "defined flips", "scores SHA-256"], body))
        inp = s["inputs"]
        w(f"- Stimuli `{inp['stimuli']['sha256']}`; pinned behavioural config file "
          f"`{inp['behavioral_config']['sha256']}` (content `{inp['behavioral_config']['content_hash']}`).")
        w(f"- Model revisions: Base `{inp['runs']['base']['revision']}`, Instruct "
          f"`{inp['runs']['instruct']['revision']}`; answer tokens ` A`=362, ` B`=426.")
        arc, rarc = inp["evidence_archive"], inp["robustness_archive"]
        w(f"- Reference evidence archived read-only at `{arc['server']}` (server) and "
          f"`{arc['mac']}` (Mac); `SHA256SUMS` hash `{arc['sha256sums_sha256']}`. Robustness "
          f"evidence at `{rarc['server']}` and `{rarc['mac']}`; `SHA256SUMS` hash "
          f"`{rarc['sha256sums_sha256']}`. The pinned config is included in both archives.")
        ref = self.meta.get("reference_check") or {}
        w(f"- Variant-runner equivalence: the reference Base prompts re-rendered by the new "
          f"variant code are byte-identical to the original prompts (tests), and the variant "
          f"runner reproduced all {ref.get('initials_compared', '—')} reference Base initial "
          f"logits bit for bit (`{ref.get('status', '—')}`, "
          f"{len(ref.get('bitwise_mismatches', []) or [])} mismatches).\n")

    def estimand(self, w) -> None:
        d = self.spec["design"]
        b = self.spec["inference"]["bootstrap"]
        w("## 2. Estimand\n")
        w("- **Block:** run × `scenario_id` × `order_id` × `opening_id` for a non-tied initial "
          "prompt: RS(m1), RS(m2), NS(m1), NS(m2), RP, NP (m1, m2 = the group's two allocated "
          "markers, one per family).")
        w("- **Cells:** RS and NS are the unweighted mean of their two marker rows; RP and NP "
          "are the single shared controls, entered once and never duplicated.")
        w("- **Contrasts** are formed within each block: NS − NP, RS − RP, RS − NS, RP − NP and "
          "(RS − RP) − (NS − NP).")
        w("- **Weighting:** valid blocks equally within `decision_id`, then each decision "
          "equally. Openings enter equally because every valid initial has all three.")
        w("- **Outcomes:** primary `movement_toward_counter` = m_after − m_before; secondary "
          "`flip`, defined only where the post reading is not an exact tie.")
        w("- **Ties:** an exact initial tie forms no block (its branches do not exist). Exact "
          "post ties stay in movement with m_after = 0 and leave flip only. No tie-breaking.")
        w(f"- **Inference:** {b['replicates']} replicates, seed {b['seed']}, NumPy PCG64 "
          f"(NumPy {self.meta['software']['numpy']}); whole decisions resampled within domain "
          "(20 climate, 20 energy, 20 technology draws each replicate), carrying both scenario "
          "variants, all openings, both orders, all conditions and every run together; "
          "two-sided percentile 95% intervals; the draw matrix is saved in "
          "`bootstrap/draws.csv.gz`. Bootstrap p = min(1, 2·min(#≤0 + 1, #≥0 + 1)/(B + 1)).")
        w("- **Multiplicity:** Holm over the four planned simple contrasts, separately by "
          "outcome and model; the interaction is reported separately; Holm over 48 "
          "individual-marker style comparisons. Everything else is exploratory and unadjusted.\n")

    def core(self, w) -> None:
        w("## 3. Core developmental contrasts\n")
        w("### Movement (primary)\n")
        w(self.contrast_table("core_contrasts_movement", ["base", "instruct"], holm=True))
        cells = {(r["run"], r["quantity"]): r for r in self.t["condition_descriptives_movement"]}
        w("Cell means (decision-weighted): " + "; ".join(
            f"{RUNS[m]} " + ", ".join(f"{c} {f(cells[(m, c)]['estimate'], 2)}"
                                      for c in ("RS", "RP", "NS", "NP"))
            for m in ("base", "instruct")) + ".\n")
        w("### Flip (secondary)\n")
        w(self.contrast_table("core_contrasts_flip", ["base", "instruct"], holm=True))
        fl = {(r["run"], r["quantity"]): r for r in self.t["condition_descriptives_flip"]}
        rates = "; ".join(f"{RUNS[m]} " + ", ".join(
            f"{c} {f(fl[(m, c)]['estimate'], 3)} ({fl[(m, c)]['raw_flips']}/"
            f"{fl[(m, c)]['raw_flip_defined']} rows)" for c in ("RS", "RP", "NS", "NP"))
            for m in ("base", "instruct"))
        w(f"Flip rates: {rates}. Rates at or near 1 leave flip almost no room to vary; where "
          "every defined row flips, the flip contrasts are exactly zero by construction and "
          "carry no information about the factors.\n")
        w("### Base versus Instruct on the common valid subset\n")
        cs = self.one("base_vs_instruct", outcome="movement", quantity="NS_minus_NP")
        w(f"{len(self.ev.runs['base'].initials) - len(self.ev.runs['base'].exclusions)} Base and "
          f"{len(self.ev.runs['instruct'].initials) - len(self.ev.runs['instruct'].exclusions)} "
          f"Instruct initials are valid; {self.meta['common_valid_initials']} are valid in both "
          f"({cs['n_decisions']} decisions). Differences "
          "below (Instruct − Base) are associations between two checkpoints with different "
          "prompt formats, not effects of instruction tuning.\n")
        body = []
        for c in CONTRASTS:
            r = self.one("base_vs_instruct", outcome="movement", quantity=c)
            body.append([LABEL[c], f(r["base_estimate"], sign=True),
                         f(r["instruct_estimate"], sign=True), ci(r), side(r)])
        w(table(["contrast", "Base", "Instruct", "Instruct − Base [95% CI]", "interval"], body))
        w("### Option-order stability\n")
        body = []
        for m in ("base", "instruct"):
            for c in CONTRASTS:
                r = self.one("order_heterogeneity", run=m, outcome="movement", quantity=c)
                body.append([RUNS[m], LABEL[c], f(r["estimate_first"], sign=True),
                             f(r["estimate_second"], sign=True), ci(r),
                             "yes" if r["same_sign"] else "no",
                             "yes" if r["both_intervals_exclude_zero_same_side"] else "no"])
        w(table(["model", "contrast", "o1", "o2", "o1 − o2 [95% CI]", "same sign",
                 "both intervals exclude 0 on the same side"], body))
        w("No stability criterion is fixed in the repository; the last two columns are "
          "descriptive.\n")
        w("### Initial margin\n")
        md = {r["run"]: r for r in self.t["initial_margin_descriptives"]}
        w("|m_before| over valid initials: " + "; ".join(
            f"{RUNS[m]} median {f(md[m]['median'])} (IQR {f(md[m]['q25'])}–{f(md[m]['q75'])}, "
            f"{md[m]['near_ties_below_tau']} of {md[m]['valid_initials']} below τ = 0.405)"
            for m in ("base", "instruct")) + ". Logits are read in bfloat16, whose spacing near "
          "the observed logit magnitudes is coarse (Base margins are multiples of 1/16), which "
          "is consistent with the number of exact ties.\n")
        body = []
        for m in ("base", "instruct"):
            for c in CONTRASTS:
                sl = self.one("margin_moderation", run=m, term="slope_per_logit_abs_m_before",
                              quantity=c)
                nt = self.one("near_tie_sensitivity", run=m, outcome="movement", quantity=c)
                body.append([RUNS[m], LABEL[c], ci(sl), ci(nt), f(nt["full_sample_estimate"], sign=True)])
        w(table(["model", "contrast", "slope per logit of |m_before| [95% CI]",
                 "excluding near ties (sensitivity) [95% CI]", "full sample"], body))
        w("Quartile-specific estimates are in `tables/margin_quartiles` and figure 11. The full "
          "valid sample stays primary; τ was fixed before these data and was not re-chosen.\n")

    def robustness(self, w) -> None:
        w("## 4. Robustness: Base response labels and prompt format\n")
        w("Predeclared in `configs/robustness/base_prompt_variants_v1.yaml` before any variant "
          "ran. Each variant used the cached pinned Base revision, bfloat16 and deterministic "
          "CUDA on one idle RTX A6000, scored its own initial prompts first, then its own "
          "adaptive branches under the unchanged tie policy.\n")
        for name, ref in sorted(self.ev.refusals.items()):
            diag = ref.get("diagnostic", {})
            w(f"- **{RUNS.get(name, name)}: refused at the offline tokenizer gate** — "
              f"{ref['reason']}. {diag.get('observation', '')} Under the predeclared rule no "
              "replacement label or whitespace convention was tried, and no model forward pass "
              "was made for it.")
        w("")
        body = []
        for r in self.rows("robustness_initial_diagnostics", order_id="all"):
            if not r.get("valid_initials"):
                continue
            body.append([RUNS[r["run"]], r["display_order"], str(r["exact_initial_ties"]),
                         str(r["valid_initials"]),
                         f"{r['chose_slot_A_token']} ({100 * r['chose_slot_A_token'] / r['valid_initials']:.1f}%)",
                         f"{r['chose_first_listed']} ({100 * r['chose_first_listed'] / r['valid_initials']:.1f}%)",
                         f"{r['chose_opt_1']}/{r['chose_opt_2']}",
                         f"{r['same_semantic_choice_as_reference']}/{r['valid_in_reference_too']} "
                         f"({f(r['agreement_pct'], 1)}%)"])
        w(table(["run", "display order", "initial ties", "valid initials", "chose token A",
                 "chose first-listed line", "opt_1/opt_2", "same semantic choice as reference"],
                body))
        ref = self.one("robustness_initial_diagnostics", run="base", order_id="all")
        r1 = self.one("robustness_initial_diagnostics", run="ab_bfirst", order_id="all")
        token_share = r1["chose_slot_A_token"] / r1["valid_initials"]
        first_share = r1["chose_first_listed"] / r1["valid_initials"]
        verdict = ("tracks the response token “A” rather than the first displayed line"
                   if token_share > 0.5 > first_share else
                   "tracks the first displayed line rather than the token" if first_share > 0.5 > token_share
                   else "is not explained by either the token or the position alone")
        w(f"In the reference format Base chose token A on {ref['chose_slot_A_token']} of "
          f"{ref['valid_initials']} valid initials, where A is also the first line. With the B "
          f"line displayed first (R1) it still chose token A on {r1['chose_slot_A_token']} of "
          f"{r1['valid_initials']}, now the second-listed line. In these runs the initial "
          f"preference therefore {verdict}."
          + (" Because `label_to_option` is counterbalanced, the initial semantic choice in Base "
             "mostly follows whichever option is labelled A, so Base's counterarguments mostly "
             "argue for the option labelled B. Base movement is therefore measured against a "
             "label preference, not a content-based initial stance."
             if ref["chose_slot_A_token"] / ref["valid_initials"] > 0.9 else "") + "\n")
        w("### Contrasts on each run's full valid sample\n")
        w(self.contrast_table("robustness_contrasts", ["base", "ab_bfirst", "minimal_ab"],
                              sample="full_valid", stratum="all", outcome="movement",
                              kind="contrast"))
        n = self.t["robustness_common_semantic_choice_subset"][0]["subset_initials"]
        w(f"### Common semantic-choice subset ({n} initials)\n")
        w(f"Initials valid in the reference and every completed variant with the identical "
          f"initial semantic choice, so every run sees the same stimuli; variant − reference "
          "differences isolate the prompt format.\n")
        w(self.contrast_table("robustness_common_semantic_choice_subset",
                              ["base", "ab_bfirst", "minimal_ab"], comparison="estimate",
                              outcome="movement", kind="contrast"))
        body = []
        for run in ("ab_bfirst", "minimal_ab"):
            for c in CONTRASTS:
                r = self.one("robustness_common_semantic_choice_subset", run=run,
                             comparison="variant_minus_reference", outcome="movement", quantity=c)
                body.append([RUNS[run], LABEL[c], ci(r), side(r)])
        w(table(["variant", "contrast", "variant − reference [95% CI]", "interval"], body))
        w("Order differences, openings, marker families and individual markers per variant are "
          "in `tables/robustness_contrasts`; figures 13–14 summarise the diagnostics. These "
          "comparisons are diagnostic; no format is preferred because of the effect it shows.\n")

    def exploratory(self, w) -> None:
        w("## 5. Exploratory marker and opening analyses\n")
        w("Unadjusted except the 48-comparison Holm family for individual markers. No isolated "
          "uncorrected comparison is promoted.\n")
        body = []
        for m in ("base", "instruct"):
            for o in sorted(self.ev.openings):
                for c in ("NS_minus_NP", "RS_minus_RP", "interaction"):
                    r = self.one("contrasts_by_opening", run=m, opening_id=o, outcome="movement",
                                 kind="contrast", quantity=c)
                    body.append([RUNS[m], o, LABEL[c], ci(r), side(r)])
        w("### Openings\n")
        w(table(["model", "opening", "contrast", "estimate [95% CI]", "interval"], body))
        w("Pairwise opening differences: `tables/opening_heterogeneity`.\n")
        w("### Marker families (unweighted mean of six markers)\n")
        body = []
        for m in ("base", "instruct"):
            for fam in ("conclusion_result", "inference_basis",
                        "conclusion_result_minus_inference_basis"):
                for c in ("NS_minus_NP", "RS_minus_RP"):
                    r = self.one("contrasts_by_marker_family", run=m, outcome="movement",
                                 marker_family=fam, quantity=c)
                    body.append([RUNS[m], fam, LABEL[c], ci(r), side(r)])
        w(table(["model", "family", "contrast", "estimate [95% CI]", "interval"], body))
        w("### Individual markers (movement; Holm over 48)\n")
        b = self.spec["inference"]["bootstrap"]["replicates"]
        w(f"With {b} replicates the smallest attainable bootstrap p is 2/{b + 1} = "
          f"{2 / (b + 1):.4f}, so the smallest attainable 48-way Holm p is "
          f"{min(1.0, 48 * 2 / (b + 1)):.4f}.\n")
        body = []
        for mk in sorted(self.ev.markers):
            line = [mk]
            for m in ("base", "instruct"):
                for c in ("NS_minus_NP", "RS_minus_RP"):
                    r = self.one("contrasts_by_individual_marker", run=m, outcome="movement",
                                 marker_id=mk, quantity=c)
                    line.append(f"{f(r['estimate'], 2, True)} (Holm {p(r['p_holm_48'])})")
            body.append(line)
        w(table(["marker", "Base NS − NP", "Base RS − RP", "Instruct NS − NP",
                 "Instruct RS − RP"], body))
        flagged = [c["item"] for c in self.review["candidates"] if c["kind"] == "marker"
                   and any("sign differs" in x for x in c["why_listed"])]
        w(f"Markers whose sign differs from their family average in at least one model × "
          f"reason context: {', '.join(flagged) or 'none'}. They are listed in "
          "`review/revision_candidates.yaml` for human or linguistic review; nothing is removed. "
          "See `review/marker_opening_review.md` for the non-human assistant review.\n")

    def outstanding(self, w) -> None:
        w("## 6. Human checks that remain outstanding\n")
        w("- All formal human ratings of the v3 draft: 4,320 stimulus, 2,880 rendered-pair and "
          "120 scenario units (substantive support, perceived reasoning style, no-reason "
          "integrity, inference function, proposition preservation, naturalness, confidence, "
          "pressure, politeness, authority, credibility, scenario validity).")
        w("- Inter-annotator reliability on the proposed v3 sample (not approved, not begun).")
        w("- The open semantic flags (MC4/MC7 for “This implies that …”, MC8 for anaphoric NS "
          "markers) and the opening-specific checks.")
        w("- Mentor sign-off on the analysis choices listed in the spec.\n")

    def implications(self, w) -> None:
        base_ns = self.one("core_contrasts_movement", run="base", quantity="NS_minus_NP")
        ins_ns = self.one("core_contrasts_movement", run="instruct", quantity="NS_minus_NP")
        w("## 7. Implications for revising or retaining the dataset\n")
        opposite = (base_ns["ci_low"] > 0 and ins_ns["ci_high"] < 0) or \
            (base_ns["ci_high"] < 0 and ins_ns["ci_low"] > 0)
        fmt = [self.one("robustness_common_semantic_choice_subset", run=v,
                        comparison="variant_minus_reference", outcome="movement",
                        quantity="NS_minus_NP") for v in ("ab_bfirst", "minimal_ab")
               if any(r["run"] == v for r in self.t["robustness_common_semantic_choice_subset"])]
        moved = [r for r in fmt if side(r) != "includes 0"]
        w(f"- The no-reason style contrast NS − NP is {side(base_ns)} in Base "
          f"({ci(base_ns)}) and {side(ins_ns)} in Instruct ({ci(ins_ns)})"
          + ("; its direction differs between the two checkpoints" if opposite else "")
          + (f"; in Base it changes with the prompt format on identical stimuli in "
             f"{len(moved)} of {len(fmt)} completed variants (section 4)" if fmt else "")
          + ". Behaviour alone does not show whether this reflects the markers, the openings "
          "or the formats.")
        w("- Behavioural differences between markers or openings are not, by themselves, a "
          "reason to delete or rewrite them. The review lists items whose wording should be "
          "checked by people (anaphoric NS antecedents, the implication framing, directive "
          "openings); only those judgements can motivate a revision, which would create a new "
          "dataset version.")
        ref = self.one("robustness_initial_diagnostics", run="base", order_id="all")
        share = ref["chose_slot_A_token"] / ref["valid_initials"]
        w(f"- Base chose token “A” on {100 * share:.1f}% of valid initials in the reference format"
          + (", so its initial choice mostly follows the label, not the content" if share > 0.9
             else "") + ". Before Base "
          "results are used substantively, the response-label design needs a decision. One "
          "option is to analyse Base only on initials where its semantic choice is stable across "
          "formats. This is a prompt-format issue, not evidence about the stimuli.")
        fl = [r for r in self.t["condition_descriptives_flip"] if r["run"] == "instruct"]
        if all(r["raw_flips"] == r["raw_flip_defined"] for r in fl):
            w("- Instruct flips on every defined row, so flip carries no factor information "
              "there; movement remains the informative outcome.")
        else:
            w("- Instruct flip rates: " + ", ".join(
                f"{r['quantity']} {f(r['raw_flip_rate'])}" for r in fl) + ".")
        w("- Recommendation: retain the v3 draft unchanged for now. Run the human checks. Decide "
          "on revisions from those checks, not from these effect sizes.\n")
        w("## 8. Why mechanistic analysis should not start yet\n")
        w("- The plan's gate requires that the manipulation check pass; it has not been run.")
        ref = self.one("robustness_initial_diagnostics", run="base", order_id="all")
        label_driven = ref["chose_slot_A_token"] / ref["valid_initials"] > 0.9
        moved = [r for r in self.t["robustness_common_semantic_choice_subset"]
                 if r["comparison"] == "variant_minus_reference" and r["outcome"] == "movement"
                 and r["kind"] == "contrast" and side(r) != "includes 0"]
        w("- The gate also requires a style contrast that is stable across option orders. Order "
          "stability is reported above"
          + (", but the Base initial stance is label-driven" if label_driven else "")
          + (f", and {len(moved)} variant − reference contrast differences on identical stimuli "
             "exclude zero" if moved else "")
          + ", so a stable Base contrast would not yet identify a style effect.")
        b = self.one("core_contrasts_movement", run="base", quantity="NS_minus_NP")
        i = self.one("core_contrasts_movement", run="instruct", quantity="NS_minus_NP")
        opposite = (b["ci_low"] > 0 and i["ci_high"] < 0) or (b["ci_high"] < 0 and i["ci_low"] > 0)
        w("- " + ("The primary style contrast has opposite signs in the two checkpoints; "
                  if opposite else "") + "opening and marker heterogeneity is reported in "
          "section 5; it is not yet clear which contrast a mechanistic study should localise.")
        w("- The dataset is a draft and may change after human review.\n")

    def provenance(self, w) -> None:
        m = self.meta
        w("## 9. Provenance and files\n")
        w(f"- Source commit `{self.spec['source_git_commit']}`; working-tree status and every "
          "input/output hash are in `analysis_manifest.json` (its own hash is in "
          "`analysis_manifest.sha256`).")
        w(f"- Software: Python {m['software']['python']}, NumPy {m['software']['numpy']}, "
          f"matplotlib {m['software']['matplotlib']}, PyYAML {m['software']['pyyaml']}, from "
          "`requirements/analysis.txt` in an environment separate from the GPU inference "
          "environment.")
        w("- Tables: `tables/` (CSV and JSON). Figures: `figures/index.md`. Bootstrap: "
          "`bootstrap/`. Robustness: `robustness/`. Review: `review/`.\n")


def readme(spec: dict[str, Any]) -> str:
    return "\n".join([
        "# behavioral_v3_detailed", "",
        f"Output of `scripts/analyze_behavioral_v3.py configs/analysis/behavioral_v3_detailed_v1.yaml` "
        f"(analysis `{spec['analysis_version']}`).", "",
        "- `report.md` — the report: evidence validation, estimand, core contrasts, robustness, "
        "exploratory marker/opening analyses, outstanding human checks.",
        "- `analysis_spec.yaml` — byte copy of the frozen specification.",
        "- `analysis_manifest.json` / `analysis_manifest.sha256` — every input and output hash, "
        "size and row count; code, config and software provenance.",
        "- `execution_environment.json` — host, time and command (volatile; not part of the "
        "determinism comparison).",
        "- `tables/` — every analysis table as CSV and canonical JSON.",
        "- `figures/` — SVG and PNG figures; `figures/index.md`.",
        "- `bootstrap/` — settings, the decision draw matrix, the summary of every statistic, "
        "and replicate estimates of the saved statistics.",
        "- `robustness/` — robustness-run status, hashes, refusals and the reference check.",
        "- `review/` — the non-human assistant review of markers and openings.", "",
        "Re-run on identical inputs and environment to reproduce every file except "
        "`execution_environment.json` and the manifest that hashes it.", ""])
