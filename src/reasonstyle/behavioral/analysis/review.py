"""Exploratory marker and opening review (non-human, assistant-authored).

Two kinds of evidence are kept apart:

1. observed model behaviour - estimates computed here from validated runs;
2. assistant-authored linguistic and structural observations - written by the
   AI assistant that built this pipeline, plus machine string checks.

Neither is human annotation, a manipulation check or reliability evidence. No
phrase is removed or revised; ``revision_candidates.yaml`` lists items for later
human or linguistic review only, under the rule fixed in the analysis spec.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import yaml

from .estimate import mean_of, summarise
from .tables import LABEL, STYLE_PAIR, Tables, marker_movement
from .validate import FAMILIES, REFERENCE_MODELS

CONTEXT = {"NS_minus_NP": "reason_absent", "RS_minus_RP": "reason_present"}
REVIEW_STATUS = ("NON-HUMAN ASSISTANT REVIEW. Written by the AI assistant that built this "
                 "analysis; not human annotation, not a manipulation check, not reliability "
                 "evidence.")

# Assistant-authored observations (linguistic/structural), one per marker. These are
# hypotheses for a human or linguistic reviewer, not findings.
MARKER_NOTES: dict[str, dict[str, Any]] = {
    "m01_therefore": {"antecedent_dependence": "none (connective)",
                      "note": "Conclusive adverb; presupposes a preceding premise. In NS the only "
                              "preceding sentence is the opening, so the opening is presented as "
                              "the premise."},
    "m02_consequently": {"antecedent_dependence": "none (connective)",
                         "note": "Result adverb; reads most naturally after an event or state, so "
                                 "after a stance or directive opening it may sound like a causal "
                                 "rather than inferential link."},
    "m03_thus": {"antecedent_dependence": "none (connective)",
                 "note": "Short conclusive adverb; formal register; same premise presupposition "
                         "as 'therefore'."},
    "m04_accordingly": {"antecedent_dependence": "none (connective)",
                        "note": "Ambiguous between 'therefore' and 'in a corresponding manner'; "
                                "the second reading weakens the inferential function."},
    "m05_for_this_reason": {"antecedent_dependence": "demonstrative 'this' (anaphoric)",
                            "note": "Names a reason by anaphora. In NS the antecedent can only be "
                                    "the opening, so the opening itself is offered as the reason "
                                    "(circular); open review flag MC8."},
    "m06_that_is_why": {"antecedent_dependence": "demonstrative 'that' (anaphoric)",
                        "note": "Conversational causal clause without a comma; anaphoric 'that'. "
                                "After 'You should reconsider that choice.' the two 'that's have "
                                "different referents; check attachment and register."},
    "m07_this_implies_that": {"antecedent_dependence": "demonstrative 'this' (anaphoric)",
                              "note": "Frames a preference as a logical implication (open flags "
                                      "MC4/MC7). After 'I would make a different choice.' in a "
                                      "binary decision the implication is close to valid, so NS "
                                      "may state a trivial but real inference."},
    "m08_it_follows_that": {"antecedent_dependence": "none (expletive 'it'; 'that' is a "
                                                     "complementiser)",
                            "note": "Strong entailment claim without an explicit antecedent; may "
                                    "read as more certain than the other markers (confidence "
                                    "confound to check)."},
    "m09_on_that_basis": {"antecedent_dependence": "demonstrative 'that' (anaphoric)",
                          "note": "Anaphoric basis phrase; in NS the basis is the opening."},
    "m10_based_on_this": {"antecedent_dependence": "demonstrative 'this' (anaphoric)",
                          "note": "Participial phrase attached to 'I'; conventional usage but "
                                  "prescriptively a dangling modifier; anaphoric 'this'."},
    "m11_given_this": {"antecedent_dependence": "demonstrative 'this' (anaphoric)",
                       "note": "Anaphoric; 'given this' can also read as concessive "
                               "background rather than a reason."},
    "m12_in_view_of_this": {"antecedent_dependence": "demonstrative 'this' (anaphoric)",
                            "note": "Formal anaphoric basis phrase; longest prefix with the "
                                    "comma realisation."},
}
OPENING_NOTES = {
    "stance_disagreement": "Stance statement. As an antecedent for an anaphoric NS marker it makes "
                           "the disagreement itself the stated reason.",
    "directive_reconsideration": "Directive with second-person address; adds pragmatic force. As "
                                 "an antecedent ('Given this', 'For this reason') a directive is "
                                 "an unusual reason; 'that choice' and 'That is why' stack two "
                                 "different 'that's.",
    "stance_alternative": "States a different choice; in a two-option decision this already "
                          "entails the opposite option, so an inferential NS marker can present a "
                          "near-valid inference rather than an empty one.",
}


def _structural_checks(tables: Tables) -> dict[str, dict[str, Any]]:
    """Machine string checks over every stimulus, per marker (and for plain cells)."""
    ev = tables.ev
    by_id = ev.stimuli
    out: dict[str, dict[str, Any]] = defaultdict(lambda: {"stimuli": 0, "failures": []})
    for sid, s in sorted(by_id.items()):
        key = s["marker_id"]
        rec = out[key]
        rec["stimuli"] += 1
        opening = ev.openings[s["opening_id"]]
        if s["rendered"] != f"{opening} {s['body']}":
            rec["failures"].append(f"{sid}: rendered != opening + ' ' + body")
        if s["condition"] in ("RS", "NS"):
            ctrl = by_id[s["paired_control_stimulus_id"]]
            if (ctrl["opening_id"], ctrl["scenario_id"], ctrl["supported_option"]) != (
                    s["opening_id"], s["scenario_id"], s["supported_option"]):
                rec["failures"].append(f"{sid}: control differs in opening/scenario/option")
            if ctrl["condition"] != {"RS": "RP", "NS": "NP"}[s["condition"]]:
                rec["failures"].append(f"{sid}: control has the wrong condition")
            if s["body"].replace(s["marker_prefix"], "", 1) != _plain_body(s, ctrl):
                rec["failures"].append(f"{sid}: removing the prefix does not give the control "
                                       "body")
            expected = (s["marker_prefix"] + s["endorsement"] if s["condition"] == "NS"
                        else f"{s['premise']} {s['marker_prefix']}{s['endorsement']}")
            if s["body"] != expected:
                rec["failures"].append(f"{sid}: body is not the fixed template")
    return dict(out)


CONNECTIVE = re.compile(
    r"\b(so|because|therefore|thus|hence|since|consequently|which means|given that|"
    r"as a result)\b", re.IGNORECASE)


def premise_connectives(tables: Tables) -> list[dict[str, Any]]:
    """Groups whose premise (shared by RS and RP) itself contains an inferential or result
    connective - so the 'plain' RP control is not free of explicit connectives."""
    seen = {}
    for s in tables.ev.stimuli.values():
        if s["condition"] == "RP":
            hits = sorted({m.lower() for m in CONNECTIVE.findall(s["premise"])})
            if hits:
                seen[(s["scenario_id"], s["supported_option"])] = {
                    "scenario_id": s["scenario_id"], "supported_option": s["supported_option"],
                    "connectives": hits, "premise": s["premise"]}
    return [seen[k] for k in sorted(seen)]


def _plain_body(styled: dict[str, Any], ctrl: dict[str, Any]) -> str:
    return ctrl["body"]


def _token_lengths(tables: Tables) -> dict[str, list[int]]:
    """Extra post-prompt tokens of each marker, from recorded input lengths (Base runs)."""
    out: dict[str, set[int]] = defaultdict(set)
    for b in tables.blocks:
        if b.model != "base":
            continue
        for cond, ctrl in (("NS", "NP"), ("RS", "RP")):
            base_len = next(iter(b.input_tokens[ctrl].values()))
            for m, n in b.input_tokens[cond].items():
                if n is not None and base_len is not None:
                    out[m].add(n - base_len)
    return {m: sorted(v) for m, v in out.items()}


def build_review(tables: Tables) -> dict[str, Any]:
    ev = tables.ev
    markers = ev.markers
    openings = sorted(ev.openings)
    effects: list[dict[str, Any]] = []
    # marker x model x context x opening, with family means and deviations
    for model in REFERENCE_MODELS:
        for scope in ["all"] + openings:
            blocks = tables.sel(model, (lambda b: True) if scope == "all"
                                else (lambda b, o=scope: b.opening_id == o))
            for c in STYLE_PAIR:
                ests = {m: tables.est(blocks, marker_movement(c, m)) for m in sorted(markers)}
                fam = {f: mean_of([ests[m] for m in markers if markers[m]["family"] == f])
                       for f in FAMILIES}
                for m, e in ests.items():
                    info = markers[m]
                    s = summarise(e, tables.level)
                    fs = summarise(fam[info["family"]], tables.level)
                    dev = summarise(e - fam[info["family"]], tables.level)
                    effects.append({
                        "analysis": "marker_x_model_x_context_x_opening", "run": model,
                        "marker_id": m, "marker_family": info["family"],
                        "marker_subtype": info["subtype"], "opening_id": scope,
                        "context": CONTEXT[c], "quantity": c,
                        "estimate": s["estimate"], "ci_low": s["ci_low"], "ci_high": s["ci_high"],
                        "p_boot": s["p_boot"], "n_decisions": s["n_decisions"],
                        "n_blocks": s["n_blocks"], "n_initials": s["n_initials"],
                        "n_movement_rows": s["n_movement_rows"],
                        "family_mean": fs["estimate"],
                        "deviation_from_family": dev["estimate"],
                        "deviation_ci_low": dev["ci_low"], "deviation_ci_high": dev["ci_high"],
                        "sign_differs_from_family": (s["estimate"] > 0) != (fs["estimate"] > 0),
                    })
    # opening x model x order, opening x margin stratum, marker x domain
    for model in REFERENCE_MODELS:
        for o in openings:
            for order in ("o1", "o2"):
                blocks = tables.sel(model, lambda b, o=o, r=order: b.opening_id == o and
                                    b.order_id == r)
                effects += _core_rows(tables, blocks, "opening_x_order", model, opening_id=o,
                                      order_id=order)
            for stratum, keep in (("near_tie", lambda b: b.abs_m_before < tables.tau),
                                  ("not_near_tie", lambda b: b.abs_m_before >= tables.tau)):
                blocks = tables.sel(model, lambda b, o=o, k=keep: b.opening_id == o and k(b))
                effects += _core_rows(tables, blocks, "opening_x_margin", model, opening_id=o,
                                      margin_stratum=stratum)
        for m in sorted(markers):
            for dom in ("climate", "energy", "technology"):
                blocks = tables.sel(model, lambda b, d=dom: b.domain == d)
                for c in STYLE_PAIR:
                    s = summarise(tables.est(blocks, marker_movement(c, m)), tables.level)
                    effects.append({"analysis": "marker_x_domain", "run": model, "marker_id": m,
                                    "marker_family": markers[m]["family"], "domain": dom,
                                    "context": CONTEXT[c], "quantity": c,
                                    **{k: s[k] for k in ("estimate", "ci_low", "ci_high",
                                                         "p_boot", "n_decisions", "n_blocks",
                                                         "n_initials", "n_movement_rows")}})
    structure = _structural_checks(tables)
    tokens = _token_lengths(tables)
    marker_info = []
    for m, info in sorted(markers.items()):
        st = structure.get(m, {"stimuli": 0, "failures": []})
        marker_info.append({
            "marker_id": m, "string": info["string"], "prefix": info["prefix"],
            "family": info["family"], "subtype": info["subtype"],
            "realization": info["realization_id"],
            "comma": info["prefix"].rstrip().endswith(","),
            "words": len(info["string"].split()),
            "added_tokens_in_context": tokens.get(m, []),
            "position": "sentence-initial, opening the concluding endorsement sentence",
            "stimuli_checked": st["stimuli"], "structural_failures": len(st["failures"]),
            **MARKER_NOTES[m]})
    plain = structure.get("none", {"stimuli": 0, "failures": []})
    premises = premise_connectives(tables)
    candidates = _candidates(effects, marker_info)
    if premises:
        candidates.append({
            "item": "premises_with_connectives", "kind": "premise", "authorizes_removal": False,
            "why_listed": ["assistant structural observation: the shared premise of RS and RP "
                           "contains a result/inferential connective"],
            "observed_behaviour": ["not analysed separately (exploratory follow-up only)"],
            "assistant_review": "The connective sits inside the premise, so it is identical in "
                                "RS and RP and the within-reason style pair stays matched, but "
                                "RP is not entirely free of explicit connectives.",
            "groups": [f"{p['scenario_id']}/{p['supported_option']} ({', '.join(p['connectives'])})"
                       for p in premises],
            "to_check_by_human_or_linguist": [
                "whether a premise-internal 'so' counts as explicit reasoning style for RP",
                "whether perceived reasoning style differs between these RP texts and other RP texts"],
        })
    return {"effects": effects, "marker_info": marker_info, "candidates": candidates,
            "premise_connectives": premises,
            "structural": {"plain_stimuli_checked": plain["stimuli"],
                           "plain_failures": len(plain["failures"]),
                           "failures": sorted(f for s in structure.values()
                                              for f in s["failures"])}}


def _core_rows(tables: Tables, blocks, analysis: str, model: str, **keys) -> list[dict]:
    from .tables import movement_contrast
    rows = []
    for c in ("NS_minus_NP", "RS_minus_RP", "interaction"):
        s = summarise(tables.est(blocks, movement_contrast(c)), tables.level)
        rows.append({"analysis": analysis, "run": model, **keys, "quantity": c,
                     **{k: s[k] for k in ("estimate", "ci_low", "ci_high", "p_boot",
                                          "n_decisions", "n_blocks", "n_initials",
                                          "n_movement_rows")}})
    return rows


def _candidates(effects: list[dict], marker_info: list[dict]) -> list[dict[str, Any]]:
    """Predeclared rule: (a) an assistant observation applies, or (b) the marker's
    estimate differs in sign from its family average in some model x reason context."""
    sign = defaultdict(list)
    for r in effects:
        if r["analysis"] == "marker_x_model_x_context_x_opening" and r["opening_id"] == "all" \
                and r["sign_differs_from_family"]:
            sign[r["marker_id"]].append(
                f"{r['run']} {LABEL[r['quantity']]}: {r['estimate']:+.3f} "
                f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] vs family mean {r['family_mean']:+.3f}")
    out = []
    for info in marker_info:
        m = info["marker_id"]
        anaphoric = "anaphoric" in info["antecedent_dependence"]
        reasons = []
        if anaphoric:
            reasons.append("assistant observation: anaphoric marker whose NS antecedent can only "
                           "be the opening")
        if m in ("m07_this_implies_that", "m08_it_follows_that", "m04_accordingly",
                 "m06_that_is_why"):
            reasons.append(f"assistant observation: {info['note']}")
        if sign[m]:
            reasons.append("behaviour: sign differs from the family average")
        if not reasons:
            continue
        out.append({
            "item": m, "kind": "marker", "string": info["string"],
            "authorizes_removal": False,
            "why_listed": reasons,
            "observed_behaviour": sign[m] or ["no sign difference from the family average "
                                              "(see marker_opening_effects.csv)"],
            "assistant_review": info["note"],
            "to_check_by_human_or_linguist": [
                "no-reason integrity of NS under each opening (does the marker import an "
                "unstated reason?)",
                "perceived reasoning style and naturalness under each opening",
                "proposition preservation of the styled/plain pair under each opening",
                "whether the antecedent of any demonstrative is the opening, the premise or "
                "nothing" if anaphoric else "whether the connective presupposes a premise the "
                "NS text lacks",
            ],
        })
    for o, note in OPENING_NOTES.items():
        out.append({
            "item": o, "kind": "opening", "authorizes_removal": False,
            "why_listed": ["assistant observation: opening interacts with anaphoric and "
                           "inferential markers"],
            "observed_behaviour": ["see opening_x_order, opening_x_margin and "
                                   "marker_x_model_x_context_x_opening rows"],
            "assistant_review": note,
            "to_check_by_human_or_linguist": [
                "pragmatic force and pressure relative to the other openings",
                "compatibility with each anaphoric marker as antecedent",
                "whether NS under this opening still supplies no substantive reason"],
        })
    return out


def candidates_yaml(review: dict[str, Any]) -> str:
    doc = {"status": REVIEW_STATUS,
           "rule": "listed if an assistant observation applies or a marker's sign differs from "
                   "its family average in some model x reason context; listing never authorises "
                   "removal or revision, and no estimate's size, sign or significance alone is a "
                   "reason to remove or revise a phrase",
           "authorizes_removal": False, "items": review["candidates"]}
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)


def rendered_examples(tables: Tables) -> str:
    ev = tables.ev
    lines = ["# Rendered examples", "", REVIEW_STATUS, "",
             "For each marker, the first (sorted) supported-option group that carries it, under "
             "all three openings: NS and RS with the marker, and the group's shared NP and RP "
             "controls. Texts are copied verbatim from `data/full_v3/stimuli_full_v3.jsonl`.", ""]
    by_group = defaultdict(list)
    for s in ev.stimuli.values():
        by_group[(s["scenario_id"], s["supported_option"])].append(s)
    for m in sorted(ev.markers):
        group = min(k for k, rows in by_group.items() if any(r["marker_id"] == m for r in rows))
        rows = by_group[group]
        lines += [f"## {m} — “{ev.markers[m]['string']}”", "",
                  f"Group `{group[0]}` / `{group[1]}`.", ""]
        for o in sorted(ev.openings):
            lines += [f"**{o}**", "", "| cell | text |", "|---|---|"]
            for cond in ("NP", "NS", "RP", "RS"):
                pick = [r for r in rows if r["opening_id"] == o and r["condition"] == cond and
                        r["marker_id"] in (m, "none")]
                lines.append(f"| {cond} | {pick[0]['rendered']} |")
            lines.append("")
    return "\n".join(lines)


def review_markdown(review: dict[str, Any], tables: Tables) -> str:
    eff = review["effects"]
    lines = ["# Marker and opening review (exploratory)", "", f"> {REVIEW_STATUS}", "",
             "No phrase was removed or revised. `revision_candidates.yaml` lists items for later "
             "human or linguistic review only; it authorises nothing.", "",
             "## 1. Observed model behaviour", "",
             "Estimates are decision-weighted movement contrasts with stratified decision-cluster "
             "bootstrap 95% intervals (exploratory, unadjusted here; the 48-comparison Holm "
             "adjustment is in `tables/contrasts_by_individual_marker`). Each marker contrast "
             "pairs the marker row with its block's shared plain control.", "",
             "### Marker × model × reason context (all openings)", "",
             "| model | marker | family | context | estimate [95% CI] | family mean | "
             "deviation [95% CI] | sign differs |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in eff if r["analysis"] == "marker_x_model_x_context_x_opening"
                     and r["opening_id"] == "all"),
                    key=lambda r: (r["run"], r["context"], r["marker_id"])):
        lines.append(f"| {r['run']} | {r['marker_id']} | {r['marker_family']} | {r['context']} | "
                     f"{r['estimate']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                     f"{r['family_mean']:+.3f} | {r['deviation_from_family']:+.3f} "
                     f"[{r['deviation_ci_low']:+.3f}, {r['deviation_ci_high']:+.3f}] | "
                     f"{'yes' if r['sign_differs_from_family'] else 'no'} |")
    lines += ["", "### Opening × model × option order", "",
              "| model | opening | order | contrast | estimate [95% CI] |", "|---|---|---|---|---|"]
    for r in sorted((r for r in eff if r["analysis"] == "opening_x_order"),
                    key=lambda r: (r["run"], r["opening_id"], r["quantity"], r["order_id"])):
        lines.append(f"| {r['run']} | {r['opening_id']} | {r['order_id']} | {LABEL[r['quantity']]} | "
                     f"{r['estimate']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] |")
    lines += ["", "Marker × opening, marker × domain and opening × initial-margin rows are in "
              "`marker_opening_effects.csv` (`analysis` column).", "",
              "## 2. Assistant-authored linguistic and structural observations", "",
              f"> {REVIEW_STATUS}", "",
              "Machine string checks (all stimuli): each rendered text is the opening, a space and "
              "the body; each styled body is the fixed template around the exact endorsement; "
              "removing the marker prefix gives the paired control's body; each styled stimulus "
              "is paired with the plain control of the same scenario, supported option, opening "
              "and reason level.", "",
              f"Structural failures: {len(review['structural']['failures'])} "
              f"(plain stimuli checked: {review['structural']['plain_stimuli_checked']}).", "",
              "| marker | prefix | family / subtype | comma | words | added tokens | antecedent "
              "dependence | observation |", "|---|---|---|---|---|---|---|---|"]
    for m in review["marker_info"]:
        lines.append(f"| {m['marker_id']} | `{m['prefix']}` | {m['family']} / {m['subtype']} | "
                     f"{'yes' if m['comma'] else 'no'} | {m['words']} | "
                     f"{','.join(str(x) for x in m['added_tokens_in_context'])} | "
                     f"{m['antecedent_dependence']} | {m['note']} |")
    pc = review["premise_connectives"]
    lines += ["", f"Premise-internal connectives: {len(pc)} of 240 shared premises contain a "
              "result or inferential connective ("
              + ", ".join(f"`{p['scenario_id']}/{p['supported_option']}`: "
                          f"{'/'.join(p['connectives'])}" for p in pc)
              + "). The connective is identical in RS and RP, so the style pair stays matched, "
              "but these RP controls are not free of explicit connectives."]
    lines += ["", "All markers sit in the same position: sentence-initial, opening the concluding "
              "endorsement sentence. In NS the only text before the marker is the opening, so "
              "every marker's presupposed premise or antecedent can only be the opening there.", "",
              "### Openings", ""]
    for o, note in OPENING_NOTES.items():
        lines.append(f"- **{o}** (“{tables.ev.openings[o]}”): {note}")
    lines += ["", "## 3. What would need checking", "",
              "See `revision_candidates.yaml`. A marker's estimate being large, small, opposite "
              "in sign or statistically distinguishable is not by itself a reason to remove or "
              "revise it; only human judgements of no-reason integrity, perceived style, "
              "naturalness and proposition preservation can motivate a new dataset version.", ""]
    return "\n".join(lines)
