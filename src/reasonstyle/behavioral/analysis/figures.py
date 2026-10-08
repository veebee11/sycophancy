"""The fourteen figures of ``behavioral_v3_detailed_v1`` (SVG and PNG).

Deterministic: Agg backend, fixed font, size, DPI, palette and SVG hash salt,
and no creation date in SVG or PNG metadata. Every contrast plot has a solid
zero line; models and runs are distinguished by marker shape as well as
colour, and intervals are drawn, so nothing relies on colour alone.
"""

from __future__ import annotations

import io
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .tables import CONTRASTS, LABEL  # noqa: E402

SHAPES = {"base": "o", "instruct": "s", "ab_bfirst": "^", "minimal_ab": "D"}
SHORT_OPENING = {"directive_reconsideration": "directive", "stance_alternative": "alternative",
                 "stance_disagreement": "disagreement"}
RUN_LABEL = {"base": "Base (reference)", "instruct": "Instruct", "ab_bfirst": "Base R1: B line first",
             "minimal_ab": "Base R3: minimal scaffold"}


def configure(spec: dict[str, Any]) -> dict[str, Any]:
    fig = spec["figures"]
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "svg.hashsalt": "behavioral_v3_detailed_v1",
        "svg.fonttype": "path", "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": fig["dpi"], "savefig.dpi": fig["dpi"], "path.simplify": False,
    })
    return fig


class FigureSet:
    def __init__(self, spec: dict[str, Any], tables: dict[str, list[dict[str, Any]]]):
        self.cfg = configure(spec)
        self.palette = self.cfg["palette"]
        self.t = tables
        self.files: dict[str, bytes] = {}
        self.index: list[tuple[str, str, str]] = []

    def color(self, run: str) -> str:
        return self.palette.get(run, self.palette["neutral"])

    def save(self, fig, name: str, title: str, description: str) -> None:
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        for fmt in self.cfg["formats"]:
            buf = io.BytesIO()
            meta = ({"Date": None, "Creator": "reasonstyle behavioral_v3_detailed_v1"}
                    if fmt == "svg" else {"Software": "reasonstyle behavioral_v3_detailed_v1"})
            fig.savefig(buf, format=fmt, metadata=meta)
            self.files[f"figures/{name}.{fmt}"] = buf.getvalue()
        plt.close(fig)
        self.index.append((name, title, description))

    def rows(self, table: str, **where: Any) -> list[dict[str, Any]]:
        return [r for r in self.t[table] if all(r.get(k) == v for k, v in where.items())]

    def one(self, table: str, **where: Any) -> dict[str, Any]:
        rows = self.rows(table, **where)
        if len(rows) != 1:
            raise ValueError(f"{table} {where}: {len(rows)} rows")
        return rows[0]

    def forest(self, ax, items: list[tuple[str, str, dict[str, Any]]], zero: bool = True) -> None:
        """items: (y label, run, row) from top to bottom."""
        n = len(items)
        for i, (label, run, r) in enumerate(items):
            y = n - 1 - i
            if r["estimate"] is None:
                continue
            ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=self.color(run), lw=1.4)
            ax.plot(r["estimate"], y, marker=SHAPES.get(run, "o"), color=self.color(run),
                    ms=5, ls="none")
        ax.set_yticks(range(n))
        ax.set_yticklabels([lab for lab, _, _ in reversed(items)])
        if zero:
            ax.axvline(0, color="black", lw=1.0, zorder=0)
        ax.set_xlabel("estimate in logits (movement toward the counterargument), 95% CI")

    def legend(self, ax, runs: list[str]) -> None:
        for run in runs:
            ax.plot([], [], marker=SHAPES[run], color=self.color(run), ls="-", label=RUN_LABEL[run])
        ax.legend(loc="best", fontsize=7, frameon=False)

    # -- individual figures --------------------------------------------------------
    def build(self) -> None:
        size = self.cfg["size_inches"]
        models = ["base", "instruct"]

        fig, axes = plt.subplots(1, 2, figsize=size["single"])
        for ax, m in zip(axes, models):
            rows = [self.one("condition_descriptives_movement", run=m, quantity=c)
                    for c in ("RS", "RP", "NS", "NP")]
            for x, r in enumerate(rows):
                ax.errorbar(x, r["estimate"], yerr=[[r["estimate"] - r["ci_low"]],
                                                    [r["ci_high"] - r["estimate"]]],
                            marker=SHAPES[m], color=self.color(m), capsize=3)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["RS", "RP", "NS", "NP"])
            ax.set_title(RUN_LABEL[m])
            ax.set_ylabel("mean movement (logits), 95% CI")
            ax.set_ylim(bottom=0)
        self.save(fig, "fig01_condition_movement_means", "Condition movement means by model",
                  "Decision-weighted mean movement toward the counterargument in each of the four "
                  "cells, per model; the panels have different scales.")

        fig, ax = plt.subplots(figsize=size["single"])
        items = [(f"{LABEL[c]} · {m}", m, self.one("core_contrasts_movement", run=m, quantity=c))
                 for c in CONTRASTS for m in models]
        self.forest(ax, items)
        self.legend(ax, models)
        self.save(fig, "fig02_core_contrasts_forest", "Core movement contrasts",
                  "Five planned movement contrasts per model with stratified decision-cluster "
                  "bootstrap 95% intervals.")

        fig, ax = plt.subplots(figsize=size["single"])
        runs = ["base", "instruct", "ab_bfirst", "minimal_ab"]
        for j, run in enumerate(runs):
            for x, c in enumerate(("RS", "RP", "NS", "NP")):
                r = self.one("condition_descriptives_flip", run=run, quantity=c)
                xx = x + (j - 1.5) * 0.18
                ax.errorbar(xx, r["estimate"], yerr=[[r["estimate"] - r["ci_low"]],
                                                     [r["ci_high"] - r["estimate"]]],
                            marker=SHAPES[run], color=self.color(run), capsize=2, ls="none")
        ax.axhline(1.0, color="black", lw=1.0, ls="--")
        ax.text(3.45, 1.005, "ceiling (every defined row flips)", ha="right", va="bottom", fontsize=7)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["RS", "RP", "NS", "NP"])
        ax.set_ylim(0.5, 1.04)
        ax.set_ylabel("flip rate among flip-defined rows, 95% CI")
        self.legend(ax, runs)
        self.save(fig, "fig03_flip_rates_ceiling", "Flip rates and ceiling",
                  "Decision-weighted flip rates per cell and run, with the 100% ceiling marked; "
                  "exact post ties are excluded from flip only.")

        def strat_forest(table, attr, levels, name, title, desc, kind="contrast"):
            fig, axes = plt.subplots(1, 2, figsize=size["tall"], sharey=True)
            for ax, m in zip(axes, models):
                items = []
                for c in CONTRASTS:
                    for lv in levels:
                        r = self.one(table, run=m, outcome="movement", kind=kind, quantity=c,
                                     **{attr: lv})
                        items.append((f"{LABEL[c]} · {lv}", m, r))
                self.forest(ax, items)
                ax.set_title(RUN_LABEL[m])
                ax.xaxis.label.set_size(7)
            self.save(fig, name, title, desc)

        strat_forest("contrasts_by_option_order", "order_id", ["o1", "o2"],
                     "fig04_contrasts_by_option_order", "Core contrasts by option order",
                     "Movement contrasts estimated separately in each displayed option order.")
        fig, ax = plt.subplots(figsize=size["single"])
        items = [(f"{LABEL[c]} · {m}", m, self.one("order_heterogeneity", run=m, outcome="movement",
                                                   comparison="o1_minus_o2", quantity=c))
                 for c in CONTRASTS for m in models]
        self.forest(ax, items)
        self.legend(ax, models)
        self.save(fig, "fig05_order_heterogeneity", "Order heterogeneity (o1 − o2)",
                  "Difference between the o1 and o2 estimates of each movement contrast.")
        strat_forest("contrasts_by_domain", "domain", ["climate", "energy", "technology"],
                     "fig06_contrasts_by_domain", "Core contrasts by domain",
                     "Movement contrasts within each policy domain (20 decisions each).")
        openings = sorted({r["opening_id"] for r in self.t["contrasts_by_opening"]})
        strat_forest("contrasts_by_opening", "opening_id", openings,
                     "fig07_contrasts_by_opening", "Core contrasts by opening",
                     "Movement contrasts within each of the three openings.")

        markers = sorted({r["marker_id"] for r in self.t["contrasts_by_individual_marker"]})
        fig, axes = plt.subplots(1, 2, figsize=size["tall"], sharey=True)
        for ax, c in zip(axes, ("NS_minus_NP", "RS_minus_RP")):
            items = [(f"{mk} · {m}", m, self.one("contrasts_by_individual_marker", run=m,
                                                 outcome="movement", marker_id=mk, quantity=c))
                     for mk in markers for m in models]
            self.forest(ax, items)
            ax.set_title(LABEL[c])
            ax.xaxis.label.set_size(7)
        self.legend(axes[1], models)
        self.save(fig, "fig08_marker_forest", "Individual markers: NS − NP and RS − RP",
                  "Per-marker style contrasts paired with each block's shared plain control "
                  "(exploratory; Holm across 48 in the table).")

        fig, ax = plt.subplots(figsize=size["single"])
        items = []
        for c in ("NS_minus_NP", "RS_minus_RP", "RS_minus_NS", "interaction"):
            for fam in ("conclusion_result", "inference_basis"):
                for m in models:
                    items.append((f"{LABEL[c]} · {fam} · {m}", m,
                                  self.one("contrasts_by_marker_family", run=m, outcome="movement",
                                           marker_family=fam, quantity=c)))
        self.forest(ax, items)
        ax.tick_params(axis="y", labelsize=6)
        self.legend(ax, models)
        self.save(fig, "fig09_marker_family", "Marker-family summaries",
                  "Unweighted means of the six marker estimates in each family.")

        fig, axes = plt.subplots(2, 2, figsize=size["tall"])
        for i, m in enumerate(models):
            for j, c in enumerate(("NS_minus_NP", "RS_minus_RP")):
                ax = axes[i][j]
                grid = [[self.one("marker_by_opening", run=m, marker_id=mk, opening_id=o,
                                  quantity=c)["estimate"] for o in openings] for mk in markers]
                vmax = max(abs(v) for row in grid for v in row) or 1.0
                im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
                for y, row in enumerate(grid):
                    for x, v in enumerate(row):
                        ax.text(x, y, f"{v:+.2f}", ha="center", va="center", fontsize=6)
                ax.set_xticks(range(len(openings)))
                ax.set_xticklabels([SHORT_OPENING.get(o, o) for o in openings], fontsize=6)
                ax.set_yticks(range(len(markers)))
                ax.set_yticklabels(markers, fontsize=6)
                ax.set_title(f"{RUN_LABEL[m]}: {LABEL[c]}", fontsize=8)
                fig.colorbar(im, ax=ax, shrink=0.7)
        self.save(fig, "fig10_marker_opening_heatmaps", "Marker × opening",
                  "Per-marker style contrasts within each opening; every cell is labelled with its "
                  "signed estimate (intervals are in marker_by_opening).")

        fig, axes = plt.subplots(1, 2, figsize=size["tall"], sharey=True)
        for ax, m in zip(axes, models):
            items = [(f"{LABEL[c]} · {q}", m, self.one("margin_quartiles", run=m, quartile=q,
                                                         quantity=c))
                     for c in CONTRASTS for q in ("Q1", "Q2", "Q3", "Q4")]
            self.forest(ax, items)
            ax.set_title(RUN_LABEL[m])
            ax.xaxis.label.set_size(7)
        self.save(fig, "fig11_margin_quartiles", "Contrasts by initial-margin quartile",
                  "Movement contrasts within model-specific quartiles of |m_before|.")

        fig, axes = plt.subplots(1, 2, figsize=size["wide"])
        all_runs = ["base", "instruct", "ab_bfirst", "minimal_ab"]
        for ax, (table, key, cats) in zip(axes, (
                ("initial_choice_by_model_order_label", "slot", ["A", "B", "tie"]),
                ("initial_semantic_choice_by_model_order", "initial_option",
                 ["opt_1", "opt_2", "tie"]))):
            labels, bottoms = [], []
            for k, run in enumerate(all_runs):
                for o in ("o1", "o2"):
                    x = len(labels)
                    labels.append(f"{run}\n{o}")
                    bottom = 0.0
                    for h, cat in enumerate(cats):
                        r = self.one(table, run=run, order_id=o, **{key: cat})
                        ax.bar(x, r["pct"], bottom=bottom, color=["#4D4D4D", "#BBBBBB", "#FFFFFF"][h],
                               edgecolor="black", lw=0.5, hatch=["", "//", ".."][h],
                               label=cat if (k == 0 and o == "o1") else None)
                        bottom += r["pct"]
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
            ax.set_ylabel("% of initial readings")
            ax.legend(fontsize=7, frameon=False, title="slot (token)" if key == "slot" else
                      "semantic option")
        self.save(fig, "fig12_initial_choices", "Initial A/B and semantic-option choices",
                  "Share of initial readings choosing each answer slot and each semantic option, "
                  "by run and option order (slots: R1 shows the B line first).")

        fig, ax = plt.subplots(figsize=size["single"])
        diag = [r for r in self.t["robustness_initial_diagnostics"] if r["order_id"] == "all"
                and r.get("valid_initials")]
        metrics = [("token A", lambda r: 100 * r["chose_slot_A_token"] / r["valid_initials"]),
                   ("first-listed line", lambda r: 100 * r["chose_first_listed"] / r["valid_initials"]),
                   ("exact initial ties", lambda r: 100 * r["exact_initial_ties"] / r["initial_readings"]),
                   ("same semantic choice as reference",
                    lambda r: r["agreement_pct"])]
        for j, r in enumerate(diag):
            for i, (name, fn) in enumerate(metrics):
                ax.plot(fn(r), len(metrics) - 1 - i + (j - 1) * 0.15, marker=SHAPES[r["run"]],
                        color=self.color(r["run"]), ls="none", ms=6)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([m for m, _ in reversed(metrics)])
        ax.set_xlim(0, 100)
        ax.set_xlabel("% (token, position and agreement among valid initials; ties among all)")
        self.legend(ax, [r["run"] for r in diag])
        self.save(fig, "fig13_robustness_diagnostics", "Response-label / line-position diagnostics",
                  "Token-A share, first-listed share, exact-tie share and agreement with the "
                  "reference semantic choice for the reference and each completed Base variant; "
                  "R2 (1/2 labels) was refused at its tokenizer gate.")

        fig, axes = plt.subplots(1, 2, figsize=size["tall"], sharey=True)
        variants = ["base", "ab_bfirst", "minimal_ab"]
        for ax, (title, table, extra) in zip(axes, (
                ("each run's full valid sample", "robustness_contrasts",
                 {"sample": "full_valid", "stratum": "all", "kind": "contrast"}),
                ("common semantic-choice subset", "robustness_common_semantic_choice_subset",
                 {"comparison": "estimate", "kind": "contrast"}))):
            items = [(f"{LABEL[c]} · {RUN_LABEL[v]}", v,
                      self.one(table, run=v, outcome="movement", quantity=c, **extra))
                     for c in CONTRASTS for v in variants]
            self.forest(ax, items)
            ax.set_title(title)
            ax.tick_params(axis="y", labelsize=6)
            ax.xaxis.label.set_size(7)
        self.save(fig, "fig14_robustness_contrasts", "Core contrasts across Base prompt formats",
                  "Movement contrasts for the reference Base format and the two completed "
                  "variants, on each run's full valid sample and on the subset with identical "
                  "initial semantic choices.")

    def index_markdown(self) -> str:
        lines = ["# Figures", "",
                 "Generated by `scripts/analyze_behavioral_v3.py` from the tables in `../tables/`. "
                 "Each figure exists as SVG and PNG. Contrast plots show a zero line; models and "
                 "runs differ by marker shape as well as colour.", ""]
        for name, title, desc in self.index:
            lines += [f"## {title}", "", f"[![{title}]({name}.png)]({name}.svg)", "",
                      f"{desc} Files: [`{name}.svg`]({name}.svg), [`{name}.png`]({name}.png).", ""]
        return "\n".join(lines)
