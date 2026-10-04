"""Deterministic, read-only review export for the draft v3 stimulus set.

Inspection by decision (with its scenarios, supported options and conditions),
by marker, by marker family, by opening and by condition. Each styled body
names the one plain control it shares with the group's other marker variant.
No timestamps; regenerating writes identical bytes.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..hashing import sha256_of
from .build import Build
from .spec import FAMILIES, Spec

_NOTE = ("> **Generated, read-only review view of a DRAFT dataset.** Regenerate with "
         "`uv run python scripts/build_v3.py review`. Machine checks passed are listed; human "
         "judgements are outstanding and are never shown as passed.")


def _md(text: str) -> str:
    return text.replace("|", "\\|")


def build_review(spec: Spec, built: Build, manifest: dict[str, Any]) -> dict[str, str]:
    files: dict[str, str] = {}
    by_decision: dict[str, list[dict]] = defaultdict(list)
    for b in built.bodies:
        by_decision[b["decision_id"]].append(b)
    stimuli_of: dict[str, list[str]] = defaultdict(list)
    for s in built.stimuli:
        stimuli_of[s["body_id"]].append(s["stimulus_id"])
    markers = spec.markers
    c = manifest["counts"]

    head = [f"# {spec.version} — review (draft)\n", _NOTE, "",
            f"- source corpus `{manifest['source_corpus']}` sha256 `{manifest['source_corpus_sha256']}`",
            f"- allocation seed `{spec.seed}`, sha256 `{manifest['allocation']['sha256']}`",
            f"- status **{manifest['status']}**; formal annotation and behavioural evaluation "
            f"not started", ""]

    # -- index -------------------------------------------------------------------
    L = head + ["## ▶ Complete dataset: [all_decisions.md](all_decisions.md)\n",
                f"All {c['scenarios']} scenarios, both supported options, both allocated markers, "
                f"all six bodies per group and every one of the {c['stimuli']} rendered "
                f"counterarguments, with condition, marker, opening and paired-control ids.\n",
                "## Counts\n", "| quantity | value |", "|---|---|"]
    for key in ("decisions", "scenarios", "groups", "unique_bodies", "stimuli"):
        L.append(f"| {key} | {c[key]} |")
    for key in ("decisions_per_domain", "by_condition", "by_opening", "by_family", "by_domain",
                "by_supported_option", "by_marker"):
        L.append(f"| {key} | " + ", ".join(f"{k}: {v}" for k, v in c[key].items()) + " |")
    L += ["", "**Plain controls are shared.** " + manifest["plain_controls"], "",
          "## Balance\n", "```json", json.dumps(manifest["balance"], indent=1, sort_keys=True),
          "```", "", "## Machine validation\n",
          f"{manifest['machine_validation']['errors']} errors, "
          f"{manifest['machine_validation']['warnings']} warning(s).", ""]
    for w in manifest["machine_validation"]["warning_detail"]:
        L.append(f"- `{w['code']}` `{w['where']}` — {_md(w['message'])}")
    L += ["", "## Annotation units (all outstanding, none passed)\n",
          "| level | units | ratings |", "|---|---|---|"]
    for level, info in manifest["annotation_units"].items():
        L.append(f"| {level} | {info['units']} | " + ", ".join(
            f"{k}: {v}" for k, v in info["ratings"].items()) + " |")
    L += ["", "**Open semantic review flags:**", ""]
    L += [f"- {_md(flag)}" for flag in manifest["open_semantic_review_flags"]]
    L += ["", "## Pages\n", "- [conditions](conditions.md) · [openings](openings.md) · "
          "[families](families.md) · [annotation units](annotation.md) · "
          "[reliability proposal](reliability_proposal.md)", "", "### By decision\n",
          "| decision | domain | scenarios |",
          "|---|---|---|"]
    for d in sorted(by_decision):
        bs = by_decision[d]
        L.append(f"| [{d}](decisions/{d}.md) | {bs[0]['domain']} | "
                 f"{', '.join(sorted({b['scenario_id'] for b in bs}))} |")
    L += ["", "### By marker\n", "| marker | string | family | subtype | realization |",
          "|---|---|---|---|---|"]
    for mid, m in markers.items():
        L.append(f"| [{mid}](markers/{mid}.md) | {m.string} | {m.family} | {m.subtype} | "
                 f"{m.realization_id} |")
    files["index.md"] = "\n".join(L) + "\n"

    # -- decisions ---------------------------------------------------------------
    for d in sorted(by_decision):
        bs = by_decision[d]
        L = head + [f"## Decision `{d}` — {bs[0]['domain']}\n"]
        for scenario_id in sorted({b["scenario_id"] for b in bs}):
            first = next(b for b in bs if b["scenario_id"] == scenario_id)
            L += [f"### Scenario `{scenario_id}` (variant {first['variant_id']})\n",
                  f"> {first['scenario_text']}\n",
                  f"- `opt_1` — {first['options']['opt_1']}",
                  f"- `opt_2` — {first['options']['opt_2']}", ""]
            for option in ("opt_1", "opt_2"):
                group = [b for b in bs if (b["scenario_id"], b["supported_option"]) == (scenario_id, option)]
                rp = next(b for b in group if b["condition"] == "RP")
                np_ = next(b for b in group if b["condition"] == "NP")
                L += [f"#### Supported option `{option}`\n",
                      f"- premise: {rp['premise']}", f"- endorsement: {np_['endorsement']}", "",
                      "| condition | marker | family / subtype | body | paired control |",
                      "|---|---|---|---|---|"]
                for b in group:
                    marker = b["marker_string"] or "—"
                    fam = (f"{b['marker_family']} / {b['marker_subtype']}"
                           if b["marker_family"] != "none" else "—")
                    control = (f"`{b['paired_control_body_id']}`" if b["paired_control_body_id"]
                               else "shared by " + ", ".join(f"`{x}`" for x in b["styled_variant_body_ids"]))
                    L.append(f"| {b['condition']} | {marker} | {fam} | {_md(b['body'])} | {control} |")
                L += ["", "Body ids and their three stimuli (one per opening):", ""]
                for b in group:
                    L.append(f"- `{b['body_id']}` → " + ", ".join(f"`{s}`" for s in stimuli_of[b["body_id"]]))
                L.append("")
        files[f"decisions/{d}.md"] = "\n".join(L) + "\n"

    # -- markers -------------------------------------------------------------------
    for mid, m in markers.items():
        rows = [b for b in built.bodies if b["marker_id"] == mid]
        rs = [b for b in rows if b["condition"] == "RS"]
        L = head + [f"## Marker `{mid}` — “{m.string}”\n",
                    f"- family `{m.family}`, subtype `{m.subtype}`, realization `{m.realization_id}`",
                    f"- prefix before the endorsement: `{m.prefix}`",
                    f"- groups: {len(rs)}; decisions: {len({b['decision_id'] for b in rs})}; "
                    f"styled stimuli: {len(rows) * 3}", "",
                    "| decision | domain | scenario | option | RS body | NS body | shared RP / NP |",
                    "|---|---|---|---|---|---|---|"]
        for b in sorted(rs, key=lambda x: (x["scenario_id"], x["supported_option"])):
            ns = next(x for x in rows if x["condition"] == "NS" and (x["scenario_id"], x["supported_option"])
                      == (b["scenario_id"], b["supported_option"]))
            L.append(f"| {b['decision_id']} | {b['domain']} | {b['scenario_id']} | {b['supported_option']} | "
                     f"`{b['body_id']}` | `{ns['body_id']}` | `{b['paired_control_body_id']}` / "
                     f"`{ns['paired_control_body_id']}` |")
        files[f"markers/{mid}.md"] = "\n".join(L) + "\n"

    # -- families, openings, conditions ------------------------------------------
    L = head + ["## Marker families\n"]
    for family in FAMILIES:
        L += [f"### `{family}`\n", spec.raw["families"][family]["description"], "",
              "| marker | string | subtype | prefix |", "|---|---|---|---|"]
        for mid in spec.family_markers(family):
            m = markers[mid]
            L.append(f"| [{mid}](markers/{mid}.md) | {m.string} | {m.subtype} | `{m.prefix}` |")
        L.append(f"\nStyled stimuli in this family: {c['by_family'][family]}.\n")
    files["families.md"] = "\n".join(L) + "\n"

    example = built.bodies[0]
    L = head + ["## Openings (an explicit factor)\n", "| id | text | stimuli |", "|---|---|---|"]
    for oid, text in spec.openings.items():
        L.append(f"| `{oid}` | {text} | {c['by_opening'][oid]} |")
    L += ["", f"Every body is rendered once per opening; the text after the opening is identical. "
          f"Example, body `{example['body_id']}`:", ""]
    for s in built.stimuli:
        if s["body_id"] == example["body_id"]:
            L += [f"- `{s['opening_id']}`:", "", "```text", s["rendered"], "```", ""]
    files["openings.md"] = "\n".join(L) + "\n"

    L = head + ["## Conditions\n", "| condition | reason | style | stimuli | unique bodies |",
                "|---|---|---|---|---|"]
    body_counts = Counter(b["condition"] for b in built.bodies)
    for cond, reason, style in (("RS", "present", "explicit"), ("NS", "absent", "explicit"),
                                ("RP", "present", "plain"), ("NP", "absent", "plain")):
        L.append(f"| {cond} | {reason} | {style} | {c['by_condition'][cond]} | {body_counts[cond]} |")
    L += ["", manifest["plain_controls"],
          "", "Pairing: every RS marker variant → its group's single RP; every NS marker variant → "
          "its group's single NP. Removing the marker prefix from a styled body reproduces its "
          "paired plain body exactly (machine-checked for all 960 styled bodies)."]
    files["conditions.md"] = "\n".join(L) + "\n"
    files["all_decisions.md"] = _all_decisions(spec, built, head)

    units = manifest["annotation_units"]
    L = head + ["## Annotation units\n", manifest["annotation_note"], ""]
    for level, info in units.items():
        L += [f"### {level} — {info['units']} units\n", "| rating | units rated |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in info["ratings"].items()]
        for extra in ("body_level_relationships", "ratings_proposed", "inherited_ratings"):
            if extra in info:
                L.append(f"\n- {extra}: {info[extra]}")
        L.append("")
    files["annotation.md"] = "\n".join(L) + "\n"

    rel = manifest["reliability_proposal"]
    L = head + ["## Proposed reliability sample — NOT approved, annotation not begun\n",
                f"- status `{rel['status']}`, seed `{rel['seed']}`, reuses the v2 pin: "
                f"{rel['reuses_v2_pin']}",
                "- siblings: same `decision_id` (same body under other openings, the two marker "
                "variants sharing a control, the opposite supported-option group, the other "
                "scenario variant)",
                "- separation: siblings at least 20 positions apart, meaning at least 19 "
                "intervening items", "", "| sample | size | annotator | min all-pairs distance | "
                "satisfied | method |", "|---|---|---|---|---|---|"]
    for name, size in rel["sizes"].items():
        for annotator, r in rel["separation"][name].items():
            L.append(f"| {name} | {size} | {annotator} | {r['achieved_all_pairs']} | "
                     f"{r['satisfied']} | {r['method']} |")
    L += ["", "### Coverage\n", "```json", json.dumps(rel["coverage"], indent=1, sort_keys=True),
          "```"]
    files["reliability_proposal.md"] = "\n".join(L) + "\n"
    return files


def write_review(files: dict[str, str], root: str | Path) -> Path:
    root = Path(root)
    for name, text in sorted(files.items()):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / "MANIFEST.json").write_text(json.dumps(
        {"files": {n: sha256_of(t) for n, t in sorted(files.items())}}, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    return root


def _all_decisions(spec: Spec, built: Build, head: list[str]) -> str:
    """Every scenario, group, body and rendered stimulus in one file.

    Bodies are listed in tables; each rendered counterargument is the only
    ``text`` code block, so the file holds exactly one block per stimulus."""
    stimuli_of: dict[str, list[dict]] = defaultdict(list)
    for s in built.stimuli:
        stimuli_of[s["body_id"]].append(s)
    order = list(spec.openings)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for b in built.bodies:
        groups[(b["decision_id"], b["scenario_id"], b["supported_option"])].append(b)
    L = head + [f"## All decisions — {len(built.stimuli)} rendered counterarguments\n",
                "Order: decision → scenario → supported option → body (RS and NS per marker, "
                "then the shared RP and NP) → opening. Each rendered counterargument appears "
                "exactly once, in a `text` block, with its ids.\n"]
    for decision_id in sorted({k[0] for k in groups}):
        first = next(b for k, bs in groups.items() if k[0] == decision_id for b in bs)
        L.append(f"\n# Decision `{decision_id}` — {first['domain']}\n")
        for scenario_id in sorted({k[1] for k in groups if k[0] == decision_id}):
            sb = groups[(decision_id, scenario_id, "opt_1")][0]
            L += [f"## Scenario `{scenario_id}` (variant {sb['variant_id']})\n",
                  f"> {sb['scenario_text']}\n",
                  f"- **`opt_1`** — {sb['options']['opt_1']}",
                  f"- **`opt_2`** — {sb['options']['opt_2']}", ""]
            for option in ("opt_1", "opt_2"):
                bs = groups[(decision_id, scenario_id, option)]
                styled = [b for b in bs if b["marker_id"] != "none"]
                markers = sorted({(b["marker_family"], b["marker_id"], b["marker_string"]) for b in styled})
                L += [f"### `{scenario_id}` · supported option `{option}` — "
                      f"{sb['options'][option]}\n",
                      "Allocated markers: " + "; ".join(
                          f"`{mid}` “{string}” ({family})" for family, mid, string in markers),
                      "", "| body id | condition | marker | body | paired control body |",
                      "|---|---|---|---|---|"]
                for b in bs:
                    control = (f"`{b['paired_control_body_id']}`" if b["paired_control_body_id"]
                               else "shared by " + ", ".join(
                                   f"`{x}`" for x in b["styled_variant_body_ids"]))
                    L.append(f"| `{b['body_id']}` | {b['condition']} | "
                             f"{b['marker_string'] or '—'} | {_md(b['body'])} | {control} |")
                L.append("")
                for b in bs:
                    for st in sorted(stimuli_of[b["body_id"]], key=lambda x: order.index(x["opening_id"])):
                        L += [f"- `{st['stimulus_id']}` · condition **{st['condition']}** · marker "
                              f"`{st['marker_id']}` · opening `{st['opening_id']}` · paired control "
                              f"{'`' + st['paired_control_stimulus_id'] + '`' if st['paired_control_stimulus_id'] else '— (plain control)'}",
                              "", "```text", st["rendered"], "```", ""]
    return "\n".join(L) + "\n"
