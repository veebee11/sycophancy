"""Deterministic human-review export.

A read-only Markdown view of the canonical JSONL corpus, regenerable
byte-for-byte. It exists so the corpus can be read and verified without
inspecting raw JSONL, and it is emphatically **not** a second source of truth:
judgements are recorded in ``data/annotations/``, never in these files.

Determinism. No generated file, ``MANIFEST.json`` included, carries a
wall-clock timestamp — that would make the export differ from one day to the
next and defeat the whole point of being able to confirm you reviewed the exact
corpus used in the experiment. The only time value permitted is
``corpus.freeze_timestamp``, which is itself frozen configuration.

Two audiences.

*Curator view* — labelled and complete. RS, RP, NS and NP are shown plainly with
marker metadata, measurements, machine findings and the outstanding human-review
codes, so every experimental cell can be verified.

*Blinded packets* — for the independent reliability annotators only. Produced at
all three annotation levels, with condition, marker metadata, measurements,
sibling cells and the intended supported option hidden. The unblinding key is
written to a separate directory that is never handed to an annotator.

Nothing here is specific to any one corpus: the exporter groups whatever
scenarios it is given by ``decision_id``.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .annotations import (
    BlindItemKey,
    BlindPairKey,
    BlindScenarioKey,
    ItemAnnotation,
    PairAnnotation,
    ScenarioAnnotation,
    load_annotations,
)
from ..config import ExperimentConfig
from .store import corpus_content_hash
from .findings import ValidationReport
from ..hashing import canonical_json, file_sha256, sha256_of
from ..prompting.render import (
    RenderingError,
    branches_share_prefix,
    option_orders,
    render_branches,
    render_initial,
)
from .schemas import CORE_CONDITIONS, SEMANTIC_OPTIONS, ScenarioRecord
from .segmentation import Segmenter
from .validate import HUMAN_REVIEW_CODES

__all__ = [
    "ReviewExport",
    "transcript_appendix",
    "BalanceResult",
    "SampleResult",
    "SeparationResult",
    "build_review_export",
    "order_with_separation",
    "stratified_sample",
    "strip_highlighting",
]

CONDITION_GLOSS = {
    "RS": "reason present · explicit style",
    "RP": "reason present · plain",
    "NS": "no reason · explicit style",
    "NP": "no reason · plain",
}
COMPARISONS = (
    ("RS", "RP", "style_with_reason", "style effect when a reason is present"),
    ("NS", "NP", "style_without_reason", "style effect with no reason — the primary contrast"),
    ("RS", "NS", "content_with_style", "content effect with the same marker realization"),
    ("RP", "NP", "content_plain", "content effect in plain language"),
)
PAIRS = (("RS", "RP", "RS_RP"), ("NS", "NP", "NS_NP"))

_READ_ONLY = (
    "> **Generated file — read only.** Regenerate with "
    "`uv run python scripts/export_for_review.py`.\n"
    "> Every judgement belongs in `data/annotations/`. Nothing written here is read back."
)


@dataclass(frozen=True)
class SeparationResult:
    """Whether sibling cells could be held apart in a blinded packet."""

    requested: int
    achieved: int | None            # None when the packet contains no sibling pair
    satisfied: bool
    n_sibling_pairs: int

    def note(self) -> str:
        if self.n_sibling_pairs == 0:
            return (f"No two sampled items come from the same group, so the "
                    f"minimum sibling separation of {self.requested} is trivially met.")
        if self.satisfied:
            return (f"Minimum sibling separation {self.requested} met "
                    f"(achieved {self.achieved} across {self.n_sibling_pairs} sibling pair(s)).")
        return (f"**Minimum sibling separation of {self.requested} is INFEASIBLE for this "
                f"sample**: {self.n_sibling_pairs} sibling pair(s) among too few items. "
                f"Best achievable separation is {self.achieved}. The constraint was reported, "
                f"not relaxed — reduce it deliberately or enlarge the sample.")


@dataclass(frozen=True)
class ReviewExport:
    """Every generated file, as text, plus the manifest."""

    files: dict[str, str]
    manifest: dict[str, Any]
    separation: dict[str, SeparationResult]
    balance: dict[str, BalanceResult]

    def write(self, root: str | Path) -> Path:
        root = Path(root)
        for name, text in sorted(self.files.items()):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        (root / "MANIFEST.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return root


# --------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------


def highlight_markers(text: str, marker: str) -> str:
    """Bold a marker. Display only — the canonical text is never modified."""
    return re.sub(rf"\b({re.escape(marker)})\b", r"**\1**", text, flags=re.IGNORECASE)


def strip_highlighting(text: str) -> str:
    """Inverse of :func:`highlight_markers`, so the guarantee is testable."""
    return text.replace("**", "")


def word_diff(a: str, b: str, label_a: str, label_b: str) -> str:
    """A compact, deterministic word-level diff. Shared runs are elided."""
    aw, bw = a.split(), b.split()
    out: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, aw, bw, autojunk=False).get_opcodes():
        if tag == "equal":
            run = aw[i1:i2]
            out.append(" ".join(run) if len(run) <= 6
                       else f"{' '.join(run[:3])} … {' '.join(run[-3:])}")
        else:
            left = " ".join(aw[i1:i2]) or "∅"
            right = " ".join(bw[j1:j2]) or "∅"
            out.append(f"「{label_a}: {left} ⁞ {label_b}: {right}」")
    return " ".join(out)


def _provenance_block(cfg: ExperimentConfig, corpus_hash: str, source: Path,
                      segmenter: Segmenter) -> str:
    freeze = cfg.raw["corpus"].get("freeze_timestamp")
    lines = [
        f"> **Corpus** `{corpus_hash[:12]}` · **Config** {cfg.config_version} "
        f"`{cfg.content_hash[:12]}` · **Segmenter** {segmenter.info.library} "
        f"{segmenter.info.version}",
        f"> Source `{source.as_posix()}`",
    ]
    lines.append(f"> Corpus frozen at {freeze}" if freeze else "> Corpus not yet frozen")
    lines.append(_READ_ONLY)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Curator view
# --------------------------------------------------------------------------


def _human_review_lines(report: ValidationReport, scenario_id: str, option: str) -> list[str]:
    grouped: defaultdict[str, set[str]] = defaultdict(set)
    for f in report.human_review:
        if f.scenario_id != scenario_id or f.supported_option != option:
            continue
        where = f.condition or ("/".join(f.detail["pair"]) if "pair" in f.detail else f.scope)
        grouped[f.code].add(where)
    return [f"- `{code}` ({', '.join(sorted(where))}) — {HUMAN_REVIEW_CODES[code]}"
            for code, where in sorted(grouped.items())]


def _findings_lines(report: ValidationReport, scenario_id: str, option: str | None) -> list[str]:
    rows = [f for f in report.errors + report.warnings
            if f.scenario_id == scenario_id and (option is None or f.supported_option == option)]
    if not rows:
        return ["*No machine errors or warnings.*"]
    return [f"- **{f.severity}** `{f.code}` ({f.condition or f.scope}) — {f.message}"
            for f in rows]


def _comparison_lines(record: ScenarioRecord, option: str) -> list[str]:
    block = record.counterarguments[option]
    L = ["\n#### Comparison views\n"]
    for first, second, name, gloss in COMPARISONS:
        L.append(f"**{first} − {second}** · `{name}` — {gloss}\n")
        L.append(f"> {word_diff(block.cells[first].body, block.cells[second].body, first, second)}\n")
    return L


def transcript_appendix(record: ScenarioRecord, cfg: ExperimentConfig,
                       template_name: str = "base_scaffold_v1") -> list[str]:
    """The canonical, model-independent transcript for one scenario.

    Presented so the two invariances can be checked by eye: turn 1 depends only
    on the display order, turn 3 only on the initial choice, and turns 1-2 are
    shared byte-for-byte by all four branches.
    """
    try:
        orders = option_orders(cfg)
        initials = [render_initial(record, o, cfg, template_name) for o in orders]
    except RenderingError:
        return []      # config predates the frozen question; nothing to show

    L = [f"\n### Canonical transcript — `{record.scenario_id}`\n",
         "> **This is the canonical transcript, not the exact model input.** It is an "
         "ordered list of turns and is independent of any model. The string a model "
         "actually receives is produced later: for an instruction-tuned model by that "
         "model's own tokenizer chat template, and the answer continuation after the cue "
         "(`\" A\"` versus `\"A\"`) is settled at the model-compatibility stage. No model "
         "or tokenizer has been loaded.\n",
         f"Answer cue: `{initials[0].transcript.answer_cue}` — the answer is the single "
         f"next token after it.\n"]

    L.append("\n**Turn 1 · user** — depends only on the display order\n")
    for order, initial in zip(orders, initials):
        mapping = ", ".join(f"{k} = `{v}`" for k, v in sorted(order.label_to_option.items()))
        L += [f"*Order `{order.order_id}` — {mapping}*\n",
              "```text\n" + initial.transcript.turns[0].content + "\n```\n"]

    L.append("**Turn 2 · assistant** — the model's own initial answer, as a display label\n")
    L += ["| Initial semantic choice | label under `o1` | label under `o2` |", "|---|---|---|"]
    for option in SEMANTIC_OPTIONS:
        L.append(f"| `{option}` | {orders[0].label_for(option)} | {orders[1].label_for(option)} |")

    L.append("\n**Turn 3 · user** — depends only on the initial choice, never on the order\n")
    for initial_option in SEMANTIC_OPTIONS:
        target = record.opposing_option(initial_option)
        branches = {o.order_id: render_branches(record, o, cfg, template_name, initial_option)
                    for o in orders}
        prefixes = {oid: bs[0].shared_prefix_hash for oid, bs in branches.items()}
        L += [f"\n*If the model initially chooses `{initial_option}`, all four branches "
              f"counter with the counterarguments supporting `{target}`.*\n",
              "| Order | shared prefix (turns 1–2) | identical across the four branches |",
              "|---|---|---|"]
        for oid, bs in branches.items():
            L.append(f"| `{oid}` | `{prefixes[oid][:12]}` | "
                     f"{'yes' if branches_share_prefix(bs) else '**no**'} |")
        L.append("")
        for branch in branches[orders[0].order_id]:
            L += [f"**{branch.condition}** — {CONDITION_GLOSS[branch.condition]} "
                  f"(prompt hash `{branch.prompt_hash[:12]}`)\n",
                  "```text\n" + branch.transcript.turns[2].content + "\n```\n"]
    return L


def decision_markdown(decision_id: str, records: Sequence[ScenarioRecord],
                      report: ValidationReport, cfg: ExperimentConfig,
                      corpus_hash: str, source: Path, segmenter: Segmenter,
                      recorded: dict[str, int]) -> str:
    ordered = sorted(records, key=lambda r: r.variant_id)
    first = ordered[0]
    errors = [f for f in report.errors if f.decision_id == decision_id]
    warnings = [f for f in report.warnings if f.decision_id == decision_id]
    outstanding = [f for f in report.human_review if f.decision_id == decision_id]

    L = [f"# `{decision_id}` — {first.domain}\n",
         _provenance_block(cfg, corpus_hash, source, segmenter), "",
         f"**Machine status** {len(errors)} error(s), {len(warnings)} warning(s)  ·  "
         f"**Human review** {len(outstanding)} outstanding, "
         f"{recorded.get(decision_id, 0)} recorded\n",
         "## Semantic options (immutable; identical across both variants)\n",
         "| ID | Option |", "|---|---|"]
    L += [f"| `{k}` | {first.options[k]} |" for k in SEMANTIC_OPTIONS]
    L += ["\n## Shared counterargument opening\n", f"> {first.counterargument_opening}\n",
          "Prepended verbatim to all eight counterarguments of each variant, so the four "
          "cells of a group cannot differ in their opening.\n"]

    for record in ordered:
        L += [f"\n---\n\n## Variant {record.variant_id} — `{record.scenario_id}`\n",
              "### Scenario\n", f"> {record.scenario_text}\n"]
        for option in SEMANTIC_OPTIONS:
            block = record.counterarguments[option]
            L += [f"\n### Counterarguments supporting `{option}` — {record.options[option]}\n",
                  f"**Marker family** `{block.marker_family}` · **marker string** "
                  f"“{block.marker_string}” · **realization** `{block.marker_realization_id}`\n",
                  "| Cell | Reason | Style | Words body/full | Sentences body/full |",
                  "|---|---|---|---|---|"]
            for condition in CORE_CONDITIONS:
                m = block.cells[condition].measurements
                wc = f"{m.word_count_body} / {m.word_count_full}" if m else "—"
                sc = f"{m.sentence_count_body} / {m.sentence_count_full}" if m else "—"
                L.append(f"| **{condition}** | {'present' if condition[0] == 'R' else 'absent'} "
                         f"| {'explicit' if condition[1] == 'S' else 'plain'} | {wc} | {sc} |")

            L += ["\n#### Verbatim counterargument bodies\n",
                  "*Canonical experimental text, exactly as stored.*\n"]
            for condition in CORE_CONDITIONS:
                L += [f"**{condition}** — {CONDITION_GLOSS[condition]}\n",
                      "```text\n" + block.cells[condition].body + "\n```\n"]

            L += ["#### Marker highlighting — display only\n",
                  f"*Bolding “{block.marker_string}”. The blocks above are canonical; this "
                  f"rendering is never used as experimental input.*\n"]
            L += [f"- **{c}**: {highlight_markers(block.cells[c].body, block.marker_string)}"
                  for c in CORE_CONDITIONS]
            L += _comparison_lines(record, option)
            L += ["#### Machine findings\n", *_findings_lines(report, record.scenario_id, option), ""]
            L += ["#### Human review still required\n",
                  *_human_review_lines(report, record.scenario_id, option), ""]
        L += ["#### Scenario-level machine findings\n",
              *_findings_lines(report, record.scenario_id, None), ""]

    appendix = [line for record in ordered for line in transcript_appendix(record, cfg)]
    if appendix:
        L += ["\n---\n", "## Appendix — canonical transcripts\n", *appendix]
    return "\n".join(L) + "\n"


def index_markdown(records: Sequence[ScenarioRecord], report: ValidationReport,
                   cfg: ExperimentConfig, corpus_hash: str, source: Path,
                   segmenter: Segmenter, recorded: dict[str, int]) -> str:
    by_decision: defaultdict[str, list[ScenarioRecord]] = defaultdict(list)
    for record in records:
        by_decision[record.decision_id].append(record)

    L = ["# Corpus review index\n", _provenance_block(cfg, corpus_hash, source, segmenter), "",
         "| Decision | Domain | Variants | Texts | Machine | Human review | File |",
         "|---|---|---|---|---|---|---|"]
    for decision_id in sorted(by_decision):
        group = sorted(by_decision[decision_id], key=lambda r: r.variant_id)
        errors = len([f for f in report.errors if f.decision_id == decision_id])
        warnings = len([f for f in report.warnings if f.decision_id == decision_id])
        outstanding = len([f for f in report.human_review if f.decision_id == decision_id])
        status = "machine-valid" if errors == 0 else f"**{errors} error(s)**"
        if warnings:
            status += f", {warnings} warning(s)"
        L.append(f"| `{decision_id}` | {group[0].domain} "
                 f"| {', '.join(f'v{r.variant_id}' for r in group)} "
                 f"| {sum(r.counterargument_count for r in group)} | {status} "
                 f"| {outstanding} outstanding, {recorded.get(decision_id, 0)} recorded "
                 f"| [decisions/{decision_id}.md](decisions/{decision_id}.md) |")

    L += [f"\n**Totals** — {len(by_decision)} decision(s), {len(records)} scenario(s), "
          f"{report.n_texts} counterargument texts, {len(report.errors)} error(s), "
          f"{len(report.warnings)} warning(s), {len(report.human_review)} outstanding "
          f"human judgements.\n",
          "Machine validation checks form only: presence, absence, counts, structural "
          "consistency and pattern matches. Substantive support, support direction, "
          "no-reason integrity, proposition preservation, naturalness and pragmatic "
          "commitment are human judgements and are listed per decision.\n",
          "- [All decisions, combined and searchable](all_decisions.md)",
          "- Blinded reliability packets: `blind/<annotator>/`",
          "- Unblinding key (never share with an annotator): `blind_key/`\n"]
    return "\n".join(L) + "\n"


def combined_markdown(decision_files: dict[str, str], cfg: ExperimentConfig,
                      corpus_hash: str, source: Path, segmenter: Segmenter) -> str:
    L = ["# All decisions — combined searchable view\n",
         _provenance_block(cfg, corpus_hash, source, segmenter), "", "## Contents\n"]
    ids = sorted(decision_files)
    L += [f"- [`{d}`](#{d.replace('_', '-')})" for d in ids]
    L.append("")
    for decision_id in ids:
        body = decision_files[decision_id]
        body = body.split("\n", 1)[1] if body.startswith("# ") else body
        L += [f"\n<a id=\"{decision_id.replace('_', '-')}\"></a>",
              f"\n# `{decision_id}`\n", body]
    return "\n".join(L)


# --------------------------------------------------------------------------
# Blinded reliability packets
# --------------------------------------------------------------------------


def _rng(cfg: ExperimentConfig, corpus_hash: str, salt: str) -> random.Random:
    """Seeded from config *and* the corpus hash, so the same corpus always
    yields the same packet and a changed corpus never silently reuses it."""
    seed = cfg.raw["determinism"]["seeds"][
        cfg.raw["annotation"]["reliability_subsample"]["seed_ref"]]
    digest = hashlib.sha256(f"{seed}:{corpus_hash}:{salt}".encode()).hexdigest()
    return random.Random(int(digest, 16) % (2 ** 32))


def _facets(record: ScenarioRecord, option: str, condition: str | None) -> dict[str, Any]:
    return {"domain": record.domain,
            "supported_option": option,
            "condition": condition,
            "marker_family": record.counterarguments[option].marker_family}


def item_sampling_units(records: Sequence[ScenarioRecord],
                        stratify_by: Sequence[str] = ("domain", "condition", "marker_family"),
                        balance_by: Sequence[str] = ("supported_option",),
                        ) -> list[tuple[tuple, tuple, tuple]]:
    """``(unit, stratum, balance)`` for every counterargument cell.

    The facets are read from the configuration rather than fixed here: a
    stratified sample must be able to reach every stratum it claims, and with
    38 sampled pilot items ``domain x condition x marker_family`` (36 strata) is
    what fits. ``supported_option`` is balanced as a marginal count instead.

    The marker family is the one assigned to the whole four-condition group.
    RP and NP carry no marker themselves — their cell-level ``marker_family``
    is null — so the cell field is never used.
    """
    units = []
    for r in records:
        for o in SEMANTIC_OPTIONS:
            for c in CORE_CONDITIONS:
                facets = _facets(r, o, c)
                units.append(((r.scenario_id, o, c),
                              tuple(facets[f] for f in stratify_by),
                              tuple(facets[f] for f in balance_by)))
    return units


@dataclass(frozen=True)
class BalanceResult:
    """How evenly a sample splits across the two values of its balance key.

    ``imbalance`` is ``|count(first) - count(second)|`` in the drawn sample.
    ``ideal`` is what a perfect split would give: 0 for an even sample, 1 for an
    odd one. ``best_under_stratification`` is the smallest imbalance any sample
    of the same size and the same largest-remainder stratum quotas can reach,
    and ``best_from_units`` the smallest reachable ignoring strata, from the
    unit counts alone. The sampler always attains the first; the other two say
    why an ideal split was or was not possible.
    """

    values: tuple[Any, ...]
    counts: tuple[int, ...]
    sample_size: int
    imbalance: int
    ideal: int
    best_under_stratification: int
    best_from_units: int

    @property
    def satisfied(self) -> bool:
        return self.imbalance == self.ideal

    def note(self) -> str:
        split = " / ".join(f"{_balance_label(v)} {n}" for v, n in zip(self.values, self.counts))
        if self.satisfied:
            return (f"Supported option balanced marginally: {split} "
                    f"(difference {self.imbalance}, the minimum for {self.sample_size} items).")
        cause = ("the available units" if self.best_from_units > self.ideal
                 else "the stratum quotas")
        return (f"**Supported-option balance is INFEASIBLE for this sample**: {split}. "
                f"Best achievable difference is {self.best_under_stratification} "
                f"(ideal {self.ideal}), limited by {cause}. Reported, not hidden.")

    def as_dict(self) -> dict[str, Any]:
        return {"counts": {_balance_label(v): n for v, n in zip(self.values, self.counts)},
                "sample_size": self.sample_size, "imbalance": self.imbalance,
                "ideal": self.ideal, "best_under_stratification": self.best_under_stratification,
                "best_from_units": self.best_from_units, "satisfied": self.satisfied}


def _balance_label(value: Any) -> str:
    return "/".join(map(str, value)) if isinstance(value, tuple) else str(value)


@dataclass(frozen=True)
class SampleResult:
    chosen: list[Any]
    quotas: dict[tuple, int]
    balance: BalanceResult | None      # None when the units carry no balance key


def stratified_sample(units: Sequence[tuple[Any, ...]], fraction: float,
                      rng: random.Random) -> SampleResult:
    """Proportional allocation over strata, balanced marginally on a binary key.

    **Size.** ``max(1, round(N * fraction))`` units, as before.

    **Strata.** Each stratum receives ``floor(n * fraction)`` units, and the
    remaining units go to strata in descending order of their exact fractional
    remainder (largest remainder). Remainders are computed with exact
    fractions, so float noise never splits a genuine tie. Only strata *tied*
    at the boundary remainder are interchangeable; which of them receives the
    extra unit is the one freedom the stratification leaves.

    **Balance.** Units may carry a third element, a balance key with at most two
    distinct values. Among every sample consistent with the quotas above, the
    sampler finds the smallest reachable ``|count(first) - count(second)|`` by
    exact dynamic programming over (strata, boundary picks, first-value count),
    and draws a sample attaining it. So a perfect split (0 for even, 1 for odd)
    is always reached when one exists, and otherwise the best achievable
    imbalance is reported in the result.

    **Randomness** only breaks ties between samples meeting the same objective:
    which optimal count, which boundary strata, and which units within a stratum.
    The generator is consumed in a fixed order, so the same units and seed give
    the same sample.
    """
    strata: defaultdict[tuple, list[Any]] = defaultdict(list)
    balance: dict[Any, Any] = {}
    for unit, key, *rest in units:
        strata[key].append(unit)
        if rest:
            balance[unit] = rest[0]
    if not units:
        return SampleResult([], {}, None)
    target = max(1, round(len(units) * fraction))
    keys = sorted(strata)

    exact = Fraction(str(fraction))
    floors = {k: int(len(strata[k]) * exact) for k in keys}
    remainders = {k: len(strata[k]) * exact - floors[k] for k in keys}
    extra = target - sum(floors.values())
    eligible = [k for k in keys if floors[k] < len(strata[k])]
    fixed_plus: set[tuple] = set()
    boundary: list[tuple] = []
    picks = 0
    for value in sorted({remainders[k] for k in eligible}, reverse=True):
        if extra == 0:
            break
        group = [k for k in eligible if remainders[k] == value]
        if len(group) <= extra:
            fixed_plus.update(group)
            extra -= len(group)
        else:
            boundary, picks, extra = group, extra, 0
    if extra:
        raise ValueError(f"cannot place {target} units in {len(units)}")
    in_boundary = set(boundary)
    base = {k: floors[k] + (k in fixed_plus) for k in keys}

    values = tuple(sorted(set(balance.values()), key=repr)) if balance else ()
    if len(values) > 2:
        raise ValueError(f"marginal balancing supports two values; got {len(values)}")
    first = values[0] if values else None
    n_first = {k: sum(1 for u in strata[k] if balance.get(u) == first) for k in keys}

    def ranges(k: tuple) -> list[tuple[int, int]]:
        """``(extra pick, first-value count)`` options for one stratum."""
        out = []
        for e in ((0, 1) if k in in_boundary else (0,)):
            q, n1 = base[k] + e, n_first[k]
            n2 = len(strata[k]) - n1
            if not balance:
                out.append((e, 0))
                continue
            out.extend((e, a) for a in range(max(0, q - n2), min(q, n1) + 1))
        return out

    # reach[i]: every (boundary picks, first-value count) reachable by strata[:i]
    reach: list[set[tuple[int, int]]] = [{(0, 0)}]
    for k in keys:
        opts = ranges(k)
        reach.append({(j + e, a + d) for j, a in reach[-1] for e, d in opts
                      if j + e <= picks})
    finals = sorted(a for j, a in reach[-1] if j == picks)
    best = min(abs(2 * a - target) for a in finals)
    a_star = rng.choice([a for a in finals if abs(2 * a - target) == best])

    quotas, take_first = {}, {}
    j, a = picks, a_star
    for i in range(len(keys) - 1, -1, -1):
        k = keys[i]
        opts = ranges(k)
        rng.shuffle(opts)
        e, d = next((e, d) for e, d in opts if (j - e, a - d) in reach[i])
        quotas[k], take_first[k] = base[k] + e, d
        j, a = j - e, a - d

    chosen: list[Any] = []
    for k in keys:
        pool = list(strata[k])
        rng.shuffle(pool)
        if not balance:
            chosen.extend(pool[:quotas[k]])
            continue
        ones = [u for u in pool if balance[u] == first]
        others = [u for u in pool if balance[u] != first]
        chosen.extend(ones[:take_first[k]] + others[:quotas[k] - take_first[k]])

    if not balance:
        return SampleResult(chosen, quotas, None)
    count_first = sum(1 for u in chosen if balance[u] == first)
    counts = (count_first, target - count_first) if len(values) == 2 else (count_first,)
    total_first = sum(n_first.values())
    lo, hi = max(0, target - (len(units) - total_first)), min(target, total_first)
    from_units = min(abs(2 * a - target) for a in range(lo, hi + 1))
    result = BalanceResult(values=values, counts=counts, sample_size=len(chosen),
                           imbalance=abs(2 * count_first - target), ideal=target % 2,
                           best_under_stratification=best, best_from_units=from_units)
    assert len(chosen) == target and result.imbalance == best
    return SampleResult(chosen, quotas, result)


def _stratified_sample(units: Sequence[tuple[Any, ...]], fraction: float,
                       rng: random.Random) -> list[Any]:
    """The sampled units only; see :func:`stratified_sample`."""
    return stratified_sample(units, fraction, rng).chosen


def order_with_separation(items: Sequence[Any], sibling_key, minimum: int,
                           rng: random.Random, attempts: int = 2000
                           ) -> tuple[list[Any], SeparationResult]:
    """Shuffle so that items sharing a ``sibling_key`` stay ``minimum`` apart.

    If the constraint cannot be met, the best achievable arrangement is returned
    together with a report saying so. It is never silently relaxed.
    """
    groups: defaultdict[Any, list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        groups[sibling_key(item)].append(i)
    n_pairs = sum(len(v) * (len(v) - 1) // 2 for v in groups.values())

    if n_pairs == 0:
        order = list(items)
        rng.shuffle(order)
        return order, SeparationResult(minimum, None, True, 0)

    def min_gap(order: Sequence[Any]) -> int:
        positions: defaultdict[Any, list[int]] = defaultdict(list)
        for i, item in enumerate(order):
            positions[sibling_key(item)].append(i)
        gaps = [b - a for pos in positions.values()
                for a, b in zip(sorted(pos), sorted(pos)[1:])]
        return min(gaps) if gaps else minimum

    best, best_gap = None, -1
    for _ in range(attempts):
        candidate = list(items)
        rng.shuffle(candidate)
        gap = min_gap(candidate)
        if gap > best_gap:
            best, best_gap = candidate, gap
        if gap >= minimum:
            return candidate, SeparationResult(minimum, gap, True, n_pairs)
    assert best is not None
    return best, SeparationResult(minimum, best_gap, False, n_pairs)


def _blind_item_packet(records, cfg, corpus_hash, source, segmenter, annotator,
                       chosen, keys, separation) -> str:
    key_by_id = {k.blind_id: k for k in keys}
    by_scenario = {r.scenario_id: r for r in records}
    L = [f"# Blinded item review — annotator `{annotator}`\n",
         _provenance_block(cfg, corpus_hash, source, segmenter), "",
         f"{len(chosen)} items in randomised order. Each item is shown alone: the "
         f"condition, the marker metadata, the counts and the other cells of its group "
         f"are hidden, and no intended answer is stated anywhere in this file.\n",
         f"*{separation.note()}*\n",
         "For each item record, in `data/annotations/`: which option the reply supports "
         "(**P**, **Q** or **unclear**), whether it supplies a premise that makes that "
         "option more desirable, and the rating scales.\n", "---"]
    for blind_id in [k.blind_id for k in chosen]:
        key = key_by_id[blind_id]
        record = by_scenario[key.scenario_id]
        L += [f"\n### Item `{blind_id}`\n", "**Scenario.**\n", f"> {record.scenario_text}\n",
              f"- **Option P** — {record.options[key.option_labels['P']]}",
              f"- **Option Q** — {record.options[key.option_labels['Q']]}\n",
              "**A reader replies:**\n",
              "```text\n" + record.render(key.supported_option, key.condition) + "\n```\n",
              "- Which option does this reply support?  P / Q / unclear",
              "- Does it supply a premise that makes that option more desirable?  yes / no",
              "- Rate 1–5: substantive support · perceived reasoning style · speaker "
              "commitment · unstated support · naturalness · confidence · pressure · "
              "politeness · authority · credibility\n"]
    return "\n".join(L) + "\n"


def _blind_pair_packet(records, cfg, corpus_hash, source, segmenter, annotator,
                       chosen, keys, separation) -> str:
    key_by_id = {k.blind_id: k for k in keys}
    by_scenario = {r.scenario_id: r for r in records}
    L = [f"# Blinded pair review — annotator `{annotator}`\n",
         _provenance_block(cfg, corpus_hash, source, segmenter), "",
         f"{len(chosen)} pairs in randomised order. Each pair shows two replies to the "
         f"same scenario, in randomised left/right order. Nothing indicates how either "
         f"text was constructed, and the sides carry no meaning.\n",
         f"*{separation.note()}*\n",
         "Judge only whether the two texts express **the same substantive claims** "
         "(1 = clearly different claims, 5 = identical claims).\n", "---"]
    for blind_id in [k.blind_id for k in chosen]:
        key = key_by_id[blind_id]
        record = by_scenario[key.scenario_id]
        L += [f"\n### Pair `{blind_id}`\n", "**Scenario.**\n", f"> {record.scenario_text}\n"]
        for side in ("1", "2"):
            body = record.counterarguments[key.supported_option].cells[key.side_conditions[side]].body
            L += [f"**Text {side}**\n", "```text\n" + body + "\n```\n"]
        L.append("- Do these express the same substantive claims?  1–5\n")
    return "\n".join(L) + "\n"


def _blind_scenario_packet(records, cfg, corpus_hash, source, segmenter, annotator,
                           chosen, keys) -> str:
    key_by_id = {k.blind_id: k for k in keys}
    by_scenario = {r.scenario_id: r for r in records}
    L = [f"# Blinded scenario review — annotator `{annotator}`\n",
         _provenance_block(cfg, corpus_hash, source, segmenter), "",
         f"{len(chosen)} scenarios in randomised order, shown **without any "
         f"counterarguments**. Judge the scenario itself.\n", "---"]
    for blind_id in [k.blind_id for k in chosen]:
        key = key_by_id[blind_id]
        record = by_scenario[key.scenario_id]
        L += [f"\n### Scenario `{blind_id}`\n", f"> {record.scenario_text}\n",
              f"- **Option P** — {record.options[key.option_labels['P']]}",
              f"- **Option Q** — {record.options[key.option_labels['Q']]}\n",
              "- Are both options feasible?  1–5",
              "- Is neither option dominated by the other?  1–5",
              "- Does the scenario leave the weighting of the competing goals open?  1–5\n"]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# Top-level build
# --------------------------------------------------------------------------


def _recorded_counts(annotations_dir: Path, records: Sequence[ScenarioRecord]) -> dict[str, int]:
    """How many judgements exist per decision, read from the annotation tables."""
    decision_of = {r.scenario_id: r.decision_id for r in records}
    counts: defaultdict[str, int] = defaultdict(int)
    for name, model in (("item.jsonl", ItemAnnotation), ("pair.jsonl", PairAnnotation),
                        ("scenario.jsonl", ScenarioAnnotation)):
        for annotation in load_annotations(annotations_dir / name, model):
            decision = decision_of.get(annotation.scenario_id)
            if decision:
                counts[decision] += 1
    return dict(counts)


def build_review_export(
    records: Sequence[ScenarioRecord],
    report: ValidationReport,
    cfg: ExperimentConfig,
    segmenter: Segmenter,
    source: str | Path,
    annotations_dir: str | Path = "data/annotations",
) -> ReviewExport:
    """Build every review file in memory. Deterministic for a given corpus."""
    source = Path(source)
    # The canonical corpus hash, taken from the validation report, which was
    # computed on the records as stored. `records` here may carry derived
    # measurements; those must never change the corpus identity, or the hash
    # printed in the review would not match the JSONL that was reviewed.
    corpus_hash = report.corpus_content_hash
    recorded = _recorded_counts(Path(annotations_dir), records)

    by_decision: defaultdict[str, list[ScenarioRecord]] = defaultdict(list)
    for record in records:
        by_decision[record.decision_id].append(record)

    files: dict[str, str] = {}
    decision_files: dict[str, str] = {}
    for decision_id in sorted(by_decision):
        text = decision_markdown(decision_id, by_decision[decision_id], report, cfg,
                                 corpus_hash, source, segmenter, recorded)
        decision_files[decision_id] = text
        files[f"decisions/{decision_id}.md"] = text

    files["index.md"] = index_markdown(records, report, cfg, corpus_hash, source,
                                       segmenter, recorded)
    files["all_decisions.md"] = combined_markdown(decision_files, cfg, corpus_hash,
                                                  source, segmenter)

    # -- blinded reliability packets ------------------------------------------
    sub = cfg.raw["annotation"]["reliability_subsample"]
    fraction, minimum = sub["fraction"], sub["min_sibling_separation"]
    n_annotators = sub["independent_annotators"]

    stratify_by, balance_by = sub["stratify_by"], sub["balance_marginally"]
    item_units = item_sampling_units(records, stratify_by, balance_by)
    item_sample = stratified_sample(item_units, fraction, _rng(cfg, corpus_hash, "item-sample"))
    item_chosen = item_sample.chosen
    item_keys: list[BlindItemKey] = []
    label_rng = _rng(cfg, corpus_hash, "item-labels")
    for i, (scenario_id, option, condition) in enumerate(sorted(item_chosen), start=1):
        options = list(SEMANTIC_OPTIONS)
        label_rng.shuffle(options)
        item_keys.append(BlindItemKey(
            blind_id=f"i{i:04d}", scenario_id=scenario_id, supported_option=option,
            condition=condition, option_labels={"P": options[0], "Q": options[1]}))

    # Pairs stratify on the pair itself instead of the condition, and balance
    # the supported option marginally in the same way.
    pair_units = [((r.scenario_id, o, pid),
                   tuple(pid if f == "condition" else _facets(r, o, None)[f]
                         for f in stratify_by),
                   tuple(_facets(r, o, None)[f] for f in balance_by))
                  for r in records for o in SEMANTIC_OPTIONS for _, _, pid in PAIRS]
    pair_sample = stratified_sample(pair_units, fraction, _rng(cfg, corpus_hash, "pair-sample"))
    pair_chosen = pair_sample.chosen
    balance = {name: result for name, result in (("item", item_sample.balance),
                                                  ("pair", pair_sample.balance))
               if result is not None}
    pair_keys: list[BlindPairKey] = []
    side_rng = _rng(cfg, corpus_hash, "pair-sides")
    for i, (scenario_id, option, pair_id) in enumerate(sorted(pair_chosen), start=1):
        conditions = [c for a, b, pid in PAIRS if pid == pair_id for c in (a, b)]
        side_rng.shuffle(conditions)
        pair_keys.append(BlindPairKey(
            blind_id=f"p{i:04d}", scenario_id=scenario_id, supported_option=option,
            pair_id=pair_id, side_conditions={"1": conditions[0], "2": conditions[1]}))

    scenario_units = [(r.scenario_id, (r.domain,)) for r in records]
    scenario_chosen = stratified_sample(scenario_units, fraction,
                                        _rng(cfg, corpus_hash, "scenario-sample")).chosen
    scenario_keys: list[BlindScenarioKey] = []
    scen_rng = _rng(cfg, corpus_hash, "scenario-labels")
    for i, scenario_id in enumerate(sorted(scenario_chosen), start=1):
        options = list(SEMANTIC_OPTIONS)
        scen_rng.shuffle(options)
        scenario_keys.append(BlindScenarioKey(
            blind_id=f"s{i:04d}", scenario_id=scenario_id,
            option_labels={"P": options[0], "Q": options[1]}))

    separation: dict[str, SeparationResult] = {}
    for n in range(1, n_annotators + 1):
        annotator = f"annotator_{n}"
        # Same items for every annotator (kappa requires it); independent order.
        items, item_sep = order_with_separation(
            item_keys, lambda k: (k.scenario_id, k.supported_option), minimum,
            _rng(cfg, corpus_hash, f"item-order-{annotator}"))
        pairs, pair_sep = order_with_separation(
            pair_keys, lambda k: (k.scenario_id, k.supported_option), minimum,
            _rng(cfg, corpus_hash, f"pair-order-{annotator}"))
        scenarios, _ = order_with_separation(
            scenario_keys, lambda k: k.scenario_id, 0,
            _rng(cfg, corpus_hash, f"scenario-order-{annotator}"))
        separation[f"{annotator}/item"] = item_sep
        separation[f"{annotator}/pair"] = pair_sep

        files[f"blind/{annotator}/item_packet.md"] = _blind_item_packet(
            records, cfg, corpus_hash, source, segmenter, annotator, items, item_keys, item_sep)
        files[f"blind/{annotator}/pair_packet.md"] = _blind_pair_packet(
            records, cfg, corpus_hash, source, segmenter, annotator, pairs, pair_keys, pair_sep)
        files[f"blind/{annotator}/scenario_packet.md"] = _blind_scenario_packet(
            records, cfg, corpus_hash, source, segmenter, annotator, scenarios, scenario_keys)

    for name, keys in (("item", item_keys), ("pair", pair_keys), ("scenario", scenario_keys)):
        files[f"blind_key/{name}_key.jsonl"] = "".join(
            canonical_json(k.model_dump(mode="json")) + "\n" for k in keys)
    files["blind_key/README.md"] = (
        "# Unblinding key — never share with an annotator\n\n"
        + _provenance_block(cfg, corpus_hash, source, segmenter) + "\n\n"
        "These files map each blind id back to its scenario, supported option and "
        "condition, and record which semantic option was displayed as P and which as Q. "
        "Handing this directory to a reliability annotator invalidates the blinding.\n"
        + "".join(f"\n**{name.capitalize()} sample.** {result.note()}\n"
                  for name, result in sorted(balance.items()))
    )

    manifest = {
        "config_version": cfg.config_version,
        "config_content_hash": cfg.content_hash,
        "corpus_content_hash": corpus_hash,
        "corpus_source": source.as_posix(),
        "corpus_file_sha256": file_sha256(source) if source.exists() else None,
        "corpus_freeze_timestamp": cfg.raw["corpus"].get("freeze_timestamp"),
        "segmenter": segmenter.info.as_dict(),
        "corpus_scope": report.corpus_scope,
        "n_decisions": len(by_decision),
        "n_scenarios": len(records),
        "n_texts": report.n_texts,
        "machine_errors": len(report.errors),
        "machine_warnings": len(report.warnings),
        "outstanding_human_review": len(report.human_review),
        "sibling_separation": {
            k: {"requested": v.requested, "achieved": v.achieved,
                "satisfied": v.satisfied, "sibling_pairs": v.n_sibling_pairs}
            for k, v in sorted(separation.items())},
        "supported_option_balance": {k: v.as_dict() for k, v in sorted(balance.items())},
        "files": {name: sha256_of(text) for name, text in sorted(files.items())},
    }
    return ReviewExport(files=files, manifest=manifest, separation=separation, balance=balance)
