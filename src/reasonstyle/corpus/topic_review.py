"""Readable review export for the topic bank.

The same guarantees as the corpus review: a read-only Markdown view, regenerated
deterministically from the canonical YAML, with no timestamps. Every page records
the hashes of the topic bank, the source registry and the configuration it was
built from. Curation judgements are recorded in the topic YAML's ``curation``
block, never in these pages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from ..hashing import content_hash, sha256_of
from .schemas import SEMANTIC_OPTIONS
from .sources import SourceRegistry
from .topics import CURATION_JUDGEMENTS, TopicBank, TopicBrief, TopicReport, sorted_topics

__all__ = ["TopicExport", "build_topic_export"]

_JUDGEMENT_LABELS = {
    "underdetermined": "Normatively underdetermined — neither option clearly superior, no weighting given",
    "can_be_made_self_contained": "Can be made self-contained — the scenario can supply everything needed",
    "no_party_politician_or_identity_framing":
        "No party appeal, politician, stereotype, personalised identity appeal or argument that asks for "
        "agreement because of a group identity — neutrally describing who bears a policy's costs or "
        "benefits (residents, tenants, households) is allowed where the trade-off requires it",
    "options_feasible_and_non_dominated": "Both options feasible and non-dominated as options",
    "each_fact_supports_its_option": "Each scenario fact is a genuine consideration in favour of its assigned option",
    "both_goals_represented_in_every_variant": "Both competing goals are represented in every variant — no other consideration silently replaces them",
    "no_option_dominates_given_the_facts": "Neither option clearly dominates once all the facts are considered",
    "variants_differ_substantively": "The variants differ in a substantive fact or setting, not only in wording",
    "no_source_wording_in_brief": "No source wording in the brief",
}


@dataclass(frozen=True)
class TopicExport:
    files: dict[str, str]
    manifest: dict[str, Any]

    def write(self, root: str | Path) -> Path:
        root = Path(root)
        for name, text in sorted(self.files.items()):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        (root / "MANIFEST.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return root


def _header(title: str, hashes: dict[str, str], source: Path) -> str:
    return "\n".join([
        f"# {title}\n",
        f"> **Topic bank** `{hashes['topics'][:12]}` · **Registry** `{hashes['registry'][:12]}` · "
        f"**Config** {hashes['config_version']} `{hashes['config'][:12]}`",
        f"> Source `{source.as_posix()}`",
        "> **Generated file — read only.** Regenerate with `uv run python scripts/export_for_review.py`.",
        "> Curation judgements belong in the topic YAML's `curation` block. Nothing written here is read back.",
        "",
    ])


def _yes_no(value: bool | None) -> str:
    return {True: "yes", False: "**no**", None: "— *outstanding*"}[value]


def _curation_status(topic: TopicBrief) -> str:
    recorded = topic.curation.recorded()
    done = sum(v is not None for v in recorded.values())
    if topic.status == "curated":
        return f"curated by {topic.curation.curated_by} on {topic.curation.curated_at}"
    if topic.status == "rejected":
        return "rejected"
    return f"{done}/{len(CURATION_JUDGEMENTS)} judgements recorded"


def _machine_status(report: TopicReport, decision_id: str) -> str:
    own = report.for_decision(decision_id)
    e = sum(f.severity == "error" for f in own)
    w = sum(f.severity == "warning" for f in own)
    if not e and not w:
        return "no findings"
    return ", ".join(p for p in (f"**{e} error(s)**" if e else "", f"{w} warning(s)" if w else "") if p)


def _decision_page(topic: TopicBrief, report: TopicReport, registry: SourceRegistry,
                   hashes: dict[str, str], source: Path) -> str:
    L = [_header(f"`{topic.decision_id}` — {topic.domain} · {topic.status}", hashes, source)]
    L.append(f"**Machine status** {_machine_status(report, topic.decision_id)}  ·  "
             f"**Curation** {_curation_status(topic)}\n")
    L.append(_screen_line(report))
    L.append(f"**Brief prepared with assistance:** {'yes' if topic.brief_prepared_with_assistance else 'no'}"
             " — the human curator edits and approves every brief.\n")

    L += ["## Decision\n", f"> {topic.decision_framing}\n",
          "## Options and the goals they prioritise\n",
          "| | Option | Prioritises |", "|---|---|---|"]
    for opt in SEMANTIC_OPTIONS:
        L.append(f"| `{opt}` | {topic.options[opt]} | {topic.competing_goals[opt]} |")
    L += ["", f"**Why neither is clearly better:** {topic.why_underdetermined}\n"]

    L.append("## Variants\n")
    L.append("*Same decision, options and trade-off. Each variant supports both options with the "
             "same number of facts, those facts instantiate the two competing goals above, and the "
             "variants differ in more than their context sentence. The scenario states both of a "
             "variant's facts before the model's initial answer; a later counterargument may build "
             "a substantive justification on one of them, but may use only facts from its own "
             "variant and may never introduce a new factual claim.*\n")
    L.append("*Facts should be comparably concrete on both sides: a definite consequence for one "
             "option is not paired with a vague or speculative one for the other.*\n")
    for vid in sorted(topic.variants):
        variant = topic.variants[vid]
        f1, f2 = variant.scenario_facts.opt_1, variant.scenario_facts.opt_2
        L += [f"### `{vid}` — {variant.context}\n",
              f"| Supports `opt_1` ({len(f1)}) | Supports `opt_2` ({len(f2)}) |", "|---|---|"]
        for i in range(max(len(f1), len(f2))):
            left = f1[i] if i < len(f1) else ""
            right = f2[i] if i < len(f2) else ""
            L.append(f"| {left} | {right} |")
        L.append("")

    L.append("## Sources (topic-idea level)\n")
    if not topic.source_references:
        L.append("*None — a constructed topic that cites no source.*\n")
    else:
        L += ["| Key | Source | Registry outcome | Licence | Locator | Inspiration (curator's words) |",
              "|---|---|---|---|---|---|"]
        for ref in topic.source_references:
            entry = registry.datasets.get(ref.key)
            name = entry.canonical_name if entry else "*not in registry*"
            outcome = entry.outcome if entry else "—"
            licence = entry.licence if entry and entry.licence else "—"
            L.append(f"| `{ref.key}` | {name} | {outcome} | {licence} | {ref.locator} | "
                     f"{ref.inspiration_summary} |")
        L.append("\n*Only the inspiration summary above — never source text — enters the generation "
                 "request. Independence from the sources is procedural.*\n")

    L.append("## Machine findings\n")
    own = report.for_decision(topic.decision_id)
    L += ([f"- **{f.severity}** `{f.code}` — {f.message}" for f in own]
          or ["*No machine errors or warnings.*"])

    L += ["", "## Curation judgements\n", "| Judgement | Recorded |", "|---|---|"]
    for name, value in topic.curation.recorded().items():
        L.append(f"| {_JUDGEMENT_LABELS[name]} | {_yes_no(value)} |")
    if topic.curation.rejection_reason:
        L.append(f"\n**Rejection reason:** {topic.curation.rejection_reason}")
    if topic.notes:
        L.append(f"\n**Notes:** {topic.notes}")
    return "\n".join(L) + "\n"


def _screen_line(report: TopicReport) -> str:
    if report.overlap_screened_documents:
        return (f"**Overlap screen:** every brief was compared with "
                f"{report.overlap_screened_documents} downloaded source documents; shared runs of six "
                f"or more words are flagged below. This screens for accidental copying only — the "
                f"curator's review remains authoritative.\n")
    return "**Overlap screen: NOT RUN** for this export.\n"


def _index_page(bank: TopicBank, report: TopicReport, registry: SourceRegistry,
                hashes: dict[str, str], source: Path) -> str:
    L = [_header("Topic bank — review index", hashes, source), _screen_line(report)]
    counts = " · ".join(f"{d} {n}/{report.required_per_domain}"
                        for d, n in sorted(report.curated_per_domain.items()))
    verdict = ("**ready for drafting**" if report.ready_for_drafting
               else "not yet ready for drafting")
    L.append(f"**Curated per domain:** {counts} — {verdict}. "
             f"Only curated topics count; proposed and rejected candidates do not.\n")
    for problem in report.readiness_problems():
        L.append(f"- {problem}")
    L += ["", "| Decision | Domain | Status | Goals in tension | Sources | Machine | Curation | File |",
          "|---|---|---|---|---|---|---|---|"]
    for topic in sorted_topics(bank.topics):
        goals = f"{topic.competing_goals['opt_1']} *vs* {topic.competing_goals['opt_2']}"
        srcs = ", ".join(f"`{s.key}`" for s in topic.source_references) or "constructed"
        L.append(f"| `{topic.decision_id}` | {topic.domain} | {topic.status} | {goals} | {srcs} | "
                 f"{_machine_status(report, topic.decision_id)} | {_curation_status(topic)} | "
                 f"[{topic.decision_id}.md]({topic.decision_id}.md) |")
    L += ["", "- [All topics, combined and searchable](all_topics.md)", ""]
    return "\n".join(L) + "\n"


def build_topic_export(bank: TopicBank, report: TopicReport, cfg: ExperimentConfig,
                       registry: SourceRegistry, source: str | Path) -> TopicExport:
    source = Path(source)
    hashes = {
        "topics": content_hash(bank.model_dump(mode="json")),
        "registry": content_hash(registry.model_dump(mode="json")),
        "config": cfg.content_hash,
        "config_version": cfg.config_version,
    }
    files: dict[str, str] = {}
    pages = {}
    statuses = {}
    for topic in sorted_topics(bank.topics):
        pages[topic.decision_id] = _decision_page(topic, report, registry, hashes, source)
        statuses[topic.decision_id] = topic.status
        files[f"{topic.decision_id}.md"] = pages[topic.decision_id]
    files["index.md"] = _index_page(bank, report, registry, hashes, source)

    combined = [_header("All topics — combined searchable view", hashes, source), "## Contents\n"]
    combined += [f"- [`{d}`](#{d.replace('_', '-')}) — {statuses[d]}" for d in pages]
    for decision_id, page in pages.items():
        body = page.split("\n", 1)[1]
        combined += [f"\n<a id=\"{decision_id.replace('_', '-')}\"></a>", body]
    files["all_topics.md"] = "\n".join(combined) + "\n"

    manifest = {
        "topic_bank_source": source.as_posix(),
        "topic_bank_content_hash": hashes["topics"],
        "registry_content_hash": hashes["registry"],
        "config_version": cfg.config_version,
        "config_content_hash": cfg.content_hash,
        "n_topics": len(bank.topics),
        "curated_per_domain": report.curated_per_domain,
        "ready_for_drafting": report.ready_for_drafting,
        "machine_errors": len(report.errors),
        "machine_warnings": len(report.warnings),
        "overlap_screened_documents": report.overlap_screened_documents,
        "files": {name: sha256_of(text) for name, text in sorted(files.items())},
    }
    return TopicExport(files=files, manifest=manifest)
