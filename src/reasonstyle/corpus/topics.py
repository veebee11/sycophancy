"""The topic bank: one curated brief per underlying decision.

A brief is the only thing the generator will ever be told about a topic. It is
written — or edited and approved — by the human curator, in their own words.
Source datasets contribute only topic ideas, recorded as a registry key, a
locator and a short ``inspiration_summary``; no source wording enters the brief,
so independence from the sources is guaranteed by the procedure itself.

The semantic options and competing goals belong to the decision. Each variant
has its own context and its own **scenario facts**: constructed statements the
scenario will state, supporting each option with the same number of facts so
neither looks stronger merely because it has more said for it. An RS or RP
counterargument may later use only facts from its own variant.

Structural problems (a missing field, an unknown key) fail at load. Everything
else is reported as a finding, so one run lists every problem at once.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..config import ExperimentConfig
from .findings import Finding
from .schemas import SEMANTIC_OPTIONS, SemanticOption
from .sources import SourceRegistry

__all__ = [
    "CURATION_JUDGEMENTS",
    "Curation",
    "TopicBank",
    "TopicBrief",
    "TopicBankError",
    "TopicReport",
    "TopicSource",
    "Variant",
    "check_topics",
    "load_topic_bank",
]

NonEmpty = Annotated[str, Field(min_length=1)]
Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", min_length=3, max_length=64)]

#: The curator's topic-level judgements. All must be true before ``curated``.
#: Fact balance is judged as three separate questions: does each fact support
#: its own option, do the facts actually instantiate the declared goals (rather
#: than, say, funding or timing standing in for them), and does either option
#: dominate once every fact is considered.
CURATION_JUDGEMENTS = (
    "underdetermined",
    "can_be_made_self_contained",
    "no_party_politician_or_identity_framing",
    "options_feasible_and_non_dominated",
    "each_fact_supports_its_option",
    "both_goals_represented_in_every_variant",
    "no_option_dominates_given_the_facts",
    "variants_differ_substantively",
    "no_source_wording_in_brief",
)


class TopicBankError(ValueError):
    """The topic file is structurally malformed."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Facts(_Model):
    opt_1: list[NonEmpty]
    opt_2: list[NonEmpty]


class Variant(_Model):
    context: NonEmpty
    scenario_facts: Facts


class TopicSource(_Model):
    """A topic idea's origin. Never source text: only where it came from and
    what it suggested, described in the curator's own words."""

    key: Identifier
    locator: NonEmpty
    inspiration_summary: NonEmpty


class Curation(_Model):
    underdetermined: bool | None = None
    can_be_made_self_contained: bool | None = None
    no_party_politician_or_identity_framing: bool | None = None
    options_feasible_and_non_dominated: bool | None = None
    each_fact_supports_its_option: bool | None = None
    both_goals_represented_in_every_variant: bool | None = None
    no_option_dominates_given_the_facts: bool | None = None
    variants_differ_substantively: bool | None = None
    no_source_wording_in_brief: bool | None = None
    curated_by: str | None = None
    curated_at: date | None = None
    rejection_reason: str | None = None

    def recorded(self) -> dict[str, bool | None]:
        return {name: getattr(self, name) for name in CURATION_JUDGEMENTS}


class TopicBrief(_Model):
    decision_id: Identifier
    domain: NonEmpty
    status: Literal["proposed", "curated", "rejected"] = "proposed"
    decision_framing: NonEmpty
    options: dict[SemanticOption, NonEmpty]
    competing_goals: dict[SemanticOption, NonEmpty]
    why_underdetermined: NonEmpty
    variants: dict[str, Variant]
    source_references: list[TopicSource] = Field(default_factory=list)
    brief_prepared_with_assistance: bool
    curation: Curation = Field(default_factory=Curation)
    notes: str | None = None

    @model_validator(mode="after")
    def _both_options(self) -> TopicBrief:
        for field in ("options", "competing_goals"):
            if set(getattr(self, field)) != set(SEMANTIC_OPTIONS):
                raise ValueError(f"{field} needs exactly {list(SEMANTIC_OPTIONS)}")
        return self

    def prose_fields(self) -> list[tuple[str, str, str]]:
        """``(field, limit key, text)`` for everything with a word limit."""
        out = [("decision_framing", "decision_framing", self.decision_framing),
               ("why_underdetermined", "why_underdetermined", self.why_underdetermined)]
        for opt in SEMANTIC_OPTIONS:
            out.append((f"options.{opt}", "option", self.options[opt]))
            out.append((f"competing_goals.{opt}", "competing_goal", self.competing_goals[opt]))
        for vid, variant in sorted(self.variants.items()):
            out.append((f"variants.{vid}.context", "variant_context", variant.context))
            for opt in SEMANTIC_OPTIONS:
                for i, fact in enumerate(getattr(variant.scenario_facts, opt), start=1):
                    out.append((f"variants.{vid}.scenario_facts.{opt}[{i}]", "scenario_fact", fact))
        for i, src in enumerate(self.source_references, start=1):
            out.append((f"source_references[{i}].locator", "source_locator", src.locator))
            out.append((f"source_references[{i}].inspiration_summary", "inspiration_summary",
                        src.inspiration_summary))
        return out


class TopicBank(_Model):
    topics: list[TopicBrief]


def load_topic_bank(path: str | Path) -> TopicBank:
    path = Path(path)
    try:
        return TopicBank.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValidationError) as exc:
        raise TopicBankError(f"{path}: {exc}") from exc


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class TopicReport:
    findings: tuple[Finding, ...]
    curated_per_domain: dict[str, int]
    required_per_domain: int

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "warning")

    def for_decision(self, decision_id: str) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.decision_id == decision_id)

    def readiness_problems(self) -> list[str]:
        """Drafting needs exactly the required number of curated decisions per
        domain, and no machine errors. Proposed and rejected topics don't count."""
        problems = [f"{d}: {n} curated, {self.required_per_domain} required"
                    for d, n in sorted(self.curated_per_domain.items())
                    if n != self.required_per_domain]
        if self.errors:
            problems.append(f"{len(self.errors)} machine error(s)")
        return problems

    @property
    def ready_for_drafting(self) -> bool:
        return not self.readiness_problems()


def check_topics(bank: TopicBank, cfg: ExperimentConfig,
                 registry: SourceRegistry) -> TopicReport:
    t = cfg.raw["topics"]
    domains = list(cfg.raw["domains"]["ids"])
    variant_ids = set(t["variants"])
    lo, hi = t["facts_per_option"]["min"], t["facts_per_option"]["max"]
    limits = t["max_words"]
    framing_terms = [re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
                     for term in t["framing_warning_terms"]]
    forbidden = cfg.compiled_forbidden()
    leakage = cfg.compiled_leakage()

    findings: list[Finding] = []

    def add(code, severity, message, topic, **detail):
        findings.append(Finding(code=code, severity=severity, message=message,
                                scope="decision", decision_id=topic.decision_id, detail=detail))

    ids = Counter(topic.decision_id for topic in bank.topics)
    framings = Counter(topic.decision_framing.casefold().strip() for topic in bank.topics)

    for topic in bank.topics:
        if ids[topic.decision_id] > 1:
            add("E_TOPIC_DUPLICATE_ID", "error", "decision_id appears more than once", topic)
        if framings[topic.decision_framing.casefold().strip()] > 1:
            add("E_TOPIC_DUPLICATE_FRAMING", "error", "the same decision framing appears twice", topic)

        if topic.domain not in domains:
            add("E_TOPIC_UNKNOWN_DOMAIN", "error", f"domain {topic.domain!r} is not one of {domains}",
                topic)
        elif not topic.decision_id.startswith(f"{topic.domain}_"):
            add("E_TOPIC_ID_DOMAIN_MISMATCH", "error",
                f"decision_id should begin with {topic.domain}_", topic)

        if topic.options["opt_1"].casefold() == topic.options["opt_2"].casefold():
            add("E_TOPIC_OPTIONS_NOT_DISTINCT", "error", "the two options are the same", topic)
        if topic.competing_goals["opt_1"].casefold() == topic.competing_goals["opt_2"].casefold():
            add("E_TOPIC_GOALS_NOT_DISTINCT", "error", "the two competing goals are the same", topic)

        # -- variants ----------------------------------------------------
        if set(topic.variants) != variant_ids:
            add("E_TOPIC_VARIANTS", "error",
                f"needs exactly variants {sorted(variant_ids)}; got {sorted(topic.variants)}", topic)
        contexts = [v.context.casefold().strip() for v in topic.variants.values()]
        if len(set(contexts)) < len(contexts):
            add("E_TOPIC_VARIANTS_IDENTICAL", "error",
                "variants must differ in context while keeping the same decision", topic)
        # Variants must differ in more than their context sentence: at least one
        # scenario fact must change. Whether the change is substantive is the
        # curator's judgement; the machine only rules out textual identity.
        fact_sets = [tuple(" ".join(f.casefold().split())
                           for f in v.scenario_facts.opt_1 + v.scenario_facts.opt_2)
                     for v in topic.variants.values()]
        if len(fact_sets) > 1 and len(set(fact_sets)) == 1:
            add("E_TOPIC_VARIANTS_SHARE_ALL_FACTS", "error",
                "the variants differ only in context; at least one scenario fact must differ",
                topic)

        for vid, variant in sorted(topic.variants.items()):
            counts = {opt: len(getattr(variant.scenario_facts, opt)) for opt in SEMANTIC_OPTIONS}
            for opt, n in counts.items():
                if not lo <= n <= hi:
                    add("E_TOPIC_FACT_COUNT", "error",
                        f"variant {vid} has {n} fact(s) for {opt}; {lo}-{hi} required",
                        topic, variant=vid, option=opt, count=n)
            if t["equal_fact_counts_within_variant"] and len(set(counts.values())) > 1:
                add("E_TOPIC_FACT_IMBALANCE", "error",
                    f"variant {vid} supports the options with unequal numbers of facts {counts}",
                    topic, variant=vid, counts=counts)

        # -- every field that reaches the generator --------------------------
        for field, limit_key, text in topic.prose_fields():
            n = len(_WORD.findall(text))
            if n > limits[limit_key]:
                add("E_TOPIC_TOO_LONG", "error",
                    f"{field} has {n} words; the limit is {limits[limit_key]}",
                    topic, field=field, words=n, limit=limits[limit_key])
            for spec, pattern in forbidden:
                match = pattern.search(text)
                if match:
                    severity = "error" if spec.severity == "hard_fail" else "warning"
                    add(f"{'E' if severity == 'error' else 'W'}_TOPIC_FORBIDDEN_PHRASE", severity,
                        f"{field}: {spec.family} phrase {match.group(0)!r}",
                        topic, field=field, pattern_id=spec.id)
            for src, pattern in leakage:
                if pattern.search(text):
                    add("E_TOPIC_LABEL_LEAKAGE", "error",
                        f"{field} refers to a display label", topic, field=field)
            for term in framing_terms:
                match = term.search(text)
                if match:
                    add("W_TOPIC_FRAMING_TERMS", "warning",
                        f"{field} mentions {match.group(0)!r}: check for party or identity framing",
                        topic, field=field)

        # -- sources: every cited key must be citable -----------------------
        for problem in registry.citation_problems(s.key for s in topic.source_references):
            add("E_TOPIC_SOURCE_NOT_CITABLE", "error", problem, topic)

        # -- curation --------------------------------------------------------
        cur = topic.curation
        recorded = cur.recorded()
        if topic.status == "curated":
            missing = [k for k, v in recorded.items() if v is None]
            failed = [k for k, v in recorded.items() if v is False]
            if missing or failed or not cur.curated_by or not cur.curated_at:
                add("E_TOPIC_CURATION_INCOMPLETE", "error",
                    "a curated topic needs every judgement true, plus who and when",
                    topic, outstanding=missing, failed=failed)
        if topic.status == "rejected" and not cur.rejection_reason:
            add("E_TOPIC_CURATION_INCOMPLETE", "error", "a rejected topic needs a reason", topic)
        if topic.status != "rejected" and cur.rejection_reason:
            add("E_TOPIC_CURATION_INCOMPLETE", "error",
                "rejection_reason is recorded only for a rejected topic", topic)

    curated = Counter(topic.domain for topic in bank.topics if topic.status == "curated")
    return TopicReport(
        findings=tuple(findings),
        curated_per_domain={d: curated.get(d, 0) for d in domains},
        required_per_domain=t["curated_per_domain_for_drafting"],
    )


def topics_by_status(bank: TopicBank) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"proposed": [], "curated": [], "rejected": []}
    for topic in bank.topics:
        out[topic.status].append(topic.decision_id)
    return out


def sorted_topics(topics: Sequence[TopicBrief]) -> list[TopicBrief]:
    return sorted(topics, key=lambda t: t.decision_id)
