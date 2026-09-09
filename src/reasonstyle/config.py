"""Typed loading and integrity checking of the frozen experiment configuration.

The YAML file is the source of truth for every research choice (plan §15.2).
This module does three things and nothing else:

1. loads it with ``extra="forbid"``, so a misspelled key is a load-time error
   rather than a silently ignored field;
2. enforces the cross-field invariants that make the configuration a *valid*
   description of the 2x2 design — including the immutability guards on the
   four core conditions and the four core contrasts;
3. exposes the provenance block (config hash, file hash, versions) that every
   generated artefact must carry.

It deliberately contains no scenario, rendering or model logic.

Immutability guards
-------------------
``FROZEN_CORE_CONDITIONS`` and ``FROZEN_CORE_CONTRASTS`` restate the design that
the thesis defends. They are *guards*, not the source of values: the config
still supplies everything used at runtime, but a config that renamed a cell or
silently reversed ``NS - NP`` into ``NP - NS`` will fail to load. Diagnostic
conditions may be added freely; they can never enter a core contrast.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .hashing import content_hash, file_sha256

__all__ = ["ConfigError", "ExperimentConfig", "load_config"]


class ConfigError(ValueError):
    """Raised when the configuration is structurally or semantically invalid."""


# --- Immutability guards (see module docstring) -----------------------------

FROZEN_CORE_CONDITIONS: dict[str, tuple[str, str]] = {
    "RS": ("present", "explicit"),
    "RP": ("present", "plain"),
    "NS": ("absent", "explicit"),
    "NP": ("absent", "plain"),
}

# Coefficient maps over the core conditions. Each must sum to zero. The
# coefficient form makes the interaction a first-class contrast and turns a
# reversed sign into a load-time failure rather than a silent sign flip.
FROZEN_CORE_CONTRASTS: dict[str, dict[str, int]] = {
    "style_without_reason": {"NS": 1, "NP": -1},
    "style_with_reason": {"RS": 1, "RP": -1},
    "content_with_style": {"RS": 1, "NS": -1},
    "content_plain": {"RP": 1, "NP": -1},
    "interaction": {"RS": 1, "RP": -1, "NS": -1, "NP": 1},
}

# The four planned simple contrasts carrying the Holm adjustment (plan §7.1).
FROZEN_HOLM_FAMILY: set[str] = {
    "style_without_reason",
    "style_with_reason",
    "content_with_style",
    "content_plain",
}

TAU_TOLERANCE = 1e-9


# --- Structured sections ----------------------------------------------------


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConditionSpec(_Base):
    reason: Literal["present", "absent"]
    style: Literal["explicit", "plain"]
    gloss: str | None = None


class ConditionRegistry(_Base):
    core: dict[str, ConditionSpec]
    # D8: empty in v1. A later commitment-matched control lives here and is
    # never admitted to `contrasts.core`.
    diagnostic: dict[str, ConditionSpec] = Field(default_factory=dict)


class ContrastSpec(_Base):
    coefficients: dict[str, int]
    family: Literal["style", "content", "interaction"]
    kind: Literal["simple", "interaction"]
    gloss: str


class MultiplicitySpec(_Base):
    method: Literal["holm"]
    family: list[str]


class ContrastRegistry(_Base):
    core: dict[str, ContrastSpec]
    primary: str
    multiplicity: MultiplicitySpec


class ForbiddenPattern(_Base):
    id: str
    family: Literal["authority", "evidence", "consensus", "pressure", "certainty"]
    severity: Literal["hard_fail", "warning"]
    pattern: str
    positive: list[str]
    negative: list[str]


class NearTieSpec(_Base):
    role: Literal["robustness_only"]
    definition: str
    probability_threshold: float
    tau_logit: float
    derivation: str
    primary_analysis_excludes: bool


class WordMatchSpec(_Base):
    definition: str
    word_regex: str
    ratio_warn: float
    ratio_fail: float
    # v2+: the measurements the ratio is enforced on. The scenario-level
    # opening is shared by all eight texts, so measuring only the full text
    # would let a long prefix dilute a real imbalance in the manipulated body.
    applies_to: list[str] | None = None


class SentenceMatchSpec(_Base):
    rule: Literal["exact_equality"]
    tolerance: int
    punctuation_may_differ: bool
    semicolon_terminates_sentence: bool
    split_regex: str
    ambiguous_period_requires_human_confirmation: bool
    rationale: str


class MatchingSpec(_Base):
    scope: Literal["group"]
    words: WordMatchSpec
    sentences: SentenceMatchSpec
    opening: dict[str, Any]
    model_token_counts: dict[str, Any]
    naturalness: dict[str, Any]


class TemplateSpec(_Base):
    applies_to: Literal["base", "instruct"]
    gloss: str
    template_text: str | None
    template_sha256: str | None
    answer_cue: str | None
    answer_continuation: dict[str, str | None]
    leading_whitespace: str | None
    answer_token_ids: dict[str, int | None]
    verification_status: Literal["unverified", "verified"]
    verified_at: str | None


class PromptSpec(_Base):
    answer_labels: list[str]
    label_assignment: Literal["renderer_only"]
    option_orders: list[dict[str, str]]
    answer_slot_rule: str
    templates: dict[str, TemplateSpec]
    verification_rule: str


class RawConfig(_Base):
    """Top-level shape. Sections without cross-field invariants stay as
    mappings, but every section name is declared so a typo is caught."""

    config_version: str
    schema_version: str
    created: str
    status: Literal["draft", "frozen"]
    provenance: dict[str, Any]
    freeze: dict[str, Any]
    factors: dict[str, list[str]]
    conditions: ConditionRegistry
    contrasts: ContrastRegistry
    domains: dict[str, Any]
    corpus: dict[str, Any]
    markers: dict[str, Any]
    forbidden: dict[str, Any]
    matching: MatchingSpec
    leakage: dict[str, Any]
    prompts: PromptSpec
    models: dict[str, Any]
    determinism: dict[str, Any]
    splits: dict[str, Any]
    probe_evaluation: dict[str, Any]
    near_tie: NearTieSpec
    # Sections introduced in v2. Optional so that v1 still loads unchanged;
    # required from v2 onward by `_validate_version_requirements`.
    segmentation: dict[str, Any] | None = None
    corpus_provenance: dict[str, Any] | None = None
    analysis: dict[str, Any]
    mechanistic: dict[str, Any]
    annotation: dict[str, Any]
    interpretation: dict[str, Any]


# --- Public wrapper ---------------------------------------------------------


class ExperimentConfig:
    """A loaded, validated configuration plus its provenance hashes."""

    def __init__(self, raw: dict[str, Any], parsed: RawConfig, path: Path | None):
        self.raw = raw
        self.parsed = parsed
        self.path = path
        self.content_hash = content_hash(raw)
        self.file_sha256 = file_sha256(path) if path is not None else None

    # -- identity ---------------------------------------------------------
    @property
    def config_version(self) -> str:
        return self.parsed.config_version

    def provenance_block(self) -> dict[str, Any]:
        """The block every generated artefact must embed (plan §15.3)."""
        return {
            "config_version": self.config_version,
            "schema_version": self.parsed.schema_version,
            "config_content_hash": self.content_hash,
            "config_file_sha256": self.file_sha256,
            "config_status": self.parsed.status,
        }

    # -- conditions and contrasts -----------------------------------------
    def core_condition_ids(self) -> list[str]:
        """Core cells in the frozen canonical order RS, RP, NS, NP.

        Not file order: run tables and contrast output must not change because
        someone reordered two keys in the YAML.
        """
        order = list(FROZEN_CORE_CONDITIONS)
        return sorted(self.parsed.conditions.core, key=order.index)

    def diagnostic_condition_ids(self) -> list[str]:
        """Diagnostic cells, sorted for determinism (no frozen order exists)."""
        return sorted(self.parsed.conditions.diagnostic)

    def all_condition_ids(self) -> list[str]:
        return self.core_condition_ids() + self.diagnostic_condition_ids()

    def contrast(self, name: str) -> ContrastSpec:
        try:
            return self.parsed.contrasts.core[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ConfigError(f"unknown core contrast {name!r}") from exc

    # -- near ties (robustness only) ---------------------------------------
    @property
    def tau(self) -> float:
        return self.parsed.near_tie.tau_logit

    def is_near_tie(self, m_before: float) -> bool:
        """Robustness-only flag. Never used to exclude from the primary sample."""
        return abs(m_before) < self.tau

    # -- markers -----------------------------------------------------------
    def marker_families(self, role: str | None = None) -> dict[str, list[str]]:
        """Marker families, optionally restricted to ``'confirmatory'`` or
        ``'exploratory'``. With no role, every family is returned."""
        m = self.raw["markers"]
        families = {**m["primary_families"], **m["exploratory_families"]}
        if role is None:
            return families
        names = m["roles"][role]
        return {k: v for k, v in families.items() if k in names}

    def confirmatory_families(self) -> list[str]:
        """Families entering the confirmatory style analysis. Concession is
        excluded: it marks a different discourse relation."""
        return list(self.raw["markers"]["roles"]["confirmatory"])

    def exploratory_families(self) -> list[str]:
        return list(self.raw["markers"]["roles"]["exploratory"])

    def permitted_markers(self) -> list[str]:
        return [m for fam in self.marker_families().values() for m in fam]

    def forbidden_patterns(self) -> list[ForbiddenPattern]:
        return [ForbiddenPattern.model_validate(d) for d in self.raw["forbidden"]["patterns"]]

    def compiled_forbidden(
        self, severity: str | None = None
    ) -> list[tuple[ForbiddenPattern, re.Pattern[str]]]:
        """``(spec, compiled)`` per forbidden pattern, optionally filtered by
        severity. ``hard_fail`` rejects; ``warning`` routes to human review."""
        flags = re.IGNORECASE if self.raw["forbidden"]["case_insensitive"] else 0
        return [
            (spec, re.compile(spec.pattern, flags))
            for spec in self.forbidden_patterns()
            if severity is None or spec.severity == severity
        ]

    def compiled_leakage(self) -> list[tuple[str, re.Pattern[str]]]:
        flags = re.IGNORECASE if self.raw["leakage"]["case_insensitive"] else 0
        return [(src, re.compile(src, flags)) for src in self.raw["leakage"]["patterns"]]

    # -- segmentation (v2+) -------------------------------------------------
    @property
    def segmentation(self) -> dict[str, Any] | None:
        return self.parsed.segmentation

    def marker_realizations(self) -> dict[str, dict[str, Any]]:
        """The realization registry, empty for configs that predate it."""
        return dict(self.raw["markers"]["realization"].get("registry", {}))

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig(version={self.config_version!r}, "
            f"status={self.parsed.status!r}, hash={self.content_hash[:12]})"
        )


# --- Invariant checks -------------------------------------------------------


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _validate_invariants(cfg: ExperimentConfig) -> None:
    raw, p = cfg.raw, cfg.parsed

    # -- filename encodes the version, so a version bump cannot overwrite ----
    if cfg.path is not None:
        expected = f"experiment_{p.config_version}"
        _check(
            cfg.path.stem == expected,
            f"filename stem {cfg.path.stem!r} must be {expected!r} so config "
            f"versions cannot overwrite one another",
        )

    # -- factors -----------------------------------------------------------
    _check(set(p.factors) == {"reason", "style"}, "factors must be exactly reason and style")
    _check(set(p.factors["reason"]) == {"present", "absent"}, "reason levels")
    _check(set(p.factors["style"]) == {"explicit", "plain"}, "style levels")

    # -- core conditions are frozen ----------------------------------------
    core = p.conditions.core
    _check(
        set(core) == set(FROZEN_CORE_CONDITIONS),
        f"core conditions must be exactly {sorted(FROZEN_CORE_CONDITIONS)}; got {sorted(core)}",
    )
    for cid, (reason, style) in FROZEN_CORE_CONDITIONS.items():
        _check(
            (core[cid].reason, core[cid].style) == (reason, style),
            f"core condition {cid} must be reason={reason}, style={style}",
        )
    cells = {(c.reason, c.style) for c in core.values()}
    _check(len(cells) == 4, "the four core conditions must cover the full 2x2 without repeats")

    # -- diagnostic registry is additive only -------------------------------
    overlap = set(p.conditions.diagnostic) & set(core)
    _check(not overlap, f"diagnostic condition ids collide with core: {sorted(overlap)}")

    # -- core contrasts are frozen and reference core conditions only -------
    ccs = p.contrasts.core
    _check(
        set(ccs) == set(FROZEN_CORE_CONTRASTS),
        f"core contrasts must be exactly {sorted(FROZEN_CORE_CONTRASTS)}; got {sorted(ccs)}",
    )
    for name, frozen_coeffs in FROZEN_CORE_CONTRASTS.items():
        got = {k: v for k, v in ccs[name].coefficients.items() if v != 0}
        _check(
            got == frozen_coeffs,
            f"contrast {name} must have coefficients {frozen_coeffs}; got {got}",
        )
    for name, spec in ccs.items():
        _check(
            sum(spec.coefficients.values()) == 0,
            f"contrast {name} coefficients must sum to zero; got {sum(spec.coefficients.values())}",
        )
        for cid in spec.coefficients:
            _check(
                cid in core,
                f"contrast {name} references {cid!r}, which is not a core condition "
                f"(diagnostic conditions may never enter a core contrast)",
            )
        expected_kind = "interaction" if len(spec.coefficients) > 2 else "simple"
        _check(spec.kind == expected_kind, f"contrast {name} should have kind={expected_kind}")

    _check(p.contrasts.primary in ccs, f"primary contrast {p.contrasts.primary!r} is not a core contrast")
    _check(
        set(p.contrasts.multiplicity.family) == FROZEN_HOLM_FAMILY,
        f"the Holm family must be exactly {sorted(FROZEN_HOLM_FAMILY)}",
    )
    for name in p.contrasts.multiplicity.family:
        _check(ccs[name].kind == "simple", f"the Holm family holds simple contrasts; {name} is not one")

    # -- corpus arithmetic --------------------------------------------------
    dom, corp = raw["domains"], raw["corpus"]
    n_dom = len(dom["ids"])
    _check(
        dom["decisions_per_domain_full"] * n_dom == corp["decisions_full"],
        "decisions_per_domain_full * n_domains must equal decisions_full",
    )
    _check(
        dom["decisions_per_domain_pilot"] * n_dom == corp["decisions_pilot"],
        "decisions_per_domain_pilot * n_domains must equal decisions_pilot "
        "(pilot is stratified 4/4/4)",
    )
    _check(corp["supported_options"] == ["opt_1", "opt_2"], "semantic option ids are immutable")
    per_scenario = len(corp["supported_options"]) * corp["cells_per_group"]
    _check(
        corp["counterarguments_per_scenario"] == per_scenario,
        "counterarguments_per_scenario must equal len(supported_options) * cells_per_group",
    )
    for n_dec, key in ((corp["decisions_full"], "texts_full"), (corp["decisions_pilot"], "texts_pilot")):
        expected = n_dec * corp["variants_per_decision"] * per_scenario
        _check(corp[key] == expected, f"{key} must be {expected}; got {corp[key]}")

    # -- markers -----------------------------------------------------------
    mk = raw["markers"]
    primary, exploratory = mk["primary_families"], mk["exploratory_families"]
    _check(bool(primary), "the primary marker inventory must not be empty")
    overlap_fams = set(primary) & set(exploratory)
    _check(not overlap_fams, f"families listed as both primary and exploratory: {sorted(overlap_fams)}")

    fams = {**primary, **exploratory}
    seen: dict[str, str] = {}
    for fam, markers in fams.items():
        _check(bool(markers), f"marker family {fam!r} is empty")
        for m in markers:
            _check(m not in seen, f"marker {m!r} appears in both {seen.get(m)!r} and {fam!r}")
            seen[m] = fam

    roles = mk["roles"]
    _check(
        set(roles["confirmatory"]) == set(primary),
        "the confirmatory role must name exactly the primary marker families",
    )
    _check(
        set(roles["exploratory"]) == set(exploratory),
        "the exploratory role must name exactly the exploratory marker families",
    )
    _check(
        not (set(roles["confirmatory"]) & set(roles["exploratory"])),
        "a marker family cannot be both confirmatory and exploratory",
    )

    man = mk["analysis"]
    _check(man["report_family_specific_effects"] is True, "family-specific effects must be reported")
    _check(man["test_family_by_style_interaction"] is True, "a family x style interaction must be tested")
    _check(
        man["pooling_rule"] == "only_if_directionally_compatible",
        "families are pooled only if their effects are directionally compatible",
    )
    _check(man["leave_one_marker_out"] is True, "leave-one-marker-out evaluation must be supported")
    _check(
        man["leave_one_realization_template_out"] is True,
        "leave-one-realization-template-out evaluation must be supported",
    )
    _check(
        man["lexical_match_establishes_discourse_function"] is False,
        "a lexical match must never be treated as evidence of discourse function",
    )
    _check(man["human_validation_authoritative"] is True, "human validation stays authoritative")
    _check(
        mk["realization"]["template_id_field"] == "marker_realization_id",
        "the marker realization template id field must be recorded",
    )

    assign = mk["assignment"]
    _check(
        set(assign["styled_cells"]) | set(assign["plain_cells"]) == set(core),
        "styled_cells and plain_cells must partition the core conditions",
    )
    _check(
        not (set(assign["styled_cells"]) & set(assign["plain_cells"])),
        "a condition cannot be both styled and plain",
    )
    for cid in assign["styled_cells"]:
        _check(core[cid].style == "explicit", f"{cid} is listed as styled but has style={core[cid].style}")
    for cid in assign["plain_cells"]:
        _check(core[cid].style == "plain", f"{cid} is listed as plain but has style={core[cid].style}")
    _check(assign["one_family_per_group"] is True, "D3: one marker family per group")
    _check(assign["holdout_keeps_group_together"] is True, "D3: held-out-family split keeps the group intact")
    _check(
        assign["balance"]["mechanism"] == "allocation_constraint_then_post_split_check",
        "marker family is balanced by allocation constraint plus post-split check, "
        "never by decision-level stratification",
    )

    # -- style frozen at the realization, not merely the family (D12) --------
    _check(
        assign["group_level_fields"] == ["marker_family", "marker_string", "marker_realization_id"],
        "the realization group is (marker_family, marker_string, marker_realization_id): "
        "sharing only a family would let RS - NS compare 'therefore' against "
        "'because', or one marker placement against another",
    )
    _check(assign["same_realization_across_cells"] is True,
           "all four cells of a group must instantiate the same marker realization")
    inh = assign["cell_inheritance"]
    _check(set(inh) == set(core), "cell_inheritance must cover exactly the four core conditions")
    for cid in assign["styled_cells"]:
        _check(inh[cid]["markers_present"] is True, f"styled cell {cid} must record markers_present: true")
        _check(inh[cid]["instantiates"] == "marker_realization",
               f"styled cell {cid} must instantiate the group's marker realization")
    for cid in assign["plain_cells"]:
        _check(inh[cid]["markers_present"] is False, f"plain cell {cid} must record markers_present: false")
        _check(inh[cid]["instantiates"] == "paired_plain_transformation",
               f"plain cell {cid} must instantiate the paired plain transformation")

    # -- allocation minima so a held-out-marker test cannot be nominal -------
    alloc = mk["allocation"]
    for key in ("min_decisions_per_marker",
                "min_decisions_per_leave_one_marker_test",
                "min_decisions_per_held_out_family_test"):
        _check(
            isinstance(alloc[key], int) and alloc[key] >= 2,
            f"{key} must be an integer of at least 2: a leave-one-out evaluation "
            f"on one or two decisions is not an evaluation",
        )
    for key in ("max_share_within_one_domain", "max_share_within_one_supported_option"):
        _check(0.0 < alloc[key] <= 1.0, f"{key} must lie in (0, 1]")
    _check(
        alloc["max_share_within_one_domain"] >= 1.0 / len(dom["ids"]),
        "the per-domain share cap cannot be below an even split across domains",
    )
    _check(alloc["confirmatory_families_in_every_split"] is True,
           "every corpus split must contain every confirmatory marker family")
    _check(alloc["report_as_corpus_statistic"] is True,
           "marker allocation must be reported as a corpus statistic")

    # -- forbidden patterns: unique ids, and fixtures that bind -------------
    _check(raw["forbidden"]["case_insensitive"] is True, "all forbidden patterns compile case-insensitively")
    specs = cfg.forbidden_patterns()
    _check(bool(specs), "the forbidden inventory must not be empty")
    ids = [sp.id for sp in specs]
    _check(len(ids) == len(set(ids)), "forbidden pattern ids must be unique")
    _check(
        any(sp.severity == "hard_fail" for sp in specs),
        "at least one pattern must be a hard failure",
    )
    for spec, pat in cfg.compiled_forbidden():
        _check(bool(spec.positive) and bool(spec.negative),
               f"forbidden pattern {spec.id!r} needs both positive and negative fixtures")
        for example in spec.positive:
            _check(
                pat.search(example) is not None,
                f"forbidden pattern {spec.id!r} no longer catches its own positive "
                f"fixture {example!r}: it has been weakened",
            )
        for example in spec.negative:
            _check(
                pat.search(example) is None,
                f"forbidden pattern {spec.id!r} matches its negative fixture "
                f"{example!r}: it is over-broad",
            )

    # -- D7: permitted markers must not collide with forbidden/leakage ------
    for spec, pat in cfg.compiled_forbidden():
        for marker in cfg.permitted_markers():
            for variant in (marker, marker.lower(), marker.upper(), marker.capitalize()):
                _check(
                    pat.search(variant) is None,
                    f"forbidden pattern {spec.id!r} (family {spec.family!r}, "
                    f"{spec.severity}) matches permitted marker {variant!r}: the "
                    f"style manipulation would be unusable",
                )
    for src, pat in cfg.compiled_leakage():
        for marker in cfg.permitted_markers():
            _check(
                pat.search(marker) is None,
                f"leakage pattern {src!r} matches permitted marker {marker!r}",
            )

    # -- matching tolerances (D1, D2) ---------------------------------------
    w = p.matching.words
    _check(1.0 < w.ratio_warn <= w.ratio_fail, "require 1 < ratio_warn <= ratio_fail")
    _check(abs(w.ratio_warn - 1.10) < 1e-12 and abs(w.ratio_fail - 1.15) < 1e-12, "D2: warn 1.10, fail 1.15")
    re.compile(w.word_regex)
    s = p.matching.sentences
    _check(s.tolerance == 0, "D1: exact sentence-count equality, tolerance 0")
    _check(s.semicolon_terminates_sentence is False, "D1: a semicolon does not terminate a sentence")
    re.compile(s.split_regex)
    _check(p.matching.opening["level"] == "scenario", "D6: the opening is stored at scenario level")
    _check(
        p.matching.opening["shared_across"] == "all_eight_counterarguments",
        "D6: one opening is shared by all eight counterarguments of a scenario",
    )
    _check(p.matching.naturalness["human_rating_required"] is True, "D1: naturalness must be human-rated")
    _check(
        p.matching.model_token_counts["role"] == "diagnostic_only",
        "D2: token counts never substitute for the word-count rule",
    )

    # -- leakage (D5) -------------------------------------------------------
    _check(raw["leakage"]["bare_letter_rejected"] is False, "D5: a bare letter is also an English article")

    # -- prompts and answer slot (D10) --------------------------------------
    _check(p.prompts.answer_labels == ["A", "B"], "display labels are A and B")
    orders = {(o["A"], o["B"]) for o in p.prompts.option_orders}
    _check(
        orders == {("opt_1", "opt_2"), ("opt_2", "opt_1")},
        "both option orders must be present and must map to semantic option ids",
    )
    _check(len({o["order_id"] for o in p.prompts.option_orders}) == 2, "option order ids must be distinct")
    for name, tpl in p.prompts.templates.items():
        _check(
            set(tpl.answer_continuation) == {"A", "B"} and set(tpl.answer_token_ids) == {"A", "B"},
            f"template {name}: answer fields must be keyed by the display labels",
        )
        ids_set = any(v is not None for v in tpl.answer_token_ids.values())
        if tpl.verification_status == "unverified":
            _check(
                not ids_set,
                f"template {name}: token ids must stay null while verification_status is "
                f"'unverified' — Stage 5 verifies and a config bump pins them",
            )
        else:
            _check(
                all(v is not None for v in tpl.answer_token_ids.values())
                and tpl.template_sha256 is not None
                and tpl.verified_at is not None,
                f"template {name}: a verified template must pin token ids, template hash and date",
            )
    applies = sorted(t.applies_to for t in p.prompts.templates.values())
    _check(applies == ["base", "instruct"], "exactly one base and one instruct template")

    # -- models -------------------------------------------------------------
    for variant in ("base", "instruct"):
        tpl_name = raw["models"][variant]["template"]
        _check(tpl_name in p.prompts.templates, f"model {variant} references unknown template {tpl_name!r}")
        _check(
            p.prompts.templates[tpl_name].applies_to == variant,
            f"model {variant} references a template declared for "
            f"{p.prompts.templates[tpl_name].applies_to!r}",
        )

    # -- splits -------------------------------------------------------------
    sp = raw["splits"]
    fr = sp["fractions"]
    _check(abs(sum(fr.values()) - 1.0) < 1e-9, f"split fractions must sum to 1.0; got {sum(fr.values())}")
    _check(sp["grouping_unit"] == "decision_id", "splits must be grouped on decision_id")
    _check(
        sp["seed_ref"] in raw["determinism"]["seeds"],
        "splits.seed_ref must name a declared seed",
    )
    _check(
        sp["stratify_by"] == ["domain"],
        "only decision-level labels may stratify the corpus split; marker family "
        "is a property of a (scenario_id, supported_option) group, not of a decision",
    )
    _check(
        sp["marker_family_balance"]["is_stratification_key"] is False,
        "marker family is an allocation constraint and post-split check, not a "
        "stratification key",
    )
    _check(set(sp["decisions"]) == set(fr), "split decision counts and fractions must cover the same splits")
    _check(
        sum(sp["decisions"].values()) == corp["decisions_full"],
        f"split decision counts must sum to decisions_full ({corp['decisions_full']})",
    )
    for name, frac in fr.items():
        expected = round(frac * corp["decisions_full"])
        _check(
            sp["decisions"][name] == expected,
            f"split {name}: {frac} of {corp['decisions_full']} decisions is {expected}, "
            f"but {sp['decisions'][name]} is declared",
        )

    # -- probe evaluation schemes are not corpus splits ----------------------
    pe = raw["probe_evaluation"]
    _check(pe["primary_split_ref"] == "splits", "probe evaluation must reference the primary split")
    required_schemes = {
        "grouped_held_out_decisions",
        "leave_one_domain_out",
        "held_out_marker_family",
        "leave_one_marker_out",
        "leave_one_realization_template_out",
    }
    _check(
        required_schemes <= set(pe["schemes"]),
        f"probe evaluation must define {sorted(required_schemes)}",
    )
    _check(
        pe["schemes"]["leave_one_domain_out"]["folds"] == len(dom["ids"]),
        f"leave-one-domain-out needs one fold per domain ({len(dom['ids'])})",
    )
    for name in ("held_out_marker_family", "leave_one_marker_out", "leave_one_realization_template_out"):
        _check(
            pe["schemes"][name]["unit_kept_together"] == "group",
            f"scheme {name} must keep the complete four-cell group together (D3)",
        )

    # Decision-level disjointness applies to EVERY scheme. Keeping the group
    # intact is necessary but insufficient: a sibling group from the same
    # decision could otherwise sit in training while this one is evaluated.
    for name, scheme in pe["schemes"].items():
        _check(
            scheme.get("no_decision_overlap") is True,
            f"probe scheme {name} must declare no_decision_overlap: true — "
            f"otherwise another group from the same underlying decision can leak "
            f"scenario content into training",
        )
        _check(
            scheme.get("grouping_unit") == "decision_id",
            f"probe scheme {name} must be grouped on decision_id",
        )
    _check(len(pe["invariants"]) >= 3, "the probe-evaluation invariants must be stated in the config")
    for name in ("held_out_marker_family", "leave_one_marker_out", "leave_one_realization_template_out"):
        ref = pe["schemes"][name]["min_decisions_ref"]
        node: Any = raw
        for part in ref.split("."):
            _check(part in node, f"scheme {name} references missing config path {ref!r}")
            node = node[part]
        _check(isinstance(node, int) and node >= 2, f"scheme {name} minimum support {ref!r} must be >= 2")

    # -- near ties (D4) -----------------------------------------------------
    nt = p.near_tie
    _check(0.5 < nt.probability_threshold < 1.0, "the near-tie probability threshold must lie in (0.5, 1)")
    expected_tau = math.log(nt.probability_threshold / (1.0 - nt.probability_threshold))
    _check(
        abs(nt.tau_logit - expected_tau) < TAU_TOLERANCE,
        f"tau_logit {nt.tau_logit!r} does not match log(p/(1-p)) = {expected_tau!r} "
        f"for p = {nt.probability_threshold}",
    )
    _check(nt.primary_analysis_excludes is False, "§6.4: the full sample remains primary")

    # -- analysis -----------------------------------------------------------
    an = raw["analysis"]
    _check(an["cluster_unit"] == "decision_id", "§7.1: the bootstrap clusters on decision_id")
    _check(an["multiplicity"] == p.contrasts.multiplicity.method, "multiplicity rule stated twice and disagrees")
    _check(
        an["bootstrap"]["seed_ref"] in raw["determinism"]["seeds"],
        "analysis.bootstrap.seed_ref must name a declared seed",
    )

    # -- mechanistic (D11) --------------------------------------------------
    mech = raw["mechanistic"]
    _check(mech["layers"]["indexing"] == "0..n_layers-1", "D11: blocks are indexed 0..n_layers-1")
    _check(mech["layers"]["primary_hook"] == "resid_post", "D11: the primary hook site is resid_post")
    _check(
        mech["layers"]["embeddings_analysed_in_v1"] is False,
        "D11: embeddings are not a v1 analysis target",
    )
    _check(
        mech["layers"]["embeddings_name"] == "embed",
        "D11: embeddings are named 'embed', never a negative block index",
    )
    for pair in mech["subset"]["contrasts"]:
        for cid in pair:
            _check(cid in core, f"mechanistic subset references non-core condition {cid!r}")
    _check(
        mech["subset"]["direction_field"] == "counter_target_opt",
        "the patched direction is resolved at runtime through counter_target_opt",
    )
    _check(
        mech["subset"]["n_decisions"] <= corp["decisions_full"],
        "the mechanistic subset cannot exceed the corpus",
    )

    # -- annotation (D8, D13) -----------------------------------------------
    ann = raw["annotation"]
    levels = ann["levels"]
    _check(set(levels) == {"item", "pair", "scenario"}, "annotations are stored at item, pair and scenario level")

    seen_rating: dict[str, str] = {}
    for level, spec in levels.items():
        _check(bool(spec["ratings"]), f"annotation level {level!r} has no ratings")
        _check("annotator_id" in spec["keys"], f"annotation level {level!r} must be keyed by annotator")
        for r in spec["ratings"]:
            _check(
                r not in seen_rating,
                f"rating {r!r} is declared at both {seen_rating.get(r)!r} and {level!r} level: "
                f"a judgement belongs to exactly one level",
            )
            seen_rating[r] = level
    _check("scenario_id" in levels["scenario"]["keys"], "scenario ratings are keyed by scenario_id")
    _check(
        "condition" not in levels["pair"]["keys"] and "condition" not in levels["scenario"]["keys"],
        "only item-level annotations are keyed by condition",
    )
    for pair in levels["pair"]["pairs"]:
        _check(len(pair) == 2 and all(c in core for c in pair), f"annotation pair {pair} must name two core conditions")
        _check(
            core[pair[0]].reason == core[pair[1]].reason,
            f"proposition preservation is judged within a reason level; {pair} crosses it",
        )
    _check(
        ann["denormalization"]["duplicate_pair_and_scenario_judgements_per_item"] is False,
        "pair- and scenario-level judgements are not duplicated onto every item",
    )

    prov = ann["rating_provenance"]
    required_additions = {
        "perceived_speaker_commitment",
        "perceived_naturalness",
        "perceived_unstated_support",
    }
    _check(
        required_additions.issubset(set(prov["additions_v1"])),
        f"D8: annotations must include {sorted(required_additions)}",
    )
    buckets = {name: set(values) for name, values in prov.items()}
    for a, b in ((x, y) for x in buckets for y in buckets if x < y):
        overlap = buckets[a] & buckets[b]
        _check(not overlap, f"ratings claimed by both {a!r} and {b!r}: {sorted(overlap)}")
    declared: set[str] = set().union(*buckets.values()) if buckets else set()
    _check(
        declared == set(seen_rating),
        "every declared rating must have provenance, and every rating with "
        "provenance must be declared at exactly one level "
        f"(missing provenance: {sorted(set(seen_rating) - declared)}; "
        f"provenance without a level: {sorted(declared - set(seen_rating))})",
    )
    _check(
        ann["agreement"]["expected_value_preset"] is False,
        "§6.5: do not set an expected agreement value in advance",
    )
    _check(
        ann["reliability_subsample"]["seed_ref"] in raw["determinism"]["seeds"],
        "annotation.reliability_subsample.seed_ref must name a declared seed",
    )

    # -- interpretation ceiling ---------------------------------------------
    _check(
        raw["interpretation"]["primary_claim_mode"] == "descriptive",
        "the primary claim is descriptive",
    )
    _check(
        required_additions.issuperset(set(raw["interpretation"]["benign_alternatives_measured"]))
        and bool(raw["interpretation"]["benign_alternatives_measured"]),
        "the benign alternatives to NS-NP must be measured by declared annotation fields",
    )


def _config_version_number(version: str) -> int:
    """``"v2"`` -> ``2``. Used only to gate which sections are required."""
    m = re.fullmatch(r"v(\d+)", version)
    _check(m is not None, f"config_version must look like 'v<n>'; got {version!r}")
    assert m is not None
    return int(m.group(1))


def _validate_v2_sections(cfg: ExperimentConfig) -> None:
    """Checks for sections introduced in config v2 (Implementation Stage 2a)."""
    raw, p = cfg.raw, cfg.parsed

    # -- segmentation (D14) -------------------------------------------------
    seg = raw["segmentation"]
    _check(bool(seg["library"]) and bool(seg["version"]), "the segmenter library and version must be pinned")
    _check(
        seg["authority"] == "machine_count_authoritative",
        "the machine sentence count is authoritative (D1)",
    )
    _check(
        seg["human_override"]["permitted"] is False
        and seg["human_override"]["requires_recorded_annotation"] is True,
        "a segmentation correction requires a recorded annotation; it is never silent",
    )
    _check(
        seg["guarantees"]["semicolon_does_not_terminate_sentence"] is True,
        "D1 relies on a semicolon achieving explicit framing inside one sentence",
    )
    _check(seg["clean"] is False, "the segmenter must never rewrite the text it measures")
    for kind, pattern in seg["ambiguity_patterns"].items():
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ConfigError(f"ambiguity pattern {kind!r} does not compile: {exc}") from exc
    restrictions = seg["text_restrictions"]
    for flag in ("single_paragraph", "prohibit_bullet_lists", "prohibit_numbered_lists"):
        _check(restrictions[flag] is True, f"text restriction {flag!r} must be enabled")
    _check(bool(restrictions["known_abbreviations"]), "the known-abbreviation list must not be empty")

    # -- word ratio scope ---------------------------------------------------
    _check(
        p.matching.words.applies_to == ["full_text", "body"],
        "the word-count ratio is enforced on both the complete counterargument "
        "and the manipulated body, so a shared opening cannot dilute it",
    )

    # -- marker realization registry (D3b) ----------------------------------
    realization = raw["markers"]["realization"]
    _check(
        realization["recorded_per"] == "group",
        "the realization is stored once per (scenario_id, supported_option) group "
        "and inherited; it is never duplicated per cell",
    )
    id_pattern = re.compile(realization["id_pattern"])
    registry = realization["registry"]
    _check(bool(registry), "the realization registry must not be empty")
    families = set(cfg.marker_families())
    covered: set[str] = set()
    for rid, entry in registry.items():
        _check(id_pattern.fullmatch(rid) is not None, f"realization id {rid!r} does not match the id pattern")
        _check(entry["family"] in families, f"realization {rid!r} names unknown marker family {entry['family']!r}")
        _check(bool(entry["description"]), f"realization {rid!r} needs a description")
        covered.add(entry["family"])
    missing = families - covered
    _check(not missing, f"marker families with no realization: {sorted(missing)}")

    # -- model selection is not frozen yet ----------------------------------
    models = raw["models"]
    _check(
        models["selection_status"] in {"unfrozen", "frozen"},
        "models.selection_status must be 'unfrozen' or 'frozen'",
    )
    for variant in ("base", "instruct"):
        entry = models[variant]
        _check(entry["role"] == variant, f"model {variant} must declare role: {variant}")
        if models["selection_status"] == "unfrozen":
            _check(
                entry["repo_id"] is None and entry["revision"] is None,
                f"model {variant}: selection is unfrozen, so repo_id and revision "
                f"must stay null until the compatibility test freezes them",
            )
        else:
            _check(
                entry["repo_id"] is not None and entry["revision"] is not None,
                f"model {variant}: a frozen selection must pin repo_id and revision",
            )

    # -- corpus provenance (D15) --------------------------------------------
    prov = raw["corpus_provenance"]
    src = prov["source_references"]
    _check(src["cardinality"] == "zero_or_more", "a scenario may cite several sources, or none")
    _check(src["open_vocabulary"] is True, "source references are an open structure, not an enum")
    _check(src["constructed_allows_empty_list"] is True, "a constructed scenario may cite no source")
    _check(
        "constructed" in src["scenario_level_type_values"],
        "a scenario must be able to declare itself constructed",
    )
    _check("dataset_name" in src["fields"], "a source reference must name its dataset")
    for field in ("dataset_version", "source_item_id", "source_url", "access_date", "reuse_licence"):
        _check(field in src["fields"], f"the source-reference structure must carry {field!r}")
    gen = prov["generation_metadata"]
    _check(
        gen["separate_from_source_references"] is True,
        "LLM generation metadata is not a source reference and not a licence claim",
    )
    for field in ("generator_model", "generator_model_revision", "prompt_hash",
                  "generation_parameters", "seed", "generated_at"):
        _check(field in gen["fields"], f"generation metadata must carry {field!r}")
    _check(
        set(src["fields"]) & set(gen["fields"]) == set(),
        "source-reference and generation-metadata fields must not overlap",
    )


def _validate_version_requirements(cfg: ExperimentConfig) -> None:
    version = _config_version_number(cfg.parsed.config_version)
    if version >= 2:
        for section in ("segmentation", "corpus_provenance"):
            _check(cfg.raw.get(section) is not None, f"config {cfg.parsed.config_version} must declare {section!r}")
        _validate_v2_sections(cfg)
    else:
        for section in ("segmentation", "corpus_provenance"):
            _check(
                cfg.raw.get(section) is None,
                f"section {section!r} was introduced in v2; it may not appear in "
                f"{cfg.parsed.config_version} — create a new config version instead",
            )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load, type-check and invariant-check a versioned experiment config."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} does not contain a YAML mapping")
    parsed = RawConfig.model_validate(raw)
    cfg = ExperimentConfig(raw=raw, parsed=parsed, path=path)
    _validate_invariants(cfg)
    _validate_version_requirements(cfg)
    return cfg
