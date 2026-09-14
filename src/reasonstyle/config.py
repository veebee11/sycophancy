"""Loading and checking the experiment configuration.

One editable file, ``configs/experiment.yaml``, holds every value the code
reads; git records its history. When a configuration is actually used to
produce research results it is copied to ``configs/frozen/<name>.yaml`` and
never changes again — from that point its hash is what every artefact refers
back to.

This module loads the file, rejects unrecognised keys, and enforces the
invariants that make it a valid description of the experiment. The reasoning
behind the values is in ``docs/design_notes.md``, not here.

``FROZEN_CORE_CONDITIONS`` and ``FROZEN_CORE_CONTRASTS`` restate the design the
thesis defends. They are guards, not the source of values: the config still
supplies everything used at runtime, but a config that renamed a cell or
reversed a contrast will not load.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .hashing import content_hash, file_sha256, sha256_of

__all__ = ["ConfigError", "ExperimentConfig", "load_config"]

DEV_CONFIG = Path("configs/experiment.yaml")
FROZEN_DIR = "frozen"


#: The repository this package lives in — src/reasonstyle/config.py -> root.
#: Used only as the last place to look for a file the config points at.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """The configuration is structurally or semantically invalid."""


# --- design guards ----------------------------------------------------------

FROZEN_CORE_CONDITIONS: dict[str, tuple[str, str]] = {
    "RS": ("present", "explicit"),
    "RP": ("present", "plain"),
    "NS": ("absent", "explicit"),
    "NP": ("absent", "plain"),
}

#: Coefficient maps over the core conditions; each must sum to zero.
FROZEN_CORE_CONTRASTS: dict[str, dict[str, int]] = {
    "style_without_reason": {"NS": 1, "NP": -1},
    "style_with_reason": {"RS": 1, "RP": -1},
    "content_with_style": {"RS": 1, "NS": -1},
    "content_plain": {"RP": 1, "NP": -1},
    "interaction": {"RS": 1, "RP": -1, "NS": -1, "NP": 1},
}

#: The four planned simple contrasts carrying the Holm adjustment.
FROZEN_HOLM_FAMILY = {"style_without_reason", "style_with_reason",
                      "content_with_style", "content_plain"}

TAU_TOLERANCE = 1e-9


# --- typed shape ------------------------------------------------------------


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConditionSpec(_Base):
    reason: Literal["present", "absent"]
    style: Literal["explicit", "plain"]
    gloss: str | None = None


class ConditionRegistry(_Base):
    core: dict[str, ConditionSpec]
    #: Empty for now. A later commitment-matched control lives here and can
    #: never be admitted to a core contrast.
    diagnostic: dict[str, ConditionSpec] = Field(default_factory=dict)


class ContrastSpec(_Base):
    coefficients: dict[str, int]
    family: Literal["style", "content", "interaction"]
    kind: Literal["simple", "interaction"]
    gloss: str | None = None


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
    probability_threshold: float
    tau_logit: float


class WordMatchSpec(_Base):
    word_regex: str
    ratio_warn: float
    ratio_fail: float
    #: Enforced on both, so a shared opening cannot dilute an imbalance in the
    #: manipulated body.
    applies_to: list[str]


class SentenceMatchSpec(_Base):
    rule: Literal["exact_equality"]
    tolerance: int
    split_regex: str


class MatchingSpec(_Base):
    scope: Literal["group"]
    words: WordMatchSpec
    sentences: SentenceMatchSpec
    opening: dict[str, Any]


class TemplateSpec(_Base):
    applies_to: Literal["base", "instruct"]
    materialization: Literal["plain_scaffold", "tokenizer_chat_template"]
    template_text: str | None
    template_sha256: str | None
    role_labels: dict[str, str] | None
    turn_separator: str | None
    answer_cue: str
    answer_continuation: dict[str, str | None]
    leading_whitespace: str | None
    answer_token_ids: dict[str, int | None]
    verification_status: Literal["unverified", "verified"]
    verified_at: str | None


class PromptSpec(_Base):
    answer_labels: list[str]
    option_orders: list[dict[str, str]]
    question_text: str
    instruction_text: str
    option_line_format: str
    turn_structure: list[dict[str, Any]]
    templates: dict[str, TemplateSpec]
    #: Drafting templates for the generator. Separate from ``templates``, which
    #: are evaluation scaffolds for the models under test.
    drafting: dict[str, Any]


class RawConfig(_Base):
    config_version: str
    status: Literal["draft", "frozen"]
    provenance: dict[str, Any]
    factors: dict[str, list[str]]
    conditions: ConditionRegistry
    contrasts: ContrastRegistry
    domains: dict[str, Any]
    corpus: dict[str, Any]
    markers: dict[str, Any]
    forbidden: dict[str, Any]
    matching: MatchingSpec
    segmentation: dict[str, Any]
    leakage: dict[str, Any]
    prompts: PromptSpec
    models: dict[str, Any]
    determinism: dict[str, Any]
    splits: dict[str, Any]
    probe_evaluation: dict[str, Any]
    near_tie: NearTieSpec
    analysis: dict[str, Any]
    mechanistic: dict[str, Any]
    corpus_provenance: dict[str, Any]
    topics: dict[str, Any]
    annotation: dict[str, Any]
    review: dict[str, Any]


# --- public wrapper ---------------------------------------------------------


class ExperimentConfig:
    """A loaded, checked configuration plus its provenance hashes."""

    def __init__(self, raw: dict[str, Any], parsed: RawConfig, path: Path | None):
        self.raw = raw
        self.parsed = parsed
        self.path = path
        self.content_hash = content_hash(raw)
        self.file_sha256 = file_sha256(path) if path is not None else None

    def resolve_path(self, relative: str | Path) -> Path:
        """Locate a file the configuration points at, such as a prompt template.

        Tried in order: beside the config, one level above it (the usual
        ``configs/experiment.yaml`` -> repository root), then the repository
        this package was installed from. The last case is what lets a test
        write a mutated config into a temporary directory without having to
        copy the prompt files with it; the hash check still applies, so a
        template found this way is still the pinned one.
        """
        relative = Path(relative)
        bases = [p for p in ((self.path.parent, self.path.parent.parent) if self.path else ())]
        bases.append(_REPO_ROOT)
        for base in bases:
            candidate = base / relative
            if candidate.is_file():
                return candidate
        return bases[0] / relative

    @property
    def config_version(self) -> str:
        return self.parsed.config_version

    @property
    def is_frozen(self) -> bool:
        return self.parsed.status == "frozen"

    def provenance_block(self) -> dict[str, Any]:
        """The block every generated artefact embeds."""
        return {
            "config_version": self.config_version,
            "config_content_hash": self.content_hash,
            "config_file_sha256": self.file_sha256,
            "config_status": self.parsed.status,
        }

    # -- conditions and contrasts ------------------------------------------
    def core_condition_ids(self) -> list[str]:
        """In the canonical order RS, RP, NS, NP — never file order."""
        order = list(FROZEN_CORE_CONDITIONS)
        return sorted(self.parsed.conditions.core, key=order.index)

    def diagnostic_condition_ids(self) -> list[str]:
        return sorted(self.parsed.conditions.diagnostic)

    def contrast(self, name: str) -> ContrastSpec:
        try:
            return self.parsed.contrasts.core[name]
        except KeyError as exc:
            raise ConfigError(f"unknown contrast {name!r}") from exc

    # -- near ties (robustness only) ----------------------------------------
    @property
    def tau(self) -> float:
        return self.parsed.near_tie.tau_logit

    def is_near_tie(self, m_before: float) -> bool:
        return abs(m_before) < self.tau

    # -- markers -------------------------------------------------------------
    def marker_families(self, role: str | None = None) -> dict[str, list[str]]:
        m = self.raw["markers"]
        families = {**m["primary_families"], **m["exploratory_families"]}
        if role is None:
            return families
        names = m["roles"][role]
        return {k: v for k, v in families.items() if k in names}

    def confirmatory_families(self) -> list[str]:
        return list(self.raw["markers"]["roles"]["confirmatory"])

    def exploratory_families(self) -> list[str]:
        return list(self.raw["markers"]["roles"]["exploratory"])

    def permitted_markers(self) -> list[str]:
        return [m for fam in self.marker_families().values() for m in fam]

    def marker_realizations(self) -> dict[str, dict[str, Any]]:
        return dict(self.raw["markers"]["realization"]["registry"])

    # -- patterns ------------------------------------------------------------
    def forbidden_patterns(self) -> list[ForbiddenPattern]:
        return [ForbiddenPattern.model_validate(d) for d in self.raw["forbidden"]["patterns"]]

    def compiled_forbidden(self, severity: str | None = None
                           ) -> list[tuple[ForbiddenPattern, re.Pattern[str]]]:
        flags = re.IGNORECASE if self.raw["forbidden"]["case_insensitive"] else 0
        return [(spec, re.compile(spec.pattern, flags))
                for spec in self.forbidden_patterns()
                if severity is None or spec.severity == severity]

    def compiled_leakage(self) -> list[tuple[str, re.Pattern[str]]]:
        flags = re.IGNORECASE if self.raw["leakage"]["case_insensitive"] else 0
        return [(src, re.compile(src, flags)) for src in self.raw["leakage"]["patterns"]]

    def __repr__(self) -> str:
        return (f"ExperimentConfig({self.config_version!r}, {self.parsed.status!r}, "
                f"{self.content_hash[:12]})")


# --- invariants -------------------------------------------------------------


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _check_location(cfg: ExperimentConfig) -> None:
    """A draft lives at configs/experiment.yaml; a frozen config lives in
    configs/frozen/ under its own version name and never changes."""
    if cfg.path is None:
        return
    if cfg.parsed.status == "frozen":
        _check(cfg.path.parent.name == FROZEN_DIR,
               f"a frozen config must live in configs/{FROZEN_DIR}/; found {cfg.path}")
        _check(cfg.path.stem == cfg.config_version,
               f"frozen config {cfg.path.name} must be named after its version "
               f"{cfg.config_version!r}")
    else:
        _check(cfg.path.parent.name != FROZEN_DIR,
               f"{cfg.path} is in configs/{FROZEN_DIR}/ but is still marked draft")


def _check_conditions_and_contrasts(cfg: ExperimentConfig) -> None:
    p = cfg.parsed
    core = p.conditions.core

    _check(set(p.factors) == {"reason", "style"}, "factors are exactly reason and style")
    _check(set(core) == set(FROZEN_CORE_CONDITIONS),
           f"core conditions must be {sorted(FROZEN_CORE_CONDITIONS)}; got {sorted(core)}")
    for cid, (reason, style) in FROZEN_CORE_CONDITIONS.items():
        _check((core[cid].reason, core[cid].style) == (reason, style),
               f"core condition {cid} must be reason={reason}, style={style}")
    _check(len({(c.reason, c.style) for c in core.values()}) == 4,
           "the four core conditions must cover the 2x2 without repeats")

    overlap = set(p.conditions.diagnostic) & set(core)
    _check(not overlap, f"diagnostic ids collide with core: {sorted(overlap)}")

    ccs = p.contrasts.core
    _check(set(ccs) == set(FROZEN_CORE_CONTRASTS),
           f"contrasts must be {sorted(FROZEN_CORE_CONTRASTS)}; got {sorted(ccs)}")
    for name, frozen in FROZEN_CORE_CONTRASTS.items():
        got = {k: v for k, v in ccs[name].coefficients.items() if v != 0}
        _check(got == frozen, f"contrast {name} must have coefficients {frozen}; got {got}")
    for name, spec in ccs.items():
        _check(sum(spec.coefficients.values()) == 0,
               f"contrast {name} must sum to zero")
        for cid in spec.coefficients:
            _check(cid in core,
                   f"contrast {name} references {cid!r}, not a core condition "
                   f"(a diagnostic condition may never enter a core contrast)")
        expected = "interaction" if len(spec.coefficients) > 2 else "simple"
        _check(spec.kind == expected, f"contrast {name} should have kind={expected}")

    _check(p.contrasts.primary in ccs, f"primary contrast {p.contrasts.primary!r} is unknown")
    _check(set(p.contrasts.multiplicity.family) == FROZEN_HOLM_FAMILY,
           f"the Holm family must be {sorted(FROZEN_HOLM_FAMILY)}")
    for name in p.contrasts.multiplicity.family:
        _check(ccs[name].kind == "simple", f"the Holm family holds simple contrasts; {name} is not")


def _check_corpus_shape(cfg: ExperimentConfig) -> None:
    dom, corp = cfg.raw["domains"], cfg.raw["corpus"]
    n_dom = len(dom["ids"])
    _check(dom["decisions_per_domain_full"] * n_dom == corp["decisions_full"],
           "decisions_per_domain_full * domains must equal decisions_full")
    _check(dom["decisions_per_domain_pilot"] * n_dom == corp["decisions_pilot"],
           "the pilot must be evenly stratified across domains")
    _check(corp["supported_options"] == ["opt_1", "opt_2"], "semantic option ids are immutable")
    per_scenario = len(corp["supported_options"]) * corp["cells_per_group"]
    _check(corp["counterarguments_per_scenario"] == per_scenario,
           "counterarguments_per_scenario must be options x cells")
    for n_dec, key in ((corp["decisions_full"], "texts_full"),
                       (corp["decisions_pilot"], "texts_pilot")):
        expected = n_dec * corp["variants_per_decision"] * per_scenario
        _check(corp[key] == expected, f"{key} must be {expected}; got {corp[key]}")

    _check(isinstance(corp["body_sentences"], int) and corp["body_sentences"] >= 2,
           "a body needs at least two sentences to carry a premise and a conclusion")
    words = corp["scenario_words"]
    _check(0 < words["min"] < words["max"], "the scenario word band must be a real range")
    opening = corp["counterargument_opening"]
    _check(bool(opening.strip()), "the corpus-wide opening sentence must be set")
    # The opening is prepended to all eight texts of every scenario, so any
    # deliberation word in it would be reasoning content present even in the
    # plain, no-reason cells.
    deliberation = re.compile(
        r"\b(read|reading|weigh\w*|consider\w*|think\w*|reason\w*|reflect\w*|"
        r"analys\w*|analyz\w*|evaluat\w*|judg\w*|assess\w*)\b", re.IGNORECASE)
    match = deliberation.search(opening)
    _check(match is None,
           f"counterargument_opening must not suggest deliberation or reasoning: "
           f"{match.group(0)!r}" if match else "")
    pc = corp["premise_containment"]
    _check(0 < pc["min_content_word_coverage"] <= 1, "coverage is a fraction in (0, 1]")
    _check(isinstance(pc["min_word_length"], int) and pc["min_word_length"] >= 3,
           "the containment screen ignores very short words")


def _check_markers(cfg: ExperimentConfig) -> None:
    mk = cfg.raw["markers"]
    primary, exploratory = mk["primary_families"], mk["exploratory_families"]
    core = set(cfg.parsed.conditions.core)

    _check(not (set(primary) & set(exploratory)),
           "a family cannot be both primary and exploratory")
    seen: dict[str, str] = {}
    for fam, markers in {**primary, **exploratory}.items():
        _check(bool(markers), f"marker family {fam!r} is empty")
        for m in markers:
            _check(m not in seen, f"marker {m!r} appears in both {seen.get(m)!r} and {fam!r}")
            seen[m] = fam

    roles = mk["roles"]
    _check(set(roles["confirmatory"]) == set(primary),
           "the confirmatory role must name exactly the primary families")
    _check(set(roles["exploratory"]) == set(exploratory),
           "the exploratory role must name exactly the exploratory families")

    realization = mk["realization"]
    id_pattern = re.compile(realization["id_pattern"])
    families = set(cfg.marker_families())
    covered: set[str] = set()
    _check(bool(realization["registry"]), "the realization registry must not be empty")
    for rid, entry in realization["registry"].items():
        _check(id_pattern.fullmatch(rid) is not None, f"realization id {rid!r} is malformed")
        _check(entry["family"] in families,
               f"realization {rid!r} names unknown family {entry['family']!r}")
        covered.add(entry["family"])
    _check(not (families - covered),
           f"marker families with no realization: {sorted(families - covered)}")

    assign = mk["assignment"]
    _check(set(assign["styled_cells"]) | set(assign["plain_cells"]) == core,
           "styled and plain cells must partition the core conditions")
    _check(not (set(assign["styled_cells"]) & set(assign["plain_cells"])),
           "a condition cannot be both styled and plain")
    _check(assign["group_level_fields"] ==
           ["marker_family", "marker_string", "marker_realization_id"],
           "the realization group is (family, string, realization): sharing only a "
           "family would let RS - NS compare one marker or placement against another")
    inh = assign["cell_inheritance"]
    _check(set(inh) == core, "cell_inheritance must cover the four core conditions")
    for cid in assign["styled_cells"]:
        _check(inh[cid]["markers_present"] is True, f"styled cell {cid} must carry markers")
        _check(inh[cid]["instantiates"] == "marker_realization",
               f"styled cell {cid} must instantiate the group's realization")
    for cid in assign["plain_cells"]:
        _check(inh[cid]["markers_present"] is False, f"plain cell {cid} must carry no markers")
        _check(inh[cid]["instantiates"] == "paired_plain_transformation",
               f"plain cell {cid} must instantiate the paired plain transformation")

    alloc = mk["allocation"]
    for key in ("min_decisions_per_marker", "min_decisions_per_leave_one_marker_test",
                "min_decisions_per_held_out_family_test"):
        _check(isinstance(alloc[key], int) and alloc[key] >= 2,
               f"{key} must be at least 2: a leave-one-out test on one or two "
               f"decisions is not an evaluation")
    for key in ("max_share_within_one_domain", "max_share_within_one_supported_option"):
        _check(0.0 < alloc[key] <= 1.0, f"{key} must lie in (0, 1]")
    _check(alloc["max_share_within_one_domain"] >= 1.0 / len(cfg.raw["domains"]["ids"]),
           "the per-domain cap cannot be below an even split")


def _check_patterns(cfg: ExperimentConfig) -> None:
    _check(cfg.raw["forbidden"]["case_insensitive"] is True,
           "forbidden patterns compile case-insensitively")
    specs = cfg.forbidden_patterns()
    ids = [s.id for s in specs]
    _check(len(ids) == len(set(ids)), "forbidden pattern ids must be unique")
    _check(any(s.severity == "hard_fail" for s in specs), "at least one pattern must hard-fail")

    for spec, pattern in cfg.compiled_forbidden():
        _check(bool(spec.positive) and bool(spec.negative),
               f"pattern {spec.id!r} needs positive and negative fixtures")
        for example in spec.positive:
            _check(pattern.search(example) is not None,
                   f"pattern {spec.id!r} no longer catches its own positive fixture "
                   f"{example!r}: it has been weakened")
        for example in spec.negative:
            _check(pattern.search(example) is None,
                   f"pattern {spec.id!r} matches its negative fixture {example!r}: "
                   f"it is over-broad")

    # A forbidden or leakage pattern that swallowed a permitted marker would
    # make the style manipulation unusable.
    for spec, pattern in cfg.compiled_forbidden():
        for marker in cfg.permitted_markers():
            for variant in (marker, marker.lower(), marker.upper(), marker.capitalize()):
                _check(pattern.search(variant) is None,
                       f"forbidden pattern {spec.id!r} matches permitted marker {variant!r}")
    for src, pattern in cfg.compiled_leakage():
        for marker in cfg.permitted_markers():
            _check(pattern.search(marker) is None,
                   f"leakage pattern {src!r} matches permitted marker {marker!r}")
    _check(cfg.raw["leakage"]["bare_letter_rejected"] is False,
           "a bare letter is also an English article and is not rejected")


def _check_matching(cfg: ExperimentConfig) -> None:
    w = cfg.parsed.matching.words
    _check(1.0 < w.ratio_warn <= w.ratio_fail, "require 1 < ratio_warn <= ratio_fail")
    _check(w.applies_to == ["full_text", "body"],
           "the word ratio is enforced on the whole counterargument and on the "
           "manipulated body, so a shared opening cannot dilute it")
    re.compile(w.word_regex)
    s = cfg.parsed.matching.sentences
    _check(s.tolerance == 0, "sentence counts must match exactly")
    re.compile(s.split_regex)
    _check(cfg.parsed.matching.opening["level"] == "scenario",
           "the counterargument opening is stored once per scenario")


def _check_segmentation(cfg: ExperimentConfig) -> None:
    seg = cfg.raw["segmentation"]
    _check(bool(seg["library"]) and bool(seg["version"]),
           "the segmenter library and version must be pinned")
    _check(seg["clean"] is False, "the segmenter must never rewrite the text it measures")
    _check(seg["human_override"]["permitted"] is False
           and seg["human_override"]["requires_recorded_annotation"] is True,
           "a segmentation correction requires a recorded annotation; never silent")
    for kind, pattern in seg["ambiguity_patterns"].items():
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ConfigError(f"ambiguity pattern {kind!r} does not compile: {exc}") from exc
    restrictions = seg["text_restrictions"]
    for flag in ("single_paragraph", "prohibit_bullet_lists", "prohibit_numbered_lists"):
        _check(restrictions[flag] is True, f"text restriction {flag!r} must be enabled")
    _check(bool(restrictions["known_abbreviations"]), "the abbreviation list must not be empty")


def _check_prompts(cfg: ExperimentConfig) -> None:
    p = cfg.parsed.prompts
    _check(p.answer_labels == ["A", "B"], "display labels are A and B")
    orders = {(o["A"], o["B"]) for o in p.option_orders}
    _check(orders == {("opt_1", "opt_2"), ("opt_2", "opt_1")},
           "both option orders must be present and map onto semantic ids")
    _check(len({o["order_id"] for o in p.option_orders}) == 2, "order ids must be distinct")

    for field in (p.question_text, p.instruction_text, p.option_line_format):
        _check(bool(field), "the frozen question fields must be non-empty")
    _check("{label}" in p.option_line_format and "{option_text}" in p.option_line_format,
           "the option line must place the label beside the option text")
    _check([t["role"] for t in p.turn_structure] == ["user", "assistant", "user"],
           "the frozen conversation is user -> assistant -> user")
    _check("counterargument" in p.turn_structure[2]["contains"]
           and "counterargument" not in p.turn_structure[0]["contains"],
           "the counterargument belongs to turn 3, after the initial answer")

    for name, tpl in p.templates.items():
        _check(set(tpl.answer_continuation) == {"A", "B"}
               and set(tpl.answer_token_ids) == {"A", "B"},
               f"template {name}: answer fields are keyed by the display labels")
        if tpl.verification_status == "unverified":
            _check(all(v is None for v in tpl.answer_token_ids.values()),
                   f"template {name}: token ids stay null until a tokenizer is inspected")
        else:
            _check(all(v is not None for v in tpl.answer_token_ids.values())
                   and tpl.template_sha256 is not None and tpl.verified_at is not None,
                   f"template {name}: a verified template pins ids, hash and date")
        if tpl.materialization == "plain_scaffold":
            _check(tpl.template_text is not None and tpl.template_sha256 is not None,
                   f"template {name}: a plain scaffold is authored and hashed here")
            _check(sha256_of(tpl.template_text) == tpl.template_sha256,
                   f"template {name}: template_sha256 does not match template_text")
            for placeholder in ("{turns}", "{assistant_label}", "{answer_cue}"):
                _check(placeholder in tpl.template_text,
                       f"template {name}: scaffold is missing {placeholder}")
            _check(tpl.role_labels is not None
                   and set(tpl.role_labels) == {"user", "assistant"},
                   f"template {name}: a plain scaffold needs a label per role")
            _check(tpl.turn_separator is not None, f"template {name}: needs a turn separator")
        else:
            _check(tpl.template_text is None and tpl.template_sha256 is None,
                   f"template {name}: a chat template belongs to its tokenizer and is "
                   f"applied by the model adapter, not authored here")

    _check(sorted(t.applies_to for t in p.templates.values()) == ["base", "instruct"],
           "exactly one base and one instruct template")

    models = cfg.raw["models"]
    for variant in ("base", "instruct"):
        entry = models[variant]
        _check(entry["template"] in p.templates,
               f"model {variant} references unknown template {entry['template']!r}")
        _check(p.templates[entry["template"]].applies_to == variant,
               f"model {variant} references a template declared for another variant")
        if models["selection_status"] == "unfrozen":
            _check(entry["repo_id"] is None and entry["revision"] is None,
                   f"model {variant}: selection is unfrozen, so repo_id and revision "
                   f"stay null until the compatibility test")
        else:
            _check(entry["repo_id"] is not None and entry["revision"] is not None,
                   f"model {variant}: a frozen selection must pin repo_id and revision")


def _check_drafting(cfg: ExperimentConfig) -> None:
    """The generator's prompt templates and the generator's own settings.

    The template text lives in files so that it stays readable; the recorded
    hash is what fixes it. Editing a template without bumping the hash must
    fail at load, not silently change what the generator was asked.
    """
    drafting = cfg.parsed.prompts.drafting
    _check(set(drafting) == {"scenario_draft_v1", "group_draft_v1", "repair_v1"},
           "the drafting templates are the scenario, group and repair prompts")
    for name, spec in drafting.items():
        path = cfg.resolve_path(spec["path"])
        _check(path.is_file(), f"drafting template {name}: {spec['path']} not found")
        text = path.read_text(encoding="utf-8")
        _check(sha256_of(text) == spec["template_sha256"],
               f"drafting template {name}: {spec['path']} does not match its recorded hash")
        for placeholder in spec["placeholders"]:
            _check(f"${{{placeholder}}}" in text,
                   f"drafting template {name}: ${{{placeholder}}} is declared but absent")
        found = set(re.findall(r"\$\{([a-z_0-9]+)\}", text))
        _check(found == set(spec["placeholders"]),
               f"drafting template {name}: template uses {sorted(found)} but declares "
               f"{sorted(spec['placeholders'])}")
        schema = spec["response_schema"]
        _check(schema["type"] == "object" and schema["additionalProperties"] is False,
               f"drafting template {name}: the response schema must be a closed object")
        _check(set(schema["required"]) == set(schema["properties"]),
               f"drafting template {name}: every property must be required")

    _check(set(drafting["group_draft_v1"]["response_schema"]["properties"])
           == set(cfg.parsed.conditions.core),
           "the group response schema must have exactly one field per core condition")
    _check(drafting["repair_v1"]["response_schema"]
           == drafting["group_draft_v1"]["response_schema"],
           "a repair returns the same shape as the draft it repairs")

    gen = cfg.raw["models"]["generator"]
    _check(gen["backend"] in gen["supported_backends"],
           f"generator backend {gen['backend']!r} is not one of {gen['supported_backends']}")

    model = gen["model"]
    _check(bool(model["repo_id"]), "the generator repository id must be recorded")
    lowered = model["repo_id"].casefold()
    clash = [f for f in model["excluded_families"] if f.casefold() in lowered]
    _check(not clash,
           f"the generator {model['repo_id']!r} belongs to an evaluated family {clash}: "
           f"the corpus would share an ancestor with a model under test")
    _check(model["trust_remote_code"] is False,
           "remote code execution stays off for the generator")

    dec = gen["decoding"]
    _check(dec["thinking"] == "disabled",
           "the generator runs in non-thinking mode for this short structured task")
    _check(isinstance(dec["max_tokens"], int) and dec["max_tokens"] > 0,
           "max_tokens must be a positive integer")
    _check(0 < dec["temperature"] <= 1 and 0 < dec["top_p"] <= 1,
           "temperature and top_p are recorded explicitly and lie in (0, 1]")
    _check(isinstance(dec["seed"], int),
           "the seed is recorded — it does not make output reproducible across GPUs "
           "or library versions, but it belongs in the record")
    _check(dec["n"] == 1, "one response per call; no cherry-picking among samples")

    # No credential of any kind belongs in a configuration, and this generator
    # needs none: it is a local endpoint on the loopback interface.
    for forbidden in ("api_key", "api_key_env", "token", "auth"):
        _check(forbidden not in gen, f"{forbidden!r} must never appear in the configuration")
    url = gen["vllm"]["base_url"]
    _check(url.startswith("http://127.0.0.1") or url.startswith("http://localhost"),
           f"the generator endpoint must be local; got {url!r}")
    _check(gen["vllm"]["require_offline_env"] == {"HF_HUB_OFFLINE": "1"},
           "offline mode is required so that a missing model is an error, not a download")

    # Authorisation to run is deliberately absent from the config: it lives in
    # an explicit argument plus an environment variable at the call site.
    _check("live_calls_enabled" not in gen,
           "run authorisation does not belong in the experiment configuration")

    alloc = cfg.raw["markers"]["allocation"]
    confirmatory = cfg.raw["markers"]["roles"]["confirmatory"]
    inventory = cfg.raw["markers"]["primary_families"]
    _check(alloc["pilot_families"] == confirmatory,
           "the pilot allocates the confirmatory families; the exploratory family "
           "is deferred to the full corpus")
    _check(set(alloc["pilot_strings"]) == set(alloc["pilot_families"]),
           "every pilot family needs its pilot strings")
    for family, strings in alloc["pilot_strings"].items():
        _check(len(strings) == len(set(strings)) == 2,
               f"{family}: the pilot uses exactly two distinct strings")
        unknown = [s for s in strings if s not in inventory[family]]
        _check(not unknown, f"{family}: {unknown} are not in the marker inventory")

    rep = cfg.raw["corpus"]["repair"]
    _check(rep["max_calls_per_group"] == rep["max_repair_calls"] + 1,
           "the call budget is the original attempt plus the repairs")
    _check(rep["on_exhaustion"] == "needs_manual_review",
           "an exhausted group is marked for manual review, never silently accepted")


def _check_splits_and_probes(cfg: ExperimentConfig) -> None:
    raw = cfg.raw
    sp, corp, dom = raw["splits"], raw["corpus"], raw["domains"]
    seeds = raw["determinism"]["seeds"]

    _check(abs(sum(sp["fractions"].values()) - 1.0) < 1e-9, "split fractions must sum to 1.0")
    _check(sp["grouping_unit"] == "decision_id", "splits are grouped on decision_id")
    _check(sp["seed_ref"] in seeds, "splits.seed_ref must name a declared seed")
    _check(sp["stratify_by"] == ["domain"],
           "only decision-level labels may stratify the corpus split; marker family "
           "belongs to a (scenario, supported_option) group, not to a decision")
    _check(sp["marker_family_balance"]["is_stratification_key"] is False,
           "marker family is an allocation constraint, not a stratification key")
    _check(sum(sp["decisions"].values()) == corp["decisions_full"],
           "split decision counts must sum to decisions_full")
    for name, frac in sp["fractions"].items():
        expected = round(frac * corp["decisions_full"])
        _check(sp["decisions"][name] == expected,
               f"split {name}: {frac} of {corp['decisions_full']} decisions is {expected}, "
               f"but {sp['decisions'][name]} is declared")

    pe = raw["probe_evaluation"]
    required = {"grouped_held_out_decisions", "leave_one_domain_out", "held_out_marker_family",
                "leave_one_marker_out", "leave_one_realization_template_out"}
    _check(required <= set(pe["schemes"]), f"probe evaluation must define {sorted(required)}")
    _check(pe["schemes"]["leave_one_domain_out"]["folds"] == len(dom["ids"]),
           "leave-one-domain-out needs one fold per domain")
    for name, scheme in pe["schemes"].items():
        # Keeping the four-cell group together is necessary but insufficient: a
        # sibling group from the same decision would otherwise leak into training.
        _check(scheme.get("no_decision_overlap") is True,
               f"probe scheme {name} must declare no_decision_overlap")
        _check(scheme.get("grouping_unit") == "decision_id",
               f"probe scheme {name} must be grouped on decision_id")
    for name in ("held_out_marker_family", "leave_one_marker_out",
                 "leave_one_realization_template_out"):
        _check(pe["schemes"][name]["unit_kept_together"] == "group",
               f"scheme {name} must keep the complete four-cell group together")
        ref = pe["schemes"][name]["min_decisions_ref"]
        node: Any = raw
        for part in ref.split("."):
            _check(part in node, f"scheme {name} references missing path {ref!r}")
            node = node[part]
        _check(isinstance(node, int) and node >= 2, f"scheme {name} minimum support must be >= 2")


def _check_analysis(cfg: ExperimentConfig) -> None:
    raw, nt = cfg.raw, cfg.parsed.near_tie
    seeds = raw["determinism"]["seeds"]

    _check(0.5 < nt.probability_threshold < 1.0,
           "the near-tie probability threshold must lie in (0.5, 1)")
    expected = math.log(nt.probability_threshold / (1.0 - nt.probability_threshold))
    _check(abs(nt.tau_logit - expected) < TAU_TOLERANCE,
           f"tau_logit {nt.tau_logit!r} does not match log(p/(1-p)) = {expected!r}")

    an = raw["analysis"]
    _check(an["cluster_unit"] == "decision_id", "the bootstrap clusters on decision_id")
    _check(an["multiplicity"] == cfg.parsed.contrasts.multiplicity.method,
           "the multiplicity rule is stated twice and disagrees")
    _check(an["bootstrap"]["seed_ref"] in seeds, "the bootstrap seed must be declared")

    mech = raw["mechanistic"]
    core = set(cfg.parsed.conditions.core)
    _check(mech["layers"]["indexing"] == "0..n_layers-1", "blocks are indexed 0..n_layers-1")
    _check(mech["layers"]["primary_hook"] == "resid_post", "the primary hook is resid_post")
    _check(mech["layers"]["embeddings_name"] == "embed",
           "embeddings are named 'embed', never a negative block index")
    for pair in mech["subset"]["contrasts"]:
        for cid in pair:
            _check(cid in core, f"the mechanistic subset references non-core {cid!r}")
    _check(mech["subset"]["direction_field"] == "counter_target_opt",
           "the patched direction is resolved at runtime")
    _check(mech["subset"]["n_decisions"] <= raw["corpus"]["decisions_full"],
           "the mechanistic subset cannot exceed the corpus")


def _check_annotation_and_review(cfg: ExperimentConfig) -> None:
    raw = cfg.raw
    core = cfg.parsed.conditions.core
    ann, levels = raw["annotation"], raw["annotation"]["levels"]

    _check(set(levels) == {"item", "pair", "scenario"},
           "annotations are stored at item, pair and scenario level")
    seen: dict[str, str] = {}
    for level, spec in levels.items():
        _check(bool(spec["ratings"]), f"annotation level {level!r} has no ratings")
        _check("annotator_id" in spec["keys"], f"level {level!r} must be keyed by annotator")
        for r in spec["ratings"]:
            _check(r not in seen,
                   f"rating {r!r} is declared at both {seen.get(r)!r} and {level!r}: "
                   f"a judgement belongs to exactly one level")
            seen[r] = level
    _check("condition" not in levels["pair"]["keys"]
           and "condition" not in levels["scenario"]["keys"],
           "only item-level annotations are keyed by condition")
    for pair in levels["pair"]["pairs"]:
        _check(len(pair) == 2 and all(c in core for c in pair),
               f"annotation pair {pair} must name two core conditions")
        _check(core[pair[0]].reason == core[pair[1]].reason,
               f"proposition preservation is judged within a reason level; {pair} crosses it")

    prov = ann["rating_provenance"]
    buckets = {name: set(values) for name, values in prov.items()}
    for a, b in ((x, y) for x in buckets for y in buckets if x < y):
        _check(not (buckets[a] & buckets[b]),
               f"ratings claimed by both {a!r} and {b!r}: {sorted(buckets[a] & buckets[b])}")
    declared: set[str] = set().union(*buckets.values()) if buckets else set()
    _check(declared == set(seen),
           f"every rating needs provenance and exactly one level "
           f"(missing provenance: {sorted(set(seen) - declared)}; "
           f"provenance without a level: {sorted(declared - set(seen))})")

    sub = ann["reliability_subsample"]
    _check(sub["seed_ref"] in raw["determinism"]["seeds"],
           "the reliability subsample seed must be declared")
    _check(isinstance(sub["min_sibling_separation"], int)
           and sub["min_sibling_separation"] >= 0,
           "min_sibling_separation must be a non-negative integer")

    # A stratified sample must be able to reach every stratum it claims to
    # stratify by; otherwise the description in the thesis would be false.
    facets = {
        "domain": len(raw["domains"]["ids"]),
        "condition": len(core),
        "marker_family": len(raw["markers"]["roles"]["confirmatory"]),
        "supported_option": len(raw["corpus"]["supported_options"]),
    }
    unknown = set(sub["stratify_by"]) - set(facets)
    _check(not unknown, f"unknown stratification facet(s): {sorted(unknown)}")
    _check(not (set(sub["stratify_by"]) & set(sub["balance_marginally"])),
           "a facet is either a crossed stratum or balanced marginally, never both")
    n_strata = math.prod(facets[f] for f in sub["stratify_by"])
    sampled = round(raw["corpus"]["texts_pilot"] * sub["fraction"])
    _check(n_strata <= sampled,
           f"{n_strata} strata cannot be covered by a {sampled}-item sample: "
           f"move a facet from stratify_by to balance_marginally")

    blinded = raw["review"]["blinded_view"]
    _check(blinded["applies_to"] == "reliability_sample_only",
           "only the independent reliability packets are blinded")
    _check(blinded["option_labels"] == ["P", "Q"],
           "blinded option labels are P and Q, never the experiment's A and B")
    _check(set(blinded["support_direction_answers"]) == {"P", "Q", "unclear"},
           "a blinded annotator must be able to answer 'unclear'")
    _check(set(blinded["levels"]) == {"item", "pair", "scenario"},
           "blinded packets are produced at all three levels")
    _check(blinded["key_directory"] == "blind_key",
           "the unblinding key lives in its own directory")


def _check_provenance(cfg: ExperimentConfig) -> None:
    prov = cfg.raw["corpus_provenance"]
    src, gen = prov["source_references"], prov["generation_metadata"]
    _check("constructed" in src["scenario_level_type_values"],
           "a scenario must be able to declare itself constructed")
    for field in ("dataset_name", "dataset_version", "source_item_id", "source_url",
                  "access_date", "reuse_licence"):
        _check(field in src["fields"], f"the source-reference structure must carry {field!r}")
    _check(src["fields"]["dataset_name"]["required"] is True,
           "a source reference must name its dataset")
    for field in ("generator_model", "generator_model_revision", "prompt_hash",
                  "generation_parameters", "seed", "generated_at"):
        _check(field in gen["fields"], f"generation metadata must carry {field!r}")
    _check(not (set(src["fields"]) & set(gen["fields"])),
           "source-reference and generation-metadata fields must not overlap")


def _check_topics(cfg: ExperimentConfig) -> None:
    t = cfg.raw["topics"]
    corp, dom = cfg.raw["corpus"], cfg.raw["domains"]
    _check(len(t["variants"]) == corp["variants_per_decision"],
           "topics.variants must list one key per scenario variant")
    lo, hi = t["facts_per_option"]["min"], t["facts_per_option"]["max"]
    _check(1 <= lo <= hi, "facts_per_option needs 1 <= min <= max")
    _check(t["curated_per_domain_for_drafting"] == dom["decisions_per_domain_pilot"],
           "curated_per_domain_for_drafting must equal the pilot's decisions per domain")
    for field, limit in t["max_words"].items():
        _check(isinstance(limit, int) and limit > 0, f"topics.max_words.{field} must be positive")
    for pattern in t["specificity_patterns"]:
        re.compile(pattern)
    _check(isinstance(t["source_overlap_min_words"], int) and t["source_overlap_min_words"] >= 3,
           "source_overlap_min_words must be an integer of at least 3")


def load_config(path: str | Path) -> ExperimentConfig:
    """Load, type-check and invariant-check a configuration."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} does not contain a YAML mapping")
    cfg = ExperimentConfig(raw=raw, parsed=RawConfig.model_validate(raw), path=path)

    _check_location(cfg)
    _check_conditions_and_contrasts(cfg)
    _check_corpus_shape(cfg)
    _check_markers(cfg)
    _check_patterns(cfg)
    _check_matching(cfg)
    _check_segmentation(cfg)
    _check_prompts(cfg)
    _check_drafting(cfg)
    _check_splits_and_probes(cfg)
    _check_analysis(cfg)
    _check_annotation_and_review(cfg)
    _check_provenance(cfg)
    _check_topics(cfg)
    return cfg
