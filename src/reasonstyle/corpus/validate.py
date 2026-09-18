"""Corpus validation: the machine-checkable rules, and nothing beyond them.

What this module can establish is *form*: presence, absence, counts, structural
consistency, and whether a phrase matches a pattern. What it cannot establish is
*substance*. It cannot tell whether a reason is relevant, valid under the
scenario facts, or genuinely supporting; whether a no-reason cell is truly
reason-free; whether RS and RP express the same propositions; whether text is
natural; or what a marker signals about speaker commitment.

Those are emitted as ``human_review`` findings, **unconditionally**, for every
item, pair, group and scenario. A corpus with zero errors is *machine-valid*,
never "validated": approval additionally requires those judgements to be
recorded by a reviewer.

The module is model-agnostic — it measures text, and knows nothing of
tokenizers, checkpoints or inference backends.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Literal

from ..config import ExperimentConfig
from .store import corpus_content_hash
from .findings import Finding, ValidationReport
from .schemas import CORE_CONDITIONS, SEMANTIC_OPTIONS, Condition, Measurements
from .schemas import ScenarioRecord, SegmenterRef
from .segmentation import Segmenter

__all__ = ["CorpusScope", "FOUR_WAY", "PAIRWISE", "embedded_option_text",
           "endorsement_text", "matching_mode",
           "validate_corpus", "validate_group",
           "validate_scenario_text", "with_measurements"]

CorpusScope = Literal["fixture", "pilot", "full"]

#: Judgements no lexical validator can make. Emitted for every relevant unit so
#: that a clean machine run is never mistaken for a validated corpus.
HUMAN_REVIEW_CODES: dict[str, str] = {
    "H_SUPPORT_DIRECTION": "confirm this text supports its declared semantic option",
    "H_SUBSTANTIVE_SUPPORT": "judge whether a relevant premise genuinely supports the option",
    "H_NATURALNESS": "judge naturalness under exact sentence-count matching",
    "H_PRAGMATIC_COMMITMENT": "rate perceived speaker commitment and unstated support",
    # "No reason" means no TASK-RELEVANT reason. A short self-referential
    # clause is allowed, and is often needed to host the marker; what it may
    # not contain is listed here, so the judgement is not read as a literal ban
    # on every subordinate clause.
    "H_NO_REASON_INTEGRITY":
        "confirm this no-reason cell gives no task-relevant support: no scenario fact, no "
        "consequence of either option, no value or trade-off, no evidence, authority or "
        "expertise, and no new factual claim (a clause referring only to the speaker's own "
        "preference is acceptable)",
    "H_PROPOSITION_PRESERVATION": "confirm both cells express the same substantive claims",
    "H_REALIZATION_YIELDS_REASON_FREE_NS": "confirm this realization can yield a genuinely reason-free NS",
    "H_SCENARIO_VALIDITY": "confirm both options are feasible, non-dominated and not value-weighted",
}

_PAIRS: tuple[tuple[Condition, Condition], ...] = (("RS", "RP"), ("NS", "NP"))


def _normalized(text: str) -> str:
    """Case- and whitespace-insensitive form, for substring comparison."""
    return " ".join(text.casefold().split())


def _content_tokens(body: str, word_re: re.Pattern[str], marker: str | None,
                    permitted: set[str]) -> Counter[str]:
    """Content words of one cell body, as a multiset.

    The assigned marker is removed first — its presence in the styled cell and
    absence from the plain cell is the difference the pair is *supposed* to
    have — and so are the configured function words, which are what a
    styled-to-plain transformation is permitted to change. Multiplicity is
    kept: a content word used twice in one body and once in the other is a
    difference.
    """
    tokens = Counter(w.casefold() for w in word_re.findall(body))
    for token in (word_re.findall(marker) if marker else ()):
        tokens[token.casefold()] -= 1        # one occurrence of the marker phrase
    return Counter({w: n for w, n in tokens.items() if n > 0 and w not in permitted})


class _Collector:
    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def add(self, code: str, severity: str, message: str, scope: str, **loc) -> None:
        self.findings.append(
            Finding(code=code, severity=severity, message=message, scope=scope, **loc)  # type: ignore[arg-type]
        )


def _count_words(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


def _containment(body: str, scenario_text: str, pattern: re.Pattern[str],
                 spec: dict) -> tuple[list[str], float]:
    """Which content words of a reason cell are absent from its scenario.

    Frame vocabulary — the endorsement and self-reference wording every cell
    shares — is configured and never counted: it is not scenario content, so
    its absence says nothing about premise containment. Very short words are
    skipped for the same reason.
    """
    ignore = set(spec["ignore_words"])
    minimum = spec["min_word_length"]
    in_scenario = {w.casefold() for w in pattern.findall(scenario_text)}
    content = [w for w in (m.casefold() for m in pattern.findall(body))
               if len(w) >= minimum and w not in ignore]
    if not content:
        return [], 1.0
    unmatched = sorted({w for w in content if w not in in_scenario})
    matched = sum(1 for w in content if w in in_scenario)
    return unmatched, matched / len(content)


def _measure(opening: str, block, condition: Condition,
             segmenter: Segmenter, word_re: re.Pattern[str]) -> tuple[Measurements, tuple[str, ...]]:
    body = block.cells[condition].body
    full = f"{opening} {body}"
    seg_full, seg_body = segmenter.segment(full), segmenter.segment(body)
    info = segmenter.info
    measurements = Measurements(
        word_count_full=_count_words(full, word_re),
        word_count_body=_count_words(body, word_re),
        sentence_count_full=seg_full.count,
        sentence_count_body=seg_body.count,
        segmenter=SegmenterRef(library=info.library, version=info.version, language=info.language),
        ambiguity_kinds=seg_full.ambiguity_kinds(),
    )
    return measurements, seg_full.ambiguity_kinds()


def with_measurements(
    records: Sequence[ScenarioRecord], cfg: ExperimentConfig, segmenter: Segmenter
) -> list[ScenarioRecord]:
    """Return copies of ``records`` with every cell's measurements filled in.

    Measurements are computed here and never written by hand.
    """
    word_re = re.compile(cfg.parsed.matching.words.word_regex)
    out: list[ScenarioRecord] = []
    for record in records:
        blocks = {}
        for option, block in record.counterarguments.items():
            cells = {}
            for condition, cell in block.cells.items():
                measurements, _ = _measure(record.counterargument_opening, block,
                                           condition, segmenter, word_re)
                cells[condition] = cell.model_copy(update={"measurements": measurements})
            blocks[option] = block.model_copy(update={"cells": cells})
        out.append(record.model_copy(update={"counterarguments": blocks}))
    return out


def validate_scenario_text(
    scenario_text: str,
    cfg: ExperimentConfig,
    segmenter: Segmenter,
    *,
    loc: dict | None = None,
) -> list[Finding]:
    """The machine rules a scenario text can be held to on its own, at drafting
    time, before any counterargument exists.

    The same codes and the same configured patterns the corpus validator uses:
    the drafting check is never weaker than the check the text faces later. It
    is stricter in one respect — forbidden authority and evidence language is
    screened in the scenario itself, not only in counterarguments — because a
    scenario that smuggles in an appeal to expertise should be caught before it
    is built upon. Whether the scenario is genuinely underdetermined,
    self-contained and neutral stays a human judgement
    (``H_SCENARIO_VALIDITY``).
    """
    c = _Collector()
    loc = dict(loc or {})
    word_re = re.compile(cfg.parsed.matching.words.word_regex)
    band = cfg.raw["corpus"]["scenario_words"]
    restrictions = cfg.raw["segmentation"]["text_restrictions"]

    words = _count_words(scenario_text, word_re)
    if not band["min"] <= words <= band["max"]:
        c.add("W_SCENARIO_WORDS", "warning",
              f"the scenario is {words} words, outside the drafting band "
              f"{band['min']}-{band['max']}",
              "scenario", detail={"words": words, **band}, **loc)

    for pattern, compiled in cfg.compiled_leakage():
        match = compiled.search(scenario_text)
        if match is not None:
            c.add("E_LABEL_LEAKAGE", "error",
                  f"the scenario references a display label: {match.group(0)!r}",
                  "scenario", detail={"pattern": pattern, "match": match.group(0)}, **loc)

    for spec, pattern in cfg.compiled_forbidden():
        match = pattern.search(scenario_text)
        if match is None:
            continue
        severity = "error" if spec.severity == "hard_fail" else "warning"
        c.add(f"{'E' if severity == 'error' else 'W'}_FORBIDDEN", severity,
              f"{spec.family} phrase {match.group(0)!r} matched {spec.id}",
              "scenario",
              detail={"pattern_id": spec.id, "family": spec.family,
                      "match": match.group(0)}, **loc)

    prohibited = {"bullet_list": "prohibit_bullet_lists",
                  "numbered_list": "prohibit_numbered_lists",
                  "line_break": "single_paragraph"}
    for kind in segmenter.segment(scenario_text).ambiguity_kinds():
        if kind in prohibited and restrictions[prohibited[kind]]:
            c.add("E_PROHIBITED_FORMATTING", "error",
                  f"the scenario contains {kind.replace('_', ' ')}, which the config prohibits",
                  "scenario", detail={"kind": kind}, **loc)
        else:
            c.add("W_AMBIGUOUS_SEGMENTATION", "warning",
                  f"{kind.replace('_', ' ')} present; the segmenter is unreliable here and the "
                  f"sentence count needs human confirmation",
                  "scenario", detail={"kind": kind}, **loc)

    c.add("H_SCENARIO_VALIDITY", "human_review",
          HUMAN_REVIEW_CODES["H_SCENARIO_VALIDITY"], "scenario", **loc)
    return c.findings


#: The matching regimes. ``four_way`` is v1: all four cells within one word
#: ratio and one sentence count. ``pairwise`` is v2: RS/RP matched to each other
#: and NS/NP to each other, with the cross-pair comparison recorded but never
#: failed. A configuration that names neither gets ``four_way``, so every v1
#: configuration keeps its exact behaviour and its exact codes.
FOUR_WAY, PAIRWISE = "four_way", "pairwise"


def matching_mode(cfg: ExperimentConfig) -> str:
    """Which matching regime this configuration asks for."""
    return cfg.raw["matching"].get("compare", FOUR_WAY)


def _pairs_of(cfg: ExperimentConfig) -> list[tuple[Condition, Condition]]:
    return [tuple(pair) for pair in cfg.parsed.matching.pair_content.compare_pairs]


def _pair_id(pair) -> str:
    return "/".join(pair)


def _strip_once(haystack: str, needle: str) -> tuple[str, int]:
    """``(remainder, occurrences)`` for a normalised substring removal.

    Comparison is on normalised text — case folded, whitespace collapsed — so a
    capitalisation or spacing difference is not mistaken for a missing clause.
    """
    normalised_needle = _normalized(needle)
    if not normalised_needle:
        return _normalized(haystack), 0
    normalised = _normalized(haystack)
    occurrences = normalised.count(normalised_needle)
    return (normalised.replace(normalised_needle, " ", 1).strip() if occurrences
            else normalised), occurrences


def _residue(body: str, endorsement: str, marker: str | None,
             word_re: re.Pattern[str]) -> list[str]:
    """The content words left once the endorsement and the marker are removed.

    What v2 uses to decide, mechanically, that a no-premise cell states no
    premise: if nothing remains, there is nowhere for a fact, a consequence, a
    value or a new claim to live. It is a structural check, not a semantic one,
    and it replaces nothing a person judges — H_NO_REASON_INTEGRITY stays
    unconditional.
    """
    remainder, _ = _strip_once(body, endorsement)
    if marker:
        remainder, _ = _strip_once(remainder, marker)
    return [w for w in word_re.findall(remainder)]


def embedded_option_text(option_text: str) -> str:
    """An option line, as it reads inside a sentence rather than as one.

    The topic bank states an option as a standalone imperative; the endorsement
    embeds it. The trailing stop goes, and the first letter is lowered only when
    it is ordinary sentence capitalisation — a word whose second letter is
    already upper case is an acronym or a name and is left alone.
    """
    text = option_text.strip().rstrip(".").strip()
    if len(text) > 1 and text[0].isupper() and not text[1].isupper():
        text = text[0].lower() + text[1:]
    return text


def endorsement_text(option_text: str, cfg: ExperimentConfig) -> str | None:
    """The fixed endorsement clause for one supported option.

    ``None`` when the configuration defines no template, which is what every v1
    configuration means: there the shared clause was an instruction to the model
    rather than a string anything could check. Derived here, in the corpus
    layer, so corpus validation runs the v2 checks without any caller having to
    remember to pass the clause in.
    """
    from string import Template
    template = cfg.raw["corpus"].get("endorsement_template")
    if not template:
        return None
    return Template(template).substitute(
        supported_option_text=embedded_option_text(option_text))


def _duplicate_key(cfg: ExperimentConfig, scenario_text: str, opening: str,
                   body: str) -> str:
    """What has to be unique: the whole experimental input, or just the reply.

    ``rendered_experimental_input`` (v2) keys on the scenario as well, because
    that is what the model is given. A configuration that says nothing keys on
    the rendered counterargument alone, which is exactly what v1 did.
    """
    spec = cfg.raw["corpus"].get("duplicate_text") or {}
    rendered = f"{opening} {body}"
    if spec.get("compare") == "rendered_experimental_input":
        return f"{_normalized(scenario_text)}\u241f{_normalized(rendered)}"
    return rendered


def _expected_repetition(cfg: ExperimentConfig, first_seen: str, here: str,
                         condition: Condition, block) -> bool:
    """Whether this repetition is the one the design predicts, and only that.

    Narrow on purpose. Every clause has to hold: a configured condition (NS or
    NP), the same decision, the same supported option, the same condition, the
    same marker, and a *different* variant. Anything else — a repeat inside one
    variant, across decisions, across options, or in a reason-present cell — is
    a collision and stays ``E_DUPLICATE_TEXT``.
    """
    spec = (cfg.raw["corpus"].get("duplicate_text") or {}).get("expected_repetition")
    if not spec or condition not in set(spec.get("conditions") or ()):
        return False
    try:
        first_scenario, first_option, first_condition = first_seen.split("/")
    except ValueError:                                      # pragma: no cover - defensive
        return False
    this_scenario, this_option, this_condition = here.split("/")
    first_decision, _, first_variant = first_scenario.rpartition("_v")
    this_decision, _, this_variant = this_scenario.rpartition("_v")
    checks = {
        "same_decision": first_decision == this_decision and bool(first_decision),
        "same_supported_option": first_option == this_option,
        "same_condition": first_condition == this_condition,
        "different_variant": first_variant != this_variant,
        # The marker is group-level and identical in both by construction when
        # the allocation says so; a differing marker would make the two bodies
        # differ anyway, so this holds trivially where the texts matched.
        "same_marker": True,
    }
    return all(value for name, value in checks.items() if spec.get(name))


def _pairwise_matching(c, cfg: ExperimentConfig, measurements, gloc: dict, words_cfg,
                       marker: str | None, word_re: re.Pattern[str]) -> None:
    """v2 matching: RS to RP, NS to NP, and the cross-pair difference recorded.

    The v1 rule asked all four cells to match in length. A reason-present cell
    carries a premise and a no-premise cell may not, so the only way to satisfy
    it was to pad the no-premise cells — and padding is content, which is what
    the no-premise condition exists not to have. v2 matches within each pair,
    where the minimal edit is a single connective, and reports the cross-pair
    difference as a **measurement rather than a failure**: it is a property of
    the design, not a defect in a draft.
    """
    expected = cfg.raw["corpus"].get("body_sentences_by_pair") or {}
    for pair in _pairs_of(cfg):
        pair_id = _pair_id(pair)
        full = {k: measurements[k].sentence_count_full for k in pair}
        if len(set(full.values())) != 1:
            c.add("E_PAIR_SENTENCE_COUNT_MISMATCH", "error",
                  f"{pair_id} must have equal sentence counts; got {full}",
                  "pair", **{**gloc, "detail": {"pair": list(pair), "counts": full}})
        else:
            body = {k: measurements[k].sentence_count_body for k in pair}
            actual = next(iter(set(body.values())))
            want = expected.get(pair_id)
            if want is not None and actual != want:
                c.add("E_PAIR_BODY_SENTENCE_COUNT", "error",
                      f"each body of {pair_id} is exactly {want} sentence(s); these are "
                      f"{actual}", "pair",
                      **{**gloc, "detail": {"pair": list(pair), "expected": want,
                                            "actual": actual}})
        # Length inside a pair is checked as an EXACT WORD BUDGET, not a ratio.
        # The two cells of a pair are supposed to differ by the assigned marker
        # and nothing else, so the permitted difference is the marker's own
        # length plus the configured function-word allowance — a fixed number of
        # words, not a proportion. A ratio cannot express that: NS is NP plus a
        # three-word marker, which is a 25% difference against a twelve-word
        # endorsement and a 6% difference against a fifty-word one, while being
        # exactly the same manipulation. Enforcing a ratio here would penalise
        # short endorsements for being short and would push a drafter to pad the
        # plain cell, which is the v1 failure this design exists to remove.
        marker_words = len(word_re.findall(marker or ""))
        # The budget is the marker and nothing else. A configured
        # permitted-difference token would widen it, so a design that needs one
        # has to demonstrate it and declare it; v2 declares none.
        allowance = marker_words + len(cfg.parsed.matching.pair_content.permitted_differences)
        for scope_name, key in (("full_text", "word_count_full"),
                                ("body", "word_count_body")):
            if scope_name not in (words_cfg.applies_to or ["full_text"]):
                continue
            values = {k: getattr(measurements[k], key) for k in pair}
            delta = abs(values[pair[0]] - values[pair[1]])
            ratio = (max(values.values()) / min(values.values())
                     if min(values.values()) else float("inf"))
            detail = {"pair": list(pair), "measurement": scope_name, "counts": values,
                      "delta": delta, "permitted_delta": allowance,
                      "marker_words": marker_words, "ratio": round(ratio, 4)}
            if delta > allowance:
                c.add(f"E_PAIR_WORD_DELTA_{scope_name.upper()}", "error",
                      f"{pair_id} differ by {delta} {scope_name} word(s); the marker "
                      f"{marker!r} is {marker_words} word(s) and the permitted difference "
                      f"is {allowance}. The two cells of a pair differ by the marker and "
                      f"nothing else.", "pair", **{**gloc, "detail": detail})
            else:
                c.add(f"I_PAIR_WORD_DELTA_{scope_name.upper()}", "info",
                      f"{pair_id} differ by {delta} {scope_name} word(s) (ratio "
                      f"{ratio:.3f}); within the {allowance}-word budget the marker "
                      f"allows. The ratio is recorded because the analysis needs it, "
                      f"not because it is a threshold here.", "pair",
                      **{**gloc, "detail": detail})

    # The cross-pair comparison: recorded for every group, never a failure. A
    # reason-present body is longer than a no-premise body by construction, and
    # the analysis has to know by how much — that difference is a covariate to
    # report, not an imbalance to repair away.
    for scope_name, key in (("full_text", "word_count_full"), ("body", "word_count_body")):
        values = {k: getattr(v, key) for k, v in measurements.items()}
        lo, hi = min(values.values()), max(values.values())
        c.add("I_CROSS_PAIR_WORD_RATIO", "info",
              f"across the two pairs the {scope_name} word ratio is "
              f"{(hi / lo if lo else float('inf')):.3f}; under pairwise matching this is "
              f"recorded, not failed, because a premise-bearing cell is longer than a "
              f"premise-free one by construction",
              "group", **{**gloc, "detail": {"measurement": scope_name, "counts": values,
                                             "ratio": round(hi / lo if lo else 0.0, 4)}})


def _v2_structure(c, cfg: ExperimentConfig, block, endorsement: str | None,
                  word_re: re.Pattern[str], gloc: dict) -> None:
    """The v2 cell shapes, checked mechanically.

    RS is premise + marker + endorsement, RP the same without the marker, NS
    marker + endorsement, NP the endorsement alone. Two things follow that a
    machine *can* decide, and they are what these checks are:

    * the fixed endorsement occurs exactly once in every cell, so no cell
      restates its preference or adds a second commitment;
    * once the endorsement and the marker are removed, nothing is left in NS or
      NP. A cell with no remaining words has nowhere to put a scenario fact, a
      consequence, a value, a trade-off, evidence or a new claim.

    The second replaces no human judgement. H_NO_REASON_INTEGRITY stays
    unconditional: a person still confirms that what the cell *does* say
    supplies no task-relevant support.
    """
    if not endorsement:
        # Never a note. A pairwise design is defined by the fixed endorsement:
        # without it the residue check cannot run, and a group would be reported
        # as clean on rules that were never applied to it.
        c.add("E_ENDORSEMENT_NOT_SUPPLIED", "error",
              "this configuration matches pairwise, so every cell is built around a "
              "fixed endorsement clause — but none was supplied to the validator. The "
              "endorsement and residue checks could not run, and a group that has not "
              "been checked is not a group that passed.", "group", **gloc)
        return
    # The first configured pair is the reason-present one (RS/RP); the second
    # is the no-explicit-premise pair (NS/NP). A load-time check pins the order.
    pairs = _pairs_of(cfg)
    premise_cells = set(pairs[0])
    no_premise_cells = {cell for pair in pairs[1:] for cell in pair}

    for condition in CORE_CONDITIONS:
        cell = block.cells[condition]
        cloc = {**gloc, "condition": condition}
        _, occurrences = _strip_once(cell.body, endorsement)
        if occurrences != 1:
            c.add("E_ENDORSEMENT_NOT_EXACTLY_ONCE", "error",
                  f"the fixed endorsement must appear exactly once in every cell; it "
                  f"appears {occurrences} time(s) here. Endorsement: {endorsement!r}",
                  "cell", detail={"endorsement": endorsement,
                                  "occurrences": occurrences}, **cloc)
        marker = block.marker_string if cell.markers_present else None
        residue = _residue(cell.body, endorsement, marker, word_re)
        if condition in no_premise_cells and residue:
            c.add("E_NO_PREMISE_CELL_HAS_EXTRA_CONTENT", "error",
                  f"{condition} is the endorsement"
                  + (f" and the marker {block.marker_string!r}" if marker else "")
                  + f", and nothing else; these words remain: {residue}. Any of them "
                  f"could carry a premise, which is what this condition may not have.",
                  "cell", detail={"residue": residue, "endorsement": endorsement,
                                  "marker": marker}, **cloc)
        if condition in premise_cells and not residue:
            c.add("E_REASON_CELL_HAS_NO_PREMISE", "error",
                  f"{condition} states the endorsement and nothing else, so it supplies "
                  f"no premise at all; a reason-present cell must give one",
                  "cell", detail={"endorsement": endorsement}, **cloc)


def validate_group(
    scenario_text: str,
    opening: str,
    block,
    cfg: ExperimentConfig,
    segmenter: Segmenter,
    *,
    loc: dict | None = None,
    seen_texts: dict[str, str] | None = None,
    endorsement: str | None = None,
) -> list[Finding]:
    """Every machine rule that applies to ONE four-cell group, plus that group's
    unconditional human-review codes.

    Extracted so a freshly drafted group can be checked before the rest of its
    scenario exists, and checked again once the corpus is assembled — by the
    same code, so a draft can never pass a weaker rule than the corpus it later
    joins. ``validate_corpus`` calls this for every group; what a single group
    cannot see (marker allocation across the corpus, duplicates in other
    scenarios, fixture shape) stays there.

    ``seen_texts`` carries cross-group duplicate detection when a corpus is
    being validated; on its own a group is compared only against itself.
    """
    c = _Collector()
    loc = dict(loc or {})
    scenario_id = loc.get("scenario_id")
    gloc = {**loc, "supported_option": block.supported_option}
    seen_texts = {} if seen_texts is None else seen_texts
    word_re = re.compile(cfg.parsed.matching.words.word_regex)
    families = cfg.marker_families()
    realizations = cfg.marker_realizations()
    marker_res = {m: re.compile(rf"\b{re.escape(m)}\b", re.IGNORECASE)
                  for m in cfg.permitted_markers()}
    words_cfg = cfg.parsed.matching.words
    restrictions = cfg.raw["segmentation"]["text_restrictions"]
    body_sentences = cfg.raw["corpus"]["body_sentences"]
    containment = cfg.raw["corpus"]["premise_containment"]
    pair_content = cfg.parsed.matching.pair_content
    permitted_differences = {w.casefold() for w in pair_content.permitted_differences}

    if set(block.cells) != set(CORE_CONDITIONS):
        c.add("E_MISSING_CELL", "error",
              f"expected the four core cells, got {sorted(block.cells)}", "group", **gloc)
        return c.findings                      # nothing else can be measured

    # -- realization group (D3b) ------------------------------------
    if block.marker_realization_id not in realizations:
        c.add("E_REALIZATION_UNKNOWN", "error",
              f"realization {block.marker_realization_id!r} is not in the registry",
              "group", detail={"realization": block.marker_realization_id}, **gloc)
    elif realizations[block.marker_realization_id]["family"] != block.marker_family:
        c.add("E_REALIZATION_FAMILY_MISMATCH", "error",
              f"realization {block.marker_realization_id!r} belongs to family "
              f"{realizations[block.marker_realization_id]['family']!r}, "
              f"not {block.marker_family!r}",
              "group", **gloc)

    if block.marker_family not in families:
        c.add("E_UNKNOWN_MARKER_FAMILY", "error",
              f"marker family {block.marker_family!r} is not in the inventory",
              "group", **gloc)
    elif block.marker_string not in families[block.marker_family]:
        c.add("E_MARKER_NOT_IN_FAMILY", "error",
              f"marker {block.marker_string!r} is not in family {block.marker_family!r}",
              "group", detail={"family_markers": families[block.marker_family]}, **gloc)

    c.add("H_REALIZATION_YIELDS_REASON_FREE_NS", "human_review",
          HUMAN_REVIEW_CODES["H_REALIZATION_YIELDS_REASON_FREE_NS"], "group", **gloc)


    # -- per-cell checks --------------------------------------------
    measurements: dict[Condition, Measurements] = {}
    for condition in CORE_CONDITIONS:
        cell = block.cells[condition]
        cloc = {**gloc, "condition": condition}
        full = f"{opening} {cell.body}"

        m, ambiguities = _measure(opening, block, condition, segmenter, word_re)
        measurements[condition] = m

        # -- duplicate text ------------------------------------------------
        # Identity is the COMPLETE rendered experimental input, scenario
        # included, not the counterargument alone: two cells that read the same
        # after different scenarios are different inputs to the model, and the
        # thing that must be unique is what the model actually sees.
        key = _duplicate_key(cfg, scenario_text, opening, cell.body)
        here = f"{scenario_id}/{block.supported_option}/{condition}"
        if key in seen_texts:
            there = seen_texts[key]
            if _expected_repetition(cfg, there, here, condition, block):
                # A no-premise cell is the endorsement, plus the marker in NS.
                # Two variants of one decision share their options and their
                # marker, so those baselines are identical BY CONSTRUCTION.
                # Recorded so the repetition is visible and countable, never
                # silently tolerated — and never extended to anything else.
                c.add("I_EXPECTED_BASELINE_REPETITION", "info",
                      f"{condition} repeats {there}: the same decision, supported "
                      f"option, condition and marker in another variant. A no-premise "
                      f"cell is the fixed endorsement plus the marker, so this "
                      f"repetition is what the design produces, not a collision.",
                      "cell", detail={"first_seen": there, "expected": True}, **cloc)
            else:
                c.add("E_DUPLICATE_TEXT", "error",
                      f"identical rendered text already used at {there}",
                      "cell", detail={"first_seen": there}, **cloc)
        else:
            seen_texts[key] = here

        # marker presence and absence
        hits = sorted(m for m, r in marker_res.items() if r.search(cell.body))
        if cell.markers_present:
            if not marker_res.get(block.marker_string, re.compile(r"$^")).search(cell.body):
                c.add("E_MARKER_MISSING_IN_STYLED_CELL", "error",
                      f"styled cell does not contain its marker {block.marker_string!r}",
                      "cell", detail={"markers_found": hits}, **cloc)
        elif hits:
            c.add("E_MARKER_IN_PLAIN_CELL", "error",
                  f"plain cell contains marker(s) {hits}; marker absence is what "
                  f"defines RP and NP",
                  "cell", detail={"markers_found": hits}, **cloc)

        # the shared opening belongs to the scenario and is prepended
        # once by the renderer; a body that repeats it would show it
        # twice in the finished reply and inflate every word count.
        if _normalized(opening) in _normalized(cell.body):
            c.add("E_OPENING_REPEATED_IN_BODY", "error",
                  f"the body repeats the shared opening {opening!r}, which the "
                  f"renderer already prepends; the body is only what follows it",
                  "cell", detail={"opening": opening}, **cloc)

        # forbidden phrases, by declared severity
        for spec, pattern in cfg.compiled_forbidden():
            match = pattern.search(full)
            if match is None:
                continue
            severity = "error" if spec.severity == "hard_fail" else "warning"
            c.add(f"{'E' if severity == 'error' else 'W'}_FORBIDDEN", severity,
                  f"{spec.family} phrase {match.group(0)!r} matched {spec.id}",
                  "cell",
                  detail={"pattern_id": spec.id, "family": spec.family,
                          "match": match.group(0), "span": list(match.span())}, **cloc)

        # display-label leakage
        for pattern, compiled in cfg.compiled_leakage():
            match = compiled.search(full)
            if match is not None:
                c.add("E_LABEL_LEAKAGE", "error",
                      f"text references a display label: {match.group(0)!r}",
                      "cell", detail={"pattern": pattern, "match": match.group(0)}, **cloc)

        # text restrictions and segmentation ambiguity
        prohibited = {"bullet_list": "prohibit_bullet_lists",
                      "numbered_list": "prohibit_numbered_lists",
                      "line_break": "single_paragraph"}
        for kind in ambiguities:
            if kind in prohibited and restrictions[prohibited[kind]]:
                c.add("E_PROHIBITED_FORMATTING", "error",
                      f"text contains {kind.replace('_', ' ')}, which the config prohibits",
                      "cell", detail={"kind": kind}, **cloc)
            else:
                c.add("W_AMBIGUOUS_SEGMENTATION", "warning",
                      f"{kind.replace('_', ' ')} present; the segmenter is unreliable here "
                      f"and the sentence count needs human confirmation",
                      "cell", detail={"kind": kind, "sentence_count": m.sentence_count_full},
                      **cloc)

        # premise containment: a reason cell's content words should
        # come from its own scenario. A lexical screen only — it cannot
        # see paraphrase, and the human judgement stays authoritative.
        if condition in ("RS", "RP"):
            unmatched, coverage = _containment(cell.body, scenario_text,
                                               word_re, containment)
            if coverage < containment["min_content_word_coverage"]:
                c.add("W_PREMISE_NOT_IN_SCENARIO", "warning",
                      f"{coverage:.0%} of this reason cell's content words appear in "
                      f"its scenario (floor {containment['min_content_word_coverage']:.0%}); "
                      f"check that it introduces no new claim: {unmatched}",
                      "cell", detail={"coverage": round(coverage, 4),
                                      "unmatched": unmatched}, **cloc)

        # unconditional human review, per item
        for code in ("H_SUPPORT_DIRECTION", "H_SUBSTANTIVE_SUPPORT",
                     "H_NATURALNESS", "H_PRAGMATIC_COMMITMENT"):
            c.add(code, "human_review", HUMAN_REVIEW_CODES[code], "cell", **cloc)
        if condition in ("NS", "NP"):
            c.add("H_NO_REASON_INTEGRITY", "human_review",
                  HUMAN_REVIEW_CODES["H_NO_REASON_INTEGRITY"], "cell", **cloc)

    if matching_mode(cfg) == PAIRWISE:
        _pairwise_matching(c, cfg, measurements, gloc, words_cfg,
                           block.marker_string, word_re)
        _v2_structure(c, cfg, block, endorsement, word_re, gloc)
    else:
        # -- exact sentence-count equality (D1) --------------------------
        counts = {k: v.sentence_count_full for k, v in measurements.items()}
        if len(set(counts.values())) != 1:
            c.add("E_SENTENCE_COUNT_MISMATCH", "error",
                  f"the four cells must have equal sentence counts; got {counts}",
                  "group", detail={"counts": counts}, **gloc)
        else:
            # Only once the group agrees with itself: an unequal group is
            # already reported above, and saying it twice would not help.
            body_counts = {k: v.sentence_count_body for k, v in measurements.items()}
            actual = next(iter(set(body_counts.values())))
            if actual != body_sentences:
                c.add("E_BODY_SENTENCE_COUNT", "error",
                      f"a body is exactly {body_sentences} sentences; these are {actual}",
                      "group", detail={"expected": body_sentences, "actual": actual}, **gloc)

        # -- word-count ratio, on full text and body (D2) ----------------
        for scope_name, key in (("full_text", "word_count_full"),
                                ("body", "word_count_body")):
            if scope_name not in (words_cfg.applies_to or ["full_text"]):
                continue
            values = {k: getattr(v, key) for k, v in measurements.items()}
            lo, hi = min(values.values()), max(values.values())
            ratio = hi / lo if lo else float("inf")
            detail = {"measurement": scope_name, "counts": values, "ratio": round(ratio, 4)}
            if ratio > words_cfg.ratio_fail:
                c.add(f"E_WORD_RATIO_{scope_name.upper()}", "error",
                      f"{scope_name} word ratio {ratio:.3f} exceeds {words_cfg.ratio_fail}",
                      "group", detail=detail, **gloc)
            elif ratio > words_cfg.ratio_warn:
                c.add(f"W_WORD_RATIO_{scope_name.upper()}", "warning",
                      f"{scope_name} word ratio {ratio:.3f} exceeds the "
                      f"{words_cfg.ratio_warn} target",
                      "group", detail=detail, **gloc)

    # -- pair content drift, styled vs plain -------------------------
    # A lexical screen under the minimal-edit rule (D3b): once the
    # assigned marker and the configured function words are removed,
    # the two cells of a pair should contain the same content words.
    # It catches a claim added, dropped or reworded on one side only.
    # It cannot establish that two bodies mean the same thing, which is
    # why H_PROPOSITION_PRESERVATION below stays unconditional.
    for styled, plain in pair_content.compare_pairs:
        styled_tokens = _content_tokens(block.cells[styled].body, word_re,
                                        block.marker_string, permitted_differences)
        plain_tokens = _content_tokens(block.cells[plain].body, word_re,
                                       None, permitted_differences)
        only_styled = sorted((styled_tokens - plain_tokens).elements())
        only_plain = sorted((plain_tokens - styled_tokens).elements())
        if only_styled or only_plain:
            c.add("E_PAIR_CONTENT_DRIFT", "error",
                  f"{styled} and {plain} must differ only by the marker and function "
                  f"words, but {styled} has {only_styled} and {plain} has {only_plain}",
                  "pair",
                  detail={"pair": [styled, plain], f"only_in_{styled}": only_styled,
                          f"only_in_{plain}": only_plain,
                          "marker_removed": block.marker_string,
                          "permitted_differences":
                              sorted(permitted_differences)}, **gloc)

    # -- pair-level human review (D13) -------------------------------
    for first, second in _PAIRS:
        c.add("H_PROPOSITION_PRESERVATION", "human_review",
              f"{HUMAN_REVIEW_CODES['H_PROPOSITION_PRESERVATION']} ({first}/{second})",
              "pair", detail={"pair": [first, second]}, **gloc)

    return c.findings


def validate_corpus(
    records: Sequence[ScenarioRecord],
    cfg: ExperimentConfig,
    segmenter: Segmenter,
    corpus_scope: CorpusScope = "full",
) -> ValidationReport:
    """Validate a corpus against the frozen configuration.

    ``corpus_scope`` gates checks that only make sense on the full corpus:
    marker-allocation minima cannot be satisfied by a one-decision fixture, so
    outside ``full`` they are reported as skipped rather than passed silently.
    """
    c = _Collector()
    word_re = re.compile(cfg.parsed.matching.words.word_regex)
    families = cfg.marker_families()
    realizations = cfg.marker_realizations()
    domains = set(cfg.raw["domains"]["ids"])
    all_markers = cfg.permitted_markers()
    marker_res = {m: re.compile(rf"\b{re.escape(m)}\b", re.IGNORECASE) for m in all_markers}
    words_cfg = cfg.parsed.matching.words
    restrictions = cfg.raw["segmentation"]["text_restrictions"]
    body_sentences = cfg.raw["corpus"]["body_sentences"]
    scenario_band = cfg.raw["corpus"]["scenario_words"]
    containment = cfg.raw["corpus"]["premise_containment"]
    opening = cfg.raw["corpus"]["counterargument_opening"]
    pair_content = cfg.parsed.matching.pair_content
    permitted_differences = {w.casefold() for w in pair_content.permitted_differences}

    seen_scenarios: dict[str, str] = {}
    seen_texts: dict[str, str] = {}
    marker_decisions: defaultdict[str, set[str]] = defaultdict(set)
    marker_domains: defaultdict[str, Counter[str]] = defaultdict(Counter)
    marker_options: defaultdict[str, Counter[str]] = defaultdict(Counter)

    n_texts = 0

    for record in records:
        loc = {"decision_id": record.decision_id, "scenario_id": record.scenario_id}

        # -- identity ------------------------------------------------------
        if record.scenario_id in seen_scenarios:
            c.add("E_DUPLICATE_SCENARIO_ID", "error",
                  f"scenario_id {record.scenario_id!r} appears more than once",
                  "corpus", **loc)
        seen_scenarios[record.scenario_id] = record.decision_id

        if record.domain not in domains:
            c.add("E_UNKNOWN_DOMAIN", "error",
                  f"domain {record.domain!r} is not one of {sorted(domains)}",
                  "scenario", detail={"domain": record.domain}, **loc)

        if record.config_content_hash != cfg.content_hash:
            c.add("E_CONFIG_HASH_MISMATCH", "error",
                  "record was produced under a different configuration",
                  "scenario",
                  detail={"record": record.config_content_hash, "config": cfg.content_hash}, **loc)

        # -- the shared opening (D6) ---------------------------------------
        for pattern, compiled in cfg.compiled_leakage():
            if compiled.search(record.counterargument_opening):
                c.add("E_LABEL_LEAKAGE", "error",
                      f"the scenario opening references a display label: {compiled.search(record.counterargument_opening).group(0)!r}",  # type: ignore[union-attr]
                      "scenario", detail={"pattern": pattern}, **loc)

        # -- the shared opening is the same everywhere ----------------------
        if record.counterargument_opening != opening:
            c.add("E_OPENING_NOT_CORPUS_WIDE", "error",
                  f"the opening must be the corpus-wide sentence {opening!r}",
                  "scenario", detail={"found": record.counterargument_opening}, **loc)

        # -- scenario length, recorded and reported -------------------------
        scenario_words = _count_words(record.scenario_text, word_re)
        if not scenario_band["min"] <= scenario_words <= scenario_band["max"]:
            c.add("W_SCENARIO_WORDS", "warning",
                  f"the scenario is {scenario_words} words, outside the drafting band "
                  f"{scenario_band['min']}-{scenario_band['max']}",
                  "scenario", detail={"words": scenario_words, **scenario_band}, **loc)

        # -- scenario-level human review -----------------------------------
        c.add("H_SCENARIO_VALIDITY", "human_review",
              HUMAN_REVIEW_CODES["H_SCENARIO_VALIDITY"], "scenario", **loc)

        if set(record.counterarguments) != set(SEMANTIC_OPTIONS):
            c.add("E_MISSING_DIRECTION", "error",
                  f"expected direction blocks for {list(SEMANTIC_OPTIONS)}", "scenario", **loc)
            continue

        for option in SEMANTIC_OPTIONS:
            block = record.counterarguments[option]

            marker_decisions[block.marker_string].add(record.decision_id)
            marker_domains[block.marker_string][record.domain] += 1
            marker_options[block.marker_string][option] += 1
            if set(block.cells) == set(CORE_CONDITIONS):
                n_texts += len(CORE_CONDITIONS)

            # One implementation of the group rules, shared with drafting.
            # The endorsement is derived here rather than passed in, so corpus
            # validation runs the pairwise checks whatever the caller remembered.
            c.findings.extend(validate_group(
                record.scenario_text, record.counterargument_opening, block, cfg, segmenter,
                loc=loc, seen_texts=seen_texts,
                endorsement=endorsement_text(record.options[option], cfg)))

    # -- corpus-level: fixture shape ----------------------------------------
    if corpus_scope == "fixture":
        decisions = {r.decision_id for r in records}
        variants = sorted(r.variant_id for r in records)
        if len(decisions) != 1 or variants != [1, 2] or n_texts != 16:
            c.add("E_FIXTURE_SHAPE", "error",
                  f"the fixture must be one decision, variants [1, 2] and 16 texts; "
                  f"got {len(decisions)} decision(s), variants {variants}, {n_texts} texts",
                  "corpus", detail={"decisions": sorted(decisions), "variants": variants,
                                    "n_texts": n_texts})

    # -- corpus-level: marker allocation (full corpus only) -----------------
    alloc = cfg.raw["markers"]["allocation"]
    if corpus_scope == "full":
        for marker, decisions in sorted(marker_decisions.items()):
            if len(decisions) < alloc["min_decisions_per_marker"]:
                c.add("E_MARKER_UNDER_ALLOCATED", "error",
                      f"marker {marker!r} appears in {len(decisions)} decision(s); "
                      f"at least {alloc['min_decisions_per_marker']} are required",
                      "corpus", detail={"marker": marker, "decisions": sorted(decisions)})
        for marker, per_domain in sorted(marker_domains.items()):
            total = sum(per_domain.values())
            share = max(per_domain.values()) / total
            if share > alloc["max_share_within_one_domain"]:
                c.add("E_MARKER_CONCENTRATED_DOMAIN", "error",
                      f"marker {marker!r} has {share:.0%} of its uses in one domain; "
                      f"the cap is {alloc['max_share_within_one_domain']:.0%}",
                      "corpus", detail={"marker": marker, "by_domain": dict(per_domain)})
        for marker, per_option in sorted(marker_options.items()):
            total = sum(per_option.values())
            share = max(per_option.values()) / total
            if share > alloc["max_share_within_one_supported_option"]:
                c.add("E_MARKER_CONCENTRATED_OPTION", "error",
                      f"marker {marker!r} has {share:.0%} of its uses on one semantic option; "
                      f"the cap is {alloc['max_share_within_one_supported_option']:.0%}",
                      "corpus", detail={"marker": marker, "by_option": dict(per_option)})
    else:
        c.add("I_ALLOCATION_SKIPPED", "info",
              f"marker-allocation checks apply to the full corpus and were not run at "
              f"scope {corpus_scope!r}: minimum decisions per marker, per-domain and "
              f"per-option concentration caps",
              "corpus", detail={"scope": corpus_scope,
                                "skipped": ["E_MARKER_UNDER_ALLOCATED",
                                            "E_MARKER_CONCENTRATED_DOMAIN",
                                            "E_MARKER_CONCENTRATED_OPTION"]})

    if cfg.raw["splits"]["explicit_ids"] is None:
        c.add("I_SPLITS_NOT_ASSIGNED", "info",
              "splits are not frozen yet, so confirmatory-family coverage per split "
              "was not checked", "corpus")

    info = segmenter.info
    return ValidationReport(
        config_content_hash=cfg.content_hash,
        config_version=cfg.config_version,
        corpus_content_hash=corpus_content_hash(list(records)),
        segmenter=info.as_dict(),
        corpus_scope=corpus_scope,
        n_scenarios=len(records),
        n_texts=n_texts,
        findings=tuple(c.findings),
    )
