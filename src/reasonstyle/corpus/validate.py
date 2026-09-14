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

__all__ = ["CorpusScope", "validate_corpus", "with_measurements"]

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


def _measure(record: ScenarioRecord, option: str, condition: Condition,
             segmenter: Segmenter, word_re: re.Pattern[str]) -> tuple[Measurements, tuple[str, ...]]:
    full = record.render(option, condition)          # type: ignore[arg-type]
    body = record.counterarguments[option].cells[condition].body   # type: ignore[index]
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
                measurements, _ = _measure(record, option, condition, segmenter, word_re)
                cells[condition] = cell.model_copy(update={"measurements": measurements})
            blocks[option] = block.model_copy(update={"cells": cells})
        out.append(record.model_copy(update={"counterarguments": blocks}))
    return out


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
            gloc = {**loc, "supported_option": option}

            if set(block.cells) != set(CORE_CONDITIONS):
                c.add("E_MISSING_CELL", "error",
                      f"expected the four core cells, got {sorted(block.cells)}", "group", **gloc)
                continue

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

            marker_decisions[block.marker_string].add(record.decision_id)
            marker_domains[block.marker_string][record.domain] += 1
            marker_options[block.marker_string][option] += 1

            # -- per-cell checks --------------------------------------------
            measurements: dict[Condition, Measurements] = {}
            for condition in CORE_CONDITIONS:
                cell = block.cells[condition]
                cloc = {**gloc, "condition": condition}
                n_texts += 1
                full = record.render(option, condition)   # type: ignore[arg-type]

                m, ambiguities = _measure(record, option, condition, segmenter, word_re)
                measurements[condition] = m

                # duplicate text
                if full in seen_texts:
                    c.add("E_DUPLICATE_TEXT", "error",
                          f"identical rendered text already used at {seen_texts[full]}",
                          "cell", detail={"first_seen": seen_texts[full]}, **cloc)
                seen_texts[full] = f"{record.scenario_id}/{option}/{condition}"

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
                    unmatched, coverage = _containment(cell.body, record.scenario_text,
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
            for scope_name, key in (("full_text", "word_count_full"), ("body", "word_count_body")):
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
                          f"{scope_name} word ratio {ratio:.3f} exceeds the {words_cfg.ratio_warn} target",
                          "group", detail=detail, **gloc)

            # -- pair-level human review (D13) -------------------------------
            for first, second in _PAIRS:
                c.add("H_PROPOSITION_PRESERVATION", "human_review",
                      f"{HUMAN_REVIEW_CODES['H_PROPOSITION_PRESERVATION']} ({first}/{second})",
                      "pair", detail={"pair": [first, second]}, **gloc)

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
