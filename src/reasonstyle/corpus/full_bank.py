"""Deterministic checks on the full topic bank before any drafting.

The per-brief checks (word limits, forbidden phrases, source overlap,
citability) live in :mod:`reasonstyle.corpus.topics`. This module checks what
only makes sense for the bank as a whole, against the pilot it grew from:

* the design's counts — decisions in play per domain, and rejected pilot
  candidates carried over but never counted;
* uniqueness of ids and of propositions (framings and option texts);
* that every pilot record is carried over byte for byte, and still parses to
  the same brief;
* that every cited registry key exists and every locator points at something
  that is really there — an article present in the pinned legal text, or a
  category value present in the pinned JRC tables;
* that every decision added after the pilot is either proposed with no
  curation at all, or curated with every judgement true, a curator and a date;
* a lexical near-duplicate screen between decisions; and
* that nothing downstream of the bank — scenarios, groups, an allocation,
  approvals, a corpus, review pages — exists yet for the full design.

A lexical screen cannot see two decisions that share a trade-off in different
words. Conceptual overlap stays the curator's judgement; the screen only makes
sure the obvious cases are never missed.
"""

from __future__ import annotations

import csv
import io
import itertools
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .topics import TopicBank, TopicBrief

__all__ = [
    "BankFinding",
    "check_full_bank",
    "check_locator",
    "near_duplicates",
    "record_blocks",
]

#: Lexical similarity (Jaccard over content words of the framing and both
#: options) at or above which two decisions are an error, and a warning.
#: Measured on the drafted bank, the most similar pair of distinct decisions
#: scores 0.40, driven by shared institutional wording rather than a shared
#: trade-off.
NEAR_DUPLICATE_ERROR = 0.60
NEAR_DUPLICATE_WARNING = 0.35

_STOP = frozenset(
    "a an the and or of to in on for with by from at as is are be its their it this that "
    "whether must decide should only all each every any more than into over under not no own "
    "new keep let allow require make use have has".split())

_RECORD = re.compile(r"^  - decision_id: (\S+)$", re.M)
_BOUNDARY = re.compile(r"^  - decision_id: |^  # =", re.M)

POLIANNA_ARTICLES = Path("POLIANNA_v1_1/POLIANNA_v1_1/03b_processed_to_json")
JRC_USECASES = "Export_PSTW_GENAI_AnnexII_usecases_dataset.csv"
JRC_GUIDELINES = "Export_PSTW_GENAI_AnnexI_guidelines_dataset.csv"

#: Locator field name -> JRC Annex II column.
_JRC_FIELDS = {
    "process type": "Process type",
    "application type": "Application type",
    "function of government": "Functions of Government (COFOG level I)",
    "interaction": "Interaction",
    "organisation category": "Responsible organisation category",
    "organisation categories": "Responsible organisation category",
}
_INTERACTION = {"government to citizen": "g2c", "government to business": "g2b",
                "government to government": "g2g"}


@dataclass(frozen=True)
class BankFinding:
    code: str
    severity: str  # "error" or "warning"
    message: str
    decision_id: str | None = None


# --------------------------------------------------------------------------
# Pilot preservation
# --------------------------------------------------------------------------

def record_blocks(text: str) -> dict[str, str]:
    """``{decision_id: record text}``, each block from its ``- decision_id``
    line to the next record or domain marker, trailing blank lines removed."""
    starts = [(m.start(), m.group(1)) for m in _RECORD.finditer(text)]
    blocks: dict[str, str] = {}
    for pos, did in starts:
        nxt = _BOUNDARY.search(text, pos + 1)
        end = nxt.start() if nxt else len(text)
        blocks[did] = text[pos:end].rstrip("\n") + "\n"
    return blocks


# --------------------------------------------------------------------------
# Locators
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    return " ".join(s.casefold().replace("-", " ").split())


def _jrc_columns(raw: Path) -> tuple[dict[str, set[str]], set[str]]:
    def rows(name):
        data = (raw / name).read_bytes().decode("utf-8-sig")
        return list(csv.reader(io.StringIO(data), delimiter=";"))
    usecases = rows(JRC_USECASES)
    header, body = usecases[0], usecases[1:]
    columns = {col: {_norm(r[i]) for r in body if i < len(r) and r[i].strip()}
               for i, col in enumerate(header)}
    principles = {_norm(h) for h in rows(JRC_GUIDELINES)[0] if h.strip()}
    return columns, principles


def _value_present(value: str, cells: set[str]) -> bool:
    v = _norm(value)
    if any(v in cell for cell in cells):
        return True
    # A locator may list several categories ("private sector, consortium");
    # each must then be present on its own.
    parts = [p.strip() for p in v.split(",") if p.strip()]
    return len(parts) > 1 and all(any(p in cell for cell in cells) for p in parts)


def _articles(segment: str) -> list[str]:
    return re.findall(r"\b(\d+[a-z]{0,2})(?:\([a-z0-9]+\))?", segment)


def check_locator(locator: str, raw: Path, *, _cache: dict | None = None) -> list[str]:
    """Problems with one locator; empty when it points at something real."""
    cache = _cache if _cache is not None else {}
    celex = re.match(r"CELEX (\S+?), arts?\. (.*?)(?: \(.*\))?$", locator)
    if celex:
        cid, segment = celex.groups()
        articles = _articles(segment)
        if not articles:
            return [f"no article number in {locator!r}"]
        problems = []
        if cid.startswith("0"):
            # A consolidated act, pinned as an EUR-Lex PDF with its text beside it.
            path = raw / f"eurlex_{cid}_EN.txt"
            if not path.exists():
                return [f"no pinned text for CELEX {cid} ({path.name})"]
            text = cache.setdefault(path, path.read_text(encoding="utf-8", errors="replace"))
            for art in articles:
                if not re.search(rf"^\s*Article {re.escape(art)}\s*$", text, re.M):
                    problems.append(f"CELEX {cid} has no Article {art}")
            for chapter in re.findall(r"Chapter ([IVXL]+[a-z]?)", segment):
                if not re.search(rf"^\s*CHAPTER {chapter}\s*$", text, re.M):
                    problems.append(f"CELEX {cid} has no Chapter {chapter}")
        else:
            folder = raw / POLIANNA_ARTICLES
            names = cache.setdefault(folder, sorted(p.name for p in folder.iterdir())
                                     if folder.exists() else [])
            if not names:
                return [f"POLIANNA articles not found under {folder}"]
            for art in articles:
                if not art.isdigit():
                    problems.append(f"POLIANNA numbers articles only by digits; got {art!r}")
                    continue
                pattern = re.compile(rf"^EU_{re.escape(cid)}_.*_Article_0*{int(art)}$")
                if not any(pattern.match(n) for n in names):
                    problems.append(f"POLIANNA has no CELEX {cid} article {art}")
        return problems

    if "Annex I" in locator:
        if "jrc" not in cache:
            cache["jrc"] = _jrc_columns(raw)
        columns, principles = cache["jrc"]
        problems = []
        for part in (p.strip() for p in locator.split(";")):
            field, _, value = part.partition(":")
            field = re.sub(r"^Annex II\s+", "", field.strip()).casefold()
            value = value.strip()
            if not value:
                problems.append(f"no value in {part!r}")
            elif field == "annex i principle":
                if _norm(value) not in principles:
                    problems.append(f"no Annex I principle {value!r}")
            elif field in _JRC_FIELDS:
                if field == "interaction":
                    value = _INTERACTION.get(_norm(value), value)
                if not _value_present(value, columns[_JRC_FIELDS[field]]):
                    problems.append(f"no Annex II {field} value {value!r}")
            else:
                problems.append(f"unknown JRC locator field {field!r}")
        return problems

    return [f"unrecognised locator format {locator!r}"]


# --------------------------------------------------------------------------
# Near duplicates
# --------------------------------------------------------------------------

def _content(topic: TopicBrief) -> set[str]:
    text = " ".join([topic.decision_framing, topic.options["opt_1"], topic.options["opt_2"]])
    return {w for w in re.findall(r"[a-z]+", text.casefold()) if w not in _STOP and len(w) > 2}


def near_duplicates(topics: Iterable[TopicBrief]) -> list[tuple[float, str, str]]:
    """Every pair of decisions with its similarity, most similar first."""
    sigs = [(t.decision_id, _content(t)) for t in topics]
    pairs = []
    for (a, sa), (b, sb) in itertools.combinations(sigs, 2):
        union = sa | sb
        pairs.append((round(len(sa & sb) / len(union), 3) if union else 1.0, a, b))
    return sorted(pairs, key=lambda p: (-p[0], p[1], p[2]))


# --------------------------------------------------------------------------
# The whole bank
# --------------------------------------------------------------------------

def check_full_bank(
    *,
    bank: TopicBank,
    bank_text: str,
    pilot: TopicBank,
    pilot_text: str,
    domains: list[str],
    per_domain: int,
    registry_keys: Iterable[str],
    raw_dir: Path | None,
    output_paths: Mapping[str, str | Path],
    root: Path,
    generation_blocked: bool,
) -> list[BankFinding]:
    out: list[BankFinding] = []

    def add(code, severity, message, did=None):
        out.append(BankFinding(code, severity, message, did))

    # -- counts --------------------------------------------------------------
    active = [t for t in bank.topics if t.status != "rejected"]
    per = Counter(t.domain for t in active)
    if len(active) != per_domain * len(domains):
        add("E_BANK_COUNT", "error",
            f"{len(active)} decisions in play; the design needs {per_domain * len(domains)}")
    for d in domains:
        if per.get(d, 0) != per_domain:
            add("E_BANK_DOMAIN_COUNT", "error", f"{d}: {per.get(d, 0)} decisions; needs {per_domain}")
    for d in set(per) - set(domains):
        add("E_BANK_DOMAIN_COUNT", "error", f"unknown domain {d!r}")

    # -- uniqueness ----------------------------------------------------------
    for did, n in Counter(t.decision_id for t in bank.topics).items():
        if n > 1:
            add("E_BANK_DUPLICATE_ID", "error", f"{did} appears {n} times", did)
    seen: dict[str, str] = {}
    for t in bank.topics:
        for label, text in [("framing", t.decision_framing),
                            ("opt_1", t.options["opt_1"]), ("opt_2", t.options["opt_2"])]:
            key = _norm(text)
            if key in seen and seen[key] != t.decision_id:
                add("E_BANK_DUPLICATE_PROPOSITION", "error",
                    f"{label} repeats a proposition of {seen[key]}", t.decision_id)
            seen.setdefault(key, t.decision_id)

    # -- shape ---------------------------------------------------------------
    for t in bank.topics:
        if set(t.variants) != {"v1", "v2"}:
            add("E_BANK_VARIANTS", "error", f"variants {sorted(t.variants)}", t.decision_id)
        if set(t.options) != {"opt_1", "opt_2"}:
            add("E_BANK_OPTIONS", "error", f"options {sorted(t.options)}", t.decision_id)

    # -- pilot preservation ----------------------------------------------------
    pilot_blocks, bank_blocks = record_blocks(pilot_text), record_blocks(bank_text)
    by_id = {t.decision_id: t for t in bank.topics}
    for p in pilot.topics:
        did = p.decision_id
        if did not in by_id:
            add("E_BANK_PILOT_MISSING", "error", "pilot record missing", did)
            continue
        if bank_blocks.get(did) != pilot_blocks.get(did):
            add("E_BANK_PILOT_CHANGED", "error", "pilot record text differs from the pilot bank", did)
        if by_id[did] != p:
            add("E_BANK_PILOT_CHANGED", "error", "pilot record parses to a different brief", did)
    pilot_ids = {p.decision_id for p in pilot.topics}
    # A decision added after the pilot is either still proposed, carrying no
    # curation at all, or curated with every judgement true and a named curator
    # and date. Nothing in between: a half-recorded approval is not an approval.
    for t in bank.topics:
        if t.decision_id in pilot_ids:
            continue
        cur = t.curation
        judgements = cur.recorded().values()
        if t.status == "proposed":
            if any(v is not None for v in judgements) or cur.curated_by or cur.curated_at:
                add("E_BANK_NEW_CURATION", "error",
                    "a proposed decision carries curation it was never given", t.decision_id)
        elif t.status == "curated":
            if not all(v is True for v in judgements) or not cur.curated_by or not cur.curated_at:
                add("E_BANK_NEW_CURATION", "error",
                    "a curated decision needs every judgement true, plus who and when",
                    t.decision_id)
        else:
            add("E_BANK_NEW_CURATION", "error",
                f"a decision added after the pilot is {t.status!r}; only the pilot's rejected "
                f"candidates are carried as rejected", t.decision_id)

    # -- sources ----------------------------------------------------------------
    keys = set(registry_keys)
    cache: dict = {}
    for t in bank.topics:
        if not t.source_references:
            add("E_BANK_NO_SOURCE", "error", "no source reference", t.decision_id)
        for s in t.source_references:
            if s.key not in keys:
                add("E_BANK_UNKNOWN_SOURCE", "error", f"{s.key!r} is not in the registry",
                    t.decision_id)
            if raw_dir is None:
                continue
            for problem in check_locator(s.locator, raw_dir, _cache=cache):
                add("E_BANK_LOCATOR", "error", problem, t.decision_id)
    if raw_dir is None:
        add("W_BANK_LOCATORS_UNCHECKED", "warning",
            "no source directory given, so no locator was checked against the pinned sources")

    # -- near duplicates ----------------------------------------------------------
    for score, a, b in near_duplicates(active):
        if score >= NEAR_DUPLICATE_ERROR:
            add("E_BANK_NEAR_DUPLICATE", "error", f"{a} and {b}: similarity {score:.2f}", b)
        elif score >= NEAR_DUPLICATE_WARNING:
            add("W_BANK_NEAR_DUPLICATE", "warning",
                f"{a} and {b}: similarity {score:.2f}; check they differ in trade-off", b)

    # -- nothing downstream exists yet ---------------------------------------------
    for name, path in sorted(output_paths.items()):
        if (root / path).exists():
            add("E_BANK_DOWNSTREAM_EXISTS", "error",
                f"paths.{name} ({path}) exists, but nothing may be generated while the "
                f"full design's generation block is in place")
    if not generation_blocked and any(t.status == "proposed" for t in bank.topics):
        add("E_BANK_GENERATION_UNBLOCKED", "error",
            "generation is not blocked in the configuration while the bank holds proposed decisions")

    return out
