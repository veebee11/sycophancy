"""The downloaded source text, for screening topic briefs against accidental copying.

Only the permitted formats are read: POLIANNA's JSON (article text rebuilt from
``Tokens.json``, and the coding scheme's descriptions) and the two pinned JRC
CSV tables. POLIANNA's pickle, dataframe CSV and plain-text files are never
opened. The text is used only to find word sequences a brief shares with it;
nothing here is copied into a brief or a review page.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

__all__ = ["load_source_texts", "ngram_index", "shared_runs", "words"]

_WORD = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*")

POLIANNA_JSON = Path("POLIANNA_v1_1/POLIANNA_v1_1/03b_processed_to_json")
POLIANNA_SCHEME = Path("POLIANNA_v1_1/POLIANNA_v1_1/01_policy_info/Coding_Scheme.json")
JRC_TABLES = ("Export_PSTW_GENAI_AnnexI_guidelines_dataset.csv",
              "Export_PSTW_GENAI_AnnexII_usecases_dataset.csv")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _tokens_text(path: Path) -> str:
    out, last = [], None
    for token in json.loads(path.read_text(encoding="utf-8")):
        if last is not None and token["start"] > last:
            out.append(" ")
        out.append(token["text"])
        last = token["stop"]
    return "".join(out)


def _scheme_text(path: Path) -> str:
    parts = []
    for layer in json.loads(path.read_text(encoding="utf-8"))["layers"]:
        parts.append(layer.get("layer_description", ""))
        for tagset in layer["tagsets"]:
            parts.append(tagset.get("tagset_description", ""))
            parts += [tag.get("tag_description", "") for tag in tagset["tags"]]
    return "\n".join(p for p in parts if p)


def load_source_texts(raw_dir: str | Path) -> dict[str, str]:
    """``{document id: text}`` for everything a topic brief may have drawn on."""
    raw = Path(raw_dir)
    texts: dict[str, str] = {}
    articles = raw / POLIANNA_JSON
    if articles.exists():
        for folder in sorted(articles.iterdir()):
            if (folder / "Tokens.json").exists():
                texts[f"polianna:{folder.name}"] = _tokens_text(folder / "Tokens.json")
    if (raw / POLIANNA_SCHEME).exists():
        texts["polianna:coding_scheme"] = _scheme_text(raw / POLIANNA_SCHEME)
    for name in JRC_TABLES:
        path = raw / name
        if not path.exists():
            continue
        reader = csv.reader(io.StringIO(path.read_bytes().decode("utf-8-sig")), delimiter=";")
        for i, row in enumerate(reader):
            text = " \n ".join(cell for cell in row if cell.strip())
            if text:
                texts[f"jrc:{path.stem}:{i}"] = text
    return texts


def ngram_index(texts: Iterable[str], n: int) -> set[tuple[str, ...]]:
    index: set[tuple[str, ...]] = set()
    for text in texts:
        toks = words(text)
        index.update(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return index


def shared_runs(text: str, index: set[tuple[str, ...]], n: int) -> list[str]:
    """Maximal runs of ``n`` or more consecutive words ``text`` shares with the
    index, returned in the brief's own (lower-cased) words."""
    toks = words(text)
    covered = [False] * len(toks)
    for i in range(len(toks) - n + 1):
        if tuple(toks[i:i + n]) in index:
            for j in range(i, i + n):
                covered[j] = True
    runs, start = [], None
    for i, flag in enumerate(covered + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append(" ".join(toks[start:i]))
            start = None
    return runs


def screen_summary(texts: Mapping[str, str]) -> dict[str, int]:
    return {"documents": len(texts), "words": sum(len(words(t)) for t in texts.values())}
