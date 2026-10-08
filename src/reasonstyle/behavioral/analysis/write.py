"""Deterministic serialisation, atomic output directories and the manifest.

CSV: UTF-8, ``\\n`` line ends, a header of the union of keys in first-seen order,
rows sorted by the table's key columns, floats as ``repr`` (shortest
round-trip), missing values empty. JSON: sorted keys, indent 2, no NaN. gzip:
``mtime=0``, no file name, level 9. Volatile execution details live only in
``execution_environment.json``.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

MANIFEST = "analysis_manifest.json"
MANIFEST_HASH = "analysis_manifest.sha256"
SORT_KEYS = ("run", "model", "outcome", "sample", "comparison", "kind", "stratum", "term",
             "domain", "order_id", "opening_id", "quartile", "marker_family", "marker_id",
             "quantity", "family", "level", "check", "initial_id", "condition", "slot",
             "initial_option", "statistic_id")


def clean(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.integer,)):
        value = int(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    return value


def _cell(value: Any) -> str:
    value = clean(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _sort_key(row: dict[str, Any]) -> tuple:
    return tuple("" if row.get(k) is None else str(row.get(k)) for k in SORT_KEYS if k in row)


def sorted_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=_sort_key)


def csv_text(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(row.get(c)) for c in columns])
    return buf.getvalue()


def json_text(obj: Any) -> str:
    return json.dumps(clean(obj), indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False) + "\n"


def gzip_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class OutputDir:
    """Collects files in a temporary sibling and moves them into place at the end."""

    def __init__(self, destination: Path, analysis_version: str, overwrite: bool):
        self.destination = Path(destination)
        self.version = analysis_version
        if self.destination.exists():
            if not overwrite:
                raise FileExistsError(f"{self.destination} exists; pass --overwrite to replace it")
            manifest = self.destination / MANIFEST
            if not manifest.is_file() or json.loads(manifest.read_text()).get(
                    "analysis_version") != analysis_version:
                raise FileExistsError(f"{self.destination} does not hold a "
                                      f"{analysis_version} analysis; refusing to overwrite it")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temp = Path(tempfile.mkdtemp(prefix=self.destination.name + ".incomplete.",
                                          dir=self.destination.parent))
        self.files: dict[str, dict[str, Any]] = {}

    def write(self, rel: str, data: str | bytes, rows: int | None = None) -> None:
        path = self.temp / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = data.encode("utf-8") if isinstance(data, str) else data
        path.write_bytes(raw)
        self.files[rel] = {"sha256": sha256_bytes(raw), "bytes": len(raw), "rows": rows}

    def table(self, name: str, rows: list[dict[str, Any]], folder: str = "tables") -> None:
        ordered = sorted_rows(rows)
        self.write(f"{folder}/{name}.csv", csv_text(ordered), rows=len(ordered))
        self.write(f"{folder}/{name}.json", json_text(ordered), rows=len(ordered))

    def finish(self, manifest: dict[str, Any]) -> Path:
        manifest = {**manifest, "outputs": {k: self.files[k] for k in sorted(self.files)}}
        text = json_text(manifest)
        (self.temp / MANIFEST).write_text(text, encoding="utf-8")
        (self.temp / MANIFEST_HASH).write_text(
            f"{sha256_bytes(text.encode('utf-8'))}  {MANIFEST}\n", encoding="utf-8")
        old = None
        if self.destination.exists():
            old = self.destination.with_name(self.destination.name + ".replaced")
            if old.exists():
                shutil.rmtree(old)
            self.destination.replace(old)
        self.temp.replace(self.destination)
        if old is not None:
            shutil.rmtree(old)
        return self.destination


def verify_directory(directory: Path) -> list[str]:
    """Problems found re-hashing a finished output directory against its manifest."""
    directory = Path(directory)
    problems = []
    text = (directory / MANIFEST).read_bytes()
    recorded = (directory / MANIFEST_HASH).read_text().split()[0]
    if sha256_bytes(text) != recorded:
        problems.append("analysis_manifest.json does not match analysis_manifest.sha256")
    manifest = json.loads(text)
    for rel, info in manifest["outputs"].items():
        path = directory / rel
        if not path.is_file():
            problems.append(f"missing {rel}")
        elif sha256_bytes(path.read_bytes()) != info["sha256"]:
            problems.append(f"hash mismatch {rel}")
    listed = set(manifest["outputs"]) | {MANIFEST, MANIFEST_HASH}
    extra = sorted(str(p.relative_to(directory)) for p in directory.rglob("*")
                   if p.is_file() and str(p.relative_to(directory)) not in listed)
    problems += [f"unlisted file {e}" for e in extra]
    return problems
