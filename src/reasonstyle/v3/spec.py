"""The v3 specification, read from ``configs/full_v3_multimarker_openings.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SPEC_PATH = Path("configs/full_v3_multimarker_openings.yaml")
FAMILIES = ("conclusion_result", "inference_basis")
CONDITIONS = ("RS", "NS", "RP", "NP")


class V3Error(RuntimeError):
    """The v3 inputs or outputs do not verify. Nothing proceeds."""


@dataclass(frozen=True)
class Marker:
    marker_id: str
    string: str
    family: str
    subtype: str
    realization_id: str
    prefix: str


@dataclass(frozen=True)
class Spec:
    raw: dict[str, Any]
    path: Path

    @property
    def version(self) -> str:
        return self.raw["dataset_version"]

    @property
    def markers(self) -> dict[str, Marker]:
        return {mid: Marker(marker_id=mid, **m) for mid, m in self.raw["markers"].items()}

    def family_markers(self, family: str) -> tuple[str, ...]:
        return tuple(self.raw["families"][family]["markers"])

    @property
    def openings(self) -> dict[str, str]:
        return dict(self.raw["openings"])

    @property
    def seed(self) -> int:
        return int(self.raw["allocation"]["seed"])

    @property
    def source(self) -> dict[str, Any]:
        return self.raw["derived_from"]

    @property
    def outputs(self) -> dict[str, str]:
        return self.raw["outputs"]


def load_spec(path: str | Path = SPEC_PATH) -> Spec:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    problems = []
    if raw.get("status") != "draft":
        problems.append(f"status is {raw.get('status')!r}; v3 is a draft")
    if tuple(raw.get("families", {})) != FAMILIES:
        problems.append(f"families must be exactly {FAMILIES}")
    for family in FAMILIES:
        ids = raw["families"][family]["markers"]
        if len(ids) != 6 or len(set(ids)) != 6:
            problems.append(f"{family} must name six distinct markers")
        for mid in ids:
            m = raw["markers"].get(mid)
            if m is None or m.get("family") != family:
                problems.append(f"{mid} is not a {family} marker")
                continue
            expected = m["string"][0].upper() + m["string"][1:]
            if not m["prefix"].startswith(expected) or m["prefix"] not in (expected + ", ",
                                                                            expected + " "):
                problems.append(f"{mid}: prefix {m['prefix']!r} is not the capitalised marker "
                                f"followed by ', ' or ' '")
            comma = m["prefix"].endswith(", ")
            if comma != (m["realization_id"] != "v3_initial_clause_no_comma"):
                problems.append(f"{mid}: punctuation does not match realization "
                                f"{m['realization_id']}")
    if len(raw.get("markers", {})) != 12:
        problems.append("there must be exactly 12 markers")
    if len(raw.get("openings", {})) != 3 or any(t != t.strip() for t in raw["openings"].values()):
        problems.append("there must be exactly three openings with no surrounding whitespace")
    if problems:
        raise V3Error("the v3 specification does not verify:\n  - " + "\n  - ".join(problems))
    return Spec(raw=raw, path=path)
