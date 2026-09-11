"""The reference-source registry.

Datasets are reference sources only: they may supply topics, competing values
and argument structures, but their text is never copied into the corpus.

Each candidate has one of three explicit outcomes:

``unverified``
    Not checked yet.
``citable``
    Access and licence confirmed, and seed-only use permitted. Only citable
    entries may be cited. An unresolved candidate that nothing cites does not
    block anything.
``excluded``
    Checked, but unavailable, unsuitable, prohibited or too unclear to use.
    Requires who checked it, when, and why it was excluded — and nothing more.
    Information that could not be found is left empty, never invented.

``usage`` is our commitment (``seed_only`` is the only value).
``usage_permitted`` is the checked finding that the licence allows it; it is
true exactly when the outcome is ``citable``.

Nothing in the registry is filled in from memory, and nothing here touches the
network.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

__all__ = [
    "CITABLE_EVIDENCE",
    "EXCLUDED_EVIDENCE",
    "SourceDataset",
    "SourceRegistry",
    "SourceRegistryError",
    "load_registry",
]

#: What must be recorded before a dataset may be marked citable. ``version`` is
#: not required: some providers state none, and it must not be invented.
CITABLE_EVIDENCE = (
    "access_url", "access_date", "licence", "licence_url",
    "attribution_required", "registration_required", "checked_by", "checked_at",
)

#: What an exclusion needs: who decided, when, and why.
EXCLUDED_EVIDENCE = ("checked_by", "checked_at", "exclusion_reason")

Outcome = Literal["unverified", "citable", "excluded"]
DatasetId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]


class SourceRegistryError(ValueError):
    """The registry is malformed or makes a claim its evidence does not support."""


class SourceDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_name: Annotated[str, Field(min_length=1)]
    citation: str | None = None
    outcome: Outcome = "unverified"
    version: str | None = None
    access_url: str | None = None
    access_date: date | None = None
    licence: str | None = None
    licence_url: str | None = None
    usage: Literal["seed_only"]
    usage_permitted: bool = False
    attribution_required: bool | None = None
    registration_required: bool | None = None
    checked_by: str | None = None
    checked_at: date | None = None
    exclusion_reason: str | None = None
    notes: str | None = None

    def _missing(self, fields: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(f for f in fields if getattr(self, f) in (None, ""))

    @property
    def missing_evidence(self) -> tuple[str, ...]:
        """Fields the declared outcome requires but that are still empty."""
        if self.outcome == "citable":
            return self._missing(CITABLE_EVIDENCE)
        if self.outcome == "excluded":
            return self._missing(EXCLUDED_EVIDENCE)
        return ()

    @property
    def is_citable(self) -> bool:
        return self.outcome == "citable"

    @model_validator(mode="after")
    def _outcome_is_supported(self) -> SourceDataset:
        for field in ("access_url", "licence_url"):
            url = getattr(self, field)
            if url is not None and not url.startswith(("https://", "http://")):
                raise ValueError(f"{field} must be an http(s) URL; got {url!r}")
        if self.access_date and self.checked_at and self.checked_at < self.access_date:
            raise ValueError("checked_at cannot precede access_date")

        if self.missing_evidence:
            raise ValueError(
                f"outcome {self.outcome!r} is not supported by the recorded evidence; "
                f"missing: {list(self.missing_evidence)}")
        if self.usage_permitted != (self.outcome == "citable"):
            raise ValueError("usage_permitted is true exactly when the outcome is 'citable'")
        if self.exclusion_reason is not None and self.outcome != "excluded":
            raise ValueError("exclusion_reason is recorded only for an excluded dataset")
        return self


class SourceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    datasets: dict[DatasetId, SourceDataset]

    def _with(self, outcome: str) -> list[str]:
        return sorted(k for k, d in self.datasets.items() if d.outcome == outcome)

    def citable(self) -> list[str]:
        """The only entries the topic bank may cite."""
        return self._with("citable")

    def citation_problems(self, keys: Iterable[str]) -> list[str]:
        """Why any of ``keys`` may not be cited; empty when all are citable.

        This is the gate that matters: every source a topic actually cites must
        be citable. A candidate nobody cites may stay unresolved.
        """
        problems = []
        for key in keys:
            dataset = self.datasets.get(key)
            if dataset is None:
                problems.append(f"{key}: not in the registry")
            elif dataset.outcome != "citable":
                problems.append(f"{key}: {dataset.outcome}, so it may not be cited")
        return problems

    def excluded(self) -> list[str]:
        return self._with("excluded")

    def unverified(self) -> list[str]:
        return self._with("unverified")


def load_registry(path: str | Path, today: date | None = None) -> SourceRegistry:
    """Load the registry and check it. Raises on anything it cannot support.

    ``today`` is injectable so the no-future-dates check is testable.
    """
    path = Path(path)
    today = today or date.today()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        registry = SourceRegistry.model_validate(raw)
    except (yaml.YAMLError, ValidationError) as exc:
        raise SourceRegistryError(f"{path}: {exc}") from exc

    problems = []
    names: dict[str, str] = {}
    for key, dataset in registry.datasets.items():
        for field in ("access_date", "checked_at"):
            value = getattr(dataset, field)
            if value is not None and value > today:
                problems.append(f"{key}: {field} {value} is in the future")
        other = names.get(dataset.canonical_name.casefold())
        if other:
            problems.append(f"{key}: canonical name duplicates {other!r}")
        names[dataset.canonical_name.casefold()] = key
    if problems:
        raise SourceRegistryError(f"{path}:\n  " + "\n  ".join(problems))
    return registry
