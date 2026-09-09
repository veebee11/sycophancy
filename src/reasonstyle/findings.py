"""Validation findings and reports.

Four severities, and the third is the load-bearing one.

``error``
    A machine-checkable rule is broken. The item cannot enter the corpus.
``warning``
    Something a reviewer must look at but which does not automatically block:
    a word ratio in the tolerance band, a warning-severity forbidden phrase, a
    construction the segmenter handles unreliably.
``human_review``
    **Not a defect.** A construct no lexical validator can decide, emitted
    *unconditionally* for every item, pair, group and scenario. Substantive
    support, support direction, no-reason integrity, proposition preservation,
    naturalness and pragmatic commitment are human judgements. They are
    findings so that a clean machine run can never be mistaken for validated
    quality.
``info``
    A check that did not apply at this corpus scope, reported so its absence is
    visible rather than silent.

Consequently :attr:`ValidationReport.ok` means "no machine errors" and nothing
more. Approval additionally requires that every ``human_review`` code has been
discharged by a recorded annotation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["Finding", "Severity", "Scope", "ValidationReport"]

Severity = Literal["error", "warning", "human_review", "info"]
Scope = Literal["corpus", "decision", "scenario", "group", "pair", "cell"]


@dataclass(frozen=True, slots=True)
class Finding:
    """One validation outcome, located as precisely as it can be."""

    code: str
    severity: Severity
    message: str
    scope: Scope
    decision_id: str | None = None
    scenario_id: str | None = None
    supported_option: str | None = None
    condition: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def locator(self) -> str:
        parts = [p for p in (self.scenario_id, self.supported_option, self.condition) if p]
        return "/".join(parts) if parts else (self.decision_id or "corpus")

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code} at {self.locator}: {self.message}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "scope": self.scope,
            "decision_id": self.decision_id,
            "scenario_id": self.scenario_id,
            "supported_option": self.supported_option,
            "condition": self.condition,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """The outcome of validating a corpus, with its provenance."""

    config_content_hash: str
    config_version: str
    corpus_content_hash: str
    segmenter: dict[str, str]
    corpus_scope: str
    n_scenarios: int
    n_texts: int
    findings: tuple[Finding, ...]

    # -- filtered views -----------------------------------------------------
    def by_severity(self, severity: Severity) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == severity)

    @property
    def errors(self) -> tuple[Finding, ...]:
        return self.by_severity("error")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return self.by_severity("warning")

    @property
    def human_review(self) -> tuple[Finding, ...]:
        return self.by_severity("human_review")

    @property
    def info(self) -> tuple[Finding, ...]:
        return self.by_severity("info")

    def codes(self, severity: Severity | None = None) -> Counter[str]:
        source = self.findings if severity is None else self.by_severity(severity)
        return Counter(f.code for f in source)

    # -- verdicts -----------------------------------------------------------
    @property
    def ok(self) -> bool:
        """No machine errors. **Not** a statement that the corpus is good."""
        return not self.errors

    @property
    def requires_human_review(self) -> bool:
        """Always true for a non-empty corpus: the H_ codes are unconditional."""
        return bool(self.human_review)

    def outstanding_human_review(self, scenario_id: str) -> tuple[str, ...]:
        """The human-review codes a given scenario must still discharge."""
        return tuple(
            sorted({f.code for f in self.human_review if f.scenario_id == scenario_id})
        )

    # -- presentation -------------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"corpus scope   : {self.corpus_scope}",
            f"config         : {self.config_version} ({self.config_content_hash[:12]})",
            f"corpus         : {self.n_scenarios} scenarios, {self.n_texts} texts "
            f"({self.corpus_content_hash[:12]})",
            f"segmenter      : {self.segmenter.get('library')} {self.segmenter.get('version')}",
            "",
            f"errors         : {len(self.errors)}",
            f"warnings       : {len(self.warnings)}",
            f"human review   : {len(self.human_review)} outstanding judgements",
            f"skipped checks : {len(self.info)}",
            "",
        ]
        for severity in ("error", "warning", "human_review", "info"):
            counts = self.codes(severity)  # type: ignore[arg-type]
            if counts:
                lines.append(f"{severity}:")
                lines += [f"  {code:<38} {n}" for code, n in sorted(counts.items())]
        lines += [
            "",
            "MACHINE-VALID" if self.ok else "REJECTED",
            "This reports machine-checkable rules only. Substantive support, support",
            "direction, no-reason integrity, proposition preservation, naturalness and",
            "pragmatic commitment are human judgements and are listed above as",
            "outstanding human review. A clean run is not an approved corpus.",
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_content_hash": self.config_content_hash,
            "config_version": self.config_version,
            "corpus_content_hash": self.corpus_content_hash,
            "segmenter": self.segmenter,
            "corpus_scope": self.corpus_scope,
            "n_scenarios": self.n_scenarios,
            "n_texts": self.n_texts,
            "ok": self.ok,
            "requires_human_review": self.requires_human_review,
            "findings": [f.as_dict() for f in self.findings],
        }
