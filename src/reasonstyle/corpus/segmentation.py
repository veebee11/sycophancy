"""Deterministic sentence segmentation, pinned by library and version.

D1 makes exact sentence-count equality across RS, RP, NS and NP a *hard*
rejection criterion, so the segmenter is not a convenience: it decides whether
a valid item enters the corpus. Two consequences shape this module.

**The version is pinned and recorded.** A segmenter upgrade can silently change
counts, which would change which items are admissible. The expected version
comes from the experiment config, is checked against what is installed, and is
carried on every :class:`Segmentation` so it can be written into artefacts.

**The segmenter is a heuristic, not ground truth.** pysbd handles decimals and
``e.g.`` correctly but mis-segments ``"In the U.S. Storage costs fell"`` and
``"approx."``. Rather than pretend otherwise, constructions known to be
unreliable are detected and reported as :class:`Ambiguity` flags. The machine
count stays authoritative; a reviewer may flag a suspected error, but a
correction requires a recorded annotation (config ``segmentation.human_override``).

This module is deliberately model-agnostic: it concerns text only, and knows
nothing about tokenizers, checkpoints or inference backends.
"""

from __future__ import annotations

import importlib.metadata
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Ambiguity",
    "PysbdSegmenter",
    "SegmentationError",
    "Segmentation",
    "Segmenter",
    "SegmenterInfo",
    "SegmenterUnavailable",
    "SegmenterVersionMismatch",
    "build_ambiguity_patterns",
    "segmenter_from_config",
]


class SegmentationError(RuntimeError):
    """Base class for segmentation failures."""


class SegmenterUnavailable(SegmentationError):
    """The pinned segmentation library is not installed."""


class SegmenterVersionMismatch(SegmentationError):
    """The installed version differs from the version the config pins."""


@dataclass(frozen=True)
class SegmenterInfo:
    """Identity of the segmenter, recorded on every artefact."""

    library: str
    version: str
    language: str

    def as_dict(self) -> dict[str, str]:
        return {"library": self.library, "version": self.version, "language": self.language}


@dataclass(frozen=True)
class Ambiguity:
    """A construction the segmenter is known to handle unreliably.

    Not an error. It routes the text to human confirmation and, upstream, to a
    text restriction telling the generator to avoid the construction.
    """

    kind: str
    span: tuple[int, int]
    text: str


@dataclass(frozen=True)
class Segmentation:
    """The result of segmenting one text."""

    text: str
    sentences: tuple[str, ...]
    ambiguities: tuple[Ambiguity, ...]
    segmenter: SegmenterInfo

    @property
    def count(self) -> int:
        return len(self.sentences)

    @property
    def is_ambiguous(self) -> bool:
        return bool(self.ambiguities)

    def ambiguity_kinds(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(a.kind for a in self.ambiguities))


@runtime_checkable
class Segmenter(Protocol):
    """The interface the validator depends on, so the library can be swapped."""

    @property
    def info(self) -> SegmenterInfo: ...

    def segment(self, text: str) -> Segmentation: ...


def build_ambiguity_patterns(spec: dict[str, Any]) -> dict[str, re.Pattern[str]]:
    """Compile the ambiguity detectors from the config ``segmentation`` block.

    The ``abbreviation`` detector is generated from ``known_abbreviations``
    rather than stored as a regex, so that list stays the single source of
    truth and cannot drift from the pattern that enforces it.
    """
    patterns = {kind: re.compile(src) for kind, src in spec["ambiguity_patterns"].items()}

    abbreviations = spec["text_restrictions"]["known_abbreviations"]
    if abbreviations:
        alternatives = "|".join(re.escape(a) for a in sorted(abbreviations, key=len, reverse=True))
        patterns["abbreviation"] = re.compile(rf"(?<![A-Za-z])(?:{alternatives})", re.IGNORECASE)
    return patterns


def _detect(text: str, patterns: dict[str, re.Pattern[str]]) -> tuple[Ambiguity, ...]:
    found = [
        Ambiguity(kind=kind, span=m.span(), text=m.group(0))
        for kind, pattern in patterns.items()
        for m in pattern.finditer(text)
    ]
    found.sort(key=lambda a: (a.span[0], a.kind))
    return tuple(found)


class PysbdSegmenter:
    """pysbd-backed segmenter with a pinned version and ambiguity reporting."""

    def __init__(
        self,
        expected_version: str,
        language: str = "en",
        clean: bool = False,
        ambiguity_patterns: dict[str, re.Pattern[str]] | None = None,
    ) -> None:
        try:
            import pysbd
        except ImportError as exc:  # pragma: no cover - environment failure
            raise SegmenterUnavailable(
                "pysbd is not installed; it is pinned in the experiment config"
            ) from exc

        installed = importlib.metadata.version("pysbd")
        if installed != expected_version:
            raise SegmenterVersionMismatch(
                f"config pins pysbd=={expected_version} but {installed} is installed. "
                f"A segmenter change can alter sentence counts and therefore which "
                f"items are admissible under D1; bump the config, do not edit it."
            )

        self._info = SegmenterInfo(library="pysbd", version=installed, language=language)
        self._patterns = ambiguity_patterns or {}
        # `clean=False` is deliberate: the segmenter must never rewrite the text
        # it is measuring.
        self._segmenter = pysbd.Segmenter(language=language, clean=clean)

    @property
    def info(self) -> SegmenterInfo:
        return self._info

    def segment(self, text: str) -> Segmentation:
        sentences = tuple(s.strip() for s in self._segmenter.segment(text) if s.strip())
        return Segmentation(
            text=text,
            sentences=sentences,
            ambiguities=_detect(text, self._patterns),
            segmenter=self._info,
        )

    def count(self, text: str) -> int:
        """Sentence count only, for call sites that need nothing else."""
        return self.segment(text).count


def segmenter_from_config(cfg: Any) -> Segmenter:
    """Build the segmenter the config pins.

    Accepts an :class:`~reasonstyle.config.ExperimentConfig` or the raw mapping.
    Raises if the config predates the segmentation block (v1).
    """
    raw = getattr(cfg, "raw", cfg)
    spec = raw.get("segmentation")
    if spec is None:
        raise SegmentationError(
            "this config declares no `segmentation` block; a config of v2 or later is required"
        )
    if spec["library"] != "pysbd":
        raise SegmentationError(f"unsupported segmentation library {spec['library']!r}")

    return PysbdSegmenter(
        expected_version=spec["version"],
        language=spec["language"],
        clean=spec["clean"],
        ambiguity_patterns=build_ambiguity_patterns(spec),
    )
