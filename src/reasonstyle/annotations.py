"""Human annotation records, at the three levels of D13.

Judgements are stored at the level at which they are actually made — item, pair
or scenario — never duplicated downward onto every item, which would inflate
apparent sample size and let one judgement disagree with itself.

Two families of record:

*Canonical* (:class:`ItemAnnotation`, :class:`PairAnnotation`,
:class:`ScenarioAnnotation`) carry the real keys and are what the analysis
reads. The single curator writes these directly.

*Blinded* (:class:`BlindItemResponse` and friends) are what an independent
reliability annotator writes. They are keyed only by a ``blind_id`` and contain
no condition, no marker metadata and no intended answer. :func:`unblind_items`
and its siblings join them to the key to produce canonical records.

A blinded annotator is never asked a question whose phrasing reveals the
design: support direction is asked as "P, Q or unclear", and no-reason
integrity is asked as "does this supply a premise?", which is answerable
without knowing whether the cell is a reason cell.

These records are never written into the generated Markdown review; the
Markdown is a read-only view.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import Condition, SemanticOption

__all__ = [
    "AnnotationError",
    "BlindItemKey",
    "BlindItemResponse",
    "BlindPairKey",
    "BlindPairResponse",
    "BlindScenarioKey",
    "BlindScenarioResponse",
    "ItemAnnotation",
    "Likert",
    "PairAnnotation",
    "PairId",
    "ScenarioAnnotation",
    "load_annotations",
    "save_annotations",
    "unblind_items",
    "unblind_pairs",
    "unblind_scenarios",
]

Likert = Annotated[int, Field(ge=1, le=5)]
PairId = Literal["RS_RP", "NS_NP"]
BlindLabel = Literal["P", "Q"]
DirectionAnswer = Literal["P", "Q", "unclear"]
Round = Literal["full_review", "reliability"]


class AnnotationError(ValueError):
    """An annotation file could not be read, or a blind record could not be joined."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------
# Canonical annotations (D13)
# --------------------------------------------------------------------------


class ItemAnnotation(_Model):
    """One reviewer's judgement of one counterargument cell."""

    scenario_id: str
    supported_option: SemanticOption
    condition: Condition
    annotator_id: str
    annotated_at: date
    round: Round
    config_content_hash: str
    corpus_content_hash: str
    blind_id: str | None = None          # set when derived from a blinded response

    # support direction: the raw answer and whether it matched the intent
    support_direction_answer: Literal["opt_1", "opt_2", "unclear"]
    support_direction_confirmed: bool

    # does the text supply a premise making the option more desirable?
    supplies_premise: bool

    substantive_support: Likert
    perceived_reasoning_style: Likert
    perceived_speaker_commitment: Likert
    perceived_unstated_support: Likert
    perceived_naturalness: Likert
    confidence: Likert
    pressure: Likert
    politeness: Likert
    authority: Likert
    credibility: Likert

    note: str | None = None

    @property
    def no_reason_integrity(self) -> bool | None:
        """Derived, and only meaningful for the no-reason cells: a cell whose
        condition supplies no reason has integrity when no premise was found."""
        if self.condition not in ("NS", "NP"):
            return None
        return not self.supplies_premise

    @model_validator(mode="after")
    def _direction_answer_is_consistent(self) -> ItemAnnotation:
        expected = self.support_direction_answer == self.supported_option
        if self.support_direction_confirmed is not expected:
            raise ValueError(
                "support_direction_confirmed must equal "
                "(support_direction_answer == supported_option)"
            )
        return self


class PairAnnotation(_Model):
    """Proposition preservation, judged once per pair — never per item."""

    scenario_id: str
    supported_option: SemanticOption
    pair_id: PairId
    annotator_id: str
    annotated_at: date
    round: Round
    config_content_hash: str
    corpus_content_hash: str
    blind_id: str | None = None

    proposition_preservation: Likert
    differences_noted: str | None = None


class ScenarioAnnotation(_Model):
    """Scenario validity, judged once per scenario."""

    scenario_id: str
    annotator_id: str
    annotated_at: date
    round: Round
    config_content_hash: str
    corpus_content_hash: str
    blind_id: str | None = None

    option_feasibility: Likert
    option_non_dominance: Likert
    normative_underdetermination: Likert
    note: str | None = None


# --------------------------------------------------------------------------
# Blinded responses and keys
# --------------------------------------------------------------------------


class BlindItemKey(_Model):
    """Unblinding key for one item. Lives in ``blind_key/``, never beside the packet."""

    blind_id: str
    level: Literal["item"] = "item"
    scenario_id: str
    supported_option: SemanticOption
    condition: Condition
    option_labels: dict[BlindLabel, SemanticOption]

    @model_validator(mode="after")
    def _labels_cover_both_options(self) -> BlindItemKey:
        if set(self.option_labels) != {"P", "Q"} or set(self.option_labels.values()) != {"opt_1", "opt_2"}:
            raise ValueError("option_labels must map P and Q onto opt_1 and opt_2")
        return self


class BlindPairKey(_Model):
    blind_id: str
    level: Literal["pair"] = "pair"
    scenario_id: str
    supported_option: SemanticOption
    pair_id: PairId
    #: which displayed side ("1"/"2") holds which condition — randomised
    side_conditions: dict[Literal["1", "2"], Condition]


class BlindScenarioKey(_Model):
    blind_id: str
    level: Literal["scenario"] = "scenario"
    scenario_id: str
    option_labels: dict[BlindLabel, SemanticOption]


class BlindItemResponse(_Model):
    """What an independent annotator writes. No condition, no intended answer."""

    blind_id: str
    annotator_id: str
    annotated_at: date

    supports_option: DirectionAnswer          # P / Q / unclear
    supplies_premise: bool

    substantive_support: Likert
    perceived_reasoning_style: Likert
    perceived_speaker_commitment: Likert
    perceived_unstated_support: Likert
    perceived_naturalness: Likert
    confidence: Likert
    pressure: Likert
    politeness: Likert
    authority: Likert
    credibility: Likert

    note: str | None = None


class BlindPairResponse(_Model):
    blind_id: str
    annotator_id: str
    annotated_at: date
    proposition_preservation: Likert
    differences_noted: str | None = None


class BlindScenarioResponse(_Model):
    blind_id: str
    annotator_id: str
    annotated_at: date
    option_feasibility: Likert
    option_non_dominance: Likert
    normative_underdetermination: Likert
    note: str | None = None


# --------------------------------------------------------------------------
# Unblinding
# --------------------------------------------------------------------------


def _key_index(keys: Iterable[Any]) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for key in keys:
        if key.blind_id in index:
            raise AnnotationError(f"duplicate blind_id {key.blind_id!r} in the key")
        index[key.blind_id] = key
    return index


def unblind_items(
    responses: Sequence[BlindItemResponse],
    keys: Sequence[BlindItemKey],
    *,
    config_content_hash: str,
    corpus_content_hash: str,
) -> list[ItemAnnotation]:
    """Join blinded responses to the key, comparing the answer with the intent."""
    index = _key_index(keys)
    out: list[ItemAnnotation] = []
    for response in responses:
        key = index.get(response.blind_id)
        if key is None:
            raise AnnotationError(f"no key entry for blind_id {response.blind_id!r}")
        answer: Literal["opt_1", "opt_2", "unclear"] = (
            "unclear" if response.supports_option == "unclear"
            else key.option_labels[response.supports_option]   # type: ignore[index]
        )
        out.append(ItemAnnotation(
            scenario_id=key.scenario_id,
            supported_option=key.supported_option,
            condition=key.condition,
            annotator_id=response.annotator_id,
            annotated_at=response.annotated_at,
            round="reliability",
            config_content_hash=config_content_hash,
            corpus_content_hash=corpus_content_hash,
            blind_id=response.blind_id,
            support_direction_answer=answer,
            support_direction_confirmed=answer == key.supported_option,
            supplies_premise=response.supplies_premise,
            substantive_support=response.substantive_support,
            perceived_reasoning_style=response.perceived_reasoning_style,
            perceived_speaker_commitment=response.perceived_speaker_commitment,
            perceived_unstated_support=response.perceived_unstated_support,
            perceived_naturalness=response.perceived_naturalness,
            confidence=response.confidence,
            pressure=response.pressure,
            politeness=response.politeness,
            authority=response.authority,
            credibility=response.credibility,
            note=response.note,
        ))
    return out


def unblind_pairs(responses, keys, *, config_content_hash: str, corpus_content_hash: str):
    index = _key_index(keys)
    out: list[PairAnnotation] = []
    for response in responses:
        key = index.get(response.blind_id)
        if key is None:
            raise AnnotationError(f"no key entry for blind_id {response.blind_id!r}")
        out.append(PairAnnotation(
            scenario_id=key.scenario_id, supported_option=key.supported_option,
            pair_id=key.pair_id, annotator_id=response.annotator_id,
            annotated_at=response.annotated_at, round="reliability",
            config_content_hash=config_content_hash, corpus_content_hash=corpus_content_hash,
            blind_id=response.blind_id,
            proposition_preservation=response.proposition_preservation,
            differences_noted=response.differences_noted))
    return out


def unblind_scenarios(responses, keys, *, config_content_hash: str, corpus_content_hash: str):
    index = _key_index(keys)
    out: list[ScenarioAnnotation] = []
    for response in responses:
        key = index.get(response.blind_id)
        if key is None:
            raise AnnotationError(f"no key entry for blind_id {response.blind_id!r}")
        out.append(ScenarioAnnotation(
            scenario_id=key.scenario_id, annotator_id=response.annotator_id,
            annotated_at=response.annotated_at, round="reliability",
            config_content_hash=config_content_hash, corpus_content_hash=corpus_content_hash,
            blind_id=response.blind_id,
            option_feasibility=response.option_feasibility,
            option_non_dominance=response.option_non_dominance,
            normative_underdetermination=response.normative_underdetermination,
            note=response.note))
    return out


# --------------------------------------------------------------------------
# JSONL storage — separate from the generated Markdown views
# --------------------------------------------------------------------------


def load_annotations(path: str | Path, model: type[BaseModel]) -> list[Any]:
    """Read a JSONL annotation table, reporting the line number of a bad record."""
    import json

    from pydantic import ValidationError

    path = Path(path)
    if not path.exists():
        return []
    out = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(model.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise AnnotationError(f"{path}:{lineno}: {exc}") from exc
    return out


def save_annotations(records: Iterable[BaseModel], path: str | Path) -> Path:
    """Write annotations as canonical JSONL."""
    from .hashing import canonical_json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{canonical_json(r.model_dump(mode='json'))}\n" for r in records),
        encoding="utf-8",
    )
    return path
