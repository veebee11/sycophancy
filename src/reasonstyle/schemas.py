"""Source-data schemas for the scenario corpus (Research_Plan_v6 §15.1).

One record per scenario. Each scenario carries two **direction blocks**, one
per immutable semantic option, and each block carries the four condition cells.

Three principles shape these models.

**Semantic option IDs only.** ``opt_1`` and ``opt_2`` are the identities.
Display letters A and B are produced by the prompt renderer and appear nowhere
in stored data. Support direction comes from ``supported_option`` plus human
review, never from letters in the text.

**The marker realization is stored once and inherited.** ``marker_family``,
``marker_string`` and ``marker_realization_id`` live on the
:class:`DirectionBlock`. Cells hold no copy of the string or the realization,
so there is no second place for them to disagree; :meth:`DirectionBlock.resolve`
composes the inherited view on demand. RS and NS are both styled with the *same*
realization — that is what makes ``RS - NS`` a content contrast, since the
realization cancels. Marker *absence* is what defines RP and NP, making
``RP - NP`` the corresponding content contrast in plain language.

**These models check shape, never substance.** Structural consistency, key
agreement and inheritance are enforced here. Whether a reason is relevant,
valid or genuinely supporting is a human judgement recorded in the annotation
tables; whether counts and phrases satisfy the corpus rules is the validator's
job. Nothing in this module claims to assess reasoning quality.

The module is model-agnostic: it knows nothing of tokenizers, checkpoints or
inference backends.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import FROZEN_CORE_CONDITIONS

__all__ = [
    "Cell",
    "Condition",
    "DirectionBlock",
    "GenerationMetadata",
    "Measurements",
    "ResolvedCell",
    "ScenarioRecord",
    "SegmenterRef",
    "SemanticOption",
    "SourceReference",
    "ValidationStatus",
]

Condition = Literal["RS", "RP", "NS", "NP"]
SemanticOption = Literal["opt_1", "opt_2"]
SourceType = Literal["constructed", "adapted", "mixed"]
ValidationState = Literal["draft", "machine_validated", "human_reviewed", "approved", "rejected"]

CORE_CONDITIONS: tuple[Condition, ...] = ("RS", "RP", "NS", "NP")
SEMANTIC_OPTIONS: tuple[SemanticOption, ...] = ("opt_1", "opt_2")

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", min_length=3, max_length=64)]
NonEmpty = Annotated[str, Field(min_length=1)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


class SourceReference(_Model):
    """One reference source a scenario draws on.

    An **open structure, not an enum**: no dataset is privileged and adding a
    new source needs no schema change. A scenario may cite several sources, or
    none — a fully constructed scenario carries ``source_type: constructed``
    and an empty list.
    """

    dataset_name: NonEmpty
    dataset_version: str | None = None       # version or revision
    source_item_id: str | None = None        # when the source exposes one
    source_url: str | None = None
    access_date: date | None = None
    reuse_licence: str | None = None         # when known
    notes: str | None = None


class GenerationMetadata(_Model):
    """How an LLM produced a *draft*.

    Kept strictly separate from :class:`SourceReference`: this is not a source
    and not a licence claim. LLM output is a draft only — no item enters the
    frozen corpus without human approval (plan §5).
    """

    generator_model: NonEmpty
    generator_model_revision: str | None = None   # immutable id when available
    prompt_hash: Sha256
    generation_parameters: dict[str, Any]
    seed: int | None = None
    generated_at: date


# --------------------------------------------------------------------------
# Measurements and validation status
# --------------------------------------------------------------------------


class SegmenterRef(_Model):
    """Identity of the segmenter that produced a sentence count."""

    library: NonEmpty
    version: NonEmpty
    language: NonEmpty


class Measurements(_Model):
    """Counts computed by the validator. Never written by hand.

    ``full`` is the rendered counterargument (shared opening + body); ``body``
    is the manipulated part alone. Both are recorded for words and sentences so
    no reader has to guess which text a count refers to. The word-count ratio is
    enforced on both, so a long shared opening cannot dilute an imbalance in the
    body. D1's exact sentence-count equality is unaffected by the choice, since
    the opening contributes the same number of sentences to all eight texts.
    """

    word_count_full: int = Field(ge=0)
    word_count_body: int = Field(ge=0)
    sentence_count_full: int = Field(ge=0)
    sentence_count_body: int = Field(ge=0)
    segmenter: SegmenterRef
    ambiguity_kinds: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _body_is_within_full(self) -> Measurements:
        if self.word_count_body > self.word_count_full:
            raise ValueError("word_count_body cannot exceed word_count_full")
        if self.sentence_count_body > self.sentence_count_full:
            raise ValueError("sentence_count_body cannot exceed sentence_count_full")
        return self


class ValidationStatus(_Model):
    """Where an item stands. ``machine_validated`` is not ``approved``."""

    status: ValidationState = "draft"
    machine_errors: int | None = Field(default=None, ge=0)
    machine_warnings: int | None = Field(default=None, ge=0)
    outstanding_human_review: tuple[str, ...] = ()
    reviewed_by: str | None = None
    reviewed_at: date | None = None

    @model_validator(mode="after")
    def _status_is_supported_by_its_evidence(self) -> ValidationStatus:
        if self.status == "draft":
            return self
        if self.machine_errors is None or self.machine_warnings is None:
            raise ValueError(f"status {self.status!r} requires machine error and warning counts")
        if self.status in {"human_reviewed", "approved"} and not (self.reviewed_by and self.reviewed_at):
            raise ValueError(f"status {self.status!r} requires reviewed_by and reviewed_at")
        if self.status == "approved":
            if self.machine_errors:
                raise ValueError("an approved item cannot carry machine errors")
            if self.outstanding_human_review:
                raise ValueError(
                    "an approved item cannot have outstanding human review: "
                    f"{list(self.outstanding_human_review)}"
                )
        return self


# --------------------------------------------------------------------------
# Cells and direction blocks
# --------------------------------------------------------------------------


class Cell(_Model):
    """One condition cell.

    ``body`` excludes the scenario-level opening, which is stored once and
    prepended at render time (D6), so exact equality of the opening across all
    eight texts holds by construction rather than by check.

    ``marker_family`` is redundant with the block's for styled cells and is
    ``None`` for plain cells; it is retained for literal compatibility with
    plan §15.1 and validated against the block.
    """

    condition: Condition
    body: NonEmpty
    markers_present: bool
    marker_family: str | None = None
    measurements: Measurements | None = None

    @model_validator(mode="after")
    def _markers_present_matches_the_frozen_cell(self) -> Cell:
        expected = FROZEN_CORE_CONDITIONS[self.condition][1] == "explicit"
        if self.markers_present is not expected:
            raise ValueError(
                f"condition {self.condition} is "
                f"{'styled' if expected else 'plain'}, so markers_present must be {expected}"
            )
        if not expected and self.marker_family is not None:
            raise ValueError(f"plain cell {self.condition} must carry marker_family: null (§15.1)")
        if expected and self.marker_family is None:
            raise ValueError(f"styled cell {self.condition} must name its marker family")
        return self


class ResolvedCell(_Model):
    """A cell with the block's realization fields resolved onto it.

    Produced by :meth:`DirectionBlock.resolve`; never stored. This is what
    "stored once and inherited" means in practice — inheritance is composed on
    demand, so the realization cannot drift between cells.
    """

    condition: Condition
    supported_option: SemanticOption
    body: NonEmpty
    markers_present: bool
    marker_family: str | None
    inherited_marker_family: str
    marker_string: str
    marker_realization_id: str
    measurements: Measurements | None = None


class DirectionBlock(_Model):
    """The four cells supporting one semantic option.

    This is the ``(scenario_id, supported_option)`` group. The marker family,
    the marker string and the syntactic realization are stored **here, once**,
    and every cell inherits them: RS and NS must instantiate the same
    realization, or ``RS - NS`` would also contrast one marker or placement
    against another.
    """

    supported_option: SemanticOption
    marker_family: NonEmpty
    marker_string: NonEmpty
    marker_realization_id: Identifier
    cells: dict[Condition, Cell]

    @model_validator(mode="after")
    def _cells_are_complete_and_consistent(self) -> DirectionBlock:
        if set(self.cells) != set(CORE_CONDITIONS):
            missing = sorted(set(CORE_CONDITIONS) - set(self.cells))
            extra = sorted(set(self.cells) - set(CORE_CONDITIONS))
            raise ValueError(f"a direction block needs exactly the four core cells (missing={missing}, extra={extra})")
        for key, cell in self.cells.items():
            if cell.condition != key:
                raise ValueError(f"cell filed under {key!r} declares condition {cell.condition!r}")
            if cell.markers_present and cell.marker_family != self.marker_family:
                raise ValueError(
                    f"styled cell {key} declares marker family {cell.marker_family!r} but its "
                    f"group is {self.marker_family!r}; all four cells share one realization"
                )
        return self

    def resolve(self, condition: Condition) -> ResolvedCell:
        """Compose the inherited view of one cell."""
        cell = self.cells[condition]
        return ResolvedCell(
            condition=cell.condition,
            supported_option=self.supported_option,
            body=cell.body,
            markers_present=cell.markers_present,
            marker_family=cell.marker_family,
            inherited_marker_family=self.marker_family,
            marker_string=self.marker_string,
            marker_realization_id=self.marker_realization_id,
            measurements=cell.measurements,
        )

    def resolved_cells(self) -> tuple[ResolvedCell, ...]:
        return tuple(self.resolve(c) for c in CORE_CONDITIONS)


# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------

_SCENARIO_ID = re.compile(r"^(?P<decision>[a-z][a-z0-9_]*)_v(?P<variant>\d+)$")


class ScenarioRecord(_Model):
    """One scenario: two semantic options and eight directional counterarguments."""

    schema_version: NonEmpty
    config_version: NonEmpty
    config_content_hash: Sha256

    decision_id: Identifier
    scenario_id: NonEmpty
    variant_id: int = Field(ge=1)
    domain: NonEmpty                     # checked against config.domains by the validator
    scenario_text: NonEmpty

    options: dict[SemanticOption, str]
    counterargument_opening: NonEmpty    # D6: stored once, shared by all eight
    counterarguments: dict[SemanticOption, DirectionBlock]

    source_type: SourceType
    source_references: tuple[SourceReference, ...] = ()
    generation: GenerationMetadata | None = None

    validation: ValidationStatus = ValidationStatus()

    @model_validator(mode="after")
    def _structure_is_consistent(self) -> ScenarioRecord:
        m = _SCENARIO_ID.fullmatch(self.scenario_id)
        if m is None:
            raise ValueError(f"scenario_id {self.scenario_id!r} must look like '<decision_id>_v<variant>'")
        if m.group("decision") != self.decision_id:
            raise ValueError(f"scenario_id {self.scenario_id!r} does not belong to decision {self.decision_id!r}")
        if int(m.group("variant")) != self.variant_id:
            raise ValueError(f"scenario_id {self.scenario_id!r} disagrees with variant_id {self.variant_id}")

        if set(self.options) != set(SEMANTIC_OPTIONS):
            raise ValueError(f"a scenario needs exactly {list(SEMANTIC_OPTIONS)}; got {sorted(self.options)}")
        for key, text in self.options.items():
            if not text.strip():
                raise ValueError(f"option {key} has no text")

        if set(self.counterarguments) != set(SEMANTIC_OPTIONS):
            raise ValueError(
                f"a scenario needs one direction block per semantic option; got {sorted(self.counterarguments)}"
            )
        for key, block in self.counterarguments.items():
            if block.supported_option != key:
                raise ValueError(
                    f"direction block filed under {key!r} declares supported_option {block.supported_option!r}"
                )

        if self.source_type == "constructed" and self.source_references:
            raise ValueError("a constructed scenario cites no source; use 'adapted' or 'mixed'")
        if self.source_type in {"adapted", "mixed"} and not self.source_references:
            raise ValueError(f"source_type {self.source_type!r} requires at least one source reference")
        return self

    # -- derived views ------------------------------------------------------

    @property
    def counterargument_count(self) -> int:
        """Always eight: two semantic options x four conditions."""
        return sum(len(b.cells) for b in self.counterarguments.values())

    def render(self, supported_option: SemanticOption, condition: Condition) -> str:
        """The full counterargument the model would see: opening + body."""
        body = self.counterarguments[supported_option].cells[condition].body
        return f"{self.counterargument_opening} {body}"

    def opposing_option(self, option: SemanticOption) -> SemanticOption:
        return "opt_2" if option == "opt_1" else "opt_1"

    def resolved_cells(self) -> tuple[ResolvedCell, ...]:
        return tuple(
            cell
            for option in SEMANTIC_OPTIONS
            for cell in self.counterarguments[option].resolved_cells()
        )
