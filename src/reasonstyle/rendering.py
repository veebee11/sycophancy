"""Prompt rendering: model-independent transcripts, option orders, four branches.

The canonical object is a :class:`Transcript` — an ordered sequence of turns
with roles. It is *model-independent*: it knows nothing of tokenizers,
checkpoints or chat templates, and it is what every hash is computed over.

Turning a transcript into the string a model actually sees is a separate step,
and it splits by template:

``plain_scaffold``
    A frozen plain-text dialogue scaffold, authored and hashed in the config.
    :func:`materialize` produces it here.

``tokenizer_chat_template``
    Deferred. An instruct model's chat template belongs to *its own tokenizer*,
    and no model has been selected, so authoring one here would hard-code a
    checkpoint. :func:`materialize` refuses, and the model adapter applies the
    tokenizer's own template at the compatibility stage.

The conversation is frozen at three turns::

    turn 1  user       scenario, options as A/B, the fixed question
                       -> answer slot: the initial argmax, m_before
    turn 2  assistant  the model's chosen display label
    turn 3  user       one counterargument, then the SAME fixed question
                       -> answer slot: m_after

Turns 1 and 2 are shared byte-for-byte by all four pushback branches, so the
four conditions never appear together in one conversation.

**Semantic identity never becomes a display letter.** ``opt_1`` and ``opt_2``
are the identities throughout; A and B exist only in
:attr:`OptionOrder.label_to_option` and in rendered text. Which counterargument
a branch carries is decided by ``counter_target_option``, a semantic id, and is
invariant under swapping the display order.

No model or tokenizer is imported anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .config import ExperimentConfig
from .hashing import content_hash
from .schemas import CORE_CONDITIONS, Condition, ScenarioRecord, SemanticOption

__all__ = [
    "AnswerContinuationUnresolved",
    "InitialPrompt",
    "MaterializationDeferred",
    "branches_share_prefix",
    "OptionOrder",
    "PushbackBranch",
    "RenderingError",
    "Transcript",
    "Turn",
    "materialize",
    "option_orders",
    "render_branches",
    "render_initial",
]

Role = Literal["user", "assistant"]


class RenderingError(ValueError):
    """A transcript could not be rendered or materialized."""


class MaterializationDeferred(RenderingError):
    """This template is materialized by the tokenizer, not here."""


class AnswerContinuationUnresolved(RenderingError):
    """The exact answer continuation is not known until the model stage."""


# --------------------------------------------------------------------------
# Canonical, model-independent transcript
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Turn:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class Transcript:
    """An ordered conversation, ending where the model must answer."""

    turns: tuple[Turn, ...]
    answer_cue: str

    @property
    def transcript_hash(self) -> str:
        return content_hash({"turns": [t.as_dict() for t in self.turns],
                             "answer_cue": self.answer_cue})

    def prefix(self, n: int) -> Transcript:
        """The first ``n`` turns, as a transcript in its own right."""
        return Transcript(turns=self.turns[:n], answer_cue=self.answer_cue)

    def as_list(self) -> list[dict[str, str]]:
        return [t.as_dict() for t in self.turns]


@dataclass(frozen=True, slots=True)
class OptionOrder:
    """A counterbalanced mapping of display letters onto semantic options."""

    order_id: str
    label_to_option: dict[str, SemanticOption]

    def option_for(self, label: str) -> SemanticOption:
        return self.label_to_option[label]

    def label_for(self, option: SemanticOption) -> str:
        for label, opt in self.label_to_option.items():
            if opt == option:
                return label
        raise RenderingError(f"option {option!r} has no display label in order {self.order_id!r}")

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.label_to_option))


def option_orders(cfg: ExperimentConfig) -> tuple[OptionOrder, ...]:
    """Both counterbalanced orders, as frozen in the config."""
    out = []
    for entry in cfg.parsed.prompts.option_orders:
        mapping = {k: v for k, v in entry.items() if k != "order_id"}
        out.append(OptionOrder(order_id=entry["order_id"], label_to_option=mapping))  # type: ignore[arg-type]
    return tuple(out)


# --------------------------------------------------------------------------
# Rendered artefacts
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InitialPrompt:
    """Turn 1 plus the answer slot: the initial-choice reading."""

    decision_id: str
    scenario_id: str
    order_id: str
    template_name: str
    transcript: Transcript
    label_to_option: dict[str, SemanticOption]
    answer_labels: tuple[str, ...]
    record_hash: str
    config_version: str
    config_content_hash: str

    @property
    def transcript_hash(self) -> str:
        return self.transcript.transcript_hash

    @property
    def prompt_hash(self) -> str:
        return content_hash({
            "transcript_hash": self.transcript_hash,
            "template_name": self.template_name,
            "order_id": self.order_id,
            "answer_labels": list(self.answer_labels),
            "config_content_hash": self.config_content_hash,
            "record_hash": self.record_hash,
        })


@dataclass(frozen=True, slots=True)
class PushbackBranch:
    """One of four independent conversations, differing only in turn 3."""

    decision_id: str
    scenario_id: str
    order_id: str
    template_name: str
    condition: Condition
    initial_option: SemanticOption
    initial_label: str
    counter_target_option: SemanticOption
    counter_target_label: str
    marker_family: str
    marker_string: str
    marker_realization_id: str
    transcript: Transcript
    label_to_option: dict[str, SemanticOption]
    answer_labels: tuple[str, ...]
    record_hash: str
    config_version: str
    config_content_hash: str

    @property
    def transcript_hash(self) -> str:
        return self.transcript.transcript_hash

    @property
    def shared_prefix_hash(self) -> str:
        """Hash of turns 1-2, identical across all four branches."""
        return self.transcript.prefix(2).transcript_hash

    @property
    def prompt_hash(self) -> str:
        return content_hash({
            "transcript_hash": self.transcript_hash,
            "template_name": self.template_name,
            "order_id": self.order_id,
            "condition": self.condition,
            "counter_target_option": self.counter_target_option,
            "answer_labels": list(self.answer_labels),
            "config_content_hash": self.config_content_hash,
            "record_hash": self.record_hash,
        })


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _question_block(cfg: ExperimentConfig) -> str:
    prompts = cfg.parsed.prompts
    return f"{prompts.question_text} {prompts.instruction_text}"


def _option_lines(record: ScenarioRecord, order: OptionOrder, cfg: ExperimentConfig) -> str:
    fmt = cfg.parsed.prompts.option_line_format
    assert fmt is not None
    return "\n".join(
        fmt.format(label=label, option_text=record.options[order.option_for(label)])
        for label in order.labels
    )


def _turn_one(record: ScenarioRecord, order: OptionOrder, cfg: ExperimentConfig) -> Turn:
    return Turn(role="user", content="\n\n".join(
        [record.scenario_text, _option_lines(record, order, cfg), _question_block(cfg)]))


def _turn_three(record: ScenarioRecord, option: SemanticOption, condition: Condition,
                cfg: ExperimentConfig) -> Turn:
    body = record.counterarguments[option].cells[condition].body
    counterargument = f"{record.counterargument_opening} {body}"
    return Turn(role="user", content="\n\n".join([counterargument, _question_block(cfg)]))


def _require_v4(cfg: ExperimentConfig) -> None:
    if cfg.parsed.prompts.question_text is None:
        raise RenderingError(
            f"config {cfg.config_version} declares no frozen question; "
            f"rendering requires a config of v4 or later")


def _template(cfg: ExperimentConfig, template_name: str):
    try:
        return cfg.parsed.prompts.templates[template_name]
    except KeyError as exc:
        raise RenderingError(f"unknown template {template_name!r}") from exc


def render_initial(record: ScenarioRecord, order: OptionOrder, cfg: ExperimentConfig,
                   template_name: str) -> InitialPrompt:
    """Render turn 1. The next token at the answer slot gives the initial choice."""
    _require_v4(cfg)
    template = _template(cfg, template_name)
    assert template.answer_cue is not None
    return InitialPrompt(
        decision_id=record.decision_id,
        scenario_id=record.scenario_id,
        order_id=order.order_id,
        template_name=template_name,
        transcript=Transcript(turns=(_turn_one(record, order, cfg),),
                              answer_cue=template.answer_cue),
        label_to_option=dict(order.label_to_option),
        answer_labels=tuple(cfg.parsed.prompts.answer_labels),
        record_hash=content_hash(record.model_dump(mode="json")),
        config_version=cfg.config_version,
        config_content_hash=cfg.content_hash,
    )


def render_branches(record: ScenarioRecord, order: OptionOrder, cfg: ExperimentConfig,
                    template_name: str, initial_option: SemanticOption
                    ) -> tuple[PushbackBranch, ...]:
    """Four independent conversations, one per core condition.

    Every branch carries a counterargument supporting the option **opposite**
    the model's initial choice, and turns 1-2 are identical across all four.
    """
    _require_v4(cfg)
    if initial_option not in record.options:
        raise RenderingError(f"{initial_option!r} is not a semantic option of {record.scenario_id}")

    template = _template(cfg, template_name)
    assert template.answer_cue is not None
    counter_target = record.opposing_option(initial_option)
    block = record.counterarguments[counter_target]

    # Built once and reused, so the branches cannot drift apart.
    shared = (_turn_one(record, order, cfg),
              Turn(role="assistant", content=order.label_for(initial_option)))

    common = dict(
        decision_id=record.decision_id, scenario_id=record.scenario_id,
        order_id=order.order_id, template_name=template_name,
        initial_option=initial_option, initial_label=order.label_for(initial_option),
        counter_target_option=counter_target,
        counter_target_label=order.label_for(counter_target),
        marker_family=block.marker_family, marker_string=block.marker_string,
        marker_realization_id=block.marker_realization_id,
        label_to_option=dict(order.label_to_option),
        answer_labels=tuple(cfg.parsed.prompts.answer_labels),
        record_hash=content_hash(record.model_dump(mode="json")),
        config_version=cfg.config_version, config_content_hash=cfg.content_hash,
    )
    return tuple(
        PushbackBranch(
            condition=condition,
            transcript=Transcript(
                turns=(*shared, _turn_three(record, counter_target, condition, cfg)),
                answer_cue=template.answer_cue),
            **common,
        )
        for condition in CORE_CONDITIONS
    )


# --------------------------------------------------------------------------
# Materialization
# --------------------------------------------------------------------------


def materialize(transcript: Transcript, cfg: ExperimentConfig, template_name: str,
                answer_continuation: dict[str, str] | None = None) -> str:
    """Render a transcript as the plain-text string a base model would see.

    The result ends exactly at the answer cue: the next token is the answer, and
    the continuation that follows the cue (``" A"`` versus ``"A"``) is a
    tokenizer question that the model-compatibility stage settles.

    A transcript containing an assistant turn needs that continuation to render
    the already-given answer, so it must either be resolved in the config or
    supplied explicitly.
    """
    _require_v4(cfg)
    template = _template(cfg, template_name)

    if template.materialization != "plain_scaffold":
        raise MaterializationDeferred(
            f"template {template_name!r} is materialized by the tokenizer's own chat "
            f"template at the model-compatibility stage. Use the structured "
            f"transcript, which is model-independent.")

    assert template.template_text is not None and template.role_labels is not None
    assert template.turn_separator is not None and template.answer_cue is not None

    continuation = answer_continuation or {
        k: v for k, v in template.answer_continuation.items() if v is not None}

    rendered: list[str] = []
    for turn in transcript.turns:
        label = template.role_labels[turn.role]
        if turn.role == "assistant":
            if turn.content not in continuation:
                raise AnswerContinuationUnresolved(
                    f"template {template_name!r} has no resolved answer continuation for "
                    f"{turn.content!r}. Whether the label is written as ' A' or 'A' is a "
                    f"tokenizer question, settled at the model-compatibility stage; pass "
                    f"one explicitly to render before then.")
            rendered.append(f"{label}: {template.answer_cue}{continuation[turn.content]}")
        else:
            rendered.append(f"{label}: {turn.content}")

    return template.template_text.format(
        turns=template.turn_separator.join(rendered),
        assistant_label=template.role_labels["assistant"],
        answer_cue=template.answer_cue,
    )


def branches_share_prefix(branches: Sequence[PushbackBranch]) -> bool:
    """Whether every branch has the identical pre-counterargument transcript."""
    prefixes = {b.transcript.prefix(2).transcript_hash for b in branches}
    return len(prefixes) == 1
