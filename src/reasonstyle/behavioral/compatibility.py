"""Prompt materialization and exact A/B next-token compatibility checks.

This module does not import a model library. A caller supplies the tokenizer
belonging to the exact cached checkpoint. Compatibility is deliberately
strict: the prompt tokenization must remain an exact prefix and each answer
continuation must add one, and only one, distinct token.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


class CompatibilityError(ValueError):
    """A prompt template cannot support the specified exact-logit design."""


class TokenizerLike(Protocol):
    chat_template: str | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def apply_chat_template(self, conversation: list[dict[str, str]], *,
                            tokenize: bool, add_generation_prompt: bool) -> str: ...


@dataclass(frozen=True)
class AnswerTokenResolution:
    leading_whitespace: bool
    answer_continuation: dict[str, str]
    answer_token_ids: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_transcript(transcript: list[dict[str, str]]) -> None:
    if not transcript or transcript[0].get("role") != "user":
        raise CompatibilityError("a behavioural transcript must begin with a user turn")
    expected = "user"
    for turn in transcript:
        if turn.get("role") != expected or not isinstance(turn.get("content"), str):
            raise CompatibilityError("transcript roles must alternate user/assistant")
        expected = "assistant" if expected == "user" else "user"
    if transcript[-1]["role"] != "user":
        raise CompatibilityError("a scored transcript must end immediately after a user turn")


def materialize_base(transcript: list[dict[str, str]], *, answer_cue: str,
                     template: dict[str, str]) -> str:
    """Render a transcript with the frozen plain-dialogue base scaffold."""
    _validate_transcript(transcript)
    separator = template["turn_separator"]
    labels = {"user": template["user_label"], "assistant": template["assistant_label"]}
    turns = [f'{labels[t["role"]]}: {t["content"]}' for t in transcript]
    turns.append(f'{labels["assistant"]}: {answer_cue}')
    return template["header"] + separator + separator.join(turns)


def materialize_instruct(transcript: list[dict[str, str]], *, answer_cue: str,
                         tokenizer: TokenizerLike) -> str:
    """Use the selected instruct checkpoint's own chat template."""
    _validate_transcript(transcript)
    if not getattr(tokenizer, "chat_template", None):
        raise CompatibilityError("the instruct tokenizer has no chat template")
    rendered = tokenizer.apply_chat_template(
        transcript, tokenize=False, add_generation_prompt=True)
    if not isinstance(rendered, str) or not rendered:
        raise CompatibilityError("the instruct chat template did not render text")
    return rendered + answer_cue


def _encoded(tokenizer: TokenizerLike, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return [int(token_id) for token_id in encoded]


def resolve_answer_tokens(tokenizer: TokenizerLike, prompt: str) -> AnswerTokenResolution:
    """Resolve a shared one-token continuation for labels A and B.

    Both labels must work with the same whitespace convention. Testing the
    whole ``prompt + continuation`` protects against a boundary merge that a
    standalone encoding of ``" A"`` would miss.
    """
    prompt_ids = _encoded(tokenizer, prompt)
    if not prompt_ids:
        raise CompatibilityError("the materialized prompt tokenizes to nothing")
    failures: list[str] = []
    for leading_whitespace in (True, False):
        prefix = " " if leading_whitespace else ""
        ids: dict[str, int] = {}
        valid = True
        for label in ("A", "B"):
            continuation = prefix + label
            combined = _encoded(tokenizer, prompt + continuation)
            if combined[:-1] != prompt_ids or len(combined) != len(prompt_ids) + 1:
                failures.append(
                    f"{continuation!r} did not add exactly one token without changing the prefix")
                valid = False
                break
            ids[label] = combined[-1]
        if valid and ids["A"] != ids["B"]:
            return AnswerTokenResolution(
                leading_whitespace=leading_whitespace,
                answer_continuation={label: prefix + label for label in ("A", "B")},
                answer_token_ids=ids,
            )
        if valid:
            failures.append("A and B resolved to the same token ID")
    raise CompatibilityError("; ".join(failures))
