"""Building drafting requests, and accepting responses.

A request is a plain object: the rendered prompt, the template it came from,
the keys that identify what it drafts, and the response schema the answer must
satisfy. It is written to a file before anything is sent, so that every request
can be read in full before a model ever sees it.

Responses are validated against the template's schema — a closed object of
non-empty strings — before they are allowed anywhere near a corpus record. A
response that does not fit is rejected, not repaired silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

from ..config import ExperimentConfig
from ..corpus.schemas import SemanticOption
from ..corpus.topics import TopicBrief
from ..hashing import content_hash, sha256_of
from .allocation import GroupAllocation

__all__ = [
    "DraftRequest",
    "RequestError",
    "ResponseRejected",
    "group_request",
    "parse_response",
    "read_response",
    "repair_request",
    "scenario_request",
    "write_request",
]


class RequestError(ValueError):
    """A request could not be built from the configuration and the brief."""


class ResponseRejected(ValueError):
    """A response does not satisfy the template's schema."""


@dataclass(frozen=True, slots=True)
class DraftRequest:
    kind: str                       # scenario | group | repair
    template_name: str
    template_sha256: str
    prompt: str
    response_schema: dict[str, Any]
    decision_id: str
    variant_id: int
    supported_option: SemanticOption | None = None
    attempt: int = 1
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_sha256(self) -> str:
        return sha256_of(self.prompt)

    @property
    def call_id(self) -> str:
        """Content-addressed: the same request is the same id, and a repair of
        the same group is a different id because the attempt differs."""
        return content_hash({
            "kind": self.kind,
            "template_sha256": self.template_sha256,
            "prompt_sha256": self.prompt_sha256,
            "decision_id": self.decision_id,
            "variant_id": self.variant_id,
            "supported_option": self.supported_option,
            "attempt": self.attempt,
        })

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "kind": self.kind,
            "template_name": self.template_name,
            "template_sha256": self.template_sha256,
            "prompt_sha256": self.prompt_sha256,
            "decision_id": self.decision_id,
            "variant_id": self.variant_id,
            "supported_option": self.supported_option,
            "attempt": self.attempt,
            "context": self.context,
            "response_schema": self.response_schema,
            "prompt": self.prompt,
        }


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------


def _template(cfg: ExperimentConfig, name: str) -> tuple[str, dict[str, Any]]:
    try:
        spec = cfg.parsed.prompts.drafting[name]
    except KeyError as exc:                                   # pragma: no cover
        raise RequestError(f"unknown drafting template {name!r}") from exc
    text = cfg.resolve_path(spec["path"]).read_text(encoding="utf-8")
    if sha256_of(text) != spec["template_sha256"]:
        raise RequestError(f"{spec['path']} does not match the recorded hash; "
                           f"the configuration must be updated deliberately")
    return text, spec


def _render(cfg: ExperimentConfig, name: str, values: dict[str, Any]) -> tuple[str, dict, str]:
    text, spec = _template(cfg, name)
    missing = set(spec["placeholders"]) - set(values)
    if missing:
        raise RequestError(f"template {name}: no value for {sorted(missing)}")
    extra = set(values) - set(spec["placeholders"])
    if extra:
        raise RequestError(f"template {name}: {sorted(extra)} are not placeholders")
    for key, value in values.items():
        if value is None or str(value).strip() == "":
            raise RequestError(f"template {name}: {key} is empty")
    prompt = Template(text).substitute(values)
    return prompt, spec["response_schema"], spec["template_sha256"]


def _variant(topic: TopicBrief, variant_id: int):
    key = f"v{variant_id}"
    try:
        return topic.variants[key]
    except KeyError as exc:
        raise RequestError(f"{topic.decision_id} has no variant {key!r}") from exc


def _single_fact(topic: TopicBrief, variant_id: int, option: SemanticOption) -> str:
    facts = getattr(_variant(topic, variant_id).scenario_facts, option)
    if len(facts) != 1:
        raise RequestError(
            f"{topic.decision_id} v{variant_id} {option}: the pilot expects exactly one fact")
    return facts[0]


# --------------------------------------------------------------------------
# The three request kinds
# --------------------------------------------------------------------------


def scenario_request(topic: TopicBrief, variant_id: int, cfg: ExperimentConfig) -> DraftRequest:
    """One scenario. Everything the generator sees comes from the curated brief."""
    if topic.status != "curated":
        raise RequestError(f"{topic.decision_id} is {topic.status}, not curated")
    words = cfg.raw["corpus"]["scenario_words"]
    prompt, schema, digest = _render(cfg, "scenario_draft_v1", {
        "decision_framing": topic.decision_framing,
        "variant_context": _variant(topic, variant_id).context,
        "option_text_opt_1": topic.options["opt_1"],
        "option_text_opt_2": topic.options["opt_2"],
        "fact_opt_1": _single_fact(topic, variant_id, "opt_1"),
        "fact_opt_2": _single_fact(topic, variant_id, "opt_2"),
        "min_words": words["min"],
        "max_words": words["max"],
    })
    return DraftRequest(
        kind="scenario", template_name="scenario_draft_v1", template_sha256=digest,
        prompt=prompt, response_schema=schema,
        decision_id=topic.decision_id, variant_id=variant_id,
        context={"domain": topic.domain},
    )


def group_request(topic: TopicBrief, variant_id: int, scenario_text: str,
                  allocation: GroupAllocation, cfg: ExperimentConfig,
                  *, attempt: int = 1) -> DraftRequest:
    """The four conditions of one group, drafted together.

    ``attempt`` exists because a schema-rejected or truncated response leaves
    nothing to repair: the next call is a fresh draft, and it must have its own
    call id rather than colliding with the draft that failed."""
    budget = cfg.raw["corpus"]["repair"]["max_calls_per_group"]
    if not 1 <= attempt <= budget:
        raise RequestError(f"attempt {attempt} is outside the budget of {budget} calls "
                           f"per group")
    option = allocation.supported_option
    if allocation.decision_id != topic.decision_id or allocation.variant_id != variant_id:
        raise RequestError("the allocation does not belong to this scenario")
    registry = cfg.raw["markers"]["realization"]["registry"]
    prompt, schema, digest = _render(cfg, "group_draft_v1", {
        "scenario_text": scenario_text,
        "option_text_opt_1": topic.options["opt_1"],
        "option_text_opt_2": topic.options["opt_2"],
        "supported_option_text": topic.options[option],
        "supported_option_fact": _single_fact(topic, variant_id, option),
        "counterargument_opening": cfg.raw["corpus"]["counterargument_opening"],
        "marker_string": allocation.marker_string,
        "realization_description": registry[allocation.marker_realization_id]["description"],
        "sentence_count": cfg.raw["corpus"]["body_sentences"],
    })
    return DraftRequest(
        kind="group", template_name="group_draft_v1", template_sha256=digest,
        prompt=prompt, response_schema=schema,
        decision_id=topic.decision_id, variant_id=variant_id, supported_option=option,
        attempt=attempt,
        context={"domain": topic.domain,
                 "marker_family": allocation.marker_family,
                 "marker_string": allocation.marker_string,
                 "marker_realization_id": allocation.marker_realization_id},
    )


def repair_request(topic: TopicBrief, variant_id: int, scenario_text: str,
                   allocation: GroupAllocation, bodies: dict[str, str],
                   findings: list[str], attempt: int, cfg: ExperimentConfig,
                   *, diagnostics: str = "", history: str = "") -> DraftRequest:
    """A repair of one group. The findings are the validator's own codes and
    messages, never advice composed by hand for this item.

    What is fixed is the **versioned template**, whose SHA-256 is pinned in the
    configuration and checked at load. The **rendered request** necessarily
    varies: it carries this scenario, these bodies, these findings, these
    measurements, the attempt number and the history of earlier attempts. Every
    rendered prompt is hashed individually (``prompt_sha256``), and that hash is
    recorded in the log and in the raw traffic, so each call is identified by
    exactly what was sent.

    ``diagnostics`` are the measurements behind those findings — per-condition
    body sentence and word counts against the configured rule — and ``history``
    says what earlier attempts did, in particular whether the last one changed
    anything at all. Both are derived mechanically from the validator and the
    stored attempts, so the template stays fixed; what varies is the measured
    state, which is exactly what a second repair needs in order not to repeat
    the first. A repair after one that changed nothing must not be the same
    request again.
    """
    budget = cfg.raw["corpus"]["repair"]
    if not 2 <= attempt <= budget["max_calls_per_group"]:
        raise RequestError(
            f"attempt {attempt} is outside the budget of {budget['max_calls_per_group']} "
            f"calls per group (one draft plus {budget['max_repair_calls']} repairs)")
    if not findings:
        raise RequestError("a repair needs the findings it is repairing")
    option = allocation.supported_option
    prompt, schema, digest = _render(cfg, "repair_v1", {
        "scenario_text": scenario_text,
        "supported_option_text": topic.options[option],
        "supported_option_fact": _single_fact(topic, variant_id, option),
        "rs": bodies["RS"], "rp": bodies["RP"], "ns": bodies["NS"], "np": bodies["NP"],
        "findings": "\n".join(f"- {f}" for f in findings),
        "marker_string": allocation.marker_string,
        "sentence_count": cfg.raw["corpus"]["body_sentences"],
        "repair_attempt": attempt,
        "attempt_history": history or "This is the first repair of this group.",
        "diagnostics": diagnostics or "  (no measurements were recorded)",
    })
    return DraftRequest(
        kind="repair", template_name="repair_v1", template_sha256=digest,
        prompt=prompt, response_schema=schema,
        decision_id=topic.decision_id, variant_id=variant_id, supported_option=option,
        attempt=attempt,
        context={"domain": topic.domain,
                 "marker_family": allocation.marker_family,
                 "marker_string": allocation.marker_string,
                 "marker_realization_id": allocation.marker_realization_id,
                 "findings": list(findings),
                 "diagnostics": diagnostics,
                 "history": history},
    )


# --------------------------------------------------------------------------
# Files in, files out
# --------------------------------------------------------------------------


def write_request(request: DraftRequest, directory: str | Path) -> Path:
    """Write the request where a human can read it before it is sent."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{request.call_id[:16]}_{request.kind}.json"
    path.write_text(json.dumps(request.as_dict(), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def read_response(path: str | Path) -> dict[str, Any]:
    """Read a response file exactly as it was stored."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_response(request: DraftRequest, payload: Any) -> dict[str, str]:
    """Validate a response against the request's schema.

    Accepts either a parsed object or the raw JSON text a model returned.
    Anything that does not fit the closed schema is rejected; nothing is
    coerced, stripped of extra keys or guessed at.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ResponseRejected(f"response is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResponseRejected(f"response must be a JSON object; got {type(payload).__name__}")

    schema = request.response_schema
    required, properties = set(schema["required"]), schema["properties"]
    missing = sorted(required - set(payload))
    if missing:
        raise ResponseRejected(f"response is missing {missing}")
    extra = sorted(set(payload) - set(properties))
    if extra:
        raise ResponseRejected(f"response has unexpected field(s) {extra}")
    for key, value in payload.items():
        if properties[key]["type"] != "string" or not isinstance(value, str):
            raise ResponseRejected(f"{key} must be a string")
        if not value.strip():
            raise ResponseRejected(f"{key} is empty")
    return {k: v.strip() for k, v in payload.items()}
