"""Rebuilding a request from its declared inputs, and comparing.

This is what turns "the generator only ever saw the curated brief" from a claim
into a check. For any logged call, the request is rebuilt from

    the curated topic brief  +  the marker allocation  +  the hashed template

and the rebuilt prompt hash must equal the one that was logged. Because no
source text is stored or supplied anywhere in that chain, a matching hash shows
that no source passage could have reached the generator.

It also catches the duller failures: a template edited after the fact, a brief
revised after drafting, an allocation regenerated with a different seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import ExperimentConfig
from ..corpus.topics import TopicBank
from ..hashing import content_hash
from .allocation import MarkerAllocation
from .requests import DraftRequest, group_request, scenario_request

__all__ = ["ProvenanceMismatch", "ProvenanceResult", "check_request_provenance",
           "rebuild_request"]


class ProvenanceMismatch(ValueError):
    """A logged call does not match the request its inputs produce."""


@dataclass(frozen=True, slots=True)
class ProvenanceResult:
    call_id: str
    kind: str
    matched: bool
    problems: tuple[str, ...]


def _topic(bank: TopicBank, decision_id: str):
    for topic in bank.topics:
        if topic.decision_id == decision_id:
            return topic
    raise ProvenanceMismatch(f"no brief for {decision_id!r} in this topic bank")


def rebuild_request(entry: dict[str, Any], bank: TopicBank, alloc: MarkerAllocation,
                    cfg: ExperimentConfig, *, scenario_text: str | None = None) -> DraftRequest:
    """Rebuild a scenario or group request from the recorded inputs."""
    topic = _topic(bank, entry["decision_id"])
    if entry["kind"] == "scenario":
        return scenario_request(topic, entry["variant_id"], cfg)
    if entry["kind"] == "group":
        if scenario_text is None:
            raise ProvenanceMismatch(
                "rebuilding a group request needs the scenario text it was given")
        group = alloc.for_group(f"{entry['decision_id']}_v{entry['variant_id']}",
                                entry["supported_option"])
        return group_request(topic, entry["variant_id"], scenario_text, group, cfg)
    raise ProvenanceMismatch(
        f"{entry['kind']!r} requests carry their own inputs and are checked against "
        f"the stored request, not rebuilt")


def check_request_provenance(entry: dict[str, Any], bank: TopicBank,
                             alloc: MarkerAllocation, cfg: ExperimentConfig, *,
                             scenario_text: str | None = None) -> ProvenanceResult:
    """Compare one logged call with the request its inputs produce.

    Returns a result rather than raising, so a whole log can be checked and
    reported at once.
    """
    problems: list[str] = []
    if entry["config_content_hash"] != cfg.content_hash:
        problems.append(f"config hash {entry['config_content_hash'][:12]} != "
                        f"{cfg.content_hash[:12]}")
    bank_hash = content_hash(bank.model_dump(mode="json"))
    if entry["topic_bank_content_hash"] != bank_hash:
        problems.append(f"topic bank hash {entry['topic_bank_content_hash'][:12]} != "
                        f"{bank_hash[:12]}")
    if entry["kind"] in ("group", "repair") and entry["allocation_content_hash"] not in (
            None, alloc.content_hash):
        problems.append(f"allocation hash {entry['allocation_content_hash'][:12]} != "
                        f"{alloc.content_hash[:12]}")

    if entry["kind"] in ("scenario", "group") and not problems:
        rebuilt = rebuild_request(entry, bank, alloc, cfg, scenario_text=scenario_text)
        if rebuilt.template_sha256 != entry["template_sha256"]:
            problems.append("the template has changed since the call was made")
        if rebuilt.prompt_sha256 != entry["prompt_sha256"]:
            problems.append("the rebuilt prompt differs from the prompt that was sent")
        if rebuilt.call_id != entry["call_id"] and rebuilt.prompt_sha256 == entry["prompt_sha256"]:
            problems.append("the call identity differs although the prompt matches")

    return ProvenanceResult(call_id=entry["call_id"], kind=entry["kind"],
                            matched=not problems, problems=tuple(problems))
