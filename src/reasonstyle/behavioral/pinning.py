"""Create a run-ready behavioural config from two compatibility reports."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ..hashing import file_sha256
from .plan import BehavioralPlanError, BehavioralSpec


def pin_compatibility(spec: BehavioralSpec,
                      reports: dict[str, tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    if spec.raw["status"] != "setup_authorized_models_unfrozen":
        raise BehavioralPlanError("only the unverified setup config can be pinned")
    if set(reports) != {"base", "instruct"}:
        raise BehavioralPlanError("base and instruct compatibility reports are required")
    pinned = deepcopy(spec.raw)
    pinned["status"] = "ready_for_smoke"
    pinned["models"]["selection_status"] = "pinned_after_compatibility"
    for variant in ("base", "instruct"):
        path, report = reports[variant]
        expected = spec.raw["models"][variant]
        checks = {
            "compatibility_status": report.get("compatibility_status") == "passed",
            "variant": report.get("variant") == variant,
            "repo_id": report.get("repo_id") == expected["repo_id"],
            "behavioral_config_sha256": report.get("behavioral_config_sha256") == spec.content_hash,
            "stimuli_sha256": report.get("stimuli_sha256") == spec.raw["dataset"]["stimuli_sha256"],
            "network_access": report.get("network_access") is False,
            "model_forward_pass": report.get("model_forward_pass") is False,
        }
        failed = [name for name, okay in checks.items() if not okay]
        if failed:
            raise BehavioralPlanError(
                f"{variant} compatibility report failed checks: {', '.join(failed)}")
        revision = report.get("revision")
        token_ids = report.get("answer_token_ids")
        continuations = report.get("answer_continuation")
        if not isinstance(revision, str) or not revision:
            raise BehavioralPlanError(f"{variant} report has no resolved revision")
        if (not isinstance(token_ids, dict) or set(token_ids) != {"A", "B"} or
                not all(isinstance(value, int) for value in token_ids.values()) or
                token_ids["A"] == token_ids["B"]):
            raise BehavioralPlanError(f"{variant} report has invalid A/B token IDs")
        if (not isinstance(continuations, dict) or set(continuations) != {"A", "B"} or
                not all(isinstance(value, str) and value
                        for value in continuations.values())):
            raise BehavioralPlanError(f"{variant} report has invalid A/B continuations")
        model = pinned["models"][variant]
        model.update({
            "revision": revision,
            "compatibility_status": "passed",
            "answer_continuation": continuations,
            "answer_token_ids": token_ids,
            "compatibility_report": {
                "path": str(path),
                "sha256": file_sha256(path),
            },
        })
        if variant == "instruct":
            template_hash = report.get("chat_template_sha256")
            if not isinstance(template_hash, str) or not template_hash:
                raise BehavioralPlanError("instruct report has no chat-template hash")
            model["chat_template_sha256"] = template_hash
    return pinned
