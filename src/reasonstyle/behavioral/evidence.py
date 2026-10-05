"""Torch-free pieces of a behavioural run: determinism environment, initial
records, and the completeness test analysis must apply before reading a run.

A run directory is complete only when it carries a ``COMPLETE`` marker and a
``RUN_METADATA.json`` with ``status: complete`` whose counts and file hashes
agree with the files on disk. A complete run may still record exclusions
(exact initial ties under ``exact_tie: refuse``); that is a finished run, not a
failed one. Anything else — no marker, an ``.incomplete.`` directory, a count or
hash mismatch — is not a run to analyse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from ..hashing import file_sha256
from .plan import BehavioralPlanError
from .runtime import TIE_REASON, RuntimeSelection, expected_post_counterargument

CUBLAS_VARIABLE = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_REQUIRED = ":4096:8"

COMPLETE = "complete"
COMPLETE_WITH_EXCLUSIONS = "complete_with_recorded_exclusions"
INCOMPLETE = "incomplete_or_failed"


def configure_cublas_workspace(deterministic: bool, environ: MutableMapping[str, str],
                               loaded_modules: Mapping[str, Any]) -> dict[str, Any]:
    """Set ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` before Torch is imported when
    deterministic algorithms are enabled. A different pre-existing value is a
    conflict and refuses; so does Torch already being imported, because the
    setting must precede CUDA initialisation. Returns what was found and done."""
    existing = environ.get(CUBLAS_VARIABLE)
    record = {"variable": CUBLAS_VARIABLE, "required": CUBLAS_REQUIRED if deterministic else None,
              "preexisting_value": existing, "set_by_runner": False,
              "deterministic_algorithms": bool(deterministic)}
    if not deterministic:
        record["value"] = existing
        return record
    if "torch" in loaded_modules:
        raise BehavioralPlanError(
            f"torch was imported before {CUBLAS_VARIABLE} could be set; refusing")
    if existing is not None and existing != CUBLAS_REQUIRED:
        raise BehavioralPlanError(
            f"{CUBLAS_VARIABLE} is already {existing!r}, which conflicts with the required "
            f"{CUBLAS_REQUIRED!r}; unset it or set it to {CUBLAS_REQUIRED!r}")
    if existing is None:
        environ[CUBLAS_VARIABLE] = CUBLAS_REQUIRED
        record["set_by_runner"] = True
    record["value"] = environ[CUBLAS_VARIABLE]
    return record


def initial_records(initials: list[dict[str, Any]], raw_initial: dict[str, dict[str, Any]],
                    selection: RuntimeSelection, *, run_id: str, variant: str
                    ) -> list[dict[str, Any]]:
    """One record per scored initial prompt, tied or not. A tie keeps its
    logits and gets ``initial_label`` and ``initial_option`` of ``None``."""
    excluded = {row["initial_id"]: row for row in selection.exclusions}
    rows = []
    for row in initials:
        measured = raw_initial[row["initial_id"]]
        label = selection.labels.get(row["initial_id"])
        tie = row["initial_id"] in excluded
        if tie == (label is not None):
            raise BehavioralPlanError(f"{row['initial_id']}: neither labelled nor excluded")
        rows.append({
            "run_id": run_id, "model_variant": variant,
            **{key: row[key] for key in ("initial_id", "decision_id", "domain",
                                          "scenario_id", "order_id")},
            "initial_label": label,
            "initial_option": row["label_to_option"][label] if label else None,
            "excluded": tie,
            "exclusion_reason": TIE_REASON if tie else None,
            "logit_a": measured["A"], "logit_b": measured["B"],
            "input_tokens": measured["input_tokens"],
            "materialized_prompt_sha256": measured["materialized_prompt_sha256"],
        })
    return rows


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run_status(directory: str | Path) -> tuple[str, list[str]]:
    """``(status, problems)`` for a run directory: ``complete``,
    ``complete_with_recorded_exclusions``, or ``incomplete_or_failed``."""
    directory = Path(directory)
    problems: list[str] = []
    if ".incomplete." in directory.name:
        problems.append("an .incomplete. directory is never a finished run")
    meta_path = directory / "RUN_METADATA.json"
    if not (directory / "COMPLETE").is_file():
        problems.append("no COMPLETE marker")
    if not meta_path.is_file():
        return INCOMPLETE, problems + ["no RUN_METADATA.json"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "complete":
        problems.append(f"metadata status is {meta.get('status')!r}")
    for name, digest in (meta.get("files") or {}).items():
        path = directory / name
        if not path.is_file() or file_sha256(path) != digest:
            problems.append(f"{name} is missing or does not match its recorded hash")
    if problems:
        return INCOMPLETE, problems
    counts = meta.get("counts") or {}
    initial = _lines(directory / "initial_scores.jsonl")
    scores = _lines(directory / "behavioral_scores.jsonl")
    exclusions = _lines(directory / "initial_exclusions.jsonl")
    ties = [x for x in exclusions if x.get("exclusion_reason") == TIE_REASON]
    want = expected_post_counterargument(len(initial), len(ties))
    checks = (
        (len(initial), counts.get("initial"), "initial"),
        (len(ties), counts.get("excluded_exact_initial_ties"), "excluded ties"),
        (len(scores), counts.get("post_counterargument"), "post-counterargument"),
        (want, counts.get("expected_post_counterargument"), "expected post-counterargument"),
        (len(scores), want, "post-counterargument against 18 x (initials - ties)"),
        (sum(1 for x in initial if x.get("excluded")), len(ties), "excluded initial records"),
    )
    for got, recorded, label in checks:
        if got != recorded:
            problems.append(f"{label}: {got} != {recorded}")
    if {x["initial_id"] for x in scores} & {x["initial_id"] for x in ties}:
        problems.append("a tied initial has post-counterargument scores")
    if problems:
        return INCOMPLETE, problems
    return (COMPLETE_WITH_EXCLUSIONS if ties else COMPLETE), []
