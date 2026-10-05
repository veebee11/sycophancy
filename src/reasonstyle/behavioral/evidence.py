"""Torch-free pieces of a behavioural run: determinism environment, initial
records, and the completeness test analysis must apply before reading a run.

A run directory is complete only when it carries a ``COMPLETE`` marker and a
``RUN_METADATA.json`` with ``status: complete`` whose counts and file hashes
agree with the files on disk. A complete run may still record exclusions
(exact initial ties under ``exact_tie: refuse``) and secondary post ties (exact
post-counterargument ties, kept for movement, undefined for flip); either is a
finished run, not a failed one. Anything else — no marker, an ``.incomplete.`` directory, a count or
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
COMPLETE_WITH_SECONDARY_TIES = "complete_with_recorded_secondary_ties"
COMPLETE_WITH_EXCLUSIONS_AND_SECONDARY_TIES = (
    "complete_with_recorded_exclusions_and_secondary_ties")
INCOMPLETE = "incomplete_or_failed"
POST_TIE_REASON = "exact_post_logit_tie"


def completion_label(n_initial_ties: int, n_post_ties: int) -> str:
    """How a finished run is labelled, by which kinds of recorded tie it holds."""
    if n_initial_ties and n_post_ties:
        return COMPLETE_WITH_EXCLUSIONS_AND_SECONDARY_TIES
    if n_initial_ties:
        return COMPLETE_WITH_EXCLUSIONS
    if n_post_ties:
        return COMPLETE_WITH_SECONDARY_TIES
    return COMPLETE


def post_tie_records(selected: list[dict[str, Any]], score_rows: list[dict[str, Any]]
                     ) -> list[dict[str, Any]]:
    """One record per behavioural-score row whose post reading is an exact tie.

    The score row itself stays in ``behavioral_scores.jsonl`` for the primary
    movement analysis; this file only records which rows have no flip value."""
    branch = {row["branch_id"]: row for row in selected}
    out = []
    for row in score_rows:
        if not row.get("post_exact_tie"):
            continue
        source = branch[row["branch_id"]]
        out.append({
            **{key: row[key] for key in (
                "run_id", "model_variant", "branch_id", "initial_id", "stimulus_id",
                "decision_id", "domain", "scenario_id", "variant_id", "supported_option",
                "condition", "marker_id", "opening_id", "order_id", "counter_target_label",
                "initial_label", "initial_logit_a", "initial_logit_b", "after_logit_a",
                "after_logit_b", "m_before", "m_after", "movement_toward_counter")},
            "label_to_option": dict(source["label_to_option"]),
            "final_label": None, "flip": None,
            "exclusion_reason": POST_TIE_REASON,
            "excluded_from": ["flip_rate"],
            "retained_in": ["movement_toward_counter"],
        })
    return out


def movement_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every behavioural-score row: post ties are part of the primary outcome."""
    return list(score_rows)


def flip_rate_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows with a defined flip: exact post ties are omitted, and only here."""
    return [row for row in score_rows if not row.get("post_exact_tie")]


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


def _post_tie_problems(scores: list[dict[str, Any]], post_ties: list[dict[str, Any]]
                       ) -> list[str]:
    problems = []
    tied_rows = {x["branch_id"]: x for x in scores if x.get("post_exact_tie")}
    recorded = {x["branch_id"]: x for x in post_ties}
    if set(tied_rows) != set(recorded) or len(recorded) != len(post_ties):
        problems.append("post_ties.jsonl does not list exactly the post-tie score rows")
    for branch_id, row in tied_rows.items():
        movement = row.get("movement_toward_counter")
        if not (row.get("after_logit_a") == row.get("after_logit_b")
                and row.get("m_after") == 0 and row.get("final_label") is None
                and row.get("flip") is None and isinstance(movement, (int, float))
                and movement == movement and movement == row["m_after"] - row["m_before"]):
            problems.append(f"{branch_id}: post-tie row is not equal logits, m_after 0, null "
                            f"final label and flip, with a valid movement")
        if branch_id in recorded and recorded[branch_id].get("exclusion_reason") != POST_TIE_REASON:
            problems.append(f"{branch_id}: post tie recorded with the wrong reason")
    for row in scores:
        if not row.get("post_exact_tie") and (row.get("final_label") not in ("A", "B")
                                              or not isinstance(row.get("flip"), bool)):
            problems.append(f"{row.get('branch_id')}: untied row lacks a final label or flip")
    return problems


def run_status(directory: str | Path) -> tuple[str, list[str]]:
    """``(status, problems)`` for a run directory: ``complete``,
    ``complete_with_recorded_exclusions`` (initial ties),
    ``complete_with_recorded_secondary_ties`` (post ties),
    ``complete_with_recorded_exclusions_and_secondary_ties`` (both), or
    ``incomplete_or_failed``."""
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
    has_post_file = "post_ties.jsonl" in (meta.get("files") or {})
    post_ties = _lines(directory / "post_ties.jsonl") if has_post_file else []
    ties = [x for x in exclusions if x.get("exclusion_reason") == TIE_REASON]
    want = expected_post_counterargument(len(initial), len(ties))
    checks = [
        (len(initial), counts.get("initial"), "initial"),
        (len(ties), counts.get("excluded_exact_initial_ties"), "excluded ties"),
        (len(scores), counts.get("post_counterargument"), "post-counterargument"),
        (want, counts.get("expected_post_counterargument"), "expected post-counterargument"),
        (len(scores), want, "post-counterargument against 18 x (initials - ties)"),
        (sum(1 for x in initial if x.get("excluded")), len(ties), "excluded initial records"),
    ]
    if has_post_file:
        checks.append((len(post_ties), counts.get("post_exact_ties"), "post exact ties"))
    for got, recorded, label in checks:
        if got != recorded:
            problems.append(f"{label}: {got} != {recorded}")
    if {x["initial_id"] for x in scores} & {x["initial_id"] for x in ties}:
        problems.append("a tied initial has post-counterargument scores")
    if not has_post_file and any(x.get("post_exact_tie") for x in scores):
        problems.append("post-tie rows exist but no post_ties.jsonl is recorded")
    problems += _post_tie_problems(scores, post_ties)
    label = completion_label(len(ties), len(post_ties))
    if meta.get("completion") is not None and meta["completion"] != label:
        problems.append(f"metadata completion {meta['completion']!r} != {label!r}")
    if problems:
        return INCOMPLETE, problems
    return label, []
