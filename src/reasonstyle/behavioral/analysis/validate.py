"""Refuse-first validation of every behavioural run the analysis reads.

Nothing is analysed unless every check passes: the analysis spec; the pinned
behavioural config (file and content hash, revisions, tokens); the v3 stimuli
and manifest; each run directory (``evidence.run_status`` plus the expected
status, run ID, revision, counts, metadata hash and score-file hash); for a
robustness run also its pinned variant config, prompt digest and tokens; and
every row against the deterministic plan and the real scoring function.
Any mismatch raises ``AnalysisError``. Run directories are only read.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ...hashing import content_hash, file_sha256
from ..evidence import POST_TIE_REASON, run_status
from ..plan import BehavioralPlanError, build_plan, load_spec
from ..runtime import TIE_REASON, expected_post_counterargument, is_exact_tie
from ..score import ScoreError, score_movement

CONDITIONS = ("RS", "RP", "NS", "NP")
FAMILIES = ("conclusion_result", "inference_basis")
PLAIN_MARKER = "none"          # the v3 stimuli's marker_id and marker_family for RP/NP
REFERENCE_MODELS = ("base", "instruct")

PLAN_FIELDS = (
    "initial_id", "stimulus_id", "decision_id", "domain", "scenario_id", "variant_id",
    "supported_option", "condition", "reason", "style", "marker_id", "marker_family",
    "marker_string", "opening_id", "order_id", "required_initial_option",
    "counter_target_label", "paired_control_stimulus_id")
SCORE_FIELDS = (
    "initial_label", "counter_target_label", "initial_logit_a", "initial_logit_b",
    "after_logit_a", "after_logit_b", "m_before", "m_after", "movement_toward_counter",
    "final_label", "flip", "initial_near_tie", "post_exact_tie")
RUN_FILES = ("initial_scores.jsonl", "behavioral_scores.jsonl", "initial_exclusions.jsonl",
             "post_ties.jsonl")
EXPECTED_CONTRASTS = {
    "NS_minus_NP": {"RS": 0, "RP": 0, "NS": 1, "NP": -1},
    "RS_minus_RP": {"RS": 1, "RP": -1, "NS": 0, "NP": 0},
    "RS_minus_NS": {"RS": 1, "RP": 0, "NS": -1, "NP": 0},
    "RP_minus_NP": {"RS": 0, "RP": 1, "NS": 0, "NP": -1},
    "interaction": {"RS": 1, "RP": -1, "NS": -1, "NP": 1},
}


class AnalysisError(ValueError):
    """The evidence cannot be analysed under the frozen analysis specification."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


@dataclass(frozen=True)
class AnalysisSpec:
    path: Path
    root: Path
    raw: dict[str, Any]

    @property
    def content_hash(self) -> str:
        return content_hash(self.raw)

    @property
    def file_sha256(self) -> str:
        return file_sha256(self.path)


def load_analysis_spec(path: str | Path, root: str | Path | None = None) -> AnalysisSpec:
    path = Path(path).resolve()
    if root is None:
        _require(path.parent.name == "analysis" and path.parent.parent.name == "configs",
                 f"analysis spec must live under configs/analysis/: {path}")
        root = path.parent.parent.parent
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "analysis spec is not a mapping")
    spec = AnalysisSpec(path=path, root=Path(root).resolve(), raw=raw)
    _check_spec(spec)
    return spec


def _check_spec(spec: AnalysisSpec) -> None:
    raw = spec.raw
    for key in ("analysis_version", "source_git_commit", "inputs", "design", "analyses",
                "inference", "figures", "outputs"):
        _require(key in raw, f"analysis spec lacks {key!r}")
    _require(raw["design"]["contrasts"] == EXPECTED_CONTRASTS,
             "contrast coefficients differ from the frozen definitions")
    boot = raw["inference"]["bootstrap"]
    _require(isinstance(boot["replicates"], int) and boot["replicates"] >= 1,
             "bootstrap replicates must be a positive integer")
    _require(isinstance(boot["seed"], int), "bootstrap seed must be an integer")
    _require(boot["bit_generator"] == "numpy.random.PCG64", "the bit generator must be PCG64")
    _require(0 < boot["confidence_level"] < 1, "confidence level must lie in (0, 1)")
    runs = raw["inputs"]["runs"]
    _require(set(runs) == set(REFERENCE_MODELS), "inputs.runs must name base and instruct")
    for name, run in {**runs, **(raw["inputs"].get("robustness_runs") or {})}.items():
        for key in ("path", "run_id", "status", "behavioral_scores_sha256",
                    "run_metadata_sha256", "counts", "repo_id", "revision"):
            _require(run.get(key) not in (None, ""), f"{name}: spec lacks {key}")
        c = run["counts"]
        _require(c["movement_rows"] == expected_post_counterargument(
            c["initial"], c["excluded_exact_initial_ties"]),
            f"{name}: spec movement count is not 18 x (initial - ties)")
        _require(c["flip_defined"] == c["movement_rows"] - c["post_exact_ties"],
                 f"{name}: spec flip count is not movement - post ties")


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _same(recorded: Any, recomputed: Any) -> bool:
    """Exact equality; a float may be stored as an integral JSON number, nothing else may
    change type (``True == 1`` is not accepted)."""
    if isinstance(recomputed, float):
        return _finite(recorded) and float(recorded) == recomputed
    return type(recorded) is type(recomputed) and recorded == recomputed


@dataclass(frozen=True)
class RunEvidence:
    name: str                       # base, instruct, or a robustness variant name
    model: str                      # base or instruct
    directory: Path
    status: str
    metadata: dict[str, Any]
    initials: list[dict[str, Any]]
    scores: list[dict[str, Any]]
    exclusions: list[dict[str, Any]]
    post_ties: list[dict[str, Any]]
    file_hashes: dict[str, str]
    response_labels: dict[str, str] = field(default_factory=lambda: {"A": "A", "B": "B"})
    display_order: tuple[str, ...] = ("A", "B")
    template: str = "base_plain_dialogue_v1"

    @property
    def is_reference(self) -> bool:
        return self.name in REFERENCE_MODELS


@dataclass(frozen=True)
class ValidatedEvidence:
    spec: AnalysisSpec
    behavioral_config_path: Path
    runs: dict[str, RunEvidence]
    stimuli: dict[str, dict[str, Any]]
    markers: dict[str, dict[str, Any]]
    openings: dict[str, str]
    input_hashes: dict[str, str]
    refusals: dict[str, dict[str, Any]]
    checks: list[dict[str, Any]]


def _check_metadata(name: str, meta: dict[str, Any], want: dict[str, Any],
                    spec: AnalysisSpec, extra: dict[str, tuple[Any, Any]]) -> list[dict]:
    inputs = spec.raw["inputs"]
    cfg = inputs["behavioral_config"]
    checks = {
        "run_id": (meta.get("run_id"), want["run_id"]),
        "status": (meta.get("status"), "complete"),
        "completion": (meta.get("completion"), want["status"]),
        "mode": (meta.get("mode"), "full"),
        "repo_id": (meta.get("repo_id"), want["repo_id"]),
        "revision": (meta.get("revision"), want["revision"]),
        "behavioral_version": (meta.get("behavioral_version"), cfg["behavioral_version"]),
        "behavioral_config_file_sha256": (meta.get("behavioral_config_file_sha256"),
                                          cfg["sha256"]),
        "behavioral_config_content_hash": (meta.get("behavioral_config_content_hash"),
                                           cfg["content_hash"]),
        "stimuli_sha256": (meta.get("stimuli_sha256"), inputs["stimuli"]["sha256"]),
        "dtype": (meta.get("dtype"), "bfloat16"),
        "device": (meta.get("device"), "cuda"),
        "deterministic_algorithms": (meta.get("deterministic_algorithms"), True),
        "api_calls": (meta.get("api_calls"), False),
        "free_form_generation": (meta.get("free_form_generation"), False),
        **extra,
    }
    counts = meta.get("counts") or {}
    wc = want["counts"]
    for key, spec_key in (("initial", "initial"),
                          ("excluded_exact_initial_ties", "excluded_exact_initial_ties"),
                          ("post_counterargument", "movement_rows"),
                          ("post_exact_ties", "post_exact_ties"),
                          ("flip_defined", "flip_defined")):
        checks[f"counts.{key}"] = (counts.get(key), wc[spec_key])
    rows = [{"run": name, "check": k, "observed": got, "expected": exp, "passed": got == exp}
            for k, (got, exp) in checks.items()]
    bad = [f"{r['check']}: {r['observed']!r} != {r['expected']!r}" for r in rows
           if not r["passed"]]
    _require(not bad, f"{name}: run metadata disagrees with the analysis spec: " + "; ".join(bad))
    return rows


def _load_run(name: str, model: str, want: dict[str, Any], spec: AnalysisSpec,
              checks: list[dict]) -> RunEvidence:
    directory = (spec.root / want["path"]).resolve()
    _require(directory.is_dir(), f"{name}: run directory {directory} does not exist")
    status, problems = run_status(directory)
    _require(not problems, f"{name}: run_status refused: " + "; ".join(problems))
    _require(status == want["status"],
             f"{name}: run status {status!r} != expected {want['status']!r}")
    checks.append({"run": name, "check": "run_status", "observed": status,
                   "expected": want["status"], "passed": True})
    meta_hash = file_sha256(directory / "RUN_METADATA.json")
    _require(meta_hash == want["run_metadata_sha256"],
             f"{name}: RUN_METADATA.json hashes to {meta_hash}, not {want['run_metadata_sha256']}")
    checks.append({"run": name, "check": "sha256 RUN_METADATA.json", "observed": meta_hash,
                   "expected": want["run_metadata_sha256"], "passed": True})
    meta = json.loads((directory / "RUN_METADATA.json").read_text(encoding="utf-8"))
    _require(set(meta.get("files") or {}) == set(RUN_FILES),
             f"{name}: RUN_METADATA.json does not record exactly the four run files")
    hashes = {}
    for fname in RUN_FILES:
        digest = file_sha256(directory / fname)
        _require(digest == meta["files"][fname], f"{name}: {fname} does not match its record")
        hashes[fname] = digest
        checks.append({"run": name, "check": f"sha256 {fname}", "observed": digest,
                       "expected": meta["files"][fname], "passed": True})
    _require(hashes["behavioral_scores.jsonl"] == want["behavioral_scores_sha256"],
             f"{name}: behavioral_scores.jsonl hashes to {hashes['behavioral_scores.jsonl']}, "
             f"not {want['behavioral_scores_sha256']}")
    hashes["RUN_METADATA.json"] = meta_hash
    hashes["COMPLETE"] = file_sha256(directory / "COMPLETE")
    variant = meta.get("prompt_variant") or {}
    return RunEvidence(
        name=name, model=model, directory=directory, status=status, metadata=meta,
        initials=_lines(directory / "initial_scores.jsonl"),
        scores=_lines(directory / "behavioral_scores.jsonl"),
        exclusions=_lines(directory / "initial_exclusions.jsonl"),
        post_ties=_lines(directory / "post_ties.jsonl"), file_hashes=hashes,
        response_labels=dict(variant.get("response_labels") or {"A": "A", "B": "B"}),
        display_order=tuple(variant.get("display_order") or ("A", "B")),
        template=variant.get("template_name") or "base_plain_dialogue_v1")


def _check_rows_against_plan(run: RunEvidence, plan, tau: float) -> None:
    name, run_id = run.name, run.metadata["run_id"]
    by_initial = {row["initial_id"]: row for row in plan.initials}
    initial_ids = [row["initial_id"] for row in run.initials]
    _require(len(initial_ids) == len(set(initial_ids)) and set(initial_ids) == set(by_initial),
             f"{name}: initial_scores.jsonl does not hold each planned initial exactly once")
    labels: dict[str, str] = {}
    logits: dict[str, tuple[float, float]] = {}
    tied: set[str] = set()
    for row in run.initials:
        iid = row["initial_id"]
        _require(row.get("run_id") == run_id and row.get("model_variant") == run.model,
                 f"{iid}: initial row names another run or model")
        a, b = row.get("logit_a"), row.get("logit_b")
        _require(_finite(a) and _finite(b), f"{iid}: initial logits are not finite numbers")
        logits[iid] = (a, b)
        plan_row = by_initial[iid]
        for key in ("decision_id", "domain", "scenario_id", "order_id"):
            _require(row.get(key) == plan_row[key], f"{iid}: initial {key} differs from plan")
        if is_exact_tie(a, b):
            _require(row.get("excluded") is True and row.get("exclusion_reason") == TIE_REASON
                     and row.get("initial_label") is None and row.get("initial_option") is None,
                     f"{iid}: exact initial tie is not recorded as an exclusion")
            tied.add(iid)
        else:
            label = "A" if a > b else "B"
            _require(row.get("excluded") is False and row.get("initial_label") == label
                     and row.get("initial_option") == plan_row["label_to_option"][label],
                     f"{iid}: initial label is not the A/B argmax")
            labels[iid] = label
    excluded = {row["initial_id"] for row in run.exclusions}
    _require(excluded == tied and len(run.exclusions) == len(tied),
             f"{name}: initial_exclusions.jsonl does not list exactly the tied initials")
    for row in run.exclusions:
        _require(row.get("exclusion_reason") == TIE_REASON and row.get("branches_selected") == 0
                 and row.get("initial_choice") is None,
                 f"{row['initial_id']}: exclusion record is not an exact-tie refusal")
    candidates = {row["branch_id"]: row for row in plan.candidates}
    want_by_initial: dict[str, set[str]] = {}
    for bid, cand in candidates.items():
        if labels.get(cand["initial_id"]) == cand["required_initial_label"]:
            want_by_initial.setdefault(cand["initial_id"], set()).add(bid)
    seen: dict[str, set[str]] = {}
    m_before_of: dict[str, float] = {}
    for row in run.scores:
        bid = row.get("branch_id")
        _require(bid in candidates, f"{name}: score row {bid!r} is not a planned branch")
        cand = candidates[bid]
        iid = row["initial_id"]
        _require(bid not in seen.get(iid, set()), f"{bid}: duplicate score row")
        seen.setdefault(iid, set()).add(bid)
        _require(row.get("run_id") == run_id and row.get("model_variant") == run.model,
                 f"{bid}: score row names another run or model")
        for key in PLAN_FIELDS:
            _require(row.get(key) == cand[key], f"{bid}: {key} differs from the plan")
        _require(iid not in tied, f"{bid}: a tied initial has a counterargument branch")
        _require(cand["required_initial_label"] == labels.get(iid),
                 f"{bid}: branch does not oppose its initial argmax")
        a, b = logits[iid]
        for key in ("after_logit_a", "after_logit_b"):
            _require(_finite(row.get(key)), f"{bid}: {key} is not a finite number")
        try:
            score = score_movement(
                initial_logit_a=a, initial_logit_b=b,
                after_logit_a=row["after_logit_a"], after_logit_b=row["after_logit_b"],
                counter_target_label=cand["counter_target_label"], near_tie_tau_logit=tau)
        except ScoreError as exc:
            raise AnalysisError(f"{bid}: {exc}") from exc
        recomputed = score.as_dict()
        for key in SCORE_FIELDS:
            _require(_same(row.get(key), recomputed[key]),
                     f"{bid}: {key} {row.get(key)!r} != recomputed {recomputed[key]!r}")
        _require(m_before_of.setdefault(iid, row["m_before"]) == row["m_before"],
                 f"{iid}: m_before differs across its branches")
    _require(seen == want_by_initial,
             f"{name}: scored branches are not exactly the 18 opposing each argmax")
    post = {row["branch_id"] for row in run.scores if row["post_exact_tie"]}
    recorded = {row["branch_id"] for row in run.post_ties}
    _require(post == recorded and len(recorded) == len(run.post_ties),
             f"{name}: post_ties.jsonl does not list exactly the post-tie rows")
    for row in run.post_ties:
        _require(row.get("exclusion_reason") == POST_TIE_REASON
                 and row.get("excluded_from") == ["flip_rate"]
                 and row.get("retained_in") == ["movement_toward_counter"],
                 f"{row['branch_id']}: post-tie record has the wrong reason or scope")


def validate_evidence(spec: AnalysisSpec) -> ValidatedEvidence:
    """Validate everything the analysis will read; refuse on the first mismatch."""
    root = spec.root
    inputs = spec.raw["inputs"]
    cfg = inputs["behavioral_config"]
    checks: list[dict[str, Any]] = []

    def record(check: str, observed: Any, expected: Any) -> None:
        _require(observed == expected, f"{check}: {observed!r} != {expected!r}")
        checks.append({"run": "inputs", "check": check, "observed": observed,
                       "expected": expected, "passed": True})

    config_path = root / cfg["path"]
    _require(config_path.is_file(), f"pinned behavioural config {config_path} is missing")
    record("sha256 behavioral_config", file_sha256(config_path), cfg["sha256"])
    try:
        behavioral = load_spec(config_path)
        plan = build_plan(behavioral)
    except BehavioralPlanError as exc:
        raise AnalysisError(f"pinned behavioural config refused: {exc}") from exc
    record("content_hash behavioral_config", behavioral.content_hash, cfg["content_hash"])
    record("behavioral_version", behavioral.version, cfg["behavioral_version"])
    record("pinned config stimuli hash", behavioral.raw["dataset"]["stimuli_sha256"],
           inputs["stimuli"]["sha256"])
    record("sha256 stimuli", file_sha256(root / inputs["stimuli"]["path"]),
           inputs["stimuli"]["sha256"])
    record("sha256 dataset_manifest", file_sha256(root / inputs["dataset_manifest"]["path"]),
           inputs["dataset_manifest"]["sha256"])
    tau = float(behavioral.raw["scoring"]["near_tie_tau_logit"])
    record("near-tie tau", tau, 0.4054651081)

    stimuli = {row["stimulus_id"]: row for row in _lines(root / inputs["stimuli"]["path"])}
    marker_cfg = yaml.safe_load((root / inputs["marker_config"]["path"]).read_text())
    markers = marker_cfg["markers"]
    _require(len(markers) == 12 and set(m["family"] for m in markers.values()) == set(FAMILIES),
             "v3 marker config does not hold 12 markers in the two families")
    answer = inputs["answer"]
    runs: dict[str, RunEvidence] = {}
    for model in REFERENCE_MODELS:
        want = inputs["runs"][model]
        pinned = behavioral.raw["models"][model]
        run = _load_run(model, model, want, spec, checks)
        checks += _check_metadata(model, run.metadata, want, spec, {
            "variant": (run.metadata.get("variant"), model),
            "answer_continuation": (run.metadata.get("answer_continuation"),
                                    answer["continuation"]),
            "answer_token_ids": (run.metadata.get("answer_token_ids"), answer["token_ids"]),
            "pinned revision": (pinned["revision"], want["revision"]),
            "pinned answer_token_ids": (pinned["answer_token_ids"], answer["token_ids"]),
        })
        _check_rows_against_plan(run, plan, tau)
        runs[model] = run
    variants_path = root / inputs["robustness_variants_file"]
    for name, want in (inputs.get("robustness_runs") or {}).items():
        vcfg_path = root / want["variant_config"]["path"]
        record(f"sha256 {name} variant config", file_sha256(vcfg_path),
               want["variant_config"]["sha256"])
        vcfg = yaml.safe_load(vcfg_path.read_text(encoding="utf-8"))
        run = _load_run(name, "base", want, spec, checks)
        checks += _check_metadata(name, run.metadata, want, spec, {
            "variant": (run.metadata.get("variant"), "base"),
            "prompt_variant.name": ((run.metadata.get("prompt_variant") or {}).get("name"), name),
            "variant_definition_hash": (run.metadata.get("variant_definition_hash"),
                                        vcfg["variant_definition_hash"]),
            "prompt_digest": (run.metadata.get("prompt_digest"), vcfg["prompt_digest"]),
            "variant_config_file_sha256": (run.metadata.get("variant_config_file_sha256"),
                                           want["variant_config"]["sha256"]),
            "answer_token_ids": (run.metadata.get("answer_token_ids"), vcfg["answer_token_ids"]),
            "answer_continuation": (run.metadata.get("answer_continuation"),
                                    vcfg["answer_continuation"]),
            "variant config source hash": (vcfg["source_config"]["file_sha256"], cfg["sha256"]),
            "variants file hash": (vcfg["variants_file"]["sha256"], file_sha256(variants_path)),
        })
        _check_rows_against_plan(run, plan, tau)
        runs[name] = run
    for run in runs.values():
        for row in run.scores:
            stim = stimuli[row["stimulus_id"]]
            _require(stim["marker_id"] == row["marker_id"] and
                     (row["marker_id"] == PLAIN_MARKER) == (row["condition"] in ("RP", "NP")),
                     f"{row['branch_id']}: marker does not match its stimulus")
    hashes = {
        "analysis_spec": spec.file_sha256,
        "behavioral_config": file_sha256(config_path),
        "stimuli": file_sha256(root / inputs["stimuli"]["path"]),
        "dataset_manifest": file_sha256(root / inputs["dataset_manifest"]["path"]),
        "marker_config": file_sha256(root / inputs["marker_config"]["path"]),
        "robustness_variants_file": file_sha256(variants_path),
    }
    for name, want in (inputs.get("robustness_runs") or {}).items():
        hashes[f"{name}/variant_config"] = file_sha256(root / want["variant_config"]["path"])
    for name, run in runs.items():
        for fname, digest in run.file_hashes.items():
            hashes[f"{name}/{fname}"] = digest
    refusals = {}
    for name, ref in (inputs.get("robustness_refusals") or {}).items():
        path = root / ref["path"]
        record(f"sha256 {name} refusal record", file_sha256(path), ref["sha256"])
        refusals[name] = {**json.loads(path.read_text()), "path": ref["path"],
                          "sha256": ref["sha256"]}
        hashes[f"{name}/refusal"] = ref["sha256"]
        if ref.get("diagnostic"):
            dpath = root / ref["diagnostic"]["path"]
            record(f"sha256 {name} diagnostic", file_sha256(dpath), ref["diagnostic"]["sha256"])
            refusals[name]["diagnostic"] = json.loads(dpath.read_text())
            hashes[f"{name}/diagnostic"] = ref["diagnostic"]["sha256"]
    return ValidatedEvidence(spec=spec, behavioral_config_path=config_path, runs=runs,
                             stimuli=stimuli, markers=markers,
                             openings=marker_cfg["openings"], input_hashes=hashes,
                             refusals=refusals, checks=checks)


def tie_counts(run: RunEvidence) -> dict[str, Any]:
    n_initial, n_ties, n_rows = len(run.initials), len(run.exclusions), len(run.scores)
    n_post = sum(1 for r in run.scores if r["post_exact_tie"])
    return {"initial_readings": n_initial, "exact_initial_ties": n_ties,
            "exact_initial_ties_pct": 100.0 * n_ties / n_initial,
            "valid_initials": n_initial - n_ties, "movement_rows": n_rows,
            "exact_post_ties": n_post,
            "exact_post_ties_pct": 100.0 * n_post / n_rows if n_rows else 0.0,
            "flip_defined_rows": n_rows - n_post}


def post_tie_breakdown(run: RunEvidence) -> Counter:
    return Counter((r["condition"], r["marker_id"], r["opening_id"])
                   for r in run.scores if r["post_exact_tie"])
