"""Deterministic, model-independent plan for the v3 behavioural experiment.

The plan contains 240 initial readings (120 scenarios x two option orders) and
8,640 possible counterargument branches. A model's initial A/B argmax selects
exactly half of those candidates: the 4,320 branches whose counterargument
supports the opposite semantic option. No model library is imported here.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..hashing import canonical_json, content_hash, file_sha256, sha256_of


class BehavioralPlanError(ValueError):
    """The behavioural specification or v3 stimulus set is inconsistent."""


@dataclass(frozen=True)
class BehavioralSpec:
    path: Path
    root: Path
    raw: dict[str, Any]

    @property
    def version(self) -> str:
        return self.raw["behavioral_version"]

    @property
    def content_hash(self) -> str:
        return content_hash(self.raw)

    @property
    def stimuli_path(self) -> Path:
        return self.root / self.raw["dataset"]["stimuli"]

    @property
    def manifest_path(self) -> Path:
        return self.root / self.raw["dataset"]["manifest"]


@dataclass(frozen=True)
class BehavioralPlan:
    initials: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    manifest: dict[str, Any]


def _root_of(path: Path) -> Path:
    path = path.resolve()
    if path.parent.name == "configs":
        return path.parent.parent
    raise BehavioralPlanError(f"behavioural config must live under configs/: {path}")


def load_spec(path: str | Path) -> BehavioralSpec:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise BehavioralPlanError("behavioural config is not a mapping")
    spec = BehavioralSpec(path=path.resolve(), root=_root_of(path), raw=raw)
    _validate_spec(spec)
    return spec


def _require(value: bool, message: str) -> None:
    if not value:
        raise BehavioralPlanError(message)


def _validate_spec(spec: BehavioralSpec) -> None:
    raw = spec.raw
    status = raw.get("status")
    _require(status in {"setup_authorized_models_unfrozen", "ready_for_smoke"},
             "unexpected behavioural setup status")
    dataset = raw.get("dataset") or {}
    _require(dataset.get("version") == "full_v3_multimarker_openings",
             "the behavioural setup must name the v3 dataset")
    for path_key, hash_key in (("stimuli", "stimuli_sha256"),
                               ("manifest", "manifest_sha256")):
        artifact = spec.root / dataset[path_key]
        _require(artifact.is_file(), f"missing dataset artifact: {artifact}")
        actual = file_sha256(artifact)
        _require(actual == dataset[hash_key],
                 f"{dataset[path_key]} hashes to {actual}, not {dataset[hash_key]}")
    manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("dataset_version") == dataset["version"],
             "v3 manifest names another dataset")
    _require(manifest.get("status") == dataset.get("status_at_setup") == "draft",
             "this setup is explicitly bound to the reviewed draft status")
    _require(manifest.get("machine_validation", {}).get("errors") == 0,
             "the v3 manifest reports machine errors")
    _require((spec.root / raw["authorization"]["record"]).is_file(),
             "the reported behavioural authorization record is missing")
    _require(set(raw.get("models", {})) >= {"base", "instruct"},
             "base and instruct model entries are required")
    expected_selection = ("provisional_pending_compatibility" if
                          status == "setup_authorized_models_unfrozen" else
                          "pinned_after_compatibility")
    _require(raw["models"]["selection_status"] == expected_selection,
             "model selection status does not match setup status")
    for variant in ("base", "instruct"):
        model = raw["models"][variant]
        if status == "setup_authorized_models_unfrozen":
            _require(model.get("revision") is None,
                     f"{variant} revision must remain unresolved before compatibility")
            _require(model.get("compatibility_status") == "unverified",
                     f"{variant} compatibility must remain unverified in the draft")
            _require(model.get("answer_continuation") == {"A": None, "B": None},
                     f"{variant} answer continuations must not be guessed")
            _require(model.get("answer_token_ids") == {"A": None, "B": None},
                     f"{variant} answer token IDs must not be guessed")
        else:
            _require(isinstance(model.get("revision"), str) and model["revision"],
                     f"{variant} needs an exact cached revision")
            _require(model.get("compatibility_status") == "passed",
                     f"{variant} has not passed compatibility")
            continuations = model.get("answer_continuation")
            token_ids = model.get("answer_token_ids")
            _require(isinstance(continuations, dict) and
                     set(continuations) == {"A", "B"} and
                     all(isinstance(v, str) and v for v in continuations.values()),
                     f"{variant} needs resolved A/B continuations")
            _require(isinstance(token_ids, dict) and set(token_ids) == {"A", "B"} and
                     all(isinstance(v, int) for v in token_ids.values()) and
                     token_ids["A"] != token_ids["B"],
                     f"{variant} needs distinct resolved A/B token IDs")
            evidence = model.get("compatibility_report")
            _require(isinstance(evidence, dict) and evidence.get("sha256"),
                     f"{variant} compatibility evidence is not pinned")
    orders = raw["prompt"]["option_orders"]
    _require(orders == [{"order_id": "o1", "A": "opt_1", "B": "opt_2"},
                        {"order_id": "o2", "A": "opt_2", "B": "opt_1"}],
             "option orders must be the exact counterbalanced mappings")
    _require(raw["prompt"]["answer_labels"] == ["A", "B"],
             "answer labels must be A and B")
    design = raw["design"]
    _require((design["scenarios"], design["initial_prompts_per_model"],
              design["branch_candidates_per_model"],
              design["selected_branches_per_model"]) == (120, 240, 8640, 4320),
             "behavioural design counts do not match full v3")


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def _prompt_hash(transcript: list[dict[str, str]], answer_cue: str) -> str:
    return content_hash({"turns": transcript, "answer_cue": answer_cue})


def _turn_one(scenario: dict[str, Any], order: dict[str, str], spec: BehavioralSpec) -> str:
    prompt = spec.raw["prompt"]
    lines = "\n".join(prompt["option_line_format"].format(
        label=label, option_text=scenario["options"][order[label]]) for label in ("A", "B"))
    return "\n\n".join((scenario["scenario_text"], lines,
                         f'{prompt["question_text"]} {prompt["instruction_text"]}'))


def _opposite(option: str) -> str:
    return {"opt_1": "opt_2", "opt_2": "opt_1"}[option]


def _label_for(order: dict[str, str], option: str) -> str:
    return next(label for label in ("A", "B") if order[label] == option)


def _load_stimuli(spec: BehavioralSpec) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in spec.stimuli_path.read_text(encoding="utf-8").splitlines()
            if line]
    _require(len(rows) == 4320, f"expected 4320 v3 stimuli, got {len(rows)}")
    ids = [row["stimulus_id"] for row in rows]
    _require(len(set(ids)) == len(ids), "duplicate v3 stimulus ID")
    return rows


def build_plan(spec: BehavioralSpec) -> BehavioralPlan:
    stimuli = _load_stimuli(spec)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stimuli:
        grouped[row["scenario_id"]].append(row)
    _require(len(grouped) == 120, f"expected 120 scenarios, got {len(grouped)}")

    initials: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    cue = spec.raw["prompt"]["answer_cue"]
    question = f'{spec.raw["prompt"]["question_text"]} '
    question += spec.raw["prompt"]["instruction_text"]
    for scenario_id in sorted(grouped):
        rows = grouped[scenario_id]
        _require(len(rows) == 36, f"{scenario_id}: expected 36 rendered stimuli")
        first = rows[0]
        _require(Counter(r["supported_option"] for r in rows) ==
                 Counter({"opt_1": 18, "opt_2": 18}),
                 f"{scenario_id}: expected 18 stimuli per semantic target")
        scenario = {k: first[k] for k in ("decision_id", "domain", "scenario_id",
                                           "variant_id", "scenario_text", "options")}
        for entry in spec.raw["prompt"]["option_orders"]:
            order = {"A": entry["A"], "B": entry["B"]}
            turn_one = _turn_one(scenario, order, spec)
            initial_transcript = [{"role": "user", "content": turn_one}]
            initial_id = f'{spec.version}.{scenario_id}.{entry["order_id"]}.initial'
            initials.append({
                "initial_id": initial_id,
                **{k: scenario[k] for k in
                   ("decision_id", "domain", "scenario_id", "variant_id")},
                "order_id": entry["order_id"], "label_to_option": order,
                "transcript": initial_transcript,
                "prompt_hash": _prompt_hash(initial_transcript, cue),
            })
            for stimulus in sorted(rows, key=lambda r: r["stimulus_id"]):
                required_initial = _opposite(stimulus["supported_option"])
                initial_label = _label_for(order, required_initial)
                counter_label = _label_for(order, stimulus["supported_option"])
                transcript = [
                    {"role": "user", "content": turn_one},
                    {"role": "assistant", "content": initial_label},
                    {"role": "user", "content":
                        "\n\n".join((stimulus["rendered"], question))},
                ]
                candidates.append({
                    "branch_id": f'{spec.version}.{entry["order_id"]}.{stimulus["stimulus_id"]}',
                    "initial_id": initial_id,
                    "stimulus_id": stimulus["stimulus_id"],
                    **{k: stimulus[k] for k in (
                        "decision_id", "domain", "scenario_id", "variant_id",
                        "supported_option", "condition", "reason", "style", "marker_id",
                        "marker_family", "marker_string", "opening_id", "opening",
                        "paired_control_stimulus_id")},
                    "order_id": entry["order_id"], "label_to_option": order,
                    "required_initial_option": required_initial,
                    "required_initial_label": initial_label,
                    "counter_target_label": counter_label,
                    "transcript": transcript,
                    "prompt_hash": _prompt_hash(transcript, cue),
                })

    initials.sort(key=lambda r: r["initial_id"])
    candidates.sort(key=lambda r: r["branch_id"])
    _require(len(initials) == 240, f"expected 240 initial prompts, got {len(initials)}")
    _require(len(candidates) == 8640,
             f"expected 8640 branch candidates, got {len(candidates)}")
    per_initial = Counter(r["initial_id"] for r in candidates)
    _require(set(per_initial.values()) == {36}, "each initial prompt needs 36 candidates")
    by_initial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_initial[row["initial_id"]].append(row)
    for initial_id, mine in by_initial.items():
        _require(Counter(r["required_initial_option"] for r in mine) ==
                 Counter({"opt_1": 18, "opt_2": 18}),
                 f"{initial_id}: either argmax must select exactly 18 branches")

    initial_text = _jsonl(initials)
    candidate_text = _jsonl(candidates)
    manifest = {
        "behavioral_version": spec.version,
        "status": spec.raw["status"],
        "behavioral_config_sha256": spec.content_hash,
        "dataset_version": spec.raw["dataset"]["version"],
        "stimuli_sha256": spec.raw["dataset"]["stimuli_sha256"],
        "counts": {
            "scenarios": 120, "option_orders": 2, "initials_per_model": 240,
            "branch_candidates_per_model": 8640,
            "selected_branches_per_model": 4320, "models": 2,
        },
        "initials_sha256": sha256_of(initial_text),
        "candidates_sha256": sha256_of(candidate_text),
        "models": {v: dict(spec.raw["models"][v]) for v in ("base", "instruct")},
        "live_model_call_made": False,
    }
    return BehavioralPlan(initials=initials, candidates=candidates, manifest=manifest)


def plan_texts(plan: BehavioralPlan) -> dict[str, str]:
    return {
        "initials": _jsonl(plan.initials),
        "candidates": _jsonl(plan.candidates),
        "manifest": json.dumps(plan.manifest, indent=2, sort_keys=True) + "\n",
    }
