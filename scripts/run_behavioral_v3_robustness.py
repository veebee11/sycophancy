#!/usr/bin/env python3
"""Exact next-token scoring of one pinned Base prompt-format variant.

Phases (each to its own new directory; nothing is overwritten or mixed):

``reference-check``  score the 240 initials of the *reference* rendering through
                     this runner and require bitwise equality with the reference
                     run's recorded logits before any variant is trusted.
``initial``          score and record the variant's 240 initial prompts only.
``full``             re-score the initials (they must equal the recorded
                     initial-phase logits exactly), select the 18 branches that
                     oppose each non-tied argmax, score them, and write a run in
                     the reference run's format with a COMPLETE marker.

Offline only: the cached pinned Base revision, bfloat16, deterministic CUDA,
no network, no generation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from reasonstyle.behavioral.compatibility import CompatibilityError
from reasonstyle.behavioral.evidence import (
    POST_TIE_REASON,
    completion_label,
    configure_cublas_workspace,
    initial_records,
    post_tie_records,
    run_status,
)
from reasonstyle.behavioral.plan import BehavioralPlanError, build_plan, load_spec
from reasonstyle.behavioral.robustness import (
    PromptVariant,
    build_variant_plan,
    resolve_label_tokens,
)
from reasonstyle.behavioral.runtime import (
    TIE_REASON,
    expected_post_counterargument,
    runtime_selection,
)
from reasonstyle.behavioral.score import ScoreError, score_movement
from reasonstyle.generation.environment import (
    ModelNotCached,
    library_versions,
    resolve_cached_model,
)
from reasonstyle.hashing import canonical_json, file_sha256

BRANCH_FIELDS = (
    "branch_id", "initial_id", "stimulus_id", "decision_id", "domain", "scenario_id",
    "variant_id", "supported_option", "condition", "reason", "style", "marker_id",
    "marker_family", "marker_string", "opening_id", "order_id", "required_initial_option",
    "counter_target_label", "paired_control_stimulus_id")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8")


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def _score(rows, id_key, tokenizer, model, torch, device, batch_size, token_ids):
    """Identical batching and readout to scripts/run_behavioral_v3.py (base)."""
    out: dict[str, dict[str, Any]] = {}
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [r["prompt"] for r in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        inputs = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        last = inputs["attention_mask"].sum(dim=1) - 1
        for i, row in enumerate(batch):
            pos = int(last[i].item())
            out[row[id_key]] = {
                "A": float(logits[i, pos, token_ids["A"]].float().item()),
                "B": float(logits[i, pos, token_ids["B"]].float().item()),
                "input_tokens": int(inputs["attention_mask"][i].sum().item()),
                "materialized_prompt_sha256": row["prompt_sha256"],
            }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-config", required=True)
    parser.add_argument("--phase", required=True, choices=("reference-check", "initial", "full"))
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--initial-evidence", help="initial-phase directory (full phase)")
    parser.add_argument("--reference-run", help="reference run directory (reference-check)")
    args = parser.parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    destination = Path(args.out)
    temporary: Path | None = None
    try:
        vcfg_path = Path(args.variant_config)
        vcfg = yaml.safe_load(vcfg_path.read_text(encoding="utf-8"))
        if vcfg.get("status") != "ready_for_robustness_run":
            raise BehavioralPlanError("variant config is not a pinned robustness config")
        spec = load_spec(vcfg["source_config"]["path"])
        if (file_sha256(spec.path) != vcfg["source_config"]["file_sha256"]
                or spec.content_hash != vcfg["source_config"]["content_hash"]):
            raise BehavioralPlanError("source behavioural config differs from the pinned variant")
        if file_sha256(vcfg["compatibility_report"]["path"]) != vcfg["compatibility_report"]["sha256"]:
            raise BehavioralPlanError("compatibility report differs from the pinned variant")
        gate = spec.raw["runtime"]["authorization_environment_variable"]
        if os.environ.get(gate) != "1":
            raise BehavioralPlanError(f"set {gate}=1 only for an authorized run")
        if destination.exists():
            raise BehavioralPlanError(f"refusing to overwrite {destination}")
        v = vcfg["variant"]
        variant = PromptVariant(name=v["name"], run_id=v["run_id"],
                                response_labels=v["response_labels"],
                                display_order=tuple(v["display_order"]),
                                template_name=v["template_name"], template=v["template"])
        if variant.definition_hash != vcfg["variant_definition_hash"]:
            raise BehavioralPlanError("variant definition hash mismatch")
        if (args.phase == "reference-check") != (variant.name == "reference"):
            raise BehavioralPlanError("the reference-check phase is for the reference variant only")
        model_cfg = spec.raw["models"]["base"]
        cached = resolve_cached_model(model_cfg["repo_id"], hf_home=args.hf_home)
        if cached.revision != model_cfg["revision"] or cached.revision != vcfg["model"]["revision"]:
            raise BehavioralPlanError(f"cached revision {cached.revision} != pinned")
        plan = build_plan(spec)
        vplan = build_variant_plan(spec, plan, variant)
        if vplan.prompt_digest != vcfg["prompt_digest"]:
            raise BehavioralPlanError("rendered prompts differ from the pinned prompt digest")
        deterministic = bool(spec.raw["runtime"]["deterministic_algorithms"])
        cublas = configure_cublas_workspace(deterministic, os.environ, sys.modules)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise BehavioralPlanError("CUDA with bfloat16 is required; refusing a CPU run")
        seed = int(spec.raw["runtime"]["seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic)
        tokenizer = AutoTokenizer.from_pretrained(cached.snapshot_path, local_files_only=True)
        token_ids = vcfg["answer_token_ids"]
        for row in (vplan.initials[0], vplan.initials[-1], vplan.candidates[0],
                    vplan.candidates[-1]):
            if resolve_label_tokens(tokenizer, row["prompt"], variant.response_labels) != token_ids:
                raise CompatibilityError("runtime token resolution differs from the pinned gate")
        device = torch.device("cuda")
        model = AutoModelForCausalLM.from_pretrained(
            cached.snapshot_path, local_files_only=True, torch_dtype=torch.bfloat16)
        model.to(device)
        model.eval()
        batch_size = int(spec.raw["runtime"]["batch_size"])
        initial_logits = _score(vplan.initials, "initial_id", tokenizer, model, torch, device,
                                batch_size, token_ids)
        run_id = destination.name
        common_meta = {
            "run_id": run_id, "phase": args.phase, "mode": "full", "variant": "base",
            "prompt_variant": variant.as_dict(),
            "variant_definition_hash": variant.definition_hash,
            "variant_config_file_sha256": file_sha256(vcfg_path),
            "prompt_digest": vplan.prompt_digest,
            "behavioral_version": spec.version,
            "behavioral_config_file_sha256": file_sha256(spec.path),
            "behavioral_config_content_hash": spec.content_hash,
            "stimuli_sha256": spec.raw["dataset"]["stimuli_sha256"],
            "repo_id": cached.repo_id, "revision": cached.revision,
            "dtype": "bfloat16", "device": "cuda", "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "platform": platform.platform(), "libraries": library_versions(),
            "seed": seed, "batch_size": batch_size,
            "answer_continuation": vcfg["answer_continuation"], "answer_token_ids": token_ids,
            "slot_semantics": "A/B in every file are slots; the model read and answered with "
                              "prompt_variant.response_labels[slot]",
            "deterministic_algorithms": deterministic, "cublas_workspace_config": cublas,
            "api_calls": False, "free_form_generation": False, "network_access": False,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".incomplete.",
                                          dir=destination.parent))
        selection = runtime_selection(plan, initial_logits)
        records = initial_records(plan.initials, initial_logits, selection,
                                  run_id=run_id, variant="base")
        if args.phase == "reference-check":
            reference = {r["initial_id"]: r for r in _lines(Path(args.reference_run)
                                                            / "initial_scores.jsonl")}
            diffs = [r["initial_id"] for r in records
                     if (r["logit_a"], r["logit_b"], r["input_tokens"]) !=
                     (reference[r["initial_id"]]["logit_a"], reference[r["initial_id"]]["logit_b"],
                      reference[r["initial_id"]]["input_tokens"])
                     or r["materialized_prompt_sha256"] !=
                     reference[r["initial_id"]]["materialized_prompt_sha256"]]
            _jsonl(temporary / "initial_scores.jsonl", records)
            result = {**common_meta, "status": "complete" if not diffs else "mismatch",
                      "reference_run": str(args.reference_run),
                      "reference_initial_scores_sha256": file_sha256(
                          Path(args.reference_run) / "initial_scores.jsonl"),
                      "initials_compared": len(records), "bitwise_mismatches": diffs,
                      "files": {"initial_scores.jsonl":
                                file_sha256(temporary / "initial_scores.jsonl")}}
            (temporary / "REFERENCE_CHECK.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(destination)
            print(json.dumps({k: result[k] for k in ("status", "initials_compared")}))
            return 0 if not diffs else 1
        if args.phase == "initial":
            _jsonl(temporary / "initial_scores.jsonl", records)
            _jsonl(temporary / "initial_exclusions.jsonl",
                   [{"run_id": run_id, "model_variant": "base", **x}
                    for x in selection.exclusions])
            meta = {**common_meta, "status": "initial_phase_complete",
                    "counts": {"initial": len(records),
                               "excluded_exact_initial_ties": len(selection.exclusions)},
                    "files": {n: file_sha256(temporary / n) for n in
                              ("initial_scores.jsonl", "initial_exclusions.jsonl")}}
            (temporary / "INITIAL_METADATA.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (temporary / "INITIAL_COMPLETE").write_text("initial_phase_complete\n")
            temporary.replace(destination)
            print(json.dumps(meta["counts"]))
            return 0
        # full phase: the initials must reproduce the validated initial phase exactly
        evidence = Path(args.initial_evidence)
        imeta = json.loads((evidence / "INITIAL_METADATA.json").read_text())
        if not (evidence / "INITIAL_COMPLETE").is_file() or \
                imeta["variant_definition_hash"] != variant.definition_hash or \
                imeta["prompt_digest"] != vplan.prompt_digest:
            raise BehavioralPlanError("initial-phase evidence is missing or for another variant")
        for name, digest in imeta["files"].items():
            if file_sha256(evidence / name) != digest:
                raise BehavioralPlanError(f"initial-phase {name} does not match its hash")
        recorded = {r["initial_id"]: r for r in _lines(evidence / "initial_scores.jsonl")}
        drift = [r["initial_id"] for r in records
                 if (r["logit_a"], r["logit_b"]) != (recorded[r["initial_id"]]["logit_a"],
                                                     recorded[r["initial_id"]]["logit_b"])]
        if drift:
            raise BehavioralPlanError(f"{len(drift)} initial logits differ from the initial phase")
        by_branch = {r["branch_id"]: r for r in vplan.candidates}
        selected = [by_branch[r["branch_id"]] for r in selection.selected]
        after = _score(selected, "branch_id", tokenizer, model, torch, device, batch_size,
                       token_ids)
        tau = float(spec.raw["scoring"]["near_tie_tau_logit"])
        scores = []
        for row in selected:
            before = initial_logits[row["initial_id"]]
            post = after[row["branch_id"]]
            score = score_movement(initial_logit_a=before["A"], initial_logit_b=before["B"],
                                   after_logit_a=post["A"], after_logit_b=post["B"],
                                   counter_target_label=row["counter_target_label"],
                                   near_tie_tau_logit=tau)
            scores.append({"run_id": run_id, "model_variant": "base",
                           **{k: row[k] for k in BRANCH_FIELDS}, **score.as_dict(),
                           "input_tokens": post["input_tokens"],
                           "materialized_prompt_sha256": post["materialized_prompt_sha256"]})
        exclusions = [{"run_id": run_id, "model_variant": "base", **x}
                      for x in selection.exclusions]
        post_rows = post_tie_records(selection.selected, scores)
        _jsonl(temporary / "initial_scores.jsonl", records)
        _jsonl(temporary / "behavioral_scores.jsonl", scores)
        _jsonl(temporary / "initial_exclusions.jsonl", exclusions)
        _jsonl(temporary / "post_ties.jsonl", post_rows)
        n_ties, n_post = len(exclusions), len(post_rows)
        expected = expected_post_counterargument(len(records), n_ties)
        if len(scores) != expected:
            raise BehavioralPlanError(f"{len(scores)} scores, expected {expected}")
        meta = {**common_meta, "status": "complete", "completion": completion_label(n_ties, n_post),
                "initial_phase_evidence": {"path": str(evidence),
                                           "initial_metadata_sha256":
                                               file_sha256(evidence / "INITIAL_METADATA.json")},
                "exact_tie_policy": "refuse: exact initial ties excluded with no branches; "
                                    "no tie-break of any kind",
                "post_tie_policy": "exact post ties kept for movement with m_after = 0; "
                                   "final_label and flip null; omitted only from flip",
                "counts": {"initial": len(records), "initial_included": len(records) - n_ties,
                           "excluded_exact_initial_ties": n_ties,
                           "post_counterargument": len(scores),
                           "expected_post_counterargument": expected,
                           "expected_formula": "18 x (initial - excluded_exact_initial_ties)",
                           "post_exact_ties": n_post, "flip_defined": len(scores) - n_post},
                "exclusions": {"file": "initial_exclusions.jsonl", "count": n_ties,
                               "reasons": {TIE_REASON: n_ties} if n_ties else {}},
                "post_ties": {"file": "post_ties.jsonl", "count": n_post,
                              "reasons": {POST_TIE_REASON: n_post} if n_post else {}},
                "files": {n: file_sha256(temporary / n) for n in (
                    "initial_scores.jsonl", "behavioral_scores.jsonl",
                    "initial_exclusions.jsonl", "post_ties.jsonl")}}
        (temporary / "RUN_METADATA.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "COMPLETE").write_text("complete\n", encoding="utf-8")
        temporary.replace(destination)
        status, problems = run_status(destination)
        if problems:
            # never leave a directory that run_status rejects under the final name
            failed = Path(tempfile.mkdtemp(prefix=destination.name + ".incomplete.failed.",
                                           dir=destination.parent))
            failed.rmdir()
            destination.replace(failed)
            temporary = failed
            raise BehavioralPlanError("run_status: " + "; ".join(problems))
        print(json.dumps({"status": status, **meta["counts"]}, sort_keys=True))
        return 0
    except (BehavioralPlanError, CompatibilityError, ModelNotCached, ScoreError, OSError,
            KeyError, TypeError, ValueError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        if temporary is not None and temporary.exists():
            print(f"incomplete evidence retained at {temporary}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
