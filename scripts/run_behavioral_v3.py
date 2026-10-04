#!/usr/bin/env python3
"""Run exact A/B next-token scoring for one pinned v3 Llama checkpoint.

The runner is offline-only, makes no API calls, and never generates a
rationale. It first scores the initial choice, selects only counterarguments
supporting the opposite semantic option, then records movement in the A/B
logit margin. A pinned config and an explicit authorization environment
variable are both required.
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

from reasonstyle.behavioral.compatibility import (
    CompatibilityError,
    materialize_base,
    materialize_instruct,
    resolve_answer_tokens,
)
from reasonstyle.behavioral.plan import BehavioralPlanError, build_plan, load_spec
from reasonstyle.behavioral.runtime import select_runtime_candidates
from reasonstyle.behavioral.score import ScoreError, score_movement
from reasonstyle.generation.environment import (
    ModelNotCached,
    library_versions,
    resolve_cached_model,
)
from reasonstyle.hashing import canonical_json, file_sha256, sha256_of


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _materialize(row: dict[str, Any], *, variant: str, spec, tokenizer) -> str:
    cue = spec.raw["prompt"]["answer_cue"]
    if variant == "base":
        return materialize_base(
            row["transcript"], answer_cue=cue,
            template=spec.raw["prompt"]["base_plain_dialogue_v1"])
    return materialize_instruct(
        row["transcript"], answer_cue=cue, tokenizer=tokenizer)


def _score_prompts(rows: list[dict[str, Any]], *, id_key: str, variant: str,
                   spec, tokenizer, model, torch, device, batch_size: int,
                   token_ids: dict[str, int]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise CompatibilityError("tokenizer has neither a pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [_materialize(row, variant=variant, spec=spec, tokenizer=tokenizer)
                   for row in batch]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True,
            add_special_tokens=(variant == "base"))
        if "attention_mask" not in encoded:
            raise CompatibilityError("tokenizer batch has no attention mask")
        model_inputs = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**model_inputs).logits
        last_positions = model_inputs["attention_mask"].sum(dim=1) - 1
        for offset, (row, prompt) in enumerate(zip(batch, prompts, strict=True)):
            position = int(last_positions[offset].item())
            output[row[id_key]] = {
                "A": float(logits[offset, position, token_ids["A"]].float().item()),
                "B": float(logits[offset, position, token_ids["B"]].float().item()),
                "input_tokens": int(model_inputs["attention_mask"][offset].sum().item()),
                "materialized_prompt_sha256": sha256_of(prompt),
            }
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True, choices=("base", "instruct"))
    parser.add_argument("--hf-home", help="local Hugging Face cache root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    destination = Path(args.out)
    temporary: Path | None = None
    try:
        spec = load_spec(args.config)
        if spec.raw["status"] != "ready_for_smoke":
            raise BehavioralPlanError("run requires a compatibility-pinned config")
        gate = spec.raw["runtime"]["authorization_environment_variable"]
        if os.environ.get(gate) != "1":
            raise BehavioralPlanError(f"set {gate}=1 only for an authorized run")
        if destination.exists():
            raise BehavioralPlanError(f"refusing to overwrite {destination}")
        variant_cfg = spec.raw["models"][args.variant]
        cached = resolve_cached_model(variant_cfg["repo_id"], hf_home=args.hf_home)
        if cached.revision != variant_cfg["revision"]:
            raise BehavioralPlanError(
                f"cached revision {cached.revision} != pinned {variant_cfg['revision']}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BehavioralPlanError(
                "torch and transformers must be installed in the pinned GPU environment") from exc
        if not torch.cuda.is_available():
            raise BehavioralPlanError("CUDA is unavailable; refusing an accidental CPU run")
        if not torch.cuda.is_bf16_supported():
            raise BehavioralPlanError("the selected CUDA device does not support bfloat16")
        seed = int(spec.raw["runtime"]["seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(
            bool(spec.raw["runtime"]["deterministic_algorithms"]))
        tokenizer = AutoTokenizer.from_pretrained(
            cached.snapshot_path, local_files_only=True)
        if args.variant == "instruct":
            actual_template = sha256_of(tokenizer.chat_template or "")
            if actual_template != variant_cfg["chat_template_sha256"]:
                raise CompatibilityError("instruct chat template differs from pinned evidence")
        plan = build_plan(spec)
        sample = (plan.initials[0], plan.initials[-1],
                  plan.candidates[0], plan.candidates[-1])
        for row in sample:
            prompt = _materialize(row, variant=args.variant, spec=spec, tokenizer=tokenizer)
            resolution = resolve_answer_tokens(tokenizer, prompt)
            if (resolution.answer_token_ids != variant_cfg["answer_token_ids"] or
                    resolution.answer_continuation != variant_cfg["answer_continuation"]):
                raise CompatibilityError("runtime A/B token resolution differs from pinned evidence")
        device = torch.device("cuda")
        model = AutoModelForCausalLM.from_pretrained(
            cached.snapshot_path, local_files_only=True, torch_dtype=torch.bfloat16)
        model.to(device)
        model.eval()
        batch_size = args.batch_size or int(spec.raw["runtime"]["batch_size"])
        if batch_size < 1:
            raise BehavioralPlanError("batch size must be positive")
        initials = plan.initials if args.mode == "full" else plan.initials[:2]
        initial_ids = {row["initial_id"] for row in initials}
        token_ids = variant_cfg["answer_token_ids"]
        raw_initial = _score_prompts(
            initials, id_key="initial_id", variant=args.variant, spec=spec,
            tokenizer=tokenizer, model=model, torch=torch, device=device,
            batch_size=batch_size, token_ids=token_ids)
        selected = select_runtime_candidates(
            plan, raw_initial, initial_ids=initial_ids)
        raw_after = _score_prompts(
            selected, id_key="branch_id", variant=args.variant, spec=spec,
            tokenizer=tokenizer, model=model, torch=torch, device=device,
            batch_size=batch_size, token_ids=token_ids)
        run_id = destination.name
        initial_rows = []
        for row in initials:
            measured = raw_initial[row["initial_id"]]
            label = "A" if measured["A"] > measured["B"] else "B"
            initial_rows.append({
                "run_id": run_id, "model_variant": args.variant,
                **{key: row[key] for key in ("initial_id", "decision_id", "domain",
                                              "scenario_id", "order_id")},
                "initial_label": label,
                "initial_option": row["label_to_option"][label],
                "logit_a": measured["A"], "logit_b": measured["B"],
                "input_tokens": measured["input_tokens"],
                "materialized_prompt_sha256": measured["materialized_prompt_sha256"],
            })
        score_rows = []
        tau = float(spec.raw["scoring"]["near_tie_tau_logit"])
        for row in selected:
            before = raw_initial[row["initial_id"]]
            after = raw_after[row["branch_id"]]
            score = score_movement(
                initial_logit_a=before["A"], initial_logit_b=before["B"],
                after_logit_a=after["A"], after_logit_b=after["B"],
                counter_target_label=row["counter_target_label"],
                near_tie_tau_logit=tau)
            score_rows.append({
                "run_id": run_id, "model_variant": args.variant,
                **{key: row[key] for key in (
                    "branch_id", "initial_id", "stimulus_id", "decision_id", "domain",
                    "scenario_id", "variant_id", "supported_option", "condition",
                    "reason", "style", "marker_id", "marker_family", "marker_string",
                    "opening_id", "order_id", "required_initial_option",
                    "counter_target_label", "paired_control_stimulus_id")},
                **score.as_dict(),
                "input_tokens": after["input_tokens"],
                "materialized_prompt_sha256": after["materialized_prompt_sha256"],
            })
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(
            prefix=destination.name + ".incomplete.", dir=destination.parent))
        _write_jsonl(temporary / "initial_scores.jsonl", initial_rows)
        _write_jsonl(temporary / "behavioral_scores.jsonl", score_rows)
        metadata = {
            "status": "complete", "run_id": run_id,
            "mode": args.mode, "variant": args.variant,
            "behavioral_version": spec.version,
            "behavioral_config_file_sha256": file_sha256(spec.path),
            "behavioral_config_content_hash": spec.content_hash,
            "stimuli_sha256": spec.raw["dataset"]["stimuli_sha256"],
            "repo_id": cached.repo_id, "revision": cached.revision,
            "dtype": "bfloat16", "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
            "platform": platform.platform(), "libraries": library_versions(),
            "seed": seed, "batch_size": batch_size,
            "answer_continuation": variant_cfg["answer_continuation"],
            "answer_token_ids": token_ids,
            "counts": {"initial": len(initial_rows), "post_counterargument": len(score_rows)},
            "files": {
                "initial_scores.jsonl": file_sha256(temporary / "initial_scores.jsonl"),
                "behavioral_scores.jsonl": file_sha256(temporary / "behavioral_scores.jsonl"),
            },
            "api_calls": False, "free_form_generation": False,
        }
        (temporary / "RUN_METADATA.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "COMPLETE").write_text("complete\n", encoding="utf-8")
        temporary.replace(destination)
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0
    except (BehavioralPlanError, CompatibilityError, ModelNotCached, ScoreError,
            OSError, KeyError, TypeError, ValueError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        if temporary is not None:
            print(f"incomplete evidence retained at {temporary}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
