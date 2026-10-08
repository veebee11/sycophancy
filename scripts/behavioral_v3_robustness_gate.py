#!/usr/bin/env python3
"""Offline compatibility gate and pinning for one Base prompt-format variant.

Loads only the cached tokenizer of the pinned Base revision (no weights, no
forward pass, no network), renders every initial and candidate prompt of the
variant, and requires the variant's two response labels to be distinct
single-token continuations of every one of them. On success it writes the
report, a per-prompt hash list and a separate pinned variant config; on failure
it writes a refusal record and pins nothing. No replacement label is tried.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from reasonstyle.behavioral.compatibility import CompatibilityError
from reasonstyle.behavioral.plan import BehavioralPlanError, build_plan, load_spec
from reasonstyle.behavioral.robustness import (
    build_variant_plan,
    compatibility_gate,
    load_variants,
    pinned_variant_config,
)
from reasonstyle.generation.environment import ModelNotCached, resolve_cached_model
from reasonstyle.hashing import canonical_json, file_sha256


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise CompatibilityError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="configs/robustness/base_prompt_variants_v1.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--report-dir", default="runs/behavioral_v3_robustness/compatibility")
    parser.add_argument("--config-dir", default="configs/robustness")
    args = parser.parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    report_dir = Path(args.report_dir)
    try:
        variants_path = Path(args.variants)
        raw, variants = load_variants(variants_path)
        variant = variants[args.variant]
        spec = load_spec(raw["source_config"]["path"])
        if (file_sha256(spec.path) != raw["source_config"]["file_sha256"]
                or spec.content_hash != raw["source_config"]["content_hash"]):
            raise BehavioralPlanError("source behavioural config differs from the variants file")
        plan = build_plan(spec)
        model_cfg = spec.raw["models"]["base"]
        cached = resolve_cached_model(model_cfg["repo_id"], hf_home=args.hf_home)
        if cached.revision != model_cfg["revision"]:
            raise BehavioralPlanError(f"cached revision {cached.revision} != pinned")
        from transformers import AutoTokenizer, __version__ as tf_version
        tokenizer = AutoTokenizer.from_pretrained(cached.snapshot_path, local_files_only=True)
        vplan = build_variant_plan(spec, plan, variant)
        try:
            report = compatibility_gate(tokenizer, vplan, raw["required_continuation_prefix"])
        except CompatibilityError as exc:
            refusal = {"compatibility_status": "refused", "variant": variant.name,
                       "variant_definition_hash": variant.definition_hash,
                       "reason": str(exc), "revision": cached.revision,
                       "network_access": False, "model_forward_pass": False}
            _write_new(report_dir / f"{variant.name}.refused.json",
                       json.dumps(refusal, indent=2, sort_keys=True) + "\n")
            raise
        report.update({"repo_id": cached.repo_id, "revision": cached.revision,
                       "snapshot_path": cached.snapshot_path,
                       "transformers_version": tf_version,
                       "tokenizer_class": tokenizer.__class__.__name__,
                       "source_config_file_sha256": file_sha256(spec.path),
                       "variants_file_sha256": file_sha256(variants_path)})
        prompts = "".join(canonical_json({"id": r.get("branch_id", r["initial_id"]),
                                          "prompt_sha256": r["prompt_sha256"]}) + "\n"
                          for r in vplan.initials + vplan.candidates)
        _write_new(report_dir / f"{variant.name}.prompts.jsonl", prompts)
        report["prompt_hash_list_sha256"] = file_sha256(report_dir / f"{variant.name}.prompts.jsonl")
        report_path = report_dir / f"{variant.name}.json"
        _write_new(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
        pinned = pinned_variant_config(variants_path, variant, spec, report, report_path,
                                       cached.revision)
        _write_new(Path(args.config_dir) / f"base_{variant.name}_v1.pinned.yaml",
                   yaml.safe_dump(pinned, sort_keys=False))
        print(json.dumps({k: report[k] for k in ("variant", "answer_continuation",
                                                 "answer_token_ids", "prompts_checked",
                                                 "prompt_digest")}, sort_keys=True))
        return 0
    except (CompatibilityError, BehavioralPlanError, ModelNotCached, OSError, KeyError,
            ValueError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
