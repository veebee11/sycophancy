#!/usr/bin/env python3
"""Inspect one exact cached Llama checkpoint without a network connection.

This gate verifies the resolved commit, renders representative v3 prompts, and
proves that A and B are distinct single-token next-token continuations. It
does not run model weights, produce behavioural observations, or edit config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from reasonstyle.behavioral.compatibility import (
    CompatibilityError,
    materialize_base,
    materialize_instruct,
    resolve_answer_tokens,
)
from reasonstyle.behavioral.plan import build_plan, load_spec
from reasonstyle.generation.environment import ModelNotCached, resolve_cached_model
from reasonstyle.hashing import sha256_of


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True, choices=("base", "instruct"))
    parser.add_argument("--hf-home", help="local Hugging Face cache root")
    parser.add_argument("--out", help="write the JSON report atomically")
    args = parser.parse_args(argv)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        spec = load_spec(args.config)
        plan = build_plan(spec)
        configured = spec.raw["models"][args.variant]
        cached = resolve_cached_model(configured["repo_id"], hf_home=args.hf_home)
        try:
            from transformers import AutoTokenizer, __version__ as transformers_version
        except ImportError as exc:
            raise CompatibilityError(
                "transformers is not installed in this environment; run this gate in the "
                "pinned GPU environment") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            cached.snapshot_path, local_files_only=True)
        sample_rows = (plan.initials[0], plan.initials[-1],
                       plan.candidates[0], plan.candidates[-1])
        rendered: list[str] = []
        resolutions = []
        for row in sample_rows:
            if args.variant == "base":
                prompt = materialize_base(
                    row["transcript"], answer_cue=spec.raw["prompt"]["answer_cue"],
                    template=spec.raw["prompt"]["base_plain_dialogue_v1"])
            else:
                prompt = materialize_instruct(
                    row["transcript"], answer_cue=spec.raw["prompt"]["answer_cue"],
                    tokenizer=tokenizer)
            rendered.append(prompt)
            resolutions.append(resolve_answer_tokens(tokenizer, prompt))
        first = resolutions[0]
        if any(result != first for result in resolutions[1:]):
            raise CompatibilityError("A/B token resolution changes across prompt types")
        chat_template = getattr(tokenizer, "chat_template", None)
        report = {
            "compatibility_status": "passed",
            "behavioral_config_sha256": spec.content_hash,
            "stimuli_sha256": spec.raw["dataset"]["stimuli_sha256"],
            "variant": args.variant,
            "repo_id": cached.repo_id,
            "revision": cached.revision,
            "snapshot_path": cached.snapshot_path,
            "transformers_version": transformers_version,
            "tokenizer_class": tokenizer.__class__.__name__,
            "chat_template_sha256": sha256_of(chat_template) if chat_template else None,
            **first.as_dict(),
            "representative_prompts_checked": len(rendered),
            "representative_prompt_sha256": [sha256_of(prompt) for prompt in rendered],
            "network_access": False,
            "model_forward_pass": False,
        }
        rendered_report = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.out:
            destination = Path(args.out)
            if destination.exists():
                raise CompatibilityError(f"refusing to overwrite {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_text(rendered_report, encoding="utf-8")
            temporary.replace(destination)
        print(rendered_report, end="")
        return 0
    except (CompatibilityError, ModelNotCached, OSError, KeyError, TypeError,
            ValueError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        print("Nothing was downloaded and no model forward pass was made.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
