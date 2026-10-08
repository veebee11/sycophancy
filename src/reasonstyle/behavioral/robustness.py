"""Prompt-format variants for the Base robustness experiment (no model import).

Each variant re-renders the exact reference plan: the same 240 initial prompts,
the same 8,640 candidate branches, the same slot labels A/B and the same
``label_to_option`` mappings. Only three things can change, one per variant:
the displayed order of the option lines, the response labels the model reads
and answers with, or the scaffold around the transcript. Slot labels stay A/B
in every run file so runtime selection, scoring and ``run_status`` are reused
unchanged; the slot-to-response map is recorded alongside.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..hashing import canonical_json, content_hash, file_sha256, sha256_of
from .compatibility import CompatibilityError, materialize_base
from .plan import BehavioralPlan, BehavioralPlanError, BehavioralSpec

SLOTS = ("A", "B")


@dataclass(frozen=True)
class PromptVariant:
    name: str
    run_id: str | None
    response_labels: dict[str, str]
    display_order: tuple[str, str]
    template_name: str
    template: dict[str, Any]

    def question(self, spec: BehavioralSpec) -> str:
        p = spec.raw["prompt"]
        a, b = self.response_labels["A"], self.response_labels["B"]
        return f'{p["question_text"]} Answer with {a} or {b} only.'

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "run_id": self.run_id,
                "response_labels": dict(self.response_labels),
                "display_order": list(self.display_order),
                "template_name": self.template_name, "template": dict(self.template)}

    @property
    def definition_hash(self) -> str:
        return content_hash(self.as_dict())


def load_variants(path: str | Path) -> tuple[dict[str, Any], dict[str, PromptVariant]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, cfg in raw["variants"].items():
        labels = {k: str(v) for k, v in cfg["response_labels"].items()}
        order = tuple(cfg["display_order"])
        if set(labels) != set(SLOTS) or labels["A"] == labels["B"]:
            raise BehavioralPlanError(f"{name}: response labels must map A and B distinctly")
        if sorted(order) != list(SLOTS):
            raise BehavioralPlanError(f"{name}: display order must be a permutation of A, B")
        template = raw["templates"][cfg["template"]]
        out[name] = PromptVariant(name=name, run_id=cfg["run_id"], response_labels=labels,
                                  display_order=order, template_name=cfg["template"],
                                  template=template)
    return raw, out


def _scenarios(spec: BehavioralSpec) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    stimuli: dict[str, str] = {}
    for line in spec.stimuli_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out.setdefault(row["scenario_id"], {"scenario_text": row["scenario_text"],
                                            "options": row["options"]})
        stimuli[row["stimulus_id"]] = row["rendered"]
    out["__rendered__"] = stimuli
    return out


def _option_lines(variant: PromptVariant, spec: BehavioralSpec, options: dict[str, str],
                  label_to_option: dict[str, str]) -> str:
    fmt = spec.raw["prompt"]["option_line_format"]
    return "\n".join(fmt.format(label=variant.response_labels[slot],
                                option_text=options[label_to_option[slot]])
                     for slot in variant.display_order)


def render_initial(variant: PromptVariant, spec: BehavioralSpec, scenario: dict[str, Any],
                   label_to_option: dict[str, str]) -> str:
    lines = _option_lines(variant, spec, scenario["options"], label_to_option)
    question = variant.question(spec)
    if variant.template["kind"] == "dialogue":
        turn_one = "\n\n".join((scenario["scenario_text"], lines, question))
        return materialize_base([{"role": "user", "content": turn_one}],
                                answer_cue=spec.raw["prompt"]["answer_cue"],
                                template=spec.raw["prompt"]["base_plain_dialogue_v1"])
    return variant.template["initial"].format(
        scenario_text=scenario["scenario_text"], options=lines, question=question)


def render_branch(variant: PromptVariant, spec: BehavioralSpec, scenario: dict[str, Any],
                  label_to_option: dict[str, str], initial_slot: str,
                  counterargument: str) -> str:
    lines = _option_lines(variant, spec, scenario["options"], label_to_option)
    question = variant.question(spec)
    answer = variant.response_labels[initial_slot]
    if variant.template["kind"] == "dialogue":
        turn_one = "\n\n".join((scenario["scenario_text"], lines, question))
        transcript = [{"role": "user", "content": turn_one},
                      {"role": "assistant", "content": answer},
                      {"role": "user", "content": "\n\n".join((counterargument, question))}]
        return materialize_base(transcript, answer_cue=spec.raw["prompt"]["answer_cue"],
                                template=spec.raw["prompt"]["base_plain_dialogue_v1"])
    return render_initial(variant, spec, scenario, label_to_option) + \
        variant.template["continuation"].format(
            initial_answer=answer, counterargument=counterargument, question=question)


@dataclass(frozen=True)
class VariantPlan:
    variant: PromptVariant
    initials: list[dict[str, Any]]          # plan rows + "prompt", "prompt_sha256"
    candidates: list[dict[str, Any]]

    @property
    def prompt_digest(self) -> str:
        """One hash over every (id, prompt hash) pair, in plan order."""
        pairs = [[r["initial_id"], r["prompt_sha256"]] for r in self.initials]
        pairs += [[r["branch_id"], r["prompt_sha256"]] for r in self.candidates]
        return sha256_of(canonical_json(pairs))


def build_variant_plan(spec: BehavioralSpec, plan: BehavioralPlan,
                       variant: PromptVariant) -> VariantPlan:
    scenarios = _scenarios(spec)
    rendered = scenarios["__rendered__"]
    initials = []
    for row in plan.initials:
        prompt = render_initial(variant, spec, scenarios[row["scenario_id"]],
                                row["label_to_option"])
        initials.append({**{k: v for k, v in row.items() if k != "transcript"},
                         "prompt": prompt, "prompt_sha256": sha256_of(prompt)})
    candidates = []
    for row in plan.candidates:
        prompt = render_branch(variant, spec, scenarios[row["scenario_id"]],
                               row["label_to_option"], row["required_initial_label"],
                               rendered[row["stimulus_id"]])
        candidates.append({**{k: v for k, v in row.items() if k != "transcript"},
                           "prompt": prompt, "prompt_sha256": sha256_of(prompt)})
    return VariantPlan(variant=variant, initials=initials, candidates=candidates)


def resolve_label_tokens(tokenizer: Any, prompt: str, labels: dict[str, str],
                         prefix: str = " ") -> dict[str, int]:
    """Token IDs of ``prefix + label`` for both slots; each continuation must add
    exactly one token without altering the prompt's own tokens, and the two must
    differ. No other whitespace convention or label is tried."""
    base = [int(t) for t in tokenizer.encode(prompt, add_special_tokens=False)]
    if not base:
        raise CompatibilityError("the materialized prompt tokenizes to nothing")
    ids = {}
    for slot in SLOTS:
        combined = [int(t) for t in tokenizer.encode(prompt + prefix + labels[slot],
                                                     add_special_tokens=False)]
        if combined[:-1] != base or len(combined) != len(base) + 1:
            raise CompatibilityError(
                f"{prefix + labels[slot]!r} is not a single-token continuation of the prompt")
        ids[slot] = combined[-1]
    if ids["A"] == ids["B"]:
        raise CompatibilityError("the two response labels resolve to the same token")
    return ids


def compatibility_gate(tokenizer: Any, vplan: VariantPlan, prefix: str = " "
                       ) -> dict[str, Any]:
    """Check every initial and every candidate prompt; one shared token map."""
    first: dict[str, int] | None = None
    checked = 0
    for row in vplan.initials + vplan.candidates:
        ids = resolve_label_tokens(tokenizer, row["prompt"], vplan.variant.response_labels,
                                   prefix)
        if first is None:
            first = ids
        elif ids != first:
            raise CompatibilityError(
                f"token IDs differ across prompts ({row.get('branch_id', row['initial_id'])})")
        checked += 1
    assert first is not None
    labels = vplan.variant.response_labels
    return {
        "compatibility_status": "passed",
        "variant": vplan.variant.name,
        "variant_definition_hash": vplan.variant.definition_hash,
        "answer_continuation": {s: prefix + labels[s] for s in SLOTS},
        "answer_token_ids": first,
        "prompts_checked": checked,
        "initial_prompts_checked": len(vplan.initials),
        "candidate_prompts_checked": len(vplan.candidates),
        "prompt_digest": vplan.prompt_digest,
        "network_access": False,
        "model_forward_pass": False,
    }


def pinned_variant_config(variants_path: Path, variant: PromptVariant, spec: BehavioralSpec,
                          report: dict[str, Any], report_path: Path, revision: str
                          ) -> dict[str, Any]:
    return {
        "robustness_version": yaml.safe_load(variants_path.read_text())["robustness_version"],
        "status": "ready_for_robustness_run",
        "variant": variant.as_dict(),
        "variant_definition_hash": variant.definition_hash,
        "variants_file": {"path": str(variants_path), "sha256": file_sha256(variants_path)},
        "source_config": {"path": str(spec.path.relative_to(spec.root)),
                          "file_sha256": file_sha256(spec.path),
                          "content_hash": spec.content_hash},
        "model": {"repo_id": spec.raw["models"]["base"]["repo_id"], "revision": revision},
        "answer_continuation": report["answer_continuation"],
        "answer_token_ids": report["answer_token_ids"],
        "prompt_digest": report["prompt_digest"],
        "compatibility_report": {"path": str(report_path), "sha256": file_sha256(report_path)},
    }
