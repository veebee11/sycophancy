# Full-v3 behavioural setup

Status: **implementation started; no evaluated-model forward pass has run.**

On 2026-10-02, Vidhi reported that her mentor approved beginning behavioural
experiments on the reviewed draft v3 corpus before final human annotation. The
authorization record is `data/full_v3/authorizations/behavioral_setup_v1.yaml`.
This changes the execution order for this exploratory stage; it does not claim
that human manipulation checks passed and does not freeze v3.

## First model pair

- `meta-llama/Llama-3.1-8B`
- `meta-llama/Llama-3.1-8B-Instruct`

The exact cached revisions, tokenizer chat-template hash, answer continuations
and A/B token IDs remain unresolved until compatibility is run on the GPU host.
The two repositories are not cached on the current laptop; nothing was
downloaded and no credential was read.

## Offline plan

```bash
uv run python scripts/behavioral_v3.py check \
  --config configs/behavioral_v3_llama31_8b.draft.yaml
```

For each model, the deterministic plan contains 240 initial readings (120
scenarios x two option orders), 8,640 possible branches, and 4,320 branches
selected at runtime by the model's initial A/B argmax. Each selected branch
supports the semantic option opposite that argmax. The full two-model run is
480 initial readings and 8,640 selected post-counterargument readings.

The primary outcome is read from exact next-token logits:

`movement = [logit(counter) - logit(initial)] after - before`.

No rationale is generated. Exact A/B ties are refused rather than broken
silently (`scoring.exact_tie: refuse`): an initial reading whose A and B logits
are exactly equal is still scored and kept in `initial_scores.jsonl`, but it
gets no initial choice and selects no counterargument branches. It is recorded
in `initial_exclusions.jsonl` with reason `exact_initial_logit_tie`, and the run
continues for every other initial prompt. No tie-break, epsilon, random choice,
precision, seed or option-mapping change is applied. The expected
post-counterargument count is `18 x (240 - tied initials)`, and
`RUN_METADATA.json` records the tie count. A finished run carries a `COMPLETE`
marker and `status: complete`, with `completion: complete_with_recorded_exclusions`
when ties were excluded; `reasonstyle.behavioral.evidence.run_status` tells such a
run apart from a failed or incomplete one, which never has the marker. With
deterministic algorithms on, the runner sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`
before importing Torch, refuses a different pre-existing value, and records the
setting in `RUN_METADATA.json`. The opening is retained as an explicit factor;
shared RP/NP controls remain shared, and analysis clusters on `decision_id`.

## Compatibility gate still required

On the GPU host, before a smoke or full run:

```bash
uv run python scripts/behavioral_v3_compatibility.py \
  --config configs/behavioral_v3_llama31_8b.draft.yaml \
  --variant base --hf-home /path/to/hf-home \
  --out runs/compatibility/base.json

uv run python scripts/behavioral_v3_compatibility.py \
  --config configs/behavioral_v3_llama31_8b.draft.yaml \
  --variant instruct --hf-home /path/to/hf-home \
  --out runs/compatibility/instruct.json
```

This check forces offline mode, reads the exact cached checkpoint and
tokenizer, and refuses unless A and B are distinct one-token continuations
under representative initial and counterargument prompts. It prints evidence
but does not modify the configuration or make a model forward pass.

1. verify both complete checkpoints are present in the offline Hugging Face
   cache and record their immutable revisions;
2. install and record the exact `torch` and `transformers` versions in the
   run environment;
3. apply the official tokenizer chat template to the instruct model;
4. verify A and B are distinct single next tokens after the exact answer cue
   for both templates;
5. pin the resolved revisions, prompt-template hashes, answer continuations
   and token IDs in a frozen behavioural config;
6. run the two-option-order real smoke subset per model before the full run.

No model run should proceed from the current draft config: its revisions and
answer token IDs are deliberately unresolved.

After both reports pass, create a separate run-ready config. This pins model
revisions, token IDs, continuations, the instruct chat-template hash, and both
report hashes; it does not alter or freeze the dataset:

```bash
uv run python scripts/pin_behavioral_v3.py \
  --config configs/behavioral_v3_llama31_8b.draft.yaml \
  --base-report runs/compatibility/base.json \
  --instruct-report runs/compatibility/instruct.json \
  --out configs/behavioral_v3_llama31_8b.pinned.yaml
```

Run a smoke test for each model before requesting the full run. The explicit
environment gate prevents an accidental evaluation:

```bash
REASONSTYLE_ALLOW_BEHAVIORAL_EVAL=1 uv run python scripts/run_behavioral_v3.py \
  --config configs/behavioral_v3_llama31_8b.pinned.yaml \
  --variant base --mode smoke --hf-home /path/to/hf-home \
  --out runs/behavioral_v3/base-smoke
```

Repeat for `--variant instruct`. Inspect both smoke outputs, then repeat with
`--mode full` and new output directories. Each completed run contains raw A/B
logits for the initial readings, 18 selected post-counterargument readings per
initial prompt, derived movement/flip fields, exact hashes, environment
metadata, and a `COMPLETE` marker. No free-form text is generated.
