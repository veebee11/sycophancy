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
run apart from a failed or incomplete one, which never has the marker.

An exact tie in a **post-counterargument** reading is different: it is retained
for the primary continuous outcome. Its exact A and B logits are kept, its
`m_after` is exactly 0 by the unchanged formula, and
`movement_toward_counter = m_after - m_before` is computed normally. Only the
secondary categorical outcome is undefined, so the row has `post_exact_tie: true`,
`final_label: null` and `flip: null` and is omitted from flip-rate calculations
alone (`evidence.flip_rate_rows`); it stays in every movement analysis
(`evidence.movement_rows`). A post tie never reduces the behavioural-score count.
Each one is listed in `post_ties.jsonl` with reason `exact_post_logit_tie`, and
`RUN_METADATA.json` records the count, the reason counts and the file hash. A
finished run with post ties is labelled `complete_with_recorded_secondary_ties`,
or `complete_with_recorded_exclusions_and_secondary_ties` when initial ties were
also excluded. `run_status` confirms every post-tie row has equal post logits,
`m_after = 0`, null final label and flip, and a valid movement. As with initial
ties, no epsilon, random choice, precision, seed or tie-break is applied. With
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

## Behavioural-result analysis and Base robustness (2026-10-05)

**Evidence.** Both full runs passed `run_status`, every recorded hash and count, and an
exact row-by-row re-scoring. They are archived read-only, with the pinned config and
compatibility reports, at `/data/vidhi/archive/reasonstyle/behavioral_v3_full_evidence_v1`
(server) and `runs/archive/behavioral_v3_full_evidence_v1` (Mac); `SHA256SUMS` hash
`71751a24…`. The pinned config is copied, byte-identical, to the Mac's `configs/`.

**Robustness variants.** Predeclared in `configs/robustness/base_prompt_variants_v1.yaml`:
R1 `ab_bfirst` (B line displayed first), R2 `numeric_12` (labels 1/2), R3 `minimal_ab`
(no User:/Assistant: scaffold). `src/reasonstyle/behavioral/robustness.py` re-renders the
reference plan with one component changed and keeps slot labels A/B in every run file.
`scripts/behavioral_v3_robustness_gate.py` checks every one of the 8,880 prompts with the
cached tokenizer and pins a separate config; R2 was refused (`" 1"` is two tokens). The
variant runner `scripts/run_behavioral_v3_robustness.py` first reproduced all 240
reference Base initial logits bit for bit, then ran each variant's initial phase and
full adaptive phase. Archive: `/data/vidhi/archive/reasonstyle/behavioral_v3_robustness_evidence_v1`
and `runs/archive/behavioral_v3_robustness_evidence_v1`.

```bash
.venv/bin/python scripts/behavioral_v3_robustness_gate.py --variant ab_bfirst --hf-home "$HF_HOME"
REASONSTYLE_ALLOW_BEHAVIORAL_EVAL=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_behavioral_v3_robustness.py \
  --variant-config configs/robustness/base_ab_bfirst_v1.pinned.yaml --phase initial \
  --hf-home "$HF_HOME" --out runs/behavioral_v3_robustness/base-ab-bfirst-v1.initial
```

**Analysis.** Specification `configs/analysis/behavioral_v3_detailed_v1.yaml` (frozen
before results; robustness hashes pinned after those runs passed `run_status`). Code:
`src/reasonstyle/behavioral/analysis/`, `scripts/analyze_behavioral_v3.py`. Output:
`analysis/behavioral_v3_detailed/` (report, tables, figures, bootstrap, robustness,
review, manifest). The estimand averages the two RS and two NS marker rows within each
model × scenario × order × opening block, keeps RP and NP once, averages blocks within
`decision_id` and decisions equally; inference is a 9,999-replicate domain-stratified
decision-cluster bootstrap (NumPy PCG64, seed 20261005).

**Analysis environment.** NumPy and matplotlib are needed only here, so they live in a
separate environment pinned with hashes in `requirements/analysis.txt`; `pyproject.toml`,
`uv.lock` and the GPU `.venv` are unchanged.

```bash
uv venv .venv-analysis --python 3.11
uv pip install --python .venv-analysis -r requirements/analysis.txt --require-hashes
uv pip install --python .venv-analysis --no-deps -e .
.venv-analysis/bin/python scripts/analyze_behavioral_v3.py configs/analysis/behavioral_v3_detailed_v1.yaml
.venv-analysis/bin/python -m pytest --noconftest tests/test_behavioral_v3_analysis.py tests/test_behavioral_v3_robustness.py
```

The analysis tests skip in the main environment (no NumPy); `--noconftest` avoids the
project conftest's pydantic import in the analysis environment.
