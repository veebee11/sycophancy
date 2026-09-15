# Current status

*Updated 2026-09-15. Kept short; history is in git, rules in `design_notes.md`.*

## Done

- **Design and configuration.** Four frozen conditions (RS, RP, NS, NP), contrasts, marker inventory, matching and validation rules in `configs/experiment.yaml`, checked at load. The research-plan hash is pinned.
- **Corpus tooling.** Schemas, pinned segmenter, validator, annotation schemas, prompt renderer, deterministic review export.
- **Sources.** Registry with checked outcomes; POLIANNA and the GenAI4PA JRC snapshot are downloaded and verified. IBM-ArgQ is citable but optional.
- **Topic bank.** 12 curated pilot decisions (4 climate, 4 energy, 4 technology), 2 variants each. No machine errors; the overlap screen has been run.
- **Marker allocation.** 48 groups, 16 per confirmatory family, 8 per string.
- **Generation client.** Hashed prompt templates and request/response files. Local vLLM backend with two-key authorisation, offline enforcement, a read-only cache preflight, three-way revision agreement, the server launcher and the one-group smoke test (dry run only so far). The Anthropic backend has been removed.
- **Reliability sampler.** Stratified quotas plus an exactly optimal marginal opt_1/opt_2 balance; any shortfall is reported (2026-09-15).
- **No hidden sampling defaults.** The launcher passes `--generation-config vllm` and records it; all sampling fields are sent at explicit neutral values.
- **Tests.** The full suite passes on a laptop, with no server.

## Resolved

- **Reliability sample size (2026-09-15).** The implemented rule stays: `round(N × 0.20)`. Pilot: 38 items covering all 36 item strata, and 19 pairs covering all 18 pair strata. The older 48-item / 24-pair statement is withdrawn.

## Not yet built

- Scenario-only smoke path.
- Pilot sender: scenarios before groups, with scenario approval as a gate.
- Bounded repair controller (draft + ≤2 repairs, then `needs_manual_review`).
- Assembly of scenarios and group drafts into corpus JSONL, then full validation and review export of the pilot.
- Model adapter, answer-token verification, logit scoring and every later analysis stage.

A proposal for the offline pipeline items is to be approved before implementation.

## Blocker

Waiting for the lab administrator to create a Chomusuke02 account. No server
has been contacted, no model run and no weights downloaded.
`models.generator.model.revision` is `null` until the preflight resolves it.

## Next steps

1. **Read-only preflight** on Chomusuke02 (run by Vidhi): `HF_HUB_OFFLINE=1 HF_HOME=/data/$USER/hf_cache uv run python scripts/preflight_model.py --config configs/experiment.yaml`.
2. **If it succeeds:** record the resolved commit SHA in `models.generator.model.revision`, start vLLM on one announced GPU, and run the single fixture smoke call.
   **If it fails:** ask the lab about an existing shared model cache. Download nothing without approval.
3. **Offline pipeline:** approve the proposal, then implement and test it with `FakeBackend` before any pilot call.

## Open, not resolved

- **Compute.** The A6000 is for generation and possibly the first Llama-3.1-8B compatibility and behavioural tests. Larger causal sweeps may need Wisteria or an A100-class GPU, depending on measurements not yet taken.
- **Family overlap.** If an optional Qwen model is evaluated, it shares a family with the corpus generator. That would be disclosed as a limitation.
