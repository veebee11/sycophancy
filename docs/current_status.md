# Current status

*Updated 2026-09-16. Kept short; history is in git, rules in `design_notes.md`.*

## Done

- **Design and configuration.** Four frozen conditions (RS, RP, NS, NP), contrasts, marker inventory, matching and validation rules in `configs/experiment.yaml`, checked at load. The research-plan hash is pinned.
- **Corpus tooling.** Schemas, pinned segmenter, validator, annotation schemas, prompt renderer, deterministic review export.
- **Sources.** Registry with checked outcomes; POLIANNA and the GenAI4PA JRC snapshot are downloaded and verified. IBM-ArgQ is citable but optional.
- **Topic bank.** 12 curated pilot decisions (4 climate, 4 energy, 4 technology), 2 variants each. No machine errors; the overlap screen has been run.
- **Marker allocation.** 48 groups, 16 per confirmatory family, 8 per string.
- **Generation client.** Hashed prompt templates and request/response files. Local vLLM backend with two-key authorisation, offline enforcement, a read-only cache preflight, three-way revision agreement, the server launcher and the one-group smoke test, which has been run live twice, draft-only, on 2026-09-15 and 2026-09-16 (see below). The Anthropic backend has been removed.
- **Launcher fixes after the first server attempts (2026-09-15).** Library versions are read from package metadata instead of by import; setup installs `setuptools==79.0.1`; the runtime record is published once the launcher's own child first answers `/health`, kept while that process is alive, and removed on exit, and a port that already answers `/health` is refused.
- **Corrections after the first live smoke call, exercised by the second (2026-09-16).** Both drafting prompts now state that a body excludes the shared opening, that all four bodies share one endorsement clause, that each pair must keep the same content words, what NS and NP may not say, and that the four word counts are checked before returning. The validator gained `E_OPENING_REPEATED_IN_BODY` and the lexical pair screen `E_PAIR_CONTENT_DRIFT`. The smoke test now persists its validation result, records the codes, prints `stop_reason`, labels human-review counts by scope, and no longer logs a validation failure as an accepted result. The second live smoke confirmed the effect: neither the repeated opening nor the pair drift recurred.
- **Reliability sampler.** Stratified quotas plus an exactly optimal marginal opt_1/opt_2 balance; any shortfall is reported (2026-09-15).
- **No hidden sampling defaults.** The launcher passes `--generation-config vllm` and records it; all sampling fields are sent at explicit neutral values.
- **Scenario-only smoke path (2026-09-16).** `smoke_test.py --kind scenario` drafts one scenario from the synthetic fixture brief and checks it with `validate_scenario_text`. Implemented and offline-tested; **not yet run live**.
- **Bounded repair controller (2026-09-16).** One draft plus at most two validator-driven repairs, then `needs_manual_review`; recorded per call, resumable, transport failures retryable. Implemented and offline-tested against `FakeBackend`; **not yet run live**. `scripts/pipeline_smoke.py` is its synthetic four-call gate.
- **`scripts/pilot.py` (2026-09-16).** Plan and status only. It has no live code path at any argument or environment combination, and `--send` refuses.
- **Tests.** The full suite passes on a laptop, with no server.

## Resolved

- **Reliability sample size (2026-09-15).** The implemented rule stays: `round(N × 0.20)`. Pilot: 38 items covering all 36 item strata, and 19 pairs covering all 18 pair strata. The older 48-item / 24-pair statement is withdrawn.

## Not yet built

- **Curator-approval gate:** every scenario approved against its exact text hash before any group is drafted from it. Required before pilot generation.
- **Corpus assembler:** accepted drafts into `data/pilot/corpus.jsonl`, then full validation and a pilot review export. Required before pilot generation.
- Model adapter, answer-token verification, logit scoring and every later analysis and mechanistic stage.

## Server (Chomusuke02), as of 2026-09-15

All commands were run by Vidhi.

- The account was created, and SSH access through the gateway succeeded.
- `/data/vidhi` exists. The host exposes four RTX A6000 48 GB GPUs.
- An isolated Python 3.11 environment was created under `/data/vidhi/reasonstyle`.
- Recorded versions: vLLM `0.8.5.post1+cu118`, Transformers `4.51.3`, tokenizers `0.21.4`, PyTorch `2.6.0+cu118`, huggingface-hub `0.36.2`.
- `Qwen/Qwen3-14B` revision `40c069824f4251a91eefaf281ebe4c544efd3e18` was downloaded under `/data/vidhi/hf_cache`.
- The offline preflight verified all eight weight shards, 29.6 GB.
- The revision was recorded in the generator configuration. In this repository it is set in `configs/experiment.yaml`, and the fixture corpus and marker allocation were regenerated with the existing scripts to carry the new configuration hash; the marker assignments themselves are unchanged.
- **Two launcher attempts, both failed.**
  1. The first failed while building the runtime record: importing vLLM wrote to stdout and corrupted the code that writes the record.
  2. The second reached engine initialisation and failed because `setuptools` was absent.
- **The first two launcher attempts produced no endpoint.** The runtime record left by the second attempt is invalid and must not be used; the corrected launcher removes stale records before it starts.

## First live smoke call (2026-09-15)

One synthetic group, `energy_fixture_001` v1 / `opt_1`, from the fixture brief.
One call, run by Vidhi on 2026-09-15. The raw request, response and log line are
kept read-only and gitignored under `data/pilot/smoke/server_2026-09-15/`. The
corrections described here were made on 2026-09-16.

**What worked.** The call completed. The model, the resolved revision and the
server's runtime record all named the same commit, the response parsed against
the group schema, and `stop_reason` was `stop`: nothing was truncated.

**What the machine found.** Two errors — `E_WORD_RATIO_FULL_TEXT` and
`E_WORD_RATIO_BODY` — and two `W_PREMISE_NOT_IN_SCENARIO` warnings.

**What a person found**, reading the four bodies:

- Every body repeated the shared opening, which the renderer adds separately.
- RS and RP did not preserve their propositions: RS carried "extending its
  operating life", RP dropped it and changed the grammatical subject.
- NS carried a proposition NP did not, and that proposition — the option
  "maintains consistent performance" — is itself a possible policy benefit, so
  the no-reason cells were not reason-free.

The first defect had no machine check at the time; the second and third are
human judgements (`H_PROPOSITION_PRESERVATION`, `H_NO_REASON_INTEGRITY`) that a
person made. Two of the three now also have a conservative lexical screen. What
the machine catches is an exact, normalised repetition of the opening and a
difference in content words within a pair; a *paraphrased* opening, and whether
a pair means the same thing, remain human judgements. The two
`W_PREMISE_NOT_IN_SCENARIO` warnings stay warnings at the same threshold and
will be inspected during the next smoke test.

**Provenance defect.** The log line recorded `status: ok` although validation
had produced errors, and the validator output existed only in the terminal.
Both are corrected.

**No repair, no retry and no pilot generation took place.**

## Second live smoke call (2026-09-16)

One synthetic group again, draft-only, after the corrections above. Run by
Vidhi; one call, no retry and no repair.

| | |
|---|---|
| call id | `05a54889c0b4f92ae28a6740972e19b9e4f925ed99177fcaf23d8374f3e4b366` |
| prompt hash | `14df854e5228c7807287fa89150a1d96cfd287ec45084904f9bbb51281c3daae` |
| model revision | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| `stop_reason` | `stop` |
| saved status | `validation_failed` |

**Machine findings.** Two errors, `E_SENTENCE_COUNT_MISMATCH` and
`E_WORD_RATIO_BODY`, and one warning, `W_WORD_RATIO_FULL_TEXT`.

**What the corrections fixed.** The repeated shared opening and the
pair-content drift did not recur, and neither did the premise-containment
warning.

**What human review found.** RS reversed the intended inferential direction,
and NS and NP used unnatural tautological repetition. Both are human
judgements; no machine rule was added for either, and none is planned.

**Decision.** Do not run another draft-only smoke test. A single draft with no
repair path cannot show whether the remaining defects are recoverable, and
every further draft-only call spends GPU time on a question already answered.
The next live gate is the repair-enabled path: one scenario call, then one
group draft with up to two validator-driven repairs.

## Offline pipeline (2026-09-16)

Four separate states, deliberately kept apart:

**1. The controller is implemented and tested offline.**
`generation/pipeline.py`: one scenario call, one group draft plus at most two
validator-driven repairs, stopping as soon as a group is machine-valid or the
budget is spent. A scenario that fails its machine checks stops its own groups
being drafted. Every call records its attempt number, the full vLLM request
fields, the raw response, the prompt and response hashes, the revision,
snapshot, GPU, dtype, library versions, seed, offline state, topic-bank and
allocation hashes, the stop reason, token usage and the validation codes. A
restart re-reads a completed call instead of sending it again, re-parsing a
stored response under its own schema, so a truncated or malformed one recovers
as `rejected` and never as an accepted draft. A transport failure is logged as
an aborted transport event, consumes no budget position, and is retried on the
next run. `validate_group` and `validate_scenario_text` are shared by drafting
and corpus validation, so a draft is held to the rules it will face later.
Tested only against `FakeBackend`: **nothing has been sent**.

**2. The synthetic repair-path smoke has not been run yet.**
`scripts/pipeline_smoke.py` is the next live gate: the fixture decision
`energy_fixture_001`, one scenario, then one `opt_1` group with its repair
path — at most four calls, its own output directory, and the same pre-flight
checks as the one-call smoke test. It needs `--send`,
`REASONSTYLE_ALLOW_LOCAL_GENERATION=1` and offline mode, and no pilot
authorisation.

**3. Real pilot generation is hard-disabled.**
`scripts/pilot.py` plans and reports only. It has no live code path at any
argument or environment combination, and `--send` refuses.

**4. Two things must exist before pilot generation is written at all:** the
curator-approval gate (every scenario approved against its exact text hash
before any group is drafted from it) and the corpus assembler.

## Next steps

1. **Review and commit the offline controller**, then bring the server checkout up to date. The prepared environment works: two live smoke calls have since launched and completed on it.
2. **Authorise one live synthetic repair-path smoke test** on one announced GPU: `scripts/pipeline_smoke.py`, at most four calls, on `energy_fixture_001`. Stop there.
3. **Review that result**, and only then decide whether the repair path is sound enough to build the curator-approval gate and the corpus assembler on.
4. **Pilot generation** stays hard-disabled until both of those exist and are separately authorised.

## Open, not resolved

- **Compute.** The A6000 is for generation and possibly the first Llama-3.1-8B compatibility and behavioural tests. Larger causal sweeps may need Wisteria or an A100-class GPU, depending on measurements not yet taken.
- **Family overlap.** If an optional Qwen model is evaluated, it shares a family with the corpus generator. That would be disclosed as a limitation.
