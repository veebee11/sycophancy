# Current status

*Updated 2026-09-16. Kept short; history is in git, rules in `design_notes.md`.*

## Done

- **Design and configuration.** Four frozen conditions (RS, RP, NS, NP), contrasts, marker inventory, matching and validation rules in `configs/experiment.yaml`, checked at load. The research-plan hash is pinned.
- **Corpus tooling.** Schemas, pinned segmenter, validator, annotation schemas, prompt renderer, deterministic review export.
- **Sources.** Registry with checked outcomes; POLIANNA and the GenAI4PA JRC snapshot are downloaded and verified. IBM-ArgQ is citable but optional.
- **Topic bank.** 12 curated **pilot** decisions (4 climate, 4 energy, 4 technology), 2 variants each. No machine errors; the overlap screen has been run. They are the first 12 of the 60-decision main corpus (see *Corpus scope*).
- **Marker allocation.** 48 groups, 16 per confirmatory family, 8 per string.
- **Generation client.** Hashed prompt templates and request/response files. Local vLLM backend with two-key authorisation, offline enforcement, a read-only cache preflight, three-way revision agreement, the server launcher and the one-group smoke test, which has been run live twice, draft-only, on 2026-09-15 and 2026-09-16 (see below). The Anthropic backend has been removed.
- **Launcher fixes after the first server attempts (2026-09-15).** Library versions are read from package metadata instead of by import; setup installs `setuptools==79.0.1`; the runtime record is published once the launcher's own child first answers `/health`, kept while that process is alive, and removed on exit, and a port that already answers `/health` is refused.
- **Corrections after the first live smoke call, exercised by the second (2026-09-16).** Both drafting prompts now state that a body excludes the shared opening, that all four bodies share one endorsement clause, that each pair must keep the same content words, what NS and NP may not say, and that the four word counts are checked before returning. The validator gained `E_OPENING_REPEATED_IN_BODY` and the lexical pair screen `E_PAIR_CONTENT_DRIFT`. The smoke test now persists its validation result, records the codes, prints `stop_reason`, labels human-review counts by scope, and no longer logs a validation failure as an accepted result. The second live smoke confirmed the effect: neither the repeated opening nor the pair drift recurred.
- **Reliability sampler.** Stratified quotas plus an exactly optimal marginal opt_1/opt_2 balance; any shortfall is reported (2026-09-15).
- **No hidden sampling defaults.** The launcher passes `--generation-config vllm` and records it; all sampling fields are sent at explicit neutral values.
- **Scenario drafting (2026-09-16).** The scenario stage **has run live**, inside `pipeline_smoke.py`, and its scenario was accepted. The standalone `smoke_test.py --kind scenario` command is implemented and offline-tested but has not itself been run.
- **Bounded repair controller (2026-09-16).** One draft plus at most two validator-driven repairs, then `needs_manual_review`; recorded per call, resumable, transport failures retryable. It **has run live once**, through `scripts/pipeline_smoke.py`: the group exhausted the four-call ceiling and ended `needs_manual_review` because both repair requests were identical (see below). The no-progress correction is implemented and offline-tested; **it has not yet been confirmed by another live smoke**.
- **`scripts/pilot.py` (2026-09-16).** Plan and status only. It has no live code path at any argument or environment combination, and `--send` refuses.
- **Tests.** The full suite passes on a laptop, with no server.

## Corpus scope (settled 2026-09-16)

- The **12 decisions × 2 variants** now being generated are the **pilot corpus**: 24 scenarios, 48 groups, 192 counterargument texts.
- The **main corpus is 60 policy decisions** — 20 climate, 20 energy, 20 technology — giving 120 scenarios, 240 groups and 960 texts, as in Research_Plan_v6 §5 and `configs/experiment.yaml`.
- **192 texts are 12 independent policy decisions, not 192.** One decision yields 16 texts; the independence unit is `decision_id`.
- **The pilot's 12 count toward the 60** if they pass the final frozen specification and review criteria, so the normal remainder is **48 additional decisions, not 60**. One that cannot qualify is regenerated under the final procedure or replaced; the total stays 60.
- **Decision-level splits stay 36/12/12**, and the **mechanistic subset is 40 of the 60** — its count is fixed, and its selection rule is documented before mechanistic analysis begins.
- **Evaluated-model runs wait** until the complete 60-decision corpus is validated, human-reviewed and frozen.
- **Any change from 60 is a documented design amendment**, made before any main evaluated-model outcome is examined.

The order this implies:

1. **Now — engineering.** Generate and human-review the 12-decision pilot.
2. **Then — production.** Build the remaining decisions to reach 60, then validate, human-review and freeze the whole corpus.
3. **Only then.** Run the main behavioural evaluation.
4. **After that.** Mechanistic analysis over 40 of the 60, by the documented selection rule.

## Resolved

- **Reliability sample size (2026-09-15).** The implemented rule stays: `round(N × 0.20)`. Pilot: 38 items covering all 36 item strata, and 19 pairs covering all 18 pair strata. The older 48-item / 24-pair statement is withdrawn.

## Not yet built

- **Curator-approval gate:** every scenario approved against its exact text hash before any group is drafted from it. Required before pilot generation.
- **Corpus assembler:** accepted drafts into `data/pilot/corpus.jsonl`, then full validation and a pilot review export. Required before pilot generation.
- **The remaining decisions to reach 60** (normally 48 more), then validation, human review and the corpus freeze.
- **The mechanistic subset's selection rule** — 40 of the 60 — documented before mechanistic analysis.
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

**Decision taken then, since carried out.** Do not run another draft-only smoke
test: a single draft with no repair path cannot show whether the remaining
defects are recoverable, and every further draft-only call spends GPU time on a
question already answered. The repair-enabled path named as the next gate was
run on 2026-09-16; its result is the section below.

## Live synthetic repair smoke (2026-09-16)

The first run of the repair path, on the synthetic fixture. Evidence, read-only
and gitignored, in `data/pilot/smoke/pipeline_repair_2026-09-16_attempt2/`.

- **The scenario was accepted** on its single call.
- **The group draft and both repairs all returned the same three errors**: `E_SENTENCE_COUNT_MISMATCH`, `E_WORD_RATIO_BODY`, `E_WORD_RATIO_FULL_TEXT`.
- **4 of 4 permitted calls were used** — one scenario, one draft, two repairs.
- **Final outcome `needs_manual_review`.** Nothing was hand-corrected.
- **No pilot generation occurred.**
- **The correction target: both repair requests were byte-identical**, prompt hash `bcc545c175bbec26848fe1f3d10148b8631454385ea79c6708d03e1c43f2cb5c`. The second repair asked a deterministic server a question it had already answered, so the model returned the same four bodies and the budget ran out without new information ever reaching it.

The bodies showed the shape of the problem: RS and NS were one sentence, RP and
NP two, because dropping `because` split the plain member of each pair into two
sentences. The finding reported full-text counts of 2/3 while the rule is two
**body** sentences, which the prompt never made explicit.

**Status: corrected offline, not yet reconfirmed live.** The fix below is
implemented and covered by regression tests built on this run's verbatim bodies,
but no live call has exercised it. **Exactly one confirmation repair-path smoke
is the next live gate.**

**The fix (2026-09-16, offline):** a repair now carries its attempt number, the
history of what earlier attempts did, and the measurements behind the findings —
per-condition body and full-text sentence and word counts, the required count,
the allowed range from the configured ratio, and which conditions to shorten or
lengthen. A repair returning unchanged bodies is recorded as making no progress,
and the next one is told so and asked for a materially different correction. A
repair request that would exactly repeat its predecessor is refused rather than
sent. The budget, the thresholds, the validators and the seed are unchanged, and
an exhausted group is still `needs_manual_review`.

## Offline pipeline (2026-09-16)

Four separate states, deliberately kept apart:

**1. The controller is implemented, offline-tested, and has run live once.**
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
Developed against `FakeBackend`, and exercised live once on 2026-09-16 through
`scripts/pipeline_smoke.py` (see *Live synthetic repair smoke* above).

**2. That first repair smoke ended `needs_manual_review`, and the correction
awaits one confirmation smoke.** The scenario was accepted; the group used all
four permitted calls and stopped unrepaired, because both repair requests were
byte-identical. The no-progress logic that fixes it — attempt number, history,
measured diagnostics, and a refusal to repeat a repair request — is implemented
and covered by regression tests on that run's verbatim bodies, but **no live
call has exercised it yet**. Exactly one confirmation run of
`scripts/pipeline_smoke.py` is the next live gate: the fixture decision
`energy_fixture_001`, one scenario, one `opt_1` group, at most four calls, its
own output directory, and the same pre-flight checks as the one-call smoke
test. It needs `--send`, `REASONSTYLE_ALLOW_LOCAL_GENERATION=1` and offline
mode, and no pilot authorisation.

**3. Real pilot generation is hard-disabled.**
`scripts/pilot.py` plans and reports only. It has no live code path at any
argument or environment combination, and `--send` refuses.

**4. Two things must exist before pilot generation is written at all:** the
curator-approval gate (every scenario approved against its exact text hash
before any group is drafted from it) and the corpus assembler.

## Next steps

1. **Review and commit the offline controller**, then bring the server checkout up to date. The prepared environment works: two live smoke calls have since launched and completed on it.
2. **Authorise one confirmation repair-path smoke test** on one announced GPU: `scripts/pipeline_smoke.py` again, at most four calls, on `energy_fixture_001`, to check that a repair now receives new information and can make progress. Stop there.
3. **Review that result**, and only then decide whether the repair path is sound enough to build the curator-approval gate and the corpus assembler on.
4. **Pilot generation** stays hard-disabled until both of those exist and are separately authorised.
5. **After the pilot is reviewed**, extend to the full 60 decisions, then validate, human-review and freeze before any evaluated-model run.

## Open, not resolved

- **Compute.** The A6000 is for generation and possibly the first Llama-3.1-8B compatibility and behavioural tests. Larger causal sweeps may need Wisteria or an A100-class GPU, depending on measurements not yet taken.
- **Family overlap.** If an optional Qwen model is evaluated, it shares a family with the corpus generator. That would be disclosed as a limitation.
