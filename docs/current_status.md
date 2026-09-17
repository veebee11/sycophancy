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
- **Bounded repair controller (2026-09-16).** One draft plus at most two validator-driven repairs, then `needs_manual_review`; recorded per call, resumable, transport failures retryable. It **ran live twice** through `scripts/pipeline_smoke.py`. The first run's repairs were identical requests; after the correction, the second run sent two distinct, diagnosed repairs and recorded the model's unchanged replies as no progress — **the corrected mechanism was confirmed live**. Both runs ended `needs_manual_review`. No further synthetic repair smoke is planned.
- **Live calls so far: four smoke runs, ten model calls in total** — two draft-only group smokes (1 call each) and two repair-path smokes (4 calls each). No pilot generation.
- **`scripts/pilot.py` (2026-09-16).** The two-stage pilot runner: `scenarios` and `groups` as separate commands that cannot be combined, plus `scenario-review`, `approvals`, `assemble`, `status` and `log`. A live stage needs `--send` **and** both authorisation keys, and runs on the complete pilot only — a subset is refused. Built and offline-tested; **no pilot call has been authorised or made**.
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

- **The scenario `redraft` path.** The gate reports `redraft` and blocks the set on it, but nothing re-drafts a scenario. There is deliberately no automatic redraft budget: it needs an explicit decision.
- **A live pilot run.** The two-stage path is implemented and offline-tested; no pilot call has been authorised or made.
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

**Status: corrected, and the correction was confirmed live** by the run
recorded in the next section.

**The fix (2026-09-16, offline):** a repair now carries its attempt number, the
history of what earlier attempts did, and the measurements behind the findings —
per-condition body and full-text sentence and word counts, the required count,
the allowed range from the configured ratio, and which conditions to shorten or
lengthen. A repair returning unchanged bodies is recorded as making no progress,
and the next one is told so and asked for a materially different correction. A
repair request that would exactly repeat its predecessor is refused rather than
sent. The budget, the thresholds, the validators and the seed are unchanged, and
an exhausted group is still `needs_manual_review`.

## Confirmation repair smoke (2026-09-16)

The gate the previous section asked for. Evidence, read-only and gitignored, in
`data/pilot/smoke/pipeline_repair_confirmation_2026-09-16/`.

- **The correction worked as designed.** The two repair requests were
  **distinct** — prompt hashes `4cca3674ad7cb3e2…` and `7caa5aad170dbc4f…`, not
  the single repeated hash of the first run — and both carried the numerical
  diagnostics: per-condition body and full-text sentence and word counts, the
  required count, the 1.10 target against the 1.15 ceiling, and which conditions
  to shorten or lengthen. The second repair also stated that the first had
  changed nothing.
- **The model did not use them.** Qwen3-14B returned **the original four bodies
  unchanged on both repairs**; every attempt is recorded with `no_progress: true`.
- **All three structural errors remained** at every attempt:
  `E_SENTENCE_COUNT_MISMATCH`, `E_WORD_RATIO_BODY`, `E_WORD_RATIO_FULL_TEXT`.
- **Terminal outcome `needs_manual_review`**, after the scenario was accepted and
  the four permitted calls were used.
- **No pilot generation occurred.**

So the repair *mechanism* is now correct and observable — distinct requests,
measured findings, no-progress recorded — while this model, on this item, did not
act on them. No further repair-prompt change and no model change were made in
response; what that means for drafting is a question for the pilot, not for
another prompt revision.

## Live scenario stage and human review (2026-09-16 / 17)

**The scenario stage ran live on 2026-09-16.** All 24 scenarios were drafted,
one call each, and **24/24 passed the machine checks**. The run is read-only
evidence under `data/pilot/run/pilot_scenarios_2026-09-16/` (gitignored).

**Vidhi reviewed all 24 on 2026-09-17**, recorded in the tracked
`data/pilot/scenario_approvals.yaml`: **15 approved**, **9 marked `redraft`**.
Each decision binds the exact text, the call, the configuration hash and the
topic-bank hash, and every one of the seven judgements is answered.

The nine, with what failed:

| scenario | failed judgements |
|---|---|
| `climate_01_v1` | added facts; option neutrality |
| `climate_01_v2` | both facts stated; added facts |
| `climate_04_v1` | added facts; length without filler |
| `energy_01_v1` | length without filler (140 words) |
| `energy_03_v1` | option neutrality |
| `energy_03_v2` | both facts stated; added facts |
| `technology_03_v1` | both facts stated; added facts |
| `technology_03_v2` | added facts; length without filler (132 words) |
| `technology_04_v1` | added facts |

**No group generation has occurred**, and the gate keeps it blocked: 15 of 24
scenarios approved is not a pilot.

**The redraft stage is implemented as of this pass and has not run live.**
`pilot.py redraft-scenarios` drafts exactly the scenarios marked `redraft` —
the set comes from the approvals file, never from `--only` — one call each, no
group call, no automatic second attempt, and the same authorisation and
pre-flight requirements as any other live stage. Its template,
`prompts/scenario_redraft_v1.txt`, is versioned and hashed **outside**
`configs/experiment.yaml`: recording it there would change the configuration
hash and make all 15 approvals stale. Each redraft is a new call recording
`supersedes_call_id`, both text hashes, the reviewer's reason, the template and
prompt hashes, the model revision, the server record, the sampling fields, usage
and its machine findings. A redraft that comes back unchanged or machine-invalid
supersedes nothing. All nine stay unapproved until Vidhi reads their new text.

## Offline pipeline (2026-09-16)

Four separate states, deliberately kept apart:

**1. The controller is implemented, offline-tested, and has run live twice.**
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
Developed against `FakeBackend`, and exercised live **twice** on 2026-09-16
through `scripts/pipeline_smoke.py` (both runs are recorded above).

**2. The repair mechanism is confirmed; the synthetic smokes are finished.**
Both runs ended `needs_manual_review`: the first because the two repair requests
were byte-identical, the second — after the correction — because the model
returned unchanged bodies despite two distinct, diagnosed repairs. That second
outcome is the mechanism working: an unrepaired group is routed to review rather
than accepted or silently retried. **No further synthetic repair smoke is
planned.** Whether Qwen3-14B repairs this particular fixture automatically is
not a prerequisite for anything.

**3. Pilot generation is implemented and unauthorised.**
`scripts/pilot.py` has the live path, in two commands that cannot be combined.
Each needs `--send`, `REASONSTYLE_ALLOW_LOCAL_GENERATION=1`,
`REASONSTYLE_ALLOW_PILOT_GENERATION=1` and `HF_HUB_OFFLINE=1`, passes the same
pre-flight as the smoke tests, and refuses any subset of the pilot. Nothing has
been run against a server.

**4. The two prerequisites are now built, offline-tested, and unused.**
`generation/approvals.py` is the curator-approval gate: an approval binds the
exact scenario text, the accepted call id, the configuration hash and the
topic-bank hash, and a change to any one of them makes it stale. `run_pilot`
drafts a group only from an approved scenario. `generation/assemble.py` builds
corpus records from approved scenarios and machine-valid groups, and carries an
auditable manual-correction record — original call id and text, corrected text,
editor, reason, validator result, approval state — kept beside the generated
material rather than replacing it. **A human approval never overrides a machine
error:** corrected text is re-validated by the same code, and an assembled
record still leaves `validation.status: draft` until the human judgements are
recorded. **The one remaining live blocker is the pilot execution path itself**,
which is separately authorised work and is not written.

### The workflow these two pieces imply

1. **Generate all scenarios.** The first invocation drafts every requested
   scenario and stops: no group is drafted while any scenario is unread.
2. **Approve all scenarios.** The curator records a decision per scenario,
   bound to its exact text, call, configuration and topic bank. The gate is
   all-or-nothing: one pending, stale, refused or machine-blocked scenario stops
   group drafting for the whole set, approved scenarios included.
3. **Draft the groups.** The second invocation recovers the scenario calls from
   disk rather than re-sending them, re-checks the complete gate, and only then
   drafts groups with their bounded repair.
4. **Correct and re-validate where required.** A human correction is a separate
   record naming the call it corrects; corrected text goes through the same
   validator, and material that still fails is still refused.
5. **Assemble.** Approved scenarios plus machine-valid groups become corpus
   records, which remain `validation.status: draft`.
6. **Complete the item, pair and scenario human review**, which is what turns a
   draft corpus into an approved one.

**The orchestration is now implemented, offline.** `scripts/pilot.py` has the
two stages as two commands — `scenarios` and `groups`, which cannot be combined
— plus `scenario-review`, `approvals`, `assemble` and `status`. Each live stage
needs `--send`, `REASONSTYLE_ALLOW_LOCAL_GENERATION=1`,
`REASONSTYLE_ALLOW_PILOT_GENERATION=1` and `HF_HUB_OFFLINE=1`, and runs the same
cache, revision, runtime-record and server-setting checks as the smoke tests
before its first call. Ceilings: 24 scenario calls with no repair path; 48 group
drafts and at most 144 calls including repairs. `assemble` writes
`data/pilot/corpus.jsonl` and its manifest atomically, refuses an existing
corpus without `--overwrite`, and validates the whole corpus before writing.

**Still not implemented:** the scenario `redraft` path. The gate reports
`redraft` and blocks the set on it, but nothing re-drafts a scenario; there is
no automatic scenario-redraft budget, and a `redraft` decision needs an explicit
later choice. **No pilot call has been authorised or made.**

## Next steps

1. **Review and commit the gate, the assembler and the two-phase controller**, then bring the server checkout up to date. The prepared environment works: four live smoke runs, ten model calls in total, have completed on it.
2. **Authorise stage one** when ready: `pilot.py scenarios --send` with both keys, 24 calls, no repair.
3. **Read the scenarios** (`pilot.py scenario-review`), record approvals, then **authorise stage two** (`pilot.py groups --send`), then `pilot.py assemble`. A `redraft` decision needs a separate choice, since nothing re-drafts automatically.
4. **After the pilot is reviewed**, extend to the full 60 decisions, then validate, human-review and freeze before any evaluated-model run.

## Open, not resolved

- **Compute.** The A6000 is for generation and possibly the first Llama-3.1-8B compatibility and behavioural tests. Larger causal sweeps may need Wisteria or an A100-class GPU, depending on measurements not yet taken.
- **Family overlap.** If an optional Qwen model is evaluated, it shares a family with the corpus generator. That would be disclosed as a limitation.
