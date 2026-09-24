# Current status

*Updated 2026-09-24. Kept short; history is in git, rules in `design_notes.md`.
Sections under* Historical record (v1) *are kept as they were written and are
superseded wherever they say something has not yet run.*

## Where things stand

| | |
|---|---|
| v1 group design (Qwen, then hosted) | **historical evidence**; neither v1 group set was corrected or assembled |
| v2 pairwise group design | adopted; the design of the final corpus |
| v2 pilot (12 decisions) | **generated and assembled**: `data/pilot/corpus_v2.jsonl`, `validation_status: draft`, 1032 human judgements outstanding |
| manipulation-check protocol | recorded before any rating was examined; pending mentor review |
| full topic bank (60 decisions) | **curated 2026-09-21**; 0 citability errors, 0 outstanding curation judgements |
| frozen-pilot import and seed-aware planning | **implemented and tested**; Phase 2 committed as `39b1a2528ef8c0700f87f4905f6bc0b51e1f1f75`; all offline dry runs passed 2026-09-23 |
| full marker allocation | **built and committed** (`data/full/marker_allocation_full_v2.yaml`, 240 groups) |
| final full configuration | **frozen 2026-09-23** at `configs/frozen/v2_full.yaml` (`v2_full`, content hash `7548650b42e7…`); block lifted deliberately |
| paid-call authorisation | **96 initial scenario calls only**, one per new scenario; redrafts, groups, repairs, evaluation and mechanistic analysis are excluded and refused |
| full scenario generation | **ran live 2026-09-23**: 96 of 96 scenario calls completed, all accepted and machine-valid, 0 errors; `data/full/run_v2/` (gitignored) |
| full scenario review | **complete, 2026-09-24: 96 of 96 approved**, 0 pending, stale or machine-blocked, in `data/full/scenario_approvals_full_v2.yaml`. First pass (2026-09-23): 94 approved, 2 sent to redraft (`technology_08_v1`, `technology_13_v1`); both redrafted and approved since |
| scenario-redraft authorisation | **authorised, run and complete, 2026-09-24**: `data/full/authorizations/scenario_redraft_v1.yaml`, `status: authorized`. Both authorised calls ran successfully, machine-valid, and both replacement texts were human-reviewed and approved. It never edited `configs/frozen/v2_full.yaml` — see `design_notes.md`, *Separate stage authorisations* |
| full group generation | **not started; unauthorised.** The scenario gate is complete (96/96 approved); group drafting, redrafts and repairs remain excluded from `generation_authorization` and need their own separate authorisation, refused before a backend is built |

## Records and their authority

- **Authoritative data and provenance:** YAML, JSONL and manifests — configs,
  topic bank, source registry and download manifest, approvals, correction
  ledgers, allocations, corpora and their manifests, and the generation logs
  and raw responses (the last kept locally and gitignored).
- **Committed Markdown** records design (`design_notes.md`) and status (this
  file). It describes the data; where the two disagree, the data wins.
- **`review/`** holds deterministic reading views regenerated from the data.
  They are local and gitignored, never the committed record.
- **Chat reports are not project records.** What matters from them is written
  into the files above.

## Model calls made, recomputed from the stored logs (2026-09-21)

Counted from the `generation_log.jsonl` files under `data/pilot/` (gitignored),
one line per call. Overlapping Qwen snapshot directories are de-duplicated by
call id; within a single log every line counts, because the first repair smoke
sent two byte-identical repair requests. Only metadata fields were read.

| Stage | Model | Calls | What they were |
|---|---|---|---|
| Local Qwen synthetic smokes | `Qwen/Qwen3-14B` | **9 stored** | 1 draft-only group (2026-09-15); 2 repair smokes × 4 (2026-09-16) |
| Local Qwen v1 pilot | `Qwen/Qwen3-14B` | **173** | 24 scenarios, 9 scenario redrafts, 48 group drafts, 92 repairs |
| Hosted v1 smoke | `gpt-5.6-sol` | **1** | one synthetic group (2026-09-18) |
| Hosted v1 pilot | `gpt-5.6-sol` | **141** + 1 transport error | 24 scenarios, 4 scenario redrafts, 48 group drafts, 65 repairs; one draft attempt failed in transport with no response and was retried |
| v2 smokes | `gpt-5.6-sol` | **3** | single groups: 1 `therefore`, 2 `it follows that` (2026-09-20) |
| v2 pilot groups | `gpt-5.6-sol` | **48** | one call per group, no repairs needed |

Totals: **182 local Qwen calls stored** and **193 hosted calls** (142 v1, 51
v2), plus one hosted transport error. A second draft-only Qwen smoke on
2026-09-16 is documented below (call `05a54889…`) but its log is not stored in
this working copy — call ids are derived from the request, and that id now
belongs to the hosted smoke of the same synthetic group — so the earlier
documented Qwen figure of 183 includes one call not countable from local logs.
**No call of any kind has been made for the full corpus.**

## Historical: the v1 runs

v1 required all four cells of a group to sit inside one word ratio.

- **Qwen3-14B, local (2026-09-15 to 17).** Scenario gate completed (24/24
  approved after nine redrafts and five audited corrections); group stage 2 of
  48 machine-valid, 46 `needs_manual_review`, no repair ever producing an
  accepted group. The 46 were never corrected and nothing was assembled.
- **`gpt-5.6-sol`, hosted (2026-09-18).** Same rules, same prompts. 24
  scenarios drafted; review approved 20 and sent 4 for a bounded redraft, all 4
  then approved (`data/pilot/scenario_approvals_openai.yaml`, 24/24). Group
  stage 36 of 48 machine-valid (3 on the first draft, 33 after repair), 12
  `needs_manual_review`. Not corrected or assembled: the v2 design superseded
  the group layer before that.

Both runs stay as read-only evidence, and the detail is kept below under
*Historical record (v1)*.

## The v2 pairwise design, and why it replaced v1

v1's four-way length rule could only be met by padding the no-premise cells:
Qwen failed it in 44 of 46 groups (median body ratio 1.51 against 1.15), the
hosted model in 9 of its 12 failures, and where it was met it was met with
self-referential filler — a second commitment in a condition meant to carry
one. v2 (`configs/experiment_v2_pilot.yaml`; `design_notes.md`,
*Version-2 group design*) replaces it:

- a **fixed endorsement**, `I support the option to ${supported_option_text}`,
  word for word, once, in every cell;
- **pairwise, marker-only matching**: RS is RP plus the marker, NS is NP plus the
  same marker in the same position, an exact word budget rather than a ratio;
- NS and NP state **no explicit task-relevant premise**; reason cells are
  longer by design, so `RS − NS` and `RP − NP` become exploratory and `NS − NP`
  stays primary confirmatory;
- **four marker strings in two families** — `therefore`, `consequently`
  (`conclusion_indicator`); `it follows that`, `this implies`
  (`metadiscursive_inference`); premise indicators deferred;
- consistent terminal punctuation across all four cells, added after a v2 smoke
  returned an NP cell without its final period.

## Completed v2 pilot

- **12 decisions**, 4 per domain (`data/topics/pilot_topics.yaml`).
- **24 approved scenarios reused read-only** from the hosted scenario run,
  verified against that run's configuration hash, call ids, text hashes and
  topic-bank hash; their approvals keep the hash they were granted under.
- **Marker allocation** `data/pilot/marker_allocation_v2.yaml`: 48 groups, 12
  per marker string, 24 per family.
- **48 v2 groups, 192 texts.** **All 48 groups were machine-valid on their first
  generation call**; no repair was needed.
- **4 groups / 8 RS–RP cells carry approved modal corrections** ("would" →
  "could"), recorded in `data/pilot/manual_corrections_v2.yaml`, each bound to
  its exact call and original text and re-validated.
- **Assembled:** `data/pilot/corpus_v2.jsonl` (24 scenario records) and
  `data/pilot/corpus_v2.manifest.json` — 0 machine errors, 4 machine warnings,
  corpus SHA-256 `7e0dae8415ab…`.
- **Validation status `draft`.** **1032 formal human judgements** (item, pair
  and scenario) are outstanding; none has been recorded.

## Manipulation-check protocol

`configs/manipulation_checks_v1.yaml` fixes the rules that decide whether the
manipulations worked — reason/no-reason and styled/plain separation,
proposition preservation, equivalence bounds and what low agreement means —
**recorded on 2026-09-20 before any annotation rating was examined**. Status
`researcher_approved_pending_mentor_review`. It is a separate file so the v2
pilot's configuration hash is unchanged, and nothing in generation reads it.

## Full topic bank (curated 2026-09-21)

`data/topics/full_topics_v2.yaml`:

- **60 curated decisions — 20 climate, 20 energy, 20 technology**: the **12
  pilot decisions preserved byte for byte**, plus **48 curated additions**
  (climate_06–21, energy_06–21, technology_06–21), curated by Vidhi Bhutani on
  2026-09-21 with all nine judgements true;
- the **3 rejected pilot candidates** (climate_03, energy_04, technology_05) are
  carried as rejected and excluded from every count;
- per-brief checker: **0 errors, 0 warnings — 0 source-citability errors, 0
  outstanding curation judgements**;
- whole-bank validator (`scripts/check_full_topic_bank.py`): 0 errors and **two
  lexical-overlap warnings retained**, not suppressed — climate_10/climate_14
  and energy_01/energy_06, each 0.40. Both pairs are recorded as substantively
  distinct review notes in `docs/full_topics_v2_climate_coverage.md`.

Before curation, every variant context was audited for tilt and 62 of 96 were
rewritten to state only the setting; climate_09 was reworked so both points of
obligation are operationally plausible; climate_18 was replaced because the
earlier version misstated Effort Sharing art. 8.

## Sources

Three consolidated EU acts were added from EUR-Lex on 2026-09-21 to support the
climate additions: `eurlex_ets_directive` (Directive 2003/87/EC,
02003L0087-20240301), `eurlex_effort_sharing` (Regulation 2018/842,
02018R0842-20230516) and `eurlex_lulucf` (Regulation 2018/841,
02018R0841-20230511). Each is **citable, seed-only, under CC BY 4.0 with
attribution required**, confirmed by Vidhi Bhutani on 2026-09-21 against the
EUR-Lex legal notice and Decision 2011/833/EU. A consolidated text has no legal
effect, which the registry records. The official PDFs are pinned by SHA-256 in
`data/sources/downloads.yaml` (EUR-Lex publishes no checksum of its own); the
raw files are never committed. The registry holds 6 citable, 5 excluded and 1
unverified candidate.

## Full corpus, Phase 2: seed import and full allocation (2026-09-21, offline)

Implemented, tested and **committed as `39b1a2528ef8c0700f87f4905f6bc0b51e1f1f75`**
("Add frozen pilot import and full marker allocation"). At the time this
section was written (2026-09-21) no request had been sent, no key had been
read, no scenario or group had been generated and no corpus had been
assembled; the scenario stage that changed that is recorded below, under
*Scenario stage and review (2026-09-23)*.

- **Frozen-pilot importer** (`src/reasonstyle/generation/corpus_source.py`).
  Before returning a record it verifies the seed configuration hash against
  the manifest and the pins; corpus/manifest agreement and their pinned
  SHA-256s; all 192 texts under the pilot's own configuration, with nothing
  restamped; exactly 12 decisions, 24 scenarios, 48 groups and 192 texts; that
  the 12 are the curated pilot decisions with unchanged definitions in the full
  bank; scenario text hashes and approvals; group call ids and all 8 approved
  corrections; the pinned pilot allocation and every record's marker; and
  that seed plus remainder is exactly the 60 curated decisions. The committed
  files suffice; the gitignored run directory is checked read-only only when
  present. Its provenance block keeps the pilot's own hashes.
- **Seed pins.** `seed_corpus.pins` in `configs/frozen/v2_full.yaml`
  records six hashes of the committed seed; a missing or malformed pin is a
  refusal. The draft configuration's content hash is now
  `1907344906a543f914c5fe5980c2c7a0059f9ab0d557c886450247e185116409`.
- **Full marker allocation**, `data/full/marker_allocation_full_v2.yaml`,
  allocation hash `ba39dbcb24f5…`: the 48 pilot rows imported exactly and 192
  new rows allocated around them. Every marker has 60 groups — 20 per domain,
  30 per supported option, 30 per variant — and every family 120. A fresh
  rebuild is byte-identical and `--check` rejects any edit. **Committed in the
  same commit as the importer above.** The v1 and v2 pilot allocations still
  rebuild byte for byte.
- **Seed-aware planning.** Drafting commands on the full configuration run on
  the 48 new decisions only: 96 scenarios, 192 groups, 768 texts. Every send
  path refuses a seed decision before recovery or backend, and outputs under
  `data/pilot/` are refused.
- **Combined assembly interface** (`combine_corpus`): seed lines pass through
  byte for byte, new records must be exactly the planned scenarios under the
  full configuration, order is by scenario id, and the manifest names both
  sources. It is tested but **not used**; no full corpus exists.
- **Offline preflight and status**: `pilot.py preflight` and `pilot.py status`
  on the full configuration. Read-only; no credential, no backend, no
  connection.
- **Offline dry runs, run 2026-09-23.** All five passed, writing nothing:
  `preflight`; the scenario stage (96 planned); the group stage (192 planned);
  `status`; and the synthetic combined-assembly test. No credential was read,
  no connection opened, no run directory created and nothing written under
  `data/pilot/`; `data/full/marker_allocation_full_v2.yaml` was byte-identical
  before and after.
- **Status derives its readiness live.** `pilot.py status` rebuilds and
  compares the allocation, verifies the seed and performs the dry run each
  time it runs. There is no stored claim that a check once passed. Before
  2026-09-23 the dry-run line was a hard-coded "outstanding" that checked
  nothing; that is fixed.
- **Generation remains blocked.** Of the four requirements, three are met —
  the allocation, the verified seed import, and the orchestration tests and
  offline dry runs. **Explicit authorisation of the paid calls is the only
  one outstanding.**

## Final configuration frozen (2026-09-23)

- `configs/experiment_v2_full.draft.yaml` became **`configs/frozen/v2_full.yaml`**
  — version `v2_full`, `status: frozen`, in the directory the project reserves
  for frozen configurations — with its file and content hashes recorded in
  `configs/frozen/MANIFEST.json` and verified by tests.
- **Content hash `7548650b42e7cb7407f521d10d9e8c2d4f9f25b5074d98cc523b4a5e35a99c94`.**
  The model, endpoint, prompts, decoding settings, corpus design, marker
  inventory, seed pins and every scientific rule are unchanged from the draft.
- **The generation block was lifted deliberately**, recording who lifted it and
  when, after all four requirements were met and checked.
- **The authorisation is narrower than the block.**
  `corpus.generation_authorization` records: scope
  `scenario_stage_initial_only`, at most **96 paid calls**, one per scenario,
  authorised by Vidhi Bhutani on 2026-09-23. Scenario redrafts, group
  drafting, repairs, behavioural evaluation and mechanistic analysis are
  excluded. The runner refuses an unauthorised stage before a backend is built,
  and the send path refuses an unauthorised call kind, so a resumed run cannot
  make one either.
- **The allocation was rebound** to the frozen configuration's hash. Its 240
  assignments and its content hash `ba39dbcb24f5…` are unchanged, and all 48
  pilot rows remain identical.
- **Superseded below.** The paragraph as written here said no full-corpus call
  had yet been made and named scenario review as the next gate; both are now
  true of the past tense only — see *Scenario stage and review (2026-09-23)*.

## Scenario stage and review (2026-09-23)

- **The authorised 96-call scenario stage ran live on 2026-09-23**, writing to
  `data/full/run_v2/` (gitignored: `generation_log.jsonl`, `raw/`, `results/`).
  96 of 96 calls completed — one call per new scenario, attempt 1, no retry —
  under model `gpt-5.6-sol` (requested and returned, every call), configuration
  hash `7548650b42e7…` and allocation hash `ba39dbcb24f5…`. Every call recorded
  `store: false`, `background: false`, no tools, no conversation state and no
  `previous_response_id`. **All 96 outcomes are `accepted`/machine-valid, 0
  errors.** The 48 decision ids covered are exactly the full bank's 48 new
  additions (`climate_06`–`21`, `energy_06`–`21`, `technology_06`–`21`); no
  pilot decision was drafted. **0 group, repair or redraft calls occurred** —
  the authorisation covers the scenario stage only, and nothing else was sent.
- **Independent scenario review** read all 96 against their briefs and
  recommended 94 approvals and 2 redrafts (`technology_08_v1`,
  `technology_13_v1`, both failing `no_added_facts_or_quantities`: each draft
  added a claim the supplied facts do not establish). Vidhi Bhutani reviewed
  and accepted the recommendation on 2026-09-23.
- **Recorded** in `data/full/scenario_approvals_full_v2.yaml` — the
  authoritative *tracked* file (declared in `paths.approvals`), not a
  gitignored `review/` export: 96 entries, one per new scenario, each bound to
  its exact call id, scenario-text SHA-256, configuration hash and
  topic-bank hash from the recorded run. **94 `approved`** (all seven
  judgements true, decided by Vidhi Bhutani on 2026-09-23). **2 `redraft`**
  (`no_added_facts_or_quantities: false`, the other six judgements true, each
  with its reviewer reason bound to the exact call and text it concerns).
- **Verified read-only** against the project's own gate tooling
  (`pilot.py approvals`, filtered to the 48 new decision ids, and
  `pilot.py status`): both now agree — **94 approved, 2 redraft, 0 pending,
  stale, or machine-blocked** among the 96. `pilot.py status`'s next-gate
  wording was stale until 2026-09-24 (it always read "scenario review of the
  96 drafted scenarios" regardless of what the run and approvals file actually
  held); it now derives the next gate live from the recorded scenarios and
  approvals, the same way the rest of `status` already derived the allocation
  and dry-run lines. The `data/full/run_v2/` evidence was hashed before and
  after every review/export step and confirmed byte-identical throughout.
- **Redraft calls and group generation remain unauthorised.** `redraft-scenarios`
  needs its own recorded authorisation, distinct from the scenario-stage
  authorisation already spent; group drafting needs a further one still. Neither
  exists in `configs/frozen/v2_full.yaml`, and the runner refuses both before a
  backend is built.

## Scenario-redraft authorisation: proposed, authorised, run and approved (2026-09-24)

**Why a separate record, not an edit to the frozen configuration.** The
frozen configuration's own `corpus.generation_authorization` excludes
scenario redrafts by name, and every one of the 94 scenario approvals is
bound to the configuration's exact content hash
(`7548650b42e7cb7407f521d10d9e8c2d4f9f25b5074d98cc523b4a5e35a99c94`). Editing
that block to cover redrafts — even only to add one kind to `allowed_kinds`
— would change the hash and stale all 94 approvals at once. Full reasoning:
`design_notes.md`, *Separate stage authorisations*.

- **The mechanism**: `reasonstyle.generation.stage_authorization`, plus a new
  `redraft-scenarios --stage-authorization PATH` flag on `scripts/pilot.py`.
  A record binds to the exact configuration (version and content hash), the
  exact approvals file (path and its own SHA-256), and an exact, named set of
  targets — each bound to the rejected call id and rejected text SHA-256 the
  approvals file currently holds. Every field is checked
  (`stage_authorization_problems`) before any credential is read or backend
  is built, and only a record whose every check passes widens the send path
  (`CallStore.allowed_kinds`) to accept a `scenario_redraft` call — nothing
  else, and never by inference from the configuration's own authorisation.
  Without `--stage-authorization`, `redraft-scenarios` behaves exactly as
  before this flag existed: refused by the configuration's own
  `generation_authorization`, which excludes the kind.
- **The record**: `data/full/authorizations/scenario_redraft_v1.yaml`. It
  names exactly `technology_08_v1` and `technology_13_v1` — the two, and only
  the two, the authoritative approval file marked `redraft` — each bound to
  its rejected call id and rejected text SHA-256; `max_paid_calls: 2` (exactly
  the number of authorised targets, no unused headroom), `calls_per_target: 1`;
  and explicit exclusions for the original 96-call stage, any other scenario,
  a second call for either target, groups, repairs, evaluation and
  mechanistic analysis.
- **Fails closed, and was checked before being trusted.** Proposed
  `status: proposed`, `authorized_by: null`, `authorized_at: null`
  (2026-09-24) — **not operative**: `redraft-scenarios --stage-authorization`
  refused with three separate reasons (`status` not `'authorized'`;
  `authorized_by` names no reviewer; `authorized_at` carries no valid date)
  while every scientific binding already matched reality. **Authorised the
  same day** by Vidhi Bhutani — only `status`, `authorized_by: Vidhi Bhutani`
  and `authorized_at: '2026-09-24'` changed (committed separately,
  `Authorize two full v2 scenario redrafts`); no target, hash, reason,
  ceiling or exclusion was touched. `stage_authorization_problems` then
  reported zero problems.
- **A validated stage authorisation widens what may be sent, deliberately.**
  The frozen configuration's own authorisation excludes `scenario_redraft` by
  name; passing every check in a stage authorisation is exactly what lets the
  send path (`CallStore.allowed_kinds`) accept that kind at all, for the two
  named targets only. This is not merely a narrowing of the configuration's
  own authorisation — it grants a permission the configuration withholds.
- **One call per target, enforced even across transport failures.** A
  request that reaches the backend can incur cost whether or not a usable
  response comes back, so a transport failure under a stage authorisation
  still spends that target's one authorised call: the ordinary pipeline rule
  that a transport failure "recovers as nothing happened" and may be retried
  on the next run (`CallStore.recover`) is deliberately overridden here.
  `transport_blocked_targets` refuses to dispatch a target again under the
  *same* authorisation once a transport-failure entry carrying that
  authorisation's own SHA-256 exists for it; retrying needs a new, explicit
  retry authorisation record with a different hash. A target that completed
  successfully still resumes from disk rather than being re-sent
  (`CallStore.recover`), and `calls_per_target: 1` / `max_paid_calls: 2`
  bound the stage's total regardless of how many times it is invoked — but
  none of that makes a transport-failed attempt free to retry automatically.
  Tested in `tests/test_stage_authorization.py`.
- **Traceability, for every outcome.** Every dispatch this mechanism
  authorises records the authorising record's own file SHA-256 (and path) in
  its log entry's `extra` block — a successful call, a rejected one, *and* a
  transport failure alike — so a request that incurred cost always traces
  back to the exact record that permitted it, whether or not it produced a
  usable response.
- **Both authorised calls ran successfully.** Live, via the OpenAI Responses
  API, exactly the two calls the record authorised, exactly once each: new
  call ids `9dc3b60770…` (`technology_08_v1`, supersedes `9b06829bdc91…`) and
  `842b455255…` (`technology_13_v1`, supersedes `20ef0cb34d31…`), both `ok` /
  `accepted`, both traced to the authorisation's own file SHA-256
  (`0904c9e4ee72…`) in their log entries. `data/full/run_v2/` went from 96 to
  98 recorded calls; no other call was made.
- **Both replacement texts were human-reviewed and approved** by Vidhi
  Bhutani on 2026-09-24 — `technology_08_v1` no longer generalises the
  supplied fact into an "overall capability" claim; `technology_13_v1` no
  longer implies an unstated officer-review safeguard. Recorded in
  `data/full/scenario_approvals_full_v2.yaml`: both entries rebound to the
  new call id and new text SHA-256, `decision: approved`, all seven
  judgements true, `reason: null`. The redraft log and this authorisation
  record keep the superseded calls and the original rejection reasons; the
  approvals file does not restate them.
- **All 96 new scenarios are now approved.** `pilot.py approvals` and
  `pilot.py status` agree: 0 of 96 not approved; `status`'s next gate reads
  "group drafting authorisation (every scenario is approved)". (Fixed the
  same day: `status` previously judged a redrafted scenario by its
  *pre-redraft* text, so it could disagree with `approvals` about a target
  the approvals file had already caught up with — `_full_status` now reads
  the supersession-resolved current text, exactly as `approvals` does.)
- **The next gate is a separate group-stage authorisation.** No group call
  has occurred, and none is authorised: `configs/frozen/v2_full.yaml`'s own
  `generation_authorization` still excludes `group` by name, and
  `data/full/authorizations/` holds no group-stage record. One would need its
  own proposal, its own review, and its own explicit authorisation, exactly
  as the scenario-redraft record did.

## Full corpus: what does not exist yet

- **No full group** has been generated, and no full corpus has been assembled.
  `data/full/` holds the tracked allocation, the 96 recorded scenario calls
  (gitignored run directory) and the tracked scenario approvals above.
- **No group, repair, evaluation or mechanistic-analysis call has been made.**
  Only the 96 initial scenario calls plus the 2 authorised redraft calls have
  been sent — 98 total, every one traceable to what authorised it.
- `configs/frozen/v2_full.yaml` keeps the generation block **lifted**, but its
  `generation_authorization` still covers the initial scenario stage only;
  every later stage — including the group stage next — needs its own
  separately tracked authorisation record before it can run.
- The Phase 1 work was committed as "Prepare curated full v2 topic bank"; the
  Phase 2 work above as "Add frozen pilot import and full marker allocation".

## Corpus scope (settled 2026-09-16)

- The **12 decisions × 2 variants** now being generated are the **pilot corpus**: 24 scenarios, 48 groups, 192 counterargument texts.
- The **main corpus is 60 policy decisions** — 20 climate, 20 energy, 20 technology — giving 120 scenarios, 240 groups and 960 texts, as in Research_Plan_v6 §5 and `configs/experiment.yaml`.
- **192 texts are 12 independent policy decisions, not 192.** One decision yields 16 texts; the independence unit is `decision_id`.
- **The pilot's 12 count toward the 60** if they pass the final frozen specification and review criteria, so the normal remainder is **48 additional decisions, not 60**. One that cannot qualify is regenerated under the final procedure or replaced; the total stays 60.
- **Decision-level splits stay 36/12/12**, and the **mechanistic subset is 40 of the 60** — its count is fixed, and its selection rule is documented before mechanistic analysis begins.
- **Evaluated-model runs wait** until the complete 60-decision corpus is validated, human-reviewed and frozen.
- **Any change from 60 is a documented design amendment**, made before any main evaluated-model outcome is examined.

The order this implies (updated 2026-09-21):

1. **Engineering — done.** The 12-decision v2 pilot is generated and assembled as a draft; its human review is outstanding.
2. **Production — in progress.** The 60-decision topic bank is curated. Next: full marker allocation, the frozen-pilot import and seed-aware orchestration, then the remaining 48 decisions' drafting. Building the full draft text corpus may precede mentor review.
3. **Gate.** Mentor review, then formal annotation and freezing of the whole corpus.
4. **Only then.** The main behavioural evaluation.
5. **After that.** Mechanistic analysis over 40 of the 60, by the documented selection rule.

## Resolved

- **Reliability sample size (2026-09-15).** The implemented rule stays: `round(N × 0.20)`. Pilot: 38 items covering all 36 item strata, and 19 pairs covering all 18 pair strata. The older 48-item / 24-pair statement is withdrawn.

## Next steps

1. ~~Build the full marker allocation~~ — built in Phase 2, committed as `39b1a2528ef8c0700f87f4905f6bc0b51e1f1f75`.
2. ~~Implement the orchestration~~ — importer, planning, seed guard and combined-assembly interface implemented and tested in Phase 2, committed in the same commit.
3. ~~Dry-run the full stages offline~~, ~~freeze the final configuration~~ and ~~lift the generation block~~ — done 2026-09-23.
4. ~~Run the authorised 96-call scenario stage~~ and ~~record its human review~~ — done 2026-09-23: 94 approved, 2 sent to redraft.
5. ~~Prepare, authorise and run the two redraft calls~~, ~~record their human review~~ — done 2026-09-24: both calls ran successfully and both replacements are approved. **All 96 new scenarios are now approved.**
6. **Next: authorise the group stage.** It needs its own separately tracked authorisation record, on the same fail-closed mechanism — proposed, reviewed and explicitly authorised before any group call is sent. None exists yet, and none is authorised by anything above.
7. Mentor review of the manipulation-check protocol, then formal annotation, freezing, behavioural evaluation and mechanistic analysis, in that order.

## Open, not resolved

- **Compute.** The A6000 is for generation and possibly the first Llama-3.1-8B compatibility and behavioural tests. Larger causal sweeps may need Wisteria or an A100-class GPU, depending on measurements not yet taken.
- **Family overlap.** If an optional Qwen model is evaluated, it shares a family with the corpus generator. That would be disclosed as a limitation.
---

# Historical record (v1)

> Kept as written, for provenance. These sections describe the v1 group design
> and its Qwen and hosted runs as of the dates in their headings. Where they say
> something "has not run", is "next" or is "not yet" done, that statement is
> **superseded** by the sections above: the hosted generator has run, v1 was
> replaced by the v2 pairwise design, and the v2 pilot has been generated and
> assembled.

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

**Vidhi's first review of all 24 on 2026-09-17** produced **15 approved** and
**9 marked `redraft`**. That historical state authorised the bounded redraft
run; the tracked `data/pilot/scenario_approvals.yaml` now holds the later final
24/24 decision. Each decision binds the exact text, the call, the configuration
hash and the topic-bank hash, and every one of the seven judgements is answered.

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

**No group generation occurred during either scenario stage.** The initial
review blocked the gate at 15 of 24 approved scenarios.

**The redraft stage ran live on 2026-09-17.**
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
supersedes nothing.

Vidhi and the research reviewer read all nine returned texts. Four were approved
as returned: `climate_01_v2`, `climate_04_v1`, `energy_03_v1` and
`technology_04_v1`. Five still retained an added claim, omitted or altered a
supplied fact, or retained repetition: `climate_01_v1`, `energy_01_v1`,
`energy_03_v2`, `technology_03_v1` and `technology_03_v2`. Those five now have
approved human corrections in `data/pilot/scenario_corrections.yaml`. The
ledger names the exact Qwen call and original text, stores both hashes, and is
applied only after the corrected text passes the scenario validator. The current
approval file binds the final exact text and source call for all 24 scenarios.
The gate is now clear; this does not generate or approve any counterargument.

## Live group stage (2026-09-17)

**The 48-group stage ran live on 2026-09-17**, on the run directory holding the
24 scenario calls and the nine redrafts, against the 24 approved scenario texts
(five of them human-corrected). Evidence, read-only and gitignored, in
`data/pilot/run/pilot_groups_snapshot_2026-09-17_complete/`. Its
`generation_log.jsonl` has **173 lines**: 33 scenario and redraft calls from the
earlier stages, and **140 group and repair calls** from this one.

- **48 groups attempted; 140 of the 144 permitted calls used** — two groups that
  passed on their first draft, and 46 that each used one draft and two repairs.
- **2 groups are machine-valid**: `energy_01_v1 / opt_2` and
  `energy_03_v1 / opt_2`, both accepted on their initial draft.
- **46 groups ended `needs_manual_review`.**
- **No repair produced an accepted group.** Not one of the 92 repair calls turned
  a failing group into a valid one.
- **Repair 1 returned the text it was given, unchanged, for 33 of the 46.**
  **Repair 2 did so for 41 of the 46.** **Thirty groups had identical bodies at
  all three attempts.**

Final errors among the 46, counted by group:

| Code | Groups |
|---|---|
| `E_WORD_RATIO_BODY` | 44 |
| `E_WORD_RATIO_FULL_TEXT` | 42 |
| `E_BODY_SENTENCE_COUNT` | 15 |
| `E_SENTENCE_COUNT_MISMATCH` | 15 |
| `E_PAIR_CONTENT_DRIFT` | 6 |
| `E_DUPLICATE_TEXT` | 4 |
| `E_MARKER_MISSING_IN_STYLED_CELL` | 3 |
| `E_OPENING_REPEATED_IN_BODY` | 2 |
| `E_MARKER_IN_PLAIN_CELL` | 1 |

The dominant failure is length matching: the reason-bearing cells are far longer
than the no-reason cells, so the body ratio, and with it the full-text ratio,
exceeds the 1.15 ceiling. The sentence-count failures are the same shape as the
synthetic smokes — dropping the connective splits the plain member of a pair
into a second sentence. The repair mechanism behaved exactly as the confirmation
smoke showed it does: distinct, diagnosed requests carrying the measurements,
and unchanged replies recorded as no progress rather than accepted.

**Read, not yet judged.** `pilot.py group-review` writes a deterministic,
read-only Markdown view of the whole run — the two machine-valid groups and the
46 kept clearly apart, each group with its scenario, options, marker, final
call, four bodies, rendered counterarguments, counts, exact findings with their
measurements, outstanding human judgements, and a per-attempt history that marks
an unchanged repair as unchanged. It sends nothing, writes nothing into the run
directory, and proposes no corrected wording.

**No generator, model, prompt, validator or threshold decision has been made in
response to this run.** The observation that this model, under this
configuration, does not repair these groups is recorded; what to change — if
anything — is a decision for after the 46 have been read, not an inference from
the counts above. `data/pilot/manual_corrections.yaml` does not exist and has
not been drafted.

## Hosted generator, as implemented on 2026-09-18 (since run)

**Why.** The Qwen pilot finished with **2 of 48 groups machine-valid**; the
dominant failure is cross-condition length matching, and **no repair produced
an accepted group** (*Live group stage*, above). `gpt-5.6-sol` is being tried as
a second generator under exactly the same rules. The reasoning is in
`design_notes.md`, *Corpus construction*, amendment of 2026-09-18.

**What exists.** `openai_responses` is a first-class backend beside
`local_vllm_openai`. The Responses API, the exact model id `gpt-5.6-sol` (the
moving alias `gpt-5.6` is refused at config load), `reasoning.effort` `none`,
temperature 0.3, `top_p` left at the provider default and not sent,
`max_output_tokens` 700, strict JSON-schema structured output through
`text.format`, `store: false`, `background: false`, no tools, no files, no
conversation and no previous-response state. Each request is stateless and
produces exactly one draft. `store: false` keeps a response out of retrievable
Responses API state; it is not a retention guarantee, since abuse-monitoring
retention may still apply under the account's data-control policy, and Zero Data
Retention is not claimed.

**What had not happened as of 2026-09-18 — superseded.** *The hosted generator
has since run (smoke, 24 scenarios, 4 redrafts, 48 groups with repairs; see* Model
calls made *above). The following was true when written:* **No API request had
been made, no key had been read and no dataset content had been generated.** Every test runs against fake
responses with networking disabled; none needs a key or costs anything. The
smoke call, the scenario stage, the approval gate and the group stage are all
still ahead, in that order.

**What is preserved.** The Qwen configuration is byte-identical and its content
hash still `9da99ff12674…`, so all 24 scenario approvals stay valid. The
completed group run and its snapshot are untouched: the hosted generator writes
to `data/pilot/run_openai/`, and a stage refuses a run directory belonging to
another generator or already holding another generator's calls. Approvals,
correction ledgers, corpus and review exports default to separate per-generator
files. Nothing was relaxed to accommodate a different model — a test asserts the
two configurations differ in the generator block and nowhere else, so the
prompts, conditions, validators, word ratios, repair budget, curator gate and
human-review codes are the same ones the Qwen pilot was held to.

**Authorisation and the credential.** A live hosted pilot stage needs three
things: `--send`, `REASONSTYLE_ALLOW_OPENAI_GENERATION=1` and
`REASONSTYLE_ALLOW_PILOT_GENERATION=1`. The local-generation key does not
authorise a paid external call. A dry run needs no key and opens no connection.
The credential is read from `OPENAI_API_KEY` at the moment of the call and
nowhere else, and never reaches a request record, a log line, an error message,
a printed line or a hash.

**Ceilings.** One smoke call; 24 scenario calls with no repair path; 48 group
drafts with at most 144 calls including repairs — 168 calls for a complete
pilot, plus at most one redraft call per rejected scenario if that path is used.

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

**3. Pilot generation is implemented, and both halves have run.**
`scripts/pilot.py` has the live path, in two commands that cannot be combined.
Each needs `--send`, `REASONSTYLE_ALLOW_LOCAL_GENERATION=1`,
`REASONSTYLE_ALLOW_PILOT_GENERATION=1` and `HF_HUB_OFFLINE=1`, passes the same
pre-flight as the smoke tests, and refuses any subset of the pilot. The 24-call
scenario stage, the nine-call redraft stage and the 140-call group stage all ran
on Chomusuke02.

**4. The gate was satisfied and the groups drafted; the assembler remains unused.**
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
recorded. Scenario corrections follow the same provenance rule without altering
the generated evidence. The group stage has since run, and 46 of its groups
carry no correction yet, so nothing assembles: the assembler refuses a group
that is neither machine-valid nor corrected.

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

The redraft path and exact-source-bound scenario correction ledger are now
implemented and exercised. `pilot.py approvals` reports 24/24 approved and zero
scenario blockers, which is the gate the group stage then passed.
