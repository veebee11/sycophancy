# Full-corpus orchestration — implementation audit

*Written 2026-09-20, Phase 1; updated 2026-09-21. No full-corpus call has been
made. This is an audit of what the future full run needs, which existing
modules already provide it, and the smallest changes that close the gap.*

**Completed foundation work (committed with this audit):**

- the full configuration draft, `configs/experiment_v2_full.draft.yaml`
  (60/120/240/960, seed-corpus declaration, output paths under `data/full/`);
- the curated topic bank, `data/topics/full_topics_v2.yaml` — 60 decisions,
  curated 2026-09-21;
- the source registry, with the three EUR-Lex acts confirmed citable;
- validation: the per-brief topic checker and the whole-bank validator,
  `scripts/check_full_topic_bank.py`;
- the generation block, which keeps the draft configuration from drafting
  anything.

**Orchestration work (this audit's subject; implemented offline in Phase 2 — see
below):**

- the verified, read-only **import of the frozen v2 pilot** corpus;
- **union completeness checks** — pilot plus new material covers exactly the 60
  decisions, 120 scenarios and 240 groups, once each;
- **seed-aware generation** that skips the pilot's 12 decisions rather than
  redrafting them;
- **combined assembly** of one 60-decision corpus and manifest from both
  sources.

## Phase 2 status (2026-09-21, offline; uncommitted)

The importer, the union checks, seed-aware planning and the combined-assembly
interface are **implemented and tested**, and the full marker allocation is
**built**. Generation remains blocked: no request was sent, no key was read, no
full scenario or group exists and no corpus was assembled. How each change
proposed below was realised:

| Proposed change | What was built |
|---|---|
| 1. `generation/corpus_source.py` | Built as proposed, and stricter: besides corpus/manifest agreement and re-validation, it checks scenario approvals, correction provenance against the committed ledger, topic definitions, and six hashes pinned in `seed_corpus.pins`. The committed files suffice; the gitignored run directory is read only when present. |
| 2. `assemble_pilot` gains `seed_records` | **Not done that way.** `assemble_pilot` is unchanged, which keeps the pilot's behaviour byte-identical. A separate `combine_corpus` carries the seed's committed lines byte for byte and adds new records. It is tested and not used. |
| 3. manifest `sources` list | In `combine_corpus`'s manifest: `sources` (seed provenance, then full run) and a per-scenario `scenario_sources` map. |
| 4. stages skip the seed | Done in the runner, not by adding skip arguments: drafting commands on a seed-aware config run on the 48 new decisions only. A `CallStore.refused_decisions` guard in the single send path (`_send_or_resume`) and at the start of both stages makes a seed request impossible from any entry point. |
| 5. union completeness check | In the importer and the planner: seed plus remainder must be exactly the 60 curated decisions, with no overlap. |
| 6. `data/full/` run artefacts gitignored | `data/full/run_v2/` and `data/full/smoke*/` are ignored; the allocation stays trackable. |

Also built:

- **The full marker allocation**, `data/full/marker_allocation_full_v2.yaml`
  (`scripts/allocate_markers.py --seed-allocation`): 240 groups, the pilot's 48
  imported exactly, every marker and family balanced exactly across domain,
  supported option and variant. It is a committed-data candidate, **not yet
  committed**.
- **`pilot.py preflight`** on the full configuration: read-only; it reports the
  seed checks, the plan and the call budget, and fails before any networking on
  a mismatch.
- **`pilot.py status`**, which reports the seed, the new material, the eventual
  corpus and the blockers separately.

**Offline dry runs, 2026-09-23.** All five passed and wrote nothing: the
preflight; the scenario stage (96 planned); the group stage (192 planned);
`status`; and the synthetic combined-assembly test. Each keeps the block
active, reads no credential, opens no connection, creates no run directory and
leaves the allocation byte-identical. `status` and `preflight` derive their
readiness from checks performed in the run itself — the allocation rebuilt and
compared, the seed verified, the dry run actually rendered — rather than from
any stored claim.

Still outstanding before generation: **explicit authorisation of the paid
calls**, the only unmet requirement in the generation block.

The full run is **not** a bigger pilot run. It is a pilot run plus an import:
12 of the 60 decisions already exist as a completed, corrected, assembled
corpus, and the whole point is to reuse them rather than pay to regenerate text
a person has already read. That single fact is what most of the work below
serves.

## What has to be true

| | |
|---|---|
| verify and import the frozen v2 pilot corpus, read-only | 12 decisions, 24 scenarios, 48 groups, 192 texts |
| generate only what is new | 48 decisions, 96 scenarios, 192 groups, 768 texts |
| preserve both provenances | every record traceable to the run that produced it |
| never restamp a pilot approval | an approval keeps the configuration hash it was granted under |
| never write into `data/pilot/run_v2/` | the pilot run is finished evidence |
| assemble one 60-decision corpus | from two sources, with one manifest |

## What already does the job

**`generation/scenario_source.py`** is the closest fit and was built for exactly
this shape of problem. It already verifies another design's approved scenarios
against that design's configuration hash, call ids, text hashes and topic-bank
hash; refuses a source not marked `read_only`; derives the expected scenario-id
set from the curated topic bank and refuses a missing, unexpected, unapproved
or unbound scenario; and returns a provenance block recorded on every call. The
full run needs the same guarantees one level up — over an assembled *corpus*
rather than a set of scenarios.

**`generation/pipeline.py`** needs no structural change. `CallStore` already
carries a `scenario_source` block onto every log line, `run_group_stage`
already accepts a pre-verified scenario set with `gate_verified_elsewhere`, and
the budget, recovery and no-duplicate-paid-call behaviour are corpus-size
independent.

**`generation/assemble.py`** already takes `approvals`,
`approval_config_content_hash` and `scenario_source` separately, and already
writes both provenance layers into the manifest. Assembling 60 decisions from
two provenance layers is the case it was generalised for.

**`corpus/validate.py`**, the prompts, the marker allocation checks, the group
review and the review exporter are all corpus-size independent and need nothing.

**`scripts/pilot.py`** already resolves every path from the configuration's own
`paths` block, so the full design cannot reach a pilot artefact by default, and
`run_directory_problems` already refuses a run directory belonging to another
design.

## The smallest changes that close the gap

1. **`generation/corpus_source.py`** *(new, ~150 lines; mirrors
   `scenario_source.py`)*. Verify and load a frozen corpus as a seed:
   read `seed_corpus.corpus` and `seed_corpus.manifest`, check
   `corpus_matches_manifest`, check the manifest's `config_content_hash`
   against `seed_corpus.config`, re-validate the loaded records with
   `validate_corpus`, and check every scenario text hash and group call id
   against the manifest. Return the records plus a provenance block. Refuse on
   any mismatch. **This is the load-bearing new component.**

2. **`assemble_pilot` gains a `seed_records` argument** *(~20 lines)*. Records
   that arrive verified are carried into the output unchanged and are never
   re-derived from a run directory. Their per-scenario manifest entries keep
   the pilot's own call ids, corrections and hashes; new records get the full
   run's. The existing `AssemblyRefused` paths are unchanged.

3. **The manifest gains a `sources` list** *(~10 lines)*. Two entries — the
   frozen pilot and the full run — each with its configuration version and
   hash, its run directory and the decisions it contributed. One corpus, two
   traceable origins.

4. **`run_group_stage` and `run_scenario_stage` skip what the seed supplies**
   *(~15 lines)*. Both already iterate topics; they need a `skip` set of
   decision ids drawn from the seed, so the 12 reused decisions are never
   requested. This is what holds the bill to 768 new texts.

5. **A completeness check over the union** *(~20 lines, beside
   `expected_scenario_ids`)*. The seed's 12 plus the run's 48 must be exactly
   the 60 curated decisions — no overlap, no gap. Refuse otherwise.

6. **`data/full/` added to `.gitignore` for run artefacts only** *(2 lines)*,
   matching the existing rules. The corpus, manifest, approvals and corrections
   stay tracked.

Nothing above requires a change to the validators, the prompts, the four
conditions, the matching rules, the repair budget or the approval gate. That is
the point of having generalised `assemble_pilot` and `scenario_source` already.

## What is deliberately *not* being built

- No "resume a 240-group run" machinery. The existing per-call recovery already
  makes a stage resumable, and a bigger corpus does not change that.
- No parallel or batched request path. The budget discipline — one draft, at
  most two repairs, no hidden retry — matters more than wall-clock time, and
  batching would complicate the one-call-one-record guarantee.
- No automatic correction of anything. Corrections stay a separate, separately
  approved human act.

## Cost and ceiling, for the authorisation that comes later

Ceilings and expectations are kept apart. An authorisation covers the ceiling;
the expectation is only a planning estimate.

| Stage | Calls |
|---|---|
| Initial scenario drafts (48 decisions × 2 variants) | 96 |
| Scenario redrafts, if every scenario were sent back once | up to 96 |
| Group stage (192 groups × at most 3 calls each) | up to 576 |
| **Initial scenarios + maximum group stage** | **672** |
| **Absolute worst case, every scenario also redrafted** | **768** |

Redrafts are not automatic: each is a separately recorded human decision on a
reviewed scenario, so the 96 redraft calls are a ceiling on what review could
request, not a repair loop.

*Expected*, at the pilot's measured rates — 4 of 24 OpenAI scenarios sent back
for a redraft, and all 48 v2 groups accepted on their first call — is about
96 + 16 + 192 = **304 calls**. That estimate assumes the new decisions behave
like the pilot's; it is not a limit, and the authorisation must cover 768.

## Gates

The researcher's recorded sequence:

- **Building the full draft text corpus may occur before mentor review.**
  Drafting text is not blocked on it.
- **Mentor review remains a gate** before formal annotation, corpus freezing,
  behavioural evaluation and mechanistic analysis. Drafting the corpus first
  does not substitute for that review.
- **Full generation additionally requires** a completed full marker allocation,
  a verified import of the seed pilot corpus, dry-run validation of the
  orchestration, and **explicit authorisation of the paid calls**. Until all
  four are in place, the generation block in
  `configs/experiment_v2_full.draft.yaml` stays on.
