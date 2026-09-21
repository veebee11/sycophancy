# Reason or the Language of Reason?

What actually moves an LLM's position when a user pushes back — the substantive
reason, or the language that makes something sound reasoned?

Vidhi Bhutani, University of Tokyo. The research plan is
[`Research_Plan_v6.md`](Research_Plan_v6.md); design decisions are in
[`docs/design_notes.md`](docs/design_notes.md); where the work stands is in
[`docs/current_status.md`](docs/current_status.md).

## Current workflow (2026-09-21)

Where the work stands, in detail: [`docs/current_status.md`](docs/current_status.md).

- The **v2 pilot** (12 decisions, 48 groups, 192 texts) is generated and
  assembled as a draft corpus, `data/pilot/corpus_v2.jsonl`; its human review
  is outstanding.
- The **manipulation-check rules** are fixed in
  [`configs/manipulation_checks_v1.yaml`](configs/manipulation_checks_v1.yaml),
  recorded before any rating was examined and pending mentor review.
- The **full topic bank** — 60 curated decisions, 20 per domain — is
  [`data/topics/full_topics_v2.yaml`](data/topics/full_topics_v2.yaml); the
  sources behind its climate additions are set out in
  [`docs/full_topics_v2_climate_coverage.md`](docs/full_topics_v2_climate_coverage.md).
- **Next:** full marker allocation and the seed-aware orchestration audited in
  [`docs/full_corpus_orchestration_audit.md`](docs/full_corpus_orchestration_audit.md).
  Full generation is blocked in `configs/experiment_v2_full.draft.yaml`, and no
  full-corpus call has been made.

**What is authoritative.** YAML, JSONL and manifests are the data and its
provenance. Committed Markdown records design and status. **`review/` contains
generated, gitignored reading views** — deterministic local exports rebuilt
from the data — and is **not** the authoritative committed record. Chat reports
are not project records.

## The design in one paragraph

Each scenario is a normatively underdetermined public-policy trade-off with two
**immutable semantic options**, `opt_1` and `opt_2`. The model is asked to choose;
its A/B logits give an initial argmax. The transcript is then forked into four
independent branches, each carrying a counterargument that supports the option
*opposite* the initial choice:

|                    | style explicit | style plain |
|--------------------|----------------|-------------|
| **reason present** | `RS`           | `RP`        |
| **reason absent**  | `NS`           | `NP`        |

With `m = logit(counter-supported option) − logit(initially selected option)`,
the outcome is `movement_toward_counter = m_after − m_before`. The primary
contrast is `NS − NP`: style with no reason.

**A and B are display labels only.** They never carry semantic identity, and
support direction is never inferred from them.

## Corpus scope

The **pilot** is 12 policy decisions, four per domain, with two scenario
variants each: 24 scenarios, 48 four-condition groups, 192 counterargument
texts. It has been generated and assembled under the v2 pairwise design; its
human review is outstanding.

The **main corpus** is 60 policy decisions — 20 climate, 20 energy, 20
technology — giving 120 scenarios, 240 groups and 960 texts. The 12 pilot
decisions count toward that 60 if they meet the final frozen specification and
review criteria, so the normal remainder is 48 additional decisions; a pilot
decision that cannot qualify is regenerated or replaced, keeping the total at
60. Evaluated-model runs begin only once the full 60-decision corpus is
validated, human-reviewed and frozen. Mechanistic analysis then uses 40 of the
60, by a rule documented before that analysis starts.

The independence unit is the underlying decision, never the text: one decision
yields 16 texts, so 192 texts are 12 decisions, not 192. Details in
[`docs/design_notes.md`](docs/design_notes.md) under *Corpus shape*.

## Layout

```
Research_Plan_v6.md           the research plan (its hash is pinned in the config)
configs/experiment.yaml       the one editable config; frozen copies go in configs/frozen/
prompts/                      hashed drafting templates: scenario, group, repair
data/sources/                 reference-source registry, pinned downloads, inventory
                              (raw files in data/sources/raw/ are never committed)
data/topics/pilot_topics.yaml curated topic briefs: 12 pilot decisions, 4 per domain
data/topics/full_topics_v2.yaml  the full bank: 60 curated decisions, 20 per domain
data/pilot/                   marker allocation (committed); run artefacts (gitignored)
data/fixtures/                small synthetic corpus and briefs used by tests and the smoke test
src/reasonstyle/
  corpus/                     schemas, sources, topics, segmentation, validation,
                              annotation, review export
  generation/                 marker allocation, requests, backends, environment, log, provenance
  prompting/                  model-independent transcript rendering
scripts/                      every script takes a required --config
scripts/server/               Chomusuke02 setup and the vLLM launcher
docs/                         design notes, current status, proposals
tests/
```

## Setup

Python 3.11, managed by `uv`.

```bash
uv sync --group dev
```

Client dependencies are `pydantic`, `pyyaml`, `pysbd` and `pytest`. Libraries
for later stages (`torch`/`transformers`, `pandas`, `scikit-learn`, statistics)
are added when first needed. vLLM is installed only on the GPU host, by
`scripts/server/setup_chomusuke.sh`, and is not in `pyproject.toml`.

## Corpus workflow

Reference datasets supply topic ideas only; their text never enters a brief or a
prompt. A local open-weights model drafts from the curated briefs, and every
draft is machine-validated and then reviewed by a person.

| Step | Script | State |
|---|---|---|
| 1. Fetch and verify reference sources | `fetch_sources.py`, `verify_sources.py` | done |
| 2. Check the topic bank (overlap screen, 4 curated per domain) | `prepare_topic_bank.py` | done |
| 3. Allocate marker family, string and realization to all 48 groups | `allocate_markers.py` | done |
| 4. Server preflight, launch, one-call smoke test (`--kind group` or `--kind scenario`) | `preflight_model.py`, `server/serve_vllm.sh`, `smoke_test.py` | two draft-only group smokes run live (15 and 16 Sep 2026); the standalone `--kind scenario` command has not been run |
| 5. Draft the pilot's 24 scenarios (`pilot.py scenarios`), **stop for curator approval**, then draft its 48 groups with bounded repair (`pilot.py groups`) | `pilot.py` | scenario stage ran live on 2026-09-16 and all 24 final scenario texts are approved; the group stage ran live on 2026-09-17 — 140 calls, 2 groups machine-valid, 46 `needs_manual_review` |
| 5b. Redraft the scenarios the curator rejected, one call each | `pilot.py redraft-scenarios` | nine calls ran live on 2026-09-17; four redrafts were approved as returned and five were corrected in an audited, revalidated ledger |
| 5c. File-based request/response path, separate from the live runner | `emit_requests.py`, `import_responses.py` | built; not part of the two-stage live path |
| 6. Repair failing groups (≤2 repairs) | `pipeline_smoke.py` | built; run live twice on 2026-09-16, both ending `needs_manual_review`. The mechanism is confirmed: distinct, diagnosed repairs, unchanged output routed to review. No further synthetic repair smoke is planned |
| 6b. Assemble corpus JSONL and its manifest, corpus-wide | `pilot.py assemble` | built, offline-tested; writes atomically, refuses a partial pilot, validates before writing. Waiting on the 46 group corrections |
| 6c. Read the recorded group run, read-only | `pilot.py group-review` | built and run against the complete 2026-09-17 evidence |
| 7. Validate and export the pilot for human review | `export_for_review.py` | built |
| 8. Approve every scenario between the two stages, then assemble what passes | `pilot.py scenario-review`, `pilot.py approvals`, `pilot.py assemble` | scenario gate complete: 24/24 approved. The next gate is inspecting and correcting the 46 failed groups; assembly waits for that |
| 9. Extend to the full 60 decisions, then validate, review and freeze | — | not started |

*The table above records the v1 Qwen workflow as it stood on 2026-09-17 and is
historical.* A second generator, `gpt-5.6-sol` through OpenAI's Responses API,
**has since run**: the hosted v1 pilot on 2026-09-18 and the v2 pilot groups on
2026-09-20. Counts, per stage, are in [`docs/current_status.md`](docs/current_status.md).

## A second generator (hosted; since run)

*Written 2026-09-18, before it ran. Statements below that nothing has been sent
are superseded: see* Current workflow *above.*

The Qwen pilot finished with **2 of 48 groups machine-valid**; the dominant
failure is cross-condition length matching, and no repair produced an accepted
group. `gpt-5.6-sol`, through OpenAI's **Responses API**, is implemented as a
first-class backend so the same pilot can be drafted again under **exactly the
same rules** — the same hashed prompt templates, four conditions, validators,
1.10/1.15 word ratios, one-draft-plus-two-repairs budget, curator gate and
unconditional human-review codes. Nothing was relaxed to accommodate it, and a
test asserts the two configurations differ in the generator block and nowhere
else. **Nothing has been sent yet, and no claim is made about whether it helps:
that is what the new pilot, machine *and* human review, is for.**

| | |
|---|---|
| model | `gpt-5.6-sol` — the exact id; the moving alias `gpt-5.6` is refused at config load |
| API | Responses API, `https://api.openai.com/v1/responses`. Not a browser, not the ChatGPT UI |
| reasoning | `{"effort": "none"}` |
| sampling | temperature `0.3`; `top_p` left at the provider default and **not sent** |
| output | `max_output_tokens: 700`, strict JSON schema through `text.format` |
| state | `store: false`, `background: false`, no tools, no web search, no files, no conversation, no `previous_response_id` — every request stateless, exactly one draft |
| retention | `store: false` keeps the response out of retrievable Responses API state (no fetch by response id, no `previous_response_id` chaining). It is **not** a retention guarantee: abuse-monitoring retention may still apply under the account's data-control policy, and **Zero Data Retention is not claimed** |
| config | `configs/experiment_openai_pilot.yaml` (`openai_pilot_v1`), its own content hash |
| run directory | `data/pilot/run_openai/` — gitignored, and never the Qwen run or its snapshot |

The decoding settings are recorded as their own profile, `openai_responses_v1`,
**not** as if they were the Qwen configuration: the two share temperature 0.3
and a 700-token ceiling and nothing else, and no seed is sent to this API, so
none is recorded as though it had been. If the account later exposes a more
specifically pinned snapshot, the backend records the returned model id and
reports it for approval; it never follows it silently, and a model that is not a
snapshot of the requested one is refused.

**Authorisation.** A live hosted pilot stage needs three separate yeses —
`--send`, `REASONSTYLE_ALLOW_OPENAI_GENERATION=1` and
`REASONSTYLE_ALLOW_PILOT_GENERATION=1`. The local-generation key does not
authorise a paid external call, and this one does not authorise a local run.
**A dry run needs no key and opens no connection.**

**The credential** is read from `OPENAI_API_KEY` at the moment of the call and
nowhere else. It never reaches a request record, a log line, a raw file, an
error message, a printed line or a hash; the recorded payload is the complete
request body, and the Authorization header is built inside the backend and
discarded there. **There is no automatic retry at any level** — an invisible
retry is a second paid call for the same draft. A response that arrived is
always a recorded attempt that spent its budget position, truncated or malformed
included; only a failure that produced no response at all is raised for you to
decide about.

```bash
uv run python scripts/openai_smoke_test.py --config configs/experiment_openai_pilot.yaml
```

That is the dry run: it prints the exact payload and sends nothing. The
operating sequence — dry run, one smoke call, 24 scenarios, approval, 48
groups — is in [`docs/current_status.md`](docs/current_status.md) under *Hosted
generator*, with the call ceiling for each step.

## Local generator

`Qwen/Qwen3-14B`, non-thinking, served by vLLM on `127.0.0.1` from one A6000
on Chomusuke02 (temperature 0.3, top_p 0.8, 700 tokens, seed recorded; every
other sampling field sent at its neutral value, and the server started with
`--generation-config vllm` so the model's own defaults never apply). There is
no external service, no paid API and no API key. Offline mode is
required, so a missing model is an error, never a download. The weights'
commit SHA must be resolved from the local cache and recorded in
`models.generator.model.revision` before the server starts. A live call needs
both `--send` and `REASONSTYLE_ALLOW_LOCAL_GENERATION=1`. Omitting `--send` is
the dry run. A **pilot stage** (`pilot.py scenarios` or `pilot.py groups`)
needs a second key as well, `REASONSTYLE_ALLOW_PILOT_GENERATION=1`: drafting a
corpus is a separate decision from making one smoke call. Details: [`docs/local_generation_proposal.md`](docs/local_generation_proposal.md).
The earlier Anthropic proposal in `docs/drafting_proposal.md` is withdrawn.

On the server, in order:

```bash
HF_HUB_OFFLINE=1 HF_HOME=/data/$USER/hf_cache uv run python scripts/preflight_model.py --config configs/experiment.yaml
```

```bash
GPU=0 bash scripts/server/serve_vllm.sh
```

```bash
REASONSTYLE_ALLOW_LOCAL_GENERATION=1 HF_HUB_OFFLINE=1 HF_HOME=/data/$USER/hf_cache uv run python scripts/smoke_test.py --config configs/experiment.yaml --send
```

## Everyday commands

```bash
uv run pytest
```

```bash
uv run python scripts/smoke_test.py --config configs/experiment.yaml
```

```bash
uv run python scripts/export_for_review.py --config configs/experiment.yaml
```

The review export writes a read-only Markdown view of the corpus to `review/`: an
index, one file per decision with all four conditions side by side, a combined
searchable file, and blinded packets for the reliability annotators. Add
`--check` to verify it still matches the corpus. Judgements are recorded under
`data/`, never in the generated Markdown.

## Reading the recorded group run

```bash
uv run python scripts/pilot.py group-review --config configs/experiment.yaml \
    --out data/pilot/run/pilot_groups_snapshot_2026-09-17_complete
```

`pilot.py group-review` is a **read-only reporting command**. It makes no model
call, no repair, no approval and no correction, writes nothing into the run
directory and touches no corpus; it reads the recorded run and writes
deterministic Markdown to `--group-review-out` (default `review/pilot_groups/`).
It is unaffected by `--send` or by any generation environment key.

It shows all 48 expected groups, taken from the frozen marker allocation rather
than from whatever the log happens to contain, organised by decision, variant
and supported option, with the **2 machine-valid groups and the 46 requiring
correction kept clearly apart**. For each group: the scenario that currently
stands — model draft, accepted redraft, approved human correction — both
semantic options, the allocated marker family, string and realization, the final
call id, attempt number and outcome, the RS/RP/NS/NP bodies side by side, the
complete rendered counterargument including the shared opening, body and
full-text word and sentence counts, the exact errors and warnings with the
measurements behind them, and the outstanding human judgements. A failed group
also carries a compact attempt history — each attempt's call id, prompt hash,
whether it made progress, its four bodies and its findings — in which a repair
that returned unchanged text is labelled as such.

The index also carries one **informational** section: groups for which some
earlier attempt recorded strictly fewer distinct machine-error codes than the
final one, with both attempts' numbers, call ids and codes. It is a pointer and
nothing more — the final recorded attempt remains canonical, the flag selects and
approves nothing, and fewer machine-error codes does not mean better text, since
the codes measure form and every substantive judgement is still outstanding.

Output is an `index.md`, one page per scenario under `scenarios/`, a combined
searchable `all_groups.md`, and a `MANIFEST.json` that hashes every file. No
generated file carries a timestamp, so regenerating it produces identical bytes.
The findings and counts are recomputed with `validate_group` and the validator's
own measurement function, never with a second implementation.

**Inspection only.** It proposes no corrected wording and neither generates nor
populates `data/pilot/manual_corrections.yaml`: a correction is a separate,
separately approved record bound to the exact call and text it replaces, and
corrected text is re-validated by the same validator at assembly.

## What is and is not established

Machine validation checks form: presence, absence, counts, structural
consistency, pattern matches. Substantive support, support direction, no-reason
integrity, proposition preservation, naturalness and pragmatic commitment are
human judgements, and every generated review lists them as outstanding. A clean
validation run is not an approved corpus.

Generated responses, model weights, caches and server runtime files are never
committed.
