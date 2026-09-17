# Reason or the Language of Reason?

What actually moves an LLM's position when a user pushes back — the substantive
reason, or the language that makes something sound reasoned?

Vidhi Bhutani, University of Tokyo. The research plan is
[`Research_Plan_v6.md`](Research_Plan_v6.md); design decisions are in
[`docs/design_notes.md`](docs/design_notes.md); where the work stands is in
[`docs/current_status.md`](docs/current_status.md).

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
texts. Generating and reviewing it is the current milestone.

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
| 5. Draft the pilot's 24 scenarios (`pilot.py scenarios`), **stop for curator approval**, then draft its 48 groups with bounded repair (`pilot.py groups`) | `pilot.py` | two commands, never combined, whole pilot only; built and tested offline; no pilot call authorised yet |
| 5b. Redraft the scenarios the curator rejected, one call each | `pilot.py redraft-scenarios` | built, offline-tested; nine are pending from the 2026-09-17 review, not yet run live |
| 5c. File-based request/response path, separate from the live runner | `emit_requests.py`, `import_responses.py` | built; not part of the two-stage live path |
| 6. Repair failing groups (≤2 repairs) | `pipeline_smoke.py` | built; run live twice on 2026-09-16, both ending `needs_manual_review`. The mechanism is confirmed: distinct, diagnosed repairs, unchanged output routed to review. No further synthetic repair smoke is planned |
| 6b. Assemble corpus JSONL and its manifest, corpus-wide | `pilot.py assemble` | built, offline-tested; writes atomically, refuses a partial pilot, validates before writing |
| 7. Validate and export the pilot for human review | `export_for_review.py` | built |
| 8. Approve every scenario between the two stages, then assemble what passes | `pilot.py scenario-review`, `pilot.py approvals`, `pilot.py assemble` | built and offline-tested; no pilot call authorised yet |
| 9. Extend to the full 60 decisions, then validate, review and freeze | — | not started |

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

## What is and is not established

Machine validation checks form: presence, absence, counts, structural
consistency, pattern matches. Substantive support, support direction, no-reason
integrity, proposition preservation, naturalness and pragmatic commitment are
human judgements, and every generated review lists them as outstanding. A clean
validation run is not an approved corpus.

Generated responses, model weights, caches and server runtime files are never
committed.
