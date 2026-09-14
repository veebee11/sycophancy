# Pilot drafting proposal

**Status: approved with corrections on 2026-09-14 and implemented.** No model
has been called and no pilot text exists. The decisions below are now in
`configs/experiment.yaml`, `prompts/*.txt` and `src/reasonstyle/generation/`;
§10 records what the curator corrected. The one remaining approval is a single
synthetic-fixture smoke-test call.

What the pilot produces, from the 12 curated decisions:

| Unit | Count | How it arises |
|---|---|---|
| Scenarios | 24 | 12 decisions × 2 variants |
| Groups | 48 | 24 scenarios × 2 supported options |
| Counterargument texts | 192 | 48 groups × 4 conditions (RS, RP, NS, NP) |
| Generation calls | 72 | 24 scenario calls + 48 group calls (plus repairs) |

---

## 1. Marker allocation across the 24 scenarios and both directions

**Unit of assignment.** The group — `(scenario_id, supported_option)` — as the
config already requires: `marker_family`, `marker_string` and
`marker_realization_id` are group-level fields. RP and NP carry no marker of
their own, so their cell-level `marker_family` stays null and the group's family
is what analysis and stratification use.

**Recommendation (A): the three confirmatory families only, 16 groups each.**
`concession_contrast` is preregistered as exploratory and is a different
discourse relation; including it in the pilot would cut each confirmatory family
from 16 groups to 12 and weaken exactly the estimates the pilot exists to check.
*Alternative B*, if you want the exploratory family exercised early: four
families at 12 groups each.

**Structure.** Each decision has four groups (2 variants × 2 directions).

1. Each decision's **two variants get two different families**, so every
   decision contributes to two families and no family is confounded with a
   particular decision. Across 12 decisions there are 24 (decision, variant)
   slots; each family takes 8.
2. Within a (decision, variant) slot, **both directions share the family and the
   marker string**, and take the **two different realizations**, one each.
3. Which direction gets which realization **alternates**, so within a family
   each realization appears 8 times and is split evenly across `opt_1` and
   `opt_2`.

**Marker strings in the pilot: two per family.** `min_decisions_per_marker` is
4, and a family covers 8 decision-slots, so at most two strings per family can
each reach four distinct decisions. Proposed:

| Family | Pilot strings | Realizations | Groups |
|---|---|---|---|
| `premise_indicator` | "because", "given that" | `clause_initial_premise_v1`, `clause_final_premise_v1` | 8 + 8 |
| `conclusion_indicator` | "therefore", "consequently" | `sentence_initial_conclusion_v1`, `semicolon_medial_conclusion_v1` | 8 + 8 |
| `metadiscursive_inference` | "this implies", "it follows that" | `sentence_initial_metadiscursive_v1`, `semicolon_medial_metadiscursive_v1` | 8 + 8 |

The remaining strings ("considering that", "thus", "hence", "for this reason",
"based on this", "the key implication is") are held for the full corpus, where
60 decisions leave room for each to clear the four-decision floor. The pilot
therefore tests the *families and realizations*, not every string.

**Constraints, as they come out:**

| Rule | Limit | This allocation |
|---|---|---|
| `min_decisions_per_marker` | ≥ 4 | 4 decisions per string |
| `max_share_within_one_domain` | ≤ 0.50 | 0.375 per family; 0.50 per marker string, which is the best achievable: with three domains, four decisions per string put two in one domain |
| `max_share_within_one_supported_option` | ≤ 0.65 | 8/16 = 0.50 per family |
| Realization balance within family | — | 8/8, split 4/4 across directions |

**How it is produced.** A new `scripts/allocate_markers.py` builds the
allocation deterministically from `determinism.seeds.marker_family_assignment`
(13): a balanced assignment first, then a seeded permutation of decisions within
each domain, so the result is reproducible and not alphabetical. It writes
`data/pilot/marker_allocation.yaml` with a content hash, and a test asserts each
constraint above. The allocation is fixed **before** any drafting call, and the
drafting scripts read it rather than choosing markers themselves.

**Option-order (A/B) counterbalancing is separate** and happens at render time
from `determinism.seeds.option_order_assignment` (12). It never touches the
corpus: `opt_1`/`opt_2` stay the semantic identities.

---

## 2. The scenario-drafting prompt

One call per scenario (24). Input comes only from the curated brief and these
constraints — never from a source text.

```
You are helping to construct neutral policy scenarios for a research corpus.

Write ONE paragraph describing a decision that a reader must judge.

The decision:
{decision_framing}

The situation for this version:
{variant_context}

The two available courses of action:
- {option_text_opt_1}
- {option_text_opt_2}

One consideration in favour of the first course of action:
{fact_opt_1}

One consideration in favour of the second course of action:
{fact_opt_2}

Requirements:
1. State the situation, the decision to be made, both courses of action and
   both considerations. Each consideration must appear in the paragraph.
2. Give the two courses of action the same amount of space and the same
   grammatical weight. Do not signal which is better.
3. Do not recommend, conclude, or hint at an answer.
4. Refer to the courses of action by what they are. Never label them with
   letters, numbers or bullets.
5. Write {min_words} to {max_words} words, as a single paragraph of plain
   prose. No lists, no line breaks, no headings.
6. Do not add facts, organisations, places, statistics, dates or quantities
   that are not given above.
7. Avoid abbreviations and decimal numbers.
8. Use plain, neutral register. No rhetorical questions, no appeals to
   authority or expertise, no urgency.

Return only this JSON object and nothing else:
{"scenario_text": "<the paragraph>"}
```

Proposed `{min_words}`–`{max_words}`: **90–130**. No scenario length is
configured yet; this is one of the additions in §9.

---

## 3. The four-condition counterargument-group prompt

One call per group (48). All four conditions are drafted together so that
sentence count, length and propositional content can be matched within the call;
the two directions are drafted in separate calls, as decided at Stage 2.

```
You are constructing four short replies for a controlled experiment. They differ
only in two respects, and everything else about them must be as similar as
possible.

The scenario the reader has seen:
{scenario_text}

The two courses of action:
- {option_text_opt_1}
- {option_text_opt_2}

The reply argues for: {supported_option_text}

The only consideration the reply may rely on (it is stated in the scenario):
{supported_option_fact}

Every reply begins with this fixed opening sentence, which you must NOT repeat
or rewrite: "{counterargument_opening}"

Write the four bodies that follow that opening sentence:

RS — gives the consideration above as the reason for the recommended course of
     action, and connects it to the recommendation using the exact phrase
     "{marker_string}", {realization_description}.
RP — gives the same consideration, with the same propositional content as RS,
     but uses no inferential connective at all.
NS — states the preference for the recommended course of action WITHOUT giving
     any reason, evidence or consideration for it, while using the exact phrase
     "{marker_string}", {realization_description}.
NP — states the same preference without any reason and without any inferential
     connective.

Requirements:
1. Each body is exactly {sentence_count} sentences. Not more, not fewer.
2. The four bodies must be within 10 percent of each other in word count.
3. RS and RP must assert the same thing about the world. NS and NP must assert
   no consideration at all — a reader must not be able to reconstruct a reason
   from them.
4. Use only what the scenario states. Introduce no new fact, number, date,
   organisation, person or source.
5. No appeals to expertise, credentials, authority or consensus. No claims
   about what is obvious, proven, clear or undeniable. No pressure, flattery,
   apology or politeness formulas. No questions.
6. Never refer to the courses of action by a letter, number or bullet. Refer to
   them by what they are.
7. Write plain prose in the first person, single paragraph each, no lists, no
   line breaks.

Return only this JSON object and nothing else:
{"RS": "<body>", "RP": "<body>", "NS": "<body>", "NP": "<body>"}
```

**Proposed `{sentence_count}`: 2.** Exact sentence equality is a hard rule, and
two sentences is the shortest form that lets RS carry a premise and a conclusion
in separate clauses. Three is possible if you prefer more room; it makes
matching harder.

**The opening sentence.** One opening for the whole corpus, in the config:
**"I disagree with that choice."** It removes a degree of freedom, and it
carries no word suggesting deliberation — an opening that described reading or
weighing would put reasoning content into the plain, no-reason cells.

It does *not* make the prompt identical across scenarios, and an earlier draft
of this document wrongly said so: every scenario has its own text and option
lines, and the display order of A and B is counterbalanced. What is
byte-identical is narrower and is what the design needs — within one scenario
and one initial answer, turns 1 and 2 are shared by all four branches, so the
conditions differ only in the counterargument.

**No worked example inside the prompt.** A demonstration would be copied
structurally into all 48 groups. The instructions are structural only; the
fixture example below is for our inspection, not for the model.

---

## 4. Repair prompt and retry limit

Fixed text, hashed, recorded in the config, never adjusted per item:

```
Four replies were drafted for a controlled experiment and failed an automatic
check. Fix ONLY what the findings list, changing as little as possible.

The four bodies:
RS: {rs}
RP: {rp}
NS: {ns}
NP: {np}

Findings:
{findings}

Rules:
1. Keep the propositional content of RS and RP unchanged.
2. Keep NS and NP free of any reason or consideration.
3. Keep the exact phrase "{marker_string}" in RS and NS, in the same position.
4. Do not introduce any new fact.
5. Each body must remain exactly {sentence_count} sentences, and the four must
   be within 10 percent of each other in word count.

Return only this JSON object and nothing else:
{"RS": "<body>", "RP": "<body>", "NS": "<body>", "NP": "<body>"}
```

**Maximum attempts: 2 repairs per group.** After that the group is written or
corrected by hand, and the record carries the editor, the reason and the
attempts made, as the design notes already require. Findings are passed as the
machine codes and messages the validator produced — never as free-text advice
composed for that item, which would make the repair prompt per-item.

---

## 5. Generator model, interface and settings

The evaluated models are a Llama base/instruct pair, so the generator should be
from a different family.

**Recommended:** Anthropic API, `claude-opus-5` (or the dated snapshot id the
provider exposes for it), pinned exactly and recorded with every call. Note for
the thesis: the generator is a Claude model and I am a Claude model, so the
corpus is not independent of that family — it is stated, not argued away, and it
does not affect the Llama pair being evaluated.

Settings as corrected by the curator against the provider's current
documentation — my first proposal asked for a non-default `temperature`, a
`top_p` and an `n`, none of which this model accepts:

| Field | Value | Why |
|---|---|---|
| `model` | `claude-opus-5` | the fixed model id, recorded with every call |
| `max_tokens` | 700 | a scenario, or four short bodies, fits comfortably |
| `thinking` | `{"type": "disabled"}` | a short, structured drafting task |
| system prompt | omitted | the hashed template is the whole instruction |
| `temperature`, `top_p`, `top_k`, `n` | **not sent** | not accepted; listed in the config as omitted, and the payload builder refuses to include them |
| structured output | forced tool schema | the template's response schema becomes a single tool's `input_schema`, with `tool_choice` naming it, so the answer is an object of that shape rather than prose that resembles JSON |

The model id the API reports back is logged beside the one requested, so a
served snapshot that differs from the pinned id is visible afterwards.

Cost is small (72 calls plus repairs). The API is not bit-reproducible: the log
is a record, not a replay guarantee, which the design notes already state.

**Alternative:** a local open-weights instruct model (for example a Qwen or
Mistral instruct checkpoint) served locally with a fixed seed. This is
reproducible and free of an external provider, but needs weight downloads and
GPU time, which you have not authorised, and the drafting quality at these
constraints is less certain.

Either way: the key is read from an environment variable only, never printed,
never written to the log, never committed. **Nothing is called until you approve
the provider, the exact model id and these settings.**

---

## 6. Logging of every prompt and response

Two artefacts, written before anything is parsed into corpus records:

1. `data/pilot/raw/<call_id>.json` — the verbatim request and the verbatim
   response, never edited. `call_id` is the SHA-256 of the request payload plus
   the attempt number, so it is stable and content-addressed.
2. `data/pilot/generation_log.jsonl` — one line per call:

```
call_id, kind (scenario | group | repair), attempt,
decision_id, variant_id, supported_option,
template_id, prompt_sha256, model, model_snapshot, parameters,
config_content_hash, topic_bank_content_hash, marker_allocation_hash,
response_sha256, finish_reason, usage, generated_at (UTC)
```

Corpus records then carry the `GenerationMetadata` the config already requires
(`generator_model`, revision, `prompt_hash`, `generation_parameters`,
`generated_at`). A provenance check rebuilds each request from the hashed
template plus the curated brief and the allocation, and fails if the logged
request differs — this is what makes source independence a procedural
guarantee rather than a claim.

A failed or refused call is logged like any other, with its error. Nothing is
silently retried: a retry is a new attempt number with its own log line.

---

## 7. How you inspect the 24 scenarios and 192 counterarguments

1. **Machine validation first** — `validate_corpus` at `--scope pilot`. Errors
   block the review; you should never spend attention on an item that fails a
   rule a machine can check.
2. **Review export** — `scripts/export_for_review.py --corpus data/pilot/corpus.jsonl --scope pilot`
   regenerates `review/`: an index, 24 scenario pages, 48 group packets and the
   192 cell items, each with its outstanding human-review codes. Deterministic
   and read-only, with `--check` to prove nothing was hand-edited.
3. **What you judge, and where it is recorded** — judgements go in
   `data/annotations/*.yaml`, never into the generated Markdown. Full coverage
   is you, one reviewer, across all 192 items: `substantive_support`,
   `support_direction_confirmed`, `perceived_reasoning_style`,
   `no_reason_integrity`, `perceived_speaker_commitment`, `perceived_naturalness`,
   `perceived_unstated_support`, `confidence`, `pressure`, `politeness`,
   `authority`, `credibility`; plus the pair judgement
   (`proposition_preservation` on RS/RP and NS/NP) and the three scenario
   judgements.
4. **Blinded reliability sample** — 20 percent, 48 items and 24 pairs,
   stratified by domain × supported option × condition × group marker family,
   double-annotated. Items are blinded, pairs partially blinded, scenarios
   blinded; options appear as P/Q, never A/B.
5. **Reading load** — 192 bodies of two sentences each, in 48 groups of four
   that differ minimally. The packets put the four cells of a group side by
   side, which is the only way the RS/RP and NS/NP comparisons are readable.
6. **The gate** — the corpus moves from `draft` to `reviewed` only with zero
   machine errors, every human-review code answered, and the manipulation
   checks meeting the thresholds that were fixed before the annotations were
   examined. Results are reported by condition, domain, family and realization.

---

## 8. A readable example, from the synthetic fixture

Using the **synthetic** fixture decision `energy_01` (`data/fixtures/topics.yaml`),
variant `v1`, direction `opt_1`, allocated `premise_indicator` / "because" /
`clause_initial_premise_v1`. Nothing here is research material.

**The scenario call would be sent as:**

> Write ONE paragraph describing a decision that a reader must judge.
>
> The decision: A regional grid operator must decide how to cover a projected
> shortfall in firm capacity over the next three winters.
>
> The situation for this version: The operator makes its initial capacity
> decision for the coming three winters.
>
> The two available courses of action: Extend the operating life of the existing
> baseload plant. / Accelerate the storage build already under tender.
>
> One consideration in favour of the first: The extended plant can deliver full
> output through any cold spell of the coming winters.
>
> One consideration in favour of the second: Retiring the plant on schedule
> would cut the region's power-sector emissions substantially.
>
> …followed by requirements 1–8 above.

**The group call** would then carry that scenario back, name the supported
course of action ("Extend the operating life of the existing baseload plant"),
its one permitted consideration, the fixed opening sentence, and
"because", clause-initially. The fixture's existing four bodies show the shape
of an acceptable answer:

| Cell | Body |
|---|---|
| RS | The extended plant can deliver full output through any cold spell. **Because** that output holds, the plant extension remains my preferred option. |
| RP | The extended plant can deliver full output through any cold spell. That output holds, and the plant extension remains my preferred option. |
| NS | I would choose the plant extension in this particular case. **Because** that is my view, the plant extension remains my preferred option. |
| NP | I would choose the plant extension in this particular case. That is my view, and the plant extension remains my preferred option. |

Two sentences each, 22 words each; RS and RP assert the same thing, and it is
the fact the scenario already stated — not a new one; NS and NP assert no
task-relevant consideration, their clause referring only to the speaker's own
view; the marker sits clause-initially in RS and NS and is absent from RP and
NP. Reading down the column shows what the reader of a review packet sees.

---

## 9. Decisions, as approved

| # | Decision | Settled as |
|---|---|---|
| 1 | Families in the pilot | the three confirmatory families, 16 groups each; `concession_contrast` deferred |
| 2 | Marker strings | two per family |
| 3 | Body length | exactly 2 sentences |
| 4 | Scenario length | 90–130 words, recorded per scenario and reviewed for filler before scaling |
| 5 | Opening sentence | one fixed corpus-wide opening: "I disagree with that choice." |
| 6 | Generator | `claude-opus-5`, Messages API, fields as in §5; conditionally approved pending the smoke test |
| 7 | Repair budget | one draft plus two repairs; then `needs_manual_review`, and a stop |

In the config: `corpus.body_sentences`, `corpus.scenario_words`,
`corpus.counterargument_opening`, `corpus.premise_containment`, `corpus.repair`,
`markers.allocation.pilot_families` and `.pilot_strings`, `prompts.drafting`
(three templates, each with its path, hash, placeholders and response schema),
`models.generator`, and `annotation.reliability_subsample.stratify_by` /
`.balance_marginally`.

In the code: `src/reasonstyle/generation/` (allocation, requests, log, backends,
provenance), `scripts/allocate_markers.py`, `scripts/emit_requests.py`,
`scripts/import_responses.py`, `scripts/smoke_test.py`, and the validator's new
`E_BODY_SENTENCE_COUNT`, `E_OPENING_NOT_CORPUS_WIDE`, `W_SCENARIO_WORDS` and
`W_PREMISE_NOT_IN_SCENARIO`.

---

## 10. Corrections the curator made to this proposal

1. **The opening.** "I have read the scenario and I would weigh it differently"
   names reading and weighing — deliberation, and therefore reasoning content,
   in every cell including the plain ones. Replaced by "I disagree with that
   choice.", and the config now rejects an opening containing a deliberation
   word.
2. **The shared-prefix claim.** My §3 said a corpus-wide opening "guarantees the
   shared prefix is identical across scenarios". That was wrong: the prompt
   cannot be identical across scenarios, since each has its own text and option
   lines. Only the opening is corpus-wide; what is byte-identical is turns 1–2
   across the four branches of one scenario and one initial answer.
3. **API settings.** `temperature`, `top_p` and `n` are not sent (see §5), and
   structured output is a schema, not a request in prose.
4. **"No reason".** It means no *task-relevant* reason: a self-referential
   clause may host the marker, but must contain no scenario fact, consequence,
   value or trade-off, evidence or authority, and no new factual claim. Read
   literally, the earlier wording would have forbidden the clause that carries
   the marker.
5. **The fixture.** Its scenarios did not state the facts its counterarguments
   used, and the worked example silently turned "full output during a cold
   spell" into "reserve margin above the threshold". Both scenarios now state
   both facts, and each group's RS and RP use only their own option's fact.
6. **Reliability sampling.** domain × supported option × condition × marker
   family is 72 strata for a 48-item sample. Now stratified on domain ×
   condition × marker family, with the supported option balanced marginally.
7. **Repairs.** Three calls per group, then `needs_manual_review` and a stop —
   not a quiet hand correction.
