# Decision log

Research choices that `Research_Plan_v6.md` left open, contradicted, or that
were deliberately changed. Each entry records the resolution actually
implemented and where it is enforced.

**Rule:** a resolution changes only through a new config version. Configuration
files are never edited in place once they have produced an artefact.

Status key: **frozen** = settled for v1; **open** = revisitable at the named gate.

---

## D1 — Sentence-count matching · frozen · *deviation from plan interpretation*

**Problem.** §6.2 requires all four cells matched on sentence count while the
style transformation "may change punctuation and function words". The canonical
plain→explicit edit merges two sentences into one, so the two requirements
appeared to conflict.

**Resolution (supervisor, overruling the ±1 proposal).** Exact sentence-count
equality across RS, RP, NS, NP within each `(scenario_id, supported_option)`
group is a **hard requirement**. Sentence count is itself a surface-form
confound, so it is controlled rather than recorded as a nuisance covariate.
Generation must produce naturally matched constructions — explicit framing can
be achieved within the same sentence count via connectives, semicolons or
sentence-initial markers. A semicolon does not terminate a sentence.
Punctuation may differ; the number of sentences may not.

Naturalness is human-rated (`perceived_naturalness`, D8). If exact matching
repeatedly yields unnatural language in the pilot, that triggers a **documented
config revision**, never a silent relaxation.

*Enforced:* `matching.sentences` (`rule: exact_equality`, `tolerance: 0`,
`semicolon_terminates_sentence: false`), checked at config load; validator
implements the count at Stage 2.

**Sub-issue — sentence segmentation (resolved in principle, implemented at Stage 2).**
Human confirmation cannot replace deterministic validation. Stage 2 must:

1. choose and **pin one deterministic segmentation implementation**, recording its version;
2. restrict generated texts to avoid unnecessary abbreviations, decimal-heavy constructions and bullet formatting;
3. allow human reviewers to **flag** segmentation errors;
4. **never silently override the machine count** — an override requires a recorded annotation.

## D2 — Word-count tolerance · frozen

§6.2's "±10% target, ±15% hard maximum" had no stated baseline. Within each
four-cell group: `max_words / min_words` — warn above **1.10**, fail above
**1.15**. Words are counted with a frozen regex. Model token counts are
recorded as diagnostics and **never** substitute for the word-count rule.

*Enforced:* `matching.words`; load-time check that warn ≤ fail and that both
equal their frozen values.

## D3 — Marker-family representation · frozen

§15.1 sets `marker_family: null` on plain cells, which would have made the
held-out-marker-family probe control (§7.3) undefined for half the corpus.

One intended family is assigned at the `(scenario_id, supported_option)` group
level and used across all four cells. RS and NS carry it directly. RP and NP
keep `marker_family: null` but expose `inherited_marker_family` for grouping and
held-out-family evaluation, plus `markers_present: false`. Families are balanced
across domains, support directions and splits. **The held-out-family split keeps
the complete four-cell group together.**

*Enforced:* `markers.assignment`; load-time check that `styled_cells` /
`plain_cells` partition the core conditions and agree with each cell's style
level.

### D3a — Revised marker taxonomy (Stage 1 review) · frozen

The v1 taxonomy mixed discourse relations and overlapped between `inferential`
and `premise_conclusion_framing`. Replaced by 16 markers in 4 families:

| Family | Role | Markers |
|---|---|---|
| `premise_indicator` | confirmatory | because, given that, considering that |
| `conclusion_indicator` | confirmatory | therefore, thus, hence, consequently, for this reason |
| `metadiscursive_inference` | confirmatory | this implies, it follows that, based on this, the key implication is |
| `concession_contrast` | **exploratory** | however, even so, nevertheless, despite this |

`although` is **removed**: it typically forces clause restructuring, which
conflicts with D1's exact sentence-count matching.

Concession marks a substantively different discourse relation from premise and
conclusion indication, so it is a **preregistered exploratory/robustness family
only** and cannot enter the confirmatory style estimate.

The analysis must report family-specific effects, test a family × style
interaction, pool families **only if their effects are directionally
compatible**, and support leave-one-marker-out and
leave-one-realization-template-out evaluation. `marker_realization_id` is
recorded per styled cell to make the last of these possible.

**A lexical match shows a marker is present; it does not show the marker
performed its intended discourse function.** Only human validation establishes
that. Encoded as `lexical_match_establishes_discourse_function: false`, checked
at load.

### D3b — Style frozen at the realization, not the family · frozen

Sharing a marker *family* is insufficient. If RS used "therefore" and NS used
"consequently", or the two placed their marker differently, the content contrast
**RS − NS would also be a marker contrast**. Three fields are therefore stored
once at `(scenario_id, supported_option)` group level and inherited by all four
cells:

```
marker_family:         conclusion_indicator
marker_string:         therefore
marker_realization_id: sentence_initial_conclusion_v1
```

- RS and NS instantiate the **same** marker realization; `markers_present: true`.
- RP and NP instantiate its **paired plain transformation**; `markers_present: false`.

*Enforced:* `markers.assignment.group_level_fields`,
`same_realization_across_cells`, `cell_inheritance`; checked at load. The
per-record assertion that all four cells belong to one realization group is
owed at Stage 2.

**Pilot gate.** A configured marker does not prove the resulting text is
reason-free. The pilot must establish, *for each realization*, that a genuine NS
instance can be produced without smuggling in a premise. Added to
`annotation.human_review_required`.

### D3c — Marker allocation minima · frozen, values configurable

So that a nominal held-out-marker analysis can never run on one or two
decisions:

| Constraint | v1 value |
|---|---|
| `min_decisions_per_marker` | 4 |
| `min_decisions_per_leave_one_marker_test` | 4 |
| `min_decisions_per_held_out_family_test` | 8 |
| `max_share_within_one_domain` | 0.50 |
| `max_share_within_one_supported_option` | 0.65 |
| `confirmatory_families_in_every_split` | true |
| `report_as_corpus_statistic` | true |

With 240 groups over 16 markers (~15 groups per marker) these are slack rather
than binding. They are checked at corpus construction and **reported as corpus
statistics** either way. Load-time checks reject a minimum below 2 and a
per-domain cap below an even split.

## D4 — Near-tie threshold τ · frozen · *deviation from plan*

§6.4 freezes τ "from the pilot", which creates avoidable researcher discretion
over an exclusion rule. τ is instead fixed **a priori on an interpretable
probability scale**: a near tie is an initial conditional probability of the
preferred option below **0.60**, i.e. `|m_before| < log(0.60/0.40) =
0.4054651081`. Robustness-only; the full sample stays primary.

*Enforced:* `near_tie`; load-time check that `tau_logit == log(p/(1−p))` within
1e-9, and that `primary_analysis_excludes` is false. Changing p without
recomputing τ fails to load.

## D5 — Semantic options vs display labels · frozen

`opt_1` / `opt_2` are the only semantic identities. A/B exist solely as renderer
output. Leakage detection rejects **display-label references** ("option A",
"choice B", "select A", `A.`, `B)`) but **not** a bare letter, since "A" is also
an English article. Support direction is never inferred from letters in text; it
comes from `supported_option` plus human review.

*Enforced:* `leakage` (`bare_letter_rejected: false`); tested against both
positive and negative cases.

## D6 — Shared opening · frozen

Stored **once per `scenario_id`** (not per supported option) and prepended to
all eight counterarguments. It must not mention either option, a condition,
evidence, expertise, certainty or authority. Checked by exact equality after
rendering.

**Frozen for v1** (no longer open). Any later change requires a config version
bump after the pilot.

## D7 — Forbidden phrases: collision, severity and fixtures · frozen

Forbidden phrases are word-boundary-aware regexes matched case-insensitively. A
load-time check asserts that **no permitted marker is matched by any forbidden
or leakage pattern**, in raw, lower, upper and capitalised forms.

**Severity (Stage 1 review).** 25 patterns: **18 `hard_fail`**, **7 `warning`**.

- `hard_fail` is reserved for unambiguous **authority**, **evidence**, **consensus** and **pressure** phrasing.
- `warning` covers broad patterns liable to false positives. All four **certainty** patterns are warnings ("clearly defined", "certainly" appear innocently). Three further patterns are warnings because a regex cannot tell **scenario-grounded** from **external** evidence: bare **`authorities`** ("the permitting authorities set a deadline"), **`proven`** ("proven technology", "proven reserves") and **`empirical`** ("empirically estimated demand", "empirical performance") — all three may refer to information the scenario itself supplies.
- Unsupported appeals — `studies show`, `experts agree`, `research suggests`, `consensus` — remain hard failures.
- `auth_speaker_credential` accepts 0–3 modifiers, so "as a senior energy economist", "as an experienced urban planner" and "as a technology policy expert" are caught, while the idiom "as a result … the analyst" is excluded by a negative lookahead.
- All patterns compile case-insensitively; asserted at load.
- Warnings route to human review rather than auto-rejecting.

**Fixtures bind.** Every pattern carries positive and negative examples,
verified at config load. A pattern cannot be quietly weakened to resolve a
marker collision (its positive fixture would stop matching) nor broadened until
it swallows ordinary policy language (its negative fixture would start
matching).

**Screening only.** `forbidden.screening_note` states in the config that regex
matching flags phrases and **cannot establish the semantic absence** of
evidential, authoritative or coercive content. Human review (§6.5) remains
authoritative.

## D8 — Human annotations and the pragmatic reading · frozen · *addition beyond plan*

§6.5's rubric has no speaker-commitment item, which is exactly the field that
adjudicates the benign explanation of an NS−NP effect. Three ratings are added
to the plan's list:

- `perceived_speaker_commitment` — does the wording signal stronger speaker commitment?
- `perceived_naturalness` — required by D1's exact sentence matching.
- `perceived_unstated_support` — does the wording imply the speaker holds reasons or evidence not actually stated?

An NS−NP effect initially licenses only the claim that **inferential framing
affects the model without additional stated propositional support**. It does not
establish irrationality or that the model mistakes form for reasoning.

The diagnostic-condition registry is **empty in v1** but present, so a later
commitment-matched control condition can be added without touching the four core
conditions or the four core contrasts.

*Enforced:* `annotation.ratings.additions_v1`, `interpretation`,
`conditions.diagnostic`; tests assert a diagnostic condition loads *and* can
never enter a core contrast.

## D9 — Behavioural margin arithmetic · frozen (documentation)

- `m_before ≤ 0` by construction, apart from exact ties, since `i = argmax`.
- `m_before` is **shared** by the four branches of a model × scenario × option-order run.
- Therefore the four within-group factorial contrasts are **numerically identical** computed on `movement` or on `m_after`.
- `movement_toward_counter` remains the stored primary outcome: it is the interpretable quantity, and it is what cross-scenario modelling, initial-margin adjustment and near-tie robustness use.

*Enforced:* `analysis.margin_notes`. A unit test demonstrating the identity is
**owed at Stage 4**.

## D10 — Template-dependent answer tokens · frozen

Single-next-token validity is a property of *(model revision, tokenizer
revision, rendered template, answer continuation)* — `"A"` and `" A"` are
different tokens — not of the tokenizer alone. v1 stores display labels,
per-template answer-continuation fields, `answer_token_ids: null` and
`verification_status: unverified`. **Nothing may claim single-token validity
before Stage 5 verifies it.**

*Enforced:* load-time check that an `unverified` template has null token ids,
and that a `verified` template pins ids, template hash and verification date.

## D11 — Layer indexing and hook point · frozen

Blocks indexed `0..n_layers-1`; primary hook `resid_post` (residual stream after
the indexed block). Embeddings are **not** analysed in v1; if added later they
are named `embed`, never a negative block index. Library-specific hook strings
and the `hidden_states` mapping stay null until the model adapter exists, while
the conceptual site is frozen now.

*Enforced:* `mechanistic.layers`; checked at load.

## D13 — Annotation levels · frozen

Judgements are stored at the level at which they are actually made. Duplicating
a pair- or scenario-level judgement onto every item would inflate apparent
sample size and let one judgement disagree with itself.

| Level | Keys | Ratings |
|---|---|---|
| **item** | scenario_id, supported_option, condition, annotator_id | substantive_support, perceived_reasoning_style, no_reason_integrity, perceived_speaker_commitment, perceived_naturalness, perceived_unstated_support, confidence, pressure, politeness, authority, credibility |
| **pair** | scenario_id, supported_option, pair_id, annotator_id | proposition_preservation, over pairs `[RS, RP]` and `[NS, NP]` |
| **scenario** | scenario_id, annotator_id | option_feasibility, option_non_dominance, normative_underdetermination |

The plan's single `scenario_validity` item is split into its three constituent
judgements so each can be rated and reported separately.

Denormalization is `false` by default and permitted only in an export
explicitly flagged as flat. Load-time checks reject a rating declared at two
levels, a rating without provenance, a `condition` key on a pair or scenario
level, and a preservation pair that crosses the reason factor (RS/NS is not a
proposition-preservation pair).

## D14 — Sentence segmentation · frozen (Implementation Stage 2a)

**`pysbd`, pinned at the resolved version** (`>=0.3.4,<0.4`; resolved 0.3.4).
Rule-based and deterministic — no statistical model, no data download, no
network — and zero transitive dependencies.

Why not the Stage 1 placeholder regex: D1 makes exact sentence-count equality a
*hard rejection criterion*, so the segmenter decides admissibility. A regex
miscounts this domain's constructions (`In the U.S. Storage costs fell`,
`40.5 GW`, `approx.`), and its errors would correlate with the style factor —
styled cells carry semicolons and connectives at different rates — biasing the
very confound D1 exists to control.

**It is still a heuristic, and the config says so.** pysbd handles semicolons,
decimals and `e.g.` correctly but mis-segments `U.S.` and `approx.`. Therefore:

- the resolved version is pinned in config and recorded on every `Segmentation`; a mismatch between config and installed version **refuses to run**;
- `clean: false` — the segmenter never rewrites the text it measures;
- unreliable constructions are **flagged**, not trusted: decimals, initialisms, abbreviations, bullet and numbered lists, line breaks;
- generated text is restricted to avoid them, and the restriction list is the single source of truth (the abbreviation detector is *generated* from it, never stored as a second regex);
- **the machine count is authoritative.** A reviewer may flag a suspected error; a correction requires a recorded annotation. Never a silent override.

**Regression-guarded:** a semicolon does not terminate a sentence. D1 permits a
semicolon as the device achieving explicit framing inside one sentence, so a
segmenter that split there would make styled and plain cells impossible to
length-match. Asserted in config and tested on four realizations.

## D15 — Corpus provenance · frozen

Two things kept deliberately apart:

- **`source_references`** — an **open, extensible list**, not an enum. Fields: `dataset_name` (required), `dataset_version`, `source_item_id`, `source_url`, `access_date`, `reuse_licence`, `notes`. A scenario may cite several sources; no dataset is privileged, and a new source needs no schema change. A fully constructed scenario carries `source_type: constructed` and an **empty list**.
- **`generation_metadata`** — how an LLM produced a *draft*: `generator_model`, `generator_model_revision`, `prompt_hash`, `generation_parameters`, `seed`, `generated_at`.

Generation metadata is never a source reference and never a licence claim.
Field sets are checked to be disjoint.

## D16 — Model-agnostic schemas · frozen

No model identity appears in any schema, validator or config-declared behaviour.
Config v2 sets `models.selection_status: unfrozen` with `repo_id` and `revision`
null; a `frozen` status must pin both. A test greps the config for vendor and
library names to keep it that way. Selection happens only after the
tokenizer/template compatibility test.

## Note on what controls RS − NS

Correcting a statement made during Implementation Stage 2 design: marker
*absence* is **not** what controls `RS − NS`. RS and NS are both styled, with
the **same marker realization**, which therefore cancels — that is what makes
`RS − NS` a content contrast. Marker absence is what *defines* RP and NP,
making **`RP − NP`** the corresponding content contrast in plain language.

## D17 — Schema shape decisions · frozen (Implementation Stage 2b)

**Terminology.** The synthetic fixture is described as **machine-valid**
(equivalently *structurally valid*): it satisfies the schema and every lexical
corpus rule a validator can check. It is never called "valid" without
qualification, because substantive support, no-reason integrity, proposition
preservation, naturalness and pragmatic commitment remain human judgements.

**No-reason cells carry no comparative property.** All four cells of a group
end with the same endorsement clause ("... remains my preferred option"), which
asserts a preference and nothing else. An earlier draft ended NS and NP with
"is the safer path", which smuggles safety in as a new comparative advantage —
the exact failure the no-reason condition must avoid. RS and RP add a scenario
premise; NS and NP add none, so `RS - NS` isolates the premise.

**Marker family vs marker realization are separate ID namespaces.**
`premise_indicator`, `conclusion_indicator`, `metadiscursive_inference` and
`concession_contrast` are **family** ids, matching `markers.primary_families`
and `markers.exploratory_families` in config v2 (per D3a, which renamed the
plan's original prose family names). `clause_initial_premise_v1`,
`semicolon_medial_conclusion_v1` and the rest are **realization** ids matching
`markers.realization.registry`. A test asserts the two sets are disjoint.

- **`Cell.body` excludes the shared opening.** The opening is stored once on the scenario (D6) and prepended by `ScenarioRecord.render()`, so exact equality across the eight texts holds *by construction* rather than by check.
- **`Measurements` records full and body counts for both words and sentences**, so no reader has to guess which text a count refers to. D1's sentence equality is unaffected by the choice, since the opening adds the same number of sentences to all eight.
- **`scenario_id` must be `<decision_id>_v<variant_id>`**, checked against both fields. A scenario cannot silently belong to the wrong decision or variant.
- **`Cell.markers_present` is checked against `FROZEN_CORE_CONDITIONS`**, not the config file, so schemas stay standalone and cannot disagree with the code-level design guard.
- **`domain` is a plain string in the schema**, checked against `config.domains.ids` by the validator (2c). Domains are a configured research choice, so they are not a hard-coded `Literal`.
- **`source_type` and `source_references` are cross-checked**: `constructed` requires an empty list; `adapted` and `mixed` require at least one reference.
- **`ValidationStatus` is self-consistent**: any status past `draft` requires machine counts; `human_reviewed`/`approved` require a reviewer and date; `approved` is impossible with machine errors or outstanding `H_` review codes. Machine validation is not approval.
- **Records are frozen** (`frozen=True`, `extra="forbid"`); an unknown field is a load error, not a silently kept extra.

## D18 — Validator severities and scope · frozen (Implementation Stage 2c)

**Four severities.** `error` breaks a machine-checkable rule and blocks the
item. `warning` routes to a reviewer without blocking. `info` records a check
that did not apply at this corpus scope, so its absence is visible rather than
silent. And:

**`human_review` is not a defect.** It is emitted **unconditionally** for every
item, pair, group and scenario, so a clean machine run can never be mistaken for
a validated corpus. Eight codes, at the level the judgement is actually made
(D13): `H_SUPPORT_DIRECTION`, `H_SUBSTANTIVE_SUPPORT`, `H_NATURALNESS`,
`H_PRAGMATIC_COMMITMENT` per cell; `H_NO_REASON_INTEGRITY` per NS/NP cell;
`H_PROPOSITION_PRESERVATION` per RS/RP and NS/NP pair;
`H_REALIZATION_YIELDS_REASON_FREE_NS` per group; `H_SCENARIO_VALIDITY` per
scenario. `ValidationReport.ok` therefore means "no machine errors" and the
summary says so in words.

**Support direction is never determined lexically.** The validator checks only
that the metadata is structurally consistent and that no display label leaks
into the text. Direction itself comes from `supported_option` plus human review.

**Corpus scope gates corpus-level checks.** `fixture | pilot | full`. Marker
allocation minima cannot be satisfied by a one-decision fixture, so outside
`full` they emit `I_ALLOCATION_SKIPPED` naming exactly what was not run. The
same fixture validated at `full` scope correctly fails.

**Text restrictions map to severity by their config verb.** `prohibit_*` and
`single_paragraph` produce `E_PROHIBITED_FORMATTING`; `avoid_*` (decimals,
abbreviations, initialisms) produce `W_AMBIGUOUS_SEGMENTATION`, because the
machine sentence count stays authoritative and only needs confirming.

**Invalid fixtures isolate one rule each.** 17 fixtures; each asserts a single
error code, so a regression cannot hide inside a cascade. One exemption:
`prohibited_formatting`, where a bullet list unavoidably breaks the length,
sentence and marker rules at once. `word_ratio_warn` exists to demonstrate the
D2 decision — its bodies differ by 1.143 while its full texts differ by only
1.094, so the body measurement fires and the full-text measurement does not.

## D19 — Human-review export · frozen (Implementation Stage 3b)

A deterministic, **read-only** Markdown view of the canonical JSONL. It exists so
the corpus can be read and verified without inspecting raw JSONL, and it is not a
second source of truth: judgements go to `data/annotations/`, and nothing written
into the Markdown is ever read back.

**Determinism.** No generated file — `MANIFEST.json` included — carries a
wall-clock timestamp; the only permitted time value is `corpus.freeze_timestamp`,
itself frozen configuration. `--check` regenerates into a temp directory and
exits non-zero on any drift, so a hand edit is detected rather than absorbed.

**Corpus identity.** The hash printed in every file is the hash of the corpus
**as stored**, taken from the validation report. Derived measurements must never
change it, or the hash shown in the review would not match the JSONL reviewed.
`MANIFEST.json` also records the source file's byte sha256.

**Two audiences.** The *curator view* is fully labelled: RS/RP/NS/NP shown
plainly with marker family, string and realization, measurements, machine
findings and outstanding human-review codes. Only the *reliability packets* are
blinded.

**Blinding, at all three levels.** Item packets show one counterargument at a
time and ask which option it supports — **P, Q or unclear** — plus whether it
supplies a premise, a question answerable without knowing the condition. Pair
packets show two texts in randomised left/right order with no condition labels.
Scenario packets show the scenario and both options with no counterarguments.
Hidden throughout: condition, marker metadata, measurements, sibling cells,
machine findings and the intended supported option. Option labels are **P/Q**,
never A/B, so blinded review can never be confused with the experiment's display
labels. The unblinding key lives in `blind_key/`, a separate directory.

**Sibling separation.** Two cells of one group differ only by a connective, so an
annotator seeing them close together can infer the design.
`min_sibling_separation: 20` is enforced during the seeded shuffle. When it is
infeasible — as on the 16-text fixture — the exporter **reports it in the packet,
in the manifest and on stdout, and never silently relaxes it**.

**Reliability packets.** Same items for every annotator, as Cohen's κ requires;
independently seeded order per annotator to control order effects.

**Corpus-independent.** The exporter groups whatever records it is given by
`decision_id`; a test adds a second decision in a second domain and asserts it
appears with no code change.

## D19a — Item review is blinded; pair review is only PARTIALLY blinded · frozen

A proposition-preservation judgement requires both texts to be visible, and the
two texts of a pair differ only by a connective. An annotator therefore sees
immediately that it is a minimal-pair manipulation, even with the condition
labels removed and the sides randomised.

This is inherent to the judgement, not a defect in the exporter, but it means the
three levels are **not equally blind**:

| Level | Blinding | What the annotator can infer |
|---|---|---|
| **item** | blinded | one text alone; condition, marker metadata, counts, sibling cells and the intended option are all hidden |
| **pair** | **partially blinded** | both texts shown; condition labels hidden and sides randomised, but the minimal-pair structure is visible |
| **scenario** | blinded | scenario and options only, no counterarguments |

**The methodology must describe item-level review as blinded and pair-level
review as partially blinded**, and must not report the pair-level reliability
figure as though it were obtained under the same blinding as item-level.

The sibling-separation constraint (D19) mitigates but does not remove this: it
keeps cells of one group apart in the *item* packet, where they would otherwise
reveal the design; it cannot help in the pair packet, where both cells are shown
together by design.

**On the real corpus the configured separation stays enforced.** If it is
infeasible, the exporter reports it — in the packet, the manifest and on stdout —
and the constraint requires explicit approval to change. It is never silently
relaxed.

## D20 — Configuration resolution is explicit · frozen

`latest_config_path()` is a **development convenience**, used by general tests so
a version bump does not require editing every call site. Any command that
generates pilot, full-corpus or experimental artefacts takes a **required
`--config` path** and prints the resolved version and hash: those artefacts embed
the config hash, so the version must be chosen deliberately rather than inherited
from whichever file is newest. A test greps `scripts/` and fails if any script
resolves the config implicitly.

Superseded configurations stay on disk, loadable and byte-stable. Their content
hashes are pinned in `FROZEN_CONFIG_HASHES`, and a test fails if any of them
changes — a config that produced an artefact is the record of what that artefact
was made under.

## D21 — Prompt rendering · frozen (Implementation Stage 3a)

**The canonical object is a model-independent transcript** — ordered turns with
roles, knowing nothing of tokenizers or checkpoints. Every hash is computed over
it. Turning it into a model's input string is a separate step:

| Template | Materialization | When |
|---|---|---|
| `base_scaffold_v1` | `plain_scaffold` — authored and hashed in config v4 | now |
| `instruct_chat_v1` | `tokenizer_chat_template` — the tokenizer's own | model-compatibility stage |

Authoring a chat template here would hard-code a checkpoint that has not been
selected, so `materialize()` refuses for that template rather than inventing one.

**The conversation is frozen at three turns**: user (scenario, options as A/B,
question) → assistant (the model's own label) → user (one counterargument, then
the *same* question). Turns 1–2 are built once and reused, so all four branches
share them byte-for-byte and the four conditions never appear together.

**Semantic identity never becomes a letter.** Turn 1 depends only on the display
order; turn 3 depends only on the initial choice. Swapping the order changes
which letters are printed and nothing else — tested by asserting that turn 3 is
character-identical across the two orders while the labels differ.

**Nothing is guessed about tokenization.** Materialization ends exactly at the
answer cue, and rendering an assistant turn requires a resolved answer
continuation; with none available it raises rather than choosing between `" A"`
and `"A"`. The question wording is `provisional_until:
model_compatibility_stage`.

**Hashes:** `transcript_hash` over the structured turns, `shared_prefix_hash`
over turns 1–2, `record_hash` over the scenario record, and `prompt_hash` binding
transcript, template, order, condition, config and record together.

**Curator appendix.** Each decision file ends with the canonical transcript for
both option orders and all four branches, labelled explicitly as *not* the exact
model input. It is laid out to make the two invariances checkable by eye: turn 1
varies only with the order, turn 3 only with the initial choice.

Stage 3a renders transcripts from an existing corpus. It does not generate one.

---

## Additional frozen decisions

- **Pilot stratification.** 12 pilot decisions = 4 climate + 4 energy + 4 technology. Checked arithmetically at load.
- **Mechanistic subset.** Stores decision IDs, chosen scenario variant and both option orders. Counterargument direction is **not** knowable at config time; it is resolved at runtime via `counter_target_opt`.
- **Contrasts as coefficient maps.** The five core contrasts are stored as coefficient maps over the core conditions, each validated to sum to zero and to reference only core conditions. This makes the interaction a first-class contrast rather than a special case, and turns a reversed sign into a load-time failure. Names: `style_without_reason` {NS:1, NP:−1}, `style_with_reason` {RS:1, RP:−1}, `content_with_style` {RS:1, NS:−1}, `content_plain` {RP:1, NP:−1}, `interaction` {RS:1, RP:−1, NS:−1, NP:1}. The Holm family is the four simple contrasts.
- **Splits.** 60/20/20 grouped strictly on `decision_id` = 36/12/12 decisions, stratified **by domain only**. Marker family is a property of a `(scenario_id, supported_option)` group — one decision carries several — so it is an allocation constraint plus post-split balance check, **never a stratification key**. Held-out-marker-family, leave-one-marker-out and leave-one-realization-template-out live in a separate `probe_evaluation` section as probe evaluation schemes, not as corpus splits; each keeps the complete four-cell group together.
- **Probe evaluation is decision-disjoint.** Every scheme declares `no_decision_overlap: true` and `grouping_unit: decision_id`. Keeping the four-cell group intact is necessary but **insufficient**: without decision-level disjointness, a sibling group from the same underlying decision could sit in training while this one is evaluated, leaking scenario content. For a held-out family `F`: training draws non-`F` groups from **training** decisions; evaluation draws `F` groups from **disjoint evaluation** decisions; no group belonging to an evaluation decision may enter training. The same rule governs leave-one-marker-out and leave-one-realization-template-out.
- **Dependencies.** Added at the stage that first requires them. Stage 1 installs `pydantic`, `pyyaml` and `pytest` only; `numpy` and `pandas` were removed as unused.
- **Core immutability.** The four core conditions and four core contrasts cannot change even when diagnostic conditions are added. Enforced by `FROZEN_CORE_CONDITIONS` / `FROZEN_CORE_CONTRASTS` guards in `src/reasonstyle/config.py`.
- **Config immutability.** Never edit in place after artefacts exist; create a new version. The filename must encode the version, so a bump cannot overwrite its predecessor.
- **Environment.** Python pinned `>=3.11,<3.12` so the project cannot silently run under ambient 3.13. `uv.lock` and `.python-version` are versioned.
- **Versioned data.** `data/` — including `data/pilot/` and annotations — is committed. Only runtime outputs, activation caches, weights and scratch are ignored.

---

## Owed at later stages

| Item | Impl. Stage |
|---|---|
| Unit test for the movement / `m_after` contrast identity (D9) | 4 |
| ~~Pinned deterministic sentence segmenter (D1/D14)~~ — **done, Impl. Stage 2a** | 2a |
| Synthetic fixture: **1 decision x 2 scenario variants x 8 counterarguments per scenario = 16 texts** | 2b |
| Per-record assertion that all four cells share one realization group (D3b) | 2c |
| Prompt template text and `template_sha256` | 3 |
| Answer continuations, token IDs, `verification_status: verified` (D10) | 5 |
| Model revisions, dtype, device, library versions | 5 |
| Numerical tolerances for patching (dtype-dependent) | 5 |
| `splits.explicit_ids`, `mechanistic.subset.decision_ids` | corpus freeze |
| `status: draft` → `frozen` | corpus freeze |
