# Design notes

Decisions that affect the experiment, grouped by what they concern. Where a
decision departs from `Research_Plan_v6.md`, that is stated. Implementation
history lives in git, not here.

---

## Conditions and contrasts

The four core cells are frozen: **RS** reason + explicit style, **RP** reason +
plain, **NS** no reason + explicit style, **NP** no reason + plain. They cannot
change; a fifth core cell would require both a code change and a new
configuration, deliberately.

Contrasts are coefficient maps over the core conditions, each summing to zero:

| Name | Coefficients | Reads as |
|---|---|---|
| `style_without_reason` | NS +1, NP −1 | style effect with no reason — **primary** |
| `style_with_reason` | RS +1, RP −1 | style effect when a reason is present |
| `content_with_style` | RS +1, NS −1 | content effect, marker realization held |
| `content_plain` | RP +1, NP −1 | content effect in plain language |
| `interaction` | RS +1, RP −1, NS −1, NP +1 | (RS−RP) − (NS−NP) |

The Holm family is the four simple contrasts. The coefficient form makes the
interaction a first-class contrast and turns a reversed sign into a load-time
failure.

**What controls `RS − NS`.** Both cells are styled with the *same* marker
realization, which therefore cancels. Marker *absence* is not what controls
`RS − NS`; absence is what defines RP and NP, making `RP − NP` the corresponding
content contrast in plain language.

A **diagnostic condition registry** exists but is empty. A later
commitment-matched control can be added there without touching the four core
conditions or contrasts, and can never enter a core contrast.

## Interpretation ceiling

An `NS > NP` result licenses the claim that **explicit inferential framing
affects the model without additional stated propositional support**. It does not
by itself establish irrationality, nor that the model mistakes the form of
reasoning for reasoning: markers such as "therefore" may carry pragmatic
information about speaker commitment. The primary claim stays descriptive, and
the benign alternatives are measured directly — `perceived_speaker_commitment`
and `perceived_unstated_support`.

## Corpus shape

60 underlying decisions — 20 climate, 20 energy, 20 technology — with two
scenario variants each, giving 120 scenarios and 960 counterargument texts. The
pilot is 12 decisions, four per domain: 24 scenarios, 192 texts.

Each scenario has two immutable semantic options, `opt_1` and `opt_2`, and eight
counterarguments: four conditions supporting each option. **A and B are display
labels only**, assigned by the renderer and counterbalanced. Support direction
comes from `supported_option` plus human review, never from letters in the text.

## Matching rules

**Sentence counts must be exactly equal** across the four cells of a group.
Sentence count is itself a surface-form confound, so it is controlled rather than
covaried. Explicit framing must be achieved within the same sentence count, via
connectives, semicolons or sentence-initial markers; a semicolon does not
terminate a sentence. If exact matching repeatedly produces unnatural language,
that triggers a documented configuration change, never a silent relaxation.
Naturalness is human-rated.

**Word counts:** within a group, `max/min` warns above 1.10 and fails above 1.15.
Enforced on **both** the whole counterargument and the manipulated body, because
the shared opening would otherwise dilute a real imbalance — four bodies
differing by 1.40× can pass at 1.15 once a long shared prefix is included.

**The opening** is stored once per scenario and prepended to all eight
counterarguments, so equality across them holds by construction. It must not
mention either option, a condition, evidence, expertise, certainty or authority.

**No-reason cells carry no comparative property.** All four cells of a group end
with the same endorsement clause. An earlier draft ended NS and NP with "is the
safer path", which smuggles safety in as a new comparative advantage — the exact
failure the no-reason condition must avoid.

## Markers

16 markers in four families. The three **primary** families carry the
confirmatory style analysis; **concession** is preregistered as exploratory only,
because concession is a substantively different discourse relation.

| Family | Role | Markers |
|---|---|---|
| `premise_indicator` | confirmatory | because, given that, considering that |
| `conclusion_indicator` | confirmatory | therefore, thus, hence, consequently, for this reason |
| `metadiscursive_inference` | confirmatory | this implies, it follows that, based on this, the key implication is |
| `concession_contrast` | **exploratory** | however, even so, nevertheless, despite this |

"although" is excluded: it typically forces clause restructuring, which
conflicts with exact sentence-count matching. These family names replace the
research plan's original prose names, which overlapped.

**Style is frozen at the realization, not the family.** `marker_family`,
`marker_string` and `marker_realization_id` are stored once per
`(scenario_id, supported_option)` group and inherited by all four cells. RS and
NS instantiate the same realization; RP and NP its paired plain transformation.
Sharing only a family would let `RS − NS` also compare "therefore" against
"because", or one marker placement against another.

**A lexical match shows a marker is present; it does not show the marker
performed its intended discourse function.** Only human validation establishes
that. The analysis must report family-specific effects, test a family × style
interaction, and pool families only when their effects are directionally
compatible.

**Allocation.** Every marker must appear in at least 4 decisions; no marker may
have more than 50% of its uses in one domain or 65% on one supported option;
leave-one-marker and held-out-family evaluations require at least 4 and 8
decisions respectively. These exist so that a nominal held-out-marker analysis
can never run on one or two decisions. Allocation is reported as a corpus
statistic whether or not it binds.

**A configured marker does not prove the resulting NS text is reason-free.** The
pilot must establish, for each realization, that a genuine no-reason instance can
be produced without smuggling in a premise.

## Validation

Machine validation checks **form**: presence, absence, counts, structural
consistency and pattern matches. It cannot establish whether a reason is
relevant, valid or genuinely supporting.

Four severities. `error` blocks an item. `warning` routes to a reviewer without
blocking. `info` records a check that did not apply at this corpus scope, so its
absence is visible rather than silent. And `human_review`, which is **not a
defect**: it is emitted unconditionally for every item, pair, group and scenario,
so a clean machine run can never be mistaken for a validated corpus.

| Code | Emitted for |
|---|---|
| `H_SUPPORT_DIRECTION`, `H_SUBSTANTIVE_SUPPORT`, `H_NATURALNESS`, `H_PRAGMATIC_COMMITMENT` | every cell |
| `H_NO_REASON_INTEGRITY` | every NS/NP cell |
| `H_PROPOSITION_PRESERVATION` | every RS/RP and NS/NP pair |
| `H_REALIZATION_YIELDS_REASON_FREE_NS` | every group |
| `H_SCENARIO_VALIDITY` | every scenario |
| `H_PREMISE_IS_SCENARIO_CONTAINED` *(from corpus construction)* | every RS/RP cell |

Alongside these, a machine provenance check, `E_GENERATOR_INPUT_NOT_FROM_BRIEF`,
fails an item whose logged generation request differs from the request rebuilt
from the frozen prompt template and its topic brief. No source wording is kept in
a brief, so independence from the sources is recorded as `independence:
procedural` rather than as a reviewer judgement.

`ValidationReport.ok` means "no machine errors" and nothing more. Approval
additionally requires every human-review code to be discharged by a recorded
annotation.

**Forbidden phrases** are 25 word-boundary regexes, matched case-insensitively:
18 hard failures for unambiguous authority, evidence, consensus and pressure
language; 7 warnings where a regex cannot tell scenario-grounded from external
evidence — bare "authorities", "proven", "empirical", and all certainty adverbs.
Every pattern carries positive and negative fixtures verified at config load, so
a pattern cannot be weakened to resolve a collision nor broadened until it
swallows ordinary policy language. A separate check asserts no permitted marker
is matched by any forbidden or leakage pattern.

**Display-label leakage** rejects "option A", "choice B", "select A" and bare
`A.` / `B)`, but not a bare letter, since "A" is also an English article.

**Corpus scope** gates corpus-level checks (`fixture | pilot | full`). Marker
allocation minima cannot be satisfied by a one-decision fixture, so outside
`full` they are reported as skipped, never passed silently.

## Sentence segmentation

`pysbd`, pinned at the resolved version and recorded on every measurement; a
mismatch between config and installed version refuses to run. It is rule-based
and deterministic, but still a heuristic: it handles semicolons, decimals and
"e.g." correctly and mis-segments "U.S." and "approx.". Those constructions are
therefore **flagged**, not trusted, and generated text is restricted to avoid
them. **The machine count is authoritative**; a reviewer may flag a suspected
error, but a correction requires a recorded annotation.

A regex was rejected because D1 makes sentence equality a hard rejection
criterion, and a regex's errors would correlate with the style factor — styled
cells carry semicolons and connectives at different rates — biasing the very
confound the rule exists to control.

## Prompting

The canonical object is a **model-independent transcript**: ordered turns with
roles, knowing nothing of tokenizers or checkpoints. Every hash is computed over
it.

The conversation is frozen at three turns — user (scenario, options as A/B, the
fixed question) → assistant (the model's own label) → user (one counterargument,
then the same question). Turns 1–2 are built once and reused, so all four
branches share them byte-for-byte and the four conditions never appear together
in one conversation. Turn 1 depends only on the display order; turn 3 only on the
initial choice.

The base scaffold is authored and hashed in the configuration. An instruct
model's chat template belongs to *its own tokenizer* and is applied by the model
adapter — authoring one here would hard-code a checkpoint that has not been
selected. Nothing guesses at tokenization: rendering ends exactly at the answer
cue, and the continuation after it (`" A"` versus `"A"`) is settled only by
inspecting a tokenizer. The question wording is provisional until then.

## Near ties

A near tie is an initial conditional probability of the preferred option below
0.60, i.e. `|m_before| < log(0.60/0.40) = 0.4054651081`. Fixed a priori on an
interpretable scale rather than chosen from the pilot distribution, to remove
researcher discretion over an exclusion rule. **Robustness only — the full
sample stays primary.**

`m_before ≤ 0` by construction, and is shared by the four branches of one
model × scenario × option-order run. The four within-group contrasts are
therefore numerically identical computed on movement or on `m_after`; movement is
stored because it is the interpretable quantity.

## Splits and probe evaluation

The corpus split is grouped strictly on `decision_id` (36/12/12 decisions) and
stratified **by domain only**. Marker family is a property of a
`(scenario_id, supported_option)` group — one decision carries several — so it is
an allocation constraint and post-split balance check, never a stratification
key.

Held-out-marker-family, leave-one-marker-out and leave-one-realization-template-out
are **probe evaluation schemes, not corpus splits**. Every scheme is
decision-disjoint: keeping the four-cell group intact is necessary but
insufficient, because a sibling group from the same decision would otherwise leak
scenario content into training.

## Annotation and review

Judgements are stored at the level at which they are made — duplicating a pair-
or scenario-level judgement onto every item would inflate apparent sample size
and let one judgement disagree with itself.

| Level | Ratings |
|---|---|
| **item** | substantive_support, support_direction_confirmed, perceived_reasoning_style, no_reason_integrity, perceived_speaker_commitment, perceived_naturalness, perceived_unstated_support, confidence, pressure, politeness, authority, credibility |
| **pair** | proposition_preservation, over `RS/RP` and `NS/NP` |
| **scenario** | option_feasibility, option_non_dominance, normative_underdetermination |

`perceived_speaker_commitment`, `perceived_naturalness` and
`perceived_unstated_support` are additions to the research plan's rubric; the
plan's single "scenario validity" item is split into its three constituent
judgements. Agreement: Cohen's κ for binary, Krippendorff's α for ordinal, with
no expected value set in advance.

**Every item is reviewed by one curator.** A stratified subsample is
independently annotated by two annotators — same items, as κ requires, in
independently seeded orders.

### Blinding is not uniform

| Level | Blinding | What the annotator can infer |
|---|---|---|
| **item** | blinded | one text alone; condition, marker metadata, counts, sibling cells and the intended option all hidden |
| **pair** | **partially blinded** | both texts shown; labels hidden and sides randomised, but the minimal-pair structure is visible |
| **scenario** | blinded | scenario and options only |

A proposition-preservation judgement requires both texts, and the two texts of a
pair differ only by a connective. **The methodology must describe item-level
review as blinded and pair-level review as partially blinded**, and must not
report the pair-level reliability figure as though it were obtained under
item-level blinding.

Blinded option labels are **P and Q**, never A and B, so blinded review cannot be
confused with the experiment's display labels. A blinded annotator can answer
"unclear". Questions are phrased so as not to reveal the design: no-reason
integrity is asked as "does this supply a premise?", answerable without knowing
the condition. The unblinding key lives in its own directory.

**Sibling separation.** Two cells of one group differ only by a connective, so an
annotator seeing them close together can infer the design. A minimum separation
of 20 items is enforced in the item packet. When infeasible, the exporter reports
it — in the packet, the manifest and on stdout — and it requires explicit
approval to change. It is never silently relaxed. This cannot help in the pair
packet, where both cells are shown together by design.

## Review export

A deterministic, **read-only** Markdown view of the canonical JSONL. Judgements
go to `data/.../annotations/`; nothing written into the Markdown is read back.
No generated file carries a wall-clock timestamp — that would make the export
differ from one day to the next — and `--check` regenerates into a temporary
directory and fails on any drift.

The hash printed in every file is the hash of the corpus **as stored**. Derived
measurements must never change it, or the hash shown in the review would not
match the JSONL that was reviewed.

Marker highlighting is display-only: the canonical text always appears verbatim
in a fenced block, and stripping the highlighting returns it character for
character.

## Configuration and provenance

One editable `configs/experiment.yaml` during development; git records its
history. A configuration that is actually used to produce research results is
copied to `configs/frozen/<name>.yaml`, and from then on it never changes and its
hash is pinned by test.

Any command that generates artefacts takes a **required `--config` path** and
prints the resolved version and hash. Artefacts embed the config hash, so the
version must be chosen deliberately rather than inherited from whichever file is
newest.

**Source references are an open, extensible structure, not an enum.** Reference
datasets supply topics, competing values and argument structures; their text does
not become experimental text. A scenario may cite several sources or none — a
constructed scenario carries an empty list. **LLM generation metadata is kept
separate**: it records how a draft was produced, and is neither a source
reference nor a licence claim.

**Model selection is not frozen.** No model identity appears in any schema,
validator or configured behaviour; `repo_id` and `revision` stay null until a
tokenizer/template compatibility test settles them.

## Corpus construction

**Reference datasets are reference material only.** They are used to identify
broad policy topics, competing values and general argument structures. Their
scenarios, claims, arguments and explanations are never copied, lightly
paraphrased or systematically transformed into the corpus, and no source item
is treated as the template for an experimental item. Every scenario and
counterargument is newly drafted under our own design rules; where anything a
source suggests conflicts with the scenario, matching, marker,
premise-containment or validation rules, our rules take priority. Provenance is
recorded at the topic-idea level.

**Independence is enforced by the construction process, not by comparing texts
against whole datasets** — a reviewer cannot realistically check 960 texts
against every source:

1. Reference datasets supply only broad topic ideas, value trade-offs and
   general structural inspiration.
2. A short, original topic-bank brief is written for each decision. A source
   is recorded only as a registry key, a locator (the article, table or
   category used) and an `inspiration_summary` in the curator's own words.
3. No source wording, argument, explanation or complete source item enters the
   brief or the generation request.
4. The model drafts from the curated brief and our experimental constraints.
5. Human review compares generated text with its topic brief.

A provenance check confirms from the stored request that the generator received
only the curated brief and our constraints: the logged request must match the
request rebuilt from the frozen prompt template and the item's brief. Because no
source wording is retained or supplied, independence is a **procedural
guarantee**, not an annotation claim, and no paraphrase comparison is needed.

**Assisted briefs.** The initial pilot briefs are prepared with assistance, and
each record says so (`brief_prepared_with_assistance`). Vidhi Bhutani is the
human curator who edits and approves every brief; a brief becomes `curated` only
with the curator's recorded judgements, name and date.

Intended roles: POLIANNA for climate and energy policy topics and policy
structures; the fixed April 2025 JRC snapshot of the GenAI4PA data for
technology-governance topics (the living GenAI4PA portal is a discovery source
only); IBM-ArgQ-Rank-30k for general pro/con argument structures only. ValuePrism
is optional: a source for broad value categories only if its accepted
Medium-Risk agreement can be recovered and recorded, never for its situations,
explanations or other text. Its absence does not delay corpus construction —
competing values can also be curated from the policy sources and the research
literature. The five other candidates are excluded for their recorded reasons.

Each
candidate in `data/sources/registry.yaml` is `unverified`, `citable` (access and
licence confirmed, seed-only use permitted) or `excluded` (checked, but
unavailable, unsuitable, prohibited or too unclear). An exclusion records who
checked, when and why, and nothing that could not be found is invented.
`checked_by` names the person who takes responsibility for the verification.
**The gate is at citation: every source a topic actually cites must be
citable.** A candidate that nothing cites may stay unverified without blocking
work. Exact registry contents become immutable only at the pilot freeze.
Nothing is filled in from memory.

**Provenance is recorded once, by registry key.** The topic bank records the
registry keys (such as `ibm_argq_rank_30k`) each decision draws on. Scenarios
reference their `decision_id` rather than repeating source details; any
scenario-level reference uses the same keys.

**Identity.** The topic bank has one record per underlying decision, keyed by
`decision_id`; there is no separate topic id. Every scenario references its
`decision_id`. Topics avoid explicit political-party, politician and identity
framing, though the policy decisions themselves may be politically contested.

**Variants.** The two variants of a decision must present the same decision,
options and value trade-off, differing in wording or context. The options and
competing goals are shared at the decision level; each variant has its own
context and its own **scenario facts**. These are constructed statements the
scenario will state, never factual claims copied from a source. **In the pilot,
each option gets exactly one fact per variant, of at most 25 words**, so every
option has the same amount of support and scenario lengths stay close; this can
be reconsidered after the pilot. Each fact must instantiate its option's
declared competing goal — if the trade-off is reliability against emissions,
funding or timing must not silently stand in for either. An RS or RP
counterargument may use only facts from its own variant. The two variants must
differ in more than their context sentence: at least one scenario fact changes,
while the decision, options and value trade-off stay the same. A machine can
check that options and domain agree, that contexts and facts differ and that
fact counts are equal; only a person can confirm that the facts instantiate the
goals and that the trade-off is preserved.

**Topic curation.** Each brief carries nine curator judgements: normatively
underdetermined; can be made self-contained (whether the final scenario *is*
self-contained can only be judged once it exists); no party, politician or
identity framing; both options feasible and non-dominated as options; **each
fact is a genuine consideration in favour of its assigned option**; **both
competing goals are represented in every variant**; **neither option clearly
dominates once all the facts are considered**; the variants differ
substantively; no source wording in the brief. Fact balance is deliberately
split into those three separate questions rather than judged as one. Domain balance counts only `curated`
topics, so extra candidates can be proposed and rejected freely. Drafting may
begin only with **exactly four curated decisions per domain** and no machine
errors.

**Whether pilot decisions count toward the final 60.** The 12 pilot decisions
may be included in the main corpus only if (a) the experimental definitions,
prompts and validation rules do not materially change after the pilot; (b) they
pass the same final validation and review criteria as every other item; and (c)
their inclusion is decided without looking at any model outcome results. If the
pilot leads to substantive design changes, the affected decisions are
regenerated under the final rules or excluded.

**Drafting.** Marker allocation is planned before drafting. Each scenario takes
three drafting calls: the scenario itself, then one four-condition group per
supported option — the four conditions of a group drafted together, the two
directions separately. The generator model, provider and settings are proposed
and approved before any drafting, preferably from a different model family from
those being evaluated. The exact submitted request and the raw response are
stored. This is a record, not a reproducibility guarantee.

**Repair.** Items failing validation are redrafted with a separate, fixed,
hashed repair prompt, up to a maximum number of attempts set in advance. An item
rewritten or written by hand after failed attempts records that in its
provenance, with the editor and reason, and passes the same validation and human
review as any other.

**Premise containment.** Every factual premise in an RS or RP cell must be
supported by information in its own scenario, never introduced as external
evidence. This is an unconditional human check; lexical matching is a warning
only.

**Independent annotation sample.** Stratified across domain × condition ×
marker family. Because RP and NP carry no marker themselves, **the family used
for stratification is the one assigned to the whole four-condition group**,
never the cell-level `marker_family`, which is null for plain cells. The sample
size is the larger of the configured fraction and one item per non-empty
stratum: in the pilot that is 48 items rather than 38, and 24 pairs rather than
19.

**Manipulation-check rules are fixed before the pilot annotations are
examined** — the minimum reason/no-reason separation, the minimum styled/plain
separation, acceptable proposition-preservation rates, equivalence bounds for
pressure, politeness, confidence, authority and credibility, and what happens if
inter-annotator agreement is low. Thresholds are never chosen after seeing which
values let the corpus pass.

**Pilot results are reported by condition, domain, marker family and marker
realization**, not only as one aggregate pass or fail, so that a result carried
by a few particular markers is visible.

**Freezing.** A frozen configuration is recorded in an external manifest with
its SHA-256, and the tests verify the file against that hash — including a test
that edits a temporary copy and confirms verification fails. Location and
filename checks alone would not catch an edit to a frozen file's contents.

## Known limitations

- Machine validation checks form, never substance.
- The segmenter is a heuristic; its known failure modes are flagged, not corrected.
- Pair-level reliability is measured under weaker blinding than item-level.
- LLM generation is not bit-reproducible across model versions; the generation log is a record, not a replay guarantee.
- Style and content are not perfectly separable in language; the residual confound is stated rather than argued away.
