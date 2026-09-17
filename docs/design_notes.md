# Design notes

Decisions that affect the experiment, grouped by what they concern. Where a
decision departs from `Research_Plan_v6.md`, that is stated. Implementation
history lives in git, not here.

**Decision index.** Code, tests and `configs/experiment.yaml` cite a few
decisions by number. Numbers not listed were retired or merged into the
sections below; there is no separate decision log.

| ID | Decision | Section |
|---|---|---|
| D1 | Sentence counts exactly equal across the four cells; no tolerance | Matching rules |
| D2 | Word-count ratio (warn 1.10, fail 1.15) enforced on both full text and body | Matching rules |
| D3 | Marker fields are group-level and inherited by all four cells; D3b: RS and NS share one realization, RP and NP its paired plain transformation | Markers |
| D5 | Display-label leakage rejects "option A" and similar, never a bare letter | Validation |
| D6 | One corpus-wide opening, stored once per scenario and shared by all eight texts | Matching rules |
| D8 | Deviations from the plan: an empty diagnostic-condition registry, and three added annotation ratings | Conditions and contrasts; Annotation and review |
| D13 | Judgements stored at the level they are made: item, pair or scenario | Annotation and review |
| D15 | Source references kept separate from LLM generation metadata | Configuration and provenance |

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

**Both sizes are fixed; only the main corpus is still unbuilt.** The pilot is
**12 underlying decisions**, four per domain, with two scenario variants each:
24 scenarios, 48 four-condition groups, 192 counterargument texts. That is what
is being generated and reviewed now. The main corpus is 60 decisions, and
constructing the rest of it comes after the pilot is reviewed.

**192 texts are not 192 independent items.** One decision yields 2 variants × 2
supported options × 4 conditions = 16 texts, all built on the same underlying
policy decision. The independence unit is `decision_id`, which is why the
analysis clusters on it and every split is grouped by it. A count of texts,
groups or scenarios must never be reported as a number of independent policy
decisions.

**The main corpus is 60 policy decisions** — 20 climate, 20 energy, 20
technology — giving 120 scenarios, 240 four-condition groups and 960
counterargument texts (Research_Plan_v6 §5). That target is settled, and
`configs/experiment.yaml` carries it in `corpus.decisions_full`,
`corpus.texts_full`, `domains.decisions_per_domain_full`, the `splits.decisions`
counts (36/12/12 decisions) and `mechanistic.subset.n_decisions` (40). Those are
confirmed values, not placeholders.

**The pilot's 12 decisions are candidate members of the 60**, and count toward
it when they satisfy the final frozen specification and review criteria. Where they do, they reduce what is
left to build: the normal remainder is **48 additional decisions, not 60**. A
pilot decision that cannot meet the final criteria is regenerated under the
final procedure or replaced, and the total stays at 60.

**Changing 60 is a design amendment**, documented before any main evaluated-model
outcome is examined — never a quiet adjustment afterwards.

### The order of work

1. **Now (engineering).** Generate and human-review the 12-decision pilot.
2. **Then (design gate).** Approve the coverage plan and the candidate decision
   list for the remaining decisions. The number is settled at 60.
3. **Then (production).** Generate, validate, human-review and freeze the
   expanded main corpus.
4. **Only then.** Run the main behavioural evaluation at scale.
5. **After that.** Mechanistic analysis runs over **40 of the 60 decisions**
   (Research_Plan_v6 §8). The count is fixed; the rule that selects those 40 is
   documented and preregistered before the mechanistic analysis begins.

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

**One opening for the whole corpus: "I disagree with that choice."** It is
configured once and every scenario carries it. It also must not suggest
deliberation: an earlier draft, "I have read the scenario and I would weigh it
differently", describes reading and weighing, which is reasoning content —
present, under that draft, even in the plain, no-reason cells that exist to have
none. The config check rejects any opening containing a deliberation word.

*What is and is not identical across the corpus.* Only the opening. The whole
prompt is not: each scenario has its own text and its own option lines, and the
display order of A and B is counterbalanced. What is byte-identical is narrower
and is what the design needs: **within one scenario and one initial answer, the
transcript up to the counterargument — turns 1 and 2 — is shared by all four
branches**, so the four conditions differ only in the counterargument itself.

*A consequence of the shorter opening.* A five-word prefix dilutes far less
than an eleven-word one, so the full-text and body word ratios now track each
other closely. The body measurement remains the stricter of the two and is the
one that catches an imbalance; the full-text measurement is kept because it is
what a reader actually sees.

**"No reason" means no task-relevant reason.** NS and NP must give no
*substantive policy* support for the option they endorse. A short
self-referential clause is allowed, and is usually needed to host the marker at
all — "because that is my view" carries no support for the policy. What such a
clause must never contain is: a scenario fact; a consequence, effect or outcome
of either option; a value, goal, priority or trade-off; evidence, data,
authority, expertise or experience; or any new factual claim. A reader of NS or
NP must not be able to reconstruct a consideration in favour of the endorsed
option. Read as a literal ban on every subordinate clause, the rule would
forbid the very sentence that carries the marker, and the style manipulation
could not be realized in the no-reason cells at all. The curator's
`no_reason_integrity` judgement is stated in these terms in the review export.

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

**Pair content drift.** Under the minimal-edit rule the plain cell of a pair is
its styled cell with the marker removed, so once the marker and a short,
configured list of function words are set aside, the two must contain the same
content words, with multiplicity. A difference is `E_PAIR_CONTENT_DRIFT`, and
the differing tokens go into the finding so a repair can act on them. The list
is a **small connector/filler allowance, not a general function-word
allowance**: exactly the three tokens the valid fixture needs (`and`, `here`,
`overall`). Modals, tense and aspect auxiliaries, pronouns and referents,
prepositions, the logical `or`, and negation are all excluded, because each of
them can change what is claimed; negations and permitted markers are refused by
a load-time check as well. The screen
is lexical: it shows that the same words are present, never that the two express
the same proposition, so `H_PROPOSITION_PRESERVATION` remains unconditional. It
was added after the first live smoke call produced a pair whose cells differed
by a whole clause (2026-09-16).

**The opening is never part of a body.** It is stored once per scenario and
prepended by the renderer, so a body repeating it would show it twice and
inflate every word count. `E_OPENING_REPEATED_IN_BODY` rejects a body that
contains it, compared with case and whitespace normalised. Only that exact,
normalised repetition is machine-detected: a **paraphrase of the opening
remains a human judgement**, and no approximate or embedding-based matching is
used for it.

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

**Overlap screen (mandatory, warning only).** As a safeguard against wording
slipping in while briefs are written, every generation-bound brief field is
compared with the downloaded source text, and any run of six or more
consecutive shared words raises `W_TOPIC_SOURCE_OVERLAP`. The checker and the
review export will not run without deciding on the screen: they need the
downloaded files, or an explicit `none` (for synthetic fixtures only), in
which case the output states that the screen was not run. The warning quotes
at most twelve words of the brief's own text; no source excerpt is stored in
the topic bank or the review export.

**What the screen does and does not establish.** A six-word match detects
*possible verbatim reuse*; it cannot prove semantic independence. Close
paraphrase, translated wording and reused argument structure all pass it, and
a match can equally be an ordinary collocation. Independence rests primarily on
the construction procedure above: the later generator receives only the
original curator-written brief and our constraints, never a source passage, so
there is no source text for it to reuse. The screen is a backstop against
accidental copying while briefs are written, and the curator's
`no_source_wording_in_brief` judgement remains the authority. This is how the
thesis should describe it, rather than as evidence of originality.

**Downloads.** Raw source files are fetched only from pinned URLs listed in
`data/sources/downloads.yaml`, which records the access date, size, the
provider's checksum and our own SHA-256; a mismatch stops the process before
anything is recorded. Raw files live in the gitignored `data/sources/raw/`.
Every archive member path is checked before anything is extracted; pickle
files are never extracted or loaded. Only POLIANNA's JSON files and the two
pinned JRC CSV tables are read. Reading a provision to understand its general
subject is acceptable; its wording, organisations, locations and statistics
never enter a brief. `data/sources/inventory.md` summarises the material in
our own words.

**Assisted briefs.** The initial pilot briefs are prepared with assistance, and
each record says so (`brief_prepared_with_assistance`). Vidhi Bhutani is the
human curator who edits and approves every brief; a brief becomes `curated` only
with the curator's recorded judgements, name and date.

Intended roles: POLIANNA for climate and energy policy topics and policy
structures; the fixed April 2025 JRC snapshot of the GenAI4PA data for
technology-governance topics (the living GenAI4PA portal is a discovery source
only); IBM-ArgQ-Rank-30k for general pro/con argument structures only. Each of
these is **available, not mandatory**: no brief is required to cite IBM-ArgQ,
and the curated pilot bank cites only POLIANNA and GenAI4PA. Research_Plan_v6
§10 mentions "seeding argument content from ArgQ/CMV". That is superseded: CMV
is excluded, and ArgQ remains optional. ValuePrism
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
`decision_id`.

**The identity-framing rule, precisely.** Briefs, scenarios and
counterarguments must contain no political-party appeal, no reference to a
politician, no stereotype, no personalised identity appeal and no argument that
asks for agreement because of a group identity. The rule does **not** prohibit
neutrally describing who bears a policy's costs or benefits — residents,
tenants, households without private parking, job-seekers — which is frequently
the only way to state a genuine distributive trade-off. The distinction is
between *naming who is affected* (allowed, and often necessary) and *appealing
to who someone is* (prohibited). The configured `framing_warning_terms` list
party and partisan vocabulary only; a match is a warning for the curator, and
the policy decisions themselves may be politically contested.

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

**Both facts precede the initial answer.** The scenario states both of a
variant's supporting facts before the model gives its initial answer, so that
the answer is made with the whole trade-off in view. A later RS or RP
counterargument may build a substantive justification on one of those facts,
but it introduces no new factual claim; this is the premise-containment rule
below, seen from the scenario's side.

**Specificity balance.** Facts are preferably qualitative on both sides. If one
option's fact includes a number, percentage, deadline or other precise
quantity, the other option's fact in that variant should be comparably
specific, so that precision does not lend one side extra persuasive weight.
A pattern screen (configured in `topics.specificity_patterns`) flags a variant
where only one option's facts look specific (`W_TOPIC_SPECIFICITY_MISMATCH`)
for human review. It is deliberately broad — number words such as "one" also
match, so "no one" is a false positive — and the curator judges comparability.

The same balance applies beyond numbers, and only a person can check it: a
definite consequence for one option must not be paired with a vague or
speculative one for the other. A present-tense fact on one side and a "could
eventually" on the other tilts the trade-off even when both facts are
qualitative.

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

**Whether pilot decisions count toward the main corpus.** They are candidate
members of the 60 and count when they satisfy the final criteria: (a) the
experimental definitions, prompts and validation rules do not materially change
after the pilot; (b) they pass the same final validation and review criteria,
under the frozen specification, as every other item; and (c) their inclusion is
decided without looking at any model outcome results. If the pilot leads to substantive
design changes, the affected decisions are regenerated under the final rules or
excluded, keeping the main corpus at 60 decisions (*Corpus shape*).

**Drafting.** Marker allocation is planned before drafting. Each scenario takes
three drafting calls: the scenario itself, then one four-condition group per
supported option — the four conditions of a group drafted together, the two
directions separately. The generator model, provider and settings are proposed
and approved before any drafting. The exact submitted request and the raw
response are stored. This is a record, not a reproducibility guarantee.

**Generator family.** The generator should preferably differ from the
**primary** evaluated family. Llama-3.1 is the primary family, and a Llama
generator is refused at config load. `Qwen/Qwen3-14B` is therefore acceptable.
Evaluating a Qwen model is optional (Research_Plan_v6 §8). If one is later used
as a replication model, it shares a family with the corpus generator, and that
is disclosed as a limitation; it is not prohibited. The same applies to any
other optional family. Only the primary family is guarded.

**Marker allocation (pilot).** The unit is the group, because the marker fields
are group-level. Each decision's **two variants take different families**, so no
family is confounded with a decision; within a slot both directions share the
family and string and take the two different realizations, alternating so that
realization is balanced across directions. Three confirmatory families, 16
groups each; `concession_contrast` is deferred to the full corpus. **Two strings
per family**: a family covers eight decision-slots and `min_decisions_per_marker`
is four, so a third string could not clear the floor — the pilot tests families
and realizations, not every string. With three domains and four decisions per
string, one domain necessarily holds two of them, so 50% is the *best
achievable* per-string domain share, not slack in the design. The allocation is
built deterministically from a seed, checked against every rule independently of
how it was constructed, and written with its own content hash before drafting
begins.

**Prompts are files, fixed by hash.** The three drafting templates live in
`prompts/*.txt` so they stay readable and diffable; the config records each
one's SHA-256, its placeholders and its response schema, and the loader fails if
a file and its hash disagree. Placeholders are `${name}`, not braces, because the
response is JSON.

**Requests are emitted to files, and responses imported from files.** Nothing
is sent as a side effect of building a request: a request file holds the whole
rendered prompt and can be read before any model sees it. Responses are
validated against the template's closed schema and rejected — never coerced,
trimmed or patched — if they do not fit.

**The generator is a local open-weights model on a lab GPU server**, served by
vLLM behind an OpenAI-compatible endpoint bound to `127.0.0.1`. No external
service, no paid call, and no credential anywhere: there is nothing to
authenticate to. The proposed model is `Qwen/Qwen3-14B` in non-thinking mode
(temperature 0.3, top_p 0.8, 700 new tokens, seed recorded), and the
configuration refuses a generator whose repository id names the primary
evaluated family (Llama).
Everything runs under `/data/$USER` on one explicitly chosen GPU.

**Running needs two authorisations, neither of them in the config**: the
explicit `--send` argument and `REASONSTYLE_ALLOW_LOCAL_GENERATION=1` in the
environment. Permission to run a model does not belong in a scientific
configuration that is read, copied and shared; the loader refuses a config that
grows such a key. `HF_HUB_OFFLINE=1` is required as well, so that a missing
model is an error to report rather than an unattended download.

**Compute.** Chomusuke02's A6000 is used for corpus generation. It may also
support the first Llama-3.1-8B compatibility and behavioural tests. The larger
causal sweeps may later need Wisteria or an A100-class GPU (Research_Plan_v6
§11), depending on measured memory and runtime. This is **not yet resolved**:
nothing has been measured.

**No hidden sampling defaults.** vLLM's default `--generation-config auto`
loads the model's own `generation_config.json` and uses it for any sampling
field a request leaves out. For Qwen3 that includes `top_k` and a different
temperature and top_p. The launcher therefore always passes
`--generation-config vllm` and records `generation_config` in the server
runtime record. A live run refuses a record that lacks it. In addition, every
sampling field that can change the output is sent explicitly, and the loader
requires the neutral values: `top_k -1` (disabled), `min_p 0.0`,
`repetition_penalty 1.0`, `presence_penalty 0.0`, `frequency_penalty 0.0`, on
top of temperature, top_p, max_tokens, n and seed. `top_k -1` is accepted both
by vLLM 0.8.5, which rejects 0, and by later versions. Two things are not set
per request, and only the recorded vLLM version fixes them. (1) Stop tokens:
without the model's generation config, generation stops at the tokenizer's EOS
token. The smoke test should confirm `finish_reason` is `stop`, not `length`.
(2) The structured-output backend chosen for `response_format`.

**A seed is recorded, not relied on.** Local inference is not bit-for-bit
reproducible across GPUs, drivers or library versions. What stands as the record
is the saved raw response, the hashes of the exact prompt and response, and the
environment block: resolved commit sha of the weights, decoding settings, seed,
GPU, and library versions. A model whose commit cannot be resolved from the
local cache is refused, because an unpinned generator cannot be reported.

**The GPU and the libraries are recorded by the machine that loads the
weights.** The launcher writes a server runtime record — host, GPU index and
name, dtype, revision, snapshot path, context length, seed, and the server's
vLLM, transformers and torch versions — and the client's log points at it. A
client shell may sit on another host entirely, so a GPU name read there would be
a fiction. Three places name the revision (the config, the cache, the server
record) and a run is refused unless all three agree. The cache check also
verifies the weights themselves: with a safetensors index, every shard it names
must be present and non-empty — a config and tokenizer alone are exactly what an
interrupted download leaves behind.

**Redrafting what the curator rejects.** A scenario marked `redraft` passed
every machine check: what is wrong with it is what the reviewer wrote down, so
that reason, and the frozen brief, are what the redraft request carries. The
reviewer's own preferred wording is not sent — the model stays the drafter, or
the text would no longer be generated material. One call per rejected scenario,
no automatic second attempt, and a redraft that returns the original unchanged
is recorded as such and supersedes nothing.

The redraft template is versioned and hashed **outside**
`configs/experiment.yaml`, and deliberately so: an approval binds the
configuration hash, so adding a template to the config would make every existing
approval stale and invalidate a live run. The template's SHA-256 travels with
each request, log line and result instead, which is where provenance belongs.

**Supersession is explicit.** A redraft names the call it replaces
(`supersedes_call_id`), and the current text of a scenario is resolved by
following that link from the original — never by taking whichever call is latest
in the log. An unaccepted redraft therefore leaves the original standing, and
the approvals of scenarios nobody redrafted stay valid.

**A human scenario correction is separate evidence.** If the bounded redraft
still changes or adds a supplied fact, repeats filler, or otherwise fails human
review, the generated result is not edited. A correction ledger records the
exact model call and original text, both text hashes, the corrected text, editor,
reason, date and approval state. The correction must match the current model
text and call, must change the text, and is re-run through the ordinary scenario
validator; an error refuses it. The scenario keeps the model call id as its
source while the curator approval binds the corrected text hash. This preserves
which words came from Qwen and which were edited by a person.

**Two stages, and a person between them.** Scenario drafting and group drafting
are separate commands, and no command does both: the curator's approval sits
between them, and a tool that crossed that boundary automatically would make the
gate decorative. Both stages are resumable — a completed call is recovered from
disk, never re-sent — so the sequence is: draft every scenario, read them,
record approvals, draft the groups. A scenario marked `redraft` blocks the set
and is a separate decision; there is deliberately no automatic redraft budget.

**The curator-approval gate.** A group is drafted *from* a scenario, so the
scenario is approved first, and the approval binds four things: the exact text
(by SHA-256), the accepted call it came from, the configuration hash and the
topic-bank hash. A change to any one makes the approval **stale**, so there is
no way to approve one text and generate from another. Approval is all-or-nothing
across the pilot's scenarios: excluding one would unbalance a marker allocation
built over every decision at once. **Approval adds a requirement and never
removes one** — a scenario with machine errors is reported as blocked however
the approvals file reads, and is fixed or redrafted, never approved past.

**Assembly and counterargument correction.** A record is assembled only from an approved
scenario and machine-valid groups. A human correction is never an edit to what
the model returned: it is a separate record naming the original call id and its
original text, the corrected text, the editor, the reason, the date, its
approval state and the validator result of the corrected material. The
correction is bound to the text it replaces, so it cannot be carried silently on
to a different draft; corrected text is re-validated by the same validator, and
material that still fails is still refused. The generation log and the raw
responses keep what the model produced, so model output and human edit stay
distinguishable afterwards. An assembled record remains `validation.status:
draft` until the item, pair and scenario judgements are recorded.

**Repair.** Items failing validation are redrafted with a separate repair
template, versioned and hash-pinned like the others. The *template* is fixed;
the *rendered request* varies by scenario, bodies, findings, measurements,
attempt number and the history of earlier attempts, and each rendered prompt is
individually content-hashed and recorded. The budget is **one draft plus at most two repairs, three
calls per group**. The findings passed to the repair are the validator's own
codes and messages, and the measurements behind them, so no advice is composed
by hand for an item.
A group that still fails is marked **`needs_manual_review` and stops there**: it
is never silently hand-corrected or accepted. Any later human correction is a
separate, recorded and separately approved act, with the editor and reason in
the item's provenance, and it passes the same validation and human review as
any other item.

**Logging.** Every call writes its verbatim request and response to
`data/pilot/raw/<call_id>.json` and one line to `generation_log.jsonl`: what was
asked, of which model, under which config, brief and allocation hashes, what
came back, and whether it was accepted, rejected, refused or failed. A retry is
a new attempt with its own line; nothing is silently retried.

**Premise containment.** Every factual premise in an RS or RP cell must be
supported by information in its own scenario, never introduced as external
evidence. The scenario states **both** of a variant's facts before the initial
answer, so the answer is given with the whole trade-off in view and a later
counterargument can build on a fact without adding one. This is an
unconditional human check; the lexical screen is a warning only: it reports the
share of a reason cell's content words that appear in its own scenario, ignoring
the frame vocabulary every cell shares.

**Scenario length.** 90–130 words in the pilot, with the actual count recorded
for every scenario and a warning outside the band. It is a drafting instruction
rather than a rule, and the pilot is meant to show whether the band forced
filler before it is applied to the full corpus.

**Independent annotation sample.** Stratified across domain × condition ×
marker family — 3 × 4 × 3 = 36 crossed strata. **The supported option is
balanced marginally, not crossed**: adding it would make 72 strata, which a
38-item sample cannot cover, so the sample would silently stop being stratified
in the way the thesis describes it. Instead, among all samples with the same
size and the same largest-remainder stratum quotas, the sampler draws one with
the smallest reachable opt_1/opt_2 difference. It finds that minimum exactly;
seeded randomness only picks among samples meeting it. A perfect split (0 when
even, 1 when odd) is therefore always reached when one exists. When it is not,
the best achievable difference, and whether units or quotas limit it, go into
the manifest, the unblinding-key README and stdout. Never into an annotator's
packet, where option counts would weaken the blinding. Balance may only choose
among strata *tied* at the boundary remainder; a larger remainder always wins.
A config check refuses any stratification finer than the sample can reach.
Because RP and NP carry no marker themselves, **the family used for
stratification is the one assigned to the whole four-condition group**, never
the cell-level `marker_family`, which is null for plain cells.

**Sample size** is `max(1, round(N × fraction))` with the configured fraction
0.20, allocated to strata by largest remainder. In the pilot that is **38
counterargument items** (20% of 192), covering all 36 item strata, and **19
proposition-preservation pairs** (20% of 96), covering all 18 pair strata
(domain × pair × marker family). An earlier statement of 48 items and 24 pairs
came from an older rule, one item per non-empty stratum at minimum. It is
withdrawn (decided 2026-09-15).

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
- The six-word overlap screen detects possible verbatim reuse only; independence is a property of the construction procedure, not of the screen.
- The corpus generator is a Qwen model. If an optional Qwen model is later evaluated, corpus and evaluated model share a family.
