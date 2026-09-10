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

## Known limitations

- Machine validation checks form, never substance.
- The segmenter is a heuristic; its known failure modes are flagged, not corrected.
- Pair-level reliability is measured under weaker blinding than item-level.
- LLM generation is not bit-reproducible across model versions; the generation log is a record, not a replay guarantee.
- Style and content are not perfectly separable in language; the residual confound is stated rather than argued away.
