---
title: "Reason or the Language of Reason?"
subtitle: "What actually moves an LLM's position when a user pushes back"
author:
  - "Vidhi Bhutani — B.E. Computer Science + M.Sc. Economics, BITS Pilani Goa"
  - "Supervisor: Prof. Yusuke Miyao, University of Tokyo"
  - "Version 6 — implementation-ready revision. Changes: verified the nearest-neighbour papers; corrected the factorial contrasts; specified bidirectional counterarguments, controls, analysis, patching sites, and probe limits."
date: "September 2026"
---

# 1. The question

You give a model a policy question with two defensible options. It picks one. You push back.

Sometimes the pushback contains a real reason—a relevant scenario fact, a counterexample, or a valid trade-off implication. Sometimes it only contains the *language* of reasoning—“therefore”, “although”, “this implies”, “based on this”—wrapped around no supporting premise.

**Does the model respond to the reason, or to the language that makes something sound reasoned?**

And if it responds to the language: where in the network does that happen, and is it the same place the real reasons are processed?

# 2. Why this is the right question to ask

Kim & Khashabi (EMNLP Findings 2025) found that models are more persuaded by *casually phrased* feedback than by formal critique, and by *detailed reasoning even when its conclusion is wrong*. Both findings point the same way: what looks like sensitivity to argument quality may substantially be sensitivity to surface form.

That is a behavioural observation. Recent work has examined the internal representations of factual and opinion sycophancy, and has localised factual sycophancy with probes and causal interventions. What remains untested is the fully crossed question here: when propositional content is held fixed, does explicit reasoning style change behaviour and internal processing; and when style is held fixed, does a real reason change them?

It matters practically. A model that updates on the grammar of argument can be manipulated without a good reason—only something that sounds like one. It also sharpens the standard account of sycophancy: the possible failure is not only “agrees with whoever pushes hardest,” but “mistakes the form of reasoning for reasoning.”

# 3. Connection to the lab's work

Prof. Miyao's work on persuasive dialogue studies which strategies speakers use and which intentions are detectable in text (Sakurai & Miyao, ACL 2024). The rhetorical-structure work (Yokogawa, Ishigaki, Takamura & Miyao, INLG 2024) asks whether discourse-level properties can be controlled in generation.

This project takes the same object — discourse markers, argument structure — and asks what they do on the *receiving* end: does the presence of reasoning markers move a model's position independently of whether reasoning is present?

# 4. What exists, and what is left open

**Behavioural sycophancy work** (Sharma et al. 2023; Hong et al., SYCON-Bench; Chern et al., BeHonest) varies how hard the user pushes — expertise claims, repeated disagreement, confident phrasing. It does not vary whether the pushback contains a reason, and it observes only the output.

**Mechanistic sycophancy work** (Wang et al. 2025; Chen et al. 2024) opens the network and localises compliance. But it uses factual items with correct answers, where "did the position update?" collapses into "did retrieval fail?", and it reads an output-stage signal.

**Argument and uncertainty manipulations** provide the behavioural motivation. Kim & Khashabi (2025) compare full, truncated, answer-only, and casual rebuttals; models are influenced by detailed reasoning even when its conclusion is wrong and by casual assertiveness without justification. Their conditions do not fully cross propositional support with explicit reasoning style. BASIL (Atwell et al. 2025) measures deviations from Bayesian-rational updating under user perspectives on uncertain or no-ground-truth tasks. It supplies a useful no-ground-truth framework but studies elicited output probabilities rather than internal mechanisms or the content/style factorial.

**Internal geometry of sycophancy subtypes** (Baez et al. 2026, arXiv:2607.07003) is the nearest neighbour. It defines factual sycophancy as changing from correction to acceptance of a false claim, and opinion sycophancy as changing from neutrality to endorsing the user's opinion. It uses ten generic pushback phrases, labels generated responses as sycophantic or non-sycophantic, probes residual-stream activations at the final end-of-turn token, and tests transfer and steering between the two subtypes. Its axis is therefore *factual versus opinion sycophancy*, and its labels are based on the model's completed response. This project instead crosses *reason content versus explicit reasoning style* within opinion-like policy decisions, reads the model before it generates an explanation, and uses matched-condition patching to test the causal effect of each factor. Baez et al. also show why text-only and held-out controls are necessary: their TF–IDF baselines approach the activation-probe performance.

**Other recent mechanistic work** further narrows the claim. Genadi et al. (EACL 2026) find factual sycophancy signals in residual, MLP, and especially attention-head activations, with causal steering in a sparse set of middle-layer heads. Feng et al. (ACL 2026) show that sycophancy can change dynamically during reasoning and that final reasoning can mask it. Neither independently crosses the presence of a substantive reason with the presence of explicit reasoning language.

**Domain-specific sycophancy work** establishes that “no correct answer” is not a single setting. Cheng et al. (2025) define social sycophancy as excessive preservation of a user's face in advice and support-seeking contexts. Kaur (EMNLP Findings 2025) studies political argumentative prompts and finds that response alignment increases with argument strength. This project excludes personal advice and partisan political claims and instead uses controlled public-policy trade-offs (§5).

## The gap

No reviewed work independently crosses reasoning content and explicit reasoning style in counterarguments and then causally compares their internal effects.

Stated as the one sentence the thesis defends:

> **Does explicit reasoning style move an LLM independently of substantive reasons, and do style and substance rely on the same internal components?**

The thesis uses “internal stance-related signal” operationally, not “belief” in a human or philosophical sense. A logit lens predicts output; a probe establishes linear decodability; patching establishes causal influence on the output. None alone proves that the model privately believes one option while saying another.

# 5. Domain

*(Specified in response to lab feedback: "topics with no correct answer" is too broad a category.)*

**Self-contained policy trade-offs in climate, energy, and technology governance.**

Each scenario presents a concrete decision with **two feasible options that prioritise competing goals, neither clearly superior**. For example: a grid operator choosing between accelerating storage deployment or extending an existing baseload plant, where one option optimises cost certainty and the other optimises emissions trajectory.

Three properties make this domain the right choice:

1. **Normatively underdetermined.** Both options are feasible and non-dominated, and the scenario does not specify how competing goals should be weighted. There is no unique answer key. This reduces—rather than eliminates—the factual-retrieval confound.
2. **Self-contained.** All information needed sits in the scenario. The model is not drawing on contested world knowledge, which keeps the manipulation clean and avoids the factual-recall confound.
3. **Deliberately not personal or partisan.** Social sycophancy (personal consultations) and political sycophancy (partisan claims) are separate literatures with their own confounds — identity effects, alignment-training artefacts, ethics considerations. This project stays clear of both.

Scenarios are constructed rather than harvested, so that option symmetry can be controlled. Argumentation datasets such as IBM-ArgQ-Rank and CMV Winning Arguments may seed topics or argument structures, but adaptation to a fictional policy scenario breaks any automatic guarantee of quality. Every adapted reason must therefore be checked for relevance, validity under the supplied facts, and support for the intended option.

The planned corpus contains **60 underlying decisions**—20 climate, 20 energy, and 20 technology—and **two scenario variants per decision**, giving 120 scenarios. Each scenario requires four counterarguments supporting A and four semantically parallel counterarguments supporting B; the runtime selects the direction opposite the model's initial choice. Start with 12 decisions (24 scenarios) as the pilot. Scale only after the construction and manipulation checks pass.

The resulting **960 counterargument texts are produced through LLM-assisted generation plus validation**, rather than written manually. For each scenario and supported semantic option, a generation model receives the frozen definitions of RS, RP, NS, and NP, the permitted marker families, and the matching and confound constraints, and drafts the four cells together. Every draft is then checked by the automated validator for schema completeness, direction, marker use, forbidden authority or evidence language, length and sentence-count balance, duplicates, and option leakage; failures are revised or regenerated. After these checks, a trained human reviews **every text in its scenario context**, confirming the intended support direction, reason status, reasoning style, proposition preservation, and absence of nuisance differences, and edits or rejects it where necessary. LLM output is therefore only a draft: no item enters the frozen corpus without human approval. The stratified 20% subsample in §6.5 receives an additional independent review for reliability measurement.

# 6. Design

## 6.1 The two factors

Both are manipulated independently and fully crossed:

| Factor | Levels |
|---|---|
| **Reason content** | present — a relevant fact supplied by the scenario, a counterexample, or a valid trade-off implication that supports the opposing option / absent — endorses the opposing option but supplies no premise that makes it more desirable |
| **Reasoning style** | present — the relation between premises and conclusion is made explicit with causal, inferential, premise-framing, or concessive language / absent — the same propositions appear as plain declarative sentences without explicit inferential framing |

Four conditions per scenario:

|                       | style present | style absent |
|-----------------------|---------------|--------------|
| **reason present**    | RS | RP |
| **reason absent**     | **NS** | NP |

The four labels are frozen throughout the code and paper:

- **RS = reason + style**
- **RP = reason + plain**
- **NS = no reason + style**
- **NP = no reason + plain**

The critical style-without-substance contrast is **NS − NP**. If NS moves the model more than NP, explicit reasoning style has an effect even when no substantive reason is supplied.

The corresponding style contrast when a reason is present is **RS − RP**. The content contrasts are **RS − NS** when style is present and **RP − NP** when it is absent. The factorial interaction is `(RS − RP) − (NS − NP)`.

No individual word defines the style condition. The frozen inventory contains four marker families:

1. causal or justificatory: “because”, “given that”;
2. inferential: “therefore”, “thus”, “hence”, “this implies”;
3. premise or conclusion framing: “based on this”, “the key implication is”;
4. concession or contrast: “although”, “however”, “even so”.

Marker families are balanced across conditions and domains. Authority and evidence claims such as “experts agree” and “studies show” are excluded because they add evidential or social-pressure content rather than style alone. Confidence, politeness, expertise, and consensus are held constant.

## 6.2 Construction

For each scenario and each direction (support A and support B), text is assembled from components so the crossing cannot silently break:

```
[shared opening] + [core: reason present OR absent] + [style transformation: explicit OR plain]
```

Within each reason level, style is a **minimal edit**: RS and RP contain the same propositions, as do NS and NP. The explicit version states the relation between those propositions; the plain version juxtaposes them. The transformation may change punctuation and function words but may not add evidence, authority, certainty, evaluation, or a new premise. All four versions are matched on word count (target ±10%, hard maximum ±15%), sentence count, register, politeness, and opening phrase.

Each scenario therefore stores **eight counterarguments**: four cells supporting semantic option A and four supporting semantic option B. At runtime:

1. obtain the model's initial A/B preference;
2. append its argmax answer as the initial assistant turn;
3. branch the identical transcript into four independent conversations;
4. use the four counterarguments supporting the option opposite that initial answer.

The four conditions are never shown sequentially in one conversation. Semantic options are also presented in both A/B orders; counterarguments are linked to the semantic option ID, not to the displayed letter.

**Social pressure is held constant and low across all four cells**, and scored to verify it. It is not a factor in this design. This is a deliberate narrowing: the pressure axis is well covered by existing behavioural work, and holding it flat keeps the reason/style contrast clean.

## 6.3 The dependent variable

Each scenario presents two semantic options displayed as **A** and **B**. At a fixed answer slot, let `i` be the model's initial argmax option and `c` the counterargument-supported option. Define:

```
turn 1   scenario + fixed question + "Answer:"
         -> i = argmax{logit(A), logit(B)}
         -> m_before = logit(c) - logit(i)       [normally negative]

turn 2   append the selected initial answer, then one counterargument
turn 3   fixed question + "Answer:"
         -> m_after = logit(c) - logit(i)

movement_toward_counter = m_after - m_before
```

Positive movement means the model moved toward the counterargument, whether or not it flipped. The exact next-token strings are fixed after checking that both answer labels are single tokens in the Llama tokenizer. Logits are scored deterministically; no answer explanation is generated in the primary experiment. Because there is no generated rationale, “a flaw in the model's stated rationale” is not used as a reason subtype.

This is a large improvement over a binary support/oppose tag:

- **Continuous**, so effects are visible without a flip occurring
- **No parsing and no LLM judge** — it is read directly from the model
- **A single forward pass** per reading, no free-form generation needed, so the behavioural run is fast
- **Directly compatible with logit lens** (§7.2), because the same quantity is defined at every layer

A binary flip variable is recorded alongside it for comparability with the behavioural literature.

## 6.4 Initial near-ties

The primary analysis retains all validated scenarios and uses the initial argmax to choose the opposing counterargument. Excluding items using the same run's initial margin would change the target population and could produce model-specific scenario sets.

Near-ties are therefore handled in two transparent secondary analyses:

1. report results stratified by absolute initial margin;
2. repeat the analysis after excluding `|m_before| < τ`, where `τ` is frozen from the pilot rather than chosen from the full results.

The full-sample result remains primary. The same scenario IDs are reported for both models even when their initial choices differ.

## 6.5 Validation

Automated checks run on all items: schema validity, label/direction consistency, marker-family counts, sentence and word-count bounds, forbidden authority/evidence phrases, duplicate detection, and option-name leakage.

Human validation checks the constructs that marker counting cannot establish:

- **Substantive support:** does the text provide a relevant premise that genuinely supports its intended option under the supplied scenario?
- **Perceived reasoning style:** does it explicitly sound as though a premise is being connected to a conclusion?
- **Proposition preservation:** do RS and RP express the same substantive claims, and do NS and NP express the same claims?
- **No-reason integrity:** do NS and NP avoid adding evidence, a counterexample, or a trade-off implication?
- **Confound balance:** confidence, pressure, politeness, authority, and credibility.
- **Scenario validity:** are both policy options feasible and non-dominated, and is no unique value weighting specified?

One trained reviewer checks every item. Two independent annotators rate a stratified 20% subsample covering all domains, directions, cells, and marker families. Report Cohen's κ for binary decisions and an ordinal agreement statistic for Likert ratings; do not set an expected agreement value in advance. Exclusion and revision rules are frozen before the full corpus is inspected.

The manipulation succeeds only if, in held-out ratings, real-reason cells exceed no-reason cells on substantive support, styled cells exceed plain cells on perceived reasoning style, RS/RP and NS/NP pass proposition-preservation checks, and nuisance ratings do not show a practically important imbalance. The marker inventory and validation rubric are frozen after the pilot.

# 7. Method

Ordered so that each stage is only attempted once the previous one has shown there is something to explain. Nothing mechanistic runs until the behavioural effect is verified.

## 7.1 Behavioural (stage 1)

Run the full 2×2 on both model variants, with both semantic option orders. The primary outcome is `movement_toward_counter`; flip rate is secondary.

The confirmatory model is a factorial regression with reason, style, model variant, their interactions, domain, option order, and absolute initial margin. Inference uses a cluster bootstrap over **underlying decision ID**, keeping both scenario variants, all four cells, both option orders, and both models together in every resample. This is the correct independence unit; clustering only by scenario would treat the two variants of one decision as independent.

Report the factorial main effects and interaction, plus these pre-planned within-model contrasts:

- **NS − NP:** style effect without a real reason;
- **RS − RP:** style effect with a real reason;
- **RS − NS:** content effect with explicit style;
- **RP − NP:** content effect in plain language.

Use two-sided 95% confidence intervals and adjust the four planned contrast p-values with Holm's method. Analyse flips with the same clustered resampling rather than treating the four cells as independent. Report effect sizes and intervals even when a threshold test is non-significant.

> **Gate:** mechanistic analysis of style proceeds only if the manipulation check passes and a style contrast is stable across option orderings. If NS − NP is null but RS − RP is present, the claim becomes “style modulates substantive reasons,” not “empty reasoning language persuades.” Content localisation uses RS − NS and RP − NP—not RS − RP.

## 7.2 Logit lens (stage 2)

At the final answer-slot token, cache the residual stream after every transformer block. For layer `l`, apply Llama's final RMSNorm and unembedding, then calculate the same counter-versus-initial option margin:

```
z_l = lm_head(final_rmsnorm(h_l))
m_l = z_l[counter_option] - z_l[initial_option]
```

This gives an approximate depth trajectory for each condition. Compare the same valid factorial contrasts as in §7.1: NS/NP and RS/RP for style; RS/NS and RP/NP for content. Curves receive decision-clustered confidence bands. Do not select a “critical layer” merely because one point has the smallest p-value; identify a contiguous candidate band on a discovery split and verify it on held-out decisions.

**What it can show:** where answer-relevant information becomes readable through the model's output head and how that trajectory differs by condition.

**What it cannot show:** a separate private belief. Ordinary logit-lens readings in early layers are especially unreliable; strong early/middle-layer claims require a tuned lens or must be described as exploratory.

## 7.3 Controlled linear probes (stage 3 — core representational analysis)

A supervised probe is useful here, but its target must match what the experiment can identify. A probe trained to predict the model's eventual A/B answer would largely duplicate the output readout; disagreement between that probe and the logit lens would not, by itself, demonstrate “believed A but said B.” The core probes therefore test **which experimental factor and response tendency are linearly decodable**, not an unobservable private belief.

At each layer, fit separate L2-regularised linear models on the final answer-slot activation to decode:

- reason present versus absent, balanced over style;
- reasoning style present versus absent, balanced over reason;
- signed behavioural movement, using ridge regression; flip/no-flip is a secondary classifier.

Split by underlying decision, not by row: both scenario variants, both directions, both orders, and all four conditions for one decision must stay in one partition. Use nested cross-validation to choose regularisation. In addition to a fixed grouped test set, run leave-one-domain-out and held-out-marker-family tests. Balance initial A/B choice and option order within labels.

Required controls:

- shuffled-label permutation distribution;
- a TF–IDF/text-only classifier on the counterargument;
- exact duplicate and paraphrase leakage checks;
- reporting AUROC/balanced accuracy with bootstrap intervals rather than accuracy alone;
- train-set-only standardisation and hyperparameter selection;
- selectivity: true-label performance minus shuffled-label performance.

High probe performance establishes linear decodability, not causal use. The probe weight is not automatically treated as a steering or “belief” direction. Causal interpretation comes from patching.

## 7.4 Residual-stream activation patching (stage 4)

For each matched pair, run a source condition and target condition with the same model, decision, scenario variant, supported semantic option, option order, and final query. At the final answer-slot token, replace the target residual stream after block `l` with the source activation and re-run the remaining network.

The complete contrast map is:

| Factor isolated | Style setting | Matched cells, both directions |
|---|---|---|
| style | reason present | RS ↔ RP |
| style | reason absent | NS ↔ NP |
| content | style present | RS ↔ NS |
| content | style absent | RP ↔ NP |

For source `s`, target `t`, and patched target `p_l`, the primary causal effect is the raw signed change:

```
patch_effect_l = margin(p_l) - margin(t)
```

Also report normalised recovery `(margin(p_l)-margin(t))/(margin(s)-margin(t))` only when the unpatched source–target gap exceeds a frozen minimum; otherwise the denominator is unstable.

Required sanity checks:

- self-patching changes the margin by approximately zero;
- patching the final residual state reproduces the source logits up to numerical tolerance;
- an unrelated, label-balanced source does not systematically move the target in the predicted direction;
- results replicate in both patch directions and on held-out decisions;
- candidate layers are selected on a discovery split and confirmed on a test split.

Because the counterarguments differ in tokenisation, the primary intervention patches only the common final answer-slot position. The conclusion is therefore about the **integrated causal state available at the answer slot**, not necessarily the earlier token at which the feature was first computed.

## 7.5 Attention versus MLP patching (stage 5)

Only within residual-stream layer bands confirmed in §7.4, patch the **attention output** and **MLP output** separately from source into target at the final answer slot. Do not use zero ablation as the primary comparison: it measures component damage as well as the factor of interest. If neither component alone recovers the residual-stream effect, patch both jointly to test for distributed or nonlinear dependence.

This stage asks which sublayer carries the matched-condition effect. It does not assume in advance that “style is attention” and “content is MLP.” Genadi et al. motivate attention as a plausible site for factual sycophancy, but that is not evidence for this experiment's content/style split. Individual-head and path-level analysis are out of scope unless the sublayer result is unusually sparse and time remains.

## 7.6 What this stack can and cannot establish

Stated explicitly, because the method choice was questioned in feedback and the honest answer is more useful than a confident one.

| Method | Establishes | Does not establish |
|---|---|---|
| Behavioural 2×2 | that form matters, and how much | anything internal |
| Logit lens | where the answer distribution becomes readable through the output head | a separate internal belief |
| Controlled probes | where factor/outcome information is linearly decodable and whether it generalises | that the model uses the decoded feature |
| Residual patching | which answer-slot layer states causally transmit a matched-condition effect | a complete circuit or human-like belief |
| Attention/MLP patching | which sublayer outputs carry the effect at confirmed layers | individual heads or token-level origin |

Difference-in-means over persona contrasts was dropped for the reason raised in feedback: a borrowed sycophancy direction may not be separable from other persona traits in this domain. The supervised probes use this task's own factor labels and grouped held-out evaluation, but even they are not called belief readouts. Patching supplies the causal evidence.

# 8. Base versus instruction-tuned

The full behavioural experiment and probe/logit-lens analysis run on both Llama-3.1-8B and Llama-3.1-8B-Instruct. Exact model and tokenizer revisions, dtype, library versions, prompt templates, and answer-token IDs are pinned in the preregistration config.

Hong et al. document a behavioural gap between base and tuned models on stance-holding under contentless disagreement. This motivates a comparison, but it does not make instruction tuning a controlled causal treatment: the checkpoints and required prompt formats differ in more than one way. The primary claims are therefore within-model content/style effects. Model-by-factor interactions are secondary and are described as associations with instruction tuning, not as proof of what RLHF “installed.”

The instruct model uses its official chat template. The base model uses one frozen plain-text dialogue scaffold, piloted to ensure that the next-token A/B constraint is meaningful. A small prompt-format robustness set is run on both. Raw activation magnitudes are never compared across checkpoints; comparisons use behavioural effect sizes, probe performance, relative depth, and patch effects within each model.

To control compute, the full 32-layer patching sweep is first run on the instruct model over a preregistered mechanistic subset: one scenario variant from 40 underlying decisions, both option orders, and the four “present → absent” contrasts (RS→RP, NS→NP, RS→NS, RP→NP). Reverse-direction patches and the base-model replication are run only at the confirmed layer band. This keeps confirmatory causal tests while avoiding an unnecessary sweep of every direction on every item.

# 9. Timeline and decision points

| When | Work | Gate |
|---|---|---|
| **Month 1** | Implement schemas, prompt renderer, tokenizer checks, logit scorer, and automated validation; construct 12 pilot decisions × 2 scenarios × 8 directional counterarguments; run human pilot | **Do RS/RP and NS/NP preserve propositions while the reason and perceived-style checks separate as intended?** If not, redesign before scaling |
| **Month 2** | Scale to 60 decisions × 2 scenario variants; complete validation; freeze corpus and config; run behavioural 2×2 on both models and both option orders | **Is any style effect stable across orderings? Is the content manipulation behaviourally active?** |
| **Month 3** | Logit-lens profiles; grouped factor/outcome probes; held-out-domain, marker-family, text-only, and permutation controls | **Are factor signals decodable beyond the permutation null and stable under grouped generalisation?** |
| **Month 4** | Patching sanity tests; instruct-model residual sweep on the mechanistic subset; held-out confirmation; reverse patches and base replication at confirmed layers | **Do final-state recovery and self-patch controls pass? Does the causal effect replicate on held-out decisions?** |
| **Month 5** | Attention/MLP patching at confirmed layers; robustness, release, and writing; hard freeze mid-month | — |

**Cut order if the schedule slips:** (1) individual-head analysis, which is not planned anyway; (2) attention/MLP split; (3) base-model causal replication, while retaining its behavioural arm; (4) behavioural flip analysis. The validated corpus, continuous behavioural factorial, grouped factor probes, and instruct-model residual patching are the minimum coherent project.

# 10. Risks

**The style manipulation is not clean.** Connectives carry semantic relations—“therefore” presupposes an inference and “however” presupposes a contrast—so style may change interpretation even without adding a proposition. This is the load-bearing assumption and is tested before scaling. If the broad inventory fails, narrow it to the marker family that passes proposition-preservation and perceived-style checks. Do not fall back to “studies show,” expertise, hedging, or certainty: those introduce evidence, authority, or confidence confounds.

**Null result on NS versus NP.** First check order and marker-family heterogeneity. If NS−NP remains null but RS−RP is present, study style as a moderator of real reasons. If both style contrasts are null, do not claim a style mechanism; redirect causal analysis to the valid content contrasts RS−NS and RP−NP.

**The probes exploit lexical shortcuts.** Marker-family holdout, decision-grouped splits, leave-one-domain-out tests, and the TF–IDF baseline expose this. A probe that fails those controls is reported as in-distribution decodability only and is not used to choose causal sites.

**Model has no meaningful initial preference.** The full sample remains primary; near-ties are stratified and filtered only in a preregistered robustness analysis (§6.4).

**Scenarios are too easy or too artificial.** Mitigated by seeding argument content from ArgQ/CMV and by the human manipulation check.

**Patching cost is underestimated.** A full corpus × model × direction × layer sweep would require tens of thousands of forward continuations. §8 fixes a mechanistic subset and staged replication before results are seen.

# 11. What I need

1. **Compute.** Access to one A100-class GPU, with enough allocation for roughly 2,500 behavioural/logit-lens runs plus a staged patching sweep of approximately 10,000–20,000 continuations. The model is loaded one checkpoint at a time in bfloat16; probe training is offline and does not backpropagate through the LLM.
2. **Annotators.** One reviewer for the full corpus and two independent annotators for the stratified 20% reliability/manipulation-check sample (approximately 190 counterargument items plus scenario checks).
3. **Sign-off on the gates.** In particular: the manipulation criteria, grouped probe controls, mechanistic subset, and rule for narrowing the claim after a null.
4. **A view on the representational claim (§7.3).** I have kept controlled task-specific probes as core, but I no longer treat a probe trained on answer-correlated labels as an independent belief meter. Is “decodable factor/update signal plus causal patching” the appropriate level of claim for the thesis?
5. **Scope calibration.** Is 60 decisions × 2 scenario variants × 8 directional texts manageable for five months, given that the full causal sweep is restricted to a preregistered subset?
6. **Target venue.**

# 12. Limitations

- **Logit lens is uncalibrated.** Early-layer readings are unreliable; they are exploratory unless a tuned lens is added.
- **A probe shows decodability, not use or belief.** The experimental labels and eventual response can be linearly predictable without constituting a private position. Patching tests causal influence on output, not philosophical belief.
- **Marker inventory is a choice.** Results are relative to the specific set of markers frozen in month 1, and that set will be released.
- **Style and content are not perfectly separable in language.** §10 states the mitigation; the residual confound will be stated in the paper rather than argued away.
- **Constructed policy domain, English only, single-turn pushback.** Multi-turn escalation and naturally occurring debate are out of scope.
- **Base and tuned models are prompted differently**, so part of any gap is prompting.
- **One model family.** Base-versus-instruct comparison does not establish generality across architectures.
- **Resolution.** Layer bands and sublayers, not individual heads.

# 13. Deliverables

1. **Thesis and target paper.** A controlled behavioural, representational, and causal comparison of substantive reasons and explicit reasoning style on normatively underdetermined policy decisions.
2. **A released corpus:** 60 decisions × 2 scenario variants × 2 argument directions × 4 cells, with semantic option IDs, marker families, annotation, agreement, manipulation checks, and source provenance.
3. **Released code:** schema validator, prompt renderer, behavioural/logit scorer, logit-lens cache, grouped probe pipeline, residual and component patching harnesses, analysis scripts, tests, and frozen configuration.

# 14. Key references

**Argument quality and form.** Kim & Khashabi 2025, [*Challenging the Evaluator*](https://arxiv.org/abs/2509.16533). Atwell et al. 2025, [*Quantifying Sycophancy as Deviations from Bayesian Rationality in LLMs (BASIL)*](https://arxiv.org/abs/2508.16846). Kaur 2025, [*Echoes of Agreement: Argument Driven Sycophancy in Large Language Models*](https://aclanthology.org/2025.findings-emnlp.1241/).

**Mechanism.** Wang et al. 2026, [*When Truth Is Overridden*](https://ojs.aaai.org/index.php/AAAI/article/view/40645) — logit lens and activation patching. Chen et al. 2024, [*From Yes-Men to Truth-Tellers*](https://arxiv.org/abs/2409.01658) — path patching and pinpoint tuning. Baez et al. 2026, [*Dissociating the Internal Representations of Sycophancy in LLMs*](https://arxiv.org/abs/2607.07003) — factual/opinion probe transfer, steering, and geometry. Genadi et al. 2026, [*Sycophancy Hides Linearly in the Attention Heads*](https://aclanthology.org/2026.eacl-long.324/). Feng et al. 2026, [*Good Arguments Against the People Pleasers*](https://aclanthology.org/2026.acl-long.1126/). Zhang & Nanda 2023, [activation-patching metrics and controls](https://arxiv.org/abs/2309.16042).

**Behavioural.** Sharma et al. 2023, [*Towards Understanding Sycophancy in Language Models*](https://arxiv.org/abs/2310.13548). Hong et al. 2025, [*SYCON-Bench*](https://arxiv.org/abs/2505.23840). Chern et al. 2024, [*BeHonest*](https://arxiv.org/abs/2406.13261).

**Domain scoping.** Cheng et al. 2025, [*ELEPHANT: Measuring and Understanding Social Sycophancy in LLMs*](https://arxiv.org/abs/2505.13995). Kaur 2025 covers political argumentative prompts and is listed above.

**Probing.** Belinkov 2022, *Probing Classifiers: Promises, Shortcomings, and Advances*. Hewitt & Liang 2019, [designing and interpreting probes with control tasks](https://arxiv.org/abs/1909.03368). Baez et al. and Genadi et al. provide the closest sycophancy-specific probe controls. Belrose et al. 2023, [*The Tuned Lens*](https://arxiv.org/abs/2303.08112).

**Faithfulness framing.** Turpin et al. 2023 (arXiv:2305.04388); Lanham et al. 2023 — the literature that frames output-versus-internal divergence.

**Lab.** Sakurai & Miyao 2024 (ACL 2024, 1635–1657). Yokogawa, Ishigaki, Takamura & Miyao 2024 (INLG 2024).

**Data.** IBM-ArgQ-Rank-30k (Gretz et al. 2020); CMV Winning Arguments (Tan et al. 2016).

# 15. Implementation specification

This section is the hand-off from research plan to code. Any change to a frozen field creates a new experiment version rather than silently overwriting the old one.

## 15.1 Source-data schema

Store one JSONL record per scenario:

```json
{
  "decision_id": "energy_001",
  "scenario_id": "energy_001_v1",
  "variant_id": 1,
  "domain": "energy",
  "scenario_text": "...",
  "options": {
    "opt_1": "...",
    "opt_2": "..."
  },
  "counterarguments": {
    "opt_1": {
      "RS": {"text": "...", "marker_family": "inferential"},
      "RP": {"text": "...", "marker_family": null},
      "NS": {"text": "...", "marker_family": "inferential"},
      "NP": {"text": "...", "marker_family": null}
    },
    "opt_2": {
      "RS": {"text": "...", "marker_family": "inferential"},
      "RP": {"text": "...", "marker_family": null},
      "NS": {"text": "...", "marker_family": "inferential"},
      "NP": {"text": "...", "marker_family": null}
    }
  },
  "provenance": [],
  "validation_status": "draft"
}
```

Semantic option IDs never change. Display letters A/B are assigned by the prompt renderer so option-order counterbalancing cannot corrupt counterargument direction. Annotation records live in a separate table keyed by `scenario_id`, supported semantic option, and condition ID.

## 15.2 Frozen experiment configuration

The versioned YAML configuration contains:

- model and tokenizer repository IDs plus immutable revisions;
- dtype, device, library versions, random seeds, and deterministic settings;
- instruct and base prompt templates and their hashes;
- exact answer strings/token IDs and the answer-slot index rule;
- condition map `{RS, RP, NS, NP}` and contrast definitions;
- marker inventory, forbidden phrases, and length tolerances;
- train/validation/test decision IDs and mechanistic-subset IDs;
- near-tie threshold used only for robustness;
- statistical contrasts, bootstrap seed/count, multiplicity rule, and confidence level;
- patch hook names, layer indexing convention, component definitions, and numerical tolerances.

## 15.3 Run tables and cached artefacts

The behavioural output table has one row per model × scenario × option order × condition and includes:

`run_id`, config hash, model revision, decision/scenario IDs, domain, displayed order, initial semantic choice, counterargument semantic target, condition, marker family, prompt hash, answer-token IDs, before/after logits, before/after signed margins, movement, flip, token counts, and validation flags.

Cache residual activations in sharded tensors with a manifest recording model revision, prompt hash, layer, hook point, token index, shape, and dtype. Store probe splits and fitted preprocessing with each probe. Store every patch as a tidy row containing source/target IDs, contrast, direction, layer/component, unpatched source and target margins, patched margin, raw effect, optional normalised recovery, and control type.

## 15.4 Execution order

1. Validate JSONL and annotations; fail on missing cells, wrong support direction, duplicate IDs, forbidden markers, or length violations.
2. Render both option orders and verify answer tokens and semantic-to-letter mappings.
3. Run one synthetic smoke case and one real pilot case on each model.
4. Run the 24-scenario pilot; freeze the corpus rules, marker inventory, templates, thresholds, splits, and analysis config.
5. Run the full behavioural experiment and export the immutable run table.
6. Produce factorial estimates and gates before selecting mechanistic contrasts.
7. Cache layer activations; run logit lens and grouped probes with all controls.
8. Run patching sanity tests, the discovery sweep, and held-out confirmation.
9. Run component patching only at confirmed layers.
10. Generate all tables and figures from immutable run tables, never from hand-edited intermediate files.

## 15.5 Minimum automated tests

- both answer labels are exactly one next token under every frozen template;
- swapping displayed option order preserves semantic options and counterargument direction;
- all four branches share the identical pre-counterargument transcript;
- signed movement is positive when a synthetic logit pair moves toward the counter option;
- direct final logits match logits reconstructed from the cached final residual within tolerance;
- self-patching produces approximately zero change;
- final-residual source→target patching reproduces source logits within tolerance;
- no `decision_id` crosses probe train/validation/test splits;
- no paraphrase or scenario variant crosses grouped splits;
- rerunning an identical prompt gives identical logits and prompt hashes.

Coding should begin with §§15.1–15.2, the validator, renderer, and logit-scoring smoke tests—not with large-scale scenario generation or patching.
