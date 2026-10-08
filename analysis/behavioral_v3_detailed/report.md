# Behavioural analysis of full v3 — Llama-3.1-8B Base and Instruct

Analysis `behavioral_v3_detailed_v1`; dataset `full_v3_multimarker_openings` (**draft**). Developmental, exploratory results on a researcher-reviewed draft. No human manipulation check or inter-annotator reliability study has been run, so nothing here shows that the reason or style manipulations operate as intended for human readers. This report describes observed model behaviour only. No mechanistic analysis has been run.

## 1. Evidence validation

All 143 recorded checks passed before any estimate was computed. For every run this means: `run_status`, every file SHA-256 recomputed against `RUN_METADATA.json` and against this analysis's pinned hashes, the expected counts, model revision, config hashes and answer tokens, and every row re-scored from its stored logits with the project's scoring function and matched exactly to the plan branch its initial argmax selects.

| run | run ID | status | initials | exact initial ties | movement rows | post ties | defined flips | scores SHA-256 |
|---|---|---|---|---|---|---|---|---|
| Base | `base-full-v3` | complete_with_recorded_exclusions_and_secondary_ties | 240 | 3 | 4266 | 101 | 4165 | `e958822a78a853f9…` |
| Instruct | `instruct-full-v3` | complete_with_recorded_exclusions | 240 | 10 | 4140 | 0 | 4140 | `0db96e735637bb5e…` |
| Base R1 (B line first) | `base-ab-bfirst-v1` | complete_with_recorded_exclusions_and_secondary_ties | 240 | 6 | 4212 | 288 | 3924 | `4219e6fc7144a458…` |
| Base R3 (minimal scaffold) | `base-minimal-ab-v1` | complete_with_recorded_exclusions_and_secondary_ties | 240 | 32 | 3744 | 247 | 3497 | `023ace2a3db65892…` |

- Stimuli `9922fd402b31a54265b6c9abd72b175d65f430bd29dc3829a937f13a62b1661c`; pinned behavioural config file `f682fcad2749fb298c196997e01729a09b0e40db589075d9e99a1cf8a18c1d25` (content `4e18a0cf09dc31def4030e19e0f30072474d888214924f7b7042e20fc9735238`).
- Model revisions: Base `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`, Instruct `0e9e39f249a16976918f6564b8830bc894c89659`; answer tokens ` A`=362, ` B`=426.
- Reference evidence archived read-only at `/data/vidhi/archive/reasonstyle/behavioral_v3_full_evidence_v1` (server) and `runs/archive/behavioral_v3_full_evidence_v1` (Mac); `SHA256SUMS` hash `71751a241576b4c21a058a451458d67fd884076bf57a3d831e8e1bb803476181`. Robustness evidence at `/data/vidhi/archive/reasonstyle/behavioral_v3_robustness_evidence_v1` and `runs/archive/behavioral_v3_robustness_evidence_v1`; `SHA256SUMS` hash `c890f749cb98d433059b1291a203954183c9085f00096f4447df23d16f4d7d01`. The pinned config is included in both archives.
- Variant-runner equivalence: the reference Base prompts re-rendered by the new variant code are byte-identical to the original prompts (tests), and the variant runner reproduced all 240 reference Base initial logits bit for bit (`complete`, 0 mismatches).

## 2. Estimand

- **Block:** run × `scenario_id` × `order_id` × `opening_id` for a non-tied initial prompt: RS(m1), RS(m2), NS(m1), NS(m2), RP, NP (m1, m2 = the group's two allocated markers, one per family).
- **Cells:** RS and NS are the unweighted mean of their two marker rows; RP and NP are the single shared controls, entered once and never duplicated.
- **Contrasts** are formed within each block: NS − NP, RS − RP, RS − NS, RP − NP and (RS − RP) − (NS − NP).
- **Weighting:** valid blocks equally within `decision_id`, then each decision equally. Openings enter equally because every valid initial has all three.
- **Outcomes:** primary `movement_toward_counter` = m_after − m_before; secondary `flip`, defined only where the post reading is not an exact tie.
- **Ties:** an exact initial tie forms no block (its branches do not exist). Exact post ties stay in movement with m_after = 0 and leave flip only. No tie-breaking.
- **Inference:** 9999 replicates, seed 20261005, NumPy PCG64 (NumPy 2.2.6); whole decisions resampled within domain (20 climate, 20 energy, 20 technology draws each replicate), carrying both scenario variants, all openings, both orders, all conditions and every run together; two-sided percentile 95% intervals; the draw matrix is saved in `bootstrap/draws.csv.gz`. Bootstrap p = min(1, 2·min(#≤0 + 1, #≥0 + 1)/(B + 1)).
- **Multiplicity:** Holm over the four planned simple contrasts, separately by outcome and model; the interaction is reported separately; Holm over 48 individual-marker style comparisons. Everything else is exploratory and unadjusted.

## 3. Core developmental contrasts

### Movement (primary)

| run | contrast | estimate [95% CI] | interval | p (boot) | Holm p (planned family) | decisions | initials | blocks | movement rows | flip rows |
|---|---|---|---|---|---|---|---|---|---|---|
| Base | NS − NP | +0.123 [+0.101, +0.147] | above 0 | <0.001 | <0.001 | 60 | 237 | 711 | 2133 | 0 |
| Base | RS − RP | -0.013 [-0.026, 0.000] | includes 0 | 0.059 | 0.059 | 60 | 237 | 711 | 2133 | 0 |
| Base | RS − NS | +0.137 [+0.105, +0.170] | above 0 | <0.001 | <0.001 | 60 | 237 | 711 | 2844 | 0 |
| Base | RP − NP | +0.273 [+0.238, +0.308] | above 0 | <0.001 | <0.001 | 60 | 237 | 711 | 1422 | 0 |
| Base | (RS − RP) − (NS − NP) | -0.136 [-0.159, -0.114] | below 0 | <0.001 | separate (unadjusted) | 60 | 237 | 711 | 4266 | 0 |
| Instruct | NS − NP | -0.754 [-0.822, -0.688] | below 0 | <0.001 | <0.001 | 60 | 230 | 690 | 2070 | 0 |
| Instruct | RS − RP | -0.020 [-0.053, +0.015] | includes 0 | 0.259 | 0.259 | 60 | 230 | 690 | 2070 | 0 |
| Instruct | RS − NS | +0.554 [+0.460, +0.650] | above 0 | <0.001 | <0.001 | 60 | 230 | 690 | 2760 | 0 |
| Instruct | RP − NP | -0.181 [-0.249, -0.113] | below 0 | <0.001 | <0.001 | 60 | 230 | 690 | 1380 | 0 |
| Instruct | (RS − RP) − (NS − NP) | +0.735 [+0.663, +0.809] | above 0 | <0.001 | separate (unadjusted) | 60 | 230 | 690 | 4140 | 0 |

Cell means (decision-weighted): Base RS 1.58, RP 1.60, NS 1.45, NP 1.32; Instruct RS 7.67, RP 7.69, NS 7.11, NP 7.87.

### Flip (secondary)

| run | contrast | estimate [95% CI] | interval | p (boot) | Holm p (planned family) | decisions | initials | blocks | movement rows | flip rows |
|---|---|---|---|---|---|---|---|---|---|---|
| Base | NS − NP | +0.014 [+0.002, +0.028] | above 0 | 0.013 | 0.053 | 60 | 237 | 685 | 2055 | 2041 |
| Base | RS − RP | 0.000 [-0.009, +0.008] | includes 0 | 0.973 | 0.973 | 60 | 236 | 694 | 2082 | 2067 |
| Base | RS − NS | -0.010 [-0.025, +0.002] | includes 0 | 0.112 | 0.335 | 60 | 236 | 702 | 2808 | 2767 |
| Base | RP − NP | +0.007 [-0.007, +0.022] | includes 0 | 0.327 | 0.654 | 60 | 236 | 678 | 1356 | 1356 |
| Base | (RS − RP) − (NS − NP) | -0.011 [-0.028, +0.003] | includes 0 | 0.139 | separate (unadjusted) | 60 | 235 | 673 | 4038 | 4015 |
| Instruct | NS − NP | 0.000 [0.000, 0.000] | includes 0 | 1.000 | 1.000 | 60 | 230 | 690 | 2070 | 2070 |
| Instruct | RS − RP | 0.000 [0.000, 0.000] | includes 0 | 1.000 | 1.000 | 60 | 230 | 690 | 2070 | 2070 |
| Instruct | RS − NS | 0.000 [0.000, 0.000] | includes 0 | 1.000 | 1.000 | 60 | 230 | 690 | 2760 | 2760 |
| Instruct | RP − NP | 0.000 [0.000, 0.000] | includes 0 | 1.000 | 1.000 | 60 | 230 | 690 | 1380 | 1380 |
| Instruct | (RS − RP) − (NS − NP) | 0.000 [0.000, 0.000] | includes 0 | 1.000 | separate (unadjusted) | 60 | 230 | 690 | 4140 | 4140 |

Flip rates: Base RS 0.978 (1362/1387 rows), RP 0.974 (681/698 rows), NS 0.982 (1371/1392 rows), NP 0.964 (665/688 rows); Instruct RS 1.000 (1380/1380 rows), RP 1.000 (690/690 rows), NS 1.000 (1380/1380 rows), NP 1.000 (690/690 rows). Rates at or near 1 leave flip almost no room to vary; where every defined row flips, the flip contrasts are exactly zero by construction and carry no information about the factors.

### Base versus Instruct on the common valid subset

237 Base and 230 Instruct initials are valid; 227 are valid in both (60 decisions). Differences below (Instruct − Base) are associations between two checkpoints with different prompt formats, not effects of instruction tuning.

| contrast | Base | Instruct | Instruct − Base [95% CI] | interval |
|---|---|---|---|---|
| NS − NP | +0.118 | -0.756 | -0.874 [-0.942, -0.807] | below 0 |
| RS − RP | -0.012 | -0.018 | -0.006 [-0.041, +0.031] | includes 0 |
| RS − NS | +0.146 | +0.551 | +0.405 [+0.313, +0.497] | above 0 |
| RP − NP | +0.276 | -0.188 | -0.464 [-0.537, -0.392] | below 0 |
| (RS − RP) − (NS − NP) | -0.130 | +0.738 | +0.868 [+0.798, +0.941] | above 0 |

### Option-order stability

| model | contrast | o1 | o2 | o1 − o2 [95% CI] | same sign | both intervals exclude 0 on the same side |
|---|---|---|---|---|---|---|
| Base | NS − NP | +0.129 | +0.116 | +0.013 [-0.033, +0.059] | yes | yes |
| Base | RS − RP | -0.025 | 0.000 | -0.026 [-0.060, +0.009] | no | no |
| Base | RS − NS | +0.191 | +0.080 | +0.112 [+0.044, +0.180] | yes | yes |
| Base | RP − NP | +0.345 | +0.195 | +0.150 [+0.082, +0.219] | yes | yes |
| Base | (RS − RP) − (NS − NP) | -0.154 | -0.116 | -0.038 [-0.081, +0.005] | yes | yes |
| Instruct | NS − NP | -0.774 | -0.728 | -0.047 [-0.107, +0.012] | yes | yes |
| Instruct | RS − RP | -0.024 | -0.015 | -0.010 [-0.057, +0.037] | yes | no |
| Instruct | RS − NS | +0.580 | +0.526 | +0.054 [-0.019, +0.129] | yes | yes |
| Instruct | RP − NP | -0.170 | -0.187 | +0.017 [-0.060, +0.092] | yes | yes |
| Instruct | (RS − RP) − (NS − NP) | +0.750 | +0.713 | +0.037 [-0.031, +0.102] | yes | yes |

No stability criterion is fixed in the repository; the last two columns are descriptive.

### Initial margin

|m_before| over valid initials: Base median 0.562 (IQR 0.375–0.812, 67 of 237 below τ = 0.405); Instruct median 1.250 (IQR 0.656–2.469, 42 of 230 below τ = 0.405). Logits are read in bfloat16, whose spacing near the observed logit magnitudes is coarse (Base margins are multiples of 1/16), which is consistent with the number of exact ties.

| model | contrast | slope per logit of |m_before| [95% CI] | excluding near ties (sensitivity) [95% CI] | full sample |
|---|---|---|---|---|
| Base | NS − NP | -0.036 [-0.110, +0.037] | +0.114 [+0.087, +0.142] | +0.123 |
| Base | RS − RP | -0.046 [-0.087, -0.006] | -0.023 [-0.042, -0.006] | -0.013 |
| Base | RS − NS | +0.131 [+0.044, +0.229] | +0.182 [+0.140, +0.224] | +0.137 |
| Base | RP − NP | +0.140 [+0.047, +0.247] | +0.320 [+0.278, +0.362] | +0.273 |
| Base | (RS − RP) − (NS − NP) | -0.009 [-0.088, +0.066] | -0.138 [-0.164, -0.112] | -0.136 |
| Instruct | NS − NP | -0.040 [-0.086, +0.003] | -0.786 [-0.863, -0.713] | -0.754 |
| Instruct | RS − RP | -0.013 [-0.036, +0.012] | -0.025 [-0.063, +0.015] | -0.020 |
| Instruct | RS − NS | +0.043 [-0.016, +0.102] | +0.579 [+0.471, +0.689] | +0.554 |
| Instruct | RP − NP | +0.016 [-0.027, +0.056] | -0.181 [-0.259, -0.105] | -0.181 |
| Instruct | (RS − RP) − (NS − NP) | +0.027 [-0.014, +0.070] | +0.761 [+0.678, +0.846] | +0.735 |

Quartile-specific estimates are in `tables/margin_quartiles` and figure 11. The full valid sample stays primary; τ was fixed before these data and was not re-chosen.

## 4. Robustness: Base response labels and prompt format

Predeclared in `configs/robustness/base_prompt_variants_v1.yaml` before any variant ran. Each variant used the cached pinned Base revision, bfloat16 and deterministic CUDA on one idle RTX A6000, scored its own initial prompts first, then its own adaptive branches under the unchanged tie policy.

- **Base R2 (labels 1/2): refused at the offline tokenizer gate** — ' 1' is not a single-token continuation of the prompt. Llama-3.1 tokenizer encodes the continuation " 1" after "Answer:" as two tokens: 220 (" ") and 16 ("1"); " 2" as 220 and 17. " A" is the single token 362. Under the predeclared rule no replacement label or whitespace convention was tried, and no model forward pass was made for it.

| run | display order | initial ties | valid initials | chose token A | chose first-listed line | opt_1/opt_2 | same semantic choice as reference |
|---|---|---|---|---|---|---|---|
| Base | AB | 3 | 237 | 232 (97.9%) | 232 (97.9%) | 125/112 | 237/237 (100.0%) |
| Base R1 (B line first) | BA | 6 | 234 | 215 (91.9%) | 19 (8.1%) | 131/103 | 217/231 (93.9%) |
| Base R3 (minimal scaffold) | AB | 32 | 208 | 166 (79.8%) | 166 (79.8%) | 108/100 | 165/206 (80.1%) |

In the reference format Base chose token A on 232 of 237 valid initials, where A is also the first line. With the B line displayed first (R1) it still chose token A on 215 of 234, now the second-listed line. In these runs the initial preference therefore tracks the response token “A” rather than the first displayed line. Because `label_to_option` is counterbalanced, the initial semantic choice in Base mostly follows whichever option is labelled A, so Base's counterarguments mostly argue for the option labelled B. Base movement is therefore measured against a label preference, not a content-based initial stance.

### Contrasts on each run's full valid sample

| run | contrast | estimate [95% CI] | interval | p (boot) | decisions | initials | blocks | movement rows | flip rows |
|---|---|---|---|---|---|---|---|---|---|
| Base | NS − NP | +0.123 [+0.101, +0.147] | above 0 | <0.001 | 60 | 237 | 711 | 2133 | 0 |
| Base | RS − RP | -0.013 [-0.026, 0.000] | includes 0 | 0.059 | 60 | 237 | 711 | 2133 | 0 |
| Base | RS − NS | +0.137 [+0.105, +0.170] | above 0 | <0.001 | 60 | 237 | 711 | 2844 | 0 |
| Base | RP − NP | +0.273 [+0.238, +0.308] | above 0 | <0.001 | 60 | 237 | 711 | 1422 | 0 |
| Base | (RS − RP) − (NS − NP) | -0.136 [-0.159, -0.114] | below 0 | <0.001 | 60 | 237 | 711 | 4266 | 0 |
| Base R1 (B line first) | NS − NP | +0.053 [+0.031, +0.075] | above 0 | <0.001 | 60 | 234 | 702 | 2106 | 0 |
| Base R1 (B line first) | RS − RP | -0.010 [-0.022, +0.003] | includes 0 | 0.135 | 60 | 234 | 702 | 2106 | 0 |
| Base R1 (B line first) | RS − NS | +0.025 [-0.005, +0.057] | includes 0 | 0.098 | 60 | 234 | 702 | 2808 | 0 |
| Base R1 (B line first) | RP − NP | +0.088 [+0.056, +0.120] | above 0 | <0.001 | 60 | 234 | 702 | 1404 | 0 |
| Base R1 (B line first) | (RS − RP) − (NS − NP) | -0.063 [-0.086, -0.039] | below 0 | <0.001 | 60 | 234 | 702 | 4212 | 0 |
| Base R3 (minimal scaffold) | NS − NP | +0.332 [+0.296, +0.369] | above 0 | <0.001 | 60 | 208 | 624 | 1872 | 0 |
| Base R3 (minimal scaffold) | RS − RP | -0.221 [-0.251, -0.193] | below 0 | <0.001 | 60 | 208 | 624 | 1872 | 0 |
| Base R3 (minimal scaffold) | RS − NS | -0.018 [-0.073, +0.038] | includes 0 | 0.514 | 60 | 208 | 624 | 2496 | 0 |
| Base R3 (minimal scaffold) | RP − NP | +0.535 [+0.478, +0.591] | above 0 | <0.001 | 60 | 208 | 624 | 1248 | 0 |
| Base R3 (minimal scaffold) | (RS − RP) − (NS − NP) | -0.554 [-0.589, -0.517] | below 0 | <0.001 | 60 | 208 | 624 | 3744 | 0 |

### Common semantic-choice subset (149 initials)

Initials valid in the reference and every completed variant with the identical initial semantic choice, so every run sees the same stimuli; variant − reference differences isolate the prompt format.

| run | contrast | estimate [95% CI] | interval | p (boot) | decisions | initials | blocks | movement rows | flip rows |
|---|---|---|---|---|---|---|---|---|---|
| Base | NS − NP | +0.123 [+0.097, +0.151] | above 0 | <0.001 | 57 | 149 | 447 | 1341 | 0 |
| Base | RS − RP | -0.007 [-0.025, +0.011] | includes 0 | 0.434 | 57 | 149 | 447 | 1341 | 0 |
| Base | RS − NS | +0.132 [+0.083, +0.180] | above 0 | <0.001 | 57 | 149 | 447 | 1788 | 0 |
| Base | RP − NP | +0.263 [+0.210, +0.312] | above 0 | <0.001 | 57 | 149 | 447 | 894 | 0 |
| Base | (RS − RP) − (NS − NP) | -0.130 [-0.157, -0.104] | below 0 | <0.001 | 57 | 149 | 447 | 2682 | 0 |
| Base R1 (B line first) | NS − NP | +0.042 [+0.012, +0.074] | above 0 | 0.006 | 57 | 149 | 447 | 1341 | 0 |
| Base R1 (B line first) | RS − RP | -0.002 [-0.021, +0.015] | includes 0 | 0.775 | 57 | 149 | 447 | 1341 | 0 |
| Base R1 (B line first) | RS − NS | +0.030 [-0.012, +0.074] | includes 0 | 0.162 | 57 | 149 | 447 | 1788 | 0 |
| Base R1 (B line first) | RP − NP | +0.075 [+0.026, +0.123] | above 0 | 0.003 | 57 | 149 | 447 | 894 | 0 |
| Base R1 (B line first) | (RS − RP) − (NS − NP) | -0.045 [-0.075, -0.014] | below 0 | 0.004 | 57 | 149 | 447 | 2682 | 0 |
| Base R3 (minimal scaffold) | NS − NP | +0.363 [+0.320, +0.406] | above 0 | <0.001 | 57 | 149 | 447 | 1341 | 0 |
| Base R3 (minimal scaffold) | RS − RP | -0.211 [-0.247, -0.176] | below 0 | <0.001 | 57 | 149 | 447 | 1341 | 0 |
| Base R3 (minimal scaffold) | RS − NS | +0.005 [-0.049, +0.058] | includes 0 | 0.868 | 57 | 149 | 447 | 1788 | 0 |
| Base R3 (minimal scaffold) | RP − NP | +0.578 [+0.525, +0.631] | above 0 | <0.001 | 57 | 149 | 447 | 894 | 0 |
| Base R3 (minimal scaffold) | (RS − RP) − (NS − NP) | -0.574 [-0.613, -0.534] | below 0 | <0.001 | 57 | 149 | 447 | 2682 | 0 |

| variant | contrast | variant − reference [95% CI] | interval |
|---|---|---|---|
| Base R1 (B line first) | NS − NP | -0.081 [-0.101, -0.062] | below 0 |
| Base R1 (B line first) | RS − RP | +0.005 [-0.012, +0.021] | includes 0 |
| Base R1 (B line first) | RS − NS | -0.102 [-0.131, -0.073] | below 0 |
| Base R1 (B line first) | RP − NP | -0.188 [-0.222, -0.151] | below 0 |
| Base R1 (B line first) | (RS − RP) − (NS − NP) | +0.086 [+0.061, +0.112] | above 0 |
| Base R3 (minimal scaffold) | NS − NP | +0.239 [+0.193, +0.285] | above 0 |
| Base R3 (minimal scaffold) | RS − RP | -0.204 [-0.244, -0.166] | below 0 |
| Base R3 (minimal scaffold) | RS − NS | -0.128 [-0.209, -0.045] | below 0 |
| Base R3 (minimal scaffold) | RP − NP | +0.315 [+0.236, +0.397] | above 0 |
| Base R3 (minimal scaffold) | (RS − RP) − (NS − NP) | -0.443 [-0.485, -0.400] | below 0 |

Order differences, openings, marker families and individual markers per variant are in `tables/robustness_contrasts`; figures 13–14 summarise the diagnostics. These comparisons are diagnostic; no format is preferred because of the effect it shows.

## 5. Exploratory marker and opening analyses

Unadjusted except the 48-comparison Holm family for individual markers. No isolated uncorrected comparison is promoted.

### Openings

| model | opening | contrast | estimate [95% CI] | interval |
|---|---|---|---|---|
| Base | directive_reconsideration | NS − NP | +0.196 [+0.170, +0.223] | above 0 |
| Base | directive_reconsideration | RS − RP | +0.010 [-0.006, +0.025] | includes 0 |
| Base | directive_reconsideration | (RS − RP) − (NS − NP) | -0.187 [-0.214, -0.161] | below 0 |
| Base | stance_alternative | NS − NP | +0.018 [-0.009, +0.046] | includes 0 |
| Base | stance_alternative | RS − RP | -0.031 [-0.048, -0.014] | below 0 |
| Base | stance_alternative | (RS − RP) − (NS − NP) | -0.049 [-0.078, -0.020] | below 0 |
| Base | stance_disagreement | NS − NP | +0.155 [+0.129, +0.183] | above 0 |
| Base | stance_disagreement | RS − RP | -0.016 [-0.033, 0.000] | below 0 |
| Base | stance_disagreement | (RS − RP) − (NS − NP) | -0.172 [-0.197, -0.148] | below 0 |
| Instruct | directive_reconsideration | NS − NP | -0.791 [-0.872, -0.712] | below 0 |
| Instruct | directive_reconsideration | RS − RP | +0.091 [+0.056, +0.127] | above 0 |
| Instruct | directive_reconsideration | (RS − RP) − (NS − NP) | +0.882 [+0.802, +0.964] | above 0 |
| Instruct | stance_alternative | NS − NP | -0.831 [-0.906, -0.756] | below 0 |
| Instruct | stance_alternative | RS − RP | -0.006 [-0.046, +0.036] | includes 0 |
| Instruct | stance_alternative | (RS − RP) − (NS − NP) | +0.825 [+0.744, +0.907] | above 0 |
| Instruct | stance_disagreement | NS − NP | -0.641 [-0.728, -0.557] | below 0 |
| Instruct | stance_disagreement | RS − RP | -0.145 [-0.185, -0.102] | below 0 |
| Instruct | stance_disagreement | (RS − RP) − (NS − NP) | +0.497 [+0.405, +0.592] | above 0 |

Pairwise opening differences: `tables/opening_heterogeneity`.

### Marker families (unweighted mean of six markers)

| model | family | contrast | estimate [95% CI] | interval |
|---|---|---|---|---|
| Base | conclusion_result | NS − NP | +0.113 [+0.092, +0.133] | above 0 |
| Base | conclusion_result | RS − RP | -0.065 [-0.078, -0.051] | below 0 |
| Base | inference_basis | NS − NP | +0.128 [+0.107, +0.149] | above 0 |
| Base | inference_basis | RS − RP | +0.036 [+0.019, +0.053] | above 0 |
| Base | conclusion_result_minus_inference_basis | NS − NP | -0.015 [-0.030, 0.000] | includes 0 |
| Base | conclusion_result_minus_inference_basis | RS − RP | -0.101 [-0.116, -0.087] | below 0 |
| Instruct | conclusion_result | NS − NP | -0.755 [-0.805, -0.704] | below 0 |
| Instruct | conclusion_result | RS − RP | -0.012 [-0.052, +0.032] | includes 0 |
| Instruct | inference_basis | NS − NP | -0.739 [-0.784, -0.695] | below 0 |
| Instruct | inference_basis | RS − RP | -0.016 [-0.060, +0.030] | includes 0 |
| Instruct | conclusion_result_minus_inference_basis | NS − NP | -0.016 [-0.060, +0.031] | includes 0 |
| Instruct | conclusion_result_minus_inference_basis | RS − RP | +0.004 [-0.027, +0.037] | includes 0 |

### Individual markers (movement; Holm over 48)

With 9999 replicates the smallest attainable bootstrap p is 2/10000 = 0.0002, so the smallest attainable 48-way Holm p is 0.0096.

| marker | Base NS − NP | Base RS − RP | Instruct NS − NP | Instruct RS − RP |
|---|---|---|---|---|
| m01_therefore | +0.19 (Holm 0.010) | +0.01 (Holm 1.000) | -0.71 (Holm 0.010) | +0.13 (Holm 0.104) |
| m02_consequently | +0.20 (Holm 0.010) | -0.01 (Holm 1.000) | -0.59 (Holm 0.010) | -0.01 (Holm 1.000) |
| m03_thus | +0.19 (Holm 0.010) | -0.01 (Holm 1.000) | -0.58 (Holm 0.010) | +0.00 (Holm 1.000) |
| m04_accordingly | +0.11 (Holm 0.010) | -0.07 (Holm 0.010) | -0.42 (Holm 0.010) | -0.11 (Holm 0.036) |
| m05_for_this_reason | +0.06 (Holm 0.045) | -0.06 (Holm 0.012) | -1.11 (Holm 0.010) | +0.02 (Holm 1.000) |
| m06_that_is_why | -0.08 (Holm 0.010) | -0.26 (Holm 0.010) | -1.13 (Holm 0.010) | -0.11 (Holm 0.010) |
| m07_this_implies_that | -0.17 (Holm 0.010) | +0.02 (Holm 1.000) | -2.08 (Holm 0.010) | -0.30 (Holm 0.010) |
| m08_it_follows_that | +0.22 (Holm 0.010) | +0.07 (Holm 0.012) | -0.45 (Holm 0.010) | -0.04 (Holm 1.000) |
| m09_on_that_basis | +0.11 (Holm 0.010) | -0.03 (Holm 1.000) | -0.58 (Holm 0.010) | -0.06 (Holm 1.000) |
| m10_based_on_this | +0.15 (Holm 0.010) | +0.04 (Holm 0.315) | -0.24 (Holm 0.010) | +0.08 (Holm 0.530) |
| m11_given_this | +0.24 (Holm 0.010) | +0.04 (Holm 0.036) | -0.46 (Holm 0.010) | +0.11 (Holm 0.051) |
| m12_in_view_of_this | +0.21 (Holm 0.010) | +0.07 (Holm 0.010) | -0.63 (Holm 0.010) | +0.12 (Holm 0.530) |

Markers whose sign differs from their family average in at least one model × reason context: m01_therefore, m03_thus, m05_for_this_reason, m06_that_is_why, m07_this_implies_that, m09_on_that_basis, m10_based_on_this, m11_given_this, m12_in_view_of_this. They are listed in `review/revision_candidates.yaml` for human or linguistic review; nothing is removed. See `review/marker_opening_review.md` for the non-human assistant review.

## 6. Human checks that remain outstanding

- All formal human ratings of the v3 draft: 4,320 stimulus, 2,880 rendered-pair and 120 scenario units (substantive support, perceived reasoning style, no-reason integrity, inference function, proposition preservation, naturalness, confidence, pressure, politeness, authority, credibility, scenario validity).
- Inter-annotator reliability on the proposed v3 sample (not approved, not begun).
- The open semantic flags (MC4/MC7 for “This implies that …”, MC8 for anaphoric NS markers) and the opening-specific checks.
- Mentor sign-off on the analysis choices listed in the spec.

## 7. Implications for revising or retaining the dataset

- The no-reason style contrast NS − NP is above 0 in Base (+0.123 [+0.101, +0.147]) and below 0 in Instruct (-0.754 [-0.822, -0.688]); its direction differs between the two checkpoints; in Base it changes with the prompt format on identical stimuli in 2 of 2 completed variants (section 4). Behaviour alone does not show whether this reflects the markers, the openings or the formats.
- Behavioural differences between markers or openings are not, by themselves, a reason to delete or rewrite them. The review lists items whose wording should be checked by people (anaphoric NS antecedents, the implication framing, directive openings); only those judgements can motivate a revision, which would create a new dataset version.
- Base chose token “A” on 97.9% of valid initials in the reference format, so its initial choice mostly follows the label, not the content. Before Base results are used substantively, the response-label design needs a decision. One option is to analyse Base only on initials where its semantic choice is stable across formats. This is a prompt-format issue, not evidence about the stimuli.
- Instruct flips on every defined row, so flip carries no factor information there; movement remains the informative outcome.
- Recommendation: retain the v3 draft unchanged for now. Run the human checks. Decide on revisions from those checks, not from these effect sizes.

## 8. Why mechanistic analysis should not start yet

- The plan's gate requires that the manipulation check pass; it has not been run.
- The gate also requires a style contrast that is stable across option orders. Order stability is reported above, but the Base initial stance is label-driven, and 9 variant − reference contrast differences on identical stimuli exclude zero, so a stable Base contrast would not yet identify a style effect.
- The primary style contrast has opposite signs in the two checkpoints; opening and marker heterogeneity is reported in section 5; it is not yet clear which contrast a mechanistic study should localise.
- The dataset is a draft and may change after human review.

## 9. Provenance and files

- Source commit `cc3baebc4fb4d8c3755e5f316395dd6ee9a2e266`; working-tree status and every input/output hash are in `analysis_manifest.json` (its own hash is in `analysis_manifest.sha256`).
- Software: Python 3.11.16, NumPy 2.2.6, matplotlib 3.10.9, PyYAML 6.0.3, from `requirements/analysis.txt` in an environment separate from the GPU inference environment.
- Tables: `tables/` (CSV and JSON). Figures: `figures/index.md`. Bootstrap: `bootstrap/`. Robustness: `robustness/`. Review: `review/`.
