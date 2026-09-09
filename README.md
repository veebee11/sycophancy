# Reason or the Language of Reason?

What actually moves an LLM's position when a user pushes back — the substantive
reason, or the language that makes something sound reasoned?

Implementation of `Research_Plan_v6.md`. Vidhi Bhutani, University of Tokyo.

## The design in one paragraph

Each scenario is a normatively underdetermined public-policy trade-off with two
**immutable semantic options**, `opt_1` and `opt_2`. The model is asked to choose;
its A/B logits give an initial argmax. The transcript is then forked into four
independent branches, each carrying a counterargument that supports the option
*opposite* the initial choice, in one of four conditions:

|                    | style explicit | style plain |
|--------------------|----------------|-------------|
| **reason present** | `RS`           | `RP`        |
| **reason absent**  | `NS`           | `NP`        |

Primary outcome, where `m = logit(counter-supported option) − logit(initially
selected option)`:

```
movement_toward_counter = m_after − m_before
```

Valid contrasts: `NS−NP` (style, no reason), `RS−RP` (style, with reason),
`RS−NS` (content, explicit), `RP−NP` (content, plain), and the interaction
`(RS−RP) − (NS−NP)`.

**A/B are display labels only.** They are assigned by the prompt renderer and
counterbalanced across both orders. They are never the semantic identity of an
option, and support direction is never inferred from them.

**Interpretation ceiling.** An `NS > NP` result licenses the claim that explicit
inferential framing affects the model without additional *stated* propositional
support. It does not by itself establish irrationality, because markers such as
"therefore" may carry pragmatic information about speaker commitment. Human
ratings of perceived speaker commitment and perceived unstated support are
collected to address that alternative.

## Implementation stage status

*(Distinct from the Analysis stages in the research plan — Analysis Stage 2 is the logit lens.)*

| Stage | Scope | Status |
|---|---|---|
| 0 | Repository audit | done |
| 1 | Skeleton, frozen config, hashing, decision log | done |
| 2a | Pinned sentence segmenter, config v2 | done |
| 2b | Scenario schema + machine-valid fixture (1 decision x 2 variants x 8 = **16 texts**) | done |
| 2c | Validator, findings, invalid fixtures | done |

| 3 | Prompt renderer | not started |
| 4 | Backend-independent logit scoring | not started |
| 5 | One-model smoke test | not started |
| 6 | Pilot behavioural pipeline | not started |

No model weights are downloaded and no GPU code exists before Stage 5.

## Setup

Python is pinned to 3.11 (`requires-python = ">=3.11,<3.12"`), managed by `uv`.

```bash
uv sync --group dev
```

Dependencies are added at the stage that first needs them, so the lockfile stays
justified. Stage 1 installs `pydantic`, `pyyaml` and `pytest` only.

| Deferred package | First needed for |
|---|---|
| `pandas`, `numpy` | backend-independent scoring and run tables |
| `torch`, `transformers` | model-adapter smoke test |
| `scikit-learn` | controlled linear probes |
| `scipy` / `statsmodels` | factorial analysis (package not yet chosen) |

## Tests

```bash
uv run pytest
```

## Configuration is immutable

A config must never be edited in place once it has produced an artefact; a
change creates a new version. `configs/experiment_v1.yaml` is frozen as
committed; `configs/experiment_v2.yaml` supersedes it from Implementation
Stage 2a onward. `null` fields are
*unresolved*, not empty — each is pinned by a later config version at the stage
named in its comment. Every generated artefact records the config hash, the
prompt hash and the model revision.

Research decisions and deviations from the plan are logged in
[docs/decisions.md](docs/decisions.md).
