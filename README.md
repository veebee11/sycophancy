# Reason or the Language of Reason?

What actually moves an LLM's position when a user pushes back — the substantive
reason, or the language that makes something sound reasoned?

Vidhi Bhutani, University of Tokyo. The research plan is
[`Research_Plan_v6.md`](Research_Plan_v6.md); design decisions are in
[`docs/design_notes.md`](docs/design_notes.md).

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

## Layout

```
configs/experiment.yaml     the one editable config; frozen copies go in configs/frozen/
data/fixtures/corpus.jsonl  a small synthetic corpus used by the tests
src/reasonstyle/
  corpus/                   schemas, storage, segmentation, validation, annotation, review
  prompting/                transcript rendering
scripts/                    each takes a required --config
docs/design_notes.md        decisions that affect the experiment
```

## Setup

Python 3.11, managed by `uv`.

```bash
uv sync --group dev
```

Dependencies are added at the point they are first needed. Currently:
`pydantic`, `pyyaml`, `pysbd`, and `pytest`. Still to come — `pandas`/`numpy` for
run tables, `torch`/`transformers` for the model adapter, `scikit-learn` for
probes, and a statistics package for the factorial analysis.

## Running things

```bash
uv run pytest
```

```bash
uv run python scripts/export_for_review.py --config configs/experiment.yaml
```

Writes a read-only Markdown view of the corpus to `review/` — an index, one file
per decision showing all four conditions side by side, a combined searchable
file, and blinded packets for the reliability annotators. Add `--check` to verify
it still matches the corpus. Judgements are recorded under `data/`, never in the
generated Markdown.

## What is and is not established

Machine validation checks form: presence, absence, counts, structural
consistency, pattern matches. Substantive support, support direction, no-reason
integrity, proposition preservation, naturalness and pragmatic commitment are
human judgements, and every generated review lists them as outstanding. A clean
validation run is not an approved corpus.
