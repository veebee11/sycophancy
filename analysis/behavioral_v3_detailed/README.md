# behavioral_v3_detailed

Output of `scripts/analyze_behavioral_v3.py configs/analysis/behavioral_v3_detailed_v1.yaml` (analysis `behavioral_v3_detailed_v1`).

- `report.md` — the report: evidence validation, estimand, core contrasts, robustness, exploratory marker/opening analyses, outstanding human checks.
- `analysis_spec.yaml` — byte copy of the frozen specification.
- `analysis_manifest.json` / `analysis_manifest.sha256` — every input and output hash, size and row count; code, config and software provenance.
- `execution_environment.json` — host, time and command (volatile; not part of the determinism comparison).
- `tables/` — every analysis table as CSV and canonical JSON.
- `figures/` — SVG and PNG figures; `figures/index.md`.
- `bootstrap/` — settings, the decision draw matrix, the summary of every statistic, and replicate estimates of the saved statistics.
- `robustness/` — robustness-run status, hashes, refusals and the reference check.
- `review/` — the non-human assistant review of markers and openings.

Re-run on identical inputs and environment to reproduce every file except `execution_environment.json` and the manifest that hashes it.
