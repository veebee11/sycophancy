#!/usr/bin/env python3
"""Validate the full-v3 behavioural evidence, then write the detailed analysis.

    python scripts/analyze_behavioral_v3.py configs/analysis/behavioral_v3_detailed_v1.yaml

Refuses on a different Git commit, any evidence mismatch or an existing output
directory (``--overwrite`` replaces only a directory this pipeline wrote).
Makes no model call; runs in the isolated analysis environment
(``requirements/analysis.txt``), never the GPU inference environment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import socket
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from reasonstyle.behavioral.analysis import figures as figmod
from reasonstyle.behavioral.analysis.report import Report, readme
from reasonstyle.behavioral.analysis.review import (
    build_review,
    candidates_yaml,
    rendered_examples,
    review_markdown,
)
from reasonstyle.behavioral.analysis.tables import TABLE_NAMES, Tables
from reasonstyle.behavioral.analysis.validate import (
    AnalysisError,
    load_analysis_spec,
    validate_evidence,
)
from reasonstyle.behavioral.analysis.write import (
    OutputDir,
    csv_text,
    gzip_bytes,
    json_text,
    sorted_rows,
    verify_directory,
)
from reasonstyle.hashing import file_sha256

CODE_FILES = ["scripts/analyze_behavioral_v3.py", "requirements/analysis.txt"] + [
    f"src/reasonstyle/behavioral/analysis/{n}.py" for n in (
        "__init__", "validate", "design", "estimate", "tables", "figures", "review",
        "report", "write")] + ["src/reasonstyle/behavioral/robustness.py",
                               "src/reasonstyle/behavioral/evidence.py",
                               "src/reasonstyle/behavioral/score.py",
                               "src/reasonstyle/behavioral/plan.py",
                               "src/reasonstyle/behavioral/runtime.py"]
ROBUSTNESS_TABLES = ("robustness_initial_diagnostics", "robustness_contrasts",
                     "robustness_common_semantic_choice_subset")


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True).stdout


def run(spec_path: str, overwrite: bool, out: str | None) -> Path:
    spec = load_analysis_spec(spec_path)
    root = spec.root
    raw = spec.raw
    head = git(root, "rev-parse", "HEAD").strip()
    if head != raw["source_git_commit"]:
        raise AnalysisError(f"git HEAD {head} != required {raw['source_git_commit']}")
    status = git(root, "status", "--porcelain", "--untracked-files=all").splitlines()
    evidence = validate_evidence(spec)
    ref_cfg = raw["inputs"]["robustness_reference_check"]
    ref_path = root / ref_cfg["path"]
    if file_sha256(ref_path) != ref_cfg["sha256"]:
        raise AnalysisError("reference-check record does not match its pinned hash")
    reference_check = json.loads(ref_path.read_text())
    if reference_check["status"] != "complete" or reference_check["bitwise_mismatches"]:
        raise AnalysisError("the variant runner did not reproduce the reference initials")

    tables = Tables(evidence)
    tabs = tables.build()
    review = build_review(tables)
    software = {"python": platform.python_version(), "numpy": np.__version__,
                "matplotlib": version("matplotlib"), "pyyaml": yaml.__version__}
    meta = {"software": software, "reference_check": reference_check,
            "common_valid_initials": len(tables.common)}
    destination = Path(out) if out else root / raw["outputs"]["root"]
    od = OutputDir(destination, raw["analysis_version"], overwrite)
    od.write("analysis_spec.yaml", spec.path.read_bytes())
    for name in TABLE_NAMES:
        od.table(name, tabs[name])
    for name in ROBUSTNESS_TABLES:
        od.table(name, tabs[name], folder="robustness")

    boot = tables.boot
    bcfg = raw["inference"]["bootstrap"]
    od.write("bootstrap/settings.json", json_text({
        **bcfg, "numpy_version": np.__version__, "decisions": boot.decisions,
        "domains": boot.domains, "position_labels": boot.position_labels,
        "draw_matrix_shape": list(boot.draws.shape)}))
    draws = [{"replicate": i + 1, **dict(zip(boot.position_labels, row))}
             for i, row in enumerate(boot.decision_ids())]
    od.write("bootstrap/draws.csv.gz",
             gzip_bytes(csv_text(draws, ["replicate"] + boot.position_labels)), rows=len(draws))
    summary = sorted_rows(tables.summary)
    od.write("bootstrap/summary.csv", csv_text(summary), rows=len(summary))
    saved = sorted(tables.saved)
    header = "replicate," + ",".join(f'"{s}"' for s in saved) + "\n"
    matrix = np.column_stack([tables.saved[s].reps for s in saved])
    body = "".join(f"{i + 1}," + ",".join("" if np.isnan(v) else format(float(v), ".12g")
                                          for v in row) + "\n" for i, row in enumerate(matrix))
    od.write("bootstrap/replicate_estimates.csv.gz", gzip_bytes(header + body),
             rows=matrix.shape[0])

    figs = figmod.FigureSet(raw, tabs)
    figs.build()
    for rel, data in sorted(figs.files.items()):
        od.write(rel, data)
    od.write("figures/index.md", figs.index_markdown())

    effects = sorted_rows(review["effects"])
    od.write("review/marker_opening_effects.csv", csv_text(effects), rows=len(effects))
    od.write("review/marker_opening_review.md", review_markdown(review, tables))
    od.write("review/rendered_examples.md", rendered_examples(tables))
    od.write("review/revision_candidates.yaml", candidates_yaml(review))

    inputs = raw["inputs"]
    robustness_status = {
        "completed": {name: {"run_id": r.metadata["run_id"], "status": r.status,
                             "file_sha256": r.file_hashes,
                             "prompt_variant": r.metadata.get("prompt_variant"),
                             "prompt_digest": r.metadata.get("prompt_digest"),
                             "variant_definition_hash": r.metadata.get("variant_definition_hash")}
                      for name, r in evidence.runs.items() if not r.is_reference},
        "refused": {name: {k: v for k, v in ref.items()} for name, ref in evidence.refusals.items()},
        "reference_check": reference_check,
        "archive": inputs["robustness_archive"],
    }
    od.write("robustness/run_status.json", json_text(robustness_status))
    od.write("report.md", Report(evidence, tabs, review, meta).render())
    od.write("README.md", readme(raw))
    od.write("execution_environment.json", json_text({
        "volatile": True, "utc_time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": socket.gethostname(), "platform": platform.platform(),
        "python_executable": sys.executable, "argv": sys.argv, "cwd": str(Path.cwd()),
        "software": software}))

    expected = ({f"tables/{n}.csv" for n in TABLE_NAMES} |
                {f"figures/{n}.{x}" for n, _, _ in figs.index for x in ("svg", "png")} |
                {"bootstrap/settings.json", "bootstrap/draws.csv.gz", "bootstrap/summary.csv",
                 "bootstrap/replicate_estimates.csv.gz", "figures/index.md", "report.md",
                 "review/marker_opening_effects.csv", "review/marker_opening_review.md",
                 "review/rendered_examples.md", "review/revision_candidates.yaml"})
    missing = sorted(expected - set(od.files))
    if missing or len(figs.index) != 14:
        raise AnalysisError(f"incomplete outputs: {missing}, {len(figs.index)} figures")
    manifest = {
        "analysis_version": raw["analysis_version"],
        "source_git_commit": head,
        "working_tree_status": status,
        "code_sha256": {p: file_sha256(root / p) for p in CODE_FILES},
        "analysis_script_sha256": file_sha256(root / "scripts/analyze_behavioral_v3.py"),
        "specification": {"path": str(spec.path.relative_to(root)), "sha256": spec.file_sha256,
                          "content_hash": spec.content_hash},
        "inputs": {k: {"path": k, "sha256": v} for k, v in sorted(evidence.input_hashes.items())},
        "input_paths": {name: str(r.directory.relative_to(root)) for name, r in evidence.runs.items()},
        "model_revisions": {name: r.metadata["revision"] for name, r in evidence.runs.items()},
        "prompt_and_config_hashes": {
            "behavioral_config_sha256": inputs["behavioral_config"]["sha256"],
            "behavioral_config_content_hash": inputs["behavioral_config"]["content_hash"],
            "robustness_variants_file_sha256": evidence.input_hashes["robustness_variants_file"],
            **{f"{n}_prompt_digest": r.metadata.get("prompt_digest")
               for n, r in evidence.runs.items() if not r.is_reference},
            **{f"{n}_variant_config_sha256": inputs["robustness_runs"][n]["variant_config"]["sha256"]
               for n in inputs.get("robustness_runs", {})}},
        "software": software,
        "bootstrap": {**bcfg, "numpy_version": np.__version__},
        "evidence_checks_passed": len(evidence.checks),
        "completeness": {"expected_files_present": True, "tables": len(TABLE_NAMES),
                         "figures": len(figs.index), "missing": missing},
        "robustness": robustness_status,
        "model_run": False, "mechanistic_analysis": False, "human_annotation": False,
    }
    final = od.finish(manifest)
    problems = verify_directory(final)
    if problems:
        raise AnalysisError("output verification failed: " + "; ".join(problems))
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out", help="override the spec's output root (tests, determinism checks)")
    args = parser.parse_args(argv)
    try:
        final = run(args.spec, args.overwrite, args.out)
    except (AnalysisError, FileExistsError, OSError, KeyError, ValueError,
            subprocess.CalledProcessError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
    final = final.resolve()
    for rel in ("report.md", "figures/index.md", "tables", "robustness", "review"):
        print(final / rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
