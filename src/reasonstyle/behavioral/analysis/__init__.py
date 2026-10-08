"""Behavioural-result analysis for the completed full-v3 runs.

Evidence is validated before anything is computed (``validate``), the 18 rows
of each non-tied initial prompt are turned into balanced 2x2 units without
duplicating the shared plain controls (``design``), and every interval comes
from a decision-level cluster bootstrap (``stats``). Only the standard library
is used, so the analysis never touches the GPU inference environment.
"""

from .validate import AnalysisError, AnalysisSpec, ValidatedEvidence, load_analysis_spec, validate_evidence

__all__ = ["AnalysisError", "AnalysisSpec", "ValidatedEvidence", "load_analysis_spec",
           "validate_evidence"]
