"""Reference-source registry: every outcome must be supported by its evidence.

All values below are synthetic test data (example.org URLs), not real licence
information.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest
import yaml

from reasonstyle.corpus.sources import SourceRegistryError, load_registry

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "sources" / "registry.yaml"
SCRIPT = ROOT / "scripts" / "verify_sources.py"
TODAY = date(2026, 9, 11)

CITABLE = {
    "canonical_name": "Synthetic Citable Dataset",
    "citation": None,
    "outcome": "citable",
    "version": "1.0",
    "access_url": "https://example.org/dataset",
    "access_date": "2026-09-01",
    "licence": "CC BY 4.0 (synthetic test value)",
    "licence_url": "https://example.org/licence",
    "usage": "seed_only",
    "usage_permitted": True,
    "attribution_required": True,
    "registration_required": False,
    "checked_by": "test",
    "checked_at": "2026-09-02",
    "exclusion_reason": None,
    "notes": None,
}

#: An exclusion carries only what was actually found — here, nothing but the reason.
EXCLUDED = {
    "canonical_name": "Synthetic Excluded Dataset",
    "outcome": "excluded",
    "usage": "seed_only",
    "usage_permitted": False,
    "checked_by": "test",
    "checked_at": "2026-09-02",
    "exclusion_reason": "official link broken; no alternative distribution found",
}


def write(tmp_path, datasets) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"datasets": datasets}))
    return path


def citable(**changes):
    return {**copy.deepcopy(CITABLE), **changes}


def excluded(**changes):
    return {**copy.deepcopy(EXCLUDED), **changes}


def load(tmp_path, datasets):
    return load_registry(write(tmp_path, datasets), today=TODAY)


# --- the committed registry --------------------------------------------------
#
# Only validity is checked here. Which candidates are citable today is a matter
# of evidence that may change before the pilot freeze, so it is not asserted.


def test_the_committed_registry_is_valid():
    registry = load_registry(REGISTRY, today=TODAY)
    assert registry.datasets


# --- the three outcomes ------------------------------------------------------


def test_a_fully_evidenced_dataset_can_be_citable(tmp_path):
    assert load(tmp_path, {"a": citable()}).citable() == ["a"]


def test_a_citable_dataset_need_not_state_a_version(tmp_path):
    """Some providers state none; it must not be invented."""
    assert load(tmp_path, {"a": citable(version=None)}).citable() == ["a"]


def test_an_exclusion_needs_no_unavailable_information(tmp_path):
    """A broken link or an unclear licence is recorded, not papered over."""
    registry = load(tmp_path, {"a": excluded()})
    assert registry.excluded() == ["a"]
    assert registry.datasets["a"].licence is None
    assert registry.datasets["a"].access_url is None


def test_only_citable_entries_are_offered_for_citation(tmp_path):
    registry = load(tmp_path, {
        "a": citable(),
        "b": excluded(),
        "c": {"canonical_name": "Not Checked", "usage": "seed_only"},
    })
    assert registry.citable() == ["a"]
    assert registry.excluded() == ["b"]
    assert registry.unverified() == ["c"]


# --- outcomes the evidence does not support ----------------------------------


@pytest.mark.parametrize("missing", ["licence", "licence_url", "access_url", "access_date",
                                     "attribution_required", "registration_required",
                                     "checked_by", "checked_at"])
def test_citable_without_complete_evidence_is_rejected(tmp_path, missing):
    with pytest.raises(SourceRegistryError, match="not supported by the recorded evidence"):
        load(tmp_path, {"a": citable(**{missing: None})})


@pytest.mark.parametrize("missing", ["checked_by", "checked_at", "exclusion_reason"])
def test_excluded_without_who_when_and_why_is_rejected(tmp_path, missing):
    with pytest.raises(SourceRegistryError, match="not supported by the recorded evidence"):
        load(tmp_path, {"a": excluded(**{missing: None})})


@pytest.mark.parametrize(("entry", "why"), [
    (lambda: citable(usage_permitted=False), "citable but not permitted"),
    (lambda: excluded(usage_permitted=True), "excluded but permitted"),
    (lambda: {"canonical_name": "X", "usage": "seed_only", "usage_permitted": True},
     "unverified but permitted"),
])
def test_usage_permitted_must_agree_with_the_outcome(tmp_path, entry, why):
    with pytest.raises(SourceRegistryError, match="exactly when the outcome is 'citable'"):
        load(tmp_path, {"a": entry()})


def test_an_exclusion_reason_on_a_citable_entry_is_rejected(tmp_path):
    with pytest.raises(SourceRegistryError, match="only for an excluded dataset"):
        load(tmp_path, {"a": citable(exclusion_reason="changed my mind")})


def test_only_seed_only_use_is_allowed(tmp_path):
    with pytest.raises(SourceRegistryError):
        load(tmp_path, {"a": citable(usage="full_text")})


def test_checking_cannot_precede_access(tmp_path):
    with pytest.raises(SourceRegistryError, match="cannot precede"):
        load(tmp_path, {"a": citable(checked_at="2026-08-01")})


def test_dates_cannot_be_in_the_future(tmp_path):
    with pytest.raises(SourceRegistryError, match="in the future"):
        load(tmp_path, {"a": excluded(checked_at="2026-12-01")})


def test_urls_must_be_web_addresses(tmp_path):
    with pytest.raises(SourceRegistryError, match="http"):
        load(tmp_path, {"a": citable(licence_url="see the paper")})


def test_an_unknown_field_is_rejected(tmp_path):
    with pytest.raises(SourceRegistryError):
        load(tmp_path, {"a": citable(downloaded=True)})


def test_duplicate_canonical_names_are_rejected(tmp_path):
    with pytest.raises(SourceRegistryError, match="duplicates"):
        load(tmp_path, {"a": citable(),
                        "b": excluded(canonical_name="synthetic citable DATASET")})


# --- citation is the gate ---------------------------------------------------


UNVERIFIED = {"canonical_name": "Synthetic Unchecked Dataset", "usage": "seed_only"}


def test_a_citable_source_may_be_cited(tmp_path):
    assert load(tmp_path, {"a": citable()}).citation_problems(["a"]) == []


def test_an_unverified_source_may_not_be_cited(tmp_path):
    problems = load(tmp_path, {"u": UNVERIFIED}).citation_problems(["u"])
    assert problems == ["u: unverified, so it may not be cited"]


def test_an_excluded_source_may_not_be_cited(tmp_path):
    problems = load(tmp_path, {"x": excluded()}).citation_problems(["x"])
    assert problems == ["x: excluded, so it may not be cited"]


def test_an_unknown_source_may_not_be_cited(tmp_path):
    assert load(tmp_path, {"a": citable()}).citation_problems(["nope"]) == [
        "nope: not in the registry"]


def test_an_uncited_unresolved_candidate_blocks_nothing(tmp_path):
    """An optional candidate left unverified must not hold up the sources in use."""
    registry = load(tmp_path, {"a": citable(), "u": UNVERIFIED})
    assert registry.citation_problems(["a"]) == []


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_the_script_accepts_the_committed_registry():
    assert run("--registry", str(REGISTRY)).returncode == 0


def test_the_script_passes_when_every_cited_source_is_citable(tmp_path):
    path = write(tmp_path, {"a": citable(), "u": UNVERIFIED})
    assert run("--registry", str(path), "--cite", "a").returncode == 0


@pytest.mark.parametrize("key", ["u", "x", "missing"])
def test_the_script_fails_when_a_cited_source_is_not_citable(tmp_path, key):
    path = write(tmp_path, {"a": citable(), "u": UNVERIFIED, "x": excluded()})
    result = run("--registry", str(path), "--cite", "a", key)
    assert result.returncode == 1
    assert "may not be cited" in result.stderr


def test_the_script_rejects_an_unsupported_outcome(tmp_path):
    result = run("--registry", str(write(tmp_path, {"a": excluded(exclusion_reason=None)})))
    assert result.returncode == 1
    assert "invalid" in result.stderr
