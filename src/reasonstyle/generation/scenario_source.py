"""Reusing another design's approved scenarios, read-only and fully verified.

The v2 group design changes how a counterargument is built. It does not change
what a scenario is, and the 24 hosted scenarios have already been drafted, read
and approved — four of them through a bounded redraft. Drafting them again
would spend money to obtain different text that a person would have to read and
approve again, for no scientific gain.

So v2 **reads** them. What makes that safe rather than convenient is that the
reuse is declared in the configuration, verified against the source's own
hashes, and recorded on every call:

* the source configuration's content hash, as the approvals were granted under
  — never rewritten to the v2 hash, because an approval binds the configuration
  the curator read the text under;
* each accepted call id, as the approval names it;
* each scenario text's SHA-256, as the approval binds it;
* the topic bank's content hash.

Nothing is copied. The v1 run directory is opened for reading and never
written; no scenario call is duplicated into the v2 run; no scenario is
redrafted. Every v2 group request and result carries **both** provenance
layers: the v1 scenario source it read from, and the v2 group-design
configuration it was drafted under.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, load_config
from ..corpus.topics import load_topic_bank
from ..hashing import content_hash, sha256_of
from .approvals import APPROVED, load_approvals

__all__ = [
    "ScenarioSource",
    "expected_scenario_ids",
    "ScenarioSourceError",
    "load_scenario_source",
    "scenario_source_spec",
]


class ScenarioSourceError(RuntimeError):
    """The declared scenario source cannot be verified. Nothing proceeds."""


@dataclass(frozen=True)
class ScenarioSource:
    """Verified scenarios from another design, plus what to record about them.

    ``approvals`` are the source's own approval records, unaltered. They are
    what every command that needs a gate uses — generation, status, review and
    assembly alike — so no part of the workflow invents a v2 approval, and none
    of them rewrites a v1 one with a v2 hash.
    """

    scenarios: dict[str, dict[str, Any]]
    approvals: dict[str, Any]
    provenance: dict[str, Any]

    @property
    def count(self) -> int:
        return len(self.scenarios)

    @property
    def config_content_hash(self) -> str:
        """The hash the approvals were granted under, not the reusing design's."""
        return self.provenance["config_content_hash"]


def expected_scenario_ids(bank, cfg: ExperimentConfig) -> set[str]:
    """Every scenario id this corpus requires, from the bank and the config.

    Derived, never counted off a directory: a check that asserted "24 are here"
    would pass on a source holding the wrong 24.
    """
    curated = sorted(t.decision_id for t in bank.topics if t.status == "curated")
    wanted = cfg.raw["corpus"]["decisions_pilot"]
    if len(curated) != wanted:
        raise ScenarioSourceError(
            f"the topic bank holds {len(curated)} curated decisions, not {wanted}")
    variants = range(1, cfg.raw["corpus"]["variants_per_decision"] + 1)
    return {f"{decision}_v{variant}" for decision in curated for variant in variants}


def scenario_source_spec(cfg: ExperimentConfig) -> dict[str, Any] | None:
    """The declared source, or ``None`` when this design drafts its own."""
    return cfg.parsed.scenario_source


def load_scenario_source(cfg: ExperimentConfig, *, topics_path: str | Path,
                         root: Path | None = None) -> ScenarioSource:
    """Read and verify the declared scenario source. Writes nothing, anywhere.

    Every check is a refusal, not a warning: a v2 group drafted against a
    scenario whose approval does not verify would be built on text nobody
    approved, which is the one thing the curator gate exists to prevent.
    """
    spec = scenario_source_spec(cfg)
    if not spec:
        raise ScenarioSourceError(
            "this configuration declares no scenario_source, so there is nothing to "
            "reuse; it must draft its own scenarios through the scenario stage")
    if spec.get("mode") != "reuse_approved_v1":
        raise ScenarioSourceError(f"unknown scenario_source mode {spec.get('mode')!r}")
    if spec.get("access") != "read_only":
        raise ScenarioSourceError(
            f"scenario_source.access is {spec.get('access')!r}; a reused source is "
            f"read_only, and a design that wrote to it would be altering another "
            f"design's evidence")

    base = Path(root) if root else Path()
    source_config = base / spec["config"]
    source_run = base / spec["run"]
    source_approvals = base / spec["approvals"]
    for path in (source_config, source_approvals):
        if not path.is_file():
            raise ScenarioSourceError(f"scenario source {path} does not exist")
    if not source_run.is_dir():
        raise ScenarioSourceError(f"scenario source run {source_run} does not exist")

    source_cfg = load_config(source_config)
    bank = load_topic_bank(topics_path)
    bank_hash = content_hash(bank.model_dump(mode="json"))

    # Imported here: the pipeline imports this module's package, and the store
    # is only needed to read a directory that is never written.
    from .pipeline import CallStore
    from .redraft import current_scenarios

    approvals = load_approvals(source_approvals)
    store = CallStore(source_run, source_cfg, topic_bank_content_hash=bank_hash)
    scenarios = current_scenarios(store, approvals)

    # What the source MUST contain, derived from the curated topic bank and the
    # configured variants — never from what this machine happens to hold. A
    # source missing one scenario, or carrying one nobody expected, is refused:
    # a corpus drafted from it would be a different corpus quietly.
    expected = expected_scenario_ids(bank, cfg)
    problems: list[str] = []
    missing = sorted(expected - set(approvals))
    if missing:
        problems.append(f"the source approvals are missing {len(missing)} expected "
                        f"scenario(s): {missing}")
    unexpected = sorted(set(approvals) - expected)
    if unexpected:
        problems.append(f"the source approvals carry {len(unexpected)} scenario(s) this "
                        f"corpus does not expect: {unexpected}")

    verify = set(spec.get("verify") or ())
    verified: dict[str, dict[str, Any]] = {}
    for scenario_id, approval in sorted(approvals.items()):
        record = scenarios.get(scenario_id) or {}
        text = record.get("scenario_text")
        if not text:
            problems.append(f"{scenario_id}: the source run records no usable scenario text")
            continue
        if approval.decision != APPROVED:
            problems.append(f"{scenario_id}: the source approval is "
                            f"{approval.decision!r}, not {APPROVED!r}")
            continue
        if "config_content_hash" in verify and \
                approval.config_content_hash != source_cfg.content_hash:
            problems.append(
                f"{scenario_id}: approved under configuration "
                f"{approval.config_content_hash[:12]}, but {spec['config']} now hashes to "
                f"{source_cfg.content_hash[:12]}")
        if "topic_bank_content_hash" in verify and \
                approval.topic_bank_content_hash != bank_hash:
            problems.append(f"{scenario_id}: approved against a different topic bank")
        if "call_id" in verify and approval.call_id != record.get("call_id"):
            problems.append(
                f"{scenario_id}: the approval names call {approval.call_id[:12]}, the run "
                f"records {str(record.get('call_id'))[:12]}")
        if "scenario_text_sha256" in verify and \
                approval.scenario_text_sha256 != sha256_of(text):
            problems.append(f"{scenario_id}: the scenario text has changed since it "
                            f"was approved")
        verified[scenario_id] = dict(record)

    if problems:
        raise ScenarioSourceError(
            "the declared scenario source does not verify, so no group may be drafted "
            "from it:\n  - " + "\n  - ".join(problems))
    if sorted(verified) != sorted(expected):                # pragma: no cover - guarded above
        raise ScenarioSourceError(
            f"the verified set is not the expected set: verified {len(verified)}, "
            f"expected {len(expected)}")

    provenance = {
        "mode": spec["mode"],
        "access": "read_only",
        "config": spec["config"],
        # The hash the approvals were granted under. NOT rewritten to the hash of
        # the design that is reusing them: that would claim a curator approved
        # text under a configuration they never saw.
        "config_content_hash": source_cfg.content_hash,
        "config_version": source_cfg.config_version,
        # The declared paths, as the configuration writes them: a record
        # that carried this machine's absolute paths would not be portable.
        "run": spec["run"],
        "approvals": spec["approvals"],
        "topic_bank_content_hash": bank_hash,
        "verified": sorted(verify),
        "scenario_count": len(verified),
        "expected_scenario_count": len(expected),
        "scenarios": {scenario_id: {"call_id": record["call_id"],
                                    "scenario_text_sha256": sha256_of(
                                        record["scenario_text"]),
                                    "superseded_call_id": record.get("supersedes_call_id")}
                      for scenario_id, record in sorted(verified.items())},
    }
    return ScenarioSource(scenarios=verified, approvals=dict(approvals),
                          provenance=provenance)
