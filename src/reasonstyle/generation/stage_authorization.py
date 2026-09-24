"""A separate authorisation for a paid stage the frozen configuration's own
``corpus.generation_authorization`` does not cover.

The frozen configuration's embedded authorisation is scoped narrowly and
deliberately (``scenario_stage_initial_only``, at most 96 scenario calls) and
excludes redrafts, groups, repairs, evaluation and mechanistic analysis by
name. Recording a later stage's authorisation *inside* that block would edit
``configs/frozen/v2_full.yaml``, which would change its content hash — and
every one of the 94 existing scenario approvals is bound to that exact hash
(``approvals.ScenarioApproval.config_content_hash``). An edited configuration
would make all 94 stale at a stroke, for the sake of authorising two calls.

So a later stage's authorisation is instead its own small, separately tracked
record, named by ``kind`` (``data/full/authorizations/<kind>_v1.yaml``),
bound explicitly to the configuration it applies to — by version *and*
content hash, so it goes stale the instant the configuration it was granted
under changes — and to the exact approvals-file state and exact call/text
identities of every scenario it covers. It never edits the configuration, and
loading or checking it never computes or restamps a configuration hash.

**This does widen what may be sent.** The frozen configuration's own
authorisation excludes ``scenario_redraft`` by name; a validated stage
authorisation is exactly what lets the send path accept that kind at all
(``CallStore.allowed_kinds``, set by the caller once every check below has
passed). It is not merely a narrowing of the configuration's authorisation —
it grants a permission the configuration itself withholds, deliberately, for
one exact, named, reviewed set of calls.

**What one record does not do.** It authorises one stage, for one
configuration, for an exact, named set of targets, each bound to the call and
text the approvals file currently holds for it. It does not authorise a
second call for a target it already covers — including a second *attempt*
after a transport failure, which still consumed the one call this record
grants that target (:func:`reasonstyle.generation.redraft.transport_blocked_targets`).
And a target the approvals file no longer marks the covered decision for is
refused, not silently skipped.

**Fails closed.** Every check below defends its own field's shape before
reading it: a missing, null, wrongly typed or malformed value is a refusal,
the same as an explicit mismatch, and never an exception. A record broken in
a way this module has not anticipated is refused for exactly that shape of
brokenness, not trusted by default.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from ..hashing import file_sha256

__all__ = [
    "StageAuthorizationError",
    "load_stage_authorization",
    "stage_authorization_problems",
]

_STATUS_AUTHORIZED = "authorized"


class StageAuthorizationError(ValueError):
    """A stage-authorisation record could not be read, or is malformed."""


def load_stage_authorization(path: str | Path) -> dict[str, Any]:
    """Read a stage-authorisation record. Refuses a missing or unreadable file.

    Two keys are added that are not part of the record itself, for
    convenience and provenance: ``_path`` (where it was read from) and
    ``_file_sha256`` (its exact bytes at the moment of reading) — the same
    hash that gets recorded on every call this authorisation covers.
    """
    path = Path(path)
    if not path.is_file():
        raise StageAuthorizationError(f"{path} does not exist")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise StageAuthorizationError(f"{path} is not readable YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise StageAuthorizationError(f"{path} must be a mapping")
    record = dict(raw)
    record["_path"] = path
    record["_file_sha256"] = file_sha256(path)
    return record


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_plain_int(value: Any) -> bool:
    """``True`` only for an actual ``int`` — ``bool`` is a subclass of ``int``
    in Python, and ``True``/``False`` are never a legitimate call count."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_iso_timestamp(value: Any) -> bool:
    """A non-empty ISO-8601 date or datetime string, and nothing else."""
    if not _is_nonempty_str(value):
        return False
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        pass
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False


def stage_authorization_problems(record: dict[str, Any], *, cfg, kind: str,
                                 record_version: str, targets: list[Any],
                                 approvals_path: str | Path) -> list[str]:
    """Every reason this record does not authorise this call, right now.

    Checked before any credential is read or backend is built — the caller is
    expected to refuse on a non-empty list before doing either. Every check is
    independent and all that apply are reported together, so a record wrong
    in several ways is not fixed one refusal at a time. Nothing here raises:
    a missing, null or wrongly shaped value is reported as a problem, exactly
    like an explicit mismatch, so a caller can rely on this function to
    finish and return a list, never to throw.

    ``targets`` is the *live* redraft-target list — ``RedraftTarget`` (or
    anything with ``scenario_id``, ``original_call_id`` and
    ``original_text_sha256``) — computed fresh from the current approvals
    file, never taken from the record: the record is checked against what is
    actually true now, not asked to vouch for itself.
    """
    problems: list[str] = []

    if record.get("record_version") != record_version:
        problems.append(
            f"the stage authorisation declares record_version "
            f"{record.get('record_version')!r}, not the expected {record_version!r}")

    status = record.get("status")
    if status != _STATUS_AUTHORIZED:
        problems.append(f"the stage authorisation is {status!r}, not "
                        f"{_STATUS_AUTHORIZED!r}")

    if not _is_nonempty_str(record.get("authorized_by")):
        problems.append(
            "the stage authorisation names no reviewer: authorized_by must be a non-empty "
            "string, naming who authorised it")
    if not _is_iso_timestamp(record.get("authorized_at")):
        problems.append(
            "the stage authorisation carries no valid authorisation date: authorized_at "
            "must be a non-empty ISO-8601 date or timestamp")

    cfg_block = record.get("config")
    if not isinstance(cfg_block, dict):
        problems.append(
            f"the stage authorisation's config field must be a mapping with "
            f"config_version and config_content_hash; it is {type(cfg_block).__name__}")
        cfg_block = {}
    else:
        if cfg_block.get("config_version") != cfg.config_version:
            problems.append(
                f"the stage authorisation is bound to configuration "
                f"{cfg_block.get('config_version')!r}, not {cfg.config_version!r}")
        if cfg_block.get("config_content_hash") != cfg.content_hash:
            problems.append(
                "the stage authorisation is bound to a different configuration content "
                f"hash ({str(cfg_block.get('config_content_hash'))[:12]}, not "
                f"{cfg.content_hash[:12]}) — it was granted under a configuration that no "
                "longer matches this one")

    if record.get("kind") != kind:
        problems.append(f"the stage authorisation covers kind {record.get('kind')!r}, "
                        f"not {kind!r}")

    approvals_path = Path(approvals_path)
    recorded_approvals_path = record.get("approvals_file")
    # Resolved, not just compared as written: a caller may pass an absolute
    # path where the record names a relative one (or vice versa) and still
    # mean the same file — this check is about identity, not spelling.
    same_approvals_file = (_is_nonempty_str(recorded_approvals_path)
                           and Path(recorded_approvals_path).resolve() == approvals_path.resolve())
    if not same_approvals_file:
        problems.append(
            f"the stage authorisation is bound to approvals file "
            f"{recorded_approvals_path!r}, not {str(approvals_path)!r}")
    elif not approvals_path.is_file():
        problems.append(f"the approvals file {approvals_path} does not exist")
    else:
        actual = file_sha256(approvals_path)
        if record.get("approvals_file_sha256") != actual:
            problems.append(
                "the stage authorisation's approvals-file SHA-256 does not match the "
                "current file — it has changed since the authorisation was granted, so "
                "the reviewed-target set it names can no longer be trusted")

    # max_paid_calls: required, a genuine positive int (never a bool), and
    # exactly the number of authorised targets — no unused headroom a later,
    # unreviewed target could be slipped into.
    ceiling = record.get("max_paid_calls")
    if not _is_plain_int(ceiling) or ceiling <= 0:
        problems.append(
            f"max_paid_calls must be a required positive integer; it is {ceiling!r}")
    elif ceiling != len(targets):
        problems.append(
            f"max_paid_calls is {ceiling}; it must exactly equal the {len(targets)} "
            f"authorised target(s), with no unused authorisation headroom")

    per_target = record.get("calls_per_target")
    if not _is_plain_int(per_target) or per_target != 1:
        problems.append(
            f"calls_per_target must be a required integer equal to exactly 1; it is "
            f"{per_target!r}")

    declared = record.get("targets")
    if not isinstance(declared, dict):
        problems.append(
            f"the stage authorisation's targets field must be a mapping of scenario id to "
            f"target entry; it is {type(declared).__name__}")
        declared = {}

    declared_ids = set(declared)
    live_ids = {t.scenario_id for t in targets}
    missing = sorted(live_ids - declared_ids)
    if missing:
        problems.append(
            f"the approvals file marks {missing} for redraft, but the stage authorisation "
            f"does not name {'them' if len(missing) > 1 else 'it'}")
    extra = sorted(declared_ids - live_ids)
    if extra:
        problems.append(
            f"the stage authorisation names {extra}, which the approvals file does not "
            f"currently mark redraft")

    for target in targets:
        entry = declared.get(target.scenario_id)
        if entry is None:
            continue                       # already reported above, as missing
        if not isinstance(entry, dict):
            problems.append(
                f"{target.scenario_id}: its entry in the stage authorisation's targets "
                f"must be a mapping with call_id and scenario_text_sha256; it is "
                f"{type(entry).__name__}")
            continue
        if entry.get("call_id") != target.original_call_id:
            problems.append(
                f"{target.scenario_id}: the stage authorisation names the rejected call as "
                f"{str(entry.get('call_id'))[:12]}, but the approvals file's rejected call is "
                f"now {target.original_call_id[:12]} — a stale binding")
        if entry.get("scenario_text_sha256") != target.original_text_sha256:
            problems.append(
                f"{target.scenario_id}: the stage authorisation's rejected-text SHA-256 does "
                f"not match the text the approvals file currently holds for it — a stale "
                f"binding")

    return problems
