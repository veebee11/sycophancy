"""Frozen configurations are verified against their external manifest.

A frozen configuration never changes: it is what a run's records point at, so a
later edit would silently redefine what was already generated under it. The
manifest records each file's SHA-256 and its content hash, and the tests below
check the committed file against both — including a tamper check on a copy,
because a location and filename test alone would not catch an edited file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reasonstyle.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "configs" / "frozen"
MANIFEST = json.loads((FROZEN / "MANIFEST.json").read_text(encoding="utf-8"))["configs"]


def test_every_frozen_config_is_in_the_manifest():
    on_disk = sorted(p.name for p in FROZEN.glob("*.yaml"))
    assert on_disk == sorted(MANIFEST)


@pytest.mark.parametrize("name", sorted(MANIFEST))
def test_a_frozen_config_matches_its_recorded_hashes(name):
    path = FROZEN / name
    recorded = MANIFEST[name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == recorded["file_sha256"]
    cfg = load_config(path)
    assert cfg.content_hash == recorded["content_hash"]
    assert cfg.config_version == recorded["config_version"]
    assert cfg.parsed.status == "frozen"


@pytest.mark.parametrize("name", sorted(MANIFEST))
def test_an_edited_copy_fails_verification(tmp_path, name):
    """The check is on contents, not on where the file sits."""
    copy = tmp_path / "frozen" / name
    copy.parent.mkdir(parents=True, exist_ok=True)
    text = (FROZEN / name).read_text(encoding="utf-8")
    copy.write_text(text.replace("temperature: 0.3", "temperature: 0.7", 1), encoding="utf-8")
    assert hashlib.sha256(copy.read_bytes()).hexdigest() != MANIFEST[name]["file_sha256"]
    assert load_config(copy).content_hash != MANIFEST[name]["content_hash"]


def test_a_frozen_config_outside_the_frozen_directory_is_refused(tmp_path):
    copy = tmp_path / "v2_full.yaml"
    copy.write_text((FROZEN / "v2_full.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ConfigError, match="must live in configs/frozen/"):
        load_config(copy)


def test_the_frozen_full_config_records_its_scoped_authorisation():
    cfg = load_config(FROZEN / "v2_full.yaml")
    auth = cfg.raw["corpus"]["generation_authorization"]
    assert auth["status"] == "authorized"
    assert auth["scope"] == "scenario_stage_initial_only"
    assert auth["allowed_kinds"] == ["scenario"]
    assert auth["max_paid_calls"] == 96
    assert cfg.raw["corpus"]["generation_block"]["blocked"] is False
    generator = cfg.raw["models"]["generator"]
    assert generator["model"]["id"] == "gpt-5.6-sol"
    assert "gpt-5.6" in generator["model"]["refused_aliases"]
    assert generator["openai"]["store"] is False
    assert generator["openai"]["background"] is False
    assert generator["openai"]["tools"] == "none"
    assert generator["openai"]["conversations"] is False
    assert generator["openai"]["previous_response_state"] is False
    assert generator["openai"]["automatic_retries"] == 0
