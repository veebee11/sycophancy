"""Hashes must be reproducible across runs, machines and irrelevant reorderings."""

from __future__ import annotations

import json

import pytest

from reasonstyle.hashing import canonical_json, content_hash, file_sha256, sha256_of, short


def test_canonical_json_sorts_keys_and_is_tight():
    assert canonical_json({"b": 1, "a": [2, 3]}) == '{"a":[2,3],"b":1}'


def test_content_hash_is_independent_of_key_order():
    assert content_hash({"a": 1, "b": {"x": 1, "y": 2}}) == content_hash({"b": {"y": 2, "x": 1}, "a": 1})


def test_content_hash_changes_when_a_value_changes():
    assert content_hash({"tau": 0.4054651081}) != content_hash({"tau": 0.4054651082})


def test_content_hash_distinguishes_types():
    """'1' and 1 must not collide: a stringified token id is not a token id."""
    assert content_hash({"id": 1}) != content_hash({"id": "1"})


def test_non_finite_values_are_rejected():
    """A NaN would hash successfully but break JSON interchange downstream."""
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_sha256_of_accepts_str_and_bytes_identically():
    assert sha256_of("abc") == sha256_of(b"abc")


def test_file_sha256_is_byte_exact(tmp_path):
    """Semantically inert edits (a comment) must still change the file hash."""
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("x: 1\n")
    b.write_text("# a comment\nx: 1\n")
    assert file_sha256(a) != file_sha256(b)
    assert content_hash(json.loads('{"x":1}')) == content_hash({"x": 1})


def test_short_prefix():
    digest = sha256_of("abc")
    assert short(digest) == digest[:12]
    assert len(short(digest, 8)) == 8
