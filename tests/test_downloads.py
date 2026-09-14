"""Download verification and safe extraction. No network: all synthetic."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from reasonstyle.corpus.downloads import (
    ChecksumMismatch,
    UnsafeArchive,
    git_blob_sha1,
    safe_extract,
    verify,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_sources.py"


def test_git_blob_sha1_matches_git():
    # `printf 'hello\n' | git hash-object --stdin`
    assert git_blob_sha1(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_verify_returns_our_sha256_when_the_provider_checksum_matches():
    data = b"synthetic source file\n"
    sha = verify(data, size=len(data), algorithm="git_blob_sha1", expected=git_blob_sha1(data))
    assert len(sha) == 64


def test_a_wrong_size_is_refused():
    with pytest.raises(ChecksumMismatch, match="size"):
        verify(b"abc", size=4, algorithm="md5", expected="x")


def test_a_wrong_provider_checksum_is_refused():
    with pytest.raises(ChecksumMismatch, match="md5"):
        verify(b"abc", size=3, algorithm="md5", expected="0" * 32)


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


@pytest.mark.parametrize("bad", ["../escape.txt", "/etc/passwd", "a/../../escape.txt"])
def test_an_unsafe_member_path_refuses_the_whole_archive(tmp_path, bad):
    archive = _zip(tmp_path / "a.zip", {"ok.json": b"{}", bad: b"x"})
    target = tmp_path / "out"
    with pytest.raises(UnsafeArchive):
        safe_extract(archive, target)
    assert not target.exists() or not any(target.rglob("*"))    # nothing was written


def test_pickle_members_are_never_extracted(tmp_path):
    archive = _zip(tmp_path / "a.zip", {"data/article.json": b"{}", "data/frame.pkl": b"\x80\x04",
                                        "data/also.pickle": b"\x80\x04"})
    result = safe_extract(archive, tmp_path / "out")
    assert result.extracted == ("data/article.json",)
    assert set(result.skipped_pickles) == {"data/frame.pkl", "data/also.pickle"}
    assert not list((tmp_path / "out").rglob("*.pkl"))
    assert not list((tmp_path / "out").rglob("*.pickle"))


def test_the_script_stops_on_a_checksum_mismatch(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "file.csv").write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "downloads.yaml"
    manifest.write_text(yaml.safe_dump({"files": [{
        "name": "file.csv", "registry_key": "x", "url": "https://example.org/file.csv",
        "size_bytes": 8, "provider_checksum": {"algorithm": "git_blob_sha1", "value": "0" * 40},
        "extract": False, "sha256": None, "access_date": None}]}))
    result = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(manifest),
                             "--raw-dir", str(raw), "--verify-only"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "CHECKSUM MISMATCH" in result.stderr
    assert yaml.safe_load(manifest.read_text())["files"][0]["sha256"] is None   # nothing recorded
