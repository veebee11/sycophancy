"""Fetch and verify the reference-source files listed in the download manifest.

    uv run python scripts/fetch_sources.py --manifest data/sources/downloads.yaml
    uv run python scripts/fetch_sources.py --manifest data/sources/downloads.yaml --verify-only

For each file: download it only if it is not already present, check its size and
the provider's checksum, compute our own SHA-256, and — on first download —
record the SHA-256 and access date in the manifest. A file whose bytes differ
from a previously recorded SHA-256 is refused. Any mismatch stops the run.

ZIP archives are extracted into a folder beside them after every member path has
been checked. Pickle members are never extracted.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
from datetime import date
from pathlib import Path

import yaml

from reasonstyle.corpus.downloads import ChecksumMismatch, UnsafeArchive, safe_extract, verify


def _header(path: Path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#") and line.strip():
            break
        lines.append(line)
    return "\n".join(lines).rstrip() + "\n\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--raw-dir", default="data/sources/raw")
    ap.add_argument("--verify-only", action="store_true",
                    help="never download; fail if a file is missing")
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    changed = False

    for entry in manifest["files"]:
        path = raw / entry["name"]
        if not path.exists():
            if args.verify_only:
                print(f"missing: {path}", file=sys.stderr)
                return 1
            print(f"downloading {entry['name']} ...")
            with urllib.request.urlopen(entry["url"], timeout=120) as response:
                data = response.read()
            fetched = True
        else:
            data = path.read_bytes()
            fetched = False

        checksum = entry["provider_checksum"]
        try:
            sha256 = verify(data, size=entry["size_bytes"], algorithm=checksum["algorithm"],
                            expected=checksum["value"])
        except ChecksumMismatch as exc:
            print(f"CHECKSUM MISMATCH for {entry['name']}: {exc}", file=sys.stderr)
            return 1
        if entry.get("sha256") and entry["sha256"] != sha256:
            print(f"CHECKSUM MISMATCH for {entry['name']}: sha256 {sha256}, "
                  f"recorded {entry['sha256']}", file=sys.stderr)
            return 1

        if fetched:
            with tempfile.NamedTemporaryFile(dir=raw, delete=False) as tmp:
                tmp.write(data)
            Path(tmp.name).replace(path)
        if not entry.get("sha256"):
            entry["sha256"] = sha256
            entry["access_date"] = date.today().isoformat()
            changed = True
        print(f"  ok  {entry['name']}: {len(data):,} bytes, {checksum['algorithm']} matches, "
              f"sha256 {sha256}")

        if entry.get("extract"):
            target = raw / Path(entry["name"]).stem
            try:
                result = safe_extract(path, target)
            except UnsafeArchive as exc:
                print(f"UNSAFE ARCHIVE {entry['name']}: {exc}", file=sys.stderr)
                return 1
            print(f"      extracted {len(result.extracted)} files to {target}/; "
                  f"skipped {len(result.skipped_pickles)} pickle member(s), never opened")

    if changed:
        body = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True, width=200)
        manifest_path.write_text(_header(manifest_path) + body, encoding="utf-8")
        print(f"recorded sha256 and access date in {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
