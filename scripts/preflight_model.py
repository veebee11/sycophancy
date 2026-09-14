"""Read-only check that the generator's weights are already on this machine.

    HF_HUB_OFFLINE=1 HF_HOME=/data/$USER/hf_cache \
    uv run python scripts/preflight_model.py --config configs/experiment.yaml

Reads the Hugging Face cache directory and nothing else: it opens no network
connection and downloads nothing, by design. If the model is absent, or its
snapshot carries metadata but not its weights, it says so and exits non-zero —
that is a decision for a person, not something to fix by fetching thirty
gigabytes.

Weights are verified, not assumed: when a safetensors index is present every
shard it names must exist, because an interrupted download leaves a complete
config and tokenizer behind and no weights at all.

The resolved commit sha is what goes into the record and into the server's
``--revision``; a repository id alone does not identify the weights that
produced a draft. ``--print-revision`` prints only that sha, for the launcher.
"""

from __future__ import annotations

import argparse
import json
import sys

from reasonstyle.config import load_config
from reasonstyle.generation.backends import offline_problems
from reasonstyle.generation.environment import (
    ModelNotCached,
    cache_root,
    library_versions,
    resolve_cached_model,
    revision_agreement,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", help="override the configured repository id")
    ap.add_argument("--hf-home", help="cache location; defaults to HF_HOME or the user cache")
    ap.add_argument("--require-config-match", action="store_true",
                    help="fail unless models.generator.model.revision equals the cached commit")
    ap.add_argument("--print-revision", action="store_true",
                    help="print only the resolved commit sha, for scripting")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    quiet = args.print_revision
    out = sys.stderr if quiet else sys.stdout

    cfg = load_config(args.config)
    configured = cfg.raw["models"]["generator"]["model"]
    repo_id = args.model or configured["repo_id"]
    root = cache_root(args.hf_home)

    print(f"repository   {repo_id}", file=out)
    print(f"cache        {root}", file=out)
    for problem in offline_problems(cfg):
        print(f"  NOTE {problem}", file=sys.stderr)

    try:
        cached = resolve_cached_model(repo_id, hf_home=args.hf_home)
    except ModelNotCached as exc:
        print(f"\nNOT AVAILABLE: {exc}", file=sys.stderr)
        print("Nothing was downloaded. Choose a model that is already complete on this "
              "machine, or ask for a download to be approved separately.", file=sys.stderr)
        return 1

    size = f"{cached.size_bytes / 1e9:.1f} GB" if cached.size_bytes else "unknown"
    print(f"revision     {cached.revision}", file=out)
    print(f"refs         {', '.join(cached.refs) or '(none: snapshot only)'}", file=out)
    print(f"snapshot     {cached.snapshot_path}", file=out)
    print(f"weights      {len(cached.weight_files)} file(s) verified present", file=out)
    print(f"size on disk {size}", file=out)
    print(f"libraries    {library_versions()}", file=out)

    problems = revision_agreement(configured["revision"], cached)
    if problems:
        for problem in problems:
            print(f"  revision: {problem}", file=sys.stderr)
        if args.require_config_match:
            print("refusing: the configuration and the cache must name the same commit.",
                  file=sys.stderr)
            return 1
    else:
        print(f"config       revision matches ({configured['revision']})", file=out)

    if args.print_revision:
        print(cached.revision)
    if args.json:
        print(json.dumps({**cached.as_dict(), "cache_root": str(root)}, indent=2), file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
