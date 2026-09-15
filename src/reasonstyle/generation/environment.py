"""What produced a draft: the model, the machine and the libraries.

Two jobs.

**Resolving a revision from the local cache, read-only.** The generator's
repository id names a model; the commit that was actually loaded is what has to
be reported. It is read from the Hugging Face cache directory on disk — the
layout is stable and documented — and nothing here contacts the Hub or downloads
anything. A repository that is not cached is an error to report, never a
download to start.

**Recording the environment.** A seed does not make a local model's output
bit-for-bit reproducible: kernel selection, driver version, batch composition
and library versions all move the result. So the record is what stands: the
resolved revision, the decoding settings, the GPU, the library versions, and the
hashes of the exact prompt and the exact response. Those are the authoritative
account of what was produced, and the raw response file is kept beside them.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CachedModel",
    "ModelNotCached",
    "RunEnvironment",
    "cache_root",
    "library_versions",
    "load_server_runtime",
    "missing_weights",
    "resolve_cached_model",
    "revision_agreement",
]

OFFLINE_ENV = {"HF_HUB_OFFLINE": "1"}

#: What a model directory must contain before a server can load it. Config and
#: tokenizer alone are not enough: an interrupted download leaves exactly those
#: small files behind and none of the weights.
REQUIRED_METADATA = ("config.json", "tokenizer_config.json")
WEIGHT_INDEXES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
WEIGHT_PATTERNS = ("model*.safetensors", "pytorch_model*.bin", "*.gguf")


class ModelNotCached(RuntimeError):
    """The model is not in the local cache — report it, do not fetch it."""


@dataclass(frozen=True, slots=True)
class CachedModel:
    repo_id: str
    revision: str                   # the resolved commit sha
    snapshot_path: str
    refs: tuple[str, ...] = ()      # branch names pointing at this commit
    size_bytes: int | None = None
    weight_files: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _readable(path: Path) -> bool:
    """A cache entry is a symlink into ``blobs/``; a dangling one is not a file."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def missing_weights(snapshot: Path) -> tuple[list[str], list[str]]:
    """``(missing, present)`` weight files for a cached snapshot.

    A sharded model carries an index naming every shard, so each one is checked
    by name — the usual failure is a download interrupted with some shards
    absent, which leaves an index that still lists them. Without an index, at
    least one weight file must be present.
    """
    for index_name in WEIGHT_INDEXES:
        index = snapshot / index_name
        if not _readable(index):
            continue
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        except (json.JSONDecodeError, OSError) as exc:
            return ([f"{index_name} is unreadable: {exc}"], [])
        shards = sorted(set(weight_map.values()))
        missing = [name for name in shards if not _readable(snapshot / name)]
        return (missing, [name for name in shards if name not in missing])

    present = sorted({p.name for pattern in WEIGHT_PATTERNS
                      for p in snapshot.glob(pattern) if _readable(p)})
    if not present:
        return (["no weight file (no safetensors index, no model weights)"], [])
    return ([], present)


def cache_root(hf_home: str | os.PathLike[str] | None = None) -> Path:
    """Where the Hugging Face hub cache lives, without importing the hub."""
    if hf_home:
        return Path(hf_home) / "hub"
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_dir(root: Path, repo_id: str) -> Path:
    return root / ("models--" + repo_id.replace("/", "--"))


def resolve_cached_model(repo_id: str, *, hf_home: str | os.PathLike[str] | None = None,
                         required_files: tuple[str, ...] = REQUIRED_METADATA,
                         require_weights: bool = True,
                         ) -> CachedModel:
    """Resolve ``repo_id`` to a cached commit, reading the filesystem only.

    Raises :class:`ModelNotCached` with a precise reason — no repository, no
    snapshot, missing metadata, or missing weights — so that the caller can stop
    and report instead of downloading. The cache layout read here is the one the
    Hub libraries write and the one a server reads from ``HF_HOME``, so a
    successful check means the launcher will find the same snapshot.
    """
    root = cache_root(hf_home)
    repo = _repo_dir(root, repo_id)
    if not repo.is_dir():
        raise ModelNotCached(
            f"{repo_id} is not in the cache at {root} (looked for {repo.name}/)")

    snapshots = sorted(p for p in (repo / "snapshots").glob("*") if p.is_dir())
    if not snapshots:
        raise ModelNotCached(f"{repo_id} has no snapshot in {repo}")

    refs: dict[str, list[str]] = {}
    refs_dir = repo / "refs"
    if refs_dir.is_dir():
        for ref in sorted(refs_dir.glob("**/*")):
            if ref.is_file():
                refs.setdefault(ref.read_text(encoding="utf-8").strip(), []).append(
                    str(ref.relative_to(refs_dir)))

    # Prefer the commit a ref points at; otherwise the single snapshot present.
    chosen = next((s for s in snapshots if s.name in refs), snapshots[-1])
    missing = [name for name in required_files if not _readable(chosen / name)]
    if missing:
        raise ModelNotCached(
            f"{repo_id} snapshot {chosen.name} is incomplete: missing {missing}. "
            f"A partial cache must not be completed by downloading during a run.")

    weights: tuple[str, ...] = ()
    if require_weights:
        absent, present = missing_weights(chosen)
        if absent:
            raise ModelNotCached(
                f"{repo_id} snapshot {chosen.name} has metadata but not its weights: "
                f"missing {absent[:5]}{' …' if len(absent) > 5 else ''} "
                f"({len(present)} shard(s) present). An interrupted download looks "
                f"exactly like this; completing it is a separate, approved step.")
        weights = tuple(present)

    size = sum(p.stat().st_size for p in repo.rglob("*") if p.is_file() and not p.is_symlink())
    return CachedModel(repo_id=repo_id, revision=chosen.name, snapshot_path=str(chosen),
                       refs=tuple(refs.get(chosen.name, ())), size_bytes=size or None,
                       weight_files=weights)


def revision_agreement(configured: str | None, cached: CachedModel,
                       server: str | None = None) -> list[str]:
    """Disagreements between the config, the cache and a running server.

    Three places name a revision and all three must say the same thing; a draft
    produced by weights other than the recorded ones is unreportable.
    """
    problems = []
    if not configured:
        problems.append(
            f"models.generator.model.revision is not recorded; the cached commit is "
            f"{cached.revision}. Record it in the configuration before running.")
    elif configured != cached.revision:
        problems.append(f"the configuration pins {configured} but the cache holds "
                        f"{cached.revision}")
    if server and server != cached.revision:
        problems.append(f"the server reports {server} but the cache holds {cached.revision}")
    return problems


def server_settings_problems(cfg, server: dict[str, Any]) -> list[str]:
    """Launch settings the server must have used for its output to be reportable.

    ``generation_config`` must be the configured mode ("vllm"): under vLLM's
    default "auto" the model's own generation_config.json silently supplies any
    sampling field a request omits. A record written by an older launcher, with
    no such field, is refused as well — its sampling defaults are unknown.
    """
    expected = cfg.raw["models"]["generator"]["vllm"]["generation_config"]
    actual = server.get("generation_config")
    if actual != expected:
        return [f"the server ran with generation_config={actual!r}, not {expected!r}: "
                f"restart it with scripts/server/serve_vllm.sh, which passes "
                f"--generation-config {expected}"]
    return []


def load_server_runtime(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The record the launcher wrote when it started the server.

    GPU index, GPU name, dtype, revision, library versions and launch settings
    come from the machine that actually loaded the weights — never inferred from
    the client's own shell, which may be on a different host entirely.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no server runtime record at {path}: start the server with "
            f"scripts/server/serve_vllm.sh, which writes one.")
    return json.loads(path.read_text(encoding="utf-8"))


def library_versions() -> dict[str, str | None]:
    """Versions of what is installed, without importing heavy packages."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in ("vllm", "transformers", "tokenizers", "torch", "huggingface-hub"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


@dataclass(frozen=True, slots=True)
class RunEnvironment:
    """Everything recorded about how one response was produced."""

    repo_id: str
    revision: str | None
    snapshot_path: str | None
    backend: str
    endpoint: str | None
    decoding: dict[str, Any]
    seed: int | None
    #: Straight from the launcher's runtime record: the GPU that loaded the
    #: weights, its name, the dtype, the context length and the library versions
    #: on the server. Never inferred from the client's shell.
    server: dict[str, Any] | None = None
    dtype: str | None = None
    gpu: str | None = None
    libraries: dict[str, str | None] = field(default_factory=dict)
    client_host: str | None = None
    offline_env: dict[str, str | None] = field(default_factory=dict)
    prompt_sha256: str | None = None
    response_sha256: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    #: Stated wherever the record is shown. A seed is part of the record, not a
    #: reproducibility guarantee for a local model.
    reproducibility_note = (
        "The seed is recorded but does not guarantee bit-for-bit reproduction "
        "across GPUs, drivers or library versions. The saved raw response, this "
        "environment record and the hashes are the authoritative record.")

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "reproducibility_note": self.reproducibility_note}


def describe_run(cfg, *, cached: CachedModel | None, endpoint: str | None,
                 server: dict[str, Any] | None = None, **measured: Any) -> RunEnvironment:
    """Build the environment record from the configuration, the cache and the
    server's own runtime record.

    The GPU and dtype come from ``server`` — the machine that loaded the
    weights — because the client may be a laptop on another host, and a GPU name
    read from the client's shell would be a fiction.
    """
    gen = cfg.raw["models"]["generator"]
    server = server or {}
    return RunEnvironment(
        repo_id=gen["model"]["repo_id"],
        revision=cached.revision if cached else server.get("revision"),
        snapshot_path=cached.snapshot_path if cached else server.get("snapshot_path"),
        backend=gen["backend"],
        endpoint=endpoint,
        decoding=dict(gen["decoding"]),
        seed=gen["decoding"]["seed"],
        server=server or None,
        dtype=server.get("dtype"),
        gpu=server.get("gpu_name"),
        libraries=library_versions(),
        client_host=platform.node(),
        offline_env={k: os.environ.get(k) for k in OFFLINE_ENV},
        **measured,
    )
