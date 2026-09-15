#!/usr/bin/env bash
# Prepare the generation environment on a lab GPU server.
#
#   BASE=/data/$USER/reasonstyle bash scripts/server/setup_chomusuke.sh
#
# Everything lives under /data/$USER: the checkout, the virtual environment and
# the model cache. Nothing is written to $HOME and nothing touches the NAS.
# No credential is needed or stored: the generator is a local model.
set -euo pipefail

BASE="${BASE:-/data/$USER/reasonstyle}"
export HF_HOME="${HF_HOME:-/data/$USER/hf_cache}"

if [[ ! -d "/data/$USER" ]]; then
  echo "/data/$USER does not exist on this host — experiments must run under /data/\$USER." >&2
  exit 1
fi

echo "checkout   $BASE"
echo "hf cache   $HF_HOME"
echo "python     3.11"

mkdir -p "$HF_HOME"
if [[ ! -d "$BASE" ]]; then
  echo "Clone or copy the repository to $BASE first, then re-run this script." >&2
  exit 1
fi

cd "$BASE"
uv venv --python 3.11 .venv
uv pip install --python .venv -e .
# vLLM is installed here, not declared in pyproject.toml: it is needed only on
# the GPU host, it brings its own torch build, and its exact version is recorded
# in the environment block of every run rather than in a laptop lockfile.
uv pip install --python .venv "vllm>=0.8.5"
# Triton imports setuptools at runtime when a GPU worker starts, and `uv venv`
# does not seed it: on Chomusuke02 the engine failed at startup with
# "No module named 'setuptools'". Installed explicitly, at one fixed version,
# so the environment is reproducible.
uv pip install --python .venv "setuptools==79.0.1"
.venv/bin/python -c "import setuptools" || {
  echo "setuptools is not importable in .venv; the vLLM engine would fail to start." >&2
  exit 1
}

echo
echo "done. Before any run, in the same shell:"
echo "  export HF_HOME=$HF_HOME"
echo "  export HF_HUB_OFFLINE=1     # a missing model must be an error, not a download"
echo "  cd $BASE"
