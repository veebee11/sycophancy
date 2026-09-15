#!/usr/bin/env bash
# Start the local vLLM endpoint for corpus drafting, on ONE explicit GPU.
#
#   GPU=0 bash scripts/server/serve_vllm.sh
#   GPU=2 MODEL=Qwen/Qwen3-14B PORT=8011 bash scripts/server/serve_vllm.sh
#
# Binds 127.0.0.1 only, so the endpoint is not reachable from other hosts.
# Runs offline: if the weights are not already complete in the cache the run
# stops rather than downloading tens of gigabytes unattended.
#
# The commit sha is resolved by the read-only preflight and passed to vLLM as
# --revision, so the configuration, the cached snapshot and the weights the
# server actually loads all name the same commit.
#
# Everything comes from the project virtual environment: no activation needed,
# and no dependence on whatever "vllm" happens to be first on PATH.
#
# Announce the GPU and the expected duration in #01-servers before starting.
set -euo pipefail

BASE="${BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GPU="${GPU:?set GPU to the index of the single GPU to use, e.g. GPU=0}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
PORT="${PORT:-8011}"
SEED="${SEED:-20260914}"
MAX_LEN="${MAX_LEN:-8192}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.90}"
CONFIG="${CONFIG:-$BASE/configs/experiment.yaml}"
RUNTIME_RECORD="${RUNTIME_RECORD:-$BASE/data/pilot/server_runtime.json}"
# Fixed, not an environment override. vLLM's default, "auto", loads the model's
# own generation_config.json and uses it as the default for any sampling field a
# request omits (for Qwen3 that includes top_k and a different temperature and
# top_p). "vllm" loads none of it: every sampling value is either sent by the
# client, as configured in models.generator.decoding, or vLLM's neutral default.
GENERATION_CONFIG=vllm

export HF_HOME="${HF_HOME:-/data/$USER/hf_cache}"
export HF_HUB_OFFLINE=1
unset HF_HUB_CACHE 2>/dev/null || true   # one cache layout only: $HF_HOME/hub

VENV="$BASE/.venv"
PYTHON="$VENV/bin/python"
VLLM="$VENV/bin/vllm"

if [[ ! -d "/data/$USER" ]]; then
  echo "/data/$USER does not exist on this host; experiments run under /data/\$USER." >&2
  exit 1
fi
for tool in "$PYTHON" "$VLLM"; do
  if [[ ! -x "$tool" ]]; then
    echo "$tool not found. Run scripts/server/setup_chomusuke.sh first." >&2
    exit 1
  fi
done

# The preflight reads $HF_HOME/hub — the same layout vLLM reads through
# HF_HOME — so a pass here means the server will find this exact snapshot.
# --download-dir is deliberately NOT passed: it would point vLLM at a flat
# directory instead of the hub cache, and the check above would prove nothing.
echo "resolving the cached revision (read-only, offline)…"
REVISION="$("$PYTHON" "$BASE/scripts/preflight_model.py" \
  --config "$CONFIG" --model "$MODEL" --require-config-match --print-revision)"
if [[ -z "$REVISION" ]]; then
  echo "no revision resolved; refusing to start." >&2
  exit 1
fi

SNAPSHOT="$HF_HOME/hub/models--${MODEL//\//--}/snapshots/$REVISION"
if [[ ! -d "$SNAPSHOT" ]]; then
  echo "the resolved snapshot $SNAPSHOT is not where vLLM will look." >&2
  exit 1
fi

GPU_NAME="$(nvidia-smi --id="$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
VLLM_VERSION="$("$PYTHON" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
TF_VERSION="$("$PYTHON" -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo unknown)"
TORCH_VERSION="$("$PYTHON" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"

# The runtime record: what the client's log points at, so GPU and library
# details come from the machine that loads the weights, not from a client shell.
mkdir -p "$(dirname "$RUNTIME_RECORD")"
"$PYTHON" - "$RUNTIME_RECORD" <<PY
import json, platform, sys, datetime
record = {
    "host": platform.node(),
    "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "repo_id": "$MODEL",
    "revision": "$REVISION",
    "snapshot_path": "$SNAPSHOT",
    "gpu_index": "$GPU",
    "gpu_name": "$GPU_NAME",
    "dtype": "$DTYPE",
    "max_model_len": int("$MAX_LEN"),
    "gpu_memory_utilization": float("$GPU_MEM_FRACTION"),
    "seed": int("$SEED"),
    "generation_config": "$GENERATION_CONFIG",
    "endpoint": "http://127.0.0.1:$PORT/v1",
    "hf_home": "$HF_HOME",
    "hf_hub_offline": "1",
    "libraries": {"vllm": "$VLLM_VERSION", "transformers": "$TF_VERSION",
                  "torch": "$TORCH_VERSION"},
}
open(sys.argv[1], "w").write(json.dumps(record, indent=2) + "\n")
print("runtime record", sys.argv[1])
PY

echo "model      $MODEL"
echo "revision   $REVISION"
echo "snapshot   $SNAPSHOT"
echo "gpu        index $GPU ($GPU_NAME), one device"
echo "cache      $HF_HOME (offline)"
echo "gen config $GENERATION_CONFIG (model generation_config.json NOT loaded)"
echo "endpoint   http://127.0.0.1:$PORT/v1"

CUDA_VISIBLE_DEVICES="$GPU" exec "$VLLM" serve "$MODEL" \
  --revision "$REVISION" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_MEM_FRACTION" \
  --seed "$SEED" \
  --generation-config "$GENERATION_CONFIG" \
  --disable-log-requests
