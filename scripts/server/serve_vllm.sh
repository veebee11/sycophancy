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
# The runtime record is written as server_runtime.json.pending, promoted by an
# atomic rename once this launcher's own child FIRST answers /health, and kept
# while that child process is alive. Health is not re-checked after that. Both
# files are removed when the launcher exits for any reason.
# A port that already answers is refused: this host is shared, and a service
# someone else started must never be recorded as ours.
#
# Announce the GPU and the expected duration in #01-servers before starting.
set -euo pipefail

# --- runtime-record lifecycle -------------------------------------------------
# Functions only; they read PYTHON, PORT and RUNTIME_RECORD, and the child's pid
# is kept in CHILD. The tests source this file to exercise them with a fake child.

health_ok() {
  # True only if something answers 2xx at the configured port's /health. Proxies
  # are bypassed explicitly, so a lab proxy can never answer for localhost.
  "$PYTHON" - "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 <<'PY'
import sys, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
opener.open(sys.argv[1], timeout=2).close()
PY
}

remove_runtime_records() {
  rm -f "$RUNTIME_RECORD" "$RUNTIME_RECORD.pending"
}

stop_child_and_clean_up() {
  local status=$?
  trap - EXIT INT TERM HUP
  if [[ -n "${CHILD:-}" ]] && kill -0 "$CHILD" 2>/dev/null; then
    kill -TERM "$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
  fi
  remove_runtime_records
  exit "$status"
}

prepare_launch() {
  # Refuse BEFORE touching any record: if the port answers, the record on disk
  # may describe a server that is still running, and it is not ours to delete.
  if health_ok; then
    echo "http://127.0.0.1:$PORT/health already responds. This host is shared, so that" \
         "service is not assumed to be ours. Choose another PORT; nothing was launched." >&2
    return 1
  fi
  trap stop_child_and_clean_up EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  remove_runtime_records
}

supervise() {
  # Run "$@" in the background, publish the pending record only once that child
  # is alive and healthy, then wait on it. Termination reaches the child through
  # the traps above (a background child of a script ignores Ctrl-C by itself).
  local timeout="${HEALTH_TIMEOUT_SECONDS:-900}" poll="${HEALTH_POLL_SECONDS:-5}"
  local deadline=$((SECONDS + timeout)) status=0
  "$@" &
  CHILD=$!
  echo "server pid $CHILD; waiting up to ${timeout}s for http://127.0.0.1:$PORT/health"
  while :; do
    if ! kill -0 "$CHILD" 2>/dev/null; then
      wait "$CHILD" || status=$?
      CHILD=""
      echo "the server exited (status $status) before becoming healthy;" \
           "no runtime record was published." >&2
      return 1
    fi
    if health_ok && kill -0 "$CHILD" 2>/dev/null; then
      break
    fi
    if (( SECONDS >= deadline )); then
      echo "the server was not healthy within ${timeout}s; stopping it." \
           "No runtime record was published." >&2
      return 1
    fi
    sleep "$poll"
  done
  mv -f "$RUNTIME_RECORD.pending" "$RUNTIME_RECORD"   # rename(2): atomic
  echo "ready      runtime record $RUNTIME_RECORD"
  wait "$CHILD" || status=$?
  CHILD=""
  return "$status"
}

# Sourced (by the tests): define the functions above and stop here.
[[ "${BASH_SOURCE[0]}" == "$0" ]] || return 0

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
CHILD=""

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

# Refuses if the port already answers; otherwise installs the cleanup traps and
# removes any stale active or pending record left by an earlier attempt.
prepare_launch

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
# Installed distribution versions, read from package metadata WITHOUT importing
# the packages. Importing vllm logs to stdout ("INFO ... detected platform"),
# which would be captured here and break the record written below. Metadata
# also keeps the full local version, e.g. 0.8.5.post1+cu118.
dist_version() { "$PYTHON" -c 'import sys; from importlib.metadata import version; print(version(sys.argv[1]))' "$1" 2>/dev/null || echo unknown; }
VLLM_VERSION="$(dist_version vllm)"
TF_VERSION="$(dist_version transformers)"
TORCH_VERSION="$(dist_version torch)"

# The runtime record: what the client's log points at, so GPU and library
# details come from the machine that loads the weights, not from a client shell.
# Written as .pending; supervise promotes it when the server first answers /health.
mkdir -p "$(dirname "$RUNTIME_RECORD")"
"$PYTHON" - "$RUNTIME_RECORD.pending" <<PY
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
print("pending runtime record", sys.argv[1])
PY

echo "model      $MODEL"
echo "revision   $REVISION"
echo "snapshot   $SNAPSHOT"
echo "gpu        index $GPU ($GPU_NAME), one device"
echo "cache      $HF_HOME (offline)"
echo "gen config $GENERATION_CONFIG (model generation_config.json NOT loaded)"
echo "endpoint   http://127.0.0.1:$PORT/v1"

supervise env CUDA_VISIBLE_DEVICES="$GPU" "$VLLM" serve "$MODEL" \
  --revision "$REVISION" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_MEM_FRACTION" \
  --seed "$SEED" \
  --generation-config "$GENERATION_CONFIG" \
  --disable-log-requests
