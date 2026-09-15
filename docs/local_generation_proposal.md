# Local generation on Chomusuke — proposed changes

**Status: approved with corrections on 2026-09-14 and implemented locally.**
No server has been contacted, no GPU job launched, no model run and no weights
downloaded. `models.generator.model.revision` is still `null`: it is filled in
only from the read-only preflight on Chomusuke02, which awaits a server account.
See [`current_status.md`](current_status.md).

The Anthropic path described in `drafting_proposal.md` §5 is withdrawn in full:
there is no Anthropic dependency, no API key anywhere, and no external paid call.

Sections 1–8 and 11–12 describe what is implemented. §9 records the model
choice, and §10 the decisions that settled the proposal's open questions.

What changed from this proposal, as the curator directed:

| | |
|---|---|
| Backend | only `VLLMOpenAIBackend` is built; the in-process Transformers backend is deferred unless vLLM proves unavailable |
| Model | `Qwen/Qwen3-14B`, **provided it is already cached on Chomusuke** — a read-only preflight resolves the revision, and nothing is downloaded |
| Decoding | non-thinking, temperature 0.3, top_p 0.8, max_tokens 700, seed 20260914 recorded — and **not** described as guaranteeing reproduction across GPUs or library versions |
| Offline | `HF_HUB_OFFLINE=1` required for both preflight and run; a missing model is an error, never a download |
| Smoke-test material | the synthetic fixture id `energy_fixture_001`, never a pilot topic (see §11) |
| Authorisation | out of the experiment config entirely: `--send` **and** `REASONSTYLE_ALLOW_LOCAL_GENERATION=1` |

Target: one A6000 (48 GB) on `chomusuke02`, a four-GPU host. Everything runs
under `/data/$USER`, with no NAS and no API key, and use is announced in
`#01-servers`.

---

## 1. What is removed

| File | Change |
|---|---|
| `src/reasonstyle/generation/backends.py` | delete `AnthropicBackend`, `anthropic_payload`, `_API_URL`, `_API_VERSION`, the `x-api-key` header path |
| `configs/experiment.yaml` | delete `models.generator.api_key_env`, `provider: anthropic`, `interface: messages_api`, `request_fields.thinking`, `omitted_fields` |
| `src/reasonstyle/config.py` | delete the Anthropic-specific assertions in `_check_drafting` |
| `scripts/smoke_test.py` | rewritten (§5) |
| `tests/test_generation.py` | the six Anthropic tests replaced by local-backend tests |

Nothing about **requests, the response schema, parsing, hashing, repair limits
or provenance** changes: those are provider-independent already, and the local
backends plug into the same `send(request, cfg, *, allow_live=False)` shape.

## 2. What is added

### `src/reasonstyle/generation/backends.py`

```
VLLMOpenAIBackend          # POST to a local OpenAI-compatible endpoint (urllib)
vllm_payload()             # the request body, built from the config alone
authorization_problems()   # why a run may not proceed
offline_problems()         # HF_HUB_OFFLINE must be on
BackendUnavailable         # the endpoint is not running
FakeBackend                # unchanged; tests and dry runs
```

Only the vLLM backend is built. An in-process Transformers backend is deferred
unless vLLM proves unavailable or unsuitable; adding it later means a class with
the same `send` signature and nothing else moving.

A run needs **two yeses, neither of them in the config**: `allow_live=True` from
an explicit `--send`, and `REASONSTYLE_ALLOW_LOCAL_GENERATION=1` in the
environment. No key is read, sent or stored: the endpoint is on the loopback
interface, and the request carries no authorization header at all.

The client needs no GPU library: it speaks HTTP to the local server through
`urllib`, so the test suite and every script keep running on a laptop.

### `src/reasonstyle/generation/environment.py` (new)

Captures what has to be recorded with every call:

```
repo_id, revision (resolved commit sha, not a branch name), snapshot_path,
backend, endpoint, decoding, seed,
server: {gpu_index, gpu_name, dtype, max_model_len, seed, host,
         libraries: {vllm, transformers, torch}},
libraries (client side), prompt_sha256, response_sha256,
input_tokens, output_tokens
```

The revision is **resolved from the local cache**, never trusted from the
config, and three places must agree: the configuration, the cached snapshot and
the server's runtime record. A model whose commit cannot be resolved is a hard
failure, since an unpinned generator cannot be reported in a thesis.

The GPU, dtype and library versions come from the **server's runtime record**,
written by the launcher on the machine that loads the weights. The client never
infers them from its own shell — it may be a laptop on another host, where a
`CUDA_VISIBLE_DEVICES` value would be meaningless.

`missing_weights()` checks completeness: when a safetensors index is present,
every shard it names must exist and be non-empty; otherwise at least one weight
file must be. Config and tokenizer alone are exactly what an interrupted
download leaves behind.

### `src/reasonstyle/generation/log.py`

`LogEntry` gains `model_revision`, `runtime` (the environment block above),
`seed`, `gpu`. Existing fields keep their meaning; `model_returned` becomes the
repo id the backend actually loaded.

## 3. Configuration (replaces `models.generator`)

```yaml
models:
  generator:
    approval_status: approved_pending_smoke_test
    backend: local_vllm_openai
    supported_backends: [local_vllm_openai]

    model:
      repo_id: Qwen/Qwen3-14B
      revision: null                   # filled from the read-only cache check, never a download
      trust_remote_code: false
      excluded_families: [llama, meta-llama, llama-3, llama3]

    decoding:
      thinking: disabled               # Qwen3: enable_thinking=False in the chat template
      temperature: 0.3
      top_p: 0.8
      max_tokens: 700
      seed: 20260914                   # recorded; NOT a reproducibility guarantee
      n: 1
      top_k: -1                        # the rest: explicit neutral values (added 2026-09-15)
      min_p: 0.0
      repetition_penalty: 1.0
      presence_penalty: 0.0
      frequency_penalty: 0.0

    vllm:
      base_url: http://127.0.0.1:8011/v1
      guided_json: true                # vLLM constrains output to the response schema
      timeout_seconds: 300
      require_offline_env: {HF_HUB_OFFLINE: "1"}
      generation_config: vllm          # server must not load generation_config.json
```

Authorisation to run is deliberately **not** a key here, and the loader refuses
a config that grows one: permission to run a model does not belong in a
scientific configuration that gets read, copied and shared.

**Structured output.** The response schema stays exactly as it is and is
enforced server-side (`response_format: json_schema`, strict), so the model
cannot answer with anything else. `parse_response` still validates what comes
back against the closed schema, and anything that does not fit is **rejected and
logged as rejected** — never repaired automatically, never retried.

## 4. Server scripts (new, under `scripts/server/`)

`setup_chomusuke.sh` — creates the environment under `/data/$USER`, nothing in
`$HOME`, nothing on the NAS:

```bash
export BASE=/data/$USER/reasonstyle
export HF_HOME=/data/$USER/hf_cache        # weights, tokenizers, everything
uv venv "$BASE/.venv" --python 3.11
uv pip install --python "$BASE/.venv" -e .
uv pip install --python "$BASE/.venv" "vllm>=0.8.5"
```

`serve_vllm.sh` — the OpenAI-compatible endpoint, one explicit GPU. It runs the
**virtual environment's own** `vllm` and `python`, so nothing depends on an
activated shell or on what is first on `PATH`:

```bash
REVISION="$($VENV/bin/python scripts/preflight_model.py \
  --config "$CONFIG" --model "$MODEL" --require-config-match --print-revision)"

CUDA_VISIBLE_DEVICES="$GPU" exec "$VENV/bin/vllm" serve "$MODEL" \
  --revision "$REVISION" \
  --host 127.0.0.1 --port "$PORT" \
  --dtype "$DTYPE" --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_MEM_FRACTION" --seed "$SEED" \
  --generation-config vllm
```

Three things this buys:

- **The revision is required, not optional.** The launcher will not start
  without a commit sha that the configuration and the cache agree on, and it
  passes that sha to vLLM, so the weights loaded are the weights recorded.
- **One cache layout.** `HF_HOME` is exported and `HF_HUB_CACHE` unset, and
  `--download-dir` is *not* passed — it would send vLLM to a flat directory
  instead of the hub cache the preflight just verified, making a passing
  preflight meaningless. The launcher additionally checks that
  `$HF_HOME/hub/models--<repo>/snapshots/$REVISION` exists.
- **A runtime record.** Before launching, it writes
  `data/pilot/server_runtime.json`: host, GPU index and name, dtype, revision,
  snapshot path, context length, seed, endpoint, and the server's vLLM,
  transformers and torch versions. The smoke test reads that file and refuses to
  run without it.

Bound to `127.0.0.1`, so the endpoint is not reachable from outside the host.
Both scripts refuse to run if `/data/$USER` does not exist.

## 5. The smoke test (`scripts/smoke_test.py`, rewritten)

Exactly **one four-condition group** from the **synthetic fixture** —
`data/fixtures/topics.yaml`, decision `energy_fixture_001` (§11), variant v1, `opt_1`, with the
fixture scenario text and the premise-indicator marker. One call. No repair, no
second attempt, no continuation into pilot generation: the script builds one
request, sends it once, and exits.

What it verifies and reports:

| Requirement | How |
|---|---|
| the four outputs parse | `parse_response` against the group schema |
| the validator runs | assembles a one-scenario `ScenarioRecord` and runs `validate_corpus(..., scope="fixture")`, printing errors, warnings and human-review counts |
| model id and revision recorded | resolved commit sha, printed and logged; the configuration, the cache and the server's runtime record must all name the same commit, or the run is refused |
| settings and seeds recorded | the whole `decoding` block plus the seed, in the log line |
| GPU and libraries recorded | read from the server's runtime record — the machine that loaded the weights — never inferred from the client shell; a missing record refuses the run |
| request and response hashes reproducible | `prompt_sha256` re-derived from the template and the brief, `response_sha256` over the parsed object |
| a failure triggers nothing | on rejection or validator error it logs, prints, and exits non-zero; no repair request is built |

Saved to `data/pilot/smoke/`: `request_<call_id>.json`, `response_<call_id>.json`
and one `generation_log.jsonl` line. Paths are printed at the end.

## 6. Commands

The dry run is simply **omitting `--send`** — there is no separate flag. It
contacts no server and generates nothing, printing the rendered request, the
resolved revision and the exact payload:

```bash
uv run python scripts/smoke_test.py --config configs/experiment.yaml
```

On `chomusuke02`, with vLLM (two terminals):

```bash
cd /data/$USER/reasonstyle && GPU=0 MODEL=Qwen/Qwen3-14B PORT=8011 bash scripts/server/serve_vllm.sh
```

```bash
cd /data/$USER/reasonstyle && \
REASONSTYLE_ALLOW_LOCAL_GENERATION=1 HF_HUB_OFFLINE=1 HF_HOME=/data/$USER/hf_cache \
uv run python scripts/smoke_test.py --config configs/experiment.yaml --send
```

Without `--send` the script prints the rendered request, the resolved revision
and the exact payload, and stops. `--send` alone is not enough either: without
`REASONSTYLE_ALLOW_LOCAL_GENERATION=1` it prints what is missing and stops,
and without `HF_HUB_OFFLINE=1` it refuses rather than risk a download. See §12
for the three commands in order.

## 7. Dependencies

**Nothing is added to `pyproject.toml`.** vLLM is not declared as an extra: it
is needed only on the GPU host, and declaring it drags a large CUDA dependency
tree into a lockfile resolved on a laptop. `scripts/server/setup_chomusuke.sh`
installs it explicitly into the project virtual environment:

```bash
uv pip install --python .venv -e .
uv pip install --python .venv "vllm>=0.8.5"
```

The exact version that produced any text is recorded in that run's server
runtime record, which is where it matters. The client side needs nothing extra:
the backend speaks HTTP through `urllib`.

## 8. Server requirements

| | |
|---|---|
| Machine | `chomusuke02`, a four-GPU A6000 (48 GB) host |
| GPUs needed | **one**: a 14B model in bfloat16 is ~28 GB of weights, inside 48 GB with an 8k context |
| Selection | `CUDA_VISIBLE_DEVICES=<n>`; the code never picks a GPU itself |
| Disk | ~30 GB for weights under `/data/$USER/hf_cache`; no NAS path is read or written |
| Network | localhost only; the vLLM endpoint binds `127.0.0.1` |
| Etiquette | announce the GPU and expected duration in `#01-servers` before starting |
| Runtime | the pilot is 72 calls of ≤700 new tokens plus at most 96 repairs: expected minutes on one A6000, not yet measured |

**Compute beyond generation (unresolved).** The A6000 is being used for corpus
generation. It may also support the first Llama-3.1-8B compatibility and
behavioural tests. The larger causal sweeps (Research_Plan_v6 §8, §11) may later
need Wisteria or an A100-class GPU, depending on measured memory and runtime.
Nothing about that has been measured or decided yet.

## 9. Model choice (decided)

**`Qwen/Qwen3-14B` in non-thinking mode**, provided the preflight finds it
complete in a cache on Chomusuke02. It is ungated on the Hub, its thinking
switch is explicit (`enable_thinking=False`), it fits one A6000, and it is not a
Llama derivative, so it differs from the primary evaluated family.

Gemma 3 12B Instruct was considered as a fallback. It is gated, and the plan
lists Gemma 3 as a replication family. It is not configured, and would need a
separate decision.

The generator shares a family with the optional Qwen replication candidate. If
a Qwen model is later evaluated, that is disclosed as a limitation
(`design_notes.md`, *Corpus construction*).

**Decoding is lightly sampled**: temperature 0.3, top_p 0.8, max_tokens 700,
n 1, seed 20260914. That is cooler than Qwen3's own non-thinking suggestion
(temperature 0.7). `top_k`, `min_p` and the three penalties are sent at neutral
values. The server runs with `--generation-config vllm`, so the model's
`generation_config.json` supplies nothing (`design_notes.md`, *No hidden
sampling defaults*). Greedy decoding was proposed and not adopted. The seed is
recorded but does not guarantee reproduction across GPUs, drivers or library
versions. The saved raw response, hashes and runtime record are the record.

## 10. Decisions made

| Question | Decision (2026-09-14) |
|---|---|
| Backend | vLLM OpenAI-compatible endpoint only; the in-process Transformers backend is deferred unless vLLM proves unavailable |
| Model | `Qwen/Qwen3-14B`, non-thinking, if already cached on Chomusuke02 |
| Decoding | sampled: temperature 0.3, top_p 0.8, max_tokens 700, seed recorded |
| Weights | never downloaded without separate approval. If the preflight fails, ask the lab about an existing shared model cache first |

---

## 11. The synthetic id used by the smoke test

The curator asked for `fixture_001` rather than `energy_01`, because `energy_01`
is a real pilot decision. That was right, and the collision was worse than it
looked: the synthetic *brief* in `data/fixtures/topics.yaml` carried the id
`energy_01`, the same id as a curated pilot brief.

The bare `fixture_001` cannot be a brief id — a topic id must begin with its
domain, which the loader enforces — so both the synthetic brief and the
synthetic corpus record now use **`energy_fixture_001`**, with the other two
fixture briefs renamed to `climate_fixture_001` and `technology_fixture_001`.
The word "fixture" is in the id, so a smoke-test log line can never be mistaken
for pilot material, and the pilot ids are no longer shadowed. A test asserts the
smoke test names only the fixture id.

## 12. Commands, as implemented

```bash
# 1. read-only cache check: resolves the revision, downloads nothing
HF_HUB_OFFLINE=1 HF_HOME=/data/$USER/hf_cache \
uv run python scripts/preflight_model.py --config configs/experiment.yaml

# 2. start the server on ONE explicit GPU, bound to localhost, offline
cd /data/$USER/reasonstyle && GPU=0 MODEL=Qwen/Qwen3-14B PORT=8011 \
  bash scripts/server/serve_vllm.sh

# 3. the one call: synthetic fixture, one group, no retry
cd /data/$USER/reasonstyle && \
REASONSTYLE_ALLOW_LOCAL_GENERATION=1 HF_HUB_OFFLINE=1 HF_HOME=/data/$USER/hf_cache \
uv run python scripts/smoke_test.py --config configs/experiment.yaml --send
```

Without `--send`, step 3 prints the request, the resolved revision and the exact
payload and stops. With `--send` but without the environment variable it also
stops, naming what is missing.
