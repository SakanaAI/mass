#!/usr/bin/env bash
# Run one qwen-code episode and record model requests for SFT.
# Usage: run_episode_hiw.sh <prompt_file> <workspace_dir> <logs_dir> <port> <seed>
#
# Differences from the original (and NOTHING else):
#   - MODEL_ID   env override (default = the base FP8 model, as in the original)
#   - PROXY_PORT env: when set, a byte-faithful logging proxy (api_proxy.py) is
#     started on 127.0.0.1:$PROXY_PORT -> 127.0.0.1:$port for the lifetime of
#     the episode and qwen-code is pointed at it. Every model request/response
#     is appended to <logs_dir>/api_requests.jsonl (gzipped at the end).
#   - HIW_CONDITION env (free text) is recorded in run_status.json.
# The qwen-code settings (sampling params, context window, timeouts, yolo
# mode, stream-json output) are byte-identical to the original episodes.
set -u
set -o pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B-FP8}"
PROXY_PORT="${PROXY_PORT:-}"
HIW_CONDITION="${HIW_CONDITION:-}"
script_dir="$(cd "$(dirname "$0")" && pwd -P)"
PROXY_PY="${PROXY_PY:-python3}"

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <prompt_file> <workspace_dir> <logs_dir> <port> <seed>" >&2
  exit 64
fi

prompt_file="$1"
workspace="$2"
logs_dir="$3"
port="$4"
seed="$5"
endpoint="http://127.0.0.1:${port}/v1"

for command in jq curl qwen; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command is unavailable: $command" >&2
    exit 69
  fi
done
if [[ ! -f "$prompt_file" ]]; then
  echo "Prompt file does not exist: $prompt_file" >&2
  exit 66
fi
if [[ -d "$workspace" ]] && find "$workspace" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to start from a non-empty workspace: $workspace" >&2
  exit 73
fi
if [[ -f "$logs_dir/run_status.json" ]]; then
  echo "Refusing to overwrite an existing episode: $logs_dir" >&2
  exit 73
fi

server_json="$(curl --fail --silent --show-error "$endpoint/models")" || {
  echo "The local vLLM endpoint is unavailable: $endpoint" >&2
  exit 69
}
if ! jq -e --arg model "$MODEL_ID" '.data | any(.id == $model)' <<< "$server_json" >/dev/null; then
  echo "vLLM on port ${port} is reachable but not serving ${MODEL_ID}." >&2
  exit 69
fi

if [[ -n "$PROXY_PORT" ]]; then   # fail fast (exit 69, nothing created) if the proxy port is taken
  if ! python3 - "$PROXY_PORT" <<'PYPORT'
import socket, sys
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError as e:
    raise SystemExit(f"proxy port {sys.argv[1]} is occupied: {e}")
finally:
    s.close()
PYPORT
  then echo "Refusing to start: proxy port ${PROXY_PORT} is already in use" >&2; exit 69; fi
fi
mkdir -p "$workspace" "$logs_dir"
qwen_home="$logs_dir/qwen_home"
mkdir -p "$qwen_home"
cp "$prompt_file" "$logs_dir/prompt.txt"

# ---- optional logging proxy ----
proxy_pid=""
client_endpoint="$endpoint"
api_log=""
if [[ -n "$PROXY_PORT" ]]; then
  api_log="$logs_dir/api_requests.jsonl"
  "$PROXY_PY" "$script_dir/api_proxy.py" --listen "$PROXY_PORT" \
    --upstream "http://127.0.0.1:${port}" --log "$api_log" \
    > "$logs_dir/proxy.log" 2>&1 &
  proxy_pid=$!
  for _ in $(seq 1 50); do
    if curl --fail --silent --max-time 3 "http://127.0.0.1:${PROXY_PORT}/v1/models" >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done
  if ! curl --fail --silent --max-time 3 "http://127.0.0.1:${PROXY_PORT}/v1/models" | grep -q "\"${MODEL_ID}\""; then
    echo "Logging proxy on port ${PROXY_PORT} did not come up (see $logs_dir/proxy.log)" >&2
    kill "$proxy_pid" 2>/dev/null || true
    exit 69
  fi
  client_endpoint="http://127.0.0.1:${PROXY_PORT}/v1"
fi
cleanup_proxy() {
  if [[ -n "$proxy_pid" ]] && kill -0 "$proxy_pid" 2>/dev/null; then
    kill -TERM "$proxy_pid" 2>/dev/null || true
    wait "$proxy_pid" 2>/dev/null || true
  fi
}
trap cleanup_proxy EXIT

cat > "$qwen_home/settings.json" <<SETTINGS
{
  "\$version": 4,
  "env": {
    "VLLM_API_KEY": "EMPTY"
  },
  "modelProviders": {
    "openai": [
      {
        "id": "${MODEL_ID}",
        "envKey": "VLLM_API_KEY",
        "baseUrl": "${client_endpoint}",
        "generationConfig": {
          "timeout": 1800000,
          "maxRetries": 0,
          "contextWindowSize": 262144,
          "extra_body": {
            "chat_template_kwargs": {
              "enable_thinking": true
            }
          },
          "samplingParams": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0,
            "repetition_penalty": 1,
            "max_tokens": 32768,
            "seed": ${seed}
          }
        }
      }
    ]
  },
  "security": {
    "auth": {
      "selectedType": "openai"
    }
  },
  "model": {
    "name": "${MODEL_ID}"
  },
  "tools": {
    "shell": {
      "defaultTimeoutMs": 600000
    }
  },
  "telemetry": {
    "enabled": false
  }
}
SETTINGS

stream_file="$logs_dir/qwen_stream.jsonl"
stderr_file="$logs_dir/qwen_stderr.log"
status_file="$logs_dir/run_status.json"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
qwen_version="$(qwen --version)"

set +e
(
  cd "$workspace" || exit 70
  env \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" \
    NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}" OPENMM_CPU_THREADS="${OPENMM_CPU_THREADS:-2}" LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-4}" \
    QWEN_STREAM_IDLE_TIMEOUT_MS="${QWEN_STREAM_IDLE_TIMEOUT_MS:-600000}" \
    CUDA_VISIBLE_DEVICES="${EPISODE_CUDA_VISIBLE_DEVICES-0}" \
    QWEN_HOME="$qwen_home" \
    VLLM_API_KEY=EMPTY \
    QWEN_TELEMETRY_ENABLED=false \
    QWEN_CODE_SUPPRESS_YOLO_WARNING=1 \
    NO_PROXY=127.0.0.1,localhost \
    qwen \
      --model "$MODEL_ID" \
      --approval-mode yolo \
      --output-format stream-json \
      < "$logs_dir/prompt.txt"
) > "$stream_file" 2> "$stderr_file"
qwen_exit_code=$?
set -e

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
trace_has_success=false
if jq -e 'select(.type == "result" and .subtype == "success" and (.is_error == false))' \
  "$stream_file" >/dev/null 2>&1; then
  trace_has_success=true
fi
# qwen-code reports subtype=success even when its final assistant turn is an API-error string (e.g. the endpoint
# died under it: "connect ECONNREFUSED", or a request timeout). Those are infrastructure failures, not model
# outputs: mark them failed (subtype api_error_infra) so drivers re-run them instead of judging a broken workspace.
infra_error="$(python3 - "$stream_file" <<'PYEOF'
import json, sys
last = None
with open(sys.argv[1], errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "assistant":
            last = e
txt = ""
if last:
    c = (last.get("message") or {}).get("content")
    txt = c if isinstance(c, str) else " ".join(b.get("text", "") for b in (c or []) if isinstance(b, dict))
txt = txt.strip()
infra = txt.startswith("[API Error") and any(k in txt for k in ("Connection error", "ECONNREFUSED", "Request timeout", "fetch failed", "ECONNRESET", "socket hang up", "No stream activity", "terminated", "Service Unavailable", "502", "503"))
print(txt[:300] if infra else "")
PYEOF
)" || infra_error=""

# On error results, capture WHY: the final result event's subtype/message and
# the last tool calls before the halt (loop-detection evidence for the
# harness optimizer).
failure_json="$(python3 - "$stream_file" <<'PYEOF'
import json, sys
events = []
with open(sys.argv[1], errors="replace") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
result = next((e for e in reversed(events) if e.get("type") == "result"), None)
if result is None or not result.get("is_error"):
    print("null")
    raise SystemExit(0)
error = result.get("error") or {}
calls = []
for e in reversed(events):
    if e.get("type") != "assistant":
        continue
    for c in reversed((e.get("message") or {}).get("content") or []):
        if isinstance(c, dict) and c.get("type") == "tool_use":
            args = json.dumps(c.get("input") or {}, ensure_ascii=False)
            calls.append({"tool": c.get("name"), "args": args[:200]})
    if len(calls) >= 5:
        break
message = error.get("message") if isinstance(error, dict) else str(error)
print(json.dumps({
    "subtype": result.get("subtype"),
    "message": (message or "")[:1500] or None,
    "last_tool_calls_most_recent_first": calls[:5],
}, ensure_ascii=False))
PYEOF
)" || failure_json="null"
if [[ -n "$infra_error" ]]; then
  trace_has_success=false
  failure_json="$(jq --null-input --arg m "$infra_error" '{subtype: "api_error_infra", message: $m, last_tool_calls_most_recent_first: []}')"
fi

# Stop the proxy first so its log is complete, then compress the log.
cleanup_proxy
trap - EXIT
api_log_final=""
api_requests=0
if [[ -n "$api_log" && -f "$api_log" ]]; then
  api_requests="$(wc -l < "$api_log")"
  gzip -6 -f "$api_log"
  api_log_final="${api_log}.gz"
fi

jq --null-input \
  --arg started_at "$started_at" \
  --arg finished_at "$finished_at" \
  --arg model "$MODEL_ID" \
  --arg endpoint "$endpoint" \
  --arg client_endpoint "$client_endpoint" \
  --arg api_log "$api_log_final" \
  --arg hiw_condition "$HIW_CONDITION" \
  --arg env_policy "OMP=${OMP_NUM_THREADS:-2},MKL=${MKL_NUM_THREADS:-2},OPENBLAS=${OPENBLAS_NUM_THREADS:-2},NUMEXPR=${NUMEXPR_NUM_THREADS:-2},OPENMM=${OPENMM_CPU_THREADS:-2},LOKY=${LOKY_MAX_CPU_COUNT:-4},STREAM_IDLE_MS=${QWEN_STREAM_IDLE_TIMEOUT_MS:-600000},EPISODE_GPU=${EPISODE_CUDA_VISIBLE_DEVICES-0}" \
  --arg prompt_source "$prompt_file" \
  --arg qwen_code_version "$qwen_version" \
  --argjson api_requests "$api_requests" \
  --argjson seed "$seed" \
  --argjson exit_code "$qwen_exit_code" \
  --argjson trace_has_success "$trace_has_success" \
  --argjson failure "$failure_json" \
  '{
    started_at: $started_at,
    finished_at: $finished_at,
    model: $model,
    endpoint: $endpoint,
    seed: $seed,
    qwen_code_version: $qwen_code_version,
    exit_code: $exit_code,
    trace_has_success_result: $trace_has_success,
    failure: $failure,
    execution_mode: "headless_host_visible_yolo",
    operational_limits: {
      http_response_timeout_ms: 1800000,
      foreground_shell_timeout_ms: 600000,
      context_window_tokens: 262144,
      max_tokens_per_model_response: 32768
    },
    hiw: {
      condition: $hiw_condition,
      prompt_source: $prompt_source,
      client_endpoint: $client_endpoint,
      api_log: $api_log,
      api_requests: $api_requests,
      mass: true,
      env_policy: $env_policy
    }
  }' > "$status_file"

echo "Episode finished with exit code ${qwen_exit_code} (success_result=${trace_has_success})"
echo "Workspace: $workspace"
echo "Status:    $status_file"
exit "$qwen_exit_code"
