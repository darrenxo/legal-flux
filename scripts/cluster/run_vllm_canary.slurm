#!/usr/bin/env bash
#SBATCH --job-name=legalflux-vllm-canary
#SBATCH --account=bfua-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bfua/ychen129/legal_nlp/logs/legalflux-vllm-canary-%j.out
#SBATCH --error=/projects/bfua/ychen129/legal_nlp/logs/legalflux-vllm-canary-%j.err
#SBATCH --export=ALL

set -euo pipefail

export LEGAL_FLUX_PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_ROOT="${LEGAL_FLUX_ROOT:-${LEGAL_FLUX_PROJECT_ROOT}/repo}"
export HF_HOME="${HF_HOME:-${LEGAL_FLUX_WORK_ROOT}/cache/huggingface}"
export LEGAL_FLUX_MODEL_NAME="${LEGAL_FLUX_MODEL_NAME:-Qwen/Qwen3.5-9B}"
cd "$LEGAL_FLUX_ROOT"
module reset

source scripts/cluster/vllm_server.sh
PORT=$((20000 + SLURM_JOB_ID % 30000))
BASE_URL="http://127.0.0.1:${PORT}/v1"
VLLM_LOG="${LEGAL_FLUX_PROJECT_ROOT}/logs/vllm-canary-${SLURM_JOB_ID}.log"

legal_flux_start_vllm "$PORT" "$VLLM_LOG"
SERVER_PID="$LEGAL_FLUX_VLLM_PID"
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
legal_flux_wait_for_vllm "$SERVER_PID" "$BASE_URL" "$VLLM_LOG"

RESPONSE="${TMPDIR:-/tmp}/legalflux-vllm-canary-${SLURM_JOB_ID}.json"
PAYLOAD=$(printf '%s' "{\"model\":\"${LEGAL_FLUX_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Return a JSON object with status set to ready.\"}],\"temperature\":0,\"max_tokens\":128,\"chat_template_kwargs\":{\"enable_thinking\":false},\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"legalflux_canary\",\"schema\":{\"type\":\"object\",\"properties\":{\"status\":{\"type\":\"string\",\"enum\":[\"ready\"]}},\"required\":[\"status\"],\"additionalProperties\":false},\"strict\":true}}}")
curl --silent --show-error --fail --max-time 600 \
  -H 'Content-Type: application/json' \
  --data "$PAYLOAD" \
  "${BASE_URL}/chat/completions" > "$RESPONSE"

python3 - "$RESPONSE" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
choices = payload.get("choices") or []
if not choices:
    raise SystemExit("Canary response did not contain choices.")
message = choices[0].get("message") or {}
content = message.get("content") or ""
if not content.strip():
    reasoning = message.get("reasoning_content") or ""
    raise SystemExit(
        "Canary returned no visible content; "
        f"reasoning_characters={len(reasoning)}."
    )
parsed = json.loads(content)
if parsed != {"status": "ready"}:
    raise SystemExit(f"Unexpected canary content: {parsed!r}")
PY

EXECUTOR_REQUEST="${TMPDIR:-/tmp}/legalflux-executor-request-${SLURM_JOB_ID}.json"
EXECUTOR_RESPONSE="${TMPDIR:-/tmp}/legalflux-executor-response-${SLURM_JOB_ID}.json"
python3 - \
  "$LEGAL_FLUX_ROOT/schemas/legal_flux_step_artifact.json" \
  "$EXECUTOR_REQUEST" \
  "$LEGAL_FLUX_MODEL_NAME" <<'PY'
import json
import sys

schema_path, request_path, model = sys.argv[1:]
schema = json.load(open(schema_path, encoding="utf-8"))
payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": (
                "Execute one intermediate legal-reasoning step. State concisely "
                "in instantiated_result that F1 is material. Do not decide the "
                "overall claim and do not return metadata or other fields."
            ),
        }
    ],
    "temperature": 0,
    "seed": 20260619,
    "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": False},
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "legalflux_executor_canary",
            "schema": schema,
            "strict": True,
        },
    },
}
json.dump(payload, open(request_path, "w", encoding="utf-8"))
PY

curl --silent --show-error --fail --max-time 600 \
  -H 'Content-Type: application/json' \
  --data "@$EXECUTOR_REQUEST" \
  "${BASE_URL}/chat/completions" > "$EXECUTOR_RESPONSE"

python3 - "$EXECUTOR_RESPONSE" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
choices = payload.get("choices") or []
if not choices:
    raise SystemExit("Executor canary response did not contain choices.")
content = (choices[0].get("message") or {}).get("content") or ""
parsed = json.loads(content)
required = {"instantiated_result"}
if set(parsed) != required:
    raise SystemExit(f"Executor canary fields were unexpected: {sorted(parsed)}")
if len(parsed["instantiated_result"]) > 1800:
    raise SystemExit("Executor canary violated instantiated_result maxLength.")
PY

echo "vLLM canary and executor-schema canary succeeded."
head -c 2000 "$RESPONSE"
echo
