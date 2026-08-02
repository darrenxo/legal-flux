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
PAYLOAD=$(printf '{"model":"%s","messages":[{"role":"user","content":"Return the single word ready."}],"temperature":0,"max_tokens":16}' "$LEGAL_FLUX_MODEL_NAME")
curl --silent --show-error --fail --max-time 600 \
  -H 'Content-Type: application/json' \
  --data "$PAYLOAD" \
  "${BASE_URL}/chat/completions" > "$RESPONSE"

if ! grep -q '"choices"' "$RESPONSE"; then
  echo "Canary response did not contain choices:" >&2
  head -c 2000 "$RESPONSE" >&2
  exit 1
fi

echo "vLLM canary succeeded."
head -c 2000 "$RESPONSE"
echo
