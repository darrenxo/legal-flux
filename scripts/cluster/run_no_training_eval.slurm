#!/usr/bin/env bash
#SBATCH --job-name=legalflux-dev
#SBATCH --account=bfua-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --array=0-7%4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --time=08:00:00
#SBATCH --output=/projects/bfua/ychen129/legal_nlp/logs/legalflux-dev-%A_%a.out
#SBATCH --error=/projects/bfua/ychen129/legal_nlp/logs/legalflux-dev-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

export LEGAL_FLUX_PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_ROOT="${LEGAL_FLUX_ROOT:-${LEGAL_FLUX_PROJECT_ROOT}/repo}"
export HF_HOME="${HF_HOME:-${LEGAL_FLUX_WORK_ROOT}/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
cd "$LEGAL_FLUX_ROOT"
module reset
module load pytorch-conda/2.8
source "${LEGAL_FLUX_WORK_ROOT}/envs/legalflux-eval-v3/bin/activate"
source scripts/cluster/vllm_server.sh

mkdir -p "$HF_HOME" "${LEGAL_FLUX_WORK_ROOT}/runs/legal_flux"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
BASE_MODEL="${LEGAL_FLUX_BASE_MODEL:-Qwen/Qwen3.5-9B}"
export LEGAL_FLUX_MODEL_NAME="$BASE_MODEL"
export LEGAL_FLUX_MODEL_CONCURRENCY="${LEGAL_FLUX_MODEL_CONCURRENCY:-4}"
export LEGAL_FLUX_RETRIEVAL_DEVICE="${LEGAL_FLUX_RETRIEVAL_DEVICE:-cpu}"
# A no-training run must not inherit adapter role names from the submit shell.
export LEGAL_FLUX_PLANNER_MODEL="$BASE_MODEL"
export LEGAL_FLUX_EXECUTOR_MODEL="$BASE_MODEL"
export LEGAL_FLUX_REVIEWER_MODEL="$BASE_MODEL"
unset LEGAL_FLUX_SOURCE_CHECKPOINT
NUM_SHARDS="${LEGAL_FLUX_NUM_SHARDS:-8}"
RUN_TAG="${LEGAL_FLUX_RUN_TAG:-}"
CASE_LIMIT="${LEGAL_FLUX_CASE_LIMIT:-}"
CONDITION_ARGS=()
RUN_TAG_ARGS=()
CASE_LIMIT_ARGS=()

if [[ -n "${LEGAL_FLUX_CONDITIONS:-}" ]]; then
  read -r -a REQUESTED_CONDITIONS <<< "$LEGAL_FLUX_CONDITIONS"
  CONDITION_ARGS=(--conditions "${REQUESTED_CONDITIONS[@]}")
fi

if [[ -n "$RUN_TAG" ]]; then
  if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid LEGAL_FLUX_RUN_TAG: ${RUN_TAG}" >&2
    exit 1
  fi
  RUN_TAG_ARGS=(--run-tag "$RUN_TAG")
fi

if [[ -n "$CASE_LIMIT" ]]; then
  if [[ ! "$CASE_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "LEGAL_FLUX_CASE_LIMIT must be a positive integer: ${CASE_LIMIT}" >&2
    exit 1
  fi
  CASE_LIMIT_ARGS=(--case-limit "$CASE_LIMIT")
fi

if (( SLURM_ARRAY_TASK_ID >= NUM_SHARDS )); then
  echo "Shard index ${SLURM_ARRAY_TASK_ID} is outside ${NUM_SHARDS} shards." >&2
  exit 1
fi

LEDGER_BASE="${LEGAL_FLUX_WORK_ROOT}/runs/legal_flux/trajectory_dev"
if [[ -n "$RUN_TAG" ]]; then
  LEDGER_BASE+="/experiments/${RUN_TAG}"
fi
LEDGER_DIR="${LEDGER_BASE}/shards"
LEDGER_DIR+="/shard-$(printf '%05d' "$SLURM_ARRAY_TASK_ID")"
LEDGER_DIR+="-of-$(printf '%05d' "$NUM_SHARDS")"
LEDGER="${LEDGER_DIR}/generations.jsonl"
if [[ -s "$LEDGER" ]] && [[ "$(tail -c 1 "$LEDGER" | wc -l)" -eq 0 ]]; then
  printf '\n' >> "$LEDGER"
fi

PORT=$((20000 + (SLURM_ARRAY_JOB_ID * 16 + SLURM_ARRAY_TASK_ID) % 30000))
export LEGAL_FLUX_MODEL_BASE_URL="http://127.0.0.1:${PORT}/v1"
VLLM_LOG="${LEGAL_FLUX_PROJECT_ROOT}/logs/vllm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

legal_flux_start_vllm "$PORT" "$VLLM_LOG" --enforce-eager
SERVER_PID="$LEGAL_FLUX_VLLM_PID"
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
legal_flux_wait_for_vllm "$SERVER_PID" "$LEGAL_FLUX_MODEL_BASE_URL" "$VLLM_LOG"
echo "Starting shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}; case_limit=${CASE_LIMIT:-all}."

python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-generate \
  --phase trajectory-dev \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  "${CONDITION_ARGS[@]}" \
  "${RUN_TAG_ARGS[@]}" \
  "${CASE_LIMIT_ARGS[@]}" \
  --fail-on-errors
