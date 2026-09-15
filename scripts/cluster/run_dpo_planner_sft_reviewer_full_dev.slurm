#!/usr/bin/env bash
#SBATCH --job-name=legalflux-dpo-sft
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
#SBATCH --output=/projects/bfua/ychen129/legal_nlp/logs/legalflux-dpo-sft-%A_%a.out
#SBATCH --error=/projects/bfua/ychen129/legal_nlp/logs/legalflux-dpo-sft-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

export LEGAL_FLUX_PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_ROOT="${LEGAL_FLUX_ROOT:-${LEGAL_FLUX_PROJECT_ROOT}/repo}"
export HF_HOME="${HF_HOME:-${LEGAL_FLUX_WORK_ROOT}/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

cd "$LEGAL_FLUX_ROOT"
# Avoid reloading Delta's currently inconsistent default MPI module stack.
module --force purge
source "${LEGAL_FLUX_WORK_ROOT}/envs/legalflux-eval-v3/bin/activate"
source scripts/cluster/vllm_server.sh
mkdir -p "$HF_HOME" "${LEGAL_FLUX_PROJECT_ROOT}/logs" \
  "${LEGAL_FLUX_WORK_ROOT}/runs/legal_flux"

PLANNER_CHECKPOINT="${LEGAL_FLUX_DPO_PLANNER_CHECKPOINT:-}"
REVIEWER_CHECKPOINT="${LEGAL_FLUX_SFT_REVIEWER_CHECKPOINT:-}"
: "${PLANNER_CHECKPOINT:?Set LEGAL_FLUX_DPO_PLANNER_CHECKPOINT to the DPO final adapter}"
: "${REVIEWER_CHECKPOINT:?Set LEGAL_FLUX_SFT_REVIEWER_CHECKPOINT to the selected SFT checkpoint}"

PLANNER_CHECKPOINT="$(readlink -f "$PLANNER_CHECKPOINT")"
REVIEWER_CHECKPOINT="$(readlink -f "$REVIEWER_CHECKPOINT")"

resolve_serving_adapter() {
  local checkpoint="$1"
  local prepared="${checkpoint}/vllm_text_only"

  if [[ -f "${prepared}/adapter_config.json" ]] && \
     [[ -f "${prepared}/adapter_model.safetensors" ]]; then
    printf '%s\n' "$prepared"
    return
  fi
  if [[ -f "${checkpoint}/adapter_config.json" ]] && \
     [[ -f "${checkpoint}/adapter_model.safetensors" ]]; then
    printf '%s\n' "$checkpoint"
    return
  fi

  echo "No serving-ready adapter found at ${checkpoint} or ${prepared}." >&2
  return 1
}

PLANNER_SERVING_CHECKPOINT="$(resolve_serving_adapter "$PLANNER_CHECKPOINT")"
REVIEWER_SERVING_CHECKPOINT="$(resolve_serving_adapter "$REVIEWER_CHECKPOINT")"

RUN_TAG="${LEGAL_FLUX_RUN_TAG:-dpo-planner-sft-reviewer-full-v1}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid LEGAL_FLUX_RUN_TAG: ${RUN_TAG}" >&2
  exit 1
fi

PLANNER_ADAPTER_NAME="${LEGAL_FLUX_DPO_PLANNER_MODEL_NAME:-legalflux-dpo-planner}"
REVIEWER_ADAPTER_NAME="${LEGAL_FLUX_SFT_REVIEWER_MODEL_NAME:-legalflux-sft-reviewer}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
BASE_MODEL="${LEGAL_FLUX_BASE_MODEL:-Qwen/Qwen3.5-9B}"
export LEGAL_FLUX_MODEL_NAME="$BASE_MODEL"
export LEGAL_FLUX_MODEL_CONCURRENCY="${LEGAL_FLUX_MODEL_CONCURRENCY:-4}"
export LEGAL_FLUX_RETRIEVAL_DEVICE="${LEGAL_FLUX_RETRIEVAL_DEVICE:-cpu}"
export LEGAL_FLUX_PLANNER_MODEL="$PLANNER_ADAPTER_NAME"
export LEGAL_FLUX_EXECUTOR_MODEL="$BASE_MODEL"
export LEGAL_FLUX_REVIEWER_MODEL="$REVIEWER_ADAPTER_NAME"
export LEGAL_FLUX_PLANNER_CHECKPOINT="$PLANNER_CHECKPOINT"
export LEGAL_FLUX_REVIEWER_CHECKPOINT="$REVIEWER_CHECKPOINT"
# This legacy single-source field identifies the policy being evaluated. The
# role-specific checkpoint paths are recorded separately in run_plan.json.
export LEGAL_FLUX_SOURCE_CHECKPOINT="$PLANNER_CHECKPOINT"

NUM_SHARDS="${LEGAL_FLUX_NUM_SHARDS:-8}"
CASE_LIMIT="${LEGAL_FLUX_CASE_LIMIT:-}"
CASE_LIMIT_ARGS=()
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

echo "Planner adapter: ${PLANNER_SERVING_CHECKPOINT}"
echo "Executor model: ${BASE_MODEL}"
echo "Reviewer adapter: ${REVIEWER_SERVING_CHECKPOINT}"
echo "Run tag: ${RUN_TAG}"

PORT=$((20000 + (SLURM_ARRAY_JOB_ID * 16 + SLURM_ARRAY_TASK_ID) % 30000))
export LEGAL_FLUX_MODEL_BASE_URL="http://127.0.0.1:${PORT}/v1"

VLLM_LOG="${LEGAL_FLUX_PROJECT_ROOT}/logs/vllm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
legal_flux_start_vllm "$PORT" "$VLLM_LOG" \
  --enforce-eager \
  --enable-lora \
  --max-loras 2 \
  --max-cpu-loras 2 \
  --max-lora-rank 32 \
  --lora-modules \
    "${PLANNER_ADAPTER_NAME}=${PLANNER_SERVING_CHECKPOINT}" \
    "${REVIEWER_ADAPTER_NAME}=${REVIEWER_SERVING_CHECKPOINT}"
SERVER_PID="$LEGAL_FLUX_VLLM_PID"
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
legal_flux_wait_for_vllm "$SERVER_PID" "$LEGAL_FLUX_MODEL_BASE_URL" "$VLLM_LOG"

python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-generate \
  --phase trajectory-dev \
  --conditions flux_rf_style \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --run-tag "$RUN_TAG" \
  "${CASE_LIMIT_ARGS[@]}" \
  --fail-on-errors
