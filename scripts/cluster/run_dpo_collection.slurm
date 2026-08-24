#!/usr/bin/env bash
#SBATCH --job-name=legalflux-dpo-data
#SBATCH --account=bfua-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --array=0-143%4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --time=24:00:00
#SBATCH --output=/projects/bfua/ychen129/legal_nlp/logs/legalflux-dpo-data-%A_%a.out
#SBATCH --error=/projects/bfua/ychen129/legal_nlp/logs/legalflux-dpo-data-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

export LEGAL_FLUX_PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_ROOT="${LEGAL_FLUX_ROOT:-${LEGAL_FLUX_PROJECT_ROOT}/repo}"
export HF_HOME="${HF_HOME:-${LEGAL_FLUX_WORK_ROOT}/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
: "${LEGAL_FLUX_DPO_SFT_CHECKPOINT:?Set this to the selected original SFT checkpoint directory}"

NUM_SHARDS="${LEGAL_FLUX_DPO_NUM_SHARDS:-144}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
if (( SHARD_INDEX >= NUM_SHARDS )); then
  echo "Array task ${SHARD_INDEX} is outside ${NUM_SHARDS} DPO shards." >&2
  exit 1
fi

SERVING_CHECKPOINT="${LEGAL_FLUX_DPO_SFT_CHECKPOINT}/vllm_text_only"
if [[ ! -f "${SERVING_CHECKPOINT}/adapter_config.json" ]] || \
   [[ ! -f "${SERVING_CHECKPOINT}/adapter_model.safetensors" ]]; then
  echo "Prepared text-only adapter not found under ${SERVING_CHECKPOINT}." >&2
  echo "Run flux-prepare-vllm-adapter on the selected SFT checkpoint first." >&2
  exit 1
fi

cd "$LEGAL_FLUX_ROOT"
# Avoid reloading Delta's currently inconsistent default MPI module stack.
module --force purge
source "${LEGAL_FLUX_WORK_ROOT}/envs/legalflux-eval-v3/bin/activate"
source scripts/cluster/vllm_server.sh
mkdir -p "$HF_HOME" "${LEGAL_FLUX_PROJECT_ROOT}/logs"

ADAPTER_NAME="${LEGAL_FLUX_DPO_ADAPTER_NAME:-legalflux-dpo-source}"
BASE_MODEL="${LEGAL_FLUX_MODEL_NAME:-Qwen/Qwen3.5-9B}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export LEGAL_FLUX_MODEL_NAME="$BASE_MODEL"
export LEGAL_FLUX_MODEL_CONCURRENCY=1
export LEGAL_FLUX_RETRIEVAL_DEVICE="${LEGAL_FLUX_RETRIEVAL_DEVICE:-cpu}"
export LEGAL_FLUX_PLANNER_MODEL="$ADAPTER_NAME"
# DPO rewards isolate trajectory quality: only planning and forced finalization
# use the selected SFT adapter. Every template step uses the unchanged base model.
export LEGAL_FLUX_EXECUTOR_MODEL="$BASE_MODEL"
export LEGAL_FLUX_REVIEWER_MODEL="$ADAPTER_NAME"

PORT=$((20000 + (SLURM_ARRAY_JOB_ID * 256 + SLURM_ARRAY_TASK_ID) % 30000))
export LEGAL_FLUX_MODEL_BASE_URL="http://127.0.0.1:${PORT}/v1"
VLLM_LOG="${LEGAL_FLUX_PROJECT_ROOT}/logs/vllm-dpo-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

legal_flux_start_vllm "$PORT" "$VLLM_LOG" \
  --enforce-eager \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules "${ADAPTER_NAME}=${SERVING_CHECKPOINT}"
SERVER_PID="$LEGAL_FLUX_VLLM_PID"
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
legal_flux_wait_for_vllm "$SERVER_PID" "$LEGAL_FLUX_MODEL_BASE_URL" "$VLLM_LOG"

CASE_ARGS=()
if [[ -n "${LEGAL_FLUX_DPO_CASE_LIMIT:-}" ]]; then
  CASE_ARGS=(--case-limit "$LEGAL_FLUX_DPO_CASE_LIMIT")
fi

python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-build-dpo-data \
  --stage all \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  "${CASE_ARGS[@]}" \
  --fail-on-errors
