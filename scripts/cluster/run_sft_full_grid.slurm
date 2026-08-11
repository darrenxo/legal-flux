#!/usr/bin/env bash
#SBATCH --job-name=legalflux-sft-full
#SBATCH --account=bfua-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --array=0-71%4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --time=48:00:00
#SBATCH --output=/projects/bfua/ychen129/legal_nlp/logs/legalflux-sft-full-%A_%a.out
#SBATCH --error=/projects/bfua/ychen129/legal_nlp/logs/legalflux-sft-full-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

export LEGAL_FLUX_PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_ROOT="${LEGAL_FLUX_ROOT:-${LEGAL_FLUX_PROJECT_ROOT}/repo}"
export HF_HOME="${HF_HOME:-${LEGAL_FLUX_WORK_ROOT}/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
cd "$LEGAL_FLUX_ROOT"
module reset
source "${LEGAL_FLUX_WORK_ROOT}/envs/legalflux-eval-v3/bin/activate"
source scripts/cluster/vllm_server.sh
mkdir -p "$HF_HOME" "${LEGAL_FLUX_WORK_ROOT}/runs/legal_flux"

LEARNING_RATES=(5e-5 1e-4 2e-4)
EPOCHS=(2 4 6)
NUM_SHARDS=8
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SHARDS))
SHARD_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SHARDS))
LR_INDEX=$((CONFIG_INDEX / 3))
EPOCH_INDEX=$((CONFIG_INDEX % 3))
LR="${LEARNING_RATES[$LR_INDEX]:-}"
EPOCH="${EPOCHS[$EPOCH_INDEX]:-}"

if [[ -z "$LR" ]] || [[ -z "$EPOCH" ]]; then
  echo "Task ${SLURM_ARRAY_TASK_ID} does not map to an SFT configuration." >&2
  exit 1
fi

TRAIN_DIR="${LEGAL_FLUX_WORK_ROOT}/runs/legal_flux/training/template_structure_sft/lr-${LR}"
mapfile -t CHECKPOINTS < <(
  find "$TRAIN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V
)
CHECKPOINT="${CHECKPOINTS[$((EPOCH - 1))]:-}"
if [[ -z "$CHECKPOINT" ]]; then
  echo "Epoch ${EPOCH} checkpoint not found under ${TRAIN_DIR}." >&2
  exit 1
fi
SERVING_CHECKPOINT="${CHECKPOINT}/vllm_text_only"
if [[ ! -f "${SERVING_CHECKPOINT}/adapter_config.json" ]] || \
   [[ ! -f "${SERVING_CHECKPOINT}/adapter_model.safetensors" ]]; then
  echo "Prepared text-only adapter not found under ${SERVING_CHECKPOINT}." >&2
  echo "Run flux-prepare-vllm-adapter for the selected checkpoints first." >&2
  exit 1
fi

RUN_TAG="sft-full-lr-${LR}-e${EPOCH}"
ADAPTER_NAME="legalflux-${RUN_TAG}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export LEGAL_FLUX_MODEL_NAME="${LEGAL_FLUX_MODEL_NAME:-Qwen/Qwen3.5-9B}"
export LEGAL_FLUX_MODEL_CONCURRENCY="${LEGAL_FLUX_MODEL_CONCURRENCY:-4}"
export LEGAL_FLUX_RETRIEVAL_DEVICE="${LEGAL_FLUX_RETRIEVAL_DEVICE:-cpu}"
export LEGAL_FLUX_PLANNER_MODEL="$ADAPTER_NAME"
export LEGAL_FLUX_REVIEWER_MODEL="$ADAPTER_NAME"
export LEGAL_FLUX_EXECUTOR_MODEL="$LEGAL_FLUX_MODEL_NAME"

PORT=$((20000 + (SLURM_ARRAY_JOB_ID * 128 + SLURM_ARRAY_TASK_ID) % 30000))
export LEGAL_FLUX_MODEL_BASE_URL="http://127.0.0.1:${PORT}/v1"
VLLM_LOG="${LEGAL_FLUX_PROJECT_ROOT}/logs/vllm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

legal_flux_start_vllm "$PORT" "$VLLM_LOG" \
  --enforce-eager \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules "${ADAPTER_NAME}=${SERVING_CHECKPOINT}"
SERVER_PID="$LEGAL_FLUX_VLLM_PID"
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
legal_flux_wait_for_vllm "$SERVER_PID" "$LEGAL_FLUX_MODEL_BASE_URL" "$VLLM_LOG"

echo "Evaluating ${RUN_TAG}, shard ${SHARD_INDEX}/${NUM_SHARDS}."
python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-generate \
  --phase trajectory-dev \
  --conditions flux_rf_style \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --run-tag "$RUN_TAG" \
  --fail-on-errors
