#!/usr/bin/env bash
#SBATCH --job-name=legalflux-rereview
#SBATCH --account=bfua-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --array=0-7%4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --time=06:00:00
#SBATCH --output=/projects/bfua/ychen129/legal_nlp/logs/legalflux-rereview-%A_%a.out
#SBATCH --error=/projects/bfua/ychen129/legal_nlp/logs/legalflux-rereview-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

: "${LEGAL_FLUX_SOURCE_RUN_TAG:?Set LEGAL_FLUX_SOURCE_RUN_TAG to the completed source run.}"
: "${LEGAL_FLUX_RUN_TAG:?Set LEGAL_FLUX_RUN_TAG to a new replay run tag.}"

export LEGAL_FLUX_PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_ROOT="${LEGAL_FLUX_ROOT:-${LEGAL_FLUX_PROJECT_ROOT}/repo}"
export HF_HOME="${HF_HOME:-${LEGAL_FLUX_WORK_ROOT}/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export LEGAL_FLUX_MODEL_NAME="${LEGAL_FLUX_MODEL_NAME:-Qwen/Qwen3.5-9B}"
export LEGAL_FLUX_MODEL_CONCURRENCY="${LEGAL_FLUX_MODEL_CONCURRENCY:-4}"

NUM_SHARDS="${LEGAL_FLUX_NUM_SHARDS:-8}"
RUN_MODE="${LEGAL_FLUX_REVIEW_MODE:-base}"
CASE_LIMIT="${LEGAL_FLUX_CASE_LIMIT:-}"

if (( SLURM_ARRAY_TASK_ID >= NUM_SHARDS )); then
  echo "Shard index ${SLURM_ARRAY_TASK_ID} is outside ${NUM_SHARDS} shards." >&2
  exit 1
fi
if [[ ! "$LEGAL_FLUX_SOURCE_RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid LEGAL_FLUX_SOURCE_RUN_TAG: ${LEGAL_FLUX_SOURCE_RUN_TAG}" >&2
  exit 1
fi
if [[ ! "$LEGAL_FLUX_RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid LEGAL_FLUX_RUN_TAG: ${LEGAL_FLUX_RUN_TAG}" >&2
  exit 1
fi
if [[ "$LEGAL_FLUX_SOURCE_RUN_TAG" == "$LEGAL_FLUX_RUN_TAG" ]]; then
  echo "LEGAL_FLUX_RUN_TAG must differ from the source run tag." >&2
  exit 1
fi
if [[ -n "$CASE_LIMIT" ]] && [[ ! "$CASE_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "LEGAL_FLUX_CASE_LIMIT must be a positive integer: ${CASE_LIMIT}" >&2
  exit 1
fi

cd "$LEGAL_FLUX_ROOT"
module reset
source "${LEGAL_FLUX_WORK_ROOT}/envs/legalflux-eval-v3/bin/activate"
source scripts/cluster/vllm_server.sh
mkdir -p "$HF_HOME" "${LEGAL_FLUX_PROJECT_ROOT}/logs"

VLLM_ARGS=()
case "$RUN_MODE" in
  base)
    unset LEGAL_FLUX_REVIEWER_MODEL
    ;;
  sft)
    : "${SFT_LR:?Set SFT_LR for an SFT reviewer replay.}"
    : "${SFT_EPOCH:?Set SFT_EPOCH for an SFT reviewer replay.}"
    TRAIN_DIR="${LEGAL_FLUX_WORK_ROOT}/runs/legal_flux/training/template_structure_sft/lr-${SFT_LR}"
    mapfile -t CHECKPOINTS < <(
      find "$TRAIN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V
    )
    CHECKPOINT="${CHECKPOINTS[$((SFT_EPOCH - 1))]:-}"
    if [[ -z "$CHECKPOINT" ]]; then
      echo "Epoch ${SFT_EPOCH} checkpoint not found under ${TRAIN_DIR}." >&2
      exit 1
    fi
    SERVING_CHECKPOINT="${CHECKPOINT}/vllm_text_only"
    if [[ ! -f "${SERVING_CHECKPOINT}/adapter_config.json" ]] || \
       [[ ! -f "${SERVING_CHECKPOINT}/adapter_model.safetensors" ]]; then
      echo "Prepared text-only adapter not found under ${SERVING_CHECKPOINT}." >&2
      exit 1
    fi
    ADAPTER_NAME="legalflux-${LEGAL_FLUX_SOURCE_RUN_TAG}"
    export LEGAL_FLUX_REVIEWER_MODEL="$ADAPTER_NAME"
    VLLM_ARGS=(
      --enforce-eager
      --enable-lora
      --max-lora-rank 32
      --lora-modules "${ADAPTER_NAME}=${SERVING_CHECKPOINT}"
    )
    ;;
  *)
    echo "LEGAL_FLUX_REVIEW_MODE must be base or sft: ${RUN_MODE}" >&2
    exit 1
    ;;
esac

TARGET_BASE="${LEGAL_FLUX_WORK_ROOT}/runs/legal_flux/trajectory_dev/experiments/${LEGAL_FLUX_RUN_TAG}"
LEDGER_DIR="${TARGET_BASE}/shards"
LEDGER_DIR+="/shard-$(printf '%05d' "$SLURM_ARRAY_TASK_ID")"
LEDGER_DIR+="-of-$(printf '%05d' "$NUM_SHARDS")"
LEDGER="${LEDGER_DIR}/generations.jsonl"
if [[ -s "$LEDGER" ]] && [[ "$(tail -c 1 "$LEDGER" | wc -l)" -eq 0 ]]; then
  printf '\n' >> "$LEDGER"
fi

PORT=$((20000 + (SLURM_ARRAY_JOB_ID * 16 + SLURM_ARRAY_TASK_ID) % 30000))
export LEGAL_FLUX_MODEL_BASE_URL="http://127.0.0.1:${PORT}/v1"
VLLM_LOG="${LEGAL_FLUX_PROJECT_ROOT}/logs/vllm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

legal_flux_start_vllm "$PORT" "$VLLM_LOG" "${VLLM_ARGS[@]}"
SERVER_PID="$LEGAL_FLUX_VLLM_PID"
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
legal_flux_wait_for_vllm "$SERVER_PID" "$LEGAL_FLUX_MODEL_BASE_URL" "$VLLM_LOG"

CASE_LIMIT_ARGS=()
if [[ -n "$CASE_LIMIT" ]]; then
  CASE_LIMIT_ARGS=(--case-limit "$CASE_LIMIT")
fi

echo "Replaying ${RUN_MODE} final reviews from ${LEGAL_FLUX_SOURCE_RUN_TAG}, shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS}."
python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-rereview \
  --phase trajectory-dev \
  --source-run-tag "$LEGAL_FLUX_SOURCE_RUN_TAG" \
  --run-tag "$LEGAL_FLUX_RUN_TAG" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  "${CASE_LIMIT_ARGS[@]}" \
  --fail-on-errors
