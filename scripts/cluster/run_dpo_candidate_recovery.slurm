#!/usr/bin/env bash
#SBATCH --job-name=legalflux-dpo-recover
#SBATCH --account=bfua-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --time=03:00:00
#SBATCH --output=/projects/bfua/ychen129/legal_nlp/logs/legalflux-dpo-recover-%A_%a.out
#SBATCH --error=/projects/bfua/ychen129/legal_nlp/logs/legalflux-dpo-recover-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

export LEGAL_FLUX_PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_ROOT="${LEGAL_FLUX_ROOT:-${LEGAL_FLUX_PROJECT_ROOT}/repo}"
export HF_HOME="${HF_HOME:-${LEGAL_FLUX_WORK_ROOT}/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
: "${LEGAL_FLUX_DPO_SFT_CHECKPOINT:?Set this to the selected original SFT checkpoint directory}"
: "${LEGAL_FLUX_DPO_RECOVERY_SHARDS:?Set this to the explicit comma-separated shard IDs passed to sbatch --array}"
: "${LEGAL_FLUX_DPO_NUM_SHARDS:?Set this to the exact shard count of the existing DPO collection}"

NUM_SHARDS="$LEGAL_FLUX_DPO_NUM_SHARDS"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
if (( SHARD_INDEX >= NUM_SHARDS )); then
  echo "Array task ${SHARD_INDEX} is outside ${NUM_SHARDS} DPO shards." >&2
  exit 1
fi
case ",${LEGAL_FLUX_DPO_RECOVERY_SHARDS}," in
  *",${SHARD_INDEX},"*) ;;
  *)
    echo "Array task ${SHARD_INDEX} is not in the explicit recovery allowlist ${LEGAL_FLUX_DPO_RECOVERY_SHARDS}." >&2
    exit 1
    ;;
esac

SERVING_CHECKPOINT="${LEGAL_FLUX_DPO_SFT_CHECKPOINT}/vllm_text_only"
if [[ ! -f "${SERVING_CHECKPOINT}/adapter_config.json" ]] || \
   [[ ! -f "${SERVING_CHECKPOINT}/adapter_model.safetensors" ]]; then
  echo "Prepared text-only adapter not found under ${SERVING_CHECKPOINT}." >&2
  exit 1
fi

cd "$LEGAL_FLUX_ROOT"
module --force purge
source "${LEGAL_FLUX_WORK_ROOT}/envs/legalflux-eval-v3/bin/activate"
source scripts/cluster/vllm_server.sh
mkdir -p "$HF_HOME" "${LEGAL_FLUX_PROJECT_ROOT}/logs"

SHARD_DIR="${LEGAL_FLUX_WORK_ROOT}/data/processed/legal_flux/planner_training/dpo_shards/shard-$(printf '%05d' "$SHARD_INDEX")-of-$(printf '%05d' "$NUM_SHARDS")"
mkdir -p "$SHARD_DIR"
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to enforce one recovery writer per shard." >&2
  exit 1
fi
exec 9>"${SHARD_DIR}/.dpo-recovery.lock"
if ! flock -n 9; then
  echo "Another recovery process already holds the lock for shard ${SHARD_INDEX}." >&2
  exit 1
fi

ADAPTER_NAME="${LEGAL_FLUX_DPO_ADAPTER_NAME:-legalflux-dpo-source}"
BASE_MODEL="${LEGAL_FLUX_MODEL_NAME:-Qwen/Qwen3.5-9B}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export LEGAL_FLUX_MODEL_NAME="$BASE_MODEL"
export LEGAL_FLUX_MODEL_CONCURRENCY=1
export LEGAL_FLUX_RETRIEVAL_DEVICE="${LEGAL_FLUX_RETRIEVAL_DEVICE:-cpu}"
export LEGAL_FLUX_PLANNER_MODEL="$ADAPTER_NAME"
export LEGAL_FLUX_EXECUTOR_MODEL="$BASE_MODEL"
export LEGAL_FLUX_REVIEWER_MODEL="$ADAPTER_NAME"

PORT="$(python -c 'import socket; sock=socket.socket(); sock.bind(("127.0.0.1", 0)); print(sock.getsockname()[1]); sock.close()')"
export LEGAL_FLUX_MODEL_BASE_URL="http://127.0.0.1:${PORT}/v1"
VLLM_LOG="${LEGAL_FLUX_PROJECT_ROOT}/logs/vllm-dpo-recover-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

legal_flux_start_vllm "$PORT" "$VLLM_LOG" \
  --enforce-eager \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules "${ADAPTER_NAME}=${SERVING_CHECKPOINT}"
SERVER_PID="$LEGAL_FLUX_VLLM_PID"
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
legal_flux_wait_for_vllm "$SERVER_PID" "$LEGAL_FLUX_MODEL_BASE_URL" "$VLLM_LOG"
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "The vLLM process exited during its readiness check." >&2
  tail -n 200 "$VLLM_LOG" >&2 || true
  exit 1
fi

CASE_ARGS=()
if [[ -n "${LEGAL_FLUX_DPO_CASE_LIMIT:-}" ]]; then
  CASE_ARGS=(--case-limit "$LEGAL_FLUX_DPO_CASE_LIMIT")
fi

python -m legal_pilot.legal_flux_dpo_recovery \
  --config configs/legal_flux.cluster.yaml \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --max-recovery-attempts "${LEGAL_FLUX_DPO_MAX_RECOVERY_ATTEMPTS:-1}" \
  "${CASE_ARGS[@]}"
