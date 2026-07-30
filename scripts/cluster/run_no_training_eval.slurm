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
#SBATCH --time=48:00:00
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
source "${LEGAL_FLUX_WORK_ROOT}/envs/legalflux-eval-v2/bin/activate"

mkdir -p "$HF_HOME" "${LEGAL_FLUX_WORK_ROOT}/runs/legal_flux"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export LEGAL_FLUX_MODEL_NAME="${LEGAL_FLUX_MODEL_NAME:-Qwen/Qwen3.5-9B}"
export LEGAL_FLUX_MODEL_CONCURRENCY="${LEGAL_FLUX_MODEL_CONCURRENCY:-8}"
export LEGAL_FLUX_RETRIEVAL_DEVICE="${LEGAL_FLUX_RETRIEVAL_DEVICE:-cpu}"

PORT=$((8000 + SLURM_ARRAY_TASK_ID))
export LEGAL_FLUX_MODEL_BASE_URL="http://127.0.0.1:${PORT}/v1"

vllm serve "$LEGAL_FLUX_MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --language-model-only \
  > "${LEGAL_FLUX_PROJECT_ROOT}/logs/vllm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

until curl --silent --fail "${LEGAL_FLUX_MODEL_BASE_URL}/models" >/dev/null; do
  sleep 10
done

python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-generate \
  --phase trajectory-dev \
  --num-shards "$SLURM_ARRAY_TASK_COUNT" \
  --shard-index "$SLURM_ARRAY_TASK_ID"
