#!/usr/bin/env bash
#SBATCH --job-name=legal-bench
#SBATCH --account=bfua-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --array=0-15%4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --time=12:00:00
#SBATCH --output=/projects/bfua/ychen129/legal_nlp/logs/legal-bench-%A_%a.out
#SBATCH --error=/projects/bfua/ychen129/legal_nlp/logs/legal-bench-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

export LEGAL_FLUX_PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
export LEGAL_BENCHMARK_WORK_ROOT="${LEGAL_BENCHMARK_WORK_ROOT:-${LEGAL_FLUX_WORK_ROOT}}"
export LEGAL_FLUX_ROOT="${LEGAL_FLUX_ROOT:-${LEGAL_FLUX_PROJECT_ROOT}/repo}"
export HF_HOME="${HF_HOME:-${LEGAL_FLUX_WORK_ROOT}/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

cd "$LEGAL_FLUX_ROOT"
module reset
module load pytorch-conda/2.8
source "${LEGAL_FLUX_WORK_ROOT}/envs/legalflux-eval-v3/bin/activate"
source scripts/cluster/vllm_server.sh

mkdir -p "$HF_HOME" "${LEGAL_FLUX_PROJECT_ROOT}/logs"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
BASE_MODEL="${LEGAL_FLUX_BASE_MODEL:-Qwen/Qwen3.5-9B}"
export LEGAL_FLUX_MODEL_NAME="$BASE_MODEL"
export LEGAL_FLUX_MODEL_CONCURRENCY="${LEGAL_FLUX_MODEL_CONCURRENCY:-4}"

NUM_SHARDS="${LEGAL_BENCHMARK_NUM_SHARDS:-16}"
SUBSET="${LEGAL_BENCHMARK_SUBSET:-pilot}"
RUN_TAG="${LEGAL_BENCHMARK_RUN_TAG:-delta-bf16-pilot-v1}"
CASE_LIMIT="${LEGAL_BENCHMARK_CASE_LIMIT:-}"
DATASETS="${LEGAL_BENCHMARK_DATASETS:-annocaselaw realistic_ljp_facts}"
CONDITIONS="${LEGAL_BENCHMARK_CONDITIONS:-direct structured}"

if [[ ! "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "LEGAL_BENCHMARK_NUM_SHARDS must be a positive integer." >&2
  exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= NUM_SHARDS )); then
  echo "Shard ${SLURM_ARRAY_TASK_ID} is outside ${NUM_SHARDS} shards." >&2
  exit 1
fi
if [[ "$SUBSET" != "pilot" && "$SUBSET" != "full" ]]; then
  echo "LEGAL_BENCHMARK_SUBSET must be pilot or full." >&2
  exit 1
fi
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid LEGAL_BENCHMARK_RUN_TAG: ${RUN_TAG}" >&2
  exit 1
fi

read -r -a DATASET_ARGS <<< "$DATASETS"
read -r -a CONDITION_ARGS <<< "$CONDITIONS"
for dataset in "${DATASET_ARGS[@]}"; do
  cases="${LEGAL_BENCHMARK_WORK_ROOT}/data/processed/legal_benchmarks/${dataset}/cases.jsonl"
  pilot="${LEGAL_BENCHMARK_WORK_ROOT}/data/processed/legal_benchmarks/${dataset}/pilot_case_ids.json"
  if [[ ! -s "$cases" || ! -s "$pilot" ]]; then
    echo "Prepared benchmark data missing for ${dataset}: ${cases}" >&2
    exit 1
  fi
done

CASE_LIMIT_ARGS=()
if [[ -n "$CASE_LIMIT" ]]; then
  if [[ ! "$CASE_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "LEGAL_BENCHMARK_CASE_LIMIT must be a positive integer." >&2
    exit 1
  fi
  CASE_LIMIT_ARGS=(--case-limit "$CASE_LIMIT")
fi

LEDGER_DIR="${LEGAL_BENCHMARK_WORK_ROOT}/runs/legal_benchmarks/${SUBSET}/${RUN_TAG}"
LEDGER_DIR+="/shards/shard-$(printf '%05d' "$SLURM_ARRAY_TASK_ID")"
LEDGER_DIR+="-of-$(printf '%05d' "$NUM_SHARDS")"
LEDGER="${LEDGER_DIR}/generations.jsonl"
if [[ -s "$LEDGER" ]] && [[ "$(tail -c 1 "$LEDGER" | wc -l)" -eq 0 ]]; then
  printf '\n' >> "$LEDGER"
fi

PORT=$((20000 + (SLURM_ARRAY_JOB_ID * 16 + SLURM_ARRAY_TASK_ID) % 30000))
export LEGAL_FLUX_MODEL_BASE_URL="http://127.0.0.1:${PORT}/v1"
VLLM_LOG="${LEGAL_FLUX_PROJECT_ROOT}/logs/legal-bench-vllm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

legal_flux_start_vllm "$PORT" "$VLLM_LOG" --enforce-eager --dtype bfloat16
SERVER_PID="$LEGAL_FLUX_VLLM_PID"
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
legal_flux_wait_for_vllm "$SERVER_PID" "$LEGAL_FLUX_MODEL_BASE_URL" "$VLLM_LOG"

python -m legal_pilot --config configs/legal_benchmarks.cluster.yaml \
  benchmark-generate \
  --datasets "${DATASET_ARGS[@]}" \
  --subset "$SUBSET" \
  --conditions "${CONDITION_ARGS[@]}" \
  --run-tag "$RUN_TAG" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  "${CASE_LIMIT_ARGS[@]}" \
  --fail-on-errors
