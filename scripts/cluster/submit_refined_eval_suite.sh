#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
REPO="${LEGAL_FLUX_ROOT:-${PROJECT_ROOT}/repo}"
VLLM_CONTAINER="${LEGAL_FLUX_VLLM_CONTAINER:-${WORK_ROOT}/containers/vllm-openai-v0.21.0.sif}"
: "${LEGAL_FLUX_SFT_CHECKPOINT:?Set this to the selected original SFT checkpoint}"

if [[ ! -d "$LEGAL_FLUX_SFT_CHECKPOINT" ]]; then
  echo "Selected SFT checkpoint does not exist: ${LEGAL_FLUX_SFT_CHECKPOINT}" >&2
  exit 1
fi
CHECKPOINT="$(readlink -f "$LEGAL_FLUX_SFT_CHECKPOINT")"
SERVING_CHECKPOINT="${CHECKPOINT}/vllm_text_only"
if [[ ! -f "${SERVING_CHECKPOINT}/adapter_config.json" ]] || \
   [[ ! -f "${SERVING_CHECKPOINT}/adapter_model.safetensors" ]]; then
  echo "Prepared text-only adapter not found under ${SERVING_CHECKPOINT}." >&2
  echo "Run flux-prepare-vllm-adapter on the selected checkpoint first." >&2
  exit 1
fi
for required in \
  "${REPO}/configs/legal_flux.cluster.yaml" \
  "${WORK_ROOT}/envs/legalflux-eval-v3/bin/python" \
  "$VLLM_CONTAINER" \
  "${WORK_ROOT}/data/processed/legal_flux/cases.jsonl"; do
  if [[ ! -e "$required" ]]; then
    echo "Required evaluation path is missing: ${required}" >&2
    exit 1
  fi
done

NUM_SHARDS="${LEGAL_FLUX_NUM_SHARDS:-8}"
MAX_PARALLEL="${LEGAL_FLUX_MAX_PARALLEL_SHARDS:-4}"
SUITE_TAG="${LEGAL_FLUX_EVAL_SUITE_TAG:-vllm021-refined-pipeline-v1}"
if [[ ! "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "Shard counts must be positive integers." >&2
  exit 1
fi
if [[ ! "$SUITE_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid LEGAL_FLUX_EVAL_SUITE_TAG: ${SUITE_TAG}" >&2
  exit 1
fi
if (( MAX_PARALLEL > NUM_SHARDS )); then
  MAX_PARALLEL="$NUM_SHARDS"
fi
ARRAY_SPEC="0-$((NUM_SHARDS - 1))%${MAX_PARALLEL}"

BASE_SMOKE_TAG="${SUITE_TAG}-base-smoke"
BASE_FULL_TAG="${SUITE_TAG}-base-full"
SFT_SMOKE_TAG="${SUITE_TAG}-sft-smoke"
SFT_FULL_TAG="${SUITE_TAG}-sft-full"

cd "$REPO"
export LEGAL_FLUX_PROJECT_ROOT="$PROJECT_ROOT"
export LEGAL_FLUX_WORK_ROOT="$WORK_ROOT"
export LEGAL_FLUX_ROOT="$REPO"
export LEGAL_FLUX_BASE_MODEL="${LEGAL_FLUX_BASE_MODEL:-Qwen/Qwen3.5-9B}"
export LEGAL_FLUX_MODEL_NAME="$LEGAL_FLUX_BASE_MODEL"
export LEGAL_FLUX_SFT_CHECKPOINT="$CHECKPOINT"
export LEGAL_FLUX_CONDITIONS="direct structured flux_rf_style"
unset LEGAL_FLUX_CANARY_LORA_PATH LEGAL_FLUX_CANARY_LORA_NAME \
  LEGAL_FLUX_CASE_LIMIT LEGAL_FLUX_RUN_TAG

BASE_CANARY_SUBMISSION="$(
  sbatch --parsable scripts/cluster/run_vllm_canary.slurm
)"
BASE_CANARY_JOB="${BASE_CANARY_SUBMISSION%%;*}"
BASE_SMOKE_SUBMISSION="$(
  sbatch --parsable \
    --array=0 \
    --dependency="afterok:${BASE_CANARY_JOB}" \
    --export="ALL,LEGAL_FLUX_NUM_SHARDS=1,LEGAL_FLUX_CASE_LIMIT=32,LEGAL_FLUX_RUN_TAG=${BASE_SMOKE_TAG}" \
    scripts/cluster/run_no_training_eval.slurm
)"
BASE_SMOKE_JOB="${BASE_SMOKE_SUBMISSION%%;*}"
BASE_FULL_SUBMISSION="$(
  sbatch --parsable \
    --array="$ARRAY_SPEC" \
    --dependency="afterok:${BASE_SMOKE_JOB}" \
    --export="ALL,LEGAL_FLUX_NUM_SHARDS=${NUM_SHARDS},LEGAL_FLUX_RUN_TAG=${BASE_FULL_TAG}" \
    scripts/cluster/run_no_training_eval.slurm
)"
BASE_FULL_JOB="${BASE_FULL_SUBMISSION%%;*}"

SFT_CANARY_NAME="legalflux-refined-sft-canary"
SFT_CANARY_SUBMISSION="$(
  sbatch --parsable \
    --export="ALL,LEGAL_FLUX_CANARY_LORA_PATH=${SERVING_CHECKPOINT},LEGAL_FLUX_CANARY_LORA_NAME=${SFT_CANARY_NAME}" \
    scripts/cluster/run_vllm_canary.slurm
)"
SFT_CANARY_JOB="${SFT_CANARY_SUBMISSION%%;*}"
SFT_SMOKE_SUBMISSION="$(
  sbatch --parsable \
    --array=0 \
    --dependency="afterok:${SFT_CANARY_JOB}" \
    --export="ALL,LEGAL_FLUX_NUM_SHARDS=1,LEGAL_FLUX_CASE_LIMIT=32,LEGAL_FLUX_RUN_TAG=${SFT_SMOKE_TAG}" \
    scripts/cluster/run_sft_finalist_full_dev.slurm
)"
SFT_SMOKE_JOB="${SFT_SMOKE_SUBMISSION%%;*}"
SFT_FULL_SUBMISSION="$(
  sbatch --parsable \
    --array="$ARRAY_SPEC" \
    --dependency="afterok:${SFT_SMOKE_JOB}" \
    --export="ALL,LEGAL_FLUX_NUM_SHARDS=${NUM_SHARDS},LEGAL_FLUX_RUN_TAG=${SFT_FULL_TAG}" \
    scripts/cluster/run_sft_finalist_full_dev.slurm
)"
SFT_FULL_JOB="${SFT_FULL_SUBMISSION%%;*}"

SUBMISSION_DIR="${WORK_ROOT}/runs/legal_flux/submissions"
SUBMISSION_RECORD="${SUBMISSION_DIR}/${SUITE_TAG}.env"
mkdir -p "$SUBMISSION_DIR"
{
  printf 'LEGAL_FLUX_EVAL_SUITE_TAG=%q\n' "$SUITE_TAG"
  printf 'LEGAL_FLUX_SFT_CHECKPOINT=%q\n' "$CHECKPOINT"
  printf 'BASE_CANARY_JOB=%q\n' "$BASE_CANARY_JOB"
  printf 'BASE_SMOKE_JOB=%q\n' "$BASE_SMOKE_JOB"
  printf 'BASE_FULL_JOB=%q\n' "$BASE_FULL_JOB"
  printf 'SFT_CANARY_JOB=%q\n' "$SFT_CANARY_JOB"
  printf 'SFT_SMOKE_JOB=%q\n' "$SFT_SMOKE_JOB"
  printf 'SFT_FULL_JOB=%q\n' "$SFT_FULL_JOB"
  printf 'BASE_FULL_TAG=%q\n' "$BASE_FULL_TAG"
  printf 'SFT_FULL_TAG=%q\n' "$SFT_FULL_TAG"
} > "$SUBMISSION_RECORD"

printf 'BASE_CANARY_JOB=%s\n' "$BASE_CANARY_JOB"
printf 'BASE_SMOKE_JOB=%s TAG=%s\n' "$BASE_SMOKE_JOB" "$BASE_SMOKE_TAG"
printf 'BASE_FULL_JOB=%s TAG=%s\n' "$BASE_FULL_JOB" "$BASE_FULL_TAG"
printf 'SFT_CANARY_JOB=%s\n' "$SFT_CANARY_JOB"
printf 'SFT_SMOKE_JOB=%s TAG=%s\n' "$SFT_SMOKE_JOB" "$SFT_SMOKE_TAG"
printf 'SFT_FULL_JOB=%s TAG=%s\n' "$SFT_FULL_JOB" "$SFT_FULL_TAG"
printf 'ARRAY=%s\n' "$ARRAY_SPEC"
printf 'SUBMISSION_RECORD=%s\n' "$SUBMISSION_RECORD"
squeue -u "$USER"
