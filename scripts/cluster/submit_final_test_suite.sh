#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
REPO="${LEGAL_FLUX_ROOT:-${PROJECT_ROOT}/repo}"

: "${LEGAL_FLUX_SFT_CHECKPOINT:?Set this to the selected SFT checkpoint}"
: "${LEGAL_FLUX_DPO_CHECKPOINT:?Set this to the trained DPO final adapter}"

resolve_checkpoint() {
  local checkpoint
  checkpoint="$(readlink -f "$1")"
  if [[ ! -d "$checkpoint" ]]; then
    echo "Checkpoint directory does not exist: $1" >&2
    return 1
  fi
  printf '%s\n' "$checkpoint"
}

resolve_serving_adapter() {
  local checkpoint="$1"
  if [[ -f "${checkpoint}/vllm_text_only/adapter_config.json" ]] && \
     [[ -f "${checkpoint}/vllm_text_only/adapter_model.safetensors" ]]; then
    printf '%s\n' "${checkpoint}/vllm_text_only"
    return
  fi
  if [[ -f "${checkpoint}/adapter_config.json" ]] && \
     [[ -f "${checkpoint}/adapter_model.safetensors" ]]; then
    printf '%s\n' "$checkpoint"
    return
  fi
  echo "No serving-ready adapter found at ${checkpoint} or ${checkpoint}/vllm_text_only." >&2
  return 1
}

SFT_CHECKPOINT="$(resolve_checkpoint "$LEGAL_FLUX_SFT_CHECKPOINT")"
DPO_CHECKPOINT="$(resolve_checkpoint "$LEGAL_FLUX_DPO_CHECKPOINT")"
SFT_SERVING="$(resolve_serving_adapter "$SFT_CHECKPOINT")"
DPO_SERVING="$(resolve_serving_adapter "$DPO_CHECKPOINT")"

NUM_SHARDS="${LEGAL_FLUX_NUM_SHARDS:-8}"
MAX_PARALLEL="${LEGAL_FLUX_MAX_PARALLEL_SHARDS:-4}"
SUITE_TAG="${LEGAL_FLUX_FINAL_TEST_SUITE_TAG:-finaltest-vllm021-20260921-v1}"
if [[ ! "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "Shard counts must be positive integers." >&2
  exit 1
fi
if [[ ! "$SUITE_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Invalid LEGAL_FLUX_FINAL_TEST_SUITE_TAG: ${SUITE_TAG}" >&2
  exit 1
fi
if (( MAX_PARALLEL > NUM_SHARDS )); then
  MAX_PARALLEL="$NUM_SHARDS"
fi
ARRAY_SPEC="0-$((NUM_SHARDS - 1))%${MAX_PARALLEL}"

BASE_TAG="${SUITE_TAG}-base"
SFT_TAG="${SUITE_TAG}-sft"
DPO_TAG="${SUITE_TAG}-dpo"

cd "$REPO"
export LEGAL_FLUX_PROJECT_ROOT="$PROJECT_ROOT"
export LEGAL_FLUX_WORK_ROOT="$WORK_ROOT"
export LEGAL_FLUX_ROOT="$REPO"
export LEGAL_FLUX_BASE_MODEL="${LEGAL_FLUX_BASE_MODEL:-Qwen/Qwen3.5-9B}"
export LEGAL_FLUX_MODEL_NAME="$LEGAL_FLUX_BASE_MODEL"
unset LEGAL_FLUX_CASE_LIMIT LEGAL_FLUX_RUN_TAG LEGAL_FLUX_PHASE \
  LEGAL_FLUX_ADAPTER_CHECKPOINT LEGAL_FLUX_SOURCE_CHECKPOINT \
  LEGAL_FLUX_PLANNER_MODEL LEGAL_FLUX_EXECUTOR_MODEL \
  LEGAL_FLUX_REVIEWER_MODEL LEGAL_FLUX_PLANNER_CHECKPOINT \
  LEGAL_FLUX_EXECUTOR_CHECKPOINT LEGAL_FLUX_REVIEWER_CHECKPOINT
export LEGAL_FLUX_CONDITIONS="direct structured flux_rf_style"

EVAL_PYTHON="${WORK_ROOT}/envs/legalflux-eval-v3/bin/python"
if [[ ! -x "$EVAL_PYTHON" ]]; then
  echo "LegalFlux evaluation Python is missing: ${EVAL_PYTHON}" >&2
  exit 1
fi
"$EVAL_PYTHON" -m legal_pilot \
  --config configs/legal_flux.cluster.yaml \
  flux-seal-final-test \
  --sft-checkpoint "$SFT_CHECKPOINT" \
  --dpo-checkpoint "$DPO_CHECKPOINT"

BASE_CANARY_SUBMISSION="$(
  sbatch --parsable scripts/cluster/run_vllm_canary.slurm
)"
BASE_CANARY_JOB="${BASE_CANARY_SUBMISSION%%;*}"
BASE_SUBMISSION="$(
  sbatch --parsable \
    --array="$ARRAY_SPEC" \
    --dependency="afterok:${BASE_CANARY_JOB}" \
    --export="ALL,LEGAL_FLUX_PHASE=final-test,LEGAL_FLUX_NUM_SHARDS=${NUM_SHARDS},LEGAL_FLUX_RUN_TAG=${BASE_TAG}" \
    scripts/cluster/run_no_training_eval.slurm
)"
BASE_JOB="${BASE_SUBMISSION%%;*}"

SFT_CANARY_SUBMISSION="$(
  sbatch --parsable \
    --export="ALL,LEGAL_FLUX_CANARY_LORA_PATH=${SFT_SERVING},LEGAL_FLUX_CANARY_LORA_NAME=legalflux-finaltest-sft-canary" \
    scripts/cluster/run_vllm_canary.slurm
)"
SFT_CANARY_JOB="${SFT_CANARY_SUBMISSION%%;*}"
SFT_SUBMISSION="$(
  sbatch --parsable \
    --array="$ARRAY_SPEC" \
    --dependency="afterok:${SFT_CANARY_JOB}" \
    --export="ALL,LEGAL_FLUX_PHASE=final-test,LEGAL_FLUX_NUM_SHARDS=${NUM_SHARDS},LEGAL_FLUX_ADAPTER_CHECKPOINT=${SFT_CHECKPOINT},LEGAL_FLUX_RUN_TAG=${SFT_TAG}" \
    scripts/cluster/run_sft_finalist_full_dev.slurm
)"
SFT_JOB="${SFT_SUBMISSION%%;*}"

DPO_CANARY_SUBMISSION="$(
  sbatch --parsable \
    --export="ALL,LEGAL_FLUX_CANARY_LORA_PATH=${DPO_SERVING},LEGAL_FLUX_CANARY_LORA_NAME=legalflux-finaltest-dpo-canary" \
    scripts/cluster/run_vllm_canary.slurm
)"
DPO_CANARY_JOB="${DPO_CANARY_SUBMISSION%%;*}"
DPO_SUBMISSION="$(
  sbatch --parsable \
    --array="$ARRAY_SPEC" \
    --dependency="afterok:${DPO_CANARY_JOB}" \
    --export="ALL,LEGAL_FLUX_PHASE=final-test,LEGAL_FLUX_NUM_SHARDS=${NUM_SHARDS},LEGAL_FLUX_ADAPTER_CHECKPOINT=${DPO_CHECKPOINT},LEGAL_FLUX_RUN_TAG=${DPO_TAG}" \
    scripts/cluster/run_sft_finalist_full_dev.slurm
)"
DPO_JOB="${DPO_SUBMISSION%%;*}"

SUBMISSION_DIR="${WORK_ROOT}/runs/legal_flux/submissions"
SUBMISSION_RECORD="${SUBMISSION_DIR}/${SUITE_TAG}.env"
mkdir -p "$SUBMISSION_DIR"
{
  printf 'LEGAL_FLUX_FINAL_TEST_SUITE_TAG=%q\n' "$SUITE_TAG"
  printf 'LEGAL_FLUX_SFT_CHECKPOINT=%q\n' "$SFT_CHECKPOINT"
  printf 'LEGAL_FLUX_DPO_CHECKPOINT=%q\n' "$DPO_CHECKPOINT"
  printf 'BASE_CANARY_JOB=%q\n' "$BASE_CANARY_JOB"
  printf 'BASE_JOB=%q\n' "$BASE_JOB"
  printf 'BASE_TAG=%q\n' "$BASE_TAG"
  printf 'SFT_CANARY_JOB=%q\n' "$SFT_CANARY_JOB"
  printf 'SFT_JOB=%q\n' "$SFT_JOB"
  printf 'SFT_TAG=%q\n' "$SFT_TAG"
  printf 'DPO_CANARY_JOB=%q\n' "$DPO_CANARY_JOB"
  printf 'DPO_JOB=%q\n' "$DPO_JOB"
  printf 'DPO_TAG=%q\n' "$DPO_TAG"
} > "$SUBMISSION_RECORD"

printf 'BASE_CANARY_JOB=%s\n' "$BASE_CANARY_JOB"
printf 'BASE_JOB=%s TAG=%s CONDITIONS=direct,structured,flux_rf_style\n' "$BASE_JOB" "$BASE_TAG"
printf 'SFT_CANARY_JOB=%s\n' "$SFT_CANARY_JOB"
printf 'SFT_JOB=%s TAG=%s ROLES=planner+reviewer\n' "$SFT_JOB" "$SFT_TAG"
printf 'DPO_CANARY_JOB=%s\n' "$DPO_CANARY_JOB"
printf 'DPO_JOB=%s TAG=%s ROLES=planner+reviewer\n' "$DPO_JOB" "$DPO_TAG"
printf 'PHASE=final-test ARRAY=%s\n' "$ARRAY_SPEC"
printf 'SUBMISSION_RECORD=%s\n' "$SUBMISSION_RECORD"
squeue -u "$USER"
