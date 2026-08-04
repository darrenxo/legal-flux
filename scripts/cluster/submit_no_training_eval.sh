#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
REPO="${LEGAL_FLUX_ROOT:-${PROJECT_ROOT}/repo}"
cd "$REPO"

NUM_SHARDS="${LEGAL_FLUX_NUM_SHARDS:-8}"
MAX_PARALLEL="${LEGAL_FLUX_MAX_PARALLEL_SHARDS:-4}"
if [[ ! "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "LEGAL_FLUX_NUM_SHARDS must be a positive integer: ${NUM_SHARDS}" >&2
  exit 1
fi
if [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "LEGAL_FLUX_MAX_PARALLEL_SHARDS must be a positive integer: ${MAX_PARALLEL}" >&2
  exit 1
fi
ARRAY_END=$((NUM_SHARDS - 1))
if (( MAX_PARALLEL > NUM_SHARDS )); then
  MAX_PARALLEL="$NUM_SHARDS"
fi
ARRAY_SPEC="0-${ARRAY_END}%${MAX_PARALLEL}"

CANARY_SUBMISSION="$(sbatch --parsable scripts/cluster/run_vllm_canary.slurm)"
CANARY_JOB="${CANARY_SUBMISSION%%;*}"
EVAL_SUBMISSION="$(
  sbatch --parsable \
    --array="$ARRAY_SPEC" \
    --dependency="afterok:${CANARY_JOB}" \
    scripts/cluster/run_no_training_eval.slurm
)"
EVAL_JOB="${EVAL_SUBMISSION%%;*}"

printf 'STRUCTURED_CANARY=%s\n' "$CANARY_JOB"
printf 'NO_TRAINING=%s\n' "$EVAL_JOB"
printf 'CONDITIONS=%s\n' "${LEGAL_FLUX_CONDITIONS:-all configured conditions}"
printf 'RUN_TAG=%s\n' "${LEGAL_FLUX_RUN_TAG:-default run directory}"
printf 'CASE_LIMIT=%s\n' "${LEGAL_FLUX_CASE_LIMIT:-all cases}"
printf 'ARRAY=%s\n' "$ARRAY_SPEC"
squeue -u "$USER"
