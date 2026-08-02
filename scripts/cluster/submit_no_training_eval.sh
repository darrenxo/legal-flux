#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
REPO="${LEGAL_FLUX_ROOT:-${PROJECT_ROOT}/repo}"
cd "$REPO"

CANARY_SUBMISSION="$(sbatch --parsable scripts/cluster/run_vllm_canary.slurm)"
CANARY_JOB="${CANARY_SUBMISSION%%;*}"
EVAL_SUBMISSION="$(
  sbatch --parsable \
    --dependency="afterok:${CANARY_JOB}" \
    scripts/cluster/run_no_training_eval.slurm
)"
EVAL_JOB="${EVAL_SUBMISSION%%;*}"

printf 'STRUCTURED_CANARY=%s\n' "$CANARY_JOB"
printf 'NO_TRAINING=%s\n' "$EVAL_JOB"
squeue -u "$USER"
