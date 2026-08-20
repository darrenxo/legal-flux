#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
REPO="${LEGAL_FLUX_ROOT:-${PROJECT_ROOT}/repo}"
cd "$REPO"

NUM_SHARDS="${LEGAL_BENCHMARK_NUM_SHARDS:-16}"
MAX_PARALLEL="${LEGAL_BENCHMARK_MAX_PARALLEL_SHARDS:-4}"
if [[ ! "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "LEGAL_BENCHMARK_NUM_SHARDS must be a positive integer." >&2
  exit 1
fi
if [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "LEGAL_BENCHMARK_MAX_PARALLEL_SHARDS must be a positive integer." >&2
  exit 1
fi
if (( MAX_PARALLEL > NUM_SHARDS )); then
  MAX_PARALLEL="$NUM_SHARDS"
fi
ARRAY_SPEC="0-$((NUM_SHARDS - 1))%${MAX_PARALLEL}"

CANARY_SUBMISSION="$(sbatch --parsable scripts/cluster/run_vllm_canary.slurm)"
CANARY_JOB="${CANARY_SUBMISSION%%;*}"
BENCHMARK_SUBMISSION="$(
  sbatch --parsable \
    --array="$ARRAY_SPEC" \
    --dependency="afterok:${CANARY_JOB}" \
    scripts/cluster/run_legal_benchmarks.slurm
)"
BENCHMARK_JOB="${BENCHMARK_SUBMISSION%%;*}"

printf 'STRUCTURED_CANARY=%s\n' "$CANARY_JOB"
printf 'LEGAL_BENCHMARKS=%s\n' "$BENCHMARK_JOB"
printf 'SUBSET=%s\n' "${LEGAL_BENCHMARK_SUBSET:-pilot}"
printf 'RUN_TAG=%s\n' "${LEGAL_BENCHMARK_RUN_TAG:-delta-bf16-pilot-v1}"
printf 'DATASETS=%s\n' "${LEGAL_BENCHMARK_DATASETS:-annocaselaw realistic_ljp_facts}"
printf 'CONDITIONS=%s\n' "${LEGAL_BENCHMARK_CONDITIONS:-direct structured}"
printf 'ARRAY=%s\n' "$ARRAY_SPEC"
squeue -u "$USER"
