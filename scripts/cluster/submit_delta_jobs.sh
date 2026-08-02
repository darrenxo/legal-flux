#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
REPO="${LEGAL_FLUX_ROOT:-${PROJECT_ROOT}/repo}"
TRAIN_ENV="${WORK_ROOT}/envs/legalflux-train-v2"
EVAL_ENV="${WORK_ROOT}/envs/legalflux-eval-v3"
VLLM_CONTAINER="${LEGAL_FLUX_VLLM_CONTAINER:-${WORK_ROOT}/containers/vllm-openai-v0.18.1.sif}"
CASES="${WORK_ROOT}/data/processed/legal_flux/cases.jsonl"
MANIFEST="${WORK_ROOT}/data/processed/legal_flux/prepare_manifest.json"

for required in \
  "${REPO}/configs/legal_flux.cluster.yaml" \
  "${TRAIN_ENV}/bin/python" \
  "${EVAL_ENV}/bin/python" \
  "$VLLM_CONTAINER" \
  "$CASES" \
  "$MANIFEST"; do
  if [[ ! -e "$required" ]]; then
    echo "Required path is missing: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${PROJECT_ROOT}/logs" "${WORK_ROOT}/runs/legal_flux"
export LEGAL_FLUX_PROJECT_ROOT="$PROJECT_ROOT"
export LEGAL_FLUX_WORK_ROOT="$WORK_ROOT"
export LEGAL_FLUX_ROOT="$REPO"
export HF_HOME="${WORK_ROOT}/cache/huggingface"

module reset
module load pytorch-conda/2.8

source "${TRAIN_ENV}/bin/activate"
python -m legal_pilot --config "${REPO}/configs/legal_flux.cluster.yaml" \
  flux-train-template-sft --dry-run
deactivate

source "${EVAL_ENV}/bin/activate"
python -m legal_pilot --config "${REPO}/configs/legal_flux.cluster.yaml" \
  flux-generate \
  --phase trajectory-dev \
  --conditions direct structured flux_rf_style \
  --num-shards 8 \
  --shard-index 0 \
  --dry-run
deactivate

cd "$REPO"
CANARY_SUBMISSION="$(sbatch --parsable scripts/cluster/run_vllm_canary.slurm)"
CANARY_JOB="${CANARY_SUBMISSION%%;*}"
BASE_JOB="$(sbatch --parsable --dependency="afterok:${CANARY_JOB}" scripts/cluster/run_no_training_eval.slurm)"
SFT_CANARY_SUBMISSION="$(sbatch --parsable --array=0 scripts/cluster/run_template_sft_grid.slurm)"
SFT_CANARY_JOB="${SFT_CANARY_SUBMISSION%%;*}"
SFT_REST_JOB="$(sbatch --parsable --array=1-2%2 --dependency="afterok:${SFT_CANARY_JOB}" scripts/cluster/run_template_sft_grid.slurm)"

printf 'VLLM_CANARY=%s\n' "$CANARY_JOB"
printf 'NO_TRAINING=%s\n' "$BASE_JOB"
printf 'SFT_LR_5E_5=%s\n' "$SFT_CANARY_JOB"
printf 'SFT_REMAINING=%s\n' "$SFT_REST_JOB"
squeue -u "$USER"
