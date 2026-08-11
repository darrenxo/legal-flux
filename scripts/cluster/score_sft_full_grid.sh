#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
REPO="${LEGAL_FLUX_ROOT:-${PROJECT_ROOT}/repo}"

export LEGAL_FLUX_PROJECT_ROOT="$PROJECT_ROOT"
export LEGAL_FLUX_WORK_ROOT="$WORK_ROOT"
export LEGAL_FLUX_ROOT="$REPO"

module reset
source "${WORK_ROOT}/envs/legalflux-eval-v3/bin/activate"
cd "$REPO"

for lr in 5e-5 1e-4 2e-4; do
  for epoch in 2 4 6; do
    python -m legal_pilot --config configs/legal_flux.cluster.yaml \
      flux-score \
      --phase trajectory-dev \
      --run-tag "sft-full-lr-${lr}-e${epoch}"
  done
done

python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-summarize-sft-grid \
  --phase trajectory-dev \
  --prefix sft-full-
