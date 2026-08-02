#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
REPO="${LEGAL_FLUX_ROOT:-${PROJECT_ROOT}/repo}"
TRAIN_ENV="${WORK_ROOT}/envs/legalflux-train-v2"

if [[ ! -f "${REPO}/pyproject.toml" ]]; then
  echo "LegalFlux checkout not found at ${REPO}." >&2
  exit 1
fi

mkdir -p \
  "${PROJECT_ROOT}/logs" \
  "${WORK_ROOT}/cache/huggingface" \
  "${WORK_ROOT}/data/processed/legal_flux" \
  "${WORK_ROOT}/reports/legal_flux" \
  "${WORK_ROOT}/runs/legal_flux"

module reset
module load pytorch-conda/2.8

if [[ ! -x "${TRAIN_ENV}/bin/python" ]]; then
  python -m venv "$TRAIN_ENV"
fi
source "${TRAIN_ENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e "${REPO}[train]"
python -c "import peft, torch, transformers, trl; print('train:', torch.__version__, transformers.__version__, trl.__version__, peft.__version__)"
deactivate

bash "${REPO}/scripts/cluster/setup_delta_eval.sh"

echo "Delta environments and model cache are ready under ${WORK_ROOT}."
