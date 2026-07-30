#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
REPO="${LEGAL_FLUX_ROOT:-${PROJECT_ROOT}/repo}"
TRAIN_ENV="${WORK_ROOT}/envs/legalflux-train"
EVAL_ENV="${WORK_ROOT}/envs/legalflux-eval"

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
  python -m venv --system-site-packages "$TRAIN_ENV"
fi
source "${TRAIN_ENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${REPO}[train]"
deactivate

if [[ ! -x "${EVAL_ENV}/bin/python" ]]; then
  python -m venv --system-site-packages "$EVAL_ENV"
fi
source "${EVAL_ENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel uv
uv pip install vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly
uv pip install -e "${REPO}[retrieval]"

export HF_HOME="${WORK_ROOT}/cache/huggingface"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-9B')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3')"
deactivate

echo "Delta environments and model cache are ready under ${WORK_ROOT}."
