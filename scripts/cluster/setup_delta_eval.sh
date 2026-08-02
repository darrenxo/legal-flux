#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
REPO="${LEGAL_FLUX_ROOT:-${PROJECT_ROOT}/repo}"
EVAL_ENV="${WORK_ROOT}/envs/legalflux-eval-v3"
VLLM_CONTAINER="${LEGAL_FLUX_VLLM_CONTAINER:-${WORK_ROOT}/containers/vllm-openai-v0.18.1.sif}"
VLLM_IMAGE="${LEGAL_FLUX_VLLM_IMAGE:-docker://vllm/vllm-openai:v0.18.1}"

if [[ ! -f "${REPO}/pyproject.toml" ]]; then
  echo "LegalFlux checkout not found at ${REPO}." >&2
  exit 1
fi
if ! command -v apptainer >/dev/null 2>&1; then
  echo "Apptainer is unavailable. Check the Delta software environment." >&2
  exit 1
fi

mkdir -p \
  "${WORK_ROOT}/envs" \
  "${WORK_ROOT}/containers" \
  "${WORK_ROOT}/cache/apptainer" \
  "${WORK_ROOT}/cache/huggingface"

module reset
module load pytorch-conda/2.8

if [[ ! -x "${EVAL_ENV}/bin/python" ]]; then
  python -m venv "$EVAL_ENV"
fi
source "${EVAL_ENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e "${REPO}[retrieval]"
python -c "import sentence_transformers, torch; print('eval host:', torch.__version__, torch.version.cuda, sentence_transformers.__version__)"

export HF_HOME="${WORK_ROOT}/cache/huggingface"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-9B')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3')"
deactivate

if [[ ! -s "$VLLM_CONTAINER" ]]; then
  export APPTAINER_CACHEDIR="${WORK_ROOT}/cache/apptainer"
  echo "Pulling ${VLLM_IMAGE} to ${VLLM_CONTAINER}."
  apptainer pull "$VLLM_CONTAINER" "$VLLM_IMAGE"
fi
apptainer inspect "$VLLM_CONTAINER" >/dev/null

echo "Evaluation host environment: ${EVAL_ENV}"
echo "Pinned vLLM container: ${VLLM_CONTAINER}"
echo "Run the GPU canary before submitting evaluation arrays."
