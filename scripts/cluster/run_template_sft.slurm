#!/usr/bin/env bash
#SBATCH --job-name=legalflux-sft
#SBATCH --account=bfua-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --time=12:00:00
#SBATCH --output=/projects/bfua/ychen129/legal_nlp/logs/legalflux-sft-%j.out
#SBATCH --error=/projects/bfua/ychen129/legal_nlp/logs/legalflux-sft-%j.err
#SBATCH --export=ALL

set -euo pipefail

export LEGAL_FLUX_PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_ROOT="${LEGAL_FLUX_ROOT:-${LEGAL_FLUX_PROJECT_ROOT}/repo}"
export HF_HOME="${HF_HOME:-${LEGAL_FLUX_WORK_ROOT}/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$LEGAL_FLUX_ROOT"
module reset
module load pytorch-conda/2.8
source "${LEGAL_FLUX_WORK_ROOT}/envs/legalflux-train-v2/bin/activate"
mkdir -p "$HF_HOME" "${LEGAL_FLUX_WORK_ROOT}/runs/legal_flux"

python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-train-template-sft --dry-run

python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-train-template-sft
