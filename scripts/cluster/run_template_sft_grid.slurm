#!/usr/bin/env bash
#SBATCH --job-name=legalflux-sft-grid
#SBATCH --account=bfua-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --time=12:00:00
#SBATCH --output=/projects/bfua/ychen129/legal_nlp/logs/legalflux-sft-grid-%A_%a.out
#SBATCH --error=/projects/bfua/ychen129/legal_nlp/logs/legalflux-sft-grid-%A_%a.err
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
source "${LEGAL_FLUX_WORK_ROOT}/envs/legalflux-train/bin/activate"
mkdir -p "$HF_HOME" "${LEGAL_FLUX_WORK_ROOT}/runs/legal_flux"

LEARNING_RATES=(5e-5 1e-4 2e-4)
LR="${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="${LEGAL_FLUX_WORK_ROOT}/runs/legal_flux/training/template_structure_sft/lr-${LR}"
if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "${OUTPUT_DIR} is not empty. Move it aside or resume it explicitly." >&2
  exit 1
fi

python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-train-template-sft \
  --dry-run \
  --learning-rate "$LR" \
  --num-train-epochs 6 \
  --output-dir "$OUTPUT_DIR"

python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-train-template-sft \
  --learning-rate "$LR" \
  --num-train-epochs 6 \
  --output-dir "$OUTPUT_DIR"
