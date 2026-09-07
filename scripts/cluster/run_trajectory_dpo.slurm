#!/usr/bin/env bash
#SBATCH --job-name=legalflux-dpo-train
#SBATCH --account=bfua-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --time=24:00:00
#SBATCH --output=/projects/bfua/ychen129/legal_nlp/logs/legalflux-dpo-train-%j.out
#SBATCH --error=/projects/bfua/ychen129/legal_nlp/logs/legalflux-dpo-train-%j.err
#SBATCH --export=ALL

set -euo pipefail

export LEGAL_FLUX_PROJECT_ROOT="${LEGAL_FLUX_PROJECT_ROOT:-/projects/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_WORK_ROOT="${LEGAL_FLUX_WORK_ROOT:-/work/hdd/bfua/${USER}/legal_nlp}"
export LEGAL_FLUX_ROOT="${LEGAL_FLUX_ROOT:-${LEGAL_FLUX_PROJECT_ROOT}/repo}"
export HF_HOME="${HF_HOME:-${LEGAL_FLUX_WORK_ROOT}/cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${LEGAL_FLUX_DPO_SFT_CHECKPOINT:?Set this to the selected original SFT checkpoint directory}"

OUTPUT_DIR="${LEGAL_FLUX_DPO_OUTPUT_DIR:-${LEGAL_FLUX_WORK_ROOT}/runs/legal_flux/training/trajectory_dpo}"
if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "${OUTPUT_DIR} is not empty. Move it aside or resume explicitly." >&2
  exit 1
fi

cd "$LEGAL_FLUX_ROOT"
# The project training environment already provides PyTorch and CUDA support;
# avoid Delta's currently inconsistent default MPI module stack.
module --force purge
source "${LEGAL_FLUX_WORK_ROOT}/envs/legalflux-train-v2/bin/activate"
mkdir -p "$HF_HOME" "$OUTPUT_DIR"

python - <<'PY'
import inspect

import peft
import transformers
import trl
from trl import DPOConfig

required = {"model_adapter_name", "ref_adapter_name"}
available = set(inspect.signature(DPOConfig).parameters)
missing = sorted(required - available)
if missing:
    raise RuntimeError(
        "Installed TRL cannot represent the required frozen SFT reference "
        f"adapter; missing DPOConfig fields: {missing}"
    )
print(
    "DPO runtime:",
    f"transformers={transformers.__version__}",
    f"peft={peft.__version__}",
    f"trl={trl.__version__}",
)
PY

python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-train-trajectory-dpo \
  --dry-run \
  --model-name-or-path "$LEGAL_FLUX_DPO_SFT_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR"

python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-train-trajectory-dpo \
  --model-name-or-path "$LEGAL_FLUX_DPO_SFT_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR"
