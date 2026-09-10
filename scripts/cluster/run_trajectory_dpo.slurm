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

DPO_POLICY_ADAPTER="${LEGAL_FLUX_DPO_SFT_CHECKPOINT}/vllm_text_only"
if [[ ! -f "${DPO_POLICY_ADAPTER}/adapter_config.json" ]] || \
   [[ ! -f "${DPO_POLICY_ADAPTER}/adapter_model.safetensors" ]]; then
  echo "Prepared text-only SFT adapter not found under ${DPO_POLICY_ADAPTER}." >&2
  echo "Run flux-prepare-vllm-adapter on the selected SFT checkpoint first." >&2
  exit 1
fi

OUTPUT_DIR="${LEGAL_FLUX_DPO_OUTPUT_DIR:-${LEGAL_FLUX_WORK_ROOT}/runs/legal_flux/training/trajectory_dpo}"
VALIDATE_ONLY="${LEGAL_FLUX_DPO_VALIDATE_MODEL_LOAD_ONLY:-0}"
if [[ "$VALIDATE_ONLY" != "0" ]] && [[ "$VALIDATE_ONLY" != "1" ]]; then
  echo "LEGAL_FLUX_DPO_VALIDATE_MODEL_LOAD_ONLY must be 0 or 1." >&2
  exit 1
fi
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
import peft
import transformers
import trl
from packaging.version import Version

version = Version(trl.__version__)
if not Version("0.29") <= version < Version("0.30"):
    raise RuntimeError(
        "Trajectory DPO requires TRL >=0.29,<0.30 for its native frozen "
        f"SFT-reference adapter behavior; found TRL {trl.__version__}."
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

if [[ "$VALIDATE_ONLY" == "1" ]]; then
  python -m legal_pilot --config configs/legal_flux.cluster.yaml \
    flux-train-trajectory-dpo \
    --validate-model-load \
    --model-name-or-path "$LEGAL_FLUX_DPO_SFT_CHECKPOINT" \
    --output-dir "$OUTPUT_DIR"
  exit 0
fi

python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-train-trajectory-dpo \
  --model-name-or-path "$LEGAL_FLUX_DPO_SFT_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR"
