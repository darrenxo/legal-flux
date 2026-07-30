# NCSA Delta Cluster Runs

This workflow uses:

- login: `ychen129@login.delta.ncsa.illinois.edu`
- GPU account: `bfua-delta-gpu`
- batch partition: `gpuA40x4`
- code and logs: `/projects/bfua/$USER/legal_nlp`
- data, environments, caches, checkpoints, and outputs:
  `/work/hdd/bfua/$USER/legal_nlp`

Each array task requests one A40 GPU, 16 CPU cores, and 64 GB RAM. These
settings follow the NCSA Delta one-GPU batch pattern.

## 1. Push the local repository

From Windows:

```powershell
cd "C:\Users\Darrenxo\OneDrive\桌面\RA\Legal_agri\legal_flux"
git status --short
git add .
git commit -m "Add Delta LegalFlux evaluation and SFT jobs"
git push origin main
```

Prepared LegalHK data and generated runs are ignored by Git.

## 2. Connect the Delta checkout

```powershell
ssh ychen129@login.delta.ncsa.illinois.edu
```

On Delta:

```bash
PROJECT_ROOT=/projects/bfua/$USER/legal_nlp
WORK_ROOT=/work/hdd/bfua/$USER/legal_nlp

mkdir -p "$PROJECT_ROOT/logs" "$WORK_ROOT"

if [[ -d "$PROJECT_ROOT/repo/.git" ]]; then
  git -C "$PROJECT_ROOT/repo" pull --ff-only
elif [[ -d "$PROJECT_ROOT/repo" ]] && \
     [[ -n "$(find "$PROJECT_ROOT/repo" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "$PROJECT_ROOT/repo exists and is not an empty Git checkout."
  exit 1
else
  git clone https://github.com/darrenxo/legal-flux.git "$PROJECT_ROOT/repo"
fi
```

If the GitHub repository is private, authenticate with GitHub on Delta before
cloning.

## 3. Copy the prepared cases

From local PowerShell:

```powershell
cd "C:\Users\Darrenxo\OneDrive\桌面\RA\Legal_agri\legal_flux"
ssh ychen129@login.delta.ncsa.illinois.edu "mkdir -p /work/hdd/bfua/ychen129/legal_nlp/data/processed/legal_flux"
scp ".\data\processed\legal_flux\cases.jsonl" ".\data\processed\legal_flux\prepare_manifest.json" ychen129@login.delta.ncsa.illinois.edu:/work/hdd/bfua/ychen129/legal_nlp/data/processed/legal_flux/
```

The 227-template library is tracked in Git.

## 4. Build isolated environments

On Delta:

```bash
PROJECT_ROOT=/projects/bfua/$USER/legal_nlp
WORK_ROOT=/work/hdd/bfua/$USER/legal_nlp
REPO="$PROJECT_ROOT/repo"

module reset
module load pytorch-conda/2.8

python -m venv --system-site-packages "$WORK_ROOT/envs/legalflux-train"
source "$WORK_ROOT/envs/legalflux-train/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "$REPO[train]"
deactivate

python -m venv --system-site-packages "$WORK_ROOT/envs/legalflux-eval"
source "$WORK_ROOT/envs/legalflux-eval/bin/activate"
python -m pip install --upgrade pip setuptools wheel uv
uv pip install vllm --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly
uv pip install -e "$REPO[retrieval]"
deactivate
```

The job scripts load the same `pytorch-conda/2.8` module before activating
these environments.

## 5. Preflight

```bash
PROJECT_ROOT=/projects/bfua/$USER/legal_nlp
WORK_ROOT=/work/hdd/bfua/$USER/legal_nlp
REPO="$PROJECT_ROOT/repo"

mkdir -p "$PROJECT_ROOT/logs" \
  "$WORK_ROOT/cache/huggingface" \
  "$WORK_ROOT/runs/legal_flux" \
  "$WORK_ROOT/reports/legal_flux"

export LEGAL_FLUX_ROOT="$REPO"
export LEGAL_FLUX_WORK_ROOT="$WORK_ROOT"
export HF_HOME="$WORK_ROOT/cache/huggingface"

source "$WORK_ROOT/envs/legalflux-train/bin/activate"
python -m legal_pilot --config "$REPO/configs/legal_flux.cluster.yaml" \
  flux-train-template-sft --dry-run
deactivate

source "$WORK_ROOT/envs/legalflux-eval/bin/activate"
python -m legal_pilot --config "$REPO/configs/legal_flux.cluster.yaml" \
  flux-generate \
  --phase trajectory-dev \
  --conditions direct structured flux_rf_style \
  --case-limit 1 \
  --dry-run
python -m legal_pilot --config "$REPO/configs/legal_flux.cluster.yaml" \
  flux-export-dev-tune --count 256
deactivate
```

Expected SFT preflight: 227 training examples, no template holdout, six epochs,
effective batch size 16, and 90 optimizer steps.

## 6. Submit the immediate jobs

The no-training job runs direct, fixed IRAC, and untrained LegalFlux over all
2,755 `trajectory_dev` cases in eight shards. This development split is also
15% of the eligible data. `final_test` is a separate 15% split and remains
sealed until the trained pipeline is selected and frozen.

```bash
cd /projects/bfua/$USER/legal_nlp/repo

BASE_JOB=$(sbatch --parsable scripts/cluster/run_no_training_eval.slurm)
SFT_JOB=$(sbatch --parsable scripts/cluster/run_template_sft_grid.slurm)

echo "NO_TRAINING=$BASE_JOB"
echo "SFT_GRID=$SFT_JOB"
squeue -u "$USER"
```

The SFT job is an array of three learning rates: `5e-5`, `1e-4`, and `2e-4`.
Each trains for six epochs and retains every epoch checkpoint.

## 7. Inspect and score

```bash
squeue -u "$USER"
tail -f /projects/bfua/$USER/legal_nlp/logs/legalflux-dev-<job>_<task>.out
tail -f /projects/bfua/$USER/legal_nlp/logs/legalflux-sft-grid-<job>_<task>.out
```

After all eight no-training tasks finish:

```bash
PROJECT_ROOT=/projects/bfua/$USER/legal_nlp
WORK_ROOT=/work/hdd/bfua/$USER/legal_nlp
export LEGAL_FLUX_ROOT="$PROJECT_ROOT/repo"
export LEGAL_FLUX_WORK_ROOT="$WORK_ROOT"

module reset
module load pytorch-conda/2.8
source "$WORK_ROOT/envs/legalflux-eval/bin/activate"
python -m legal_pilot --config "$PROJECT_ROOT/repo/configs/legal_flux.cluster.yaml" \
  flux-score --phase trajectory-dev
```

The no-training aggregate is written to
`$WORK_ROOT/runs/legal_flux/trajectory_dev/aggregate.csv`. Training checkpoints
are written under
`$WORK_ROOT/runs/legal_flux/training/template_structure_sft/`.

Once SFT finishes, submit `run_sft_checkpoint_screen.slurm` to screen epochs
2, 4, and 6. Do not evaluate `final_test` until the planner, template library,
prompts, retrieval configuration, and executor are frozen.
