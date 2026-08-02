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

On Delta, use the checked setup script:

```bash
PROJECT_ROOT=/projects/bfua/$USER/legal_nlp
WORK_ROOT=/work/hdd/bfua/$USER/legal_nlp
REPO="$PROJECT_ROOT/repo"

bash "$REPO/scripts/cluster/setup_delta_envs.sh"
```

The job scripts load `pytorch-conda/2.8` before activating the isolated
`legalflux-train-v2` and `legalflux-eval-v3` host environments. Inference runs
inside the pinned `vllm/vllm-openai:v0.18.1` Apptainer image; this prevents pip
from selecting a vLLM build compiled for a different CUDA major version. The
setup script also stages Qwen3.5-9B and BGE-M3 in the shared model cache so
array tasks do not each start a separate download.

To rebuild only the evaluation environment and inference image:

```bash
bash "$REPO/scripts/cluster/setup_delta_eval.sh"
```

## 5. Preflight

The submission script below performs both preflights and stops before `sbatch`
if the checkout, environments, prepared cases, manifest, or pinned inference
image is missing.

## 6. Submit the immediate jobs

The no-training job runs direct, fixed IRAC, and untrained LegalFlux over all
2,755 `trajectory_dev` cases in eight shards. This development split is also
15% of the eligible data. `final_test` is a separate 15% split and remains
sealed until the trained pipeline is selected and frozen.

```bash
bash /projects/bfua/$USER/legal_nlp/repo/scripts/cluster/submit_delta_jobs.sh
```

This first submits a one-GPU vLLM canary. The no-training array has an
`afterok` dependency on that canary, so it starts only after Qwen3.5 serves a
successful chat-completion request. SFT learning-rate task 0 runs independently
as the training canary; tasks 1 and 2 have an `afterok` dependency on task 0.
This limits a new training-runtime failure to one GPU while retaining task 0 as
part of the real checkpoint grid.

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
source "$WORK_ROOT/envs/legalflux-eval-v3/bin/activate"
python -m legal_pilot --config "$PROJECT_ROOT/repo/configs/legal_flux.cluster.yaml" \
  flux-score --phase trajectory-dev
```

The no-training aggregate is written to
`$WORK_ROOT/runs/legal_flux/trajectory_dev/aggregate.csv`. Training checkpoints
are written under
`$WORK_ROOT/runs/legal_flux/training/template_structure_sft/`.

Once SFT finishes, evaluate epochs 2, 4, and 6 for all three learning rates on
the complete 2,755-case development split:

```bash
cd /projects/bfua/$USER/legal_nlp/repo
sbatch scripts/cluster/run_sft_full_grid.slurm
```

This is one 72-task array: 3 learning rates x 3 epochs x 8 data shards, with at
most four GPU tasks running concurrently. After every array task completes,
score all nine configurations and build the checkpoint ranking:

```bash
bash scripts/cluster/score_sft_full_grid.sh
```

The ranking is written under
`$WORK_ROOT/reports/legal_flux/sft_checkpoint_grid.*`. Do not evaluate
`final_test` until the planner, template library, prompts, retrieval
configuration, and executor are frozen.

## Resuming no-training shards

Generation is recorded after each completed condition-level run. Resubmitting
the same shard skips successful records and retries unfinished or failed ones.
The total partition count remains fixed at eight even when only selected array
indices are resubmitted. For example, to resume shards 0 through 3 while shards
4 through 7 are still running:

```bash
cd /projects/bfua/$USER/legal_nlp/repo
sbatch --array=0-3%4 scripts/cluster/run_no_training_eval.slurm
```

After shards 4 through 7 stop, resume only whichever indices did not complete.

If a shard reports `No ledger`, generation never started and there is nothing
to resume. After fixing the server environment, resubmit that entire shard. A
vLLM startup failure now terminates the task promptly and prints the final 200
server-log lines instead of waiting until the Slurm time limit.

For a clean no-training rerun, use:

```bash
bash scripts/cluster/submit_no_training_eval.sh
```

Its canary checks schema-constrained visible JSON, not merely server startup.
Cluster requests disable Qwen's hidden thinking because every LegalFlux role
already emits its reasoning in required JSON fields. Generation tasks use four
concurrent requests per server and exit nonzero if any record fails; successful
ledger records remain resumable.
