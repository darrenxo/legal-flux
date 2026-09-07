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

## Cross-dataset direct and structured benchmark

Prepare the three benchmark datasets locally first. AnnoCaseLaw and
Realistic_LJP_Facts download without authentication. IL-TUR requires that the
Hugging Face account associated with `HF_TOKEN` has accepted the dataset gate:

```powershell
cd "C:\Users\Darrenxo\OneDrive\桌面\RA\Legal_agri\legal_flux"
$env:PYTHONUTF8 = "1"
.\.venv-codex\Scripts\python.exe -m huggingface_hub.cli.hf auth login
.\.venv-codex\Scripts\python.exe -m legal_pilot `
  --config configs\legal_benchmarks.yaml benchmark-prepare `
  --datasets annocaselaw realistic_ljp_facts il_tur_cjpe
```

The preparer accepts either the cached `hf auth login` credential or an
`HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` environment variable and never writes the
token into benchmark outputs.

The relative `python.exe` path above requires the repository root as the
current folder. Calling the CLI as a Python module deliberately avoids stale
Windows console launchers after a virtual environment is moved. Authentication
is stored in the user's Hugging Face cache, so an absolute path to this Python
executable can be invoked from any folder.

Copy only the prepared evaluation artifacts to Delta; the 140 MB raw source
CSV and source repositories are not needed by the GPU jobs:

```powershell
ssh ychen129@login.delta.ncsa.illinois.edu "mkdir -p /work/hdd/bfua/ychen129/legal_nlp/data/processed/legal_benchmarks"
scp -r ".\data\processed\legal_benchmarks\*" `
  ychen129@login.delta.ncsa.illinois.edu:/work/hdd/bfua/ychen129/legal_nlp/data/processed/legal_benchmarks/
```

Alternatively, after all three datasets are prepared, run the checked upload
wrapper from any local folder. It verifies that every prepared dataset is
present before opening the SSH/SCP authentication prompts:

```powershell
& "C:\Users\Darrenxo\OneDrive\桌面\RA\Legal_agri\legal_flux\scripts\cluster\upload_legal_benchmarks.ps1"
```

Submit the default pilot comparison on full-precision BF16 Qwen3.5-9B:

```bash
cd /projects/bfua/$USER/legal_nlp/repo
export LEGAL_BENCHMARK_SUBSET=pilot
export LEGAL_BENCHMARK_RUN_TAG=delta-bf16-pilot-v1
export LEGAL_BENCHMARK_NUM_SHARDS=16
export LEGAL_BENCHMARK_MAX_PARALLEL_SHARDS=4
# The default is the two currently accessible datasets.
export LEGAL_BENCHMARK_DATASETS="annocaselaw realistic_ljp_facts"
bash scripts/cluster/submit_legal_benchmarks.sh
```

After accepting the IL-TUR gate and rerunning `benchmark-prepare`, include all
three datasets under a new run tag:

```bash
export LEGAL_BENCHMARK_RUN_TAG=delta-bf16-three-dataset-pilot-v1
export LEGAL_BENCHMARK_DATASETS="annocaselaw realistic_ljp_facts il_tur_cjpe"
bash scripts/cluster/submit_legal_benchmarks.sh
```

For the full comparison, the dedicated wrapper selects all 394 AnnoCaseLaw
cases, the 1,509-case Realistic_LJP_Facts test split, and IL-TUR CJPE's
1,517-case official test file. This is 3,420 cases and 6,840 paired-condition
generation jobs:

```bash
cd /projects/bfua/$USER/legal_nlp/repo
bash scripts/cluster/submit_legal_benchmarks_full.sh
```

Override `LEGAL_BENCHMARK_RUN_TAG`, `LEGAL_BENCHMARK_DATASETS`, shard count, or
maximum parallel shards before invoking the wrapper when a different isolated
run is needed.

After all array tasks finish, score the paired direct/structured predictions:

```bash
source /work/hdd/bfua/$USER/legal_nlp/envs/legalflux-eval-v3/bin/activate
export LEGAL_FLUX_ROOT=/projects/bfua/$USER/legal_nlp/repo
export LEGAL_BENCHMARK_WORK_ROOT=/work/hdd/bfua/$USER/legal_nlp
python -m legal_pilot --config configs/legal_benchmarks.cluster.yaml \
  benchmark-score --subset pilot --run-tag delta-bf16-pilot-v1
```

The local config records Ollama's Q4_K_M runtime separately from the Delta
vLLM BF16 runtime. Use a new run tag for every runtime and subset so the two
sets of ledgers cannot be mixed.

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
inside the pinned `vllm/vllm-openai:v0.21.0` Apptainer image; this prevents pip
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
CHECKPOINT=/work/hdd/bfua/$USER/legal_nlp/runs/legal_flux/training/template_structure_sft/lr-5e-5/checkpoint-30
sbatch \
  --export=ALL,LEGAL_FLUX_CANARY_LORA_PATH="$CHECKPOINT/vllm_text_only" \
  scripts/cluster/run_vllm_canary.slurm
```

This short canary attaches the prepared SFT adapter and sends one constrained
request to it while retaining the base model for the executor-schema request.
Submit the full grid only after the canary completes successfully:

```bash
export LEGAL_FLUX_SFT_GRID_RUN_TAG_PREFIX=refined-vllm021-sft-grid-v2-
sbatch scripts/cluster/run_sft_full_grid.slurm
```

This is one 72-task array: 3 learning rates x 3 epochs x 8 data shards, with at
most four GPU tasks running concurrently. Use a new run-tag prefix whenever
prompts, schemas, templates, retrieval, models, or inference settings change so
that results are not mixed with a previous grid. After every array task
completes, score all nine configurations and build the checkpoint ranking with
the same exported prefix:

```bash
bash scripts/cluster/score_sft_full_grid.sh
```

The ranking is written under
`$WORK_ROOT/reports/legal_flux/sft_checkpoint_grid.*`. Do not evaluate
`final_test` until the planner, template library, prompts, retrieval
configuration, and executor are frozen.

## Standardized vLLM 0.21 refined-pipeline rerun

To rerun the complete development comparison after selecting the best SFT
checkpoint, use the suite wrapper. It creates four isolated results:

- direct base-Qwen prompting;
- structured-IRAC base-Qwen prompting;
- refined LegalFlux with base Qwen in all three roles;
- refined LegalFlux with the selected SFT adapter as planner/reviewer and base
  Qwen as executor.

Every server verifies that the container reports vLLM 0.21.0, uses
`--enforce-eager`, disables hidden thinking through the shared client config,
and runs against the complete 2,755-case `trajectory_dev` split. Each full run
has an `afterok` dependency on its own 32-case real-pipeline smoke run.

```bash
cd /projects/bfua/$USER/legal_nlp/repo
export LEGAL_FLUX_SFT_CHECKPOINT=/work/hdd/bfua/$USER/legal_nlp/runs/legal_flux/training/template_structure_sft/lr-2e-4/checkpoint-N
export LEGAL_FLUX_EVAL_SUITE_TAG=vllm021-refined-pipeline-v1
export LEGAL_FLUX_NUM_SHARDS=8
export LEGAL_FLUX_MAX_PARALLEL_SHARDS=4
bash scripts/cluster/submit_refined_eval_suite.sh
```

Replace `checkpoint-N` with the exact selected epoch-6 checkpoint. The script
serves its `vllm_text_only` child but records the original checkpoint path for
provenance. Its full run tags are `${LEGAL_FLUX_EVAL_SUITE_TAG}-base-full` and
`${LEGAL_FLUX_EVAL_SUITE_TAG}-sft-full`. Job IDs and tags are also saved in
`$LEGAL_FLUX_WORK_ROOT/runs/legal_flux/submissions/<suite-tag>.env`.

After both full arrays complete, score them from the evaluation environment:

```bash
source /work/hdd/bfua/$USER/legal_nlp/envs/legalflux-eval-v3/bin/activate
python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-score --phase trajectory-dev \
  --run-tag "${LEGAL_FLUX_EVAL_SUITE_TAG}-base-full"
python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-score --phase trajectory-dev \
  --run-tag "${LEGAL_FLUX_EVAL_SUITE_TAG}-sft-full"
```

The base aggregate contains the direct, structured, and no-training LegalFlux
rows. The SFT aggregate contains the trained LegalFlux row.

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

This resume behavior applies only when the workflow hash is unchanged. Changes
to prompts, schemas, generation limits, retrieval code, or other hashed workflow
components produce new hashes and regenerate every condition-level result. Old
ledger rows remain intact, while the current `run_plan.json` keeps scoring scoped
to the new workflow.

After shards 4 through 7 stop, resume only whichever indices did not complete.

If a shard reports `No ledger`, generation never started and there is nothing
to resume. After fixing the server environment, resubmit that entire shard. A
vLLM startup failure now terminates the task promptly and prints the final 200
server-log lines instead of waiting until the Slurm time limit.

For a clean no-training rerun, use:

```bash
bash scripts/cluster/submit_no_training_eval.sh
```

Before another full RF-style run, exercise the real planner, retriever, executor,
and reviewer on 32 cases using one shard. This is distinct from the server/schema
canary and catches realistic output-contract failures cheaply:

```bash
LEGAL_FLUX_CONDITIONS="flux_rf_style" \
LEGAL_FLUX_RUN_TAG="rf-minimal-executor-smoke-v1" \
LEGAL_FLUX_CASE_LIMIT=32 \
LEGAL_FLUX_NUM_SHARDS=1 \
bash scripts/cluster/submit_no_training_eval.sh
```

Score the smoke run with `LEGAL_FLUX_WORK_ROOT` exported. Continue only if all 32
records are valid:

```bash
export LEGAL_FLUX_WORK_ROOT=/work/hdd/bfua/$USER/legal_nlp
python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-score --phase trajectory-dev --run-tag rf-minimal-executor-smoke-v1
```

Then submit the full untrained RF-style condition in a new isolated directory,
preserving the completed direct and structured baselines:

```bash
LEGAL_FLUX_CONDITIONS="flux_rf_style" \
LEGAL_FLUX_RUN_TAG="rf-minimal-executor-full-v1" \
bash scripts/cluster/submit_no_training_eval.sh
```

Its canary checks schema-constrained visible JSON and compiles the active
executor schema, not merely server startup.
Cluster requests disable Qwen's hidden thinking because every LegalFlux role
already emits its reasoning in required JSON fields. Generation tasks use four
concurrent requests per server and exit nonzero if any record fails; successful
ledger records remain resumable.

## DPO trajectory construction and training

Use the original selected SFT checkpoint for training and its prepared
`vllm_text_only` child for generation. A 32-anchor real-pipeline canary is:

```bash
cd /projects/bfua/$USER/legal_nlp/repo
export LEGAL_FLUX_DPO_SFT_CHECKPOINT=/work/hdd/bfua/$USER/legal_nlp/runs/legal_flux/training/template_structure_sft/lr-2e-4/checkpoint-N
export LEGAL_FLUX_DPO_NUM_SHARDS=1
export LEGAL_FLUX_DPO_CASE_LIMIT=32
sbatch --array=0 scripts/cluster/run_dpo_collection.slurm
```

Replace the example checkpoint with the actual selected checkpoint path. After
the canary succeeds, submit the full planner-train collection:

```bash
unset LEGAL_FLUX_DPO_CASE_LIMIT
export LEGAL_FLUX_DPO_NUM_SHARDS=144
sbatch --array=0-143%4 scripts/cluster/run_dpo_collection.slurm
```

Each anchor receives four stochastic plans. The templates are retrieved once
from each anchor plan, and that fixed trajectory is executed on the anchor plus
its two X-sim neighbors. The selected SFT adapter serves as planner. The
unchanged base Qwen model executes every template step, with no intermediate
reviewer calls. After all steps finish, the selected SFT adapter is called once
under the forced-finalization schema to produce `support` or `reject`. Shards
are isolated and resumable.

After all shards finish, export canonical preference pairs and run the DPO
preflight:

```bash
source /work/hdd/bfua/$USER/legal_nlp/envs/legalflux-eval-v3/bin/activate
python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-export-trajectory-dpo

source /work/hdd/bfua/$USER/legal_nlp/envs/legalflux-train-v2/bin/activate
python -m legal_pilot --config configs/legal_flux.cluster.yaml \
  flux-train-trajectory-dpo --dry-run \
  --model-name-or-path "$LEGAL_FLUX_DPO_SFT_CHECKPOINT"
```

Then submit training:

```bash
sbatch --export=ALL,LEGAL_FLUX_DPO_SFT_CHECKPOINT="$LEGAL_FLUX_DPO_SFT_CHECKPOINT" \
  scripts/cluster/run_trajectory_dpo.slurm
```

The DPO trainer loads the selected SFT checkpoint as the trainable `default`
adapter on one base model. With `ref_model=None`, TRL 0.29 copies that initial
adapter to its frozen `ref` adapter before optimization and uses the copy for
reference log-probabilities. The launcher rejects other TRL minor versions,
and the trainer verifies that `default` and `ref` begin identical while only
`default` is trainable.

The DPO checkpoint is trained only to prefer better planner trajectory JSON.
Executor artifacts and final support/reject labels are reward evidence in the
construction ledgers, not assistant targets in the DPO dataset.
