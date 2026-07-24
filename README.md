# LegalFlux

LegalFlux is an implementation of case-level, ReasonFlux-style adaptive
template-trajectory selection for all-domain structured LegalHK judgment
prediction.

The current experiment compares three conditions:

- `direct`: predict `support` or `reject` from the native case input.
- `structured`: use a fixed IRAC-style legal reasoning structure, then predict.
- `flux_rf_style`: plan abstract reasoning steps, retrieve one high-level legal
  template per step, execute each step, and let a reviewer continue, revise, or
  emit the final binary answer.

Older BoT, semantic/frontier, case-state, `flux_fixed`, `flux_adaptive`, and
`no_review` trials were moved outside this repo to:

```text
../stale_legal_case_state_trials_20260710/
```

## Repository Layout

- `configs/`: active experiment configuration.
- `prompts/`: model-facing prompts for direct, structured, and RF-style LegalFlux.
- `schemas/`: JSON schemas used to constrain model outputs.
- `src/legal_pilot/`: implementation code.
- `templates/`: reusable LegalFlux template pool plus API/manual prompt workflow.
- `scripts/`: template-pool utility scripts and RF-style error analysis.
- `tests/`: focused tests for the current pipeline and shared utilities.
- `data/`, `runs/`, `reports/`: local/generated artifacts. Case text, raw data,
  run ledgers, reports, and embedding caches are intentionally not pushed.

## Setup

```powershell
cd legal_flux
.\.venv-codex\Scripts\python.exe -m pip install -e ".[dev]"
ollama pull qwen3.5:9b
ollama pull bge-m3
```

The default config uses `qwen3.5:9b` for generation and `bge-m3:latest` through
Ollama for ambiguous or fallback template retrieval.

## Data Preparation

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-prepare
```

This downloads the LegalHK parquet locally, keeps structured rows with normalized
binary `support`/`reject` labels, and writes local splits under
`data/processed/legal_flux/`.

Default split roles after eligibility:

- `template_source`: 20%, used only to create/refine the template library.
- `planner_train`: 50%, used to sample trajectories and build DPO pairs.
- `trajectory_dev`: 15%, used for prompt, retrieval, checkpoint, and
  hyperparameter choices.
- `final_test`: 15%, sealed until the pipeline is frozen.

Heuristic family/demand labels may be used for stratified splitting and
diagnostics, but they are not fed to the model in template-source packets or
case prompts.

LegalHK processed text has unclear redistribution status, so prepared case
files and raw parquet files are ignored by Git.

## Template Pool

The committed pool is:

```text
templates/legal_flux_templates_v0.jsonl
```

Template-generation prompts are in:

```text
templates/chatgpt_prompts/
```

To regenerate the current Azure/API-ready batch packets locally:

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-export-template-batches
```

The older command name, `flux-export-chatgpt-batches`, is still accepted as a
compatibility alias.

To generate the same template pool through an API workflow instead of manual
copy/paste, set an API key and run the staged workflow. The current implemented
API client is DeepSeek-compatible:

```powershell
$env:DEEPSEEK_API_KEY = "..."
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-deepseek-templates --stage candidates --dry-run
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-deepseek-templates --stage candidates
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-deepseek-templates --stage merge
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-deepseek-templates --stage audit
```

The default config uses DeepSeek's OpenAI-compatible endpoint with
`deepseek-v4-pro`. `--dry-run` only estimates planned calls and prompt sizes; the
non-dry-run stages spend DeepSeek API credits. Raw responses, parsed candidates,
merged templates, audit notes, and manifests are written under
`reports/legal_flux/template_distillation/deepseek_api/`.

For Google Vertex AI / Gemini with Application Default Credentials, first run
`gcloud auth application-default login` and set the Vertex environment variables.
Then run a dry run and one-batch canary before launching all batches. The default
Gemini config uses `gemini-3.5-flash` with high thinking:

```powershell
$env:GOOGLE_CLOUD_PROJECT = "legalflux-gemini"
$env:GOOGLE_CLOUD_LOCATION = "global"
$env:GOOGLE_GENAI_USE_VERTEXAI = "true"
$env:GOOGLE_APPLICATION_CREDENTIALS = "$env:APPDATA\gcloud\application_default_credentials.json"
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-gemini-templates --stage candidates --dry-run
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-gemini-templates --stage candidates --limit 1
```

After inspecting the first batch output, continue with:

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-gemini-templates --stage candidates
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-gemini-templates --stage merge
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-gemini-templates --stage audit
```

Gemini outputs are written under
`reports/legal_flux/template_distillation/gemini_api/`.

To import a revised final template pool:

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-import-templates --input path\to\templates.jsonl
```

## Running Experiments

Smoke or dry run:

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-smoke --dry-run
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-smoke
```

Trajectory-dev run:

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-generate --phase trajectory-dev --dry-run
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-generate --phase trajectory-dev
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-score --phase trajectory-dev
```

Planner-training trajectory sampling and export:

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-generate --phase planner-train --samples 4
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-score --phase planner-train
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-export-template-sft
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-export-trajectory-dpo
```

`flux-export-template-sft` produces the ReasonFlux-style template-structure
learning data: template name/tags as input, and description/scenario/flow as
output. `flux-export-trajectory-dpo` builds chosen/rejected trajectory pairs
from repeated `planner_train` samples. v1 intentionally does not add trajectory
SFT; that would be a pragmatic warm start rather than the core ReasonFlux-Zero
recipe.

Final-test runs are guarded by `flux-freeze`:

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-freeze
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-generate --phase final-test
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-score --phase final-test
```

## RF-Style Mechanism

For `flux_rf_style`, each case uses fresh model calls:

1. Planner: reads the compact case and template-tag examples, then outputs
   abstract steps with `step_name`, `template_tags`, and `purpose`.
2. Retriever: exact-matches step name/tags against template names/tags. If
   matching is ambiguous or absent, BGE ranks candidates by embedding similarity.
3. Executor: applies the selected template and returns a structured intermediate
   artifact.
4. Reviewer: sees all executed artifacts so far and remaining abstract steps,
   then chooses `continue`, `revise`, or `final_answer`.

The final answer is forced to `support` or `reject`.
