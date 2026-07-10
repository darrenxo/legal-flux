# LegalFlux

LegalFlux is a pilot implementation of case-level, ReasonFlux-style adaptive
template-trajectory selection for LegalHK civil judgment prediction.

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
- `templates/`: final reusable LegalFlux template pool plus ChatGPT prompt workflow.
- `scripts/`: template-pool utility scripts and RF-style error analysis.
- `tests/`: focused tests for the current pipeline and shared utilities.
- `data/`, `runs/`, `reports/`: local/generated artifacts. Case text, raw data,
  run ledgers, reports, and embedding caches are intentionally not pushed.

## Setup

```powershell
cd legal_case_state_pilot
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

This downloads the LegalHK parquet locally, filters to binary civil examples,
screens obvious outcome leakage in `more_facts`, and writes local splits under
`data/processed/legal_flux/`.

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

To regenerate ChatGPT batch packets locally:

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-export-chatgpt-batches
```

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
