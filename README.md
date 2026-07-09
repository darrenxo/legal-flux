# LegalHK-Only Case-State Diagnostic Pilot

This package tests whether an explicit typed or validated case state improves
binary LegalHK judgment prediction over direct and ordinary structured
analysis.

The current experiment uses:

- `qwen3.5:9b` through local Ollama;
- five manually reviewed smoke cases;
- 64 separate evaluation cases, balanced 32 support and 32 reject;
- a conservative explicit-outcome leakage screen;
- `gpt-oss:20b` as the no-cost independent audit model.

It never requests or stores unrestricted hidden chain-of-thought.

## ReasonFlux-style LegalFlux

The current LegalFlux direction uses a ReasonFlux-style case-level trajectory:
the planner writes abstract step names and retrieval tags, Python retrieves one
template per step from the template pool, the executor instantiates the selected
template, and the reviewer either continues, revises the remaining abstract
steps, or emits the final `support`/`reject` answer. The planner is not shown the
full template catalog.

The default `configs/legal_flux.yaml` compares:

- `direct`
- `structured`
- `flux_rf_style`

`flux_rf_style` uses local `bge-m3:latest` embeddings through Ollama for
ambiguous or fallback template retrieval, with exact normalized tag/name matches
checked first.

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-generate --phase trajectory-dev --dry-run
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-generate --phase trajectory-dev
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_flux.yaml flux-score --phase trajectory-dev
```

## Legal Buffer-of-Thought extension

The package now also contains a separate Legal-BoT direction-finding workflow.
It leaves the completed case-state run untouched and uses the same 64 LegalHK
evaluation cases as:

- a balanced 32-case online adaptation stream; and
- a disjoint balanced 32-case frozen-buffer holdout.

The six conditions are Direct, full Legal-BoT, no problem-distiller, no
meta-buffer, no buffer-manager, and dynamic growth from generic-only
initialization. For every adaptation case, prediction and scoring occur before
any correctness-gated memory update. Gold outcomes never enter model prompts or
stored thought-templates.

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot.yaml bot-smoke --dry-run
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot.yaml bot-smoke
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot.yaml bot-score --smoke
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot.yaml bot-freeze
.\scripts\run_bot_main.ps1 -DryRun
.\scripts\run_bot_main.ps1
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot.yaml bot-score
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot.yaml bot-report
```

Expected volume is 30 smoke records and 384 main records. Dynamic conditions
use two calls per case plus a third call only when the post-prediction update
gate admits a new or merged template. Runs append JSONL records, replay recorded
buffer events on resume, and ignore events produced by older workflow hashes.

For the optional no-cost independent audit, the existing commands work with the
BoT configuration after scoring:

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot.yaml select-audit
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot.yaml audit-local
```

## Semantic append-only BoT

The next diagnostic replaces TF-IDF with local BGE-M3 embeddings served by
Ollama and removes template merging. A correct adaptation solution may produce
a candidate template; the candidate is appended unchanged only when its maximum
semantic similarity to the active buffer is below the frozen novelty threshold.
Otherwise the manager records a redundant-candidate rejection.

```powershell
# One-time model installation (already completed on this machine)
ollama pull bge-m3

.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot_semantic.yaml bot-embedding-check
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot_semantic.yaml bot-smoke --dry-run
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot_semantic.yaml bot-smoke
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot_semantic.yaml bot-score --smoke
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot_semantic.yaml bot-freeze
.\scripts\run_semantic_main.ps1
```

The semantic run compares Direct, Qwen-distilled fixed-buffer reasoning,
Qwen-distilled append-only updates, and raw-case semantic routing. BGE-M3
embeddings are cached under `data/processed/legalhk_semantic/`.

## Blinded frontier-distiller packet

Generate the outcome-blind packet and process it in a fresh Temporary Chat or
new Codex thread that has not seen experiment labels:

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot_frontier.yaml bot-export-frontier
```

The command writes the input JSONL, strict output schema, and instructions to
`reports/legal_bot_frontier/frontier_distillation/`. After saving the returned
profiles as JSONL:

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot_frontier.yaml bot-import-frontier --input PATH_TO_PROFILES.jsonl
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot_frontier.yaml bot-smoke
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legal_bot_frontier.yaml bot-freeze
.\scripts\run_frontier_main.ps1
```

Import validation rejects outcome fields, missing cases, duplicate case IDs,
and nonexistent F-number references.

## Environment

The original `.venv` points to a Python installation that is no longer present.
The currently verified environment is `.venv-codex`:

```powershell
cd legal_case_state_pilot
.\.venv-codex\Scripts\python.exe -m pip install -e ".[dev]"
```

Ollama must have `qwen3.5:9b` installed and use the NVIDIA GPU.

## Preparation and smoke test

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legalhk_only.yaml env-check
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legalhk_only.yaml prepare
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legalhk_only.yaml smoke --dry-run
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legalhk_only.yaml smoke
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legalhk_only.yaml score --smoke
```

Expected smoke volume: 20 unique generation hashes from five cases under Direct,
Structured, Typed, and Validated conditions.

The selected inputs are written to
`data/processed/legalhk_only/selection_review.jsonl` without outcome labels or
judgment text. The smoke IDs were condition-blind manually reviewed. The
evaluation split remains marked `evaluation_review_status: pending` until it is
reviewed before Phase 2 freeze.

## Main run

After reviewing smoke outputs and the evaluation inputs:

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legalhk_only.yaml freeze
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legalhk_only.yaml generate --dry-run
.\scripts\run_main.ps1
```

Expected main volume:

- 256 principal-condition hashes;
- 24 Oracle hashes;
- 36 sampling-control hashes;
- 316 unique hashes and approximately 444–508 Ollama calls.

The guarded launcher prevents automatic Windows system sleep while generation
is active. It cannot survive shutdown, restart, forced sleep, or a laptop-lid
action configured to sleep.

## Evaluation

```powershell
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legalhk_only.yaml score
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legalhk_only.yaml select-audit
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legalhk_only.yaml audit-local
.\.venv-codex\Scripts\python.exe -m legal_pilot --config configs\legalhk_only.yaml report
```

Primary metrics are intention-to-treat accuracy, macro-F1, class recall,
failure rate, paired bootstrap differences, and exact McNemar tests.

## Data limitations

- LegalHK’s `more_facts` field was enhanced from judgments. Screening removes
  explicit holdings and evaluative conclusions but cannot eliminate latent
  outcome conditioning.
- The processed dataset’s license is unclear. Do not redistribute case text.
- Sixty-four cases are diagnostic, not sufficient to precisely establish a
  small five-point improvement.
- Existing mixed OpenExempt/LegalHK artifacts remain untouched in their
  original directories.
