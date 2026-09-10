# Gemini 3.1 Pro LegalFlux Template-Pool Workflow

This folder supports the automated template-pool construction workflow for
Gemini 3.1 Pro on Vertex AI. The same artifacts can still be inspected manually before
spending API credit.

## Pass 1: Candidate templates

For each file in `01_semantic_family_batches`, send:

- one batch JSONL file
- `legal_flux_candidate_response.schema.json`
- `prompts/01_generate_candidate_templates.md`

Save the returned candidate JSONL files under the API output folder.

## Pass 2: Merge and deduplicate

After candidate templates are generated, send:

- all candidate-template JSONL files
- `batch_manifest.json`
- `coverage_summary.json`
- `legal_flux_template.schema.json`
- `prompts/02_merge_deduplicate_templates.md`

Ask for one globally consolidated pool with no required template count.

## Pass 3: Coverage audit

Audit every original batch against the consolidated pool. Gap candidates are
globally adjudicated before a final pool is written. Then import the final pool:

```powershell
python -m legal_pilot --config configs\legal_flux.yaml flux-import-templates --input path\to\final_templates.jsonl
```

These batches come only from the `template_source` split. Do not include
planner-train, trajectory-dev, or final-test cases in the template-pool creation
step.
