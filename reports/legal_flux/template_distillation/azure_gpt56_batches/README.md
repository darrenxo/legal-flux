# Azure GPT-5.6 LegalFlux Template-Pool Workflow

This folder supports the automated template-pool construction workflow for
Azure GPT-5.6 Sol. The same artifacts can still be inspected manually before
spending API credit.

## Pass 1: Candidate templates

For each file in `01_homogeneous_batches` and `02_mixed_contrast_batches`, send:

- one batch JSONL file
- `legal_flux_template.schema.json`
- `prompts/01_generate_candidate_templates.md`

Save the returned candidate JSONL files under the API output folder.

## Pass 2: Merge and deduplicate

After candidate templates are generated, send:

- all candidate-template JSONL files
- `batch_manifest.json`
- `coverage_summary.json`
- `legal_flux_template.schema.json`
- `prompts/02_merge_deduplicate_templates.md`

Ask for one final JSONL pool of 200-300 templates.

## Pass 3: Coverage audit

Send:

- the final template-pool JSONL
- `batch_manifest.json`
- `coverage_summary.json`
- `prompts/03_coverage_audit_and_gap_fill.md`

Use the audit to revise or add templates, then import the final pool:

```powershell
python -m legal_pilot --config configs\legal_flux.yaml flux-import-templates --input path\to\final_templates.jsonl
```

These batches come only from the `template_source` split. Do not include
planner-train, trajectory-dev, or final-test cases in the template-pool creation
step.
