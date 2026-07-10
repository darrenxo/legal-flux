# ChatGPT LegalFlux Template-Pool Workflow

This folder supports a no-API template-pool construction workflow.

## Pass 1: Candidate templates

For each file in `01_homogeneous_batches` and `02_mixed_contrast_batches`, open a
fresh ChatGPT conversation or continue a clean working conversation. Upload or
paste:

- one batch JSONL file
- `legal_flux_template.schema.json`
- `prompts/01_generate_candidate_templates.md`

Save the returned candidate JSONL files locally.

## Pass 2: Merge and deduplicate

After candidate templates are generated, give ChatGPT:

- all candidate-template JSONL files
- `batch_manifest.json`
- `coverage_summary.json`
- `legal_flux_template.schema.json`
- `prompts/02_merge_deduplicate_templates.md`

Ask for one final JSONL pool of 80-120 templates.

## Pass 3: Coverage audit

Give ChatGPT:

- the final template-pool JSONL
- `batch_manifest.json`
- `coverage_summary.json`
- `prompts/03_coverage_audit_and_gap_fill.md`

Use the audit to revise or add templates, then import the final pool:

```powershell
python -m legal_pilot --config configs\legal_flux.yaml flux-import-templates --input path\to\final_templates.jsonl
```

These batches come only from the `template_source` split. Do not include
trajectory-dev or final-test cases in the ChatGPT template-pool creation step.
