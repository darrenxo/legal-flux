# ChatGPT Template Prompts

These prompts document the no-API workflow used to build the template pool:

- `01_generate_candidate_templates.md`: produce reusable candidate templates from one batch.
- `02_merge_deduplicate_templates.md`: merge candidate files into a final pool.
- `03_coverage_audit_and_gap_fill.md`: audit coverage and propose gap-fill templates.

The generated batch case packets are not tracked because they contain LegalHK
case text. Regenerate them locally with `flux-export-chatgpt-batches`.
