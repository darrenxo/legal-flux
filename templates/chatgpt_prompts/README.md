# LegalFlux Template Prompts

These prompts document the template-pool workflow used for manual inspection and
API-backed generation:

- `01_generate_candidate_templates.md`: produce reusable candidate templates from one batch.
- `02_merge_deduplicate_templates.md`: merge candidate files into a final pool.
- `03_coverage_audit_and_gap_fill.md`: audit coverage and propose gap-fill templates.

The generated batch case packets are not tracked in this folder because they
contain LegalHK case text. Regenerate the current Azure/API-ready artifacts
locally with `flux-export-template-batches`. The older
`flux-export-chatgpt-batches` command remains available as a compatibility
alias.
