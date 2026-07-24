# Task: Audit final LegalFlux template-pool coverage

You will receive the final template-pool JSONL, the batch manifest, and the
coverage summary for the current template-source split.

Check whether the final pool covers the main observed reasoning families,
domains, authorities, issue-composition patterns, and reasoning demands. Then
return a concise audit report with:

1. Covered categories.
2. Under-covered categories.
3. Duplicative templates that should be merged.
4. Up to 20 additional templates if important gaps remain.

If you propose additional templates, return them as JSONL records matching
`legal_flux_template.schema.json` after the audit report.

Do not use this static prompt directly for a paid API run. Use
`flux-export-template-batches`, which writes a run-specific copy with the
current coverage summary embedded.
