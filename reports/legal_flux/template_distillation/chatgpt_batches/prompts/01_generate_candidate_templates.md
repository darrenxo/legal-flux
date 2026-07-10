# Task: Generate LegalFlux candidate templates from one batch

You will receive one JSONL batch of LegalHK template-source cases and
`legal_flux_template.schema.json`.

Create 6-12 reusable high-level legal reasoning templates that capture patterns
shared by multiple cases in this batch. These are candidate templates for a
later global merge pass, not final one-case summaries.

Return JSONL only, one object per template, matching the schema exactly.

Rules:

- Use the ReasonFlux-style fields: template ID, name, tags, description,
  application scenario, reasoning flow, and example application.
- Abstract away case-specific details. Do not copy case IDs, party names, dates,
  amounts, citations, F-number references, or final outcome words such as
  support/reject.
- Prefer medium-grained templates that can be sequenced with other templates.
- Name IDs as `CAND_<batch_id>_<nn>`, for example `CAND_homogeneous_001_01`.
- If a pattern is too local to one case, do not create a template for it.
- Keep `reasoning_flow` as ordered operational instructions, not hidden
  chain-of-thought.
