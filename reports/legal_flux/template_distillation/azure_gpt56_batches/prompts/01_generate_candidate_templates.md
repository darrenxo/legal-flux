# Task: Generate LegalFlux candidate templates from one batch

You will receive one JSONL batch of LegalHK template-source cases and
`legal_flux_template.schema.json`.

Create 10-18 reusable high-level legal reasoning templates that capture patterns
shared by multiple cases in this batch. The templates are candidate building
blocks for a later global merge pass, not final one-case summaries.

Return JSONL only, one object per template, matching the schema exactly.

Rules:

- Use the ReasonFlux-style fields: template ID, name, tags, description,
  application scenario, reasoning flow, and example application.
- Derive templates from recurring reasoning needs in the cases. Treat the batch
  label only as weak orientation; do not merely restate it.
- Keep the abstraction at a middle legal-reasoning level: not generic cognitive
  labels like deduction, induction, analogy, or verification, and not
  single-case fact patterns. A good template should name a reusable legal
  operation, threshold, evidence assessment, issue-composition move, authority
  use, remedy choice, or domain-specific reasoning pattern that appears across
  multiple cases in the batch.
- Abstract away case-specific details. Do not copy case IDs, party names, dates,
  amounts, citations, F-number references, or final outcome words such as
  support/reject.
- Prefer medium-grained templates that can be sequenced with other templates.
- Name IDs as `CAND_<batch_id>_<nn>`, for example `CAND_homogeneous_001_01`.
- If a pattern is too local to one case, do not create a template for it.
- Keep `reasoning_flow` as ordered operational instructions, not hidden
  chain-of-thought.
- Do not infer, predict, or mention the gold outcome of any source case.
