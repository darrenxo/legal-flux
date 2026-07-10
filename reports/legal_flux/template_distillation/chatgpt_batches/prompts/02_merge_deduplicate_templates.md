# Task: Merge LegalFlux candidate templates into the final pool

You will receive candidate-template JSONL files from homogeneous and mixed
contrast batches, plus the batch manifest and coverage summary.

Merge, deduplicate, and normalize the candidates into a final fixed LegalFlux
template pool of 80-120 templates.

Return JSONL only, one object per final template, matching
`legal_flux_template.schema.json`.

Rules:

- Preserve cross-batch patterns by merging near-duplicates instead of keeping
  local aliases.
- Keep distinct templates when they imply genuinely different reasoning
  behavior, not merely different legal topics.
- Ensure coverage for procedural thresholds, supplied-rule extraction, rule
  recall, issue decomposition/composition, evidence and burden assessment,
  defenses/counterarguments, precedent/analogy, remedy discretion, long-fact
  filtering, and domain-specific civil families.
- Use stable IDs `LF001`, `LF002`, ...
- Do not include case IDs, party names, dates, amounts, citations, F-number
  references, or support/reject outcome words.
- Make the pool useful for trajectory planning: each template should be a step
  that can be selected, instantiated, and composed with other steps.
