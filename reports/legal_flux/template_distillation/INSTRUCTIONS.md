# LegalFlux template-pool distillation

Use `template_source_cases.jsonl` to create a fixed pool of 80-120
high-level legal reasoning templates. The packet contains 1400
template-source cases only. It does not contain final-test cases, trajectory-dev
cases, judgment decisions, or support/reject labels.

Return JSONL, one object per template, matching `legal_flux_template.schema.json`.
Follow the ReasonFlux-style schema:

- `template_id`: stable ID such as `LF001`.
- `template_name`: short name.
- `knowledge_tags`: 2-8 abstract tags.
- `description`: what the template does.
- `application_scenario`: when a planner should select it.
- `reasoning_flow`: ordered high-level steps for applying it.
- `example_application`: abstract example without source-case facts.

Requirements:

- Produce reusable templates, not summaries of individual cases.
- Do not include source case IDs, party names, dates, money amounts, citations,
  F-number references, or final outcome words such as support/reject.
- Cover different case-level trajectories: issue spotting, supplied-rule
  extraction, rule recall, procedural threshold, evidence/burden assessment,
  precedent/analogy handling, defenses/counterarguments, remedy discretion, and
  final issue composition.
- Prefer medium-grained templates that can be sequenced by a planner. Avoid a
  single all-purpose IRAC template and avoid tiny one-sentence micro-actions.
- Return JSONL only, with no prose before or after the records.
