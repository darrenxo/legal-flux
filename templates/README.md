# Templates

Tracked reusable template artifacts:

- `legal_flux_templates_v0.jsonl`: final LegalFlux template pool used by the RF-style retriever.
- `legal_flux_templates_v0.manifest.json`: import manifest for the final pool.
- `chatgpt_prompts/`: reusable prompts for generating, merging, and auditing template candidates.

Each template JSONL row has:

- `template_id`: stable ID such as `LF001`.
- `template_name`: short human-readable name.
- `knowledge_tags`: retrieval tags used for exact candidate matching.
- `description`: what the template does.
- `application_scenario`: when a planner should select it.
- `reasoning_flow`: ordered high-level instructions for the executor.
- `example_application`: abstract example, sanitized to avoid source-case details.
