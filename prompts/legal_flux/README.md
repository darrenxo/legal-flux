# LegalFlux Prompts

Active RF-style prompts:

- `rf_plan.txt`: planner prompt. It asks for concise planning analysis followed by abstract steps with names, descriptions, and tags, but not template IDs.
- `instantiate.txt`: executor prompt. It applies one retrieved template to the case and prior artifacts.
- `rf_review.txt`: reviewer prompt. It emits review analysis before an adaptive decision, or only a rationale and binary decision when finalization is forced.

The old fixed/adaptive/finalize prompts were archived outside this repo.
