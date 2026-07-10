# LegalFlux Prompts

Active RF-style prompts:

- `rf_plan.txt`: planner prompt. It asks for abstract steps with names, tags, and purposes, but not template IDs.
- `instantiate.txt`: executor prompt. It applies one retrieved template to the case and prior artifacts.
- `rf_review.txt`: reviewer prompt. It decides whether to continue, revise remaining steps, or emit the final binary answer.

The old fixed/adaptive/finalize prompts were archived outside this repo.
