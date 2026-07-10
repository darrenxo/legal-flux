# Runs

Experiment outputs are written here and ignored by Git.

Generated files:

- `run_plan.json`: planned run hashes, phase, model digest, workflow hash, and template pool hash.
- `generations.jsonl`: append-only raw generation ledger. Each row includes case ID, condition, parsed output, selected templates, timing, token counts, and status.
- `scored.jsonl`: generation rows enriched with prediction and scoring metrics.
- `aggregate.csv`: per-condition summary metrics such as `n`, `answer_correct`, validity, calls, latency, and token means.
