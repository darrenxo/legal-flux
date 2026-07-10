# Scripts

Current utility scripts:

- `combine_chatgpt_candidate_templates.py`: byte-preserving concatenation of ChatGPT candidate-template JSONL files.
- `build_audited_legal_flux_pool.py`: removes audit-flagged duplicate templates, appends gap-fill templates, and renumbers IDs.
- `analyze_legal_flux_errors.py`: reads trajectory-dev scored outputs and writes RF-style error-analysis artifacts.

Generated CSVs from the analysis script:

- `trajectory_dev_rf_case_deltas.csv`: one row per case with gold label, predictions, retrieved template IDs/names, trajectory length, review count, and shortened rationales.
- `aggregate.csv` under a run directory: one row per condition with record count, accuracy, validity, issue-coverage proxy, calls, latency, and token means.
