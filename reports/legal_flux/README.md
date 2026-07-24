# LegalFlux Reports

Current generated report artifacts:

- `trajectory_dev_rf_error_analysis.md`: Markdown error analysis for the latest RF-style trajectory-dev run.
- `trajectory_dev_rf_case_deltas.csv`: one row per case comparing direct, structured, and RF-style predictions. Columns include case ID, bucket, gold label, condition predictions, lawsuit type, RF trajectory length, review count, calls, retrieved template IDs/names, planned step names, and shortened rationales.
- `legalhk_trajectory_heterogeneity_motivation_memo.md`: curated dataset examples showing that LegalHK court reasoning follows heterogeneous high-level trajectories.
- `legalhk_court_reasoning_motivation_examples.csv`: evidence table produced by the court-reasoning motivation scanner. Columns include case ID, lawsuit type, gold label, claim, issue/law/case counts, matched pattern labels, excerpts, and issues.
- `legalhk_court_reasoning_motivation_examples.md`: auto-generated summary of aggregate pattern counts and candidate examples from the scanner.
- `template_distillation/`: local working files for template-pool creation. The reusable prompts and final pool are copied to `templates/` for Git tracking.

This folder may contain LegalHK case text in generated packets, so most files here remain ignored.
