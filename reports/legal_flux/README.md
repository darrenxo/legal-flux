# LegalFlux Reports

Current generated report artifacts:

- `trajectory_dev_rf_error_analysis.md`: Markdown error analysis for the latest RF-style trajectory-dev run.
- `trajectory_dev_rf_case_deltas.csv`: one row per case comparing direct, structured, and RF-style predictions. Columns include case ID, bucket, gold label, condition predictions, lawsuit type, RF trajectory length, review count, calls, retrieved template IDs/names, planned step names, and shortened rationales.
- `template_distillation/`: local working files for template-pool creation. The reusable prompts and final pool are copied to `templates/` for Git tracking.

This folder may contain LegalHK case text in generated packets, so most files here remain ignored.
