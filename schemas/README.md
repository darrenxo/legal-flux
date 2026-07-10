# Schemas

JSON schemas constrain local model outputs.

- `direct_analysis*.json`: direct baseline answer schema.
- `final_analysis*.json`: structured baseline answer schema.
- `legal_flux_abstract_plan.json`: planner output with abstract steps.
- `legal_flux_step_artifact.json`: executor intermediate artifact output.
- `legal_flux_rf_review.json`: reviewer output for continue/revise/final decisions.
- `legal_flux_rf_final_review.json`: forced final-review schema.
- `legal_flux_template.json`: reusable template-pool record schema.

The `*_binary` schemas force final decisions to `support` or `reject`.
