# Schemas

JSON schemas constrain local model outputs.

- `direct_analysis*.json`: direct baseline answer schema.
- `final_analysis*.json`: concise IRAC reasoning followed by the structured baseline decision.
- `legal_flux_abstract_plan.json`: concise planning analysis followed by abstract steps.
- `legal_flux_step_artifact.json`: executor intermediate artifact output.
- `legal_flux_rf_review.json`: review analysis followed by a continue/revise/final decision.
- `legal_flux_rf_final_review.json`: forced final rationale and binary decision only.
- `legal_flux_template.json`: reusable template-pool record schema.

The `*_binary` schemas force final decisions to `support` or `reject`.
