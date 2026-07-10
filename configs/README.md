# Configs

`legal_flux.yaml` is the active experiment configuration.

Important fields:

- `model`: local Ollama generation settings.
- `legal_flux.conditions`: active comparison arms, currently `direct`, `structured`, and `flux_rf_style`.
- `legal_flux.template_pool_file`: final template library path.
- `legal_flux.rf_*`: ReasonFlux-style retrieval settings, including the BGE embedding backend.
- `paths`: local data, run, report, prompt, and schema directories.
