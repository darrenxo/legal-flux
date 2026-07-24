# legal_pilot Package

Main modules:

- `__main__.py`: CLI commands for `flux-*` workflows.
- `legal_flux.py`: template-pool loading, retrieval, hashing, and job construction.
- `legal_flux_runner.py`: direct/structured/RF-style generation runner.
- `legal_flux_setup.py`: LegalHK split preparation, template import, and freeze checks.
- `legal_flux_chatgpt.py`: local packet generation for API/manual template-pool construction.
- `legal_flux_gemini.py`: Vertex AI Gemini template-pool generation through ADC.
- `legal_flux_evaluation.py`: scoring and aggregate metrics.
- `runner.py`: direct and structured baseline calls shared by the LegalFlux runner.
- `legalhk_data.py`, `legalhk_selection.py`, `adaptive_profiles.py`: LegalHK normalization, filtering, and split/profile utilities.
- `clients.py`, `embeddings.py`, `io_utils.py`, `ledger.py`, `prompting.py`, `scoring.py`, `models.py`: shared infrastructure.
