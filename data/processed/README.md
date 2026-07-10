# Processed Data

`flux-prepare` writes prepared local artifacts here.

Common generated files:

- `legal_flux/cases.jsonl`: one JSON object per selected case. Fields include case ID, plaintiff claim, parties, facts, supplied authorities, gold label, reference issues, and metadata.
- `legal_flux/selection_review.jsonl`: outcome-blind review packet with case ID, split, claim, parties, facts, lawsuit type, profile metadata, and screening notes.
- `legal_flux/prepare_manifest.json`: split counts, filter counts, and selection notes.
- `legal_flux/rf_template_embeddings_bge_m3.json`: local BGE embedding cache for template retrieval.

These files are ignored because they contain case text or large generated caches.
