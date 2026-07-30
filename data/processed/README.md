# Processed Data

`flux-prepare` writes prepared local artifacts here.

Common generated files:

- `legal_flux/cases.jsonl`: one JSON object per selected case. Fields include case ID, plaintiff claim, parties, facts, supplied authorities, gold label, reference issues, and metadata.
- `legal_flux/selection_review.jsonl`: outcome-blind review packet with case ID, split, claim, parties, facts, lawsuit type, profile metadata, and screening notes.
- `legal_flux/prepare_manifest.json`: split counts, filter counts, and selection notes.
- `legal_flux/rf_template_embeddings_bge_m3.json`: local BGE embedding cache for template retrieval.
- `legal_flux/xsim/planner_train_bge_m3_embeddings.npy`: normalized dense case embeddings.
- `legal_flux/xsim/xsim_dense_top50.jsonl`: top-50 BGE-M3 candidates and dense scores for each planner-training anchor.
- `legal_flux/xsim/xsim_neighbors.jsonl`: anchor plus the top two cross-encoder-reranked neighbors.
- `legal_flux/xsim/xsim_manifest.json`: model names, field selection, corpus hash, and retrieval settings.
- `legal_flux/planner_training/trajectory_candidates.jsonl`: four SFT-planner trajectories and their retrieved template IDs per anchor.
- `legal_flux/planner_training/trajectory_evaluations.jsonl`: fixed-trajectory predictions on each three-case `X_sim` set.
- `legal_flux/planner_training/trajectory_dpo.jsonl`: chosen/rejected planner trajectory preference records.

These files are ignored because they contain case text or large generated caches.
