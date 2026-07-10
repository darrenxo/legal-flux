# GitHub Packaging Notes

This folder is the standalone LegalFlux project. Push this repo from
`legal_case_state_pilot/`, not from the parent `Legal_agri/` workspace.

Safe publication scope:

- project code: `src/`, `configs/`, `prompts/`, `schemas/`, `scripts/`, `tests/`
- reusable templates: `templates/legal_flux_templates_v0.jsonl`
- template workflow prompts: `templates/chatgpt_prompts/`
- project metadata: `README.md`, `GITHUB.md`, `pyproject.toml`, `.gitignore`

Do not publish by default:

- raw LegalHK parquet files
- prepared LegalHK cases
- run ledgers and scored model outputs
- BGE embedding caches
- generated ChatGPT batch case packets
- archived stale experiments in `../stale_legal_case_state_trials_20260710/`

The `.gitignore` keeps local data, runs, generated reports, virtual
environments, and build outputs out of Git while allowing the final template
pool and reusable prompt workflow to be tracked.
