# GitHub Packaging Notes

This folder is the standalone legal reasoning project. The parent workspace also
contains agricultural-contract annotation files, paper PDFs, scratch folders,
and an external ReasonFlux inspection checkout; those should not be published
with this project.

Recommended publication scope:

- include `configs/`, `prompts/`, `schemas/`, `scripts/`, `src/`, `tests/`,
  `pyproject.toml`, `README.md`, and this note;
- exclude local virtual environments, `runs/`, generated `reports/`, embedding
  caches, and raw or processed datasets unless a dataset license explicitly
  permits redistribution;
- keep older `flux_fixed` and `flux_adaptive` code as optional ablation paths,
  but use `flux_rf_style` as the main ReasonFlux-style condition.

The root `.gitignore` is configured so a root-level repository includes this
folder and ignores unrelated workspace assets. If you initialize Git directly
inside `legal_case_state_pilot`, this folder's `.gitignore` is sufficient.
