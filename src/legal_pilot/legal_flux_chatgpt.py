from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_path
from .io_utils import canonical_json, sha256_text, write_jsonl
from .legal_flux import resolve_project_file
from .models import LegalFluxTemplate, NormalizedCase
from .runner import load_cases


DEMAND_PRIORITY = [
    "procedural_threshold_check",
    "evidence_and_burden_assessment",
    "defense_or_counterargument_check",
    "precedent_or_analogy_handling",
    "remedy_discretion_check",
    "multi_issue_composition",
    "long_fact_filtering",
    "supplied_rule_extraction",
    "rule_recall_or_doctrine_identification",
    "issue_spotting_gap",
    "dual_issue_resolution",
    "focused_issue_resolution",
]


def export_legal_flux_chatgpt_batches(config: dict[str, Any]) -> dict[str, Any]:
    cases = [
        case
        for case in load_cases(config)
        if case.metadata.get("selection_split") == "template_source"
    ]
    if not cases:
        raise RuntimeError("No LegalFlux template-source cases found. Run flux-prepare first.")
    flux_config = config["legal_flux"]
    output_dir = resolve_project_file(
        flux_config.get(
            "chatgpt_batch_dir",
            "reports/legal_flux/template_distillation/chatgpt_batches",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    homogeneous_dir = output_dir / "01_homogeneous_batches"
    mixed_dir = output_dir / "02_mixed_contrast_batches"
    prompts_dir = output_dir / "prompts"
    for directory in (homogeneous_dir, mixed_dir, prompts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _clear_generated_files(homogeneous_dir, patterns=("*.jsonl",))
    _clear_generated_files(mixed_dir, patterns=("*.jsonl",))
    _clear_generated_files(prompts_dir, patterns=("*.md",))
    _clear_generated_files(output_dir, patterns=("*.json", "*.md"))

    homogeneous = _build_homogeneous_batches(
        cases,
        count=int(flux_config.get("chatgpt_homogeneous_batches", 24)),
        cases_per_batch=int(flux_config.get("chatgpt_cases_per_batch", 30)),
        seed=config["project"]["seed"],
    )
    mixed = _build_mixed_batches(
        cases,
        count=int(flux_config.get("chatgpt_mixed_batches", 6)),
        cases_per_batch=int(flux_config.get("chatgpt_cases_per_batch", 30)),
        seed=config["project"]["seed"] + 1000,
    )
    manifest_batches = []
    for index, batch in enumerate(homogeneous, start=1):
        path = homogeneous_dir / _batch_filename("homogeneous", index, batch["label"])
        write_jsonl(path, [_case_record(case) for case in batch["cases"]])
        manifest_batches.append(_batch_manifest_row("homogeneous", index, path, batch))
    for index, batch in enumerate(mixed, start=1):
        path = mixed_dir / _batch_filename("mixed", index, batch["label"])
        write_jsonl(path, [_case_record(case) for case in batch["cases"]])
        manifest_batches.append(_batch_manifest_row("mixed", index, path, batch))

    coverage = _coverage_summary(cases, manifest_batches)
    schema_path = output_dir / "legal_flux_template.schema.json"
    schema_path.write_text(
        json.dumps(LegalFluxTemplate.model_json_schema(), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "coverage_summary.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_prompts(prompts_dir, coverage)
    readme_path = output_dir / "README.md"
    readme_path.write_text(_readme(), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "template_source_cases": len(cases),
        "homogeneous_batches": len(homogeneous),
        "mixed_batches": len(mixed),
        "cases_per_batch": int(flux_config.get("chatgpt_cases_per_batch", 30)),
        "output_dir": str(output_dir),
        "schema": str(schema_path),
        "readme": str(readme_path),
        "coverage_summary": str(output_dir / "coverage_summary.json"),
        "batches": manifest_batches,
    }
    manifest["manifest_hash"] = sha256_text(canonical_json(manifest_batches))
    (output_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "template_source_cases": len(cases),
        "homogeneous_batches": len(homogeneous),
        "mixed_batches": len(mixed),
        "batch_manifest": str(output_dir / "batch_manifest.json"),
        "coverage_summary": str(output_dir / "coverage_summary.json"),
        "prompts_dir": str(prompts_dir),
        "schema": str(schema_path),
        "readme": str(readme_path),
    }


def _build_homogeneous_batches(
    cases: list[NormalizedCase],
    *,
    count: int,
    cases_per_batch: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[NormalizedCase]] = defaultdict(list)
    for case in cases:
        groups[(_primary_family(case), _demand_focus(case))].append(case)
    ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    batches: list[dict[str, Any]] = []
    for (family, demand), group in ranked:
        if len(batches) >= count:
            break
        selected = _select_diverse_cases(
            group,
            count=min(cases_per_batch, len(group)),
            seed=seed + len(batches),
        )
        if not selected:
            continue
        batches.append(
            {
                "label": f"{family}__{demand}",
                "group_key": {"primary_family": family, "demand_focus": demand},
                "cases": selected,
            }
        )
    return batches


def _build_mixed_batches(
    cases: list[NormalizedCase],
    *,
    count: int,
    cases_per_batch: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[NormalizedCase]] = defaultdict(list)
    for case in cases:
        groups[(_primary_family(case), _demand_focus(case))].append(case)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups, key=lambda key: (-len(groups[key]), key))
    batches: list[dict[str, Any]] = []
    cursor = 0
    for index in range(count):
        selected: list[NormalizedCase] = []
        visited = 0
        while len(selected) < cases_per_batch and visited < len(keys) * 3:
            key = keys[cursor % len(keys)]
            cursor += 1
            visited += 1
            if groups[key]:
                selected.append(groups[key].pop())
        if len(selected) < cases_per_batch:
            remaining = [
                case for group in groups.values() for case in group if case not in selected
            ]
            rng.shuffle(remaining)
            selected.extend(remaining[: cases_per_batch - len(selected)])
        if selected:
            batches.append(
                {
                    "label": f"contrast_{index + 1:02d}",
                    "group_key": {"mode": "mixed_contrast"},
                    "cases": selected,
                }
            )
    return batches


def _select_diverse_cases(
    cases: list[NormalizedCase],
    *,
    count: int,
    seed: int,
) -> list[NormalizedCase]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[NormalizedCase]] = defaultdict(list)
    for case in cases:
        groups[(_trajectory_prefix(case), str(case.metadata.get("lawsuit_type", "")))].append(case)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[NormalizedCase] = []
    keys = sorted(groups, key=str)
    while len(selected) < count and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
    return selected


def _batch_manifest_row(
    kind: str,
    index: int,
    path: Path,
    batch: dict[str, Any],
) -> dict[str, Any]:
    cases = batch["cases"]
    return {
        "batch_id": f"{kind}_{index:03d}",
        "kind": kind,
        "label": batch["label"],
        "path": str(path),
        "case_count": len(cases),
        "case_ids": [case.case_id for case in cases],
        "primary_family_counts": dict(Counter(_primary_family(case) for case in cases).most_common()),
        "demand_focus_counts": dict(Counter(_demand_focus(case) for case in cases).most_common()),
        "trajectory_prefix_count": len({_trajectory_prefix(case) for case in cases}),
        "file_sha256": sha256_text(path.read_text(encoding="utf-8")),
    }


def _case_record(case: NormalizedCase) -> dict[str, Any]:
    reference_state = (
        case.reference_state.model_dump(mode="json") if case.reference_state else None
    )
    return {
        "case_id": case.case_id,
        "claim": case.claim,
        "requested_remedy": case.requested_remedy,
        "parties": case.parties,
        "facts": case.facts,
        "lawsuit_type": case.metadata.get("lawsuit_type"),
        "legal_flux_profile": case.metadata.get("legal_flux_profile"),
        "reference_issues": case.reference_issues,
        "reference_state": reference_state,
        "authorities": case.authorities,
    }


def _coverage_summary(
    cases: list[NormalizedCase],
    manifest_batches: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "template_source_cases": len(cases),
        "primary_family_counts": dict(Counter(_primary_family(case) for case in cases).most_common()),
        "demand_focus_counts": dict(Counter(_demand_focus(case) for case in cases).most_common()),
        "all_reasoning_demand_counts": dict(_all_demand_counts(cases).most_common()),
        "trajectory_prefix_counts_top50": dict(
            Counter(_trajectory_prefix(case) for case in cases).most_common(50)
        ),
        "batch_count": len(manifest_batches),
        "batched_case_ids": len(
            {
                case_id
                for batch in manifest_batches
                for case_id in batch.get("case_ids", [])
            }
        ),
    }


def _all_demand_counts(cases: list[NormalizedCase]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for case in cases:
        counter.update(_split_profile(case, "reasoning_demands"))
    return counter


def _primary_family(case: NormalizedCase) -> str:
    return (_split_profile(case, "template_families") or ["general_civil_reasoning"])[0]


def _demand_focus(case: NormalizedCase) -> str:
    demands = set(_split_profile(case, "reasoning_demands"))
    for demand in DEMAND_PRIORITY:
        if demand in demands:
            return demand
    return "general_resolution"


def _trajectory_prefix(case: NormalizedCase) -> str:
    profile = case.metadata.get("legal_flux_profile") or {}
    signature = str(profile.get("trajectory_signature") or "")
    parts = [part.strip() for part in signature.split(" > ") if part.strip()]
    return " > ".join(parts[:5]) if parts else "unknown"


def _split_profile(case: NormalizedCase, key: str) -> list[str]:
    profile = case.metadata.get("legal_flux_profile") or {}
    value = str(profile.get(key) or "")
    return [part for part in value.split("|") if part]


def _batch_filename(kind: str, index: int, label: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", label).strip("_").lower()
    return f"{kind}_{index:03d}_{safe[:80]}.jsonl"


def _clear_generated_files(directory: Path, *, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()


def _write_prompts(prompts_dir: Path, coverage: dict[str, Any]) -> None:
    (prompts_dir / "01_generate_candidate_templates.md").write_text(
        _candidate_generation_prompt(),
        encoding="utf-8",
    )
    (prompts_dir / "02_merge_deduplicate_templates.md").write_text(
        _merge_prompt(),
        encoding="utf-8",
    )
    (prompts_dir / "03_coverage_audit_and_gap_fill.md").write_text(
        _coverage_audit_prompt(coverage),
        encoding="utf-8",
    )


def _candidate_generation_prompt() -> str:
    return """# Task: Generate LegalFlux candidate templates from one batch

You will receive one JSONL batch of LegalHK template-source cases and
`legal_flux_template.schema.json`.

Create 6-12 reusable high-level legal reasoning templates that capture patterns
shared by multiple cases in this batch. These are candidate templates for a
later global merge pass, not final one-case summaries.

Return JSONL only, one object per template, matching the schema exactly.

Rules:

- Use the ReasonFlux-style fields: template ID, name, tags, description,
  application scenario, reasoning flow, and example application.
- Abstract away case-specific details. Do not copy case IDs, party names, dates,
  amounts, citations, F-number references, or final outcome words such as
  support/reject.
- Prefer medium-grained templates that can be sequenced with other templates.
- Name IDs as `CAND_<batch_id>_<nn>`, for example `CAND_homogeneous_001_01`.
- If a pattern is too local to one case, do not create a template for it.
- Keep `reasoning_flow` as ordered operational instructions, not hidden
  chain-of-thought.
"""


def _merge_prompt() -> str:
    return """# Task: Merge LegalFlux candidate templates into the final pool

You will receive candidate-template JSONL files from homogeneous and mixed
contrast batches, plus the batch manifest and coverage summary.

Merge, deduplicate, and normalize the candidates into a final fixed LegalFlux
template pool of 80-120 templates.

Return JSONL only, one object per final template, matching
`legal_flux_template.schema.json`.

Rules:

- Preserve cross-batch patterns by merging near-duplicates instead of keeping
  local aliases.
- Keep distinct templates when they imply genuinely different reasoning
  behavior, not merely different legal topics.
- Ensure coverage for procedural thresholds, supplied-rule extraction, rule
  recall, issue decomposition/composition, evidence and burden assessment,
  defenses/counterarguments, precedent/analogy, remedy discretion, long-fact
  filtering, and domain-specific civil families.
- Use stable IDs `LF001`, `LF002`, ...
- Do not include case IDs, party names, dates, amounts, citations, F-number
  references, or support/reject outcome words.
- Make the pool useful for trajectory planning: each template should be a step
  that can be selected, instantiated, and composed with other steps.
"""


def _coverage_audit_prompt(coverage: dict[str, Any]) -> str:
    return f"""# Task: Audit final LegalFlux template-pool coverage

You will receive the final template-pool JSONL, the batch manifest, and the
coverage summary below.

Check whether the final pool covers the main observed reasoning families and
demands. Then return a concise audit report with:

1. Covered categories.
2. Under-covered categories.
3. Duplicative templates that should be merged.
4. Up to 20 additional templates if important gaps remain.

If you propose additional templates, return them as JSONL records matching
`legal_flux_template.schema.json` after the audit report.

Coverage summary:

```json
{json.dumps(coverage, ensure_ascii=False, indent=2)}
```
"""


def _readme() -> str:
    return """# ChatGPT LegalFlux Template-Pool Workflow

This folder supports a no-API template-pool construction workflow.

## Pass 1: Candidate templates

For each file in `01_homogeneous_batches` and `02_mixed_contrast_batches`, open a
fresh ChatGPT conversation or continue a clean working conversation. Upload or
paste:

- one batch JSONL file
- `legal_flux_template.schema.json`
- `prompts/01_generate_candidate_templates.md`

Save the returned candidate JSONL files locally.

## Pass 2: Merge and deduplicate

After candidate templates are generated, give ChatGPT:

- all candidate-template JSONL files
- `batch_manifest.json`
- `coverage_summary.json`
- `legal_flux_template.schema.json`
- `prompts/02_merge_deduplicate_templates.md`

Ask for one final JSONL pool of 80-120 templates.

## Pass 3: Coverage audit

Give ChatGPT:

- the final template-pool JSONL
- `batch_manifest.json`
- `coverage_summary.json`
- `prompts/03_coverage_audit_and_gap_fill.md`

Use the audit to revise or add templates, then import the final pool:

```powershell
python -m legal_pilot --config configs\\legal_flux.yaml flux-import-templates --input path\\to\\final_templates.jsonl
```

These batches come only from the `template_source` split. Do not include
trajectory-dev or final-test cases in the ChatGPT template-pool creation step.
"""
