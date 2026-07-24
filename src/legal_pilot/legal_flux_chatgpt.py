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

    cases_per_batch = int(flux_config.get("chatgpt_cases_per_batch", 30))
    homogeneous = _build_homogeneous_batches(
        cases,
        count=int(flux_config.get("chatgpt_homogeneous_batches", 24)),
        cases_per_batch=cases_per_batch,
        seed=config["project"]["seed"],
    )
    homogeneous_case_ids = {
        case.case_id for batch in homogeneous for case in batch["cases"]
    }
    full_coverage = bool(flux_config.get("template_batch_full_coverage", False))
    mixed_source_cases = (
        [case for case in cases if case.case_id not in homogeneous_case_ids]
        if full_coverage
        else cases
    )
    mixed = _build_mixed_batches(
        mixed_source_cases,
        count=int(flux_config.get("chatgpt_mixed_batches", 6)),
        cases_per_batch=cases_per_batch,
        seed=config["project"]["seed"] + 1000,
        fill_pool=cases,
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

    schema_path = output_dir / "legal_flux_template.schema.json"
    schema_path.write_text(
        json.dumps(LegalFluxTemplate.model_json_schema(), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    coverage = _coverage_summary(cases, manifest_batches)
    target_count = str(flux_config.get("template_pool_target", "150-250"))
    _write_prompts(prompts_dir, coverage, target_count=target_count)
    coverage["candidate_prompt_size_estimates"] = _candidate_prompt_size_estimates(
        manifest_batches=manifest_batches,
        schema_path=schema_path,
        candidate_prompt_path=prompts_dir / "01_generate_candidate_templates.md",
    )
    (output_dir / "coverage_summary.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(_readme(target_count=target_count), encoding="utf-8")
    batch_case_ids = [
        case_id
        for batch in manifest_batches
        for case_id in batch.get("case_ids", [])
    ]
    unique_batched_case_ids = set(batch_case_ids)
    source_case_ids = {case.case_id for case in cases}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "template_source_cases": len(cases),
        "homogeneous_batches": len(homogeneous),
        "mixed_batches": len(mixed),
        "cases_per_batch": cases_per_batch,
        "template_pool_target": target_count,
        "full_coverage_requested": full_coverage,
        "unique_batched_source_cases": len(unique_batched_case_ids & source_case_ids),
        "source_case_coverage_rate": (
            len(unique_batched_case_ids & source_case_ids) / len(source_case_ids)
            if source_case_ids
            else 0.0
        ),
        "duplicate_case_appearances": len(batch_case_ids) - len(set(batch_case_ids)),
        "unbatched_source_case_ids": sorted(source_case_ids - unique_batched_case_ids),
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
    groups: dict[str, list[NormalizedCase]] = defaultdict(list)
    for case in cases:
        groups[_batch_domain(case)].append(case)
    slot_counts = _allocate_homogeneous_group_slots(
        groups,
        count=count,
        cases_per_batch=cases_per_batch,
    )
    batches: list[dict[str, Any]] = []
    for offset, domain in enumerate(
        sorted(slot_counts, key=lambda key: (-slot_counts[key], -len(groups[key]), key))
    ):
        ordered = _order_group_cases_by_demand(
            groups[domain],
            seed=seed + offset,
        )
        for slot in range(slot_counts[domain]):
            if len(batches) >= count:
                return batches
            selected = ordered[slot * cases_per_batch : (slot + 1) * cases_per_batch]
            if not selected:
                break
            demand = _dominant_demand(selected)
            batches.append(
                {
                    "label": f"{domain}__{demand}",
                    "group_key": {
                        "broad_domain": domain,
                        "demand_focus": demand,
                    },
                    "cases": selected,
                }
            )
    return batches


def _allocate_homogeneous_group_slots(
    groups: dict[str, list[NormalizedCase]],
    *,
    count: int,
    cases_per_batch: int,
) -> dict[str, int]:
    total_cases = sum(len(group) for group in groups.values())
    if total_cases == 0 or count <= 0:
        return {}
    minimum_group_size = max(1, cases_per_batch // 2)
    candidates = {
        group_name: group
        for family, group in groups.items()
        for group_name in (family,)
        if len(group) >= minimum_group_size
    }
    if not candidates:
        candidates = groups
    raw = {
        group_name: count * len(group) / total_cases
        for group_name, group in candidates.items()
    }
    max_slots = {
        group_name: max(1, (len(group) + cases_per_batch - 1) // cases_per_batch)
        for group_name, group in candidates.items()
    }
    slots = {
        group_name: min(max_slots[group_name], max(1, int(raw[group_name])))
        for group_name in candidates
    }
    while sum(slots.values()) > count:
        removable = [
            group_name for group_name, amount in slots.items() if amount > 1
        ] or list(slots)
        group_name = min(
            removable,
            key=lambda key: (raw.get(key, 0.0) - slots[key], len(candidates[key])),
        )
        slots[group_name] -= 1
        if slots[group_name] <= 0:
            del slots[group_name]
    while sum(slots.values()) < count:
        expandable = [
            group_name
            for family in candidates
            for group_name in (family,)
            if slots.get(group_name, 0) < max_slots[group_name]
        ]
        if not expandable:
            break
        group_name = max(
            expandable,
            key=lambda key: (
                raw.get(key, 0.0) - slots.get(key, 0),
                len(candidates[key]),
                key,
            ),
        )
        slots[group_name] = slots.get(group_name, 0) + 1
    return slots


def _order_group_cases_by_demand(
    cases: list[NormalizedCase],
    *,
    seed: int,
) -> list[NormalizedCase]:
    groups: dict[str, list[NormalizedCase]] = defaultdict(list)
    for case in cases:
        groups[_demand_focus(case)].append(case)
    ordered: list[NormalizedCase] = []
    for offset, (demand, group) in enumerate(
        sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    ):
        ordered.extend(
            _select_diverse_cases(
                group,
                count=len(group),
                seed=seed + offset,
            )
        )
    return ordered


def _dominant_demand(cases: list[NormalizedCase]) -> str:
    if not cases:
        return "general_resolution"
    return Counter(_demand_focus(case) for case in cases).most_common(1)[0][0]


def _build_mixed_batches(
    cases: list[NormalizedCase],
    *,
    count: int,
    cases_per_batch: int,
    seed: int,
    fill_pool: list[NormalizedCase] | None = None,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    fill_pool = fill_pool or cases
    groups: dict[tuple[str, str], list[NormalizedCase]] = defaultdict(list)
    for case in cases:
        groups[(_batch_domain(case), _demand_focus(case))].append(case)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups, key=lambda key: (-len(groups[key]), key))
    batch_cases: list[list[NormalizedCase]] = [[] for _ in range(count)]
    batch_cursor = 0
    while any(groups.values()):
        progressed = False
        for key in keys:
            if not groups[key]:
                continue
            for _ in range(count):
                slot = batch_cases[batch_cursor % count]
                batch_cursor += 1
                if len(slot) < cases_per_batch:
                    slot.append(groups[key].pop())
                    progressed = True
                    break
            if not any(len(slot) < cases_per_batch for slot in batch_cases):
                break
        if not progressed:
            break

    filler = list(fill_pool)
    rng.shuffle(filler)
    filler_cursor = 0
    for slot in batch_cases:
        seen = {case.case_id for case in slot}
        attempts = 0
        while len(slot) < cases_per_batch and filler and attempts < len(filler) * 2:
            candidate = filler[filler_cursor % len(filler)]
            filler_cursor += 1
            attempts += 1
            if candidate.case_id in seen:
                continue
            slot.append(candidate)
            seen.add(candidate.case_id)

    batches: list[dict[str, Any]] = []
    for index, selected in enumerate(batch_cases, start=1):
        if not selected:
            continue
        batches.append(
            {
                "label": f"contrast_{index:02d}",
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
        "broad_domain_counts": dict(
            Counter(
                str(case.metadata.get("broad_domain", ""))
                for case in cases
            ).most_common()
        ),
        "authority_bucket_counts": dict(
            Counter(
                str(case.metadata.get("authority_bucket", ""))
                for case in cases
            ).most_common()
        ),
        "lawsuit_type_top10": dict(
            Counter(str(case.metadata.get("lawsuit_type", "")) for case in cases).most_common(10)
        ),
        "trajectory_prefix_count": len({_trajectory_prefix(case) for case in cases}),
        "file_sha256": sha256_text(path.read_text(encoding="utf-8")),
    }


def _case_record(case: NormalizedCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "claim": case.claim,
        "requested_remedy": case.requested_remedy,
        "parties": case.parties,
        "facts": case.facts,
        "lawsuit_type": case.metadata.get("lawsuit_type"),
        "reference_issues": case.reference_issues,
        "authorities": case.authorities,
        "relevant_cases": case.metadata.get("relevant_cases"),
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
        "batch_kind_counts": dict(Counter(batch["kind"] for batch in manifest_batches)),
    }


def _candidate_prompt_size_estimates(
    *,
    manifest_batches: list[dict[str, Any]],
    schema_path: Path,
    candidate_prompt_path: Path,
) -> dict[str, Any]:
    schema_text = schema_path.read_text(encoding="utf-8")
    prompt_text = candidate_prompt_path.read_text(encoding="utf-8")
    rows = []
    for batch in manifest_batches:
        batch_path = Path(batch["path"])
        char_count = (
            len(prompt_text)
            + len(schema_text)
            + len(batch_path.read_text(encoding="utf-8"))
        )
        rows.append(
            {
                "batch_id": batch["batch_id"],
                "label": batch["label"],
                "kind": batch["kind"],
                "characters": char_count,
                "tokens_at_3_5_chars_per_token": _ceil_div(char_count, 3.5),
                "tokens_at_4_chars_per_token": _ceil_div(char_count, 4.0),
            }
        )
    char_counts = [row["characters"] for row in rows]
    tokens_35 = [row["tokens_at_3_5_chars_per_token"] for row in rows]
    tokens_4 = [row["tokens_at_4_chars_per_token"] for row in rows]
    return {
        "per_batch_count": len(rows),
        "character_stats": _numeric_stats(char_counts),
        "token_estimates": {
            "chars_per_token_3_5": _numeric_stats(tokens_35),
            "chars_per_token_4": _numeric_stats(tokens_4),
        },
        "largest_batches": sorted(
            rows,
            key=lambda row: row["characters"],
            reverse=True,
        )[:5],
    }


def _numeric_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0.0, "max": 0, "total": 0}
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        median: float | int = ordered[midpoint]
    else:
        median = (ordered[midpoint - 1] + ordered[midpoint]) / 2
    return {
        "min": min(ordered),
        "median": median,
        "mean": sum(ordered) / len(ordered),
        "max": max(ordered),
        "total": sum(ordered),
    }


def _ceil_div(value: int, divisor: float) -> int:
    return int((value + divisor - 1) // divisor)


def _all_demand_counts(cases: list[NormalizedCase]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for case in cases:
        counter.update(_split_profile(case, "reasoning_demands"))
    return counter


def _primary_family(case: NormalizedCase) -> str:
    return (_split_profile(case, "template_families") or ["general_legal_reasoning"])[0]


def _batch_domain(case: NormalizedCase) -> str:
    domain = str(case.metadata.get("broad_domain") or "").strip()
    return domain or _primary_family(case)


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


def _write_prompts(
    prompts_dir: Path,
    coverage: dict[str, Any],
    *,
    target_count: str,
) -> None:
    (prompts_dir / "01_generate_candidate_templates.md").write_text(
        _candidate_generation_prompt(),
        encoding="utf-8",
    )
    (prompts_dir / "02_merge_deduplicate_templates.md").write_text(
        _merge_prompt(target_count=target_count),
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

Create 10-18 reusable high-level legal reasoning templates that capture patterns
shared by multiple cases in this batch. The templates are candidate building
blocks for a later global merge pass, not final one-case summaries.

Return JSONL only, one object per template, matching the schema exactly.

Rules:

- Use the ReasonFlux-style fields: template ID, name, tags, description,
  application scenario, reasoning flow, and example application.
- Derive templates from recurring reasoning needs in the cases. Treat the batch
  label only as weak orientation; do not merely restate it.
- Keep the abstraction at a middle legal-reasoning level: not generic cognitive
  labels like deduction, induction, analogy, or verification, and not
  single-case fact patterns. A good template should name a reusable legal
  operation, threshold, evidence assessment, issue-composition move, authority
  use, remedy choice, or domain-specific reasoning pattern that appears across
  multiple cases in the batch.
- Abstract away case-specific details. Do not copy case IDs, party names, dates,
  amounts, citations, F-number references, or final outcome words such as
  support/reject.
- Prefer medium-grained templates that can be sequenced with other templates.
- Name IDs as `CAND_<batch_id>_<nn>`, for example `CAND_homogeneous_001_01`.
- If a pattern is too local to one case, do not create a template for it.
- Keep `reasoning_flow` as ordered operational instructions, not hidden
  chain-of-thought.
- Do not infer, predict, or mention the gold outcome of any source case.
"""


def _merge_prompt(*, target_count: str) -> str:
    return """# Task: Merge LegalFlux candidate templates into the final pool

You will receive candidate-template JSONL files from homogeneous and mixed
contrast batches, plus the batch manifest and coverage summary.

Merge, deduplicate, and normalize the candidates into a final fixed LegalFlux
template pool of TARGET_COUNT templates.

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
  filtering, and domain-specific civil, criminal, immigration, public-law,
  tribunal, and procedural families.
- Use stable IDs `LF001`, `LF002`, ...
- Do not include case IDs, party names, dates, amounts, citations, F-number
  references, or support/reject outcome words.
- Make the pool useful for trajectory planning: each template should be a step
  that can be selected, instantiated, and composed with other steps.
""".replace("TARGET_COUNT", target_count)


def _coverage_audit_prompt(coverage: dict[str, Any]) -> str:
    return f"""# Task: Audit final LegalFlux template-pool coverage

You will receive the final template-pool JSONL, the batch manifest, and the
coverage summary below.

Check whether the final pool covers the main observed reasoning families,
domains, authorities, issue-composition patterns, and reasoning demands. Then
return a concise audit report with:

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


def _readme(*, target_count: str) -> str:
    return f"""# Azure GPT-5.6 LegalFlux Template-Pool Workflow

This folder supports the automated template-pool construction workflow for
Azure GPT-5.6 Sol. The same artifacts can still be inspected manually before
spending API credit.

## Pass 1: Candidate templates

For each file in `01_homogeneous_batches` and `02_mixed_contrast_batches`, send:

- one batch JSONL file
- `legal_flux_template.schema.json`
- `prompts/01_generate_candidate_templates.md`

Save the returned candidate JSONL files under the API output folder.

## Pass 2: Merge and deduplicate

After candidate templates are generated, send:

- all candidate-template JSONL files
- `batch_manifest.json`
- `coverage_summary.json`
- `legal_flux_template.schema.json`
- `prompts/02_merge_deduplicate_templates.md`

Ask for one final JSONL pool of {target_count} templates.

## Pass 3: Coverage audit

Send:

- the final template-pool JSONL
- `batch_manifest.json`
- `coverage_summary.json`
- `prompts/03_coverage_audit_and_gap_fill.md`

Use the audit to revise or add templates, then import the final pool:

```powershell
python -m legal_pilot --config configs\\legal_flux.yaml flux-import-templates --input path\\to\\final_templates.jsonl
```

These batches come only from the `template_source` split. Do not include
planner-train, trajectory-dev, or final-test cases in the template-pool creation
step.
"""
