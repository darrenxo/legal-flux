from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .adaptive_profiles import profile_frame, profile_row
from .clients import OllamaClient
from .config import resolve_path
from .legalhk_data import (
    LEGALHK_PARQUET_URL,
    download_file,
    legalhk_index,
    normalize_legalhk_case,
)
from .io_utils import read_jsonl, sha256_text, write_jsonl
from .legal_flux import (
    FLUX_PHASES,
    build_legal_flux_jobs,
    freeze_manifest_path,
    legal_flux_plan_hash,
    legal_flux_workflow_hash,
    load_template_pool,
    resolve_project_file,
    sanitize_flux_template,
    template_pool_hash,
    template_pool_path,
    validate_template_pool,
    write_template_pool,
)
from .legalhk_selection import (
    explicit_leakage_reasons,
    is_civil_legalhk_row,
    strict_evaluation_reasons,
)
from .models import LegalFluxTemplate, NormalizedCase
from .runner import load_cases


def prepare_legal_flux(config: dict[str, Any]) -> dict[str, Any]:
    raw_dir = resolve_path(config, "raw_dir")
    processed_dir = resolve_path(config, "processed_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = raw_dir / "legalhk" / "train.parquet"
    download_file(LEGALHK_PARQUET_URL, parquet_path)
    frame = pd.read_parquet(parquet_path).fillna("")
    eligible, excluded_reasons = _eligible_frame(frame, config)

    prior_case_ids = _prior_legalhk_case_ids(config)
    if config["legal_flux"].get("exclude_existing_legalhk_only_cases", True):
        prior_indices = {legalhk_index(case_id) for case_id in prior_case_ids}
        eligible = eligible.drop(index=list(prior_indices & set(eligible.index)))

    profiled = _add_profile_columns(eligible)
    splits = _draw_flux_splits(profiled, config)

    cases: list[NormalizedCase] = []
    review_rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for split, selected in splits.items():
        split_counts[split] = len(selected)
        for index, row in selected.iterrows():
            case = _normalize_flux_case(index, row, split=split)
            cases.append(case)
            review_rows.append(_review_row(case, row))

    write_jsonl(processed_dir / "cases.jsonl", [case.model_dump(mode="json") for case in cases])
    write_jsonl(processed_dir / "selection_review.jsonl", review_rows)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rows": len(frame),
        "eligible_rows_after_filters": len(eligible),
        "excluded_rows": len(frame) - len(eligible),
        "excluded_reasons": dict(sorted(excluded_reasons.items())),
        "prior_legalhk_only_cases_excluded": (
            len(prior_case_ids)
            if config["legal_flux"].get("exclude_existing_legalhk_only_cases", True)
            else 0
        ),
        "splits": split_counts,
        "split_outcomes": {
            split: dict(sorted(selected["support&reject"].value_counts().items()))
            for split, selected in splits.items()
        },
        "split_trajectory_signatures": {
            split: int(selected["trajectory_signature"].nunique())
            for split, selected in splits.items()
        },
        "split_domain_counts": {
            split: dict(sorted(selected["broad_domain"].value_counts().items()))
            for split, selected in splits.items()
        },
        "split_family_counts": {
            split: dict(selected["family_bucket"].value_counts().head(25).items())
            for split, selected in splits.items()
        },
        "split_demand_counts": {
            split: dict(selected["demand_bucket"].value_counts().head(25).items())
            for split, selected in splits.items()
        },
        "split_authority_counts": {
            split: dict(sorted(selected["authority_bucket"].value_counts().items()))
            for split, selected in splits.items()
        },
        "notes": [
            "All LegalFlux splits use structured LegalHK rows with binary labels, non-empty claims/facts, and configurable input-length/outcome-leakage screening.",
            "The default research split preserves natural label, domain, authority, length, issue-count, and heuristic family/demand distributions across template-source, planner-train, trajectory-dev, and final-test splits.",
            "Template-source, planner-train, trajectory-dev, and final-test cases are disjoint; final-test cases are not used for template generation, training, tuning, reward construction, or error-driven revisions.",
            "selection_review.jsonl omits outcome labels and judgment decisions so it can be inspected without revealing answers.",
        ],
    }
    (processed_dir / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def export_legal_flux_template_inputs(config: dict[str, Any]) -> dict[str, Any]:
    cases = [
        case
        for case in load_cases(config)
        if case.metadata.get("selection_split") == "template_source"
    ]
    if not cases:
        raise RuntimeError("No LegalFlux template-source cases found. Run flux-prepare first.")
    output_dir = resolve_project_file(
        config["legal_flux"].get(
            "template_export_dir",
            "reports/legal_flux/template_distillation",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    packet = output_dir / "template_source_cases.jsonl"
    write_jsonl(packet, [_template_source_packet(case) for case in cases])
    schema_path = output_dir / "legal_flux_template.schema.json"
    schema_path.write_text(
        json.dumps(LegalFluxTemplate.model_json_schema(), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    instructions_path = output_dir / "INSTRUCTIONS.md"
    instructions_path.write_text(
        _template_distillation_instructions(
            case_count=len(cases),
            target_count=config["legal_flux"].get("template_pool_target", "80-120"),
        ),
        encoding="utf-8",
    )
    manifest = {
        "cases": len(cases),
        "packet": str(packet),
        "schema": str(schema_path),
        "instructions": str(instructions_path),
        "packet_sha256": sha256_text(packet.read_text(encoding="utf-8")),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def import_legal_flux_templates(
    config: dict[str, Any],
    *,
    input_path: str | Path,
) -> dict[str, Any]:
    source = Path(input_path).resolve()
    rows = read_jsonl(source)
    templates = [
        sanitize_flux_template(LegalFluxTemplate.model_validate(row))
        for row in rows
    ]
    validate_template_pool(templates)
    output = template_pool_path(config)
    write_template_pool(output, templates)
    manifest = {
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(source),
        "output_path": str(output),
        "templates": len(templates),
        "template_pool_hash": template_pool_hash(templates),
        "schema": "legal_flux_template.json",
        "sanitization": [
            "Removed or generalized source case IDs, F-number references, numeric values, and explicit support/reject outcome words in template text.",
            "Rejected duplicate template IDs and templates with fewer than two tags or fewer than two reasoning-flow steps.",
        ],
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def freeze_legal_flux_phase(config: dict[str, Any]) -> dict[str, Any]:
    cases = load_cases(config)
    templates = load_template_pool(config)
    smoke_jobs = build_legal_flux_jobs(cases, config, phase="smoke")
    run_dir = resolve_path(config, "runs_dir") / "smoke"
    records = {
        row.get("run_hash"): row
        for row in read_jsonl(run_dir / "generations.jsonl")
        if row.get("run_hash")
    }
    client = OllamaClient(config["model"]["base_url"], config["model"]["timeout_seconds"])
    try:
        model_info = client.model_info(config["model"]["name"])
    finally:
        client.close()
    if not model_info:
        raise RuntimeError("Configured Ollama model is not available.")
    digest = model_info.get("digest", "unknown")
    workflow_hash = legal_flux_workflow_hash(config)
    expected = {
        _flux_expected_run_hash(
            job,
            model_digest=digest,
            workflow_hash=workflow_hash,
            template_hash=template_pool_hash(templates),
            seed=config["model"]["seed"],
        )
        for job in smoke_jobs
    }
    current = {run_hash: records.get(run_hash) for run_hash in expected}
    missing = [run_hash for run_hash, row in current.items() if row is None]
    failures = [
        row for row in current.values() if row is not None and row.get("status") != "ok"
    ]
    if missing or failures:
        raise RuntimeError(
            "LegalFlux smoke is not freeze-ready: "
            f"{len(missing)} missing and {len(failures)} failed records."
        )
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": config["model"]["name"],
            "digest": digest,
            "size": model_info.get("size"),
        },
        "workflow_hash": workflow_hash,
        "template_pool_hash": template_pool_hash(templates),
        "template_count": len(templates),
        "smoke_run_hashes": sorted(expected),
        "trajectory_dev_plan_hash": legal_flux_plan_hash(
            cases, config, phase="trajectory_dev"
        ),
        "final_test_plan_hash": legal_flux_plan_hash(
            cases, config, phase="final_test"
        ),
        "split_case_ids": {
            split: [
                case.case_id
                for case in cases
                if case.metadata.get("selection_split") == split
            ]
            for split in sorted(FLUX_PHASES - {"template_source"})
        },
    }
    path = freeze_manifest_path(config)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(path),
        "records": len(expected),
        "model": manifest["model"],
        "workflow_hash": workflow_hash,
        "template_pool_hash": manifest["template_pool_hash"],
        "final_test_plan_hash": manifest["final_test_plan_hash"],
    }


def assert_legal_flux_frozen(
    config: dict[str, Any],
    *,
    model_digest: str,
    workflow_hash: str,
    template_hash: str,
) -> None:
    path = freeze_manifest_path(config)
    if not path.exists():
        raise RuntimeError("LegalFlux final test is not frozen. Run flux-freeze first.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("model", {}).get("digest") != model_digest:
        raise RuntimeError("Ollama model digest differs from the LegalFlux freeze.")
    if manifest.get("workflow_hash") != workflow_hash:
        raise RuntimeError("LegalFlux prompts, schemas, settings, or code changed.")
    if manifest.get("template_pool_hash") != template_hash:
        raise RuntimeError("LegalFlux template pool differs from the frozen manifest.")
    cases = load_cases(config)
    if manifest.get("final_test_plan_hash") != legal_flux_plan_hash(
        cases, config, phase="final_test"
    ):
        raise RuntimeError("The frozen LegalFlux final-test case stream changed.")


def _eligible_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, Counter[str]]:
    max_characters = config["legal_flux"].get(
        "max_input_characters",
        config.get("data", {}).get("max_input_characters", 5000),
    )
    ngram_size = config["legal_flux"].get("decision_overlap_ngram", 6)
    overlap_threshold = config["legal_flux"].get("decision_overlap_threshold", 0.12)
    excluded: Counter[str] = Counter()
    eligible_indices: list[Any] = []
    for index, row in frame.iterrows():
        reasons: list[str] = []
        outcome = str(row.get("support&reject", "")).strip().lower()
        if outcome not in {"support", "reject"}:
            reasons.append("non_binary_outcome")
        if not str(row.get("plaintiff_claim", "")).strip():
            reasons.append("empty_claim")
        if not str(row.get("more_facts", "")).strip():
            reasons.append("empty_facts")
        if (
            len(str(row.get("plaintiff_claim", "")))
            + len(str(row.get("more_facts", "")))
            > max_characters
        ):
            reasons.append("input_too_long")
        if not config["legal_flux"].get("include_all_domains", True):
            if not is_civil_legalhk_row(
                plaintiff=str(row.get("plaintiff", "")),
                lawsuit_type=str(row.get("lawsuit_type", "")),
                claim=str(row.get("plaintiff_claim", "")),
            ):
                reasons.append("not_civil")
        if config["legal_flux"].get("screen_explicit_outcome_leakage", True):
            reasons.extend(
                explicit_leakage_reasons(
                    str(row.get("more_facts", "")),
                    judgment_decision=str(row.get("judgment_decision", "")),
                    ngram_size=ngram_size,
                    overlap_threshold=overlap_threshold,
                )
            )
        if config["legal_flux"].get("strict_fact_language_filter", False):
            reasons.extend(strict_evaluation_reasons(str(row.get("more_facts", ""))))
        if reasons:
            excluded.update(set(reasons))
        else:
            eligible_indices.append(index)
    eligible = frame.loc[eligible_indices].copy()
    eligible["support&reject"] = (
        eligible["support&reject"].astype(str).str.strip().str.lower()
    )
    eligible["has_defense"] = eligible["issues"].astype(str).str.contains(
        r"defen[cs]\w*|contributory|counterclaim\w*|exception|whether.*liable",
        case=False,
        regex=True,
    )
    return eligible, excluded


def _add_profile_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    profiles = profile_frame(frame).set_index("row_index")
    profile_columns = [
        "template_families",
        "reasoning_demands",
        "trajectory_signature",
        "trajectory_length",
        "family_count",
        "demand_count",
        "issue_count",
        "fact_count_estimate",
        "fact_characters",
        "related_law_count",
        "relevant_case_count",
    ]
    result = frame.join(profiles[profile_columns])
    result["broad_domain"] = result.apply(_broad_domain, axis=1)
    result["authority_bucket"] = result.apply(
        lambda row: _authority_bucket(
            int(row.get("related_law_count", 0)),
            int(row.get("relevant_case_count", 0)),
        ),
        axis=1,
    )
    result["family_bucket"] = result["template_families"].map(
        lambda value: str(value).split("|")[0] if str(value) else "unknown"
    )
    result["demand_bucket"] = result["reasoning_demands"].map(
        lambda value: str(value).split("|")[0] if str(value) else "unknown"
    )
    result["issue_bucket"] = result["issue_count"].map(
        lambda count: "none" if int(count) == 0 else "few" if int(count) <= 2 else "many"
    )
    ranked_lengths = result["more_facts"].astype(str).str.len().rank(method="first")
    result["length_bucket"] = pd.qcut(
        ranked_lengths, q=3, labels=["short", "medium", "long"]
    ).astype(str)
    result["trajectory_bucket"] = result["trajectory_signature"].map(
        lambda value: " > ".join(str(value).split(" > ")[:4])
    )
    return result


def _draw_flux_splits(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    counts = _split_counts(len(frame), config)
    smoke_indices = _proportional_stratified_indices(
        frame,
        count=counts["smoke"],
        seed=config["project"]["seed"],
        key_columns=["support&reject", "broad_domain"],
    )
    smoke = frame.loc[smoke_indices].copy()
    remaining = frame.drop(index=smoke_indices)
    main_counts = {
        split: counts[split]
        for split in ("template_source", "planner_train", "trajectory_dev", "final_test")
    }
    splits = _proportional_stratified_splits(
        remaining,
        targets=main_counts,
        seed=config["project"]["seed"] + 1,
        key_columns=[
            "support&reject",
            "broad_domain",
            "family_bucket",
            "demand_bucket",
            "issue_bucket",
            "length_bucket",
            "authority_bucket",
            "has_defense",
        ],
    )
    return {"smoke": smoke, **splits}


def _split_counts(total_rows: int, config: dict[str, Any]) -> dict[str, int]:
    smoke = int(config["legal_flux"].get("smoke_cases", 5))
    if smoke >= total_rows:
        raise ValueError(f"Requested {smoke} smoke rows from a pool of {total_rows}.")
    fractions = config["legal_flux"].get("split_fractions")
    if fractions:
        split_names = ("template_source", "planner_train", "trajectory_dev", "final_test")
        fraction_values = {
            name: float(fractions.get(name, 0.0)) for name in split_names
        }
        total_fraction = sum(fraction_values.values())
        if total_fraction <= 0:
            raise ValueError("legal_flux.split_fractions must sum to a positive value.")
        available = total_rows - smoke
        raw = {
            name: available * fraction_values[name] / total_fraction
            for name in split_names
        }
        counts = {name: int(raw[name]) for name in split_names}
        remainder = available - sum(counts.values())
        for name in sorted(
            split_names,
            key=lambda key: raw[key] - counts[key],
            reverse=True,
        )[:remainder]:
            counts[name] += 1
        return {"smoke": smoke, **counts}
    return {
        "smoke": smoke,
        "final_test": int(config["legal_flux"].get("final_test_cases", 512)),
        "trajectory_dev": int(config["legal_flux"].get("trajectory_dev_cases", 256)),
        "planner_train": int(config["legal_flux"].get("planner_train_cases", 0)),
        "template_source": int(config["legal_flux"].get("template_source_cases", 1400)),
    }


def _balanced_stratified_indices(
    frame: pd.DataFrame,
    *,
    count: int,
    seed: int,
) -> list[Any]:
    if count > len(frame):
        raise ValueError(f"Requested {count} rows from a pool of {len(frame)}.")
    rng = random.Random(seed)
    targets = {
        "support": count // 2 + count % 2,
        "reject": count // 2,
    }
    chosen: list[Any] = []
    for outcome, target in targets.items():
        subset = frame[frame["support&reject"] == outcome]
        chosen.extend(
            _stratified_indices(
                subset,
                count=target,
                rng=rng,
                key_columns=[
                    "broad_domain",
                    "issue_bucket",
                    "length_bucket",
                    "authority_bucket",
                    "has_defense",
                ],
            )
        )
    if len(chosen) < count:
        remaining = [index for index in frame.index if index not in set(chosen)]
        rng.shuffle(remaining)
        chosen.extend(remaining[: count - len(chosen)])
    if len(chosen) != count:
        raise ValueError("Could not draw the requested balanced LegalFlux split.")
    rng.shuffle(chosen)
    return chosen


def _stratified_indices(
    frame: pd.DataFrame,
    *,
    count: int,
    rng: random.Random,
    key_columns: list[str],
) -> list[Any]:
    groups: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for index, row in frame.iterrows():
        groups[tuple(row[column] for column in key_columns)].append(index)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[Any] = []
    keys = sorted(groups, key=str)
    while len(selected) < count and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
    return selected


def _proportional_stratified_indices(
    frame: pd.DataFrame,
    *,
    count: int,
    seed: int,
    key_columns: list[str],
) -> list[Any]:
    if count > len(frame):
        raise ValueError(f"Requested {count} rows from a pool of {len(frame)}.")
    splits = _proportional_stratified_splits(
        frame,
        targets={"selected": count, "rest": len(frame) - count},
        seed=seed,
        key_columns=key_columns,
    )
    return list(splits["selected"].index)


def _proportional_stratified_splits(
    frame: pd.DataFrame,
    *,
    targets: dict[str, int],
    seed: int,
    key_columns: list[str],
) -> dict[str, pd.DataFrame]:
    if sum(targets.values()) != len(frame):
        raise ValueError("Stratified split targets must sum to the frame length.")
    rng = random.Random(seed)
    remaining = frame.copy()
    selected: dict[str, pd.DataFrame] = {}
    split_names = list(targets)
    for split in split_names[:-1]:
        indices = _stratified_sample_exact_indices(
            remaining,
            count=targets[split],
            rng=rng,
            key_columns=key_columns,
        )
        selected[split] = remaining.loc[indices].copy()
        remaining = remaining.drop(index=indices)
    final_split = split_names[-1]
    if len(remaining) != targets[final_split]:
        raise ValueError(
            f"Final split {final_split} expected {targets[final_split]} rows "
            f"but received {len(remaining)}."
        )
    selected[final_split] = remaining.copy()
    return selected


def _stratified_sample_exact_indices(
    frame: pd.DataFrame,
    *,
    count: int,
    rng: random.Random,
    key_columns: list[str],
) -> list[Any]:
    if count < 0 or count > len(frame):
        raise ValueError(f"Requested {count} rows from a pool of {len(frame)}.")
    if count == 0:
        return []
    if count == len(frame):
        return list(frame.index)
    groups: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for index, row in frame.iterrows():
        groups[tuple(row[column] for column in key_columns)].append(index)
    for values in groups.values():
        rng.shuffle(values)

    raw_quotas = {
        key: count * len(indices) / len(frame)
        for key, indices in groups.items()
    }
    quotas = {
        key: min(int(raw_quotas[key]), len(groups[key]))
        for key in groups
    }
    leftover = count - sum(quotas.values())
    ranked = sorted(
        groups,
        key=lambda key: (
            raw_quotas[key] - quotas[key],
            rng.random(),
        ),
        reverse=True,
    )
    for key in ranked:
        if leftover <= 0:
            break
        if quotas[key] >= len(groups[key]):
            continue
        quotas[key] += 1
        leftover -= 1
    if leftover:
        raise ValueError("Could not allocate enough rows for stratified sample.")

    selected: list[Any] = []
    for key, quota in quotas.items():
        selected.extend(groups[key][:quota])
    rng.shuffle(selected)
    return selected


def _normalize_flux_case(index: Any, row: pd.Series, *, split: str) -> NormalizedCase:
    case = normalize_legalhk_case(index, row, split=split)
    profile = profile_row(row.to_dict())
    metadata = {
        **case.metadata,
        "selection_split": split,
        "legal_flux_profile": {
            key: profile[key]
            for key in (
                "template_families",
                "reasoning_demands",
                "trajectory_signature",
                "trajectory_length",
                "family_count",
                "demand_count",
                "issue_count",
                "fact_count_estimate",
                "fact_characters",
                "related_law_count",
                "relevant_case_count",
            )
        },
        "broad_domain": str(row.get("broad_domain", "")),
        "authority_bucket": str(row.get("authority_bucket", "")),
        "relevant_cases": str(row.get("relevant_cases", "")),
    }
    authorities = str(row.get("related_laws", "")).strip() or None
    return case.model_copy(update={"authorities": authorities, "metadata": metadata})


def _review_row(case: NormalizedCase, row: pd.Series) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "selection_split": case.metadata["selection_split"],
        "claim": case.claim,
        "requested_remedy": case.requested_remedy,
        "parties": case.parties,
        "facts": case.facts,
        "lawsuit_type": case.metadata.get("lawsuit_type"),
        "issue_count": case.metadata.get("issue_count"),
        "fact_characters": case.metadata.get("fact_characters"),
        "has_defense": case.metadata.get("has_defense"),
        "legal_flux_profile": case.metadata.get("legal_flux_profile"),
        "broad_domain": case.metadata.get("broad_domain"),
        "authority_bucket": case.metadata.get("authority_bucket"),
        "related_law_count": len(
            [line for line in str(row.get("related_laws", "")).splitlines() if line.strip()]
        ),
        "relevant_case_count": len(
            [line for line in str(row.get("relevant_cases", "")).splitlines() if line.strip()]
        ),
        "leakage_screen": "all_domain_explicit_outcome_leakage_screen",
    }


def _template_source_packet(case: NormalizedCase) -> dict[str, Any]:
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
        "reference_issues": case.reference_issues,
        "reference_state": reference_state,
        "authorities": case.authorities,
    }


def _template_distillation_instructions(*, case_count: int, target_count: Any) -> str:
    return f"""# LegalFlux template-pool distillation

Use `template_source_cases.jsonl` to create a fixed pool of {target_count}
high-level legal reasoning templates. The packet contains {case_count}
template-source cases only. It does not contain planner-train, trajectory-dev,
or final-test cases, judgment decisions, support/reject labels, or heuristic
family/demand labels.

Return JSONL, one object per template, matching `legal_flux_template.schema.json`.
Follow the ReasonFlux-style schema:

- `template_id`: stable ID such as `LF001`.
- `template_name`: short name.
- `knowledge_tags`: 2-8 abstract tags.
- `description`: what the template does.
- `application_scenario`: when a planner should select it.
- `reasoning_flow`: ordered high-level steps for applying it.
- `example_application`: abstract example without source-case facts.

Requirements:

- Produce reusable templates, not summaries of individual cases.
- Do not include source case IDs, party names, dates, money amounts, citations,
  F-number references, or final outcome words such as support/reject.
- Cover different case-level trajectories: issue spotting, supplied-rule
  extraction, rule recall, procedural threshold, evidence/burden assessment,
  precedent/analogy handling, defenses/counterarguments, remedy discretion,
  criminal/immigration/public-law pathways, and final issue composition.
- Prefer medium-grained templates that can be sequenced by a planner. Avoid a
  single all-purpose IRAC template and avoid tiny one-sentence micro-actions.
- Return JSONL only, with no prose before or after the records.
"""


def _broad_domain(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(key, ""))
        for key in ("plaintiff", "defendant", "plaintiff_claim", "lawsuit_type", "issues")
    ).lower()
    if "hksar" in text or re.search(
        r"\b(criminal|conviction\w*|sentence|sentencing|prosecution|offen[cs]\w*|"
        r"charg\w*|theft|trafficking|bail)\b",
        text,
    ):
        return "criminal"
    if re.search(
        r"\b(immigration|non-refoulement|refugee\w*|torture|deport\w*|"
        r"removal order)\b",
        text,
    ):
        return "immigration_public"
    if re.search(
        r"\b(judicial review|public law|administrative|director of|commissioner)\b",
        text,
    ):
        return "public_law"
    if re.search(
        r"\b(appeals?|leave|extension of time|set aside|strike out|case stated|"
        r"interlocutory)\b",
        text,
    ):
        return "procedure_appeal"
    if re.search(
        r"\b(labou?r|employment|employees?|wages?|severance|work injur\w*)\b",
        text,
    ):
        return "employment_labor"
    if re.search(
        r"\b(family|matrimonial|divorce|maintenance|probate|estate|trust)\b",
        text,
    ):
        return "family_trust_probate"
    if re.search(
        r"\b(compan(?:y|ies)|winding up|liquidat\w*|shareholders?|directors?|"
        r"insolven\w*)\b",
        text,
    ):
        return "company_insolvency"
    if re.search(
        r"\b(propert\w*|land|tenan\w*|leas\w*|possess\w*|premises|"
        r"conveyanc\w*)\b",
        text,
    ):
        return "property_land"
    if re.search(
        r"\b(contracts?|agreements?|debts?|loans?|payments?|invoices?|"
        r"breach(?:es|ed|ing)?)\b",
        text,
    ):
        return "contract_debt"
    if re.search(
        r"\b(torts?|negligence|injur\w*|damag\w*|defamation|accidents?)\b",
        text,
    ):
        return "tort_damages"
    return "other_legal"


def _authority_bucket(related_law_count: int, relevant_case_count: int) -> str:
    if related_law_count and relevant_case_count:
        return "laws_and_cases"
    if related_law_count:
        return "laws_only"
    if relevant_case_count:
        return "cases_only"
    return "no_authorities"


def _prior_legalhk_case_ids(config: dict[str, Any]) -> set[str]:
    path = resolve_project_file(
        config["legal_flux"].get(
            "prior_legalhk_cases_file",
            "data/processed/legalhk_only/cases.jsonl",
        )
    )
    return {
        str(row.get("case_id"))
        for row in read_jsonl(path)
        if str(row.get("case_id", "")).startswith("legalhk-")
    }


def _flux_expected_run_hash(
    job: dict[str, Any],
    *,
    model_digest: str,
    workflow_hash: str,
    template_hash: str,
    seed: int,
) -> str:
    from .legal_flux_runner import flux_run_hash

    return flux_run_hash(
        job["case"],
        condition=job["condition"],
        phase=job["phase"],
        model_digest=model_digest,
        workflow_hash=workflow_hash,
        template_hash=template_hash,
        seed=seed,
        sample_index=int(job.get("sample_index", 0)),
        temperature=job.get("temperature"),
    )
