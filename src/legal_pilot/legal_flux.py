from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .adaptive_profiles import profile_row
from .config import PROJECT_ROOT, resolve_path
from .embeddings import SimilarityBackend, TfidfSimilarityBackend
from .io_utils import canonical_json, read_jsonl, sha256_text, write_jsonl
from .models import (
    LegalFluxPlanStep,
    LegalFluxAbstractStep,
    LegalFluxTemplate,
    LegalFluxTrajectoryPlan,
    NormalizedCase,
)


FLUX_CONDITIONS = [
    "direct",
    "structured",
    "flux_fixed",
    "flux_adaptive",
    "flux_adaptive_no_review",
    "flux_rf_style",
]
FLUX_PHASES = {"smoke", "template_source", "trajectory_dev", "final_test"}


def resolve_project_file(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def template_pool_path(config: dict[str, Any]) -> Path:
    configured = config["legal_flux"].get(
        "template_pool_file",
        "data/processed/legal_flux/legal_flux_templates_v0.jsonl",
    )
    return resolve_project_file(configured)


def freeze_manifest_path(config: dict[str, Any]) -> Path:
    configured = config["legal_flux"].get(
        "freeze_manifest_file",
        "data/processed/legal_flux/legal_flux_frozen_manifest.json",
    )
    return resolve_project_file(configured)


def load_template_pool(config_or_path: dict[str, Any] | str | Path) -> list[LegalFluxTemplate]:
    path = (
        template_pool_path(config_or_path)
        if isinstance(config_or_path, dict)
        else Path(config_or_path)
    )
    if not path.exists():
        raise FileNotFoundError(f"LegalFlux template pool not found: {path}")
    templates = [LegalFluxTemplate.model_validate(row) for row in read_jsonl(path)]
    validate_template_pool(templates)
    return templates


def write_template_pool(path: Path, templates: list[LegalFluxTemplate]) -> None:
    validate_template_pool(templates)
    write_jsonl(path, [template.model_dump(mode="json") for template in templates])


def validate_template_pool(templates: list[LegalFluxTemplate]) -> None:
    if not templates:
        raise ValueError("LegalFlux template pool is empty.")
    ids = [template.template_id for template in templates]
    duplicates = sorted({template_id for template_id in ids if ids.count(template_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate LegalFlux template IDs: {duplicates}")
    for template in templates:
        if len(template.knowledge_tags) < 2:
            raise ValueError(f"Template {template.template_id} has fewer than two tags.")
        if len(template.reasoning_flow) < 2:
            raise ValueError(
                f"Template {template.template_id} has fewer than two reasoning steps."
            )


def template_pool_hash(templates: list[LegalFluxTemplate]) -> str:
    return sha256_text(
        canonical_json([template.model_dump(mode="json") for template in templates])
    )


def template_document(template: LegalFluxTemplate) -> str:
    return " ".join(
        [
            template.template_name,
            " ".join(template.knowledge_tags),
            template.description,
            template.application_scenario,
            " ".join(template.reasoning_flow),
            template.example_application,
        ]
    )


def rf_template_document(template: LegalFluxTemplate) -> str:
    return " ".join(
        [
            template.template_name,
            " ".join(template.knowledge_tags),
            template.description,
            template.application_scenario,
            " ".join(template.reasoning_flow),
        ]
    )


def abstract_step_query(step: LegalFluxAbstractStep) -> str:
    return (
        f"Step: {step.step_name}\n"
        f"Tags: {', '.join(step.template_tags)}\n"
        f"Purpose: {step.purpose}"
    )


def retrieve_template_for_abstract_step(
    step: LegalFluxAbstractStep,
    templates: list[LegalFluxTemplate],
    *,
    similarity_backend: SimilarityBackend | None = None,
) -> dict[str, Any]:
    if not templates:
        raise ValueError("Cannot retrieve from an empty LegalFlux template pool.")
    step_terms = _abstract_step_match_terms(step)
    exact_candidates = [
        template
        for template in templates
        if step_terms.intersection(_template_match_terms(template))
    ]
    if len(exact_candidates) == 1:
        return {
            "template": exact_candidates[0],
            "similarity": 1.0,
            "retrieval_mode": "exact_unique",
            "exact_candidate_ids": [exact_candidates[0].template_id],
        }
    candidates = exact_candidates or templates
    backend = similarity_backend or TfidfSimilarityBackend()
    query = abstract_step_query(step)
    documents = [rf_template_document(template) for template in candidates]
    scores = backend.similarities(query, documents)
    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    winner, score = ranked[0]
    return {
        "template": winner,
        "similarity": float(score),
        "retrieval_mode": (
            "embedding_ambiguous_exact"
            if exact_candidates
            else "embedding_full_pool"
        )
        if similarity_backend
        else (
            "tfidf_ambiguous_exact"
            if exact_candidates
            else "tfidf_full_pool"
        ),
        "exact_candidate_ids": [template.template_id for template in exact_candidates],
    }


def _abstract_step_match_terms(step: LegalFluxAbstractStep) -> set[str]:
    terms = {
        _normalize_lookup_term(term)
        for term in [step.step_name, *step.template_tags]
        if str(term).strip()
    }
    return {term for term in terms if term}


def _template_match_terms(template: LegalFluxTemplate) -> set[str]:
    terms = {
        _normalize_lookup_term(template.template_name),
        *{
            _normalize_lookup_term(tag)
            for tag in template.knowledge_tags
            if str(tag).strip()
        },
    }
    return {term for term in terms if term}


def _normalize_lookup_term(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower())
    text = re.sub(r"_+", "_", text).strip("_")
    if text.endswith("s") and len(text) > 4:
        text = text[:-1]
    return text


def template_catalog(
    templates: list[LegalFluxTemplate],
    *,
    include_flow: bool = False,
) -> list[dict[str, Any]]:
    catalog = []
    for template in templates:
        row = {
            "template_id": template.template_id,
            "template_name": template.template_name,
            "knowledge_tags": template.knowledge_tags,
            "description": template.description,
            "application_scenario": template.application_scenario,
        }
        if include_flow:
            row["reasoning_flow"] = template.reasoning_flow
        catalog.append(row)
    return catalog


def case_profile(
    case: NormalizedCase,
    *,
    include_reference_metadata: bool = False,
) -> dict[str, Any]:
    row = {
        "plaintiff_claim": case.claim,
        "claim": case.claim,
        "requested_remedy": case.requested_remedy or "",
        "lawsuit_type": case.metadata.get("lawsuit_type", ""),
        "more_facts": "\n".join(case.facts.values()),
        "issues": "\n".join(case.reference_issues) if include_reference_metadata else "",
        "related_laws": (case.authorities or "") if include_reference_metadata else "",
        "relevant_cases": case.metadata.get("relevant_cases", ""),
        "support&reject": case.gold_answer,
    }
    profile = profile_row(row)
    profile["case_id"] = case.case_id
    return profile


def case_profile_text(
    case: NormalizedCase,
    profile: dict[str, Any] | None = None,
    *,
    include_reference_metadata: bool = False,
) -> str:
    profile = profile or case_profile(
        case,
        include_reference_metadata=include_reference_metadata,
    )
    lines = [
        f"claim: {case.claim}",
        f"requested_remedy: {case.requested_remedy or 'not specified'}",
        f"lawsuit_type: {case.metadata.get('lawsuit_type', '')}",
        f"template_families: {profile.get('template_families', '')}",
        f"reasoning_demands: {profile.get('reasoning_demands', '')}",
        f"trajectory_signature: {profile.get('trajectory_signature', '')}",
    ]
    if include_reference_metadata:
        lines.append(f"reference_issues: {'; '.join(case.reference_issues)}")
    return "\n".join(lines)


def retrieve_templates(
    query: str,
    templates: list[LegalFluxTemplate],
    *,
    k: int,
) -> list[dict[str, Any]]:
    documents = [template_document(template) for template in templates]
    scores = TfidfSimilarityBackend().similarities(query, documents)
    ranked = sorted(
        zip(templates, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {
            "template": template,
            "similarity": float(score),
        }
        for template, score in ranked[:k]
    ]


def fixed_trajectory_plan(
    case: NormalizedCase,
    templates: list[LegalFluxTemplate],
    *,
    max_steps: int,
    include_reference_metadata: bool = False,
) -> LegalFluxTrajectoryPlan:
    profile = case_profile(
        case,
        include_reference_metadata=include_reference_metadata,
    )
    tokens = [
        token
        for token in str(profile.get("trajectory", "")).split("|")
        if token and token not in {"case_profile", "final_decision"}
    ]
    selected: list[LegalFluxPlanStep] = []
    used: set[str] = set()
    base_query = case_profile_text(
        case,
        profile,
        include_reference_metadata=include_reference_metadata,
    )
    for token in tokens:
        if len(selected) >= max_steps:
            break
        query = f"{token}\n{base_query}"
        match = _first_unused_template(query, templates, used)
        if match is None:
            break
        used.add(match.template_id)
        step_index = len(selected) + 1
        selected.append(
            LegalFluxPlanStep(
                step_id=f"S{step_index}",
                template_id=match.template_id,
                purpose=f"Use {match.template_name} to address {token.replace('_', ' ')}.",
                expected_artifact="A concise issue/rule/fact finding for final synthesis.",
            )
        )
    if not selected:
        fallback = retrieve_templates(base_query, templates, k=1)[0]["template"]
        selected.append(
            LegalFluxPlanStep(
                step_id="S1",
                template_id=fallback.template_id,
                purpose=f"Use {fallback.template_name} to structure the case analysis.",
                expected_artifact="A concise case analysis finding for final synthesis.",
            )
        )
    return LegalFluxTrajectoryPlan(
        case_profile=case_profile_text(case, profile),
        planned_steps=selected,
        planning_rationale=(
            "Deterministic heuristic trajectory from LegalHK profile labels; no LLM "
            "trajectory revision is used in the fixed condition."
        ),
    )


def sanitize_flux_template(
    template: LegalFluxTemplate,
    *,
    forbidden_terms: list[str] | None = None,
) -> LegalFluxTemplate:
    forbidden_terms = [term for term in (forbidden_terms or []) if term]

    def clean_text(value: str) -> str:
        result = value
        for term in forbidden_terms:
            result = re.sub(re.escape(term), "case-specific term", result, flags=re.I)
        result = re.sub(r"\bF\d+\b", "supplied fact", result)
        result = re.sub(r"\blegalhk-\d+\b", "source case", result, flags=re.I)
        result = re.sub(
            r"(?<!\w)(?:HK\$|US\$|\$|\u00a3|\u20ac|\u00a5)?\s*"
            r"\d[\d,]*(?:\.\d+)?%?(?!\w)",
            "case-specific value",
            result,
        )
        result = re.sub(
            r"\b(?:support|reject)\b",
            "resolve",
            result,
            flags=re.I,
        )
        return " ".join(result.split())

    return template.model_copy(
        update={
            "template_name": clean_text(template.template_name),
            "knowledge_tags": [clean_text(tag) for tag in template.knowledge_tags],
            "description": clean_text(template.description),
            "application_scenario": clean_text(template.application_scenario),
            "reasoning_flow": [clean_text(step) for step in template.reasoning_flow],
            "example_application": clean_text(template.example_application),
        }
    )


def sanitize_plan_steps(
    steps: list[LegalFluxPlanStep],
    *,
    templates_by_id: dict[str, LegalFluxTemplate],
    max_steps: int,
    fallback_query: str,
) -> list[LegalFluxPlanStep]:
    sanitized: list[LegalFluxPlanStep] = []
    used: set[str] = set()
    for raw_step in steps[:max_steps]:
        template_id = raw_step.template_id
        if template_id not in templates_by_id or template_id in used:
            fallback = _first_unused_template(
                f"{raw_step.purpose}\n{fallback_query}",
                list(templates_by_id.values()),
                used,
            )
            if fallback is None:
                continue
            template_id = fallback.template_id
        used.add(template_id)
        sanitized.append(
            LegalFluxPlanStep(
                step_id=f"S{len(sanitized) + 1}",
                template_id=template_id,
                purpose=raw_step.purpose,
                expected_artifact=raw_step.expected_artifact,
            )
        )
    return sanitized


def legal_flux_workflow_hash(config: dict[str, Any]) -> str:
    components = legal_flux_workflow_components(config)
    return sha256_text(canonical_json(components))


def legal_flux_workflow_components(config: dict[str, Any]) -> dict[str, Any]:
    project_root = resolve_path(config, "prompts_dir").parent
    prompt_root = resolve_path(config, "prompts_dir") / "legal_flux"
    schema_root = resolve_path(config, "schemas_dir")
    implementation_names = [
        "adaptive_profiles.py",
        "embeddings.py",
        "legal_flux.py",
        "legal_flux_runner.py",
        "legal_flux_setup.py",
        "models.py",
        "runner.py",
    ]
    template_path = template_pool_path(config)
    components: dict[str, Any] = {
        "legal_flux_config": config["legal_flux"],
        "model": {
            key: config["model"].get(key)
            for key in (
                "name",
                "context_length",
                "temperature",
                "seed",
                "analysis_max_tokens",
                "flux_plan_max_tokens",
                "flux_step_max_tokens",
                "flux_review_max_tokens",
            )
        },
        "prompts": {
            "direct.txt": _file_hash(resolve_path(config, "prompts_dir") / "direct.txt"),
            "structured.txt": _file_hash(
                resolve_path(config, "prompts_dir") / "structured.txt"
            ),
            **{
                f"legal_flux/{path.name}": _file_hash(path)
                for path in sorted(prompt_root.glob("*.txt"))
            },
        },
        "schemas": {
            path.name: _file_hash(path)
            for path in sorted(schema_root.glob("*.json"))
            if path.name
            in {
                "direct_analysis.json",
                "direct_analysis_binary.json",
                "final_analysis.json",
                "final_analysis_binary.json",
                "legal_flux_abstract_plan.json",
                "legal_flux_trajectory_plan.json",
                "legal_flux_step_artifact.json",
                "legal_flux_trajectory_review.json",
                "legal_flux_rf_final_review.json",
                "legal_flux_rf_review.json",
                "legal_flux_template.json",
            }
        },
        "implementation": {
            name: _file_hash(project_root / "src" / "legal_pilot" / name)
            for name in implementation_names
        },
        "template_pool_hash": (
            sha256_text(template_path.read_text(encoding="utf-8"))
            if template_path.exists()
            else None
        ),
    }
    return components


def legal_flux_plan_hash(
    cases: list[NormalizedCase],
    config: dict[str, Any],
    *,
    phase: str,
) -> str:
    payload = [
        {
            "case_id": job["case"].case_id,
            "variant_id": job["case"].variant_id,
            "gold_answer": job["case"].gold_answer,
            "condition": job["condition"],
            "phase": job["phase"],
        }
        for job in build_legal_flux_jobs(cases, config, phase=phase)
    ]
    return sha256_text(canonical_json(payload))


def build_legal_flux_jobs(
    cases: list[NormalizedCase],
    config: dict[str, Any],
    *,
    phase: str,
) -> list[dict[str, Any]]:
    normalized_phase = phase.replace("-", "_")
    if normalized_phase not in FLUX_PHASES - {"template_source"}:
        raise ValueError(f"Unknown LegalFlux generation phase: {phase}")
    selected = [
        case
        for case in cases
        if case.metadata.get("selection_split") == normalized_phase
    ]
    if not selected:
        raise ValueError(f"No cases found for LegalFlux phase {normalized_phase}.")
    conditions = config["legal_flux"].get("conditions", FLUX_CONDITIONS)
    return [
        {
            "case": case,
            "condition": condition,
            "phase": normalized_phase,
            "sample_index": 0,
            "temperature": config["model"]["temperature"],
            "seed": config["model"]["seed"],
        }
        for case in selected
        for condition in conditions
    ]


def _first_unused_template(
    query: str,
    templates: list[LegalFluxTemplate],
    used: set[str],
) -> LegalFluxTemplate | None:
    matches = retrieve_templates(query, templates, k=len(templates))
    for match in matches:
        template = match["template"]
        if template.template_id not in used:
            return template
    return None


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return sha256_text(path.read_text(encoding="utf-8"))
