from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, resolve_path
from .embeddings import SimilarityBackend, TfidfSimilarityBackend
from .io_utils import canonical_json, read_jsonl, sha256_text, write_jsonl
from .models import LegalFluxAbstractStep, LegalFluxTemplate, NormalizedCase


FLUX_CONDITIONS = ["direct", "structured", "flux_rf_style"]
FLUX_PHASES = {
    "smoke",
    "template_source",
    "planner_train",
    "trajectory_dev",
    "final_test",
}


def resolve_project_file(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def template_pool_path(config: dict[str, Any]) -> Path:
    configured = config["legal_flux"].get(
        "template_pool_file",
        "templates/legal_flux_templates_v0.jsonl",
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
    exclude_template_ids: set[str] | None = None,
) -> dict[str, Any]:
    if not templates:
        raise ValueError("Cannot retrieve from an empty LegalFlux template pool.")
    exclude_template_ids = exclude_template_ids or set()
    available_templates = [
        template for template in templates if template.template_id not in exclude_template_ids
    ] or templates
    step_terms = _abstract_step_match_terms(step)
    exact_candidates = [
        template
        for template in available_templates
        if step_terms.intersection(_template_match_terms(template))
    ]
    if len(exact_candidates) == 1:
        return {
            "template": exact_candidates[0],
            "similarity": 1.0,
            "retrieval_mode": "exact_unique",
            "exact_candidate_ids": [exact_candidates[0].template_id],
        }

    candidates = exact_candidates or available_templates
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
    if similarity_backend:
        mode = "embedding_ambiguous_exact" if exact_candidates else "embedding_full_pool"
    else:
        mode = "tfidf_ambiguous_exact" if exact_candidates else "tfidf_full_pool"
    return {
        "template": winner,
        "similarity": float(score),
        "retrieval_mode": mode,
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
            r"(?<!\w)(?:HK\$|US\$|\$|\u00a3|\u20ac|\u00a5)\s*"
            r"\d[\d,]*(?:\.\d+)?%?(?!\w)",
            "case-specific value",
            result,
        )
        result = re.sub(
            r"(?<!\w)\d+(?:\.\d+)?%(?!\w)",
            "case-specific value",
            result,
        )
        result = re.sub(
            r"(?<!\w)\d[\d,]{2,}(?:\.\d+)?(?!\w)",
            "case-specific value",
            result,
        )
        result = re.sub(r"\b(?:support|reject)\b", "resolve", result, flags=re.I)
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


def legal_flux_workflow_hash(config: dict[str, Any]) -> str:
    return sha256_text(canonical_json(legal_flux_workflow_components(config)))


def legal_flux_workflow_components(config: dict[str, Any]) -> dict[str, Any]:
    project_root = resolve_path(config, "prompts_dir").parent
    prompt_root = resolve_path(config, "prompts_dir")
    schema_root = resolve_path(config, "schemas_dir")
    implementation_names = [
        "__main__.py",
        "clients.py",
        "config.py",
        "embeddings.py",
        "io_utils.py",
        "ledger.py",
        "legal_flux.py",
        "legal_flux_chatgpt.py",
        "legal_flux_evaluation.py",
        "legal_flux_runner.py",
        "legal_flux_setup.py",
        "legal_flux_training.py",
        "legalhk_data.py",
        "legalhk_selection.py",
        "models.py",
        "prompting.py",
        "runner.py",
        "scoring.py",
    ]
    prompt_names = [
        "direct.txt",
        "structured.txt",
        "legal_flux/rf_plan.txt",
        "legal_flux/instantiate.txt",
        "legal_flux/rf_review.txt",
    ]
    schema_names = [
        "direct_analysis.json",
        "direct_analysis_binary.json",
        "final_analysis.json",
        "final_analysis_binary.json",
        "legal_flux_abstract_plan.json",
        "legal_flux_rf_final_review.json",
        "legal_flux_rf_review.json",
        "legal_flux_step_artifact.json",
        "legal_flux_template.json",
    ]
    template_path = template_pool_path(config)
    return {
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
            name: _file_hash(prompt_root.joinpath(*name.split("/")))
            for name in prompt_names
        },
        "schemas": {name: _file_hash(schema_root / name) for name in schema_names},
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
            "sample_index": job["sample_index"],
        }
        for job in build_legal_flux_jobs(cases, config, phase=phase)
    ]
    return sha256_text(canonical_json(payload))


def build_legal_flux_jobs(
    cases: list[NormalizedCase],
    config: dict[str, Any],
    *,
    phase: str,
    sample_count: int | None = None,
    case_limit: int | None = None,
) -> list[dict[str, Any]]:
    normalized_phase = phase.replace("-", "_")
    if normalized_phase not in FLUX_PHASES - {"template_source"}:
        raise ValueError(f"Unknown LegalFlux generation phase: {phase}")
    selected = [
        case for case in cases if case.metadata.get("selection_split") == normalized_phase
    ]
    if case_limit is not None:
        if case_limit < 1:
            raise ValueError("case_limit must be at least 1.")
        selected = selected[:case_limit]
    if not selected:
        raise ValueError(f"No cases found for LegalFlux phase {normalized_phase}.")
    if normalized_phase == "planner_train":
        conditions = config["legal_flux"].get(
            "planner_train_conditions", ["flux_rf_style"]
        )
        sample_count = (
            int(sample_count)
            if sample_count is not None
            else int(config["legal_flux"].get("planner_train_samples", 1))
        )
        temperature = float(
            config["legal_flux"].get(
                "planner_train_temperature", config["model"]["temperature"]
            )
        )
    else:
        conditions = config["legal_flux"].get("conditions", FLUX_CONDITIONS)
        sample_count = 1 if sample_count is None else int(sample_count)
        temperature = config["model"]["temperature"]
    unsupported = sorted(set(conditions) - set(FLUX_CONDITIONS))
    if unsupported:
        raise ValueError(f"Unsupported LegalFlux conditions in cleaned repo: {unsupported}")
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1.")
    return [
        {
            "case": case,
            "condition": condition,
            "phase": normalized_phase,
            "sample_index": sample_index,
            "temperature": temperature,
            "seed": int(config["model"]["seed"]) + sample_index,
        }
        for case in selected
        for condition in conditions
        for sample_index in range(sample_count)
    ]


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return sha256_text(path.read_text(encoding="utf-8"))
