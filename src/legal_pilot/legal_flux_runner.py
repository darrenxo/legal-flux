from __future__ import annotations

import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .clients import OllamaClient, OllamaResponseError
from .config import resolve_path
from .embeddings import OllamaEmbeddingBackend, SimilarityBackend, TfidfSimilarityBackend
from .io_utils import canonical_json, latest_by_run_hash, read_jsonl, sha256_text
from .ledger import JsonlLedger, make_run_hash
from .legal_flux import (
    abstract_step_query,
    build_legal_flux_jobs,
    case_profile,
    case_profile_text,
    fixed_trajectory_plan,
    legal_flux_workflow_hash,
    load_template_pool,
    retrieve_template_for_abstract_step,
    retrieve_templates,
    sanitize_plan_steps,
    template_catalog,
    template_pool_hash,
)
from .legal_flux_setup import assert_legal_flux_frozen
from .models import (
    FinalAnalysis,
    LegalFluxAbstractPlan,
    LegalFluxAbstractStep,
    LegalFluxPlanStep,
    LegalFluxRfReview,
    LegalFluxStepArtifact,
    LegalFluxTemplate,
    LegalFluxTrajectoryPlan,
    LegalFluxTrajectoryReview,
    NormalizedCase,
)
from .prompting import render_prompt
from .runner import (
    _execute_condition,
    _final_analysis_schema_path,
    _fold_extra_fields_into_text,
    _load_schema,
    _normalize_final_analysis_payload,
    _preview_prompt,
    _response_trace,
    load_cases,
)


def run_legal_flux_generation(
    config: dict[str, Any],
    *,
    phase: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    normalized_phase = phase.replace("-", "_")
    cases = load_cases(config)
    jobs = build_legal_flux_jobs(cases, config, phase=normalized_phase)
    run_dir = resolve_path(config, "runs_dir") / normalized_phase
    if dry_run:
        return {
            "phase": normalized_phase,
            "jobs": len(jobs),
            "conditions": sorted({job["condition"] for job in jobs}),
            "cases": len({job["case"].case_id for job in jobs}),
            "run_dir": str(run_dir),
            "dry_run": True,
        }

    templates = load_template_pool(config)
    template_hash = template_pool_hash(templates)
    workflow_hash = legal_flux_workflow_hash(config)
    similarity_backend = (
        _build_rf_similarity_backend(config)
        if any(job["condition"] == "flux_rf_style" for job in jobs)
        else None
    )
    client = OllamaClient(config["model"]["base_url"], config["model"]["timeout_seconds"])
    model_info = client.model_info(config["model"]["name"])
    if not model_info:
        client.close()
        raise RuntimeError(
            f"Model {config['model']['name']!r} is not installed in Ollama."
        )
    digest = model_info.get("digest", "unknown")
    if normalized_phase == "final_test":
        assert_legal_flux_frozen(
            config,
            model_digest=digest,
            workflow_hash=workflow_hash,
            template_hash=template_hash,
        )

    ledger = JsonlLedger(run_dir / "generations.jsonl")
    planned = [
        {
            "run_hash": flux_run_hash(
                job["case"],
                condition=job["condition"],
                phase=normalized_phase,
                model_digest=digest,
                workflow_hash=workflow_hash,
                template_hash=template_hash,
                seed=job["seed"],
            ),
            "case_id": job["case"].case_id,
            "condition": job["condition"],
            "phase": normalized_phase,
        }
        for job in jobs
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_plan.json").write_text(
        json.dumps(
            {
                "phase": normalized_phase,
                "model_digest": digest,
                "workflow_hash": workflow_hash,
                "template_pool_hash": template_hash,
                "job_count": len(planned),
                "jobs": planned,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    completed = 0
    skipped = 0
    errors = 0
    try:
        for job in jobs:
            record = _run_flux_job(
                client,
                config,
                job,
                templates=templates,
                model_digest=digest,
                workflow_hash=workflow_hash,
                template_hash=template_hash,
                ledger=ledger,
                similarity_backend=similarity_backend,
            )
            if record is None:
                skipped += 1
                continue
            completed += 1
            if record.get("status") != "ok":
                errors += 1
    finally:
        client.close()
        if hasattr(similarity_backend, "close"):
            similarity_backend.close()
    return {
        "phase": normalized_phase,
        "jobs": len(jobs),
        "completed": completed,
        "skipped": skipped,
        "errors": errors,
        "run_dir": str(run_dir),
        "model_digest": digest,
        "workflow_hash": workflow_hash,
        "template_pool_hash": template_hash,
    }


def flux_run_hash(
    case: NormalizedCase,
    *,
    condition: str,
    phase: str,
    model_digest: str,
    workflow_hash: str,
    template_hash: str,
    seed: int,
) -> str:
    return make_run_hash(
        dataset=case.dataset,
        case_id=case.case_id,
        variant_id=case.variant_id,
        condition=condition,
        phase=phase,
        model_digest=model_digest,
        workflow_hash=workflow_hash,
        template_pool_hash=template_hash,
        seed=seed,
    )


def _run_flux_job(
    client: OllamaClient,
    config: dict[str, Any],
    job: dict[str, Any],
    *,
    templates: list[LegalFluxTemplate],
    model_digest: str,
    workflow_hash: str,
    template_hash: str,
    ledger: JsonlLedger,
    similarity_backend: SimilarityBackend | None = None,
) -> dict[str, Any] | None:
    case: NormalizedCase = job["case"]
    condition = job["condition"]
    phase = job["phase"]
    run_hash = flux_run_hash(
        case,
        condition=condition,
        phase=phase,
        model_digest=model_digest,
        workflow_hash=workflow_hash,
        template_hash=template_hash,
        seed=job["seed"],
    )
    if ledger.contains(run_hash):
        return None
    prompt_hash = _condition_prompt_hash(config, case, condition, templates)
    base = {
        "run_hash": run_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": case.dataset,
        "case_id": case.case_id,
        "variant_id": case.variant_id,
        "condition": condition,
        "phase": phase,
        "prompt_hash": prompt_hash,
        "model_name": config["model"]["name"],
        "model_digest": model_digest,
        "workflow_hash": workflow_hash,
        "template_pool_hash": template_hash,
        "seed": job["seed"],
        "sample_index": job["sample_index"],
        "decoding": {
            "temperature": job["temperature"],
            "context_length": config["model"]["context_length"],
        },
        "gold_answer": case.gold_answer,
        "metadata": case.metadata,
    }
    try:
        if condition in {"direct", "structured"}:
            analysis, trace = _execute_condition(
                client,
                config,
                case,
                condition,
                job["temperature"],
                job["seed"],
            )
        elif condition in {
            "flux_fixed",
            "flux_adaptive",
            "flux_adaptive_no_review",
            "flux_rf_style",
        }:
            analysis, trace = _execute_flux_case(
                client,
                config,
                case,
                templates=templates,
                condition=condition,
                similarity_backend=similarity_backend,
            )
        else:
            raise ValueError(f"Unknown LegalFlux condition: {condition}")
        record = {
            **base,
            "status": "ok",
            "raw_response": trace["raw_response"],
            "parsed_json": analysis.model_dump(mode="json"),
            "trajectory_plan": trace.get("trajectory_plan"),
            "executed_steps": trace.get("executed_steps"),
            "trajectory_reviews": trace.get("trajectory_reviews"),
            "retrieved_template_ids": trace.get("retrieved_template_ids"),
            "prompt_hashes": trace.get("prompt_hashes", {}),
            "elapsed_seconds": trace["elapsed_seconds"],
            "prompt_tokens": trace["prompt_tokens"],
            "output_tokens": trace["output_tokens"],
            "schema_errors": trace.get("schema_errors", []),
            "repair_actions": trace.get("repair_actions", []),
            "calls": trace.get("calls", 1),
        }
    except Exception as exc:
        record = {
            **base,
            "status": "error",
            "raw_response": exc.raw_text if isinstance(exc, OllamaResponseError) else None,
            "parsed_json": None,
            "trajectory_plan": None,
            "executed_steps": None,
            "trajectory_reviews": None,
            "retrieved_template_ids": None,
            "prompt_hashes": {},
            "elapsed_seconds": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "schema_errors": [str(exc)],
            "repair_actions": [],
            "calls": 0,
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
    ledger.append(record)
    return record


def _execute_flux_case(
    client: Any,
    config: dict[str, Any],
    case: NormalizedCase,
    *,
    templates: list[LegalFluxTemplate],
    condition: str,
    similarity_backend: SimilarityBackend | None = None,
) -> tuple[FinalAnalysis, dict[str, Any]]:
    if condition == "flux_rf_style":
        return _execute_rf_style_case(
            client,
            config,
            case,
            templates=templates,
            similarity_backend=similarity_backend,
        )

    common = _common_generation_settings(config)
    schema_dir = resolve_path(config, "schemas_dir")
    max_steps = int(config["legal_flux"].get("max_steps", 4))
    include_reference = bool(
        config["legal_flux"].get("profile_uses_reference_metadata", False)
    )
    profile = case_profile(case, include_reference_metadata=include_reference)
    profile_text = case_profile_text(
        case,
        profile,
        include_reference_metadata=include_reference,
    )
    planner_k = int(config["legal_flux"].get("planner_catalog_size", 24))
    retrieved = retrieve_templates(profile_text, templates, k=min(planner_k, len(templates)))
    catalog_templates = [item["template"] for item in retrieved]
    templates_by_id = {template.template_id: template for template in templates}

    raw_parts: list[str] = []
    prompt_hashes: dict[str, str] = {}
    elapsed = 0.0
    prompt_tokens = 0
    output_tokens = 0
    calls = 0
    repairs: list[str] = []
    schema_errors: list[str] = []
    reviews: list[LegalFluxTrajectoryReview] = []
    artifacts: list[LegalFluxStepArtifact] = []

    if condition == "flux_fixed":
        plan = fixed_trajectory_plan(
            case,
            templates,
            max_steps=max_steps,
            include_reference_metadata=include_reference,
        )
        prompt_hashes["plan"] = sha256_text(canonical_json(plan.model_dump(mode="json")))
    else:
        plan_prompt, plan_hash = render_prompt(
            config,
            "legal_flux/plan",
            case,
            case_profile=profile_text,
            template_catalog=template_catalog(catalog_templates),
            max_steps=max_steps,
        )
        response = client.generate(
            prompt=plan_prompt,
            schema=_load_schema(schema_dir / "legal_flux_trajectory_plan.json"),
            max_tokens=config["model"]["flux_plan_max_tokens"],
            **common,
        )
        normalized_plan, plan_repairs = _normalize_plan_payload(
            response.parsed,
            max_steps=max_steps,
        )
        repairs.extend(plan_repairs)
        plan = LegalFluxTrajectoryPlan.model_validate(normalized_plan)
        repaired_steps = sanitize_plan_steps(
            plan.planned_steps,
            templates_by_id=templates_by_id,
            max_steps=max_steps,
            fallback_query=profile_text,
        )
        if len(repaired_steps) != len(plan.planned_steps) or any(
            left.model_dump() != right.model_dump()
            for left, right in zip(repaired_steps, plan.planned_steps, strict=False)
        ):
            repairs.append("planner_step_template_ids_sanitized")
        plan = plan.model_copy(update={"planned_steps": repaired_steps})
        raw_parts.append(response.raw_text)
        prompt_hashes["plan"] = plan_hash
        elapsed += response.elapsed_seconds
        prompt_tokens += response.prompt_tokens or 0
        output_tokens += response.output_tokens or 0
        calls += 1

    remaining = list(plan.planned_steps[:max_steps])
    while remaining and len(artifacts) < max_steps:
        step = remaining.pop(0)
        artifact, step_trace = _instantiate_step(
            client,
            config,
            case,
            step=step,
            template=templates_by_id[step.template_id],
            prior_artifacts=artifacts,
            common=common,
        )
        artifacts.append(artifact)
        raw_parts.append(step_trace["raw_response"])
        prompt_hashes[f"instantiate_{artifact.step_id}"] = step_trace["prompt_hash"]
        elapsed += step_trace["elapsed_seconds"]
        prompt_tokens += step_trace["prompt_tokens"] or 0
        output_tokens += step_trace["output_tokens"] or 0
        calls += 1
        repairs.extend(step_trace["repair_actions"])
        schema_errors.extend(step_trace["schema_errors"])

        if condition != "flux_adaptive" or not remaining:
            continue
        review, review_trace = _review_trajectory(
            client,
            config,
            case,
            plan=plan,
            artifacts=artifacts,
            remaining=remaining,
            catalog=catalog_templates,
            common=common,
            max_steps=max_steps - len(artifacts),
            fallback_query=profile_text,
            templates_by_id=templates_by_id,
        )
        if review.revised_remaining_steps:
            review = review.model_copy(
                update={
                    "revised_remaining_steps": _renumber_remaining_steps(
                        review.revised_remaining_steps,
                        start_index=len(artifacts) + 1,
                    )
                }
            )
        reviews.append(review)
        raw_parts.append(review_trace["raw_response"])
        prompt_hashes[f"review_{artifact.step_id}"] = review_trace["prompt_hash"]
        elapsed += review_trace["elapsed_seconds"]
        prompt_tokens += review_trace["prompt_tokens"] or 0
        output_tokens += review_trace["output_tokens"] or 0
        calls += 1
        repairs.extend(review_trace["repair_actions"])
        schema_errors.extend(review_trace["schema_errors"])
        if review.decision == "stop":
            break
        if review.decision == "revise":
            remaining = review.revised_remaining_steps[: max_steps - len(artifacts)]
            repairs.append("adaptive_remaining_trajectory_revised")

    analysis, final_trace = _finalize_flux_analysis(
        client,
        config,
        case,
        plan=plan,
        artifacts=artifacts,
        reviews=reviews,
        common=common,
    )
    raw_parts.append(final_trace["raw_response"])
    prompt_hashes["finalize"] = final_trace["prompt_hash"]
    elapsed += final_trace["elapsed_seconds"]
    prompt_tokens += final_trace["prompt_tokens"] or 0
    output_tokens += final_trace["output_tokens"] or 0
    calls += 1
    repairs.extend(final_trace["repair_actions"])
    schema_errors.extend(final_trace["schema_errors"])

    trace = {
        "raw_response": "\n---CALL---\n".join(raw_parts),
        "prompt_hashes": prompt_hashes,
        "trajectory_plan": plan.model_dump(mode="json"),
        "executed_steps": [artifact.model_dump(mode="json") for artifact in artifacts],
        "trajectory_reviews": [
            review.model_dump(mode="json") for review in reviews
        ],
        "retrieved_template_ids": [template.template_id for template in catalog_templates],
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "schema_errors": schema_errors,
        "repair_actions": repairs,
        "calls": calls,
    }
    return analysis, trace


def _execute_rf_style_case(
    client: Any,
    config: dict[str, Any],
    case: NormalizedCase,
    *,
    templates: list[LegalFluxTemplate],
    similarity_backend: SimilarityBackend | None = None,
) -> tuple[FinalAnalysis, dict[str, Any]]:
    common = _common_generation_settings(config)
    schema_dir = resolve_path(config, "schemas_dir")
    max_steps = int(config["legal_flux"].get("max_steps", 4))
    include_reference = bool(
        config["legal_flux"].get("profile_uses_reference_metadata", False)
    )
    profile = case_profile(case, include_reference_metadata=include_reference)
    profile_text = case_profile_text(
        case,
        profile,
        include_reference_metadata=include_reference,
    )

    raw_parts: list[str] = []
    prompt_hashes: dict[str, str] = {}
    elapsed = 0.0
    prompt_tokens = 0
    output_tokens = 0
    calls = 0
    repairs: list[str] = []
    schema_errors: list[str] = []
    reviews: list[LegalFluxRfReview] = []
    artifacts: list[LegalFluxStepArtifact] = []
    selected_templates: list[dict[str, Any]] = []

    plan_prompt, plan_hash = render_prompt(
        config,
        "legal_flux/rf_plan",
        case,
        case_profile=profile_text,
        max_steps=max_steps,
    )
    response = client.generate(
        prompt=plan_prompt,
        schema=_load_schema(schema_dir / "legal_flux_abstract_plan.json"),
        max_tokens=config["model"]["flux_plan_max_tokens"],
        **common,
    )
    normalized_plan, plan_repairs = _normalize_abstract_plan_payload(
        response.parsed,
        max_steps=max_steps,
    )
    repairs.extend(plan_repairs)
    abstract_plan = LegalFluxAbstractPlan.model_validate(normalized_plan)
    raw_parts.append(response.raw_text)
    prompt_hashes["rf_plan"] = plan_hash
    elapsed += response.elapsed_seconds
    prompt_tokens += response.prompt_tokens or 0
    output_tokens += response.output_tokens or 0
    calls += 1

    remaining = list(abstract_plan.planned_steps[:max_steps])
    analysis: FinalAnalysis | None = None
    while remaining and len(artifacts) < max_steps and analysis is None:
        abstract_step = remaining.pop(0)
        retrieval = retrieve_template_for_abstract_step(
            abstract_step,
            templates,
            similarity_backend=similarity_backend,
        )
        template = retrieval["template"]
        selected_record = {
            "step_id": abstract_step.step_id,
            "step_name": abstract_step.step_name,
            "template_tags": abstract_step.template_tags,
            "template_id": template.template_id,
            "template_name": template.template_name,
            "retrieval_mode": retrieval["retrieval_mode"],
            "similarity": retrieval["similarity"],
            "exact_candidate_ids": retrieval["exact_candidate_ids"],
        }
        selected_templates.append(selected_record)
        step = _abstract_step_to_plan_step(abstract_step, template)
        artifact, step_trace = _instantiate_step(
            client,
            config,
            case,
            step=step,
            template=template,
            prior_artifacts=artifacts,
            common=common,
        )
        artifacts.append(artifact)
        raw_parts.append(step_trace["raw_response"])
        prompt_hashes[f"instantiate_{artifact.step_id}"] = step_trace["prompt_hash"]
        elapsed += step_trace["elapsed_seconds"]
        prompt_tokens += step_trace["prompt_tokens"] or 0
        output_tokens += step_trace["output_tokens"] or 0
        calls += 1
        repairs.extend(step_trace["repair_actions"])
        schema_errors.extend(step_trace["schema_errors"])

        review, review_trace = _review_rf_trajectory(
            client,
            config,
            case,
            abstract_plan=abstract_plan,
            artifacts=artifacts,
            remaining=remaining,
            selected_templates=selected_templates,
            common=common,
            max_steps=max_steps - len(artifacts),
            force_final_answer=not remaining or len(artifacts) >= max_steps,
        )
        reviews.append(review)
        raw_parts.append(review_trace["raw_response"])
        prompt_hashes[f"rf_review_{artifact.step_id}"] = review_trace["prompt_hash"]
        elapsed += review_trace["elapsed_seconds"]
        prompt_tokens += review_trace["prompt_tokens"] or 0
        output_tokens += review_trace["output_tokens"] or 0
        calls += 1
        repairs.extend(review_trace["repair_actions"])
        schema_errors.extend(review_trace["schema_errors"])
        if review.decision == "final_answer":
            analysis = _analysis_from_rf_review(review)
            break
        if review.decision == "revise":
            remaining = _renumber_abstract_remaining_steps(
                review.revised_remaining_steps[: max_steps - len(artifacts)],
                start_index=len(artifacts) + 1,
            )
            repairs.append("rf_remaining_trajectory_revised")

    if analysis is None and artifacts:
        review, review_trace = _review_rf_trajectory(
            client,
            config,
            case,
            abstract_plan=abstract_plan,
            artifacts=artifacts,
            remaining=[],
            selected_templates=selected_templates,
            common=common,
            max_steps=0,
            force_final_answer=True,
        )
        reviews.append(review)
        raw_parts.append(review_trace["raw_response"])
        prompt_hashes["rf_review_final"] = review_trace["prompt_hash"]
        elapsed += review_trace["elapsed_seconds"]
        prompt_tokens += review_trace["prompt_tokens"] or 0
        output_tokens += review_trace["output_tokens"] or 0
        calls += 1
        repairs.extend(review_trace["repair_actions"])
        schema_errors.extend(review_trace["schema_errors"])
        if review.decision == "final_answer":
            analysis = _analysis_from_rf_review(review)

    if analysis is None:
        raise RuntimeError("RF-style trajectory ended without a final_answer review.")

    trace = {
        "raw_response": "\n---CALL---\n".join(raw_parts),
        "prompt_hashes": prompt_hashes,
        "trajectory_plan": abstract_plan.model_dump(mode="json"),
        "executed_steps": [artifact.model_dump(mode="json") for artifact in artifacts],
        "trajectory_reviews": [review.model_dump(mode="json") for review in reviews],
        "retrieved_template_ids": [
            item["template_id"] for item in selected_templates
        ],
        "selected_templates": selected_templates,
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "schema_errors": schema_errors,
        "repair_actions": repairs,
        "calls": calls,
    }
    return analysis, trace


def _instantiate_step(
    client: Any,
    config: dict[str, Any],
    case: NormalizedCase,
    *,
    step: LegalFluxPlanStep,
    template: LegalFluxTemplate,
    prior_artifacts: list[LegalFluxStepArtifact],
    common: dict[str, Any],
) -> tuple[LegalFluxStepArtifact, dict[str, Any]]:
    prompt, prompt_hash = render_prompt(
        config,
        "legal_flux/instantiate",
        case,
        trajectory_step=step.model_dump(mode="json"),
        selected_template=template.model_dump(mode="json"),
        prior_artifacts=[artifact.model_dump(mode="json") for artifact in prior_artifacts],
    )
    response = client.generate(
        prompt=prompt,
        schema=_load_schema(
            resolve_path(config, "schemas_dir") / "legal_flux_step_artifact.json"
        ),
        max_tokens=config["model"]["flux_step_max_tokens"],
        **common,
    )
    repaired, repairs = _normalize_step_artifact_payload(response.parsed, step)
    artifact = LegalFluxStepArtifact.model_validate(repaired)
    trace = _response_trace(response)
    trace["prompt_hash"] = prompt_hash
    trace["repair_actions"].extend(repairs)
    return artifact, trace


def _review_rf_trajectory(
    client: Any,
    config: dict[str, Any],
    case: NormalizedCase,
    *,
    abstract_plan: LegalFluxAbstractPlan,
    artifacts: list[LegalFluxStepArtifact],
    remaining: list[LegalFluxAbstractStep],
    selected_templates: list[dict[str, Any]],
    common: dict[str, Any],
    max_steps: int,
    force_final_answer: bool = False,
) -> tuple[LegalFluxRfReview, dict[str, Any]]:
    prompt, prompt_hash = render_prompt(
        config,
        "legal_flux/rf_review",
        case,
        abstract_plan=abstract_plan.model_dump(mode="json"),
        executed_artifacts=[artifact.model_dump(mode="json") for artifact in artifacts],
        remaining_steps=[step.model_dump(mode="json") for step in remaining],
        selected_templates=selected_templates,
        max_steps=max_steps,
        finalization_requirement=(
            "No remaining abstract steps are available. You must choose "
            'decision "final_answer" and provide final_decision as exactly '
            '"support" or "reject".'
            if force_final_answer
            else "If a useful remaining step is available, you may continue or revise; otherwise choose final_answer."
        ),
    )
    response = client.generate(
        prompt=prompt,
        schema=_load_schema(
            resolve_path(config, "schemas_dir")
            / (
                "legal_flux_rf_final_review.json"
                if force_final_answer
                else "legal_flux_rf_review.json"
            )
        ),
        max_tokens=config["model"]["flux_review_max_tokens"],
        **common,
    )
    normalized_review, repairs = _normalize_rf_review_payload(response.parsed)
    review = LegalFluxRfReview.model_validate(normalized_review)
    trace = _response_trace(response)
    trace["prompt_hash"] = prompt_hash
    trace["repair_actions"].extend(repairs)
    return review, trace


def _abstract_step_to_plan_step(
    step: LegalFluxAbstractStep,
    template: LegalFluxTemplate,
) -> LegalFluxPlanStep:
    return LegalFluxPlanStep(
        step_id=step.step_id,
        template_id=template.template_id,
        purpose=(
            f"{step.step_name}\n"
            f"Tags: {', '.join(step.template_tags)}\n"
            f"Purpose: {step.purpose}"
        ),
        expected_artifact=(
            "A concise intermediate legal finding for this abstract step, "
            "grounded in supplied facts."
        ),
    )


def _analysis_from_rf_review(review: LegalFluxRfReview) -> FinalAnalysis:
    if review.final_decision not in {"support", "reject"}:
        raise ValueError("RF-style final_answer review did not include support/reject.")
    return FinalAnalysis(
        issue_conclusions=[],
        final_decision=review.final_decision,
        final_rationale=review.final_rationale or review.rationale,
    )


def _review_trajectory(
    client: Any,
    config: dict[str, Any],
    case: NormalizedCase,
    *,
    plan: LegalFluxTrajectoryPlan,
    artifacts: list[LegalFluxStepArtifact],
    remaining: list[LegalFluxPlanStep],
    catalog: list[LegalFluxTemplate],
    common: dict[str, Any],
    max_steps: int,
    fallback_query: str,
    templates_by_id: dict[str, LegalFluxTemplate],
) -> tuple[LegalFluxTrajectoryReview, dict[str, Any]]:
    prompt, prompt_hash = render_prompt(
        config,
        "legal_flux/review",
        case,
        trajectory_plan=plan.model_dump(mode="json"),
        executed_artifacts=[artifact.model_dump(mode="json") for artifact in artifacts],
        remaining_steps=[step.model_dump(mode="json") for step in remaining],
        template_catalog=template_catalog(catalog),
        max_steps=max_steps,
    )
    response = client.generate(
        prompt=prompt,
        schema=_load_schema(
            resolve_path(config, "schemas_dir") / "legal_flux_trajectory_review.json"
        ),
        max_tokens=config["model"]["flux_review_max_tokens"],
        **common,
    )
    normalized_review, repairs = _normalize_review_payload(response.parsed)
    review = LegalFluxTrajectoryReview.model_validate(normalized_review)
    if review.revised_remaining_steps:
        revised = sanitize_plan_steps(
            review.revised_remaining_steps,
            templates_by_id=templates_by_id,
            max_steps=max_steps,
            fallback_query=fallback_query,
        )
        if len(revised) != len(review.revised_remaining_steps) or any(
            left.model_dump() != right.model_dump()
            for left, right in zip(revised, review.revised_remaining_steps, strict=False)
        ):
            repairs.append("review_revised_steps_sanitized")
        review = review.model_copy(update={"revised_remaining_steps": revised})
    trace = _response_trace(response)
    trace["prompt_hash"] = prompt_hash
    trace["repair_actions"].extend(repairs)
    return review, trace


def _finalize_flux_analysis(
    client: Any,
    config: dict[str, Any],
    case: NormalizedCase,
    *,
    plan: LegalFluxTrajectoryPlan,
    artifacts: list[LegalFluxStepArtifact],
    reviews: list[LegalFluxTrajectoryReview],
    common: dict[str, Any],
) -> tuple[FinalAnalysis, dict[str, Any]]:
    prompt, prompt_hash = render_prompt(
        config,
        "legal_flux/finalize",
        case,
        trajectory_plan=plan.model_dump(mode="json"),
        executed_artifacts=[artifact.model_dump(mode="json") for artifact in artifacts],
        trajectory_reviews=[review.model_dump(mode="json") for review in reviews],
    )
    response = client.generate(
        prompt=prompt,
        schema=_load_schema(_final_analysis_schema_path(config)),
        max_tokens=config["model"]["analysis_max_tokens"],
        **common,
    )
    normalized, repairs = _normalize_final_analysis_payload(response.parsed)
    analysis = FinalAnalysis.model_validate(normalized)
    trace = _response_trace(response)
    trace["prompt_hash"] = prompt_hash
    trace["repair_actions"].extend(repairs)
    return analysis, trace


def _condition_prompt_hash(
    config: dict[str, Any],
    case: NormalizedCase,
    condition: str,
    templates: list[LegalFluxTemplate],
) -> str:
    if condition in {"direct", "structured"}:
        _, prompt_hash = _preview_prompt(config, case, condition)
        return prompt_hash
    profile = case_profile_text(case)
    if condition == "flux_rf_style":
        payload = {
            "condition": condition,
            "profile": profile,
            "template_pool_hash": template_pool_hash(templates),
            "max_steps": config["legal_flux"].get("max_steps", 4),
            "rf_retrieval_backend": config["legal_flux"].get(
                "rf_retrieval_backend", "ollama_embedding"
            ),
            "rf_embedding_model": config["legal_flux"].get(
                "rf_embedding_model", "bge-m3:latest"
            ),
        }
        return sha256_text(canonical_json(payload))
    retrieved = retrieve_templates(
        profile,
        templates,
        k=min(int(config["legal_flux"].get("planner_catalog_size", 24)), len(templates)),
    )
    payload = {
        "condition": condition,
        "profile": profile,
        "template_ids": [item["template"].template_id for item in retrieved],
        "max_steps": config["legal_flux"].get("max_steps", 4),
    }
    return sha256_text(canonical_json(payload))


def _normalize_abstract_plan_payload(
    payload: dict[str, Any] | None,
    *,
    max_steps: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(payload, dict):
        return payload, []
    repaired = json.loads(json.dumps(payload))
    repairs: list[str] = []
    steps = repaired.get("planned_steps")
    if isinstance(steps, dict):
        steps = [steps]
        repaired["planned_steps"] = steps
        repairs.append("abstract_planned_steps_wrapped_as_array")
    if isinstance(steps, list):
        normalized_steps = []
        for step in steps[:max_steps]:
            if not isinstance(step, dict):
                repairs.append("abstract_invalid_step_removed")
                continue
            step = _unwrap_nested_abstract_step_object(step)
            index = len(normalized_steps) + 1
            value = step.get("step_id")
            if not isinstance(value, str):
                step["step_id"] = f"S{value}" if value is not None else f"S{index}"
                repairs.append("abstract_step_id_coerced_to_string")
            elif value.isdigit():
                step["step_id"] = f"S{value}"
                repairs.append("abstract_step_id_prefixed")
            for key in ("step_name", "purpose"):
                if step.get(key) is None:
                    step[key] = ""
                    repairs.append(f"abstract_{key}_null_filled")
                elif not isinstance(step.get(key), str):
                    step[key] = str(step[key])
                    repairs.append(f"abstract_{key}_coerced_to_string")
            tags = step.get("template_tags")
            if tags is None:
                step["template_tags"] = []
                repairs.append("abstract_template_tags_null_filled")
            elif isinstance(tags, str):
                step["template_tags"] = [
                    tag.strip()
                    for tag in re.split(r"[,;|]+", tags)
                    if tag.strip()
                ]
                repairs.append("abstract_template_tags_split_from_string")
            elif isinstance(tags, list):
                step["template_tags"] = [str(tag) for tag in tags if str(tag).strip()]
            else:
                step["template_tags"] = [str(tags)]
                repairs.append("abstract_template_tags_coerced_to_array")
            normalized_steps.append(step)
        if len(steps) > max_steps:
            repairs.append("abstract_planned_steps_truncated_to_max_steps")
        repaired["planned_steps"] = normalized_steps
    for key in ("case_profile", "planning_rationale"):
        if repaired.get(key) is None:
            repaired[key] = ""
            repairs.append(f"abstract_plan_{key}_null_filled")
        elif key in repaired and not isinstance(repaired[key], str):
            repaired[key] = str(repaired[key])
            repairs.append(f"abstract_plan_{key}_coerced_to_string")
    repairs.extend(
        _fold_extra_fields_into_text(
            repaired,
            allowed={"case_profile", "planned_steps", "planning_rationale"},
            text_key="planning_rationale",
            action_prefix="abstract_plan",
        )
    )
    return repaired, repairs


def _normalize_rf_review_payload(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(payload, dict):
        return payload, []
    repaired = json.loads(json.dumps(payload))
    repairs: list[str] = []
    decision = repaired.get("decision")
    if isinstance(decision, str):
        normalized_decision = decision.strip().lower()
        if normalized_decision in {"continue", "revise", "final_answer"}:
            repaired["decision"] = normalized_decision
            if normalized_decision != decision:
                repairs.append("rf_review_decision_lowercased")
    if repaired.get("rationale") is None:
        repaired["rationale"] = ""
        repairs.append("rf_review_rationale_null_filled")
    elif "rationale" in repaired and not isinstance(repaired["rationale"], str):
        repaired["rationale"] = str(repaired["rationale"])
        repairs.append("rf_review_rationale_coerced_to_string")
    if repaired.get("final_decision") is not None:
        final_decision = str(repaired["final_decision"]).strip().lower()
        repaired["final_decision"] = (
            final_decision if final_decision in {"support", "reject"} else None
        )
        if repaired["final_decision"] is None:
            repairs.append("rf_review_invalid_final_decision_null_filled")
    if repaired.get("final_rationale") is None:
        repaired["final_rationale"] = ""
        repairs.append("rf_review_final_rationale_null_filled")
    elif "final_rationale" in repaired and not isinstance(
        repaired["final_rationale"], str
    ):
        repaired["final_rationale"] = str(repaired["final_rationale"])
        repairs.append("rf_review_final_rationale_coerced_to_string")
    steps = repaired.get("revised_remaining_steps")
    if steps is None:
        repaired["revised_remaining_steps"] = []
        repairs.append("rf_review_revised_steps_null_filled")
    elif isinstance(steps, dict):
        repaired["revised_remaining_steps"] = [steps]
        repairs.append("rf_review_revised_steps_wrapped_as_array")
    if isinstance(repaired.get("revised_remaining_steps"), list):
        normalized_steps = []
        for step in repaired["revised_remaining_steps"]:
            if not isinstance(step, dict):
                repairs.append("rf_review_invalid_revised_step_removed")
                continue
            step = _unwrap_nested_abstract_step_object(step)
            index = len(normalized_steps) + 1
            value = step.get("step_id")
            if not isinstance(value, str):
                step["step_id"] = f"S{value}" if value is not None else f"S{index}"
                repairs.append("rf_review_step_id_coerced_to_string")
            elif value.isdigit():
                step["step_id"] = f"S{value}"
                repairs.append("rf_review_step_id_prefixed")
            for key in ("step_name", "purpose"):
                if step.get(key) is None:
                    step[key] = ""
                    repairs.append(f"rf_review_{key}_null_filled")
                elif not isinstance(step.get(key), str):
                    step[key] = str(step[key])
                    repairs.append(f"rf_review_{key}_coerced_to_string")
            tags = step.get("template_tags")
            if tags is None:
                step["template_tags"] = []
                repairs.append("rf_review_template_tags_null_filled")
            elif isinstance(tags, str):
                step["template_tags"] = [
                    tag.strip()
                    for tag in re.split(r"[,;|]+", tags)
                    if tag.strip()
                ]
                repairs.append("rf_review_template_tags_split_from_string")
            elif isinstance(tags, list):
                step["template_tags"] = [str(tag) for tag in tags if str(tag).strip()]
            else:
                step["template_tags"] = [str(tags)]
                repairs.append("rf_review_template_tags_coerced_to_array")
            normalized_steps.append(step)
        repaired["revised_remaining_steps"] = normalized_steps
    allowed = {
        "decision",
        "rationale",
        "revised_remaining_steps",
        "final_decision",
        "final_rationale",
    }
    repairs.extend(
        _fold_extra_fields_into_text(
            repaired,
            allowed=allowed,
            text_key="rationale",
            action_prefix="rf_review",
        )
    )
    return repaired, repairs


def _normalize_plan_payload(
    payload: dict[str, Any] | None,
    *,
    max_steps: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(payload, dict):
        return payload, []
    repaired = json.loads(json.dumps(payload))
    repairs: list[str] = []
    steps = repaired.get("planned_steps")
    if isinstance(steps, dict):
        steps = [steps]
        repaired["planned_steps"] = steps
        repairs.append("planned_steps_wrapped_as_array")
    if isinstance(steps, list):
        for index, step in enumerate(steps[:max_steps], start=1):
            if not isinstance(step, dict):
                continue
            step = _unwrap_nested_step_object(step)
            steps[index - 1] = step
            value = step.get("step_id")
            if not isinstance(value, str):
                step["step_id"] = f"S{value}" if value is not None else f"S{index}"
                repairs.append("plan_step_id_coerced_to_string")
            elif value.isdigit():
                step["step_id"] = f"S{value}"
                repairs.append("plan_step_id_prefixed")
            for key in ("template_id", "purpose", "expected_artifact"):
                if step.get(key) is None:
                    step[key] = ""
                    repairs.append(f"plan_{key}_null_filled")
                elif not isinstance(step.get(key), str):
                    step[key] = str(step[key])
                    repairs.append(f"plan_{key}_coerced_to_string")
        if len(steps) > max_steps:
            repaired["planned_steps"] = steps[:max_steps]
            repairs.append("planned_steps_truncated_to_max_steps")
    for key in ("case_profile", "planning_rationale"):
        if repaired.get(key) is None:
            repaired[key] = ""
            repairs.append(f"plan_{key}_null_filled")
        elif key in repaired and not isinstance(repaired[key], str):
            repaired[key] = str(repaired[key])
            repairs.append(f"plan_{key}_coerced_to_string")
    repairs.extend(
        _fold_extra_fields_into_text(
            repaired,
            allowed={"case_profile", "planned_steps", "planning_rationale"},
            text_key="planning_rationale",
            action_prefix="plan",
        )
    )
    return repaired, repairs


def _normalize_step_artifact_payload(
    payload: dict[str, Any] | None,
    step: LegalFluxPlanStep,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(payload, dict):
        return payload, []
    repaired = json.loads(json.dumps(payload))
    repairs: list[str] = []
    if repaired.get("step_id") != step.step_id:
        repaired["step_id"] = step.step_id
        repairs.append("step_id_forced_to_planned_value")
    if repaired.get("template_id") != step.template_id:
        repaired["template_id"] = step.template_id
        repairs.append("template_id_forced_to_planned_value")
    for key in ("material_fact_ids", "issue_ids"):
        value = repaired.get(key)
        if value is None:
            repaired[key] = []
            repairs.append(f"{key}_null_filled")
        elif isinstance(value, str):
            repaired[key] = [value] if value.strip() else []
            repairs.append(f"{key}_string_wrapped")
        elif isinstance(value, list):
            id_keys = ("fact_id", "id") if key == "material_fact_ids" else ("issue_id", "id")
            normalized_values: list[str] = []
            for item in value:
                if isinstance(item, str):
                    if item.strip():
                        normalized_values.append(item)
                    continue
                if isinstance(item, dict):
                    identifier = next(
                        (
                            item[id_key]
                            for id_key in id_keys
                            if isinstance(item.get(id_key), str)
                        ),
                        None,
                    )
                    if identifier:
                        normalized_values.append(identifier)
                        repairs.append(f"{key}_object_unwrapped")
                    else:
                        repairs.append(f"{key}_object_without_id_removed")
                    continue
                if item is not None:
                    normalized_values.append(str(item))
                    repairs.append(f"{key}_item_coerced_to_string")
            repaired[key] = normalized_values
    for key in ("instantiated_result", "revision_reason"):
        if repaired.get(key) is None:
            repaired[key] = ""
            repairs.append(f"{key}_null_filled")
        elif key in repaired and not isinstance(repaired[key], str):
            repaired[key] = str(repaired[key])
            repairs.append(f"{key}_coerced_to_string")
    if repaired.get("needs_revision") is None:
        repaired["needs_revision"] = False
        repairs.append("needs_revision_null_filled")
    confidence = repaired.get("confidence")
    if isinstance(confidence, str):
        normalized = confidence.strip().lower()
        if normalized in {"low", "medium", "high"} and normalized != confidence:
            repaired["confidence"] = normalized
            repairs.append("confidence_lowercased")
    elif confidence is None:
        repaired["confidence"] = "medium"
        repairs.append("confidence_null_filled")
    allowed = {
        "step_id",
        "template_id",
        "instantiated_result",
        "material_fact_ids",
        "issue_ids",
        "confidence",
        "needs_revision",
        "revision_reason",
    }
    extra = {
        key: repaired.pop(key)
        for key in list(repaired)
        if key not in allowed
    }
    non_empty_extra = {
        key: value
        for key, value in extra.items()
        if value not in (None, "", [], {})
    }
    if non_empty_extra:
        extra_text = json.dumps(non_empty_extra, ensure_ascii=False, sort_keys=True)
        if repaired.get("instantiated_result"):
            repaired["instantiated_result"] += (
                f"\nAdditional structured notes: {extra_text}"
            )
        else:
            repaired["instantiated_result"] = extra_text
        repairs.append("step_extra_fields_folded_into_result")
    elif extra:
        repairs.append("step_extra_fields_removed")
    return repaired, repairs


def _normalize_review_payload(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(payload, dict):
        return payload, []
    repaired = json.loads(json.dumps(payload))
    repairs: list[str] = []
    decision = repaired.get("decision")
    if isinstance(decision, str):
        normalized_decision = decision.strip().lower()
        if normalized_decision in {"continue", "revise", "stop"}:
            repaired["decision"] = normalized_decision
            if normalized_decision != decision:
                repairs.append("review_decision_lowercased")
    if repaired.get("rationale") is None:
        repaired["rationale"] = ""
        repairs.append("review_rationale_null_filled")
    elif "rationale" in repaired and not isinstance(repaired["rationale"], str):
        repaired["rationale"] = str(repaired["rationale"])
        repairs.append("review_rationale_coerced_to_string")
    steps = repaired.get("revised_remaining_steps")
    if steps is None:
        repaired["revised_remaining_steps"] = []
        repairs.append("review_revised_steps_null_filled")
    elif isinstance(steps, dict):
        repaired["revised_remaining_steps"] = [steps]
        repairs.append("review_revised_steps_wrapped_as_array")
    if isinstance(repaired.get("revised_remaining_steps"), list):
        normalized_steps: list[dict[str, Any]] = []
        for step in repaired["revised_remaining_steps"]:
            if isinstance(step, list):
                dict_steps = [item for item in step if isinstance(item, dict)]
                if len(dict_steps) == 1:
                    step = dict_steps[0]
                    repairs.append("review_revised_step_list_unwrapped")
                else:
                    repairs.append("review_invalid_revised_step_removed")
                    continue
            if not isinstance(step, dict):
                repairs.append("review_invalid_revised_step_removed")
                continue
            step = _unwrap_nested_step_object(step)
            index = len(normalized_steps) + 1
            value = step.get("step_id")
            if not isinstance(value, str):
                step["step_id"] = f"S{value}" if value is not None else f"S{index}"
                repairs.append("review_step_id_coerced_to_string")
            elif value.isdigit():
                step["step_id"] = f"S{value}"
                repairs.append("review_step_id_prefixed")
            for key in ("template_id", "purpose", "expected_artifact"):
                if step.get(key) is None:
                    step[key] = ""
                    repairs.append(f"review_{key}_null_filled")
                elif not isinstance(step.get(key), str):
                    step[key] = str(step[key])
                    repairs.append(f"review_{key}_coerced_to_string")
            normalized_steps.append(step)
        repaired["revised_remaining_steps"] = normalized_steps
    repairs.extend(
        _fold_extra_fields_into_text(
            repaired,
            allowed={"decision", "rationale", "revised_remaining_steps"},
            text_key="rationale",
            action_prefix="review",
        )
    )
    return repaired, repairs


def _unwrap_nested_step_object(step: dict[str, Any]) -> dict[str, Any]:
    allowed = {"step_id", "template_id", "purpose", "expected_artifact"}
    nested_keys = [
        key
        for key, value in step.items()
        if re.fullmatch(r"step[\s_-]*\d+", str(key), flags=re.I)
        and isinstance(value, dict)
    ]
    if nested_keys and not allowed.intersection(step):
        nested = dict(step[nested_keys[0]])
        return {key: value for key, value in nested.items() if key in allowed}
    merged = dict(step)
    for key in nested_keys:
        nested = step[key]
        for allowed_key in allowed:
            if allowed_key not in merged and allowed_key in nested:
                merged[allowed_key] = nested[allowed_key]
        merged.pop(key, None)
    return {key: value for key, value in merged.items() if key in allowed}


def _unwrap_nested_abstract_step_object(step: dict[str, Any]) -> dict[str, Any]:
    allowed = {"step_id", "step_name", "template_tags", "purpose"}
    nested_keys = [
        key
        for key, value in step.items()
        if re.fullmatch(r"step[\s_-]*\d+", str(key), flags=re.I)
        and isinstance(value, dict)
    ]
    if nested_keys and not allowed.intersection(step):
        nested = dict(step[nested_keys[0]])
        return {key: value for key, value in nested.items() if key in allowed}
    merged = dict(step)
    for key in nested_keys:
        nested = step[key]
        for allowed_key in allowed:
            if allowed_key not in merged and allowed_key in nested:
                merged[allowed_key] = nested[allowed_key]
        merged.pop(key, None)
    return {key: value for key, value in merged.items() if key in allowed}


def _renumber_remaining_steps(
    steps: list[LegalFluxPlanStep],
    *,
    start_index: int,
) -> list[LegalFluxPlanStep]:
    return [
        step.model_copy(update={"step_id": f"S{index}"})
        for index, step in enumerate(steps, start=start_index)
    ]


def _renumber_abstract_remaining_steps(
    steps: list[LegalFluxAbstractStep],
    *,
    start_index: int,
) -> list[LegalFluxAbstractStep]:
    return [
        step.model_copy(update={"step_id": f"S{index}"})
        for index, step in enumerate(steps, start=start_index)
    ]


def _common_generation_settings(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config["model"]["name"],
        "temperature": config["model"]["temperature"],
        "seed": config["model"]["seed"],
        "context_length": config["model"]["context_length"],
    }


def _build_rf_similarity_backend(config: dict[str, Any]) -> SimilarityBackend:
    backend = config["legal_flux"].get("rf_retrieval_backend", "ollama_embedding")
    if backend == "tfidf":
        return TfidfSimilarityBackend()
    if backend != "ollama_embedding":
        raise ValueError(f"Unknown LegalFlux RF retrieval backend: {backend}")
    cache_file = config["legal_flux"].get(
        "rf_embedding_cache_file",
        "data/processed/legal_flux/rf_template_embeddings_bge_m3.json",
    )
    cache_path = Path(cache_file)
    if not cache_path.is_absolute():
        cache_path = resolve_path(config, "prompts_dir").parent / cache_path
    return OllamaEmbeddingBackend(
        base_url=config["legal_flux"].get(
            "rf_embedding_base_url", config["model"]["base_url"]
        ),
        model=config["legal_flux"].get("rf_embedding_model", "bge-m3:latest"),
        cache_path=cache_path,
        timeout_seconds=config["legal_flux"].get(
            "rf_embedding_timeout_seconds",
            config["model"].get("timeout_seconds", 600),
        ),
    )
