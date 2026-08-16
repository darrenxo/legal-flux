from __future__ import annotations

import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .clients import (
    GenerationClient,
    GenerationResponseError,
    build_generation_client,
)
from .config import resolve_path, resolve_project_path
from .embeddings import (
    OllamaEmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
    SimilarityBackend,
    TfidfSimilarityBackend,
)
from .io_utils import canonical_json, sha256_text
from .ledger import JsonlLedger, make_run_hash
from .legal_flux import (
    build_legal_flux_jobs,
    legal_flux_workflow_hash,
    load_template_pool,
    retrieve_template_for_abstract_step,
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
    NormalizedCase,
)
from .prompting import render_prompt
from .runner import (
    _execute_condition,
    _fold_extra_fields_into_text,
    _load_schema,
    _preview_prompt,
    _response_trace,
    load_cases,
)


def run_legal_flux_generation(
    config: dict[str, Any],
    *,
    phase: str,
    dry_run: bool = False,
    sample_count: int | None = None,
    case_limit: int | None = None,
    conditions: list[str] | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    run_tag: str | None = None,
    case_ids_file: str | None = None,
    fail_on_errors: bool = False,
) -> dict[str, Any]:
    normalized_phase = phase.replace("-", "_")
    normalized_run_tag = _validated_run_tag(run_tag)
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    cases = load_cases(config)
    requested_case_ids: set[str] | None = None
    if case_ids_file:
        requested_case_ids = _load_case_ids(
            resolve_project_path(config, case_ids_file)
        )
        available = {case.case_id for case in cases}
        missing = sorted(requested_case_ids - available)
        if missing:
            raise ValueError(
                f"Case-ID file contains {len(missing)} unknown IDs; first: "
                f"{missing[:5]}"
            )
        cases = [case for case in cases if case.case_id in requested_case_ids]
    jobs = build_legal_flux_jobs(
        cases,
        config,
        phase=normalized_phase,
        sample_count=sample_count,
        case_limit=case_limit,
    )
    if requested_case_ids is not None:
        selected_ids = {job["case"].case_id for job in jobs}
        wrong_phase = sorted(requested_case_ids - selected_ids)
        if wrong_phase:
            raise ValueError(
                f"Case-ID file contains {len(wrong_phase)} cases outside "
                f"{normalized_phase}; first: {wrong_phase[:5]}"
            )
    if conditions is not None:
        requested = {condition.strip() for condition in conditions if condition.strip()}
        known = {job["condition"] for job in jobs}
        unsupported = sorted(requested - known)
        if unsupported:
            raise ValueError(
                f"Requested conditions are not configured for {normalized_phase}: "
                f"{unsupported}"
            )
        jobs = [job for job in jobs if job["condition"] in requested]
    jobs = _select_generation_shard(
        jobs,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    run_dir = _generation_run_dir(
        config,
        normalized_phase,
        num_shards=num_shards,
        shard_index=shard_index,
        run_tag=normalized_run_tag,
    )
    if dry_run:
        return {
            "phase": normalized_phase,
            "jobs": len(jobs),
            "conditions": sorted({job["condition"] for job in jobs}),
            "cases": len({job["case"].case_id for job in jobs}),
            "samples_per_case": max((job["sample_index"] for job in jobs), default=0) + 1,
            "run_dir": str(run_dir),
            "num_shards": num_shards,
            "shard_index": shard_index,
            "run_tag": normalized_run_tag,
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
    client = build_generation_client(config)
    required_models = {config["model"]["name"]}
    if any(job["condition"] == "flux_rf_style" for job in jobs):
        required_models.update(
            _role_model(config, role)
            for role in ("planner", "executor", "reviewer")
        )
    model_infos = {
        model_name: client.model_info(model_name)
        for model_name in sorted(required_models)
    }
    missing_models = [
        model_name for model_name, model_info in model_infos.items() if not model_info
    ]
    if missing_models:
        client.close()
        raise RuntimeError(
            f"Models {missing_models!r} are not exposed by "
            f"{config['model'].get('provider', 'ollama')} at "
            f"{config['model']['base_url']}."
        )
    model_digests = {
        model_name: str(model_info.get("digest", "unknown"))
        for model_name, model_info in model_infos.items()
        if model_info is not None
    }
    digest = (
        next(iter(model_digests.values()))
        if len(model_digests) == 1
        else sha256_text(canonical_json(model_digests))
    )
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
                sample_index=job["sample_index"],
                temperature=job["temperature"],
            ),
            "case_id": job["case"].case_id,
            "condition": job["condition"],
            "phase": normalized_phase,
            "sample_index": job["sample_index"],
        }
        for job in jobs
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_plan.json").write_text(
        json.dumps(
            {
                "phase": normalized_phase,
                "model_digest": digest,
                "model_digests": model_digests,
                "inference_runtime": config["model"].get("inference_runtime"),
                "inference_runtime_version": config["model"].get(
                    "inference_runtime_version"
                ),
                "chat_template_kwargs": (
                    config["model"].get("extra_body") or {}
                ).get("chat_template_kwargs", {}),
                "workflow_hash": workflow_hash,
                "template_pool_hash": template_hash,
                "num_shards": num_shards,
                "shard_index": shard_index,
                "run_tag": normalized_run_tag,
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
    concurrency = max(1, int(config["model"].get("concurrency", 1)))
    try:
        if concurrency == 1:
            records = (
                _run_flux_job(
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
                for job in jobs
            )
            for record in records:
                completed, skipped, errors = _update_run_counts(
                    record,
                    completed=completed,
                    skipped=skipped,
                    errors=errors,
                )
                _print_generation_progress(
                    completed=completed,
                    skipped=skipped,
                    errors=errors,
                    total=len(jobs),
                )
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(
                        _run_flux_job,
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
                    for job in jobs
                ]
                for future in as_completed(futures):
                    completed, skipped, errors = _update_run_counts(
                        future.result(),
                        completed=completed,
                        skipped=skipped,
                        errors=errors,
                    )
                    _print_generation_progress(
                        completed=completed,
                        skipped=skipped,
                        errors=errors,
                        total=len(jobs),
                    )
    finally:
        client.close()
        if hasattr(similarity_backend, "close"):
            similarity_backend.close()
    result = {
        "phase": normalized_phase,
        "jobs": len(jobs),
        "completed": completed,
        "skipped": skipped,
        "errors": errors,
        "run_dir": str(run_dir),
        "model_digest": digest,
        "model_digests": model_digests,
        "workflow_hash": workflow_hash,
        "template_pool_hash": template_hash,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "run_tag": normalized_run_tag,
        "concurrency": concurrency,
    }
    if fail_on_errors and errors:
        raise RuntimeError(
            f"LegalFlux generation recorded {errors} error(s) in {run_dir}. "
            "Successful records were preserved; rerun the same shard to retry "
            "only failed work."
        )
    return result


def flux_run_hash(
    case: NormalizedCase,
    *,
    condition: str,
    phase: str,
    model_digest: str,
    workflow_hash: str,
    template_hash: str,
    seed: int,
    sample_index: int = 0,
    temperature: float | None = None,
) -> str:
    return make_run_hash(
        dataset=case.dataset,
        case_id=case.case_id,
        variant_id=case.variant_id,
        condition=condition,
        phase=phase,
        sample_index=sample_index,
        temperature=temperature,
        model_digest=model_digest,
        workflow_hash=workflow_hash,
        template_pool_hash=template_hash,
        seed=seed,
    )


def _run_flux_job(
    client: GenerationClient,
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
        sample_index=job["sample_index"],
        temperature=job["temperature"],
    )
    if ledger.contains(run_hash):
        return None
    base = {
        "run_hash": run_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": case.dataset,
        "case_id": case.case_id,
        "variant_id": case.variant_id,
        "condition": condition,
        "phase": phase,
        "prompt_hash": _condition_prompt_hash(config, case, condition, templates),
        "model_name": config["model"]["name"],
        "inference_runtime": config["model"].get("inference_runtime"),
        "inference_runtime_version": config["model"].get(
            "inference_runtime_version"
        ),
        "chat_template_kwargs": (config["model"].get("extra_body") or {}).get(
            "chat_template_kwargs", {}
        ),
        "role_models": {
            role: _role_model(config, role)
            for role in ("planner", "executor", "reviewer")
        },
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
        elif condition == "flux_rf_style":
            analysis, trace = _execute_rf_style_case(
                client,
                config,
                case,
                templates=templates,
                similarity_backend=similarity_backend,
            )
        else:
            raise ValueError(f"Unknown LegalFlux condition: {condition}")
        record = {
            **base,
            "status": "ok",
            "raw_response": trace["raw_response"],
            "parsed_json": analysis.model_dump(mode="json", exclude_defaults=True),
            "trajectory_plan": trace.get("trajectory_plan"),
            "executed_steps": trace.get("executed_steps"),
            "trajectory_reviews": trace.get("trajectory_reviews"),
            "retrieved_template_ids": trace.get("retrieved_template_ids"),
            "selected_templates": trace.get("selected_templates"),
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
            "raw_response": (
                exc.raw_text if isinstance(exc, GenerationResponseError) else None
            ),
            "parsed_json": None,
            "trajectory_plan": None,
            "executed_steps": None,
            "trajectory_reviews": None,
            "retrieved_template_ids": None,
            "selected_templates": None,
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


def _generation_run_dir(
    config: dict[str, Any],
    phase: str,
    *,
    num_shards: int,
    shard_index: int,
    run_tag: str | None,
) -> Path:
    base = resolve_path(config, "runs_dir") / phase
    if run_tag:
        base = base / "experiments" / run_tag
    if num_shards == 1:
        return base
    return base / "shards" / f"shard-{shard_index:05d}-of-{num_shards:05d}"


def _validated_run_tag(run_tag: str | None) -> str | None:
    if run_tag is None:
        return None
    value = run_tag.strip()
    if not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ValueError(
            "run_tag must be 1-80 characters using letters, digits, dot, "
            "underscore, or hyphen."
        )
    return value


def _load_case_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Case-ID file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("case_ids") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("Case-ID file must be a JSON list or contain `case_ids`.")
    case_ids = {str(value).strip() for value in values if str(value).strip()}
    if not case_ids:
        raise ValueError("Case-ID file contains no case IDs.")
    return case_ids


def _select_generation_shard(
    jobs: list[dict[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> list[dict[str, Any]]:
    if num_shards == 1:
        return jobs
    case_keys = list(
        dict.fromkeys(
            (
                job["case"].dataset,
                job["case"].case_id,
                job["case"].variant_id,
            )
            for job in jobs
        )
    )
    selected_keys = {
        key
        for index, key in enumerate(case_keys)
        if index % num_shards == shard_index
    }
    return [
        job
        for job in jobs
        if (
            job["case"].dataset,
            job["case"].case_id,
            job["case"].variant_id,
        )
        in selected_keys
    ]


def _update_run_counts(
    record: dict[str, Any] | None,
    *,
    completed: int,
    skipped: int,
    errors: int,
) -> tuple[int, int, int]:
    if record is None:
        return completed, skipped + 1, errors
    return (
        completed + 1,
        skipped,
        errors + (1 if record.get("status") != "ok" else 0),
    )


def _print_generation_progress(
    *,
    completed: int,
    skipped: int,
    errors: int,
    total: int,
) -> None:
    processed = completed + skipped
    if processed % 25 != 0 and processed != total:
        return
    print(
        "LegalFlux progress: "
        f"{processed}/{total} jobs; completed={completed}, "
        f"skipped={skipped}, errors={errors}",
        flush=True,
    )


def _execute_rf_style_case(
    client: Any,
    config: dict[str, Any],
    case: NormalizedCase,
    *,
    templates: list[LegalFluxTemplate],
    similarity_backend: SimilarityBackend | None = None,
) -> tuple[FinalAnalysis, dict[str, Any]]:
    planner_common = _common_generation_settings(config, role="planner")
    executor_common = _common_generation_settings(config, role="executor")
    reviewer_common = _common_generation_settings(config, role="reviewer")
    schema_dir = resolve_path(config, "schemas_dir")
    max_steps = int(config["legal_flux"].get("max_steps", 4))

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
        max_steps=max_steps,
    )
    response = client.generate(
        prompt=plan_prompt,
        schema=_load_schema(schema_dir / "legal_flux_abstract_plan.json"),
        max_tokens=config["model"]["flux_plan_max_tokens"],
        **planner_common,
    )
    normalized_plan, plan_repairs = _normalize_abstract_plan_payload(
        response.parsed, max_steps=max_steps
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
    used_template_ids: set[str] = set()
    analysis: FinalAnalysis | None = None
    while remaining and len(artifacts) < max_steps and analysis is None:
        abstract_step = remaining.pop(0)
        retrieval = retrieve_template_for_abstract_step(
            abstract_step,
            templates,
            similarity_backend=similarity_backend,
            exclude_template_ids=used_template_ids,
        )
        template = retrieval["template"]
        used_template_ids.add(template.template_id)
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
            common=executor_common,
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
            artifacts=artifacts,
            remaining=remaining,
            selected_templates=selected_templates,
            common=reviewer_common,
            max_steps=max_steps,
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
            if review.final_decision in {"support", "reject"}:
                analysis = _analysis_from_rf_review(review)
            else:
                repairs.append("rf_final_answer_missing_label_forced_retry")
            break
        if review.decision == "revise":
            remaining = list(
                review.revised_remaining_steps[: max_steps - len(artifacts)]
            )
            repairs.append("rf_remaining_trajectory_revised")

    if analysis is None and artifacts:
        review, review_trace = _review_rf_trajectory(
            client,
            config,
            case,
            artifacts=artifacts,
            remaining=[],
            selected_templates=selected_templates,
            common=reviewer_common,
            max_steps=max_steps,
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

    return analysis, {
        "raw_response": "\n---CALL---\n".join(raw_parts),
        "prompt_hashes": prompt_hashes,
        "trajectory_plan": abstract_plan.model_dump(mode="json"),
        "executed_steps": [artifact.model_dump(mode="json") for artifact in artifacts],
        "trajectory_reviews": [review.model_dump(mode="json") for review in reviews],
        "retrieved_template_ids": [item["template_id"] for item in selected_templates],
        "selected_templates": selected_templates,
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "schema_errors": schema_errors,
        "repair_actions": repairs,
        "calls": calls,
    }


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
        schema=_load_schema(resolve_path(config, "schemas_dir") / "legal_flux_step_artifact.json"),
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
    artifacts: list[LegalFluxStepArtifact],
    remaining: list[LegalFluxAbstractStep],
    selected_templates: list[dict[str, Any]],
    common: dict[str, Any],
    max_steps: int,
    force_final_answer: bool = False,
) -> tuple[LegalFluxRfReview, dict[str, Any]]:
    remaining_step_limit = max_steps - len(artifacts)
    prompt, prompt_hash = render_prompt(
        config,
        "legal_flux/rf_review",
        case,
        executed_trajectory=_active_executed_trajectory(
            artifacts,
            selected_templates,
        ),
        remaining_steps=[step.model_dump(mode="json") for step in remaining],
        max_steps=max_steps,
        review_output_requirement=(
            "No remaining abstract steps are available. Return only "
            "final_rationale, followed by final_decision. final_decision must "
            'be exactly "support" or "reject". Interpret the label relative '
            "to the supplied plaintiff's claim. support means that the court "
            "grants or allows the claim. reject means that the court dismisses, "
            "refuses, or denies it. Make sure that final_decision agrees with "
            "final_rationale."
            if force_final_answer
            else (
                "Use the executed trajectory and supplied facts to decide whether "
                "to continue with the existing plan, revise the remaining abstract "
                "steps, or return the final decision.\n\n"
                "First provide a concise review_analysis, then set decision to "
                'exactly "continue", "revise", or "final_answer".\n\n'
                "continue means that the remaining abstract steps are still "
                "appropriate and the trajectory should simply proceed to the next "
                "planned step. If decision is continue, output only review_analysis "
                "and decision.\n\n"
                "revise means that the remaining abstract steps need revision. If "
                "decision is revise, also output revised_remaining_steps with the "
                "revised steps in order. Each revised step must contain step_name, "
                "step_description, and template_tags for template retrieval. Return "
                "at least 1 and at most "
                f"{remaining_step_limit} revised_remaining_steps.\n\n"
                "final_answer means that the executed trajectory so far is already "
                "sufficient to reach a decision, or that the configured limit of "
                f"{max_steps} executed steps has been reached. If decision is "
                "final_answer, also output final_rationale and final_decision. "
                'final_decision must be exactly "support" or "reject". Interpret '
                "the label relative to the supplied plaintiff's claim. support "
                "means that the court grants or allows the claim. reject means "
                "that the court dismisses, refuses, or denies it. Make sure that "
                "final_decision agrees with final_rationale."
            )
        ),
    )
    schema_name = (
        "legal_flux_rf_final_review.json"
        if force_final_answer
        else "legal_flux_rf_review.json"
    )
    schema = _load_schema(resolve_path(config, "schemas_dir") / schema_name)
    if not force_final_answer:
        revise_branch = next(
            branch
            for branch in schema["oneOf"]
            if branch["properties"]["decision"].get("const") == "revise"
        )
        revised_steps_schema = revise_branch["properties"][
            "revised_remaining_steps"
        ]
        revised_steps_schema["minItems"] = 1
        revised_steps_schema["maxItems"] = remaining_step_limit
    response = client.generate(
        prompt=prompt,
        schema=schema,
        max_tokens=config["model"]["flux_review_max_tokens"],
        **common,
    )
    normalized_review, repairs = _normalize_rf_review_payload(response.parsed)
    review = LegalFluxRfReview.model_validate(normalized_review)
    if review.decision == "revise":
        review = review.model_copy(
            update={
                "revised_remaining_steps": _renumber_abstract_remaining_steps(
                    review.revised_remaining_steps,
                    start_index=len(artifacts) + 1,
                )
            }
        )
    trace = _response_trace(response)
    trace["prompt_hash"] = prompt_hash
    trace["repair_actions"].extend(repairs)
    return review, trace


def _active_executed_trajectory(
    artifacts: list[LegalFluxStepArtifact],
    selected_templates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if len(artifacts) != len(selected_templates):
        raise ValueError(
            "Executed artifact and selected-template trace counts do not match."
        )
    executed: list[dict[str, str]] = []
    for artifact, selected in zip(artifacts, selected_templates, strict=True):
        step_id = str(selected.get("step_id") or "")
        template_id = str(selected.get("template_id") or "")
        step_name = str(selected.get("step_name") or "")
        template_name = str(selected.get("template_name") or "")
        if step_id != artifact.step_id:
            raise ValueError(
                "Executed artifact and selected template have different step IDs: "
                f"{artifact.step_id!r} != {step_id!r}."
            )
        if template_id != artifact.template_id:
            raise ValueError(
                "Executed artifact and selected template have different template IDs: "
                f"{artifact.template_id!r} != {template_id!r}."
            )
        if not step_name or not template_name:
            raise ValueError(
                "Selected-template trace must include step_name and template_name."
            )
        executed.append(
            {
                "step_id": artifact.step_id,
                "step_name": step_name,
                "template_id": artifact.template_id,
                "template_name": template_name,
                "instantiated_result": artifact.instantiated_result,
            }
        )
    return executed


def _abstract_step_to_plan_step(
    step: LegalFluxAbstractStep,
    template: LegalFluxTemplate,
) -> LegalFluxPlanStep:
    return LegalFluxPlanStep(
        step_id=step.step_id,
        template_id=template.template_id,
        purpose=(
            f"{step.step_name}\n"
            f"Description: {step.step_description}\n"
            f"Tags: {', '.join(step.template_tags)}"
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
        final_decision=review.final_decision,
        final_rationale=review.final_rationale or review.review_analysis,
    )


def _condition_prompt_hash(
    config: dict[str, Any],
    case: NormalizedCase,
    condition: str,
    templates: list[LegalFluxTemplate],
) -> str:
    if condition in {"direct", "structured"}:
        _, prompt_hash = _preview_prompt(config, case, condition)
        return prompt_hash
    if condition != "flux_rf_style":
        raise ValueError(f"Unsupported LegalFlux condition: {condition}")
    payload = {
        "condition": condition,
        "native_case_input": _native_case_input_payload(
            case,
            include_authority=bool(config["legal_flux"].get("include_authority_input", False)),
        ),
        "template_pool_hash": template_pool_hash(templates),
        "max_steps": config["legal_flux"].get("max_steps", 4),
        "include_authority_input": config["legal_flux"].get("include_authority_input", False),
        "rf_retrieval_backend": config["legal_flux"].get(
            "rf_retrieval_backend", "ollama_embedding"
        ),
        "rf_embedding_model": config["legal_flux"].get(
            "rf_embedding_model", "bge-m3:latest"
        ),
    }
    return sha256_text(canonical_json(payload))


def _native_case_input_payload(
    case: NormalizedCase,
    *,
    include_authority: bool = False,
) -> dict[str, Any]:
    payload = {
        "claim": case.claim,
        "parties": case.parties,
        "facts": case.facts,
    }
    if include_authority:
        payload["authorities"] = case.authorities
        payload["relevant_cases"] = case.metadata.get("relevant_cases")
    return payload


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
            for key in ("step_name", "step_description"):
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
                    tag.strip() for tag in re.split(r"[,;|]+", tags) if tag.strip()
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
    if "planning_analysis" not in repaired:
        legacy_parts = [
            str(repaired.pop(key)).strip()
            for key in ("case_profile", "planning_rationale")
            if repaired.get(key) not in (None, "")
        ]
        if legacy_parts:
            repaired["planning_analysis"] = " ".join(legacy_parts)
            repairs.append("abstract_legacy_analysis_combined")
    if repaired.get("planning_analysis") is None:
        repaired["planning_analysis"] = ""
        repairs.append("abstract_plan_planning_analysis_null_filled")
    elif "planning_analysis" in repaired and not isinstance(
        repaired["planning_analysis"], str
    ):
        repaired["planning_analysis"] = str(repaired["planning_analysis"])
        repairs.append("abstract_plan_planning_analysis_coerced_to_string")
    extra_fields = [
        key
        for key in repaired
        if key not in {"planning_analysis", "planned_steps"}
    ]
    if extra_fields:
        for key in extra_fields:
            repaired.pop(key)
        repairs.append("abstract_plan_extra_fields_removed")
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
    if "review_analysis" not in repaired and "rationale" in repaired:
        repaired["review_analysis"] = repaired.pop("rationale")
        repairs.append("rf_review_legacy_rationale_renamed")
    if repaired.get("review_analysis") is None:
        repaired["review_analysis"] = ""
        repairs.append("rf_review_analysis_null_filled")
    elif "review_analysis" in repaired and not isinstance(
        repaired["review_analysis"], str
    ):
        repaired["review_analysis"] = str(repaired["review_analysis"])
        repairs.append("rf_review_analysis_coerced_to_string")
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
            if value is None:
                step["step_id"] = f"S{index}"
            elif not isinstance(value, str):
                step["step_id"] = f"S{value}"
                repairs.append("rf_review_step_id_coerced_to_string")
            elif value.isdigit():
                step["step_id"] = f"S{value}"
                repairs.append("rf_review_step_id_prefixed")
            for key in ("step_name", "step_description"):
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
                    tag.strip() for tag in re.split(r"[,;|]+", tags) if tag.strip()
                ]
                repairs.append("rf_review_template_tags_split_from_string")
            elif isinstance(tags, list):
                step["template_tags"] = [str(tag) for tag in tags if str(tag).strip()]
            else:
                step["template_tags"] = [str(tags)]
                repairs.append("rf_review_template_tags_coerced_to_array")
            normalized_steps.append(step)
        repaired["revised_remaining_steps"] = normalized_steps
    repairs.extend(
        _fold_extra_fields_into_text(
            repaired,
            allowed={
                "decision",
                "review_analysis",
                "revised_remaining_steps",
                "final_rationale",
                "final_decision",
            },
            text_key="review_analysis",
            action_prefix="rf_review",
        )
    )
    return repaired, repairs


def _normalize_step_artifact_payload(
    payload: dict[str, Any] | None,
    step: LegalFluxPlanStep,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(payload, dict):
        return payload, []
    repairs: list[str] = []
    value = payload.get("instantiated_result")
    if value is None:
        result = ""
        repairs.append("instantiated_result_null_filled")
    elif isinstance(value, str):
        result = value
    else:
        result = str(value)
        repairs.append("instantiated_result_coerced_to_string")

    result, truncated = _truncate_step_result(result)
    if truncated:
        repairs.append("instantiated_result_truncated")
    if set(payload) - {"instantiated_result"}:
        repairs.append("obsolete_step_output_fields_removed")

    return {
        "step_id": step.step_id,
        "template_id": step.template_id,
        "instantiated_result": result,
    }, repairs


def _truncate_step_result(value: str) -> tuple[str, bool]:
    result = value.strip()
    truncated = False
    words = result.split()
    if len(words) > 180:
        result = " ".join(words[:180])
        truncated = True
    if len(result) > 1800:
        result = result[:1800].rstrip()
        truncated = True
    return result, truncated


def _unwrap_nested_abstract_step_object(step: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "step_id",
        "step_name",
        "step_description",
        "template_tags",
        "purpose",
    }
    nested_keys = [
        key
        for key, value in step.items()
        if re.fullmatch(r"step[\s_-]*\d+", str(key), flags=re.I)
        and isinstance(value, dict)
    ]
    if nested_keys and not allowed.intersection(step):
        nested = dict(step[nested_keys[0]])
        return _migrate_legacy_step_purpose(
            {key: value for key, value in nested.items() if key in allowed}
        )
    merged = dict(step)
    for key in nested_keys:
        nested = step[key]
        for allowed_key in allowed:
            if allowed_key not in merged and allowed_key in nested:
                merged[allowed_key] = nested[allowed_key]
        merged.pop(key, None)
    return _migrate_legacy_step_purpose(
        {key: value for key, value in merged.items() if key in allowed}
    )


def _migrate_legacy_step_purpose(step: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(step)
    if "step_description" not in migrated and "purpose" in migrated:
        migrated["step_description"] = migrated["purpose"]
    migrated.pop("purpose", None)
    return migrated


def _renumber_abstract_remaining_steps(
    steps: list[LegalFluxAbstractStep],
    *,
    start_index: int,
) -> list[LegalFluxAbstractStep]:
    return [
        step.model_copy(update={"step_id": f"S{index}"})
        for index, step in enumerate(steps, start=start_index)
    ]


def _common_generation_settings(
    config: dict[str, Any],
    *,
    role: str | None = None,
) -> dict[str, Any]:
    return {
        "model": _role_model(config, role) if role else config["model"]["name"],
        "temperature": config["model"]["temperature"],
        "seed": config["model"]["seed"],
        "context_length": config["model"]["context_length"],
    }


def _role_model(config: dict[str, Any], role: str) -> str:
    configured = config["legal_flux"].get(f"{role}_model")
    return str(configured or config["model"]["name"])


def _build_rf_similarity_backend(config: dict[str, Any]) -> SimilarityBackend:
    backend = config["legal_flux"].get("rf_retrieval_backend", "ollama_embedding")
    if backend == "tfidf":
        return TfidfSimilarityBackend()
    if backend == "sentence_transformer":
        return SentenceTransformerEmbeddingBackend(
            model=config["legal_flux"].get(
                "rf_sentence_transformer_model", "BAAI/bge-m3"
            ),
            device=config.get("xsim", {}).get("device") or None,
            max_length=int(config.get("xsim", {}).get("max_length", 8192)),
            batch_size=int(config.get("xsim", {}).get("dense_batch_size", 8)),
        )
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
