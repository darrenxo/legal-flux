from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .clients import ModelResponse, OllamaClient
from .config import resolve_path
from .embeddings import SimilarityBackend
from .io_utils import atomic_write_json, read_jsonl
from .ledger import JsonlLedger, make_run_hash
from .legal_flux import (
    load_template_pool,
    retrieve_template_for_abstract_step,
    template_pool_hash,
)
from .legal_flux_runner import (
    _abstract_step_to_plan_step,
    _build_rf_similarity_backend,
    _instantiate_step,
    _load_schema,
    _normalize_abstract_plan_payload,
    _review_rf_trajectory,
    _template_tag_examples,
)
from .legal_flux_xsim import load_xsim_neighbors
from .models import (
    LegalFluxAbstractPlan,
    LegalFluxStepArtifact,
    LegalFluxTemplate,
    NormalizedCase,
)
from .prompting import render_prompt
from .runner import load_cases


class GenerationClient(Protocol):
    def generate(self, **kwargs: Any) -> ModelResponse:
        ...


class InvalidFinalDecisionError(ValueError):
    pass


def build_dpo_data(
    config: dict[str, Any],
    *,
    stage: str = "all",
    case_limit: int | None = None,
    force: bool = False,
    client: GenerationClient | None = None,
    similarity_backend: SimilarityBackend | None = None,
) -> dict[str, Any]:
    if stage not in {"sample", "evaluate", "all"}:
        raise ValueError(f"Unsupported DPO construction stage: {stage}")
    settings = _dpo_settings(config)
    if settings["samples_per_anchor"] != 4:
        raise ValueError(
            "LegalFlux DPO construction is configured for exactly four "
            "candidate trajectories per anchor."
        )
    cases = _planner_train_cases(config)
    anchors = cases[:case_limit] if case_limit is not None else cases
    case_by_id = {case.case_id: case for case in cases}
    xsim = load_xsim_neighbors(config)
    missing_xsim = [case.case_id for case in anchors if case.case_id not in xsim]
    if missing_xsim:
        raise RuntimeError(
            "X_sim neighbors are missing for requested planner_train anchors. "
            f"First missing case: {missing_xsim[0]}"
        )

    templates = load_template_pool(config)
    template_hash = template_pool_hash(templates)
    output_dir = _training_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / settings["candidates_file"]
    evaluations_path = output_dir / settings["evaluations_file"]
    manifest_path = output_dir / settings["manifest_file"]
    manifest_key = {
        "template_pool_hash": template_hash,
        "planner_model": settings["planner_model"],
        "executor_model": settings["executor_model"],
        "samples_per_anchor": settings["samples_per_anchor"],
        "planner_temperature": settings["planner_temperature"],
        "max_steps": int(config["legal_flux"].get("max_steps", 4)),
    }
    _guard_existing_manifest(manifest_path, manifest_key, force=force)
    if force:
        candidates_path.unlink(missing_ok=True)
        evaluations_path.unlink(missing_ok=True)

    owned_client = client is None
    if client is None:
        required_models = set()
        if stage in {"sample", "all"}:
            required_models.add(settings["planner_model"])
        if stage in {"evaluate", "all"}:
            required_models.add(settings["executor_model"])
        client = _ollama_client(config, required_models)
    owned_similarity = (
        similarity_backend is None and stage in {"sample", "all"}
    )
    if similarity_backend is None and stage in {"sample", "all"}:
        similarity_backend = _build_rf_similarity_backend(config)

    sampled = 0
    evaluated = 0
    errors = 0
    invalid_answers = 0
    try:
        if stage in {"sample", "all"}:
            if similarity_backend is None:
                raise AssertionError("Sampling requires a template similarity backend.")
            sampled, sample_errors = _sample_candidate_trajectories(
                client,
                config,
                anchors=anchors,
                templates=templates,
                template_hash=template_hash,
                output_path=candidates_path,
                settings=settings,
                similarity_backend=similarity_backend,
            )
            errors += sample_errors
        if stage in {"evaluate", "all"}:
            (
                evaluated,
                evaluation_errors,
                invalid_answers,
            ) = _evaluate_candidate_trajectories(
                client,
                config,
                anchors=anchors,
                case_by_id=case_by_id,
                xsim=xsim,
                templates=templates,
                candidates_path=candidates_path,
                output_path=evaluations_path,
                settings=settings,
            )
            errors += evaluation_errors
    finally:
        if owned_client and hasattr(client, "close"):
            client.close()
        if owned_similarity and hasattr(similarity_backend, "close"):
            similarity_backend.close()

    manifest = {
        **manifest_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "planner_train_cases": len(cases),
        "requested_anchors": len(anchors),
        "sample_records_added": sampled,
        "evaluation_records_added": evaluated,
        "errors": errors,
        "invalid_answer_records": invalid_answers,
        "candidates_path": str(candidates_path),
        "evaluations_path": str(evaluations_path),
        "xsim_cases_per_anchor": 3,
        "reward": "mean binary accuracy over anchor plus two X_sim neighbors",
        "trajectory_evaluation": (
            "Retrieve each candidate template sequence once from the anchor plan, "
            "then execute that fixed sequence on all three X_sim cases."
        ),
    }
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def _sample_candidate_trajectories(
    client: GenerationClient,
    config: dict[str, Any],
    *,
    anchors: list[NormalizedCase],
    templates: list[LegalFluxTemplate],
    template_hash: str,
    output_path: Path,
    settings: dict[str, Any],
    similarity_backend: SimilarityBackend,
) -> tuple[int, int]:
    ledger = JsonlLedger(output_path)
    schema = _load_schema(
        resolve_path(config, "schemas_dir") / "legal_flux_abstract_plan.json"
    )
    template_examples = _template_tag_examples(config, templates)
    added = 0
    errors = 0
    for anchor in anchors:
        prompt, prompt_hash = render_prompt(
            config,
            "legal_flux/rf_plan",
            anchor,
            max_steps=int(config["legal_flux"].get("max_steps", 4)),
            template_tag_examples=template_examples,
        )
        for sample_index in range(settings["samples_per_anchor"]):
            seed = settings["seed"] + sample_index
            run_hash = make_run_hash(
                task="legal_flux_dpo_candidate",
                anchor_case_id=anchor.case_id,
                sample_index=sample_index,
                seed=seed,
                planner_model=settings["planner_model"],
                prompt_hash=prompt_hash,
                template_pool_hash=template_hash,
            )
            if ledger.contains(run_hash):
                continue
            base = {
                "run_hash": run_hash,
                "candidate_id": run_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "anchor_case_id": anchor.case_id,
                "sample_index": sample_index,
                "planner_model": settings["planner_model"],
                "planner_temperature": settings["planner_temperature"],
                "seed": seed,
                "prompt_hash": prompt_hash,
                "template_pool_hash": template_hash,
            }
            try:
                response = client.generate(
                    model=settings["planner_model"],
                    prompt=prompt,
                    schema=schema,
                    temperature=settings["planner_temperature"],
                    seed=seed,
                    context_length=int(config["model"]["context_length"]),
                    max_tokens=int(config["model"]["flux_plan_max_tokens"]),
                )
                normalized, repairs = _normalize_abstract_plan_payload(
                    response.parsed,
                    max_steps=int(config["legal_flux"].get("max_steps", 4)),
                )
                plan = LegalFluxAbstractPlan.model_validate(normalized)
                retrieval = _retrieve_fixed_trajectory(
                    plan,
                    templates,
                    similarity_backend=similarity_backend,
                )
                record = {
                    **base,
                    "status": "ok",
                    "trajectory_plan": plan.model_dump(mode="json"),
                    "retrieved_template_ids": [
                        item["template_id"] for item in retrieval
                    ],
                    "retrieval_trace": retrieval,
                    "raw_response": response.raw_text,
                    "prompt_tokens": response.prompt_tokens,
                    "output_tokens": response.output_tokens,
                    "elapsed_seconds": response.elapsed_seconds,
                    "repair_actions": repairs,
                }
            except Exception as exc:
                record = {
                    **base,
                    "status": "error",
                    "trajectory_plan": None,
                    "retrieved_template_ids": [],
                    "retrieval_trace": [],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                errors += 1
            ledger.append(record)
            added += 1
    return added, errors


def _evaluate_candidate_trajectories(
    client: GenerationClient,
    config: dict[str, Any],
    *,
    anchors: list[NormalizedCase],
    case_by_id: dict[str, NormalizedCase],
    xsim: dict[str, list[str]],
    templates: list[LegalFluxTemplate],
    candidates_path: Path,
    output_path: Path,
    settings: dict[str, Any],
) -> tuple[int, int, int]:
    if not candidates_path.exists():
        raise RuntimeError(
            f"Candidate trajectories do not exist at {candidates_path}. "
            "Run stage `sample` first."
        )
    requested_anchors = {case.case_id for case in anchors}
    candidates = [
        row
        for row in read_jsonl(candidates_path)
        if row.get("status") == "ok"
        and row.get("anchor_case_id") in requested_anchors
    ]
    template_by_id = {template.template_id: template for template in templates}
    ledger = JsonlLedger(output_path)
    added = 0
    errors = 0
    invalid_answers = 0
    for candidate in candidates:
        anchor_id = str(candidate["anchor_case_id"])
        plan = LegalFluxAbstractPlan.model_validate(candidate["trajectory_plan"])
        template_ids = [str(value) for value in candidate["retrieved_template_ids"]]
        selected_templates = [template_by_id[template_id] for template_id in template_ids]
        for target_id in xsim[anchor_id]:
            target = case_by_id[target_id]
            run_hash = make_run_hash(
                task="legal_flux_dpo_evaluation",
                candidate_id=candidate["candidate_id"],
                target_case_id=target_id,
                executor_model=settings["executor_model"],
            )
            if ledger.contains(run_hash):
                continue
            base = {
                "run_hash": run_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "candidate_id": candidate["candidate_id"],
                "anchor_case_id": anchor_id,
                "target_case_id": target_id,
                "is_anchor": target_id == anchor_id,
                "sample_index": candidate["sample_index"],
                "executor_model": settings["executor_model"],
                "gold_answer": target.gold_answer,
            }
            try:
                decision, trace = _execute_fixed_trajectory(
                    client,
                    config,
                    target,
                    abstract_plan=plan,
                    templates=selected_templates,
                    executor_model=settings["executor_model"],
                    seed=settings["seed"],
                )
                record = {
                    **base,
                    "status": "ok",
                    "final_decision": decision,
                    "answer_valid": True,
                    "answer_correct": decision == target.gold_answer,
                    **trace,
                }
            except InvalidFinalDecisionError as exc:
                record = {
                    **base,
                    "status": "ok",
                    "final_decision": None,
                    "answer_valid": False,
                    "answer_correct": False,
                    "retrieved_template_ids": template_ids,
                    "invalid_answer_reason": str(exc),
                }
                invalid_answers += 1
            except Exception as exc:
                record = {
                    **base,
                    "status": "error",
                    "final_decision": None,
                    "answer_valid": False,
                    "answer_correct": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                errors += 1
            ledger.append(record)
            added += 1
    return added, errors, invalid_answers


def _retrieve_fixed_trajectory(
    plan: LegalFluxAbstractPlan,
    templates: list[LegalFluxTemplate],
    *,
    similarity_backend: SimilarityBackend,
) -> list[dict[str, Any]]:
    used_template_ids: set[str] = set()
    result: list[dict[str, Any]] = []
    for step in plan.planned_steps:
        retrieval = retrieve_template_for_abstract_step(
            step,
            templates,
            similarity_backend=similarity_backend,
            exclude_template_ids=used_template_ids,
        )
        template = retrieval["template"]
        used_template_ids.add(template.template_id)
        result.append(
            {
                "step_id": step.step_id,
                "step_name": step.step_name,
                "template_tags": step.template_tags,
                "template_id": template.template_id,
                "template_name": template.template_name,
                "retrieval_mode": retrieval["retrieval_mode"],
                "similarity": retrieval["similarity"],
                "exact_candidate_ids": retrieval["exact_candidate_ids"],
            }
        )
    return result


def _execute_fixed_trajectory(
    client: GenerationClient,
    config: dict[str, Any],
    case: NormalizedCase,
    *,
    abstract_plan: LegalFluxAbstractPlan,
    templates: list[LegalFluxTemplate],
    executor_model: str,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    if len(abstract_plan.planned_steps) != len(templates):
        raise ValueError("Fixed trajectory step and template counts do not match.")
    common = {
        "model": executor_model,
        "temperature": 0.0,
        "seed": seed,
        "context_length": int(config["model"]["context_length"]),
    }
    artifacts: list[LegalFluxStepArtifact] = []
    selected_trace: list[dict[str, Any]] = []
    prompt_hashes: dict[str, str] = {}
    calls = 0
    elapsed_seconds = 0.0
    prompt_tokens = 0
    output_tokens = 0
    repair_actions: list[str] = []
    schema_errors: list[str] = []

    for abstract_step, template in zip(
        abstract_plan.planned_steps, templates, strict=True
    ):
        step = _abstract_step_to_plan_step(abstract_step, template)
        artifact, trace = _instantiate_step(
            client,
            config,
            case,
            step=step,
            template=template,
            prior_artifacts=artifacts,
            common=common,
        )
        artifacts.append(artifact)
        selected_trace.append(
            {
                "step_id": abstract_step.step_id,
                "step_name": abstract_step.step_name,
                "template_id": template.template_id,
                "template_name": template.template_name,
            }
        )
        prompt_hashes[f"instantiate_{abstract_step.step_id}"] = trace["prompt_hash"]
        calls += 1
        elapsed_seconds += trace["elapsed_seconds"]
        prompt_tokens += trace["prompt_tokens"] or 0
        output_tokens += trace["output_tokens"] or 0
        repair_actions.extend(trace["repair_actions"])
        schema_errors.extend(trace["schema_errors"])

    review, review_trace = _review_rf_trajectory(
        client,
        config,
        case,
        abstract_plan=abstract_plan,
        artifacts=artifacts,
        remaining=[],
        selected_templates=selected_trace,
        common=common,
        max_steps=0,
        force_final_answer=True,
    )
    if review.final_decision not in {"support", "reject"}:
        raise InvalidFinalDecisionError(
            "Fixed trajectory finalizer did not return support/reject."
        )
    prompt_hashes["fixed_finalize"] = review_trace["prompt_hash"]
    calls += 1
    elapsed_seconds += review_trace["elapsed_seconds"]
    prompt_tokens += review_trace["prompt_tokens"] or 0
    output_tokens += review_trace["output_tokens"] or 0
    repair_actions.extend(review_trace["repair_actions"])
    schema_errors.extend(review_trace["schema_errors"])
    return review.final_decision, {
        "retrieved_template_ids": [template.template_id for template in templates],
        "executed_steps": [
            artifact.model_dump(mode="json") for artifact in artifacts
        ],
        "final_rationale": review.final_rationale or review.review_analysis,
        "prompt_hashes": prompt_hashes,
        "calls": calls,
        "elapsed_seconds": elapsed_seconds,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "repair_actions": repair_actions,
        "schema_errors": schema_errors,
    }


def _planner_train_cases(config: dict[str, Any]) -> list[NormalizedCase]:
    return [
        case
        for case in load_cases(config)
        if case.metadata.get("selection_split") == "planner_train"
    ]


def _dpo_settings(config: dict[str, Any]) -> dict[str, Any]:
    values = config.get("dpo", {})
    return {
        "samples_per_anchor": int(values.get("samples_per_anchor", 4)),
        "planner_model": values.get("planner_model") or config["model"]["name"],
        "executor_model": values.get("executor_model") or config["model"]["name"],
        "planner_temperature": float(values.get("planner_temperature", 0.7)),
        "seed": int(values.get("seed", config["project"]["seed"])),
        "candidates_file": values.get(
            "candidates_file", "trajectory_candidates.jsonl"
        ),
        "evaluations_file": values.get(
            "evaluations_file", "trajectory_evaluations.jsonl"
        ),
        "manifest_file": values.get(
            "manifest_file", "trajectory_collection_manifest.json"
        ),
    }


def _training_dir(config: dict[str, Any]) -> Path:
    return resolve_path(config, "processed_dir") / "planner_training"


def _ollama_client(
    config: dict[str, Any],
    required_models: set[str],
) -> OllamaClient:
    client = OllamaClient(
        config["model"]["base_url"],
        config["model"]["timeout_seconds"],
    )
    missing = [
        model
        for model in required_models
        if not client.model_info(model)
    ]
    if missing:
        client.close()
        raise RuntimeError(f"Configured Ollama models are unavailable: {missing}")
    return client


def _guard_existing_manifest(
    manifest_path: Path,
    expected: dict[str, Any],
    *,
    force: bool,
) -> None:
    if force or not manifest_path.exists():
        return
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = {
        key: (existing.get(key), value)
        for key, value in expected.items()
        if existing.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Existing DPO construction artifacts were built with different "
            f"settings. Use --force to rebuild them. Mismatches: {mismatches}"
        )
