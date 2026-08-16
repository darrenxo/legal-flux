from __future__ import annotations

import json
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .clients import ModelResponse, build_generation_client
from .config import resolve_path
from .embeddings import SimilarityBackend
from .io_utils import atomic_write_json, canonical_json, read_jsonl, sha256_text
from .ledger import JsonlLedger, make_run_hash
from .legal_flux import (
    load_template_pool,
    legal_flux_workflow_hash,
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
    num_shards: int = 1,
    shard_index: int = 0,
    force: bool = False,
    fail_on_errors: bool = False,
    client: GenerationClient | None = None,
    similarity_backend: SimilarityBackend | None = None,
) -> dict[str, Any]:
    if stage not in {"sample", "evaluate", "all"}:
        raise ValueError(f"Unsupported DPO construction stage: {stage}")
    settings = _dpo_settings(config)
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    if settings["samples_per_anchor"] != 4:
        raise ValueError(
            "LegalFlux DPO construction is configured for exactly four "
            "candidate trajectories per anchor."
        )
    cases = _planner_train_cases(config)
    requested_anchors = cases[:case_limit] if case_limit is not None else cases
    anchors = _select_dpo_shard(
        requested_anchors,
        num_shards=num_shards,
        shard_index=shard_index,
    )
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
    workflow_hash = dpo_construction_workflow_hash(config)
    xsim_hash = _xsim_neighbors_hash(config)
    output_dir = _training_dir(config)
    artifact_dir = _dpo_artifact_dir(
        output_dir,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = artifact_dir / settings["candidates_file"]
    evaluations_path = artifact_dir / settings["evaluations_file"]
    manifest_path = artifact_dir / settings["manifest_file"]
    owned_client = client is None
    if client is None:
        client = build_generation_client(config)
    required_models = {
        settings["planner_model"],
        settings["executor_model"],
        settings["reviewer_model"],
    }
    model_digests = _model_digests(
        client,
        required_models,
        provider=str(config["model"].get("provider", "ollama")),
        base_url=str(config["model"]["base_url"]),
        require_available=owned_client,
    )
    manifest_key = {
        "workflow_hash": workflow_hash,
        "template_pool_hash": template_hash,
        "xsim_hash": xsim_hash,
        "planner_model": settings["planner_model"],
        "executor_model": settings["executor_model"],
        "reviewer_model": settings["reviewer_model"],
        "source_checkpoint": settings["source_checkpoint"],
        "model_digests": model_digests,
        "samples_per_anchor": settings["samples_per_anchor"],
        "planner_temperature": settings["planner_temperature"],
        "seed": settings["seed"],
        "max_steps": int(config["legal_flux"].get("max_steps", 4)),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "anchor_ids_hash": sha256_text(
            canonical_json([case.case_id for case in anchors])
        ),
    }
    _guard_existing_manifest(manifest_path, manifest_key, force=force)
    if force:
        candidates_path.unlink(missing_ok=True)
        evaluations_path.unlink(missing_ok=True)
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
                workflow_hash=workflow_hash,
                planner_digest=model_digests[settings["planner_model"]],
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
                template_hash=template_hash,
                workflow_hash=workflow_hash,
                executor_digest=model_digests[settings["executor_model"]],
                reviewer_digest=model_digests[settings["reviewer_model"]],
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
        "unsharded_requested_anchors": len(requested_anchors),
        "sample_records_added": sampled,
        "evaluation_records_added": evaluated,
        "errors": errors,
        "invalid_answer_records": invalid_answers,
        "candidates_path": str(candidates_path),
        "evaluations_path": str(evaluations_path),
        "xsim_cases_per_anchor": 3,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "reward": "mean binary accuracy over anchor plus two X_sim neighbors",
        "trajectory_evaluation": (
            "Retrieve each candidate template sequence once from the anchor plan, "
            "then execute every step of that fixed sequence on all three X_sim "
            "cases with no intermediate review, followed by one forced-finalization "
            "call."
        ),
    }
    atomic_write_json(manifest_path, manifest)
    if fail_on_errors and errors:
        raise RuntimeError(
            f"DPO construction preserved {errors} error record(s) under "
            f"{artifact_dir}. Rerun the same shard to retry failed work."
        )
    return {**manifest, "manifest_path": str(manifest_path)}


def _sample_candidate_trajectories(
    client: GenerationClient,
    config: dict[str, Any],
    *,
    anchors: list[NormalizedCase],
    templates: list[LegalFluxTemplate],
    template_hash: str,
    workflow_hash: str,
    planner_digest: str,
    output_path: Path,
    settings: dict[str, Any],
    similarity_backend: SimilarityBackend,
) -> tuple[int, int]:
    ledger = JsonlLedger(output_path)
    schema = _load_schema(
        resolve_path(config, "schemas_dir") / "legal_flux_abstract_plan.json"
    )
    added = 0
    errors = 0
    for anchor in anchors:
        prompt, prompt_hash = render_prompt(
            config,
            "legal_flux/rf_plan",
            anchor,
            max_steps=int(config["legal_flux"].get("max_steps", 4)),
        )
        for sample_index in range(settings["samples_per_anchor"]):
            seed = settings["seed"] + sample_index
            run_hash = make_run_hash(
                task="legal_flux_dpo_candidate",
                anchor_case_id=anchor.case_id,
                sample_index=sample_index,
                seed=seed,
                planner_model=settings["planner_model"],
                planner_digest=planner_digest,
                prompt_hash=prompt_hash,
                workflow_hash=workflow_hash,
                template_pool_hash=template_hash,
                source_checkpoint=settings["source_checkpoint"],
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
                "planner_digest": planner_digest,
                "planner_temperature": settings["planner_temperature"],
                "seed": seed,
                "prompt_hash": prompt_hash,
                "template_pool_hash": template_hash,
                "workflow_hash": workflow_hash,
                "source_checkpoint": settings["source_checkpoint"],
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
    template_hash: str,
    workflow_hash: str,
    executor_digest: str,
    reviewer_digest: str,
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
        and row.get("template_pool_hash") == template_hash
        and row.get("workflow_hash") == workflow_hash
        and row.get("source_checkpoint") == settings["source_checkpoint"]
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
                reviewer_model=settings["reviewer_model"],
                executor_digest=executor_digest,
                reviewer_digest=reviewer_digest,
                workflow_hash=workflow_hash,
                template_pool_hash=template_hash,
                source_checkpoint=settings["source_checkpoint"],
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
                "reviewer_model": settings["reviewer_model"],
                "executor_digest": executor_digest,
                "reviewer_digest": reviewer_digest,
                "workflow_hash": workflow_hash,
                "template_pool_hash": template_hash,
                "source_checkpoint": settings["source_checkpoint"],
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
                    reviewer_model=settings["reviewer_model"],
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
    reviewer_model: str,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    if len(abstract_plan.planned_steps) != len(templates):
        raise ValueError("Fixed trajectory step and template counts do not match.")
    executor_common = {
        "model": executor_model,
        "temperature": 0.0,
        "seed": seed,
        "context_length": int(config["model"]["context_length"]),
    }
    reviewer_common = {**executor_common, "model": reviewer_model}
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
            common=executor_common,
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
        artifacts=artifacts,
        remaining=[],
        selected_templates=selected_trace,
        common=reviewer_common,
        max_steps=int(config["legal_flux"].get("max_steps", 4)),
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
        "executor_calls": len(artifacts),
        "intermediate_review_calls": 0,
        "forced_finalization_calls": 1,
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
    flux_values = config.get("legal_flux", {})
    planner_model = (
        values.get("planner_model")
        or flux_values.get("planner_model")
        or config["model"]["name"]
    )
    executor_model = (
        values.get("executor_model")
        or config["model"]["name"]
    )
    reviewer_model = (
        values.get("reviewer_model")
        or flux_values.get("reviewer_model")
        or planner_model
    )
    return {
        "samples_per_anchor": int(values.get("samples_per_anchor", 4)),
        "planner_model": str(planner_model),
        "executor_model": str(executor_model),
        "reviewer_model": str(reviewer_model),
        "source_checkpoint": (
            str(values["source_checkpoint"])
            if values.get("source_checkpoint")
            else None
        ),
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


def dpo_construction_workflow_hash(config: dict[str, Any]) -> str:
    runtime_config = deepcopy(config)
    settings = _dpo_settings(config)
    legal_flux = runtime_config.setdefault("legal_flux", {})
    for role in ("planner", "executor", "reviewer"):
        legal_flux[f"{role}_model"] = settings[f"{role}_model"]
    return sha256_text(
        canonical_json(
            {
                "runtime_workflow_hash": legal_flux_workflow_hash(runtime_config),
                "dpo_construction_implementation": sha256_text(
                    Path(__file__).read_text(encoding="utf-8")
                ),
            }
        )
    )


def _select_dpo_shard(
    anchors: list[NormalizedCase],
    *,
    num_shards: int,
    shard_index: int,
) -> list[NormalizedCase]:
    ordered = sorted(anchors, key=lambda case: (case.case_id, case.variant_id))
    return [
        case
        for index, case in enumerate(ordered)
        if index % num_shards == shard_index
    ]


def _dpo_artifact_dir(
    output_dir: Path,
    *,
    num_shards: int,
    shard_index: int,
) -> Path:
    if num_shards == 1:
        return output_dir
    return (
        output_dir
        / "dpo_shards"
        / f"shard-{shard_index:05d}-of-{num_shards:05d}"
    )


def _model_digests(
    client: GenerationClient,
    required_models: set[str],
    *,
    provider: str,
    base_url: str,
    require_available: bool,
) -> dict[str, str]:
    if not require_available or not hasattr(client, "model_info"):
        return {model: f"injected:{model}" for model in sorted(required_models)}
    model_infos = {
        model: client.model_info(model)  # type: ignore[attr-defined]
        for model in sorted(required_models)
    }
    missing = [model for model, info in model_infos.items() if not info]
    if missing:
        raise RuntimeError(
            f"Models {missing!r} are not exposed by {provider} at {base_url}."
        )
    return {
        model: str(info.get("digest", "unknown"))
        for model, info in model_infos.items()
        if info is not None
    }


def _xsim_neighbors_hash(config: dict[str, Any]) -> str:
    filename = str(
        config.get("xsim", {}).get("neighbors_file", "xsim_neighbors.jsonl")
    )
    path = resolve_path(config, "processed_dir") / "xsim" / filename
    return sha256_text(path.read_text(encoding="utf-8"))


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
