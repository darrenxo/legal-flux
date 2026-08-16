from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .clients import GenerationResponseError, build_generation_client
from .config import resolve_path
from .io_utils import canonical_json, latest_by_run_hash, read_jsonl, sha256_text
from .ledger import JsonlLedger, make_run_hash
from .legal_flux import legal_flux_workflow_hash
from .legal_flux_evaluation import _filter_to_run_plans, _generation_paths
from .legal_flux_runner import (
    _analysis_from_rf_review,
    _common_generation_settings,
    _generation_run_dir,
    _print_generation_progress,
    _review_rf_trajectory,
    _role_model,
    _update_run_counts,
    _validated_run_tag,
)
from .models import (
    LegalFluxAbstractPlan,
    LegalFluxStepArtifact,
    NormalizedCase,
)
from .runner import load_cases


def run_legal_flux_final_review_replay(
    config: dict[str, Any],
    *,
    phase: str,
    source_run_tag: str,
    run_tag: str,
    dry_run: bool = False,
    case_limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    fail_on_errors: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    normalized_phase = phase.replace("-", "_")
    source_tag = _validated_run_tag(source_run_tag)
    target_tag = _validated_run_tag(run_tag)
    if source_tag is None or target_tag is None:
        raise ValueError("source_run_tag and run_tag are required.")
    if source_tag == target_tag:
        raise ValueError("The replay run_tag must differ from source_run_tag.")
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    if case_limit is not None and case_limit < 1:
        raise ValueError("case_limit must be a positive integer.")

    phase_dir = resolve_path(config, "runs_dir") / normalized_phase
    source_dir = phase_dir / "experiments" / source_tag
    source_paths = _generation_paths(source_dir)
    source_rows = latest_by_run_hash(
        [row for path in source_paths for row in read_jsonl(path)]
    )
    source_rows = _filter_to_run_plans(
        source_rows,
        [path.parent / "run_plan.json" for path in source_paths],
    )
    source_rows = [
        row
        for row in source_rows
        if row.get("status") == "ok" and row.get("condition") == "flux_rf_style"
    ]
    source_rows.sort(key=_source_sort_key)
    if case_limit is not None:
        source_rows = source_rows[:case_limit]
    selected_rows = [
        row
        for index, row in enumerate(source_rows)
        if index % num_shards == shard_index
    ]
    if not source_rows:
        raise ValueError(
            f"Source run {source_tag!r} contains no successful flux_rf_style records."
        )

    target_dir = _generation_run_dir(
        config,
        normalized_phase,
        num_shards=num_shards,
        shard_index=shard_index,
        run_tag=target_tag,
    )
    if dry_run:
        return {
            "phase": normalized_phase,
            "source_run_tag": source_tag,
            "run_tag": target_tag,
            "source_records": len(source_rows),
            "jobs": len(selected_rows),
            "run_dir": str(target_dir),
            "num_shards": num_shards,
            "shard_index": shard_index,
            "dry_run": True,
        }

    cases = {
        (case.dataset, case.case_id, case.variant_id): case
        for case in load_cases(config)
    }
    missing_cases = [
        (row["dataset"], row["case_id"], row.get("variant_id", "original"))
        for row in selected_rows
        if (
            row["dataset"],
            row["case_id"],
            row.get("variant_id", "original"),
        )
        not in cases
    ]
    if missing_cases:
        raise ValueError(
            f"Replay source contains {len(missing_cases)} cases absent from the "
            f"current case file; first: {missing_cases[:5]}"
        )

    owns_client = client is None
    generation_client = client or build_generation_client(config)
    reviewer_model = _role_model(config, "reviewer")
    try:
        model_info = generation_client.model_info(reviewer_model)
        if not model_info:
            raise RuntimeError(
                f"Reviewer model {reviewer_model!r} is not exposed at "
                f"{config['model']['base_url']}."
            )
        reviewer_digest = str(model_info.get("digest", "unknown"))
        source_reviewer_models = {
            str(
                (row.get("role_models") or {}).get("reviewer")
                or row.get("model_name")
                or ""
            )
            for row in selected_rows
        }
        source_reviewer_models.discard("")
        if source_reviewer_models and source_reviewer_models != {reviewer_model}:
            raise ValueError(
                "Configured reviewer model does not match the source run: "
                f"configured={reviewer_model!r}, "
                f"source={sorted(source_reviewer_models)!r}."
            )

        workflow_hash = sha256_text(
            canonical_json(
                {
                    "kind": "legal_flux_final_review_replay_v1",
                    "legal_flux_workflow_hash": legal_flux_workflow_hash(config),
                }
            )
        )
        jobs = [
            {
                "source": row,
                "case": cases[
                    (
                        row["dataset"],
                        row["case_id"],
                        row.get("variant_id", "original"),
                    )
                ],
                "run_hash": make_run_hash(
                    kind="legal_flux_final_review_replay_v1",
                    source_run_hash=row["run_hash"],
                    reviewer_model=reviewer_model,
                    reviewer_digest=reviewer_digest,
                    workflow_hash=workflow_hash,
                    seed=row.get("seed", config["model"]["seed"]),
                    decoding=row.get("decoding") or {},
                ),
            }
            for row in selected_rows
        ]
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "run_plan.json").write_text(
            json.dumps(
                {
                    "phase": normalized_phase,
                    "source_run_tag": source_tag,
                    "run_tag": target_tag,
                    "kind": "legal_flux_final_review_replay_v1",
                    "reviewer_model": reviewer_model,
                    "model_digest": reviewer_digest,
                    "workflow_hash": workflow_hash,
                    "num_shards": num_shards,
                    "shard_index": shard_index,
                    "job_count": len(jobs),
                    "jobs": [
                        {
                            "run_hash": job["run_hash"],
                            "source_run_hash": job["source"]["run_hash"],
                            "case_id": job["case"].case_id,
                            "condition": "flux_rf_style",
                            "phase": normalized_phase,
                        }
                        for job in jobs
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        ledger = JsonlLedger(target_dir / "generations.jsonl")
        completed = 0
        skipped = 0
        errors = 0
        concurrency = max(1, int(config["model"].get("concurrency", 1)))
        if concurrency == 1:
            records = (
                _run_final_review_replay_job(
                    generation_client,
                    config,
                    job,
                    ledger=ledger,
                    reviewer_model=reviewer_model,
                    reviewer_digest=reviewer_digest,
                    workflow_hash=workflow_hash,
                    source_run_tag=source_tag,
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
                        _run_final_review_replay_job,
                        generation_client,
                        config,
                        job,
                        ledger=ledger,
                        reviewer_model=reviewer_model,
                        reviewer_digest=reviewer_digest,
                        workflow_hash=workflow_hash,
                        source_run_tag=source_tag,
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
        if owns_client:
            generation_client.close()

    result = {
        "phase": normalized_phase,
        "source_run_tag": source_tag,
        "run_tag": target_tag,
        "jobs": len(selected_rows),
        "completed": completed,
        "skipped": skipped,
        "errors": errors,
        "run_dir": str(target_dir),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "dry_run": False,
    }
    if fail_on_errors and errors:
        raise RuntimeError(
            f"Final-review replay preserved {errors} failed records under {target_dir}."
        )
    return result


def _run_final_review_replay_job(
    client: Any,
    config: dict[str, Any],
    job: dict[str, Any],
    *,
    ledger: JsonlLedger,
    reviewer_model: str,
    reviewer_digest: str,
    workflow_hash: str,
    source_run_tag: str,
) -> dict[str, Any] | None:
    run_hash = job["run_hash"]
    if ledger.contains(run_hash):
        return None
    source = job["source"]
    case: NormalizedCase = job["case"]
    base = {
        "run_hash": run_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": case.dataset,
        "case_id": case.case_id,
        "variant_id": case.variant_id,
        "condition": "flux_rf_style",
        "phase": source["phase"],
        "replay_kind": "final_review_only",
        "source_run_tag": source_run_tag,
        "source_run_hash": source["run_hash"],
        "model_name": config["model"]["name"],
        "role_models": {"reviewer": reviewer_model},
        "model_digest": reviewer_digest,
        "workflow_hash": workflow_hash,
        "template_pool_hash": source.get("template_pool_hash"),
        "seed": source.get("seed", config["model"]["seed"]),
        "sample_index": source.get("sample_index", 0),
        "gold_answer": case.gold_answer,
        "metadata": case.metadata,
    }
    try:
        abstract_plan = LegalFluxAbstractPlan.model_validate(source["trajectory_plan"])
        artifacts = [
            LegalFluxStepArtifact.model_validate(value)
            for value in source["executed_steps"]
        ]
        if not artifacts:
            raise ValueError("Source replay record contains no executed artifacts.")
        selected_templates = source.get("selected_templates")
        if not isinstance(selected_templates, list):
            raise ValueError("Source replay record has no selected-template trace.")

        common = _common_generation_settings(config, role="reviewer")
        source_decoding = source.get("decoding") or {}
        common["temperature"] = float(
            source_decoding.get("temperature", common["temperature"])
        )
        common["seed"] = int(source.get("seed", common["seed"]))
        common["context_length"] = int(
            source_decoding.get("context_length", common["context_length"])
        )
        review, trace = _review_rf_trajectory(
            client,
            config,
            case,
            artifacts=artifacts,
            remaining=[],
            selected_templates=selected_templates,
            common=common,
            max_steps=int(config["legal_flux"].get("max_steps", 4)),
            force_final_answer=True,
        )
        analysis = _analysis_from_rf_review(review)
        record = {
            **base,
            "status": "ok",
            "prompt_hash": trace["prompt_hash"],
            "raw_response": trace["raw_response"],
            "parsed_json": analysis.model_dump(mode="json", exclude_defaults=True),
            "trajectory_plan": abstract_plan.model_dump(mode="json"),
            "executed_steps": [
                artifact.model_dump(mode="json") for artifact in artifacts
            ],
            "trajectory_reviews": [review.model_dump(mode="json")],
            "retrieved_template_ids": source.get("retrieved_template_ids"),
            "selected_templates": selected_templates,
            "prompt_hashes": {"rf_review_replay_final": trace["prompt_hash"]},
            "decoding": {
                "temperature": common["temperature"],
                "context_length": common["context_length"],
            },
            "elapsed_seconds": trace["elapsed_seconds"],
            "prompt_tokens": trace["prompt_tokens"],
            "output_tokens": trace["output_tokens"],
            "schema_errors": trace["schema_errors"],
            "repair_actions": trace["repair_actions"],
            "calls": 1,
        }
    except Exception as exc:
        record = {
            **base,
            "status": "error",
            "prompt_hash": None,
            "raw_response": (
                exc.raw_text if isinstance(exc, GenerationResponseError) else None
            ),
            "parsed_json": None,
            "trajectory_plan": source.get("trajectory_plan"),
            "executed_steps": source.get("executed_steps"),
            "trajectory_reviews": None,
            "retrieved_template_ids": source.get("retrieved_template_ids"),
            "selected_templates": source.get("selected_templates"),
            "prompt_hashes": {},
            "decoding": source.get("decoding") or {},
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


def _source_sort_key(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
    return (
        str(row.get("dataset") or ""),
        str(row.get("case_id") or ""),
        str(row.get("variant_id") or "original"),
        int(row.get("sample_index") or 0),
        str(row.get("run_hash") or ""),
    )
