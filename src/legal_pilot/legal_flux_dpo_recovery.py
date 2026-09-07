from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from jsonschema import Draft202012Validator

from .clients import ModelResponse, build_generation_client
from .config import load_config, resolve_path
from .embeddings import SimilarityBackend
from .io_utils import (
    atomic_write_json,
    canonical_json,
    latest_by_run_hash,
    read_jsonl,
    sha256_text,
)
from .ledger import JsonlLedger, make_run_hash
from .legal_flux import load_template_pool, template_pool_hash
from .legal_flux_dpo import (
    _dpo_artifact_dir,
    _dpo_settings,
    _evaluate_candidate_trajectories,
    _model_digests,
    _planner_train_cases,
    _retrieve_fixed_trajectory,
    _select_dpo_shard,
    _training_dir,
    _xsim_neighbors_hash,
    dpo_construction_workflow_hash,
)
from .legal_flux_runner import (
    _build_rf_similarity_backend,
    _load_schema,
    _normalize_abstract_plan_payload,
)
from .legal_flux_xsim import load_xsim_neighbors
from .models import LegalFluxAbstractPlan, LegalFluxTemplate, NormalizedCase
from .prompting import render_prompt


class GenerationClient(Protocol):
    def generate(self, **kwargs: Any) -> ModelResponse:
        ...


def recover_dpo_candidates(
    config: dict[str, Any],
    *,
    num_shards: int,
    shard_index: int,
    case_limit: int | None = None,
    fallback_seed_stride: int | None = None,
    max_recovery_attempts: int = 1,
    evaluate_missing: bool = True,
    client: GenerationClient | None = None,
    similarity_backend: SimilarityBackend | None = None,
) -> dict[str, Any]:
    """Recover only malformed planner candidates in one existing DPO shard.

    This deliberately lives outside ``legal_flux_dpo.py`` so that repairing an
    almost-complete historical collection does not change that collection's
    workflow hash. The logical run hash and original sampling seed remain
    unchanged; an alternate generation seed is recorded separately.
    """

    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    if max_recovery_attempts < 1:
        raise ValueError("max_recovery_attempts must be at least 1.")

    settings = _dpo_settings(config)
    samples_per_anchor = int(settings["samples_per_anchor"])
    if samples_per_anchor != 4:
        raise ValueError(
            "LegalFlux DPO recovery expects the four-candidate collection design."
        )
    seed_stride = (
        samples_per_anchor
        if fallback_seed_stride is None
        else int(fallback_seed_stride)
    )
    if seed_stride < samples_per_anchor:
        raise ValueError(
            "fallback_seed_stride must be at least samples_per_anchor so a "
            "fallback seed cannot collide with another candidate slot."
        )

    cases = _planner_train_cases(config)
    requested_anchors = cases[:case_limit] if case_limit is not None else cases
    anchors = _select_dpo_shard(
        requested_anchors,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    case_by_id = {case.case_id: case for case in cases}
    artifact_dir = _dpo_artifact_dir(
        _training_dir(config),
        num_shards=num_shards,
        shard_index=shard_index,
    )
    candidates_path = artifact_dir / settings["candidates_file"]
    evaluations_path = artifact_dir / settings["evaluations_file"]
    manifest_path = artifact_dir / settings["manifest_file"]
    recovery_manifest_path = artifact_dir / "trajectory_recovery_manifest.json"
    recovery_implementation_hash = sha256_text(
        Path(__file__).read_text(encoding="utf-8")
    )
    if not manifest_path.is_file() or not candidates_path.is_file():
        raise FileNotFoundError(
            "DPO recovery requires an existing shard manifest and candidate "
            f"ledger under {artifact_dir}."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    templates = load_template_pool(config)
    current_template_hash = template_pool_hash(templates)
    current_workflow_hash = dpo_construction_workflow_hash(config)
    current_xsim_hash = _xsim_neighbors_hash(config)

    owned_client = client is None
    if client is None:
        client = build_generation_client(config)
    owned_similarity = similarity_backend is None
    try:
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

        expected_manifest = {
            "workflow_hash": current_workflow_hash,
            "template_pool_hash": current_template_hash,
            "xsim_hash": current_xsim_hash,
            "planner_model": settings["planner_model"],
            "executor_model": settings["executor_model"],
            "reviewer_model": settings["reviewer_model"],
            "source_checkpoint": settings["source_checkpoint"],
            "model_digests": model_digests,
            "samples_per_anchor": samples_per_anchor,
            "planner_temperature": settings["planner_temperature"],
            "seed": settings["seed"],
            "max_steps": int(config["legal_flux"].get("max_steps", 4)),
            "num_shards": num_shards,
            "shard_index": shard_index,
            "anchor_ids_hash": sha256_text(
                canonical_json([case.case_id for case in anchors])
            ),
            "requested_anchors": len(anchors),
            "unsharded_requested_anchors": len(requested_anchors),
        }
        mismatches = {
            key: {"manifest": manifest.get(key), "current": value}
            for key, value in expected_manifest.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Refusing targeted DPO recovery because the current runtime does "
                "not match the existing shard manifest. Do not use --force. "
                f"Mismatches: {mismatches}"
            )

        if similarity_backend is None:
            similarity_backend = _build_rf_similarity_backend(config)
    except BaseException:
        if owned_client and hasattr(client, "close"):
            client.close()
        if (
            owned_similarity
            and similarity_backend is not None
            and hasattr(similarity_backend, "close")
        ):
            similarity_backend.close()
        raise

    candidates_added = 0
    candidate_errors = 0
    recovered_candidate_ids: list[str] = []
    attempted_candidate_ids: list[str] = []
    try:
        rows_before = read_jsonl(candidates_path)
        current_rows_before = _current_candidate_rows(
            rows_before,
            manifest=manifest,
        )
        expected_slots = _expected_candidate_slots(
            config,
            anchors=anchors,
            settings=settings,
            planner_digest=model_digests[settings["planner_model"]],
            workflow_hash=current_workflow_hash,
            template_hash=current_template_hash,
        )
        current_by_hash = {
            str(row["run_hash"]): row for row in current_rows_before
        }
        missing_candidate_ids = sorted(set(expected_slots) - set(current_by_hash))
        unexpected_candidate_ids = sorted(set(current_by_hash) - set(expected_slots))
        if missing_candidate_ids or unexpected_candidate_ids:
            raise RuntimeError(
                "Refusing targeted recovery because the shard's logical candidate "
                "IDs do not exactly match its anchor/sample slots. "
                f"Missing: {missing_candidate_ids[:5]}; unexpected: "
                f"{unexpected_candidate_ids[:5]}."
            )
        for run_hash, slot in expected_slots.items():
            _validate_candidate_identity(current_by_hash[run_hash], slot)
        expected_candidates = len(expected_slots)

        unresolved = [
            current_by_hash[run_hash]
            for run_hash in expected_slots
            if current_by_hash[run_hash].get("status") != "ok"
        ]
        ledger = JsonlLedger(candidates_path)
        schema = _load_schema(
            resolve_path(config, "schemas_dir")
            / "legal_flux_abstract_plan.json"
        )
        records_by_hash: dict[str, list[dict[str, Any]]] = {}
        for row in rows_before:
            if row.get("run_hash"):
                records_by_hash.setdefault(str(row["run_hash"]), []).append(row)

        for failed in unresolved:
            _validate_recoverable_candidate_error(failed)
            run_hash = str(failed["run_hash"])
            candidate_id = str(failed["candidate_id"])
            slot = expected_slots[run_hash]
            anchor_id = slot["anchor"].case_id
            sample_index = slot["sample_index"]
            prompt = slot["prompt"]
            prompt_hash = slot["prompt_hash"]
            original_seed = slot["seed"]

            prior_recovery_attempts = [
                int(row.get("generation_attempt", 0))
                for row in records_by_hash.get(run_hash, [])
                if row.get("recovery_method") == "alternate_seed_schema_retry"
            ]
            generation_attempt = max(prior_recovery_attempts, default=0) + 1
            if generation_attempt > max_recovery_attempts:
                raise RuntimeError(
                    f"Candidate {candidate_id} has exhausted its configured "
                    f"{max_recovery_attempts} recovery attempt(s)."
                )
            generation_seed = original_seed + generation_attempt * seed_stride
            if ledger.contains(run_hash):
                raise RuntimeError(
                    f"Candidate {candidate_id} is already marked complete by the "
                    "append-only ledger but appeared unresolved during the audit."
                )
            attempted_candidate_ids.append(candidate_id)
            base = {
                "run_hash": run_hash,
                "candidate_id": candidate_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "anchor_case_id": anchor_id,
                "sample_index": sample_index,
                "planner_model": settings["planner_model"],
                "planner_digest": model_digests[settings["planner_model"]],
                "planner_temperature": settings["planner_temperature"],
                "seed": original_seed,
                "generation_seed": generation_seed,
                "generation_attempt": generation_attempt,
                "recovery_method": "alternate_seed_schema_retry",
                "recovery_implementation_hash": recovery_implementation_hash,
                "prompt_hash": prompt_hash,
                "template_pool_hash": current_template_hash,
                "workflow_hash": current_workflow_hash,
                "source_checkpoint": settings["source_checkpoint"],
                "supersedes_error_attempts": len(records_by_hash.get(run_hash, [])),
            }
            response: ModelResponse | None = None
            normalized: dict[str, Any] | None = None
            try:
                response = client.generate(
                    model=settings["planner_model"],
                    prompt=prompt,
                    schema=schema,
                    temperature=settings["planner_temperature"],
                    seed=generation_seed,
                    context_length=int(config["model"]["context_length"]),
                    max_tokens=int(config["model"]["flux_plan_max_tokens"]),
                )
                normalized, repairs = _normalize_abstract_plan_payload(
                    response.parsed,
                    max_steps=int(config["legal_flux"].get("max_steps", 4)),
                )
                Draft202012Validator(schema).validate(normalized)
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
                    "finish_reason": _finish_reason(response),
                    "response_metadata": response.metadata,
                    "repair_actions": [
                        "dpo_candidate_alternate_seed_recovery",
                        *repairs,
                    ],
                }
                candidate_succeeded = True
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
                    "raw_response": response.raw_text if response else None,
                    "parsed_response": response.parsed if response else None,
                    "normalized_response": normalized,
                    "prompt_tokens": response.prompt_tokens if response else None,
                    "output_tokens": response.output_tokens if response else None,
                    "elapsed_seconds": response.elapsed_seconds if response else None,
                    "finish_reason": _finish_reason(response),
                    "response_metadata": response.metadata if response else None,
                }
                candidate_succeeded = False
            ledger.append(record)
            appended = next(
                (
                    row
                    for row in reversed(read_jsonl(candidates_path))
                    if row.get("run_hash") == run_hash
                ),
                None,
            )
            if appended is None or appended.get("timestamp") != record["timestamp"]:
                raise RuntimeError(
                    f"Candidate ledger did not append the recovery record for "
                    f"{candidate_id}."
                )
            candidates_added += 1
            if candidate_succeeded:
                recovered_candidate_ids.append(candidate_id)
            else:
                candidate_errors += 1

        rows_after = _current_candidate_rows(
            read_jsonl(candidates_path),
            manifest=manifest,
        )
        rows_after_by_hash = {
            str(row["run_hash"]): row for row in rows_after
        }
        if set(rows_after_by_hash) != set(expected_slots):
            raise RuntimeError(
                "Candidate ledger identities changed during targeted recovery."
            )
        for run_hash, slot in expected_slots.items():
            _validate_candidate_identity(rows_after_by_hash[run_hash], slot)
        remaining_candidate_errors = sum(
            row.get("status") != "ok" for row in rows_after_by_hash.values()
        )

        xsim = load_xsim_neighbors(config)
        missing_xsim = [
            anchor.case_id for anchor in anchors if anchor.case_id not in xsim
        ]
        if missing_xsim:
            raise RuntimeError(
                "X_sim neighbors are missing for a recovered shard. First "
                f"missing anchor: {missing_xsim[0]}"
            )
        expected_evaluation_slots = _expected_evaluation_slots(
            rows_after_by_hash,
            xsim=xsim,
            case_by_id=case_by_id,
            settings=settings,
            model_digests=model_digests,
            workflow_hash=current_workflow_hash,
            template_hash=current_template_hash,
        )
        evaluations_added = 0
        evaluation_errors = 0
        invalid_answers_added = 0
        if evaluate_missing and remaining_candidate_errors == 0:
            (
                evaluations_added,
                evaluation_errors,
                invalid_answers_added,
            ) = _evaluate_candidate_trajectories(
                client,
                config,
                anchors=anchors,
                case_by_id=case_by_id,
                xsim=xsim,
                templates=templates,
                template_hash=current_template_hash,
                workflow_hash=current_workflow_hash,
                executor_digest=model_digests[settings["executor_model"]],
                reviewer_digest=model_digests[settings["reviewer_model"]],
                candidates_path=candidates_path,
                output_path=evaluations_path,
                settings=settings,
            )

        evaluation_latest = {
            str(row["run_hash"]): row
            for row in latest_by_run_hash(read_jsonl(evaluations_path))
            if row.get("run_hash")
        }
        expected_evaluation_ids = set(expected_evaluation_slots)
        relevant_evaluation_ids = {
            run_hash
            for run_hash, row in evaluation_latest.items()
            if row.get("workflow_hash") == current_workflow_hash
            and row.get("template_pool_hash") == current_template_hash
            and row.get("source_checkpoint") == settings["source_checkpoint"]
            and row.get("anchor_case_id")
            in {anchor.case_id for anchor in anchors}
        }
        unexpected_evaluation_ids = sorted(
            relevant_evaluation_ids - expected_evaluation_ids
        )
        if unexpected_evaluation_ids:
            raise RuntimeError(
                "Evaluation ledger contains unexpected logical records for this "
                f"shard: {unexpected_evaluation_ids[:5]}"
            )
        present_evaluation_ids = expected_evaluation_ids & set(evaluation_latest)
        for run_hash in present_evaluation_ids:
            _validate_evaluation_identity(
                evaluation_latest[run_hash],
                expected_evaluation_slots[run_hash],
            )
        missing_evaluation_ids = sorted(
            expected_evaluation_ids - present_evaluation_ids
        )
        evaluation_rows = [
            evaluation_latest[run_hash]
            for run_hash in sorted(present_evaluation_ids)
        ]
        expected_evaluations = len(expected_evaluation_ids)
        remaining_evaluation_errors = sum(
            row.get("status") != "ok" for row in evaluation_rows
        )
        missing_evaluations = len(missing_evaluation_ids)
        complete = (
            remaining_candidate_errors == 0
            and (not evaluate_missing or missing_evaluations == 0)
            and evaluation_errors == 0
            and remaining_evaluation_errors == 0
        )
        result = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "recovery_implementation_hash": recovery_implementation_hash,
            "candidate_recovery_implementation_hashes": sorted(
                {
                    str(row["recovery_implementation_hash"])
                    for row in rows_after_by_hash.values()
                    if row.get("recovery_implementation_hash")
                }
            ),
            "shard_dir": str(artifact_dir),
            "num_shards": num_shards,
            "shard_index": shard_index,
            "workflow_hash": current_workflow_hash,
            "template_pool_hash": current_template_hash,
            "source_checkpoint": settings["source_checkpoint"],
            "attempted_candidate_ids": attempted_candidate_ids,
            "recovered_candidate_ids": recovered_candidate_ids,
            "candidate_records_added": candidates_added,
            "candidate_errors_added": candidate_errors,
            "logical_candidates": len(rows_after),
            "expected_candidates": expected_candidates,
            "remaining_candidate_errors": remaining_candidate_errors,
            "evaluation_records_added": evaluations_added,
            "evaluation_errors_added": evaluation_errors,
            "invalid_answers_added": invalid_answers_added,
            "logical_evaluations": len(evaluation_rows),
            "expected_evaluations": expected_evaluations,
            "remaining_evaluation_errors": remaining_evaluation_errors,
            "missing_evaluations": missing_evaluations,
            "missing_evaluation_ids": missing_evaluation_ids[:20],
            "complete": complete,
        }
        atomic_write_json(recovery_manifest_path, result)
        return {**result, "recovery_manifest_path": str(recovery_manifest_path)}
    finally:
        if owned_client and hasattr(client, "close"):
            client.close()
        if owned_similarity and hasattr(similarity_backend, "close"):
            similarity_backend.close()


def _current_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        row
        for row in latest_by_run_hash(rows)
        if row.get("workflow_hash") == manifest.get("workflow_hash")
        and row.get("template_pool_hash") == manifest.get("template_pool_hash")
        and row.get("planner_model") == manifest.get("planner_model")
        and row.get("source_checkpoint") == manifest.get("source_checkpoint")
    ]


def _expected_candidate_slots(
    config: dict[str, Any],
    *,
    anchors: list[NormalizedCase],
    settings: dict[str, Any],
    planner_digest: str,
    workflow_hash: str,
    template_hash: str,
) -> dict[str, dict[str, Any]]:
    slots: dict[str, dict[str, Any]] = {}
    max_steps = int(config["legal_flux"].get("max_steps", 4))
    for anchor in anchors:
        prompt, prompt_hash = render_prompt(
            config,
            "legal_flux/rf_plan",
            anchor,
            max_steps=max_steps,
        )
        for sample_index in range(int(settings["samples_per_anchor"])):
            seed = int(settings["seed"]) + sample_index
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
            slots[run_hash] = {
                "run_hash": run_hash,
                "candidate_id": run_hash,
                "anchor": anchor,
                "anchor_case_id": anchor.case_id,
                "sample_index": sample_index,
                "planner_model": settings["planner_model"],
                "planner_digest": planner_digest,
                "planner_temperature": settings["planner_temperature"],
                "seed": seed,
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "template_pool_hash": template_hash,
                "workflow_hash": workflow_hash,
                "source_checkpoint": settings["source_checkpoint"],
            }
    return slots


def _validate_candidate_identity(
    row: dict[str, Any],
    slot: dict[str, Any],
) -> None:
    comparable_fields = (
        "run_hash",
        "candidate_id",
        "anchor_case_id",
        "sample_index",
        "planner_model",
        "planner_digest",
        "planner_temperature",
        "seed",
        "prompt_hash",
        "template_pool_hash",
        "workflow_hash",
        "source_checkpoint",
    )
    mismatches = {
        key: {"ledger": row.get(key), "expected": slot.get(key)}
        for key in comparable_fields
        if row.get(key) != slot.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "Candidate ledger identity does not match its deterministic "
            f"anchor/sample slot: {mismatches}"
        )


def _expected_evaluation_slots(
    candidates_by_hash: dict[str, dict[str, Any]],
    *,
    xsim: dict[str, list[str]],
    case_by_id: dict[str, NormalizedCase],
    settings: dict[str, Any],
    model_digests: dict[str, str],
    workflow_hash: str,
    template_hash: str,
) -> dict[str, dict[str, Any]]:
    slots: dict[str, dict[str, Any]] = {}
    for candidate in candidates_by_hash.values():
        candidate_id = str(candidate["candidate_id"])
        anchor_id = str(candidate["anchor_case_id"])
        for target_id in xsim[anchor_id]:
            if target_id not in case_by_id:
                raise RuntimeError(
                    f"X_sim target {target_id} for anchor {anchor_id} is not in "
                    "the planner_train case set."
                )
            run_hash = make_run_hash(
                task="legal_flux_dpo_evaluation",
                candidate_id=candidate_id,
                target_case_id=target_id,
                executor_model=settings["executor_model"],
                reviewer_model=settings["reviewer_model"],
                executor_digest=model_digests[settings["executor_model"]],
                reviewer_digest=model_digests[settings["reviewer_model"]],
                workflow_hash=workflow_hash,
                template_pool_hash=template_hash,
                source_checkpoint=settings["source_checkpoint"],
            )
            if run_hash in slots:
                raise RuntimeError(
                    "X_sim produced a duplicate logical candidate/target "
                    f"evaluation for {candidate_id} and {target_id}."
                )
            slots[run_hash] = {
                "run_hash": run_hash,
                "candidate_id": candidate_id,
                "anchor_case_id": anchor_id,
                "target_case_id": target_id,
                "sample_index": candidate["sample_index"],
                "executor_model": settings["executor_model"],
                "reviewer_model": settings["reviewer_model"],
                "executor_digest": model_digests[settings["executor_model"]],
                "reviewer_digest": model_digests[settings["reviewer_model"]],
                "workflow_hash": workflow_hash,
                "template_pool_hash": template_hash,
                "source_checkpoint": settings["source_checkpoint"],
            }
    return slots


def _validate_evaluation_identity(
    row: dict[str, Any],
    slot: dict[str, Any],
) -> None:
    mismatches = {
        key: {"ledger": row.get(key), "expected": value}
        for key, value in slot.items()
        if row.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Evaluation ledger identity does not match its deterministic "
            f"candidate/target slot: {mismatches}"
        )


def _validate_recoverable_candidate_error(row: dict[str, Any]) -> None:
    if row.get("status") == "ok":
        raise ValueError("A completed candidate does not require recovery.")
    error = str(row.get("error") or "")
    if row.get("error_type") != "ValidationError" or "planned_steps" not in error:
        raise RuntimeError(
            "Targeted recovery is restricted to the observed malformed planner "
            "output that omitted required planned_steps. Refusing candidate "
            f"{row.get('candidate_id')}."
        )
    if row.get("candidate_id") != row.get("run_hash"):
        raise RuntimeError("Candidate ID and logical run hash do not match.")


def _finish_reason(response: ModelResponse | None) -> Any:
    if response is None:
        return None
    return response.metadata.get("finish_reason") or response.metadata.get(
        "done_reason"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover only malformed planned_steps candidates in an existing "
            "sharded LegalFlux DPO collection."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/legal_flux.cluster.yaml",
    )
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--fallback-seed-stride", type=int, default=None)
    parser.add_argument("--max-recovery-attempts", type=int, default=1)
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Regenerate candidates but do not fill their missing X_sim evaluations.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = recover_dpo_candidates(
        load_config(args.config),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        case_limit=args.case_limit,
        fallback_seed_stride=args.fallback_seed_stride,
        max_recovery_attempts=args.max_recovery_attempts,
        evaluate_missing=not args.skip_evaluation,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
