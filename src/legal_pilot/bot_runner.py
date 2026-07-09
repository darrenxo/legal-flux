from __future__ import annotations

import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bot import (
    CONDITION_SPECS,
    TemplateBuffer,
    build_bot_plan,
    generic_template,
    raw_case_query,
    seed_templates,
    should_update_buffer,
)
from .bot_freeze import assert_bot_frozen, bot_run_hash, bot_workflow_hash
from .clients import OllamaClient, OllamaResponseError
from .config import resolve_path
from .embeddings import SimilarityBackend
from .frontier_profiles import load_frontier_profiles
from .io_utils import (
    canonical_json,
    latest_by_run_hash,
    read_jsonl,
    sha256_text,
)
from .models import (
    BufferUpdateEvent,
    DirectAnalysis,
    DistilledLegalProblem,
    FinalAnalysis,
    FrontierLegalProblem,
    LegalThoughtTemplate,
    NormalizedCase,
)
from .prompting import render_prompt
from .runner import (
    _normalize_direct_payload,
    _normalize_final_analysis_payload,
    _response_trace,
    load_cases,
)
from .scoring import answers_exactly_match
from .semantic_setup import build_similarity_backend, resolve_project_file


def run_bot_generation(
    config: dict[str, Any],
    *,
    smoke: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    cases = load_cases(config)
    plan = build_bot_plan(cases, config, smoke=smoke)
    run_dir = (
        resolve_path(config, "runs_dir")
        / ("smoke" if smoke else config["project"]["run_name"])
    )
    if dry_run:
        return {
            "jobs": len(plan),
            "adaptation_jobs": sum(
                item.phase == "adaptation" for item in plan
            ),
            "holdout_jobs": sum(item.phase == "holdout" for item in plan),
            "conditions": len({item.condition for item in plan}),
            "run_dir": str(run_dir),
            "dry_run": True,
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    client = OllamaClient(
        config["model"]["base_url"], config["model"]["timeout_seconds"]
    )
    model_info = client.model_info(config["model"]["name"])
    if not model_info:
        client.close()
        raise RuntimeError(
            f"Model {config['model']['name']!r} is not installed in Ollama."
        )
    digest = model_info.get("digest", "unknown")
    similarity_backend = build_similarity_backend(config)
    embedding_info = (
        similarity_backend.model_info()
        if hasattr(similarity_backend, "model_info")
        else None
    )
    if (
        config["bot"].get("retrieval_backend") == "ollama_embedding"
        and not embedding_info
    ):
        client.close()
        similarity_backend.close()
        raise RuntimeError(
            f"Embedding model {config['bot']['embedding_model']!r} "
            "is not installed in Ollama."
        )
    embedding_digest = (
        embedding_info.get("digest", "unknown")
        if embedding_info
        else "builtin-tfidf"
    )
    frontier_conditions = {
        item.condition
        for item in plan
        if CONDITION_SPECS[item.condition].profile_source == "frontier"
    }
    frontier_profiles: dict[str, FrontierLegalProblem] = {}
    if frontier_conditions:
        required_ids = {
            item.case.case_id
            for item in plan
            if item.condition in frontier_conditions
        }
        frontier_profiles = load_frontier_profiles(
            resolve_project_file(config["bot"]["frontier_profiles_file"]),
            required_case_ids=required_ids,
            valid_fact_ids_by_case={
                item.case.case_id: set(item.case.facts)
                for item in plan
                if item.condition in frontier_conditions
            },
        )
    workflow_hash = bot_workflow_hash(config)
    if not smoke:
        assert_bot_frozen(
            config,
            model_digest=digest,
            embedding_digest=embedding_digest,
            workflow_hash=workflow_hash,
        )
    generation_path = run_dir / "generations.jsonl"
    existing = latest_by_run_hash(read_jsonl(generation_path))
    completed = {
        row["run_hash"]: row for row in existing if row.get("run_hash")
    }
    planned = [
        {
            "run_hash": _item_run_hash(
                config,
                item,
                digest=digest,
                embedding_digest=embedding_digest,
                workflow_hash=workflow_hash,
            ),
            "case_id": item.case.case_id,
            "condition": item.condition,
            "phase": item.phase,
            "stream_index": item.stream_index,
        }
        for item in plan
    ]
    (run_dir / "run_plan.json").write_text(
        json.dumps(
            {
                "model_digest": digest,
                "embedding_model": config["bot"].get("embedding_model"),
                "embedding_digest": embedding_digest,
                "workflow_hash": workflow_hash,
                "job_count": len(planned),
                "jobs": planned,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    completed_count = 0
    skipped = 0
    errors = 0
    try:
        for condition in dict.fromkeys(item.condition for item in plan):
            condition_records = [
                row
                for row in existing
                if row.get("condition") == condition
            ]
            buffer = restore_condition_buffer(
                condition,
                condition_records,
                workflow_hash=workflow_hash,
                model_digest=digest,
                embedding_digest=embedding_digest,
                similarity_backend=similarity_backend,
            )
            for item in [
                candidate
                for candidate in plan
                if candidate.condition == condition
            ]:
                run_hash = _item_run_hash(
                    config,
                    item,
                    digest=digest,
                    embedding_digest=embedding_digest,
                    workflow_hash=workflow_hash,
                )
                if run_hash in completed:
                    skipped += 1
                    continue
                record = _run_bot_item(
                    client,
                    config,
                    item.case,
                    condition=condition,
                    phase=item.phase,
                    stream_index=item.stream_index,
                    buffer=buffer,
                    run_hash=run_hash,
                    model_digest=digest,
                    embedding_digest=embedding_digest,
                    workflow_hash=workflow_hash,
                    frontier_profile=frontier_profiles.get(
                        item.case.case_id
                    ),
                )
                _append_record(generation_path, record)
                existing.append(record)
                completed[run_hash] = record
                completed_count += 1
                if record["status"] != "ok":
                    errors += 1
                _write_buffer_snapshot(run_dir, condition, buffer)
    finally:
        client.close()
        close = getattr(similarity_backend, "close", None)
        if close:
            close()
    return {
        "jobs": len(plan),
        "completed": completed_count,
        "skipped": skipped,
        "errors": errors,
        "run_dir": str(run_dir),
        "model_digest": digest,
        "embedding_digest": embedding_digest,
        "workflow_hash": workflow_hash,
    }


def restore_condition_buffer(
    condition: str,
    records: list[dict[str, Any]],
    *,
    workflow_hash: str | None = None,
    model_digest: str | None = None,
    embedding_digest: str | None = None,
    similarity_backend: SimilarityBackend | None = None,
) -> TemplateBuffer:
    spec = CONDITION_SPECS[condition]
    seeds = seed_templates() if spec.use_legal_seeds else []
    events = []
    for row in sorted(records, key=lambda item: item.get("stream_index", -1)):
        if (
            workflow_hash is not None
            and row.get("workflow_hash") != workflow_hash
        ):
            continue
        if (
            model_digest is not None
            and row.get("model_digest") != model_digest
        ):
            continue
        if (
            embedding_digest is not None
            and row.get("embedding_digest") != embedding_digest
        ):
            continue
        payload = row.get("buffer_update")
        if row.get("status") == "ok" and isinstance(payload, dict):
            events.append(BufferUpdateEvent.model_validate(payload))
    return TemplateBuffer.replay(
        seeds,
        events,
        similarity_backend=similarity_backend,
    )


def _run_bot_item(
    client: OllamaClient,
    config: dict[str, Any],
    case: NormalizedCase,
    *,
    condition: str,
    phase: str,
    stream_index: int,
    buffer: TemplateBuffer,
    run_hash: str,
    model_digest: str,
    embedding_digest: str,
    workflow_hash: str,
    frontier_profile: FrontierLegalProblem | None,
) -> dict[str, Any]:
    base = {
        "run_hash": run_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": case.dataset,
        "case_id": case.case_id,
        "variant_id": case.variant_id,
        "condition": condition,
        "phase": phase,
        "stream_index": stream_index,
        "model_name": config["model"]["name"],
        "model_digest": model_digest,
        "retrieval_backend": config["bot"].get(
            "retrieval_backend", "tfidf"
        ),
        "embedding_model": config["bot"].get("embedding_model"),
        "embedding_digest": embedding_digest,
        "workflow_hash": workflow_hash,
        "seed": config["model"]["seed"],
        "decoding": {
            "temperature": config["model"]["temperature"],
            "context_length": config["model"]["context_length"],
        },
        "gold_answer": case.gold_answer,
        "metadata": case.metadata,
    }
    try:
        analysis, trace, event = _execute_bot_case(
            client,
            config,
            case,
            condition=condition,
            phase=phase,
            buffer=buffer,
            frontier_profile=frontier_profile,
        )
        return {
            **base,
            "status": "ok",
            "prompt_hash": trace["prompt_hash"],
            "prompt_hashes": trace["prompt_hashes"],
            "raw_response": trace["raw_response"],
            "parsed_json": analysis.model_dump(mode="json"),
            "distilled_problem": trace["distilled_problem"],
            "retrieval": trace["retrieval"],
            "buffer_hash_before": trace["buffer_hash_before"],
            "buffer_hash_after": trace["buffer_hash_after"],
            "buffer_size_before": trace["buffer_size_before"],
            "buffer_size_after": trace["buffer_size_after"],
            "buffer_update": event.model_dump(mode="json"),
            "elapsed_seconds": trace["elapsed_seconds"],
            "prompt_tokens": trace["prompt_tokens"],
            "output_tokens": trace["output_tokens"],
            "schema_errors": trace["schema_errors"],
            "repair_actions": trace["repair_actions"],
            "calls": trace["calls"],
        }
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "prompt_hash": None,
            "prompt_hashes": {},
            "raw_response": (
                exc.raw_text if isinstance(exc, OllamaResponseError) else None
            ),
            "parsed_json": None,
            "distilled_problem": None,
            "retrieval": None,
            "buffer_hash_before": _buffer_hash(buffer),
            "buffer_hash_after": _buffer_hash(buffer),
            "buffer_size_before": len(buffer.templates),
            "buffer_size_after": len(buffer.templates),
            "buffer_update": None,
            "elapsed_seconds": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "schema_errors": [str(exc)],
            "repair_actions": [],
            "calls": 0,
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }


def _execute_bot_case(
    client: Any,
    config: dict[str, Any],
    case: NormalizedCase,
    *,
    condition: str,
    phase: str,
    buffer: TemplateBuffer,
    frontier_profile: FrontierLegalProblem | None = None,
) -> tuple[FinalAnalysis, dict[str, Any], BufferUpdateEvent]:
    spec = CONDITION_SPECS[condition]
    if condition == "direct":
        return _execute_direct(client, config, case, buffer)
    common = _common_generation_settings(config)
    raw_parts: list[str] = []
    prompt_hashes: dict[str, str] = {}
    elapsed = 0.0
    prompt_tokens = 0
    output_tokens = 0
    schema_errors: list[str] = []
    repair_actions: list[str] = []
    calls = 0
    before_hash = _buffer_hash(buffer)
    before_size = len(buffer.templates)

    if spec.profile_source == "frontier":
        if frontier_profile is None:
            raise ValueError(
                f"Condition {condition} requires a frontier profile."
            )
        distilled = frontier_profile
        problem_profile = frontier_profile.model_dump(
            mode="json", exclude={"case_id"}
        )
        query = frontier_profile.retrieval_summary
    elif spec.use_distiller:
        distill_prompt, distill_hash = render_prompt(
            config,
            "bot/distill",
            case,
            lawsuit_type=case.metadata.get("lawsuit_type", "Not specified."),
        )
        distill_response = client.generate(
            prompt=distill_prompt,
            schema=_load_bot_schema(config, "distilled_legal_problem.json"),
            max_tokens=config["model"]["distill_max_tokens"],
            **common,
        )
        distilled = DistilledLegalProblem.model_validate(
            distill_response.parsed
        )
        problem_profile: Any = distilled.model_dump(mode="json")
        query = distilled.retrieval_query
        raw_parts.append(distill_response.raw_text)
        prompt_hashes["distill"] = distill_hash
        elapsed += distill_response.elapsed_seconds
        prompt_tokens += distill_response.prompt_tokens or 0
        output_tokens += distill_response.output_tokens or 0
        calls += 1
    else:
        distilled = None
        query = raw_case_query(case)
        problem_profile = {
            "distiller_ablation": True,
            "raw_retrieval_query": query,
        }

    if spec.use_buffer:
        retrieval = buffer.retrieve(
            query, threshold=config["bot"]["retrieval_threshold"]
        )
    else:
        retrieval = generic_template_retrieval()
    analysis_prompt, analysis_hash = render_prompt(
        config,
        "bot/analyze",
        case,
        problem_profile=problem_profile,
        thought_template=retrieval.template.model_dump(mode="json"),
    )
    analysis_response = client.generate(
        prompt=analysis_prompt,
        schema=_load_schema(resolve_path(config, "schemas_dir") / "final_analysis.json"),
        max_tokens=config["model"]["analysis_max_tokens"],
        **common,
    )
    normalized, normalization_repairs = _normalize_final_analysis_payload(
        analysis_response.parsed
    )
    analysis = FinalAnalysis.model_validate(normalized)
    raw_parts.append(analysis_response.raw_text)
    prompt_hashes["analysis"] = analysis_hash
    elapsed += analysis_response.elapsed_seconds
    prompt_tokens += analysis_response.prompt_tokens or 0
    output_tokens += analysis_response.output_tokens or 0
    calls += 1
    repair_actions.extend(normalization_repairs)

    answer_correct = answers_exactly_match(
        analysis.final_decision, case.gold_answer
    )
    if config["bot"].get("candidate_policy") == "every_correct":
        update = (
            phase == "adaptation"
            and answer_correct
            and spec.use_manager
        )
    else:
        update = should_update_buffer(
            phase=phase,
            answer_correct=answer_correct,
            manager_enabled=spec.use_manager,
            used_fallback=retrieval.used_fallback,
            similarity=retrieval.similarity,
            novelty_threshold=config["bot"]["novelty_threshold"],
        )
    if update:
        template_prompt, template_hash = render_prompt(
            config,
            "bot/template_distill",
            case,
            problem_profile=problem_profile,
            thought_template=retrieval.template.model_dump(mode="json"),
            generated_output=analysis.model_dump(mode="json"),
        )
        template_response = client.generate(
            prompt=template_prompt,
            schema=_load_bot_schema(config, "legal_thought_template.json"),
            max_tokens=config["model"]["template_max_tokens"],
            **common,
        )
        candidate = LegalThoughtTemplate.model_validate(
            template_response.parsed
        )
        candidate = sanitize_template_candidate(candidate, case)
        if config["bot"].get("update_strategy") == "append_only":
            event = buffer.apply_candidate_append_only(
                candidate,
                source_case_id=case.case_id,
                novelty_threshold=config["bot"]["novelty_threshold"],
            )
        else:
            event = buffer.apply_candidate(
                candidate,
                source_case_id=case.case_id,
                merge_threshold=config["bot"]["merge_threshold"],
            )
        raw_parts.append(template_response.raw_text)
        prompt_hashes["template_distill"] = template_hash
        elapsed += template_response.elapsed_seconds
        prompt_tokens += template_response.prompt_tokens or 0
        output_tokens += template_response.output_tokens or 0
        calls += 1
    else:
        event = BufferUpdateEvent(
            action="reject",
            source_case_id=case.case_id,
            target_template_id=None,
            template=None,
            rationale=_no_update_reason(
                phase=phase,
                answer_correct=answer_correct,
                manager_enabled=spec.use_manager,
                similarity=retrieval.similarity,
                novelty_threshold=config["bot"]["novelty_threshold"],
            ),
        )

    trace = {
        "prompt_hash": sha256_text(canonical_json(prompt_hashes)),
        "prompt_hashes": prompt_hashes,
        "raw_response": "\n---CALL---\n".join(raw_parts),
        "distilled_problem": (
            distilled.model_dump(mode="json") if distilled else None
        ),
        "retrieval": retrieval.model_dump(mode="json"),
        "buffer_hash_before": before_hash,
        "buffer_hash_after": _buffer_hash(buffer),
        "buffer_size_before": before_size,
        "buffer_size_after": len(buffer.templates),
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "schema_errors": schema_errors,
        "repair_actions": repair_actions,
        "calls": calls,
    }
    return analysis, trace, event


def _execute_direct(
    client: Any,
    config: dict[str, Any],
    case: NormalizedCase,
    buffer: TemplateBuffer,
) -> tuple[FinalAnalysis, dict[str, Any], BufferUpdateEvent]:
    prompt, prompt_hash = render_prompt(config, "direct", case)
    response = client.generate(
        prompt=prompt,
        schema=_load_schema(
            resolve_path(config, "schemas_dir") / "direct_analysis.json"
        ),
        max_tokens=config["model"]["analysis_max_tokens"],
        **_common_generation_settings(config),
    )
    normalized, repairs = _normalize_direct_payload(response.parsed)
    direct = DirectAnalysis.model_validate(normalized)
    analysis = FinalAnalysis(
        issue_conclusions=[],
        final_decision=direct.final_decision,
        final_rationale=direct.final_rationale,
    )
    trace = _response_trace(response)
    trace.update(
        {
            "prompt_hash": prompt_hash,
            "prompt_hashes": {"analysis": prompt_hash},
            "distilled_problem": None,
            "retrieval": None,
            "buffer_hash_before": _buffer_hash(buffer),
            "buffer_hash_after": _buffer_hash(buffer),
            "buffer_size_before": len(buffer.templates),
            "buffer_size_after": len(buffer.templates),
        }
    )
    trace["repair_actions"].extend(repairs)
    event = BufferUpdateEvent(
        action="reject",
        source_case_id=case.case_id,
        target_template_id=None,
        template=None,
        rationale="Direct baseline has no buffer manager.",
    )
    return analysis, trace, event


def sanitize_template_candidate(
    candidate: LegalThoughtTemplate, case: NormalizedCase
) -> LegalThoughtTemplate:
    forbidden = [
        case.case_id,
        *case.parties,
        *case.facts.keys(),
    ]

    def clean(value: str) -> str:
        result = value
        for token in forbidden:
            if token:
                result = re.sub(
                    re.escape(token),
                    "party" if token in case.parties else "supplied fact",
                    result,
                    flags=re.IGNORECASE,
                )
        result = re.sub(
            r"(?<!\w)(?:HK\$|US\$|\$|£|€|¥)?\s*"
            r"\d[\d,]*(?:\.\d+)?%?(?!\w)",
            "case-specific value",
            result,
            flags=re.IGNORECASE,
        )
        result = re.sub(r"\b(?:support|reject)\b", "resolve", result, flags=re.I)
        return " ".join(result.split())

    cleaned = candidate.model_copy(
        update={
            "name": clean(candidate.name),
            "description": clean(candidate.description),
            "applicability_cues": [
                clean(value) for value in candidate.applicability_cues
            ],
            "reasoning_steps": [
                clean(value) for value in candidate.reasoning_steps
            ],
            "required_checks": [
                clean(value) for value in candidate.required_checks
            ],
            "contraindications": [
                clean(value) for value in candidate.contraindications
            ],
            "provenance_case_ids": [],
            "version": 1,
        }
    )
    if len(cleaned.reasoning_steps) < 2 or not cleaned.required_checks:
        raise ValueError("Candidate template is too thin to enter the buffer.")
    return cleaned


def generic_template_retrieval():
    from .models import TemplateRetrieval

    return TemplateRetrieval(
        template=generic_template(),
        similarity=0.0,
        used_fallback=True,
        best_candidate_template_id=None,
    )


def _item_run_hash(
    config: dict[str, Any],
    item: Any,
    *,
    digest: str,
    embedding_digest: str = "builtin-tfidf",
    workflow_hash: str,
) -> str:
    return bot_run_hash(
        case_id=item.case.case_id,
        condition=item.condition,
        phase=item.phase,
        stream_index=item.stream_index,
        model_digest=digest,
        embedding_digest=embedding_digest,
        workflow_hash=workflow_hash,
        seed=config["model"]["seed"],
    )


def _common_generation_settings(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config["model"]["name"],
        "temperature": config["model"]["temperature"],
        "seed": config["model"]["seed"],
        "context_length": config["model"]["context_length"],
    }


def _load_bot_schema(config: dict[str, Any], name: str) -> dict[str, Any]:
    return _load_schema(resolve_path(config, "schemas_dir") / "bot" / name)


def _load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _buffer_hash(buffer: TemplateBuffer) -> str:
    return sha256_text(canonical_json(buffer.model_dump()))


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _write_buffer_snapshot(
    run_dir: Path, condition: str, buffer: TemplateBuffer
) -> None:
    directory = run_dir / "buffers"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{condition}.json"
    path.write_text(
        json.dumps(buffer.model_dump(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _no_update_reason(
    *,
    phase: str,
    answer_correct: bool,
    manager_enabled: bool,
    similarity: float,
    novelty_threshold: float,
) -> str:
    if phase != "adaptation":
        return "Holdout phase freezes the buffer before evaluation."
    if not manager_enabled:
        return "This ablation disables the buffer manager."
    if not answer_correct:
        return "Prediction was incorrect, so no template was learned."
    if similarity >= novelty_threshold:
        return "An existing template was sufficiently similar."
    return "Update gate rejected the candidate."
