from __future__ import annotations

import json
import re
import traceback
from difflib import get_close_matches
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .clients import OllamaClient, OllamaResponseError
from .config import resolve_path
from .io_utils import canonical_json, read_jsonl, sha256_text
from .jobs import build_jobs
from .ledger import JsonlLedger, make_run_hash
from .models import CaseState, DirectAnalysis, FinalAnalysis, IssueConclusion, NormalizedCase
from .prompting import render_prompt
from .validation import validate_case_state


def load_cases(config: dict[str, Any]) -> list[NormalizedCase]:
    path = resolve_path(config, "processed_dir") / "cases.jsonl"
    if not path.exists():
        raise FileNotFoundError("Prepared cases not found. Run prepare first.")
    return [NormalizedCase.model_validate(row) for row in read_jsonl(path)]


def run_generation(
    config: dict[str, Any],
    *,
    smoke: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    cases = load_cases(config)
    jobs = build_jobs(cases, config, smoke=smoke)
    run_dir = resolve_path(config, "runs_dir") / ("smoke" if smoke else config["project"]["run_name"])
    ledger = JsonlLedger(run_dir / "generations.jsonl")

    if dry_run:
        return {"jobs": len(jobs), "run_dir": str(run_dir), "dry_run": True}

    client = OllamaClient(
        config["model"]["base_url"], config["model"]["timeout_seconds"]
    )
    model_info = client.model_info(config["model"]["name"])
    if not model_info:
        raise RuntimeError(
            f"Model {config['model']['name']!r} is not installed in Ollama."
        )
    digest = model_info.get("digest", "unknown")
    if not smoke:
        _assert_phase_two_frozen(config, digest)
    planned_jobs = [
        {
            "run_hash": _job_run_hash(config, job["case"], job, digest),
            "dataset": job["case"].dataset,
            "case_id": job["case"].case_id,
            "variant_id": job["case"].variant_id,
            "condition": job["condition"],
            "sample_index": job["sample_index"],
        }
        for job in jobs
    ]
    (run_dir / "run_plan.json").write_text(
        json.dumps(
            {
                "model_digest": digest,
                "job_count": len(planned_jobs),
                "jobs": planned_jobs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    completed = 0
    skipped = 0
    try:
        for job in jobs:
            case: NormalizedCase = job["case"]
            record = _run_one(client, config, case, job, digest, ledger)
            if record is None:
                skipped += 1
            else:
                completed += 1
    finally:
        client.close()
    return {
        "jobs": len(jobs),
        "completed": completed,
        "skipped": skipped,
        "run_dir": str(run_dir),
        "model_digest": digest,
    }


def _run_one(
    client: OllamaClient,
    config: dict[str, Any],
    case: NormalizedCase,
    job: dict[str, Any],
    model_digest: str,
    ledger: JsonlLedger,
) -> dict[str, Any] | None:
    condition = job["condition"]
    preview_prompt, prompt_hash = _preview_prompt(config, case, condition)
    run_hash = _job_run_hash(config, case, job, model_digest)
    if ledger.contains(run_hash):
        return None

    base = {
        "run_hash": run_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": case.dataset,
        "case_id": case.case_id,
        "variant_id": case.variant_id,
        "pair_id": case.pair_id,
        "perturbation_kind": case.perturbation_kind,
        "condition": condition,
        "prompt_hash": prompt_hash,
        "model_name": config["model"]["name"],
        "model_digest": model_digest,
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
        analysis, trace = _execute_condition(
            client, config, case, condition, job["temperature"], job["seed"]
        )
        record = {
            **base,
            "status": "ok",
            "raw_response": trace["raw_response"],
            "parsed_json": analysis.model_dump(mode="json"),
            "case_state": trace.get("case_state"),
            "elapsed_seconds": trace["elapsed_seconds"],
            "prompt_tokens": trace["prompt_tokens"],
            "output_tokens": trace["output_tokens"],
            "schema_errors": trace.get("schema_errors", []),
            "retries": trace.get("retries", 0),
            "repair_actions": trace.get("repair_actions", []),
            "calls": trace.get("calls", 1),
        }
    except Exception as exc:
        raw_response = exc.raw_text if isinstance(exc, OllamaResponseError) else None
        response_metadata = (
            {
                "done_reason": exc.payload.get("done_reason"),
                "thinking": exc.payload.get("message", {}).get("thinking"),
            }
            if isinstance(exc, OllamaResponseError)
            else None
        )
        record = {
            **base,
            "status": "error",
            "raw_response": raw_response,
            "parsed_json": None,
            "elapsed_seconds": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "schema_errors": [str(exc)],
            "retries": 0,
            "repair_actions": [],
            "calls": 0,
            "error_type": type(exc).__name__,
            "error_response_metadata": response_metadata,
            "traceback": traceback.format_exc(),
        }
    ledger.append(record)
    return record


def _job_run_hash(
    config: dict[str, Any],
    case: NormalizedCase,
    job: dict[str, Any],
    model_digest: str,
) -> str:
    _, prompt_hash = _preview_prompt(config, case, job["condition"])
    return make_run_hash(
        dataset=case.dataset,
        case_id=case.case_id,
        variant_id=case.variant_id,
        condition=job["condition"],
        prompt_hash=prompt_hash,
        model_digest=model_digest,
        seed=job["seed"],
        sample_index=job["sample_index"],
    )


def _preview_prompt(
    config: dict[str, Any], case: NormalizedCase, condition: str
) -> tuple[str, str]:
    if condition == "direct":
        return render_prompt(config, "direct", case)
    if condition in {"structured", "sampling_control"}:
        return render_prompt(config, "structured", case)
    state = (
        case.reference_state.model_dump(mode="json")
        if condition == "oracle" and case.reference_state
        else {"generated_in_first_call": True}
    )
    analysis_prompt, _ = render_prompt(
        config, "state_analysis", case, case_state=state
    )
    if condition == "oracle":
        return analysis_prompt, sha256_text(analysis_prompt)
    state_prompt, _ = render_prompt(config, "state", case)
    combined = state_prompt + "\n---STATE-ANALYSIS-TEMPLATE---\n" + analysis_prompt
    if condition == "validated":
        repair_prompt, _ = render_prompt(
            config,
            "state_repair",
            case,
            case_state={"generated_in_first_call": True},
            validation_errors=["deterministic_validation_errors"],
        )
        combined += "\n---STATE-REPAIR-TEMPLATE---\n" + repair_prompt
    return combined, sha256_text(combined)


def _execute_condition(
    client: OllamaClient,
    config: dict[str, Any],
    case: NormalizedCase,
    condition: str,
    temperature: float,
    seed: int,
) -> tuple[FinalAnalysis, dict[str, Any]]:
    schema_dir = resolve_path(config, "schemas_dir")
    model = config["model"]["name"]
    common = {
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "context_length": config["model"]["context_length"],
    }
    if condition == "direct":
        prompt, _ = render_prompt(config, "direct", case)
        response = client.generate(
            prompt=prompt,
            schema=_load_schema(_direct_analysis_schema_path(config)),
            max_tokens=config["model"]["analysis_max_tokens"],
            **common,
        )
        normalized_direct, normalization_repairs = _normalize_direct_payload(
            response.parsed
        )
        direct = DirectAnalysis.model_validate(normalized_direct)
        analysis = FinalAnalysis(
            issue_conclusions=[],
            final_decision=direct.final_decision,
            final_rationale=direct.final_rationale,
        )
        trace = _response_trace(response)
        _add_normalization_repairs(trace, normalization_repairs)
        return analysis, trace

    if condition in {"structured", "sampling_control"}:
        prompt, _ = render_prompt(config, "structured", case)
        response = client.generate(
            prompt=prompt,
            schema=_load_schema(_final_analysis_schema_path(config)),
            max_tokens=config["model"]["analysis_max_tokens"],
            **common,
        )
        normalized_analysis, normalization_repairs = (
            _normalize_final_analysis_payload(response.parsed)
        )
        trace = _response_trace(response)
        _add_normalization_repairs(trace, normalization_repairs)
        return FinalAnalysis.model_validate(normalized_analysis), trace

    repair_actions: list[str] = []
    validation_errors: list[str] = []
    calls = 1
    if condition == "oracle":
        if not case.reference_state:
            raise ValueError("Oracle condition requires reference_state.")
        state = case.reference_state
        state_raw = json.dumps(state.model_dump(mode="json"), ensure_ascii=False)
        state_prompt_tokens = 0
        state_output_tokens = 0
        state_elapsed = 0.0
    else:
        state_prompt, _ = render_prompt(config, "state", case)
        state_response = client.generate(
            prompt=state_prompt,
            schema=_load_schema(schema_dir / "case_state.json"),
            max_tokens=config["model"]["state_max_tokens"],
            **common,
        )
        recovered_state, envelope_repairs = _recover_case_state_schema_envelope(
            state_response.parsed, case
        )
        normalized_state, state_shape_repairs = _normalize_case_state_payload(
            recovered_state
        )
        state_shape_repairs = envelope_repairs + state_shape_repairs
        state = CaseState.model_validate(normalized_state)
        state_raw = state_response.raw_text
        state_prompt_tokens = state_response.prompt_tokens or 0
        state_output_tokens = state_response.output_tokens or 0
        state_elapsed = state_response.elapsed_seconds
        validation = validate_case_state(
            state,
            valid_fact_ids=set(case.facts),
            known_parties=set(case.parties),
        )
        validation_errors = state_shape_repairs + [
            f"{error.code}: {error.message}" for error in validation.errors
        ]
        repair_actions.extend(
            "deterministic_schema_key_repair"
            for _ in state_shape_repairs
        )
        if condition == "validated" and not validation.valid:
            calls += 1
            repair_prompt, _ = render_prompt(
                config,
                "state_repair",
                case,
                case_state=state.model_dump(mode="json"),
                validation_errors=validation_errors,
            )
            repaired_response = client.generate(
                prompt=repair_prompt,
                schema=_load_schema(schema_dir / "case_state.json"),
                max_tokens=config["model"]["state_max_tokens"],
                **common,
            )
            recovered_repair, repair_envelope_repairs = (
                _recover_case_state_schema_envelope(
                    repaired_response.parsed, case
                )
            )
            normalized_repair, repair_shape_repairs = _normalize_case_state_payload(
                recovered_repair
            )
            repair_shape_repairs = (
                repair_envelope_repairs + repair_shape_repairs
            )
            state = CaseState.model_validate(normalized_repair)
            repair_actions.append("one_state_repair_call")
            repair_actions.extend(
                "deterministic_schema_key_repair"
                for _ in repair_shape_repairs
            )
            state_raw += "\n---REPAIR---\n" + repaired_response.raw_text
            state_prompt_tokens += repaired_response.prompt_tokens or 0
            state_output_tokens += repaired_response.output_tokens or 0
            state_elapsed += repaired_response.elapsed_seconds
            repaired_validation = validate_case_state(
                state,
                valid_fact_ids=set(case.facts),
                known_parties=set(case.parties),
            )
            validation_errors = repair_shape_repairs + [
                f"{error.code}: {error.message}"
                for error in repaired_validation.errors
            ]
            if validation_errors:
                repair_actions.append("repair_left_validation_errors")

    analysis_prompt, _ = render_prompt(
        config, "state_analysis", case, case_state=state.model_dump(mode="json")
    )
    analysis_response = client.generate(
        prompt=analysis_prompt,
        schema=_load_schema(_final_analysis_schema_path(config)),
        max_tokens=config["model"]["analysis_max_tokens"],
        **common,
    )
    calls += 1
    normalized_analysis, analysis_repairs = _normalize_final_analysis_payload(
        analysis_response.parsed
    )
    analysis = FinalAnalysis.model_validate(normalized_analysis)
    trace = _response_trace(analysis_response)
    repair_actions.extend(analysis_repairs)
    trace.update(
        {
            "raw_response": state_raw + "\n---ANALYSIS---\n" + analysis_response.raw_text,
            "case_state": state.model_dump(mode="json"),
            "elapsed_seconds": state_elapsed + analysis_response.elapsed_seconds,
            "prompt_tokens": state_prompt_tokens
            + (analysis_response.prompt_tokens or 0),
            "output_tokens": state_output_tokens
            + (analysis_response.output_tokens or 0),
            "schema_errors": validation_errors,
            "repair_actions": repair_actions,
            "calls": calls,
        }
    )
    return analysis, trace


def _response_trace(response) -> dict[str, Any]:
    repaired = bool(response.metadata.get("json_repair_applied"))
    return {
        "raw_response": response.raw_text,
        "elapsed_seconds": response.elapsed_seconds,
        "prompt_tokens": response.prompt_tokens,
        "output_tokens": response.output_tokens,
        "schema_errors": ["malformed_json_repaired"] if repaired else [],
        "repair_actions": ["deterministic_json_repair"] if repaired else [],
        "calls": 1,
    }


def _add_normalization_repairs(
    trace: dict[str, Any], repairs: list[str]
) -> None:
    if not repairs:
        return
    trace["schema_errors"].extend(
        f"deterministic_output_normalization: {repair}" for repair in repairs
    )
    trace["repair_actions"].extend(repairs)


def _load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _direct_analysis_schema_path(config: dict[str, Any]) -> Path:
    name = (
        "direct_analysis_binary.json"
        if _force_binary_final_decision(config)
        else "direct_analysis.json"
    )
    return resolve_path(config, "schemas_dir") / name


def _final_analysis_schema_path(config: dict[str, Any]) -> Path:
    name = (
        "final_analysis_binary.json"
        if _force_binary_final_decision(config)
        else "final_analysis.json"
    )
    return resolve_path(config, "schemas_dir") / name


def _force_binary_final_decision(config: dict[str, Any]) -> bool:
    return bool(config.get("legal_flux", {}).get("force_binary_final_decision", False))


def _assert_phase_two_frozen(config: dict[str, Any], model_digest: str) -> None:
    manifest_path = resolve_path(config, "processed_dir") / "frozen_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Phase 2 is not frozen. Run freeze before generate.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model", {}).get("digest") != model_digest:
        raise RuntimeError("Ollama model digest differs from the frozen manifest.")
    for key, directory_name, pattern in (
        ("prompt_hashes", "prompts_dir", "*.txt"),
        ("schema_hashes", "schemas_dir", "*.json"),
    ):
        directory = resolve_path(config, directory_name)
        current = {
            path.name: sha256_text(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob(pattern))
        }
        if current != manifest.get(key):
            raise RuntimeError(
                f"{key} differ from the frozen manifest; explicitly rerun smoke "
                "and freeze before generating."
            )


def _normalize_direct_payload(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        return payload, []
    repaired = json.loads(json.dumps(payload))
    actions: list[str] = []
    actions.extend(
        _remove_schema_metadata_when_answer_present(
            repaired, {"final_decision", "final_rationale"}
        )
    )
    actions.extend(_repair_mapping_keys(
        repaired, {"final_decision", "final_rationale"}
    ))
    actions.extend(_drop_obsolete_task_answer(repaired))
    if "final_decision" in repaired and "final_rationale" not in repaired:
        repaired["final_rationale"] = "See issue conclusions."
        actions.append("missing_final_rationale_filled")
    if "final_decision" in repaired:
        actions.extend(
            _fold_extra_fields_into_text(
                repaired,
                allowed={"final_decision", "final_rationale"},
                text_key="final_rationale",
                action_prefix="direct",
            )
        )
    return repaired, actions


def _normalize_final_analysis_payload(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        return payload, []
    repaired = json.loads(json.dumps(payload))
    actions: list[str] = []
    actions.extend(
        _remove_schema_metadata_when_answer_present(
            repaired,
            {"issue_conclusions", "final_decision", "final_rationale"},
        )
    )
    actions.extend(_repair_mapping_keys(
        repaired,
        {
            "issue_conclusions",
            "final_decision",
            "final_rationale",
        },
    ))
    conclusions = repaired.get("issue_conclusions", [])
    if isinstance(conclusions, dict):
        conclusions = [conclusions]
        repaired["issue_conclusions"] = conclusions
        actions.append("issue_conclusions_wrapped_as_array")
    if isinstance(conclusions, list):
        flattened: list[Any] = []
        for item in conclusions:
            if not isinstance(item, dict):
                flattened.append(item)
                continue
            if any(
                key in item
                for key in (
                    "final_decision",
                    "final_rationale",
                    "task_answer",
                    "issue_conclusions",
                )
            ) and "issue_id" not in item:
                nested = item.get("issue_conclusions")
                if isinstance(nested, list):
                    flattened.extend(nested)
                for key in ("final_decision", "final_rationale"):
                    if key in item and key not in repaired:
                        repaired[key] = item[key]
                if "task_answer" in item:
                    actions.append("obsolete_task_answer_removed")
                actions.append("nested_final_analysis_flattened")
                continue
            flattened.append(item)
        repaired["issue_conclusions"] = flattened
        issue_keys = {
            "issue_id",
            "conclusion",
            "supporting_fact_ids",
            "opposing_fact_ids",
            "explanation",
        }
        issue_level_decisions: list[str] = []
        issue_level_rationales: list[str] = []
        valid_issue_conclusions = {
            "satisfied",
            "not_satisfied",
            "defeated",
            "unresolved",
        }
        for index, item in enumerate(flattened, start=1):
            if isinstance(item, dict):
                actions.extend(_repair_mapping_keys(item, issue_keys))
                if item.get("issue_id") is None:
                    item["issue_id"] = f"I{index}"
                    actions.append("issue_id_missing_filled")
                elif not isinstance(item.get("issue_id"), str):
                    item["issue_id"] = str(item["issue_id"])
                    actions.append("issue_id_coerced_to_string")
                conclusion = item.get("conclusion")
                if conclusion is None:
                    item["conclusion"] = "unresolved"
                    actions.append("issue_conclusion_missing_filled")
                elif not isinstance(conclusion, str):
                    item["conclusion"] = str(conclusion)
                    actions.append("issue_conclusion_coerced_to_string")
                elif conclusion not in valid_issue_conclusions:
                    item["conclusion"] = "unresolved"
                    actions.append("issue_conclusion_invalid_filled")
                explanation = item.get("explanation")
                if explanation is None:
                    item["explanation"] = "No issue-level explanation supplied."
                    actions.append("issue_explanation_missing_filled")
                elif not isinstance(explanation, str):
                    item["explanation"] = str(explanation)
                    actions.append("issue_explanation_coerced_to_string")
                for fact_id_key in ("supporting_fact_ids", "opposing_fact_ids"):
                    values = item.get(fact_id_key)
                    if values is None:
                        item[fact_id_key] = []
                        actions.append(f"{fact_id_key}_missing_filled")
                        continue
                    if isinstance(values, str):
                        item[fact_id_key] = [values]
                        actions.append(f"{fact_id_key}_wrapped_as_array")
                        continue
                    if not isinstance(values, list):
                        continue
                    normalized_values: list[Any] = []
                    for value in values:
                        if (
                            isinstance(value, dict)
                            and isinstance(value.get("fact_id"), str)
                        ):
                            normalized_values.append(value["fact_id"])
                            actions.append("fact_id_object_unwrapped")
                        else:
                            normalized_values.append(value)
                    item[fact_id_key] = normalized_values
                for key in list(item):
                    if "final_decision" in key:
                        value = item.pop(key)
                        if isinstance(value, str):
                            issue_level_decisions.append(value)
                    elif key == "final_rationale":
                        value = item.pop(key)
                        if isinstance(value, str) and value.strip():
                            issue_level_rationales.append(value.strip())
                actions.extend(
                    _fold_extra_fields_into_text(
                        item,
                        allowed=issue_keys,
                        text_key="explanation",
                        action_prefix="issue_conclusion",
                    )
                )
        if "final_decision" not in repaired and issue_level_decisions:
            distinct_decisions = list(dict.fromkeys(issue_level_decisions))
            if len(distinct_decisions) == 1:
                repaired["final_decision"] = distinct_decisions[0]
                actions.append("issue_level_final_decision_promoted")
        if "final_rationale" not in repaired and issue_level_rationales:
            distinct_rationales = list(dict.fromkeys(issue_level_rationales))
            repaired["final_rationale"] = " ".join(distinct_rationales)
            actions.append("issue_level_final_rationales_combined")
    actions.extend(_drop_obsolete_task_answer(repaired))
    if (
        "final_decision" in repaired
        and "issue_conclusions" in repaired
        and "final_rationale" not in repaired
    ):
        repaired["final_rationale"] = "See issue conclusions."
        actions.append("missing_final_rationale_filled")
    if "final_decision" in repaired or "issue_conclusions" in repaired:
        actions.extend(
            _fold_extra_fields_into_text(
                repaired,
                allowed={"issue_conclusions", "final_decision", "final_rationale"},
                text_key="final_rationale",
                action_prefix="final_analysis",
            )
        )
    return repaired, actions


def _normalize_case_state_payload(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        return payload, []
    repaired = json.loads(json.dumps(payload))
    actions: list[str] = []
    for key in ("$schema", "title"):
        if key in repaired:
            repaired.pop(key)
            actions.append(f"schema_metadata_removed: {key}")
    for key in list(repaired):
        match = re.fullmatch(
            r'requested_remedies(\[.*\]),"issues',
            key,
        )
        if not match:
            continue
        try:
            remedies = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        repaired["requested_remedies"] = remedies
        repaired["issues"] = repaired.pop(key)
        actions.append("concatenated_requested_remedies_issues_key_repaired")
    actions.extend(_repair_mapping_keys(
        repaired, {"claims", "requested_remedies", "issues"}
    ))
    for field_name in ("claims", "requested_remedies"):
        values = repaired.get(field_name)
        if not isinstance(values, list):
            continue
        normalized_values: list[Any] = []
        for value in values:
            if isinstance(value, dict) and isinstance(value.get("text"), str):
                normalized_values.append(value["text"])
                actions.append(f"{field_name}_text_object_unwrapped")
            else:
                normalized_values.append(value)
        repaired[field_name] = normalized_values
    if (
        "requested_remedies" not in repaired
        and isinstance(repaired.get("claims"), list)
        and repaired.get("claims")
    ):
        repaired["requested_remedies"] = list(repaired["claims"])
        actions.append("requested_remedies_copied_from_claims")
    issue_keys = {
        "issue_id",
        "issue",
        "rule_or_test",
        "burden_on",
        "elements",
        "defenses",
    }
    element_keys = {
        "element_id",
        "element",
        "supporting_fact_ids",
        "opposing_fact_ids",
        "missing_information",
        "status",
    }
    issues = repaired.get("issues", [])
    if isinstance(issues, list):
        required_issue_keys = {
            "issue_id",
            "issue",
            "rule_or_test",
            "burden_on",
            "elements",
        }
        valid_issues = [
            issue
            for issue in issues
            if isinstance(issue, dict)
            and required_issue_keys.issubset(issue)
        ]
        if valid_issues and len(valid_issues) != len(issues):
            removed_count = len(issues) - len(valid_issues)
            repaired["issues"] = valid_issues
            issues = valid_issues
            actions.extend(
                "orphan_issue_removed" for _ in range(removed_count)
            )
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        actions.extend(_repair_mapping_keys(issue, issue_keys))
        defenses = issue.get("defenses", [])
        if isinstance(defenses, str):
            issue["defenses"] = [defenses]
            actions.append("defenses_wrapped_as_array")
        elif isinstance(defenses, list):
            normalized_defenses: list[str] = []
            for defense in defenses:
                if isinstance(defense, str):
                    normalized_defenses.append(defense)
                elif isinstance(defense, dict):
                    value = defense.get("defense") or defense.get("description")
                    normalized_defenses.append(
                        str(value) if value is not None else canonical_json(defense)
                    )
                    actions.append("defense_object_serialized_to_string")
                else:
                    normalized_defenses.append(str(defense))
                    actions.append("defense_value_serialized_to_string")
            issue["defenses"] = normalized_defenses
        for element in issue.get("elements", []):
            if not isinstance(element, dict):
                continue
            actions.extend(_repair_mapping_keys(element, element_keys))
    return repaired, actions


def _recover_case_state_schema_envelope(
    payload: dict[str, Any] | None,
    case: NormalizedCase,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(payload, dict):
        return payload, []
    properties = payload.get("properties")
    if not isinstance(properties, dict):
        return payload, []
    issues = properties.get("issues")
    if not isinstance(issues, list):
        return payload, []
    claims = properties.get("claims")
    if not isinstance(claims, list):
        claims = [case.claim]
    remedies = properties.get("requested_remedies")
    if not isinstance(remedies, list):
        remedies = [case.requested_remedy or case.claim]
    return (
        {
            "claims": claims,
            "requested_remedies": remedies,
            "issues": issues,
        },
        ["case_state_schema_envelope_unwrapped"],
    )


def _remove_schema_metadata_when_answer_present(
    payload: dict[str, Any], answer_keys: set[str]
) -> list[str]:
    if not answer_keys.intersection(payload):
        return []
    actions: list[str] = []
    for key in (
        "$schema",
        "title",
        "type",
        "additionalProperties",
        "required",
        "properties",
    ):
        if key in payload:
            payload.pop(key)
            actions.append(f"schema_metadata_removed: {key}")
    return actions


def _repair_case_state_shape(
    payload: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    return _normalize_case_state_payload(payload)


def _drop_obsolete_task_answer(payload: dict[str, Any]) -> list[str]:
    if "task_answer" not in payload:
        return []
    payload.pop("task_answer")
    return ["obsolete_task_answer_removed"]


def _fold_extra_fields_into_text(
    mapping: dict[str, Any],
    *,
    allowed: set[str],
    text_key: str,
    action_prefix: str,
) -> list[str]:
    extra = {
        key: mapping.pop(key)
        for key in list(mapping)
        if key not in allowed
    }
    if not extra:
        return []
    non_empty_extra = {
        key: value
        for key, value in extra.items()
        if value not in (None, "", [], {})
    }
    if not non_empty_extra:
        return [f"{action_prefix}_extra_fields_removed"]
    extra_text = json.dumps(non_empty_extra, ensure_ascii=False, sort_keys=True)
    existing = mapping.get(text_key)
    if isinstance(existing, str) and existing.strip():
        mapping[text_key] = f"{existing}\nAdditional structured notes: {extra_text}"
    else:
        mapping[text_key] = extra_text
    return [f"{action_prefix}_extra_fields_folded_into_{text_key}"]


def _repair_mapping_keys(
    mapping: dict[str, Any], allowed: set[str]
) -> list[str]:
    actions: list[str] = []
    for key in list(mapping):
        if key in allowed:
            continue
        stripped = key.strip("[]`'\" .")
        target: str | None = None
        if stripped in allowed:
            target = stripped
        else:
            for candidate in allowed:
                if candidate in stripped:
                    target = candidate
                    break
        if target is None and stripped.endswith("_fact_ids"):
            if stripped.startswith("op"):
                target = "opposing_fact_ids"
            elif stripped.startswith("sup"):
                target = "supporting_fact_ids"
        if target is None:
            matches = get_close_matches(stripped, allowed, n=1, cutoff=0.86)
            target = matches[0] if matches else None
        if target and target not in mapping:
            mapping[target] = mapping.pop(key)
            actions.append(f"schema_key_typo: {key} -> {target}")
    return actions
