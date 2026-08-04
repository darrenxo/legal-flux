from __future__ import annotations

import json
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from .clients import GenerationClient
from .config import resolve_path
from .io_utils import read_jsonl
from .models import DirectAnalysis, FinalAnalysis, NormalizedCase
from .prompting import render_prompt


def load_cases(config: dict[str, Any]) -> list[NormalizedCase]:
    path = resolve_path(config, "processed_dir") / "cases.jsonl"
    if not path.exists():
        raise FileNotFoundError("Prepared LegalFlux cases not found. Run flux-prepare first.")
    return [NormalizedCase.model_validate(row) for row in read_jsonl(path)]


def _preview_prompt(
    config: dict[str, Any], case: NormalizedCase, condition: str
) -> tuple[str, str]:
    if condition == "direct":
        return render_prompt(config, "direct", case)
    if condition == "structured":
        return render_prompt(config, "structured", case)
    raise ValueError(f"Unsupported baseline condition: {condition}")


def _execute_condition(
    client: GenerationClient,
    config: dict[str, Any],
    case: NormalizedCase,
    condition: str,
    temperature: float,
    seed: int,
) -> tuple[FinalAnalysis, dict[str, Any]]:
    common = {
        "model": config["model"]["name"],
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
        normalized, repairs = _normalize_direct_payload(response.parsed)
        direct = DirectAnalysis.model_validate(normalized)
        trace = _response_trace(response)
        _add_normalization_repairs(trace, repairs)
        return (
            FinalAnalysis(
                issue_conclusions=[],
                final_decision=direct.final_decision,
                final_rationale=direct.final_rationale,
            ),
            trace,
        )
    if condition == "structured":
        prompt, _ = render_prompt(config, "structured", case)
        response = client.generate(
            prompt=prompt,
            schema=_load_schema(_final_analysis_schema_path(config)),
            max_tokens=config["model"]["analysis_max_tokens"],
            **common,
        )
        normalized, repairs = _normalize_final_analysis_payload(response.parsed)
        trace = _response_trace(response)
        _add_normalization_repairs(trace, repairs)
        return FinalAnalysis.model_validate(normalized), trace
    raise ValueError(f"Unsupported baseline condition: {condition}")


def _response_trace(response: Any) -> dict[str, Any]:
    repaired = bool(response.metadata.get("json_repair_applied"))
    finish_reason = response.metadata.get("finish_reason") or response.metadata.get(
        "done_reason"
    )
    repair_actions = ["deterministic_json_repair"] if repaired else []
    if finish_reason == "length":
        repair_actions.append("generation_length_limit_reached")
    return {
        "raw_response": response.raw_text,
        "elapsed_seconds": response.elapsed_seconds,
        "prompt_tokens": response.prompt_tokens,
        "output_tokens": response.output_tokens,
        "finish_reason": finish_reason,
        "schema_errors": ["malformed_json_repaired"] if repaired else [],
        "repair_actions": repair_actions,
        "calls": 1,
    }


def _add_normalization_repairs(trace: dict[str, Any], repairs: list[str]) -> None:
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
    actions.extend(_repair_mapping_keys(repaired, {"final_decision", "final_rationale"}))
    actions.extend(_drop_obsolete_task_answer(repaired))
    if "final_decision" in repaired and "final_rationale" not in repaired:
        repaired["final_rationale"] = "No rationale supplied."
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
    accepted_keys = {
        "irac_reasoning",
        "issue_conclusions",
        "final_decision",
        "final_rationale",
    }
    actions.extend(
        _remove_schema_metadata_when_answer_present(repaired, accepted_keys)
    )
    actions.extend(_repair_mapping_keys(repaired, accepted_keys))
    if "irac_reasoning" in repaired:
        if repaired["irac_reasoning"] is None:
            repaired["irac_reasoning"] = ""
            actions.append("irac_reasoning_null_filled")
        elif not isinstance(repaired["irac_reasoning"], str):
            repaired["irac_reasoning"] = str(repaired["irac_reasoning"])
            actions.append("irac_reasoning_coerced_to_string")
        actions.extend(_drop_obsolete_task_answer(repaired))
        actions.extend(
            _fold_extra_fields_into_text(
                repaired,
                allowed={"irac_reasoning", "final_decision"},
                text_key="irac_reasoning",
                action_prefix="structured_analysis",
            )
        )
        return repaired, actions
    conclusions = repaired.get("issue_conclusions", [])
    if isinstance(conclusions, dict):
        conclusions = [conclusions]
        repaired["issue_conclusions"] = conclusions
        actions.append("issue_conclusions_wrapped_as_array")
    if not isinstance(conclusions, list):
        repaired["issue_conclusions"] = []
        actions.append("issue_conclusions_invalid_filled")
        conclusions = []
    issue_keys = {
        "issue_id",
        "conclusion",
        "supporting_fact_ids",
        "opposing_fact_ids",
        "explanation",
    }
    valid_issue_conclusions = {"satisfied", "not_satisfied", "defeated", "unresolved"}
    promoted_decisions: list[str] = []
    promoted_rationales: list[str] = []
    normalized_conclusions: list[dict[str, Any]] = []
    for index, item in enumerate(conclusions, start=1):
        if not isinstance(item, dict):
            actions.append("invalid_issue_conclusion_removed")
            continue
        if (
            "issue_id" not in item
            and any(key in item for key in ("final_decision", "final_rationale"))
        ):
            if isinstance(item.get("final_decision"), str):
                promoted_decisions.append(item["final_decision"])
            if isinstance(item.get("final_rationale"), str):
                promoted_rationales.append(item["final_rationale"])
            actions.append("nested_final_analysis_promoted")
            continue
        actions.extend(_repair_mapping_keys(item, issue_keys))
        if not isinstance(item.get("issue_id"), str) or not item.get("issue_id"):
            item["issue_id"] = f"I{index}"
            actions.append("issue_id_missing_filled")
        conclusion = item.get("conclusion")
        if not isinstance(conclusion, str) or conclusion not in valid_issue_conclusions:
            item["conclusion"] = "unresolved"
            actions.append("issue_conclusion_invalid_filled")
        for fact_key in ("supporting_fact_ids", "opposing_fact_ids"):
            values = item.get(fact_key)
            if values is None:
                item[fact_key] = []
                actions.append(f"{fact_key}_missing_filled")
            elif isinstance(values, str):
                item[fact_key] = [values] if values.strip() else []
                actions.append(f"{fact_key}_wrapped_as_array")
            elif isinstance(values, list):
                item[fact_key] = [
                    value.get("fact_id", value.get("id"))
                    if isinstance(value, dict)
                    else value
                    for value in values
                    if value not in (None, "")
                ]
            else:
                item[fact_key] = []
                actions.append(f"{fact_key}_invalid_filled")
        if not isinstance(item.get("explanation"), str) or not item.get("explanation"):
            item["explanation"] = "No issue-level explanation supplied."
            actions.append("issue_explanation_missing_filled")
        actions.extend(
            _fold_extra_fields_into_text(
                item,
                allowed=issue_keys,
                text_key="explanation",
                action_prefix="issue_conclusion",
            )
        )
        normalized_conclusions.append(item)
    repaired["issue_conclusions"] = normalized_conclusions
    if "final_decision" not in repaired and promoted_decisions:
        distinct = list(dict.fromkeys(promoted_decisions))
        if len(distinct) == 1:
            repaired["final_decision"] = distinct[0]
            actions.append("nested_final_decision_promoted")
    if "final_rationale" not in repaired and promoted_rationales:
        repaired["final_rationale"] = " ".join(dict.fromkeys(promoted_rationales))
        actions.append("nested_final_rationales_combined")
    actions.extend(_drop_obsolete_task_answer(repaired))
    if "final_decision" in repaired and "final_rationale" not in repaired:
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
    extra = {key: mapping.pop(key) for key in list(mapping) if key not in allowed}
    if not extra:
        return []
    non_empty_extra = {
        key: value for key, value in extra.items() if value not in (None, "", [], {})
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


def _repair_mapping_keys(mapping: dict[str, Any], allowed: set[str]) -> list[str]:
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
