from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_path
from .io_utils import latest_by_run_hash, read_jsonl, sha256_text, write_jsonl
from .legal_flux import load_template_pool, template_pool_hash
from .legal_flux_runner import _template_tag_examples
from .models import LegalFluxAbstractPlan, LegalFluxTemplate, NormalizedCase
from .prompting import render_prompt
from .runner import load_cases


def export_template_structure_sft(config: dict[str, Any]) -> dict[str, Any]:
    templates = load_template_pool(config)
    output_dir = _training_dir(config)
    output_path = output_dir / "template_structure_sft.jsonl"
    rows = [
        _template_structure_sft_row(template)
        for template in templates
    ]
    write_jsonl(output_path, rows)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "template_structure_sft",
        "templates": len(templates),
        "output_path": str(output_path),
        "output_sha256": sha256_text(output_path.read_text(encoding="utf-8")),
        "template_pool_hash": template_pool_hash(templates),
        "notes": [
            "Input side contains template name and knowledge tags only.",
            "Assistant side reconstructs description, application scenario, and reasoning flow.",
            "This is the ReasonFlux-style template-structure learning objective; it is not trajectory SFT.",
        ],
    }
    manifest_path = output_dir / "template_structure_sft_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_path": str(manifest_path)}


def export_trajectory_dpo(
    config: dict[str, Any],
    *,
    phase: str = "planner_train",
) -> dict[str, Any]:
    normalized_phase = phase.replace("-", "_")
    if normalized_phase != "planner_train":
        raise ValueError("Trajectory DPO export is restricted to the planner_train split.")
    run_dir = resolve_path(config, "runs_dir") / normalized_phase
    scored_path = run_dir / "scored.jsonl"
    if not scored_path.exists():
        raise RuntimeError(
            "No planner-train scored generations found. Run "
            "`flux-generate --phase planner-train --samples N` and then "
            "`flux-score --phase planner-train` first."
        )
    rows = latest_by_run_hash(read_jsonl(scored_path))
    cases = {
        (case.dataset, case.case_id, case.variant_id): case
        for case in load_cases(config)
        if case.metadata.get("selection_split") == normalized_phase
    }
    templates = load_template_pool(config)
    template_examples = _template_tag_examples(config, templates)
    groups = _group_planner_train_rows(rows, cases)
    pairs = [
        pair
        for pair in (
            _dpo_pair_from_group(config, key, group, cases, template_examples)
            for key, group in groups.items()
        )
        if pair is not None
    ]
    output_dir = _training_dir(config)
    output_path = output_dir / "trajectory_dpo.jsonl"
    write_jsonl(output_path, pairs)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "trajectory_dpo",
        "phase": normalized_phase,
        "scored_path": str(scored_path),
        "candidate_rows": sum(len(group) for group in groups.values()),
        "case_groups": len(groups),
        "pairs": len(pairs),
        "output_path": str(output_path),
        "output_sha256": sha256_text(output_path.read_text(encoding="utf-8")),
        "template_pool_hash": template_pool_hash(templates),
        "reward": {
            "answer_correct": 1.0,
            "binary_prediction_valid": 0.1,
            "retrieval_success": 0.1,
            "schema_clean": 0.05,
            "step_efficiency": 0.05,
        },
        "notes": [
            "Chosen/rejected responses are planner trajectory JSON objects.",
            "Pairs are built only from repeated planner_train samples for the same case.",
            "Final-test rows are never read by this export.",
        ],
    }
    manifest_path = output_dir / "trajectory_dpo_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_path": str(manifest_path)}


def _training_dir(config: dict[str, Any]) -> Path:
    return resolve_path(config, "processed_dir") / "planner_training"


def _template_structure_sft_row(template: LegalFluxTemplate) -> dict[str, Any]:
    assistant_payload = {
        "description": template.description,
        "application_scenario": template.application_scenario,
        "reasoning_flow": template.reasoning_flow,
    }
    return {
        "id": f"template-structure-{template.template_id}",
        "task": "template_structure_sft",
        "template_id": template.template_id,
        "messages": [
            {
                "role": "system",
                "content": "You are learning the LegalFlux structured template library.",
            },
            {
                "role": "user",
                "content": (
                    "Given this LegalFlux template identifier, name, and knowledge "
                    "tags, return JSON with description, application_scenario, "
                    "and reasoning_flow.\n\n"
                    f"Template ID: {template.template_id}\n"
                    f"Template name: {template.template_name}\n"
                    "Knowledge tags: "
                    + ", ".join(template.knowledge_tags)
                ),
            },
            {
                "role": "assistant",
                "content": _json_dumps(assistant_payload),
            },
        ],
    }


def _group_planner_train_rows(
    rows: list[dict[str, Any]],
    cases: dict[tuple[str, str, str], NormalizedCase],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("case_id", "")),
            str(row.get("variant_id", "original")),
        )
        if key not in cases:
            continue
        if row.get("status") != "ok":
            continue
        if row.get("condition") != "flux_rf_style":
            continue
        if not isinstance(row.get("trajectory_plan"), dict):
            continue
        grouped[key].append(row)
    return grouped


def _dpo_pair_from_group(
    config: dict[str, Any],
    key: tuple[str, str, str],
    group: list[dict[str, Any]],
    cases: dict[tuple[str, str, str], NormalizedCase],
    template_examples: str,
) -> dict[str, Any] | None:
    if len(group) < 2:
        return None
    ranked = sorted(
        (
            (_trajectory_reward(row, config), row)
            for row in group
            if _validated_plan(row.get("trajectory_plan")) is not None
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    if len(ranked) < 2:
        return None
    chosen_reward, chosen = ranked[0]
    rejected_reward, rejected = ranked[-1]
    if chosen_reward <= rejected_reward:
        return None
    chosen_plan = _validated_plan(chosen["trajectory_plan"])
    rejected_plan = _validated_plan(rejected["trajectory_plan"])
    if chosen_plan is None or rejected_plan is None:
        return None
    chosen_text = _json_dumps(chosen_plan.model_dump(mode="json"))
    rejected_text = _json_dumps(rejected_plan.model_dump(mode="json"))
    if chosen_text == rejected_text:
        return None
    case = cases[key]
    prompt, prompt_hash = render_prompt(
        config,
        "legal_flux/rf_plan",
        case,
        max_steps=int(config["legal_flux"].get("max_steps", 4)),
        template_tag_examples=template_examples,
    )
    return {
        "id": f"trajectory-dpo-{case.case_id}-{case.variant_id}",
        "task": "trajectory_dpo",
        "case_id": case.case_id,
        "variant_id": case.variant_id,
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "chosen": chosen_text,
        "rejected": rejected_text,
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
        "chosen_run_hash": chosen.get("run_hash"),
        "rejected_run_hash": rejected.get("run_hash"),
        "metadata": {
            "chosen_answer_correct": chosen.get("answer_correct"),
            "rejected_answer_correct": rejected.get("answer_correct"),
            "chosen_sample_index": chosen.get("sample_index", 0),
            "rejected_sample_index": rejected.get("sample_index", 0),
        },
    }


def _trajectory_reward(row: dict[str, Any], config: dict[str, Any]) -> float:
    trajectory_length = int(row.get("trajectory_length") or len(row.get("executed_steps") or []))
    max_steps = max(int(config["legal_flux"].get("max_steps", 4)), 1)
    retrieved = row.get("retrieved_template_ids") or []
    executed = row.get("executed_steps") or []
    retrieval_success = 1.0 if retrieved and len(retrieved) == len(executed) else 0.0
    schema_clean = 1.0 if not row.get("schema_errors") else 0.0
    step_efficiency = max(0.0, 1.0 - max(0, trajectory_length - 1) / max_steps)
    return (
        _truthy_float(row.get("answer_correct"))
        + 0.1 * _truthy_float(row.get("binary_prediction_valid"))
        + 0.1 * retrieval_success
        + 0.05 * schema_clean
        + 0.05 * step_efficiency
    )


def _validated_plan(value: Any) -> LegalFluxAbstractPlan | None:
    try:
        return LegalFluxAbstractPlan.model_validate(value)
    except Exception:
        return None


def _truthy_float(value: Any) -> float:
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in {"1", "true", "yes"} else 0.0
    return 1.0 if bool(value) else 0.0


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
