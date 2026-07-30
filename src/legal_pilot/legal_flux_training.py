from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_path
from .io_utils import latest_by_run_hash, read_jsonl, sha256_text, write_jsonl
from .legal_flux import load_template_pool, template_pool_hash
from .legal_flux_dpo import _dpo_settings
from .legal_flux_runner import _template_tag_examples
from .legal_flux_xsim import load_xsim_neighbors
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
            "Input contains template name and knowledge tags only.",
            "Assistant reconstructs description and scope; LegalFlux maps the "
            "library's application_scenario field to ReasonFlux scope.",
            "Reasoning flow remains in the executable template library and is "
            "not an SFT target.",
            "This is template-structure SFT, not trajectory SFT.",
        ],
        "reasonflux_objective": "P(description, scope | template_name, knowledge_tags)",
        "paper_disclosed_training": {
            "epochs": 6,
            "optimizer": "AdamW",
            "lr_scheduler": "cosine",
            "learning_rate": None,
            "batch_size": None,
        },
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
    output_dir = _training_dir(config)
    settings = _dpo_settings(config)
    candidates_path = output_dir / settings["candidates_file"]
    evaluations_path = output_dir / settings["evaluations_file"]
    if not candidates_path.exists() or not evaluations_path.exists():
        raise RuntimeError(
            "DPO trajectory candidates or X_sim evaluations are missing. Run "
            "`flux-build-dpo-data --stage all` first."
        )
    candidates = [
        row
        for row in latest_by_run_hash(read_jsonl(candidates_path))
        if row.get("status") == "ok"
    ]
    evaluations = [
        row
        for row in latest_by_run_hash(read_jsonl(evaluations_path))
        if row.get("status") == "ok"
    ]
    cases = {
        case.case_id: case
        for case in load_cases(config)
        if case.metadata.get("selection_split") == normalized_phase
    }
    xsim = load_xsim_neighbors(config)
    templates = load_template_pool(config)
    template_examples = _template_tag_examples(config, templates)
    evaluations_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in evaluations:
        evaluations_by_candidate.setdefault(str(row["candidate_id"]), []).append(row)
    candidates_by_anchor: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        anchor_id = str(candidate["anchor_case_id"])
        if anchor_id in cases:
            candidates_by_anchor.setdefault(anchor_id, []).append(candidate)
    pairs = [
        pair
        for pair in (
            _dpo_pair_from_xsim_group(
                config,
                anchor_id,
                group,
                cases[anchor_id],
                xsim.get(anchor_id, []),
                evaluations_by_candidate,
                template_examples,
            )
            for anchor_id, group in candidates_by_anchor.items()
        )
        if pair is not None
    ]
    output_path = output_dir / "trajectory_dpo.jsonl"
    write_jsonl(output_path, pairs)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "trajectory_dpo",
        "phase": normalized_phase,
        "candidates_path": str(candidates_path),
        "evaluations_path": str(evaluations_path),
        "candidate_rows": len(candidates),
        "evaluation_rows": len(evaluations),
        "case_groups": len(candidates_by_anchor),
        "pairs": len(pairs),
        "output_path": str(output_path),
        "output_sha256": sha256_text(output_path.read_text(encoding="utf-8")),
        "template_pool_hash": template_pool_hash(templates),
        "reward": "mean binary accuracy over anchor plus two X_sim neighbors",
        "notes": [
            "Chosen/rejected responses are planner trajectory JSON objects.",
            "Each anchor contributes four candidate trajectories before filtering.",
            "The highest-accuracy trajectory is chosen and the lowest is rejected.",
            "Ties within the best or worst accuracy tier are resolved by mean "
            "retrieval similarity and trajectory efficiency.",
            "A completed evaluation with an invalid final label counts as incorrect.",
            "Groups with one shared accuracy level and incomplete three-case "
            "evaluations do not produce pairs.",
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
        "scope": template.application_scenario,
    }
    system = {
        "role": "system",
        "content": "You are learning the LegalFlux structured template library.",
    }
    user = {
        "role": "user",
        "content": (
            "Given this LegalFlux template name and knowledge tags, return one "
            "JSON object with exactly two fields: description and scope.\n\n"
            f"Template name: {template.template_name}\n"
            "Knowledge tags: "
            + ", ".join(template.knowledge_tags)
        ),
    }
    assistant = {
        "role": "assistant",
        "content": _json_dumps(assistant_payload),
    }
    return {
        "id": f"template-structure-{template.template_id}",
        "task": "template_structure_sft",
        "template_id": template.template_id,
        "prompt": [system, user],
        "completion": [assistant],
        "messages": [system, user, assistant],
        "reasoning_flow_metadata": template.reasoning_flow,
    }


def _dpo_pair_from_xsim_group(
    config: dict[str, Any],
    anchor_id: str,
    candidates: list[dict[str, Any]],
    case: NormalizedCase,
    xsim_case_ids: list[str],
    evaluations_by_candidate: dict[str, list[dict[str, Any]]],
    template_examples: str,
) -> dict[str, Any] | None:
    if len(candidates) < 2 or len(xsim_case_ids) != 3:
        return None
    expected_targets = set(xsim_case_ids)
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        plan = _validated_plan(candidate.get("trajectory_plan"))
        if plan is None:
            continue
        evaluations = evaluations_by_candidate.get(str(candidate["candidate_id"]), [])
        by_target = {
            str(row["target_case_id"]): row
            for row in evaluations
            if str(row.get("anchor_case_id")) == anchor_id
        }
        if set(by_target) != expected_targets:
            continue
        ordered_evaluations = [by_target[target_id] for target_id in xsim_case_ids]
        reward = sum(
            1.0 if bool(row.get("answer_correct")) else 0.0
            for row in ordered_evaluations
        ) / len(ordered_evaluations)
        tie_break = _trajectory_tie_break(candidate, plan)
        ranked.append(
            {
                "reward": reward,
                "candidate": candidate,
                "evaluations": ordered_evaluations,
                "tie_break": tie_break,
            }
        )
    if len(ranked) < 2:
        return None
    chosen_reward = max(float(item["reward"]) for item in ranked)
    rejected_reward = min(float(item["reward"]) for item in ranked)
    if chosen_reward <= rejected_reward:
        return None
    chosen_item = max(
        (item for item in ranked if item["reward"] == chosen_reward),
        key=_tie_break_sort_key,
    )
    rejected_item = min(
        (item for item in ranked if item["reward"] == rejected_reward),
        key=_tie_break_sort_key,
    )
    chosen = chosen_item["candidate"]
    rejected = rejected_item["candidate"]
    chosen_evaluations = chosen_item["evaluations"]
    rejected_evaluations = rejected_item["evaluations"]
    chosen_plan = _validated_plan(chosen["trajectory_plan"])
    rejected_plan = _validated_plan(rejected["trajectory_plan"])
    if chosen_plan is None or rejected_plan is None:
        return None
    chosen_text = _json_dumps(chosen_plan.model_dump(mode="json"))
    rejected_text = _json_dumps(rejected_plan.model_dump(mode="json"))
    if chosen_text == rejected_text:
        return None
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
        "chosen_candidate_id": chosen.get("candidate_id"),
        "rejected_candidate_id": rejected.get("candidate_id"),
        "metadata": {
            "x_sim_case_ids": xsim_case_ids,
            "chosen_answers_correct": [
                bool(row.get("answer_correct")) for row in chosen_evaluations
            ],
            "rejected_answers_correct": [
                bool(row.get("answer_correct")) for row in rejected_evaluations
            ],
            "chosen_template_ids": chosen.get("retrieved_template_ids", []),
            "rejected_template_ids": rejected.get("retrieved_template_ids", []),
            "chosen_sample_index": chosen.get("sample_index", 0),
            "rejected_sample_index": rejected.get("sample_index", 0),
            "chosen_tie_break": chosen_item["tie_break"],
            "rejected_tie_break": rejected_item["tie_break"],
        },
    }


def _trajectory_tie_break(
    candidate: dict[str, Any],
    plan: LegalFluxAbstractPlan,
) -> dict[str, float | int]:
    retrieval_trace = candidate.get("retrieval_trace") or []
    similarities = [
        float(item["similarity"])
        for item in retrieval_trace
        if item.get("similarity") is not None
    ]
    return {
        "mean_retrieval_similarity": (
            sum(similarities) / len(similarities) if similarities else 0.0
        ),
        "trajectory_steps": len(plan.planned_steps),
        "sample_index": int(candidate.get("sample_index", 0)),
    }


def _tie_break_sort_key(item: dict[str, Any]) -> tuple[float, int, int]:
    tie_break = item["tie_break"]
    return (
        float(tie_break["mean_retrieval_similarity"]),
        -int(tie_break["trajectory_steps"]),
        -int(tie_break["sample_index"]),
    )


def _validated_plan(value: Any) -> LegalFluxAbstractPlan | None:
    try:
        return LegalFluxAbstractPlan.model_validate(value)
    except Exception:
        return None


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
