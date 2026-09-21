from __future__ import annotations

import pytest

from scripts.compare_legal_flux_runs import (
    build_dossiers,
    build_pairs,
    build_summary,
    template_library_hash,
    validate_template_library_hash,
)


def _row(
    case_id: str,
    *,
    gold: str,
    predicted: str,
    plan_name: str,
    template_id: str,
    result: str,
    final_prompt_hash: str,
    template_hash: str,
    executor: str = "base-executor",
    status: str = "ok",
) -> dict:
    return {
        "dataset": "legalhk",
        "case_id": case_id,
        "variant_id": "original",
        "condition": "flux_rf_style",
        "status": status,
        "gold_answer": gold,
        "prediction": predicted if status == "ok" else None,
        "answer_correct": float(predicted == gold) if status == "ok" else 0.0,
        "trajectory_plan": {
            "planning_analysis": f"Analysis for {plan_name}",
            "planned_steps": [
                {
                    "step_id": "S1",
                    "step_name": plan_name,
                    "step_description": f"Apply {plan_name}",
                    "template_tags": ["procedure", "merits"],
                }
            ],
        },
        "selected_templates": [
            {
                "step_id": "S1",
                "step_name": plan_name,
                "template_tags": ["procedure", "merits"],
                "template_id": template_id,
                "template_name": f"Template {template_id}",
                "retrieval_mode": "embedding_full_pool",
                "similarity": 0.75,
                "exact_candidate_ids": [],
            }
        ],
        "executed_steps": [
            {
                "step_id": "S1",
                "template_id": template_id,
                "instantiated_result": result,
            }
        ],
        "trajectory_reviews": [
            {
                "review_analysis": "",
                "decision": "final_answer",
                "revised_remaining_steps": [],
                "final_rationale": result,
                "final_decision": predicted,
            }
        ],
        "parsed_json": {
            "final_rationale": result,
            "final_decision": predicted,
        },
        "prompt_hashes": {
            "rf_plan": f"planner-prompt-{case_id}",
            "rf_review_S1": final_prompt_hash,
        },
        "role_models": {
            "planner": "policy",
            "executor": executor,
            "reviewer": "policy",
        },
        "template_pool_hash": template_hash,
        "workflow_hash": "workflow",
        "model_digest": "model",
        "inference_runtime": "vllm",
        "inference_runtime_version": "0.21.0",
        "chat_template_kwargs": {"enable_thinking": False},
        "seed": 7,
        "sample_index": 0,
        "decoding": {"temperature": 0.0, "context_length": 16384},
        "repair_actions": [],
        "schema_errors": [],
        "calls": 3,
        "prompt_tokens": 100,
        "output_tokens": 50,
        "raw_response": "raw",
    }


def test_pairing_diagnostics_and_full_dossiers() -> None:
    templates = {
        "LF001": {
            "template_id": "LF001",
            "template_name": "Template LF001",
            "knowledge_tags": ["procedure", "merits"],
            "description": "Reusable test",
            "application_scenario": "A relevant case",
            "reasoning_flow": ["Identify", "Apply"],
            "example_application": "Synthetic",
        },
        "LF002": {
            "template_id": "LF002",
            "template_name": "Template LF002",
            "knowledge_tags": ["procedure", "remedy"],
            "description": "Alternative reusable test",
            "application_scenario": "Another relevant case",
            "reasoning_flow": ["Identify", "Apply"],
            "example_application": "Synthetic",
        },
    }
    pool_hash = template_library_hash(templates)

    sft = {
        ("legalhk", "legalhk-1", "original"): _row(
            "legalhk-1",
            gold="support",
            predicted="support",
            plan_name="Threshold",
            template_id="LF001",
            result="F1 satisfies the threshold.",
            final_prompt_hash="same-final-prompt",
            template_hash=pool_hash,
        ),
        ("legalhk", "legalhk-2", "original"): _row(
            "legalhk-2",
            gold="support",
            predicted="reject",
            plan_name="Weak plan",
            template_id="LF001",
            result="F1 was not applied well.",
            final_prompt_hash="sft-final-prompt-2",
            template_hash=pool_hash,
        ),
        ("legalhk", "legalhk-3", "original"): _row(
            "legalhk-3",
            gold="reject",
            predicted="reject",
            plan_name="Shared correct plan",
            template_id="LF001",
            result="F1 defeats the claim.",
            final_prompt_hash="shared-final-prompt-3",
            template_hash=pool_hash,
        ),
        ("legalhk", "legalhk-4", "original"): _row(
            "legalhk-4",
            gold="support",
            predicted="support",
            plan_name="Unpaired plan",
            template_id="LF001",
            result="F1 supports the claim.",
            final_prompt_hash="sft-final-prompt-4",
            template_hash=pool_hash,
        ),
    }
    dpo = {
        ("legalhk", "legalhk-1", "original"): _row(
            "legalhk-1",
            gold="support",
            predicted="reject",
            plan_name="Threshold",
            template_id="LF001",
            result="F1 satisfies the threshold.",
            final_prompt_hash="same-final-prompt",
            template_hash=pool_hash,
        ),
        ("legalhk", "legalhk-2", "original"): _row(
            "legalhk-2",
            gold="support",
            predicted="support",
            plan_name="Better plan",
            template_id="LF002",
            result="F1 now supports the requested remedy.",
            final_prompt_hash="dpo-final-prompt-2",
            template_hash=pool_hash,
        ),
        ("legalhk", "legalhk-3", "original"): _row(
            "legalhk-3",
            gold="reject",
            predicted="reject",
            plan_name="Shared correct plan",
            template_id="LF001",
            result="F1 defeats the claim.",
            final_prompt_hash="shared-final-prompt-3",
            template_hash=pool_hash,
        ),
        ("legalhk", "legalhk-4", "original"): _row(
            "legalhk-4",
            gold="support",
            predicted="reject",
            plan_name="Failed row",
            template_id="LF002",
            result="unused",
            final_prompt_hash="dpo-final-prompt-4",
            template_hash=pool_hash,
            status="error",
        ),
    }

    pairs = build_pairs(sft, dpo, max_steps=4)
    assert len(pairs) == 3
    assert pairs[0]["diagnostic_category"] == (
        "same_final_review_context_regression"
    )
    assert pairs[1]["diagnostic_category"] == "changed_trajectory_improvement"

    summary = build_summary(
        sft_rows=sft,
        dpo_rows=dpo,
        pairs=pairs,
        max_steps=4,
    )
    assert summary["coverage"]["dpo_ok"] == 3
    assert summary["outcome_transitions"] == {
        "sft_only": 1,
        "dpo_only": 1,
        "both_correct": 1,
    }
    assert not summary["control_audit"]["warnings"]

    cases = {
        key: {
            "dataset": key[0],
            "case_id": key[1],
            "variant_id": key[2],
            "claim": "A claim",
            "requested_remedy": "A remedy",
            "parties": ["Plaintiff", "Defendant"],
            "facts": {"F1": "A material fact"},
            "authorities": ["An authority"],
            "reference_issues": ["Whether the threshold is met"],
            "gold_answer": row["gold_answer"],
        }
        for key, row in sft.items()
    }
    dossiers = build_dossiers(
        selected=pairs[:2],
        cases=cases,
        templates=templates,
        sft_rows=sft,
        dpo_rows=dpo,
    )
    selected = dossiers[0]["sft"]["executed_trajectory"][0][
        "selected_template"
    ]
    assert selected["library_template"]["description"] == "Reusable test"
    assert dossiers[0]["case"]["facts"] == {"F1": "A material fact"}


def test_template_hash_and_gold_mismatch_are_rejected() -> None:
    templates = {
        "LF001": {
            "template_id": "LF001",
            "template_name": "Template LF001",
        }
    }
    pool_hash = template_library_hash(templates)
    left = _row(
        "legalhk-1",
        gold="support",
        predicted="support",
        plan_name="Threshold",
        template_id="LF001",
        result="F1 supports the claim.",
        final_prompt_hash="same",
        template_hash=pool_hash,
    )
    right = dict(left)
    right["gold_answer"] = "reject"
    with pytest.raises(ValueError, match="Gold-answer mismatch"):
        build_pairs(
            {("legalhk", "legalhk-1", "original"): left},
            {("legalhk", "legalhk-1", "original"): right},
        )

    with pytest.raises(ValueError, match="does not match"):
        validate_template_library_hash(
            label="SFT",
            rows={("legalhk", "legalhk-1", "original"): left},
            actual_hash="wrong-hash",
        )
