from __future__ import annotations

import pandas as pd

from legal_pilot.bot_evaluation import enrich_bot_record
from legal_pilot.bot_reporting import (
    adaptation_curve,
    bot_condition_summary,
    buffer_summary,
    paired_bot_comparisons,
    recommend_bot_next_step,
)
from legal_pilot.models import NormalizedCase


def _case() -> NormalizedCase:
    return NormalizedCase(
        dataset="legalhk",
        case_id="legalhk-1",
        claim="Claim",
        facts={"F1": "Fact"},
        gold_answer="support",
    )


def _record(
    *,
    condition: str,
    phase: str,
    index: int,
    prediction: str,
    update: str = "reject",
    fallback: bool = False,
    size_after: int = 8,
) -> dict:
    return {
        "run_hash": f"{condition}-{phase}-{index}",
        "dataset": "legalhk",
        "case_id": f"case-{index}",
        "variant_id": "original",
        "condition": condition,
        "phase": phase,
        "stream_index": index,
        "status": "ok",
        "gold_answer": "support",
        "parsed_json": {
            "issue_conclusions": [],
            "final_decision": prediction,
            "final_rationale": "Rationale",
        },
        "retrieval": (
            {
                "template": {
                    "template_id": "generic_legal_reasoning"
                    if fallback
                    else "debt_payment"
                },
                "similarity": 0.1 if fallback else 0.7,
                "used_fallback": fallback,
            }
            if condition != "direct"
            else None
        ),
        "buffer_update": {
            "action": update,
            "source_case_id": f"case-{index}",
            "target_template_id": "learned" if update != "reject" else None,
            "template": None,
            "rationale": "test",
        },
        "buffer_size_before": max(size_after - (update == "new"), 0),
        "buffer_size_after": size_after,
        "elapsed_seconds": 1.0,
        "prompt_tokens": 100,
        "output_tokens": 50,
        "calls": 2,
    }


def test_enrich_bot_record_adds_outcome_and_memory_metrics():
    row = _record(
        condition="bot_full",
        phase="adaptation",
        index=0,
        prediction="support",
        update="new",
        fallback=True,
        size_after=9,
    )

    enriched = enrich_bot_record(row, _case())

    assert enriched["answer_correct"] == 1.0
    assert enriched["prediction"] == "support"
    assert enriched["retrieval_similarity"] == 0.1
    assert enriched["fallback_used"] == 1.0
    assert enriched["buffer_update_action"] == "new"
    assert enriched["buffer_growth"] == 1


def test_condition_summary_separates_adaptation_and_frozen_holdout():
    rows = [
        _record(
            condition="bot_full",
            phase="adaptation",
            index=0,
            prediction="support",
        ),
        _record(
            condition="bot_full",
            phase="holdout",
            index=32,
            prediction="reject",
        ),
    ]
    rows[0].update(answer_correct=1.0, fallback_used=0.0)
    rows[1].update(answer_correct=0.0, fallback_used=0.0)

    summary = bot_condition_summary(rows)

    assert set(summary["phase"]) == {"adaptation", "holdout", "all"}
    holdout = summary[summary["phase"] == "holdout"].iloc[0]
    assert holdout["answer_accuracy_itt"] == 0.0


def test_adaptation_curve_reports_roundwise_learning():
    rows = []
    for index in range(8):
        row = _record(
            condition="bot_full",
            phase="adaptation",
            index=index,
            prediction="support" if index >= 4 else "reject",
        )
        row.update(
            answer_correct=float(index >= 4),
            fallback_used=float(index < 2),
            buffer_update_action="new" if index == 4 else "reject",
        )
        rows.append(row)

    curve = adaptation_curve(rows, bins=4)

    assert list(curve["round"]) == [1, 2, 3, 4]
    assert list(curve["answer_accuracy_itt"]) == [0.0, 0.0, 1.0, 1.0]


def test_buffer_summary_counts_updates_and_reuse():
    rows = []
    for index, action in enumerate(["new", "merge", "reject"]):
        row = _record(
            condition="bot_full",
            phase="adaptation",
            index=index,
            prediction="support",
            update=action,
            fallback=index == 0,
            size_after=9,
        )
        row.update(
            answer_correct=1.0,
            fallback_used=float(index == 0),
            buffer_update_action=action,
            retrieved_template_id=(
                "generic_legal_reasoning" if index == 0 else "learned"
            ),
        )
        rows.append(row)

    summary = buffer_summary(rows).iloc[0]

    assert summary["new_templates"] == 1
    assert summary["merged_templates"] == 1
    assert summary["retrieval_reuse_rate"] == 2 / 3


def test_paired_comparison_uses_direct_on_identical_cases():
    rows = []
    for condition, correct in (("direct", 1.0), ("bot_full", 0.0)):
        for index in range(2):
            row = _record(
                condition=condition,
                phase="holdout",
                index=index,
                prediction="support" if correct else "reject",
            )
            row["answer_correct"] = correct
            rows.append(row)

    paired = paired_bot_comparisons(rows, baseline="direct", seed=7)

    row = paired[paired["condition"] == "bot_full"].iloc[0]
    assert row["paired_n"] == 2
    assert row["accuracy_difference"] == -1.0


def test_recommendation_requires_dynamic_buffer_to_beat_fixed_buffer():
    summary = pd.DataFrame(
        [
            {
                "condition": "direct",
                "phase": "holdout",
                "answer_accuracy_itt": 0.72,
            },
            {
                "condition": "bot_full",
                "phase": "holdout",
                "answer_accuracy_itt": 0.70,
            },
            {
                "condition": "bot_no_manager",
                "phase": "holdout",
                "answer_accuracy_itt": 0.70,
            },
            {
                "condition": "bot_no_buffer",
                "phase": "holdout",
                "answer_accuracy_itt": 0.60,
            },
        ]
    )
    curve = pd.DataFrame(
        [
            {
                "condition": "bot_full",
                "round": 1,
                "answer_accuracy_itt": 0.50,
            },
            {
                "condition": "bot_full",
                "round": 4,
                "answer_accuracy_itt": 0.70,
            },
        ]
    )

    recommendation = recommend_bot_next_step(summary, curve)

    assert recommendation.startswith("NO EVIDENCE FOR ONLINE UPDATES")


def test_recommendation_handles_semantic_qwen_conditions():
    summary = pd.DataFrame(
        [
            {
                "condition": "direct",
                "phase": "holdout",
                "answer_accuracy_itt": 0.75,
            },
            {
                "condition": "semantic_qwen_fixed",
                "phase": "holdout",
                "answer_accuracy_itt": 0.70,
            },
            {
                "condition": "semantic_qwen_dynamic",
                "phase": "holdout",
                "answer_accuracy_itt": 0.72,
            },
            {
                "condition": "semantic_raw_fixed",
                "phase": "holdout",
                "answer_accuracy_itt": 0.68,
            },
        ]
    )

    recommendation = recommend_bot_next_step(summary, pd.DataFrame())

    assert "SEMANTIC" in recommendation
    assert "Direct" in recommendation
