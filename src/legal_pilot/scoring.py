from __future__ import annotations

import ast
import json
from typing import Any

from .io_utils import normalize_answer
from .models import FinalAnalysis, NormalizedCase


def score_record(case: NormalizedCase, analysis: FinalAnalysis) -> dict[str, Any]:
    referenced = [
        fact_id
        for conclusion in analysis.issue_conclusions
        for fact_id in conclusion.supporting_fact_ids + conclusion.opposing_fact_ids
    ]
    valid_count = sum(fact_id in case.facts for fact_id in referenced)
    unknown_count = len(referenced) - valid_count
    valid_rate = valid_count / len(referenced) if referenced else 1.0

    predicted = analysis.final_decision
    answer_correct = float(answers_exactly_match(predicted, case.gold_answer))
    conclusion_issue_ids = {item.issue_id for item in analysis.issue_conclusions}
    reference_count = len(case.reference_issues)
    issue_coverage = (
        min(len(conclusion_issue_ids) / reference_count, 1.0)
        if reference_count
        else None
    )

    return {
        "answer_correct": answer_correct,
        "binary_prediction_valid": float(predicted in {"support", "reject"}),
        "conclusion_with_fact_rate": (
            sum(
                bool(item.supporting_fact_ids or item.opposing_fact_ids)
                for item in analysis.issue_conclusions
            )
            / len(analysis.issue_conclusions)
            if analysis.issue_conclusions
            else 0.0
        ),
        "valid_fact_reference_rate": valid_rate,
        "unknown_fact_reference_count": unknown_count,
        "issue_conclusion_count": len(analysis.issue_conclusions),
        "issue_coverage_proxy": issue_coverage,
        "unresolved_issue_rate": (
            sum(item.conclusion == "unresolved" for item in analysis.issue_conclusions)
            / len(analysis.issue_conclusions)
            if analysis.issue_conclusions
            else 1.0
        ),
    }


def paired_robustness(
    originals: dict[str, dict[str, Any]],
    variants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for variant in variants:
        pair_id = variant.get("pair_id")
        original = originals.get(pair_id)
        if not original:
            continue
        same_prediction = normalize_answer(variant.get("prediction")) == normalize_answer(
            original.get("prediction")
        )
        kind = variant.get("perturbation_kind")
        expected_same = kind == "irrelevant_fact"
        results.append(
            {
                "pair_id": pair_id,
                "perturbation_kind": kind,
                "same_prediction": same_prediction,
                "expected_same": expected_same,
                "robustness_success": same_prediction == expected_same,
            }
        )
    return results


def answers_exactly_match(prediction: str | None, gold: str | None) -> bool:
    if normalize_answer(prediction) == normalize_answer(gold):
        return True
    predicted_structure = _parse_structure(prediction)
    gold_structure = _parse_structure(gold)
    return (
        predicted_structure is not None
        and gold_structure is not None
        and predicted_structure == gold_structure
    )


def _parse_structure(value: str | None) -> Any | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text[0] not in "[{":
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (dict, list, tuple)):
            return parsed
    return None
