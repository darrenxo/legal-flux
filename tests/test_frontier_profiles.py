from __future__ import annotations

import json

import pytest

from legal_pilot.frontier_profiles import (
    build_frontier_inputs,
    load_frontier_profiles,
)
from legal_pilot.models import FrontierLegalProblem, NormalizedCase


def _case(case_id: str = "legalhk-1") -> NormalizedCase:
    return NormalizedCase(
        dataset="legalhk",
        case_id=case_id,
        claim="The plaintiff seeks repayment.",
        requested_remedy="Money judgment",
        parties=["A", "B"],
        facts={"F1": "A transferred money to B.", "F2": "B denies a loan."},
        gold_answer="support",
        metadata={"lawsuit_type": "Debt claim", "selection_split": "evaluation"},
    )


def _profile(case_id: str = "legalhk-1") -> dict:
    return {
        "case_id": case_id,
        "procedural_posture": "Civil claim for repayment",
        "claim_and_remedy": "Repayment of an alleged loan",
        "material_fact_ids": ["F1", "F2"],
        "dispositive_questions": [
            "Whether the transfer was a loan rather than a gift"
        ],
        "evidence_conflicts": [
            "The transfer is admitted but its legal character is disputed"
        ],
        "missing_information": ["Any repayment terms or contemporaneous records"],
        "retrieval_summary": "disputed loan characterization repayment evidence",
    }


def test_frontier_export_is_outcome_blind():
    rows = build_frontier_inputs([_case()])

    assert len(rows) == 1
    assert rows[0]["case_id"] == "legalhk-1"
    assert rows[0]["facts"] == {"F1": "A transferred money to B.", "F2": "B denies a loan."}
    assert "gold_answer" not in rows[0]
    assert "reference_issues" not in rows[0]


def test_frontier_profiles_load_and_validate_required_case_ids(tmp_path):
    path = tmp_path / "profiles.jsonl"
    path.write_text(json.dumps(_profile()) + "\n", encoding="utf-8")

    profiles = load_frontier_profiles(path, required_case_ids={"legalhk-1"})

    assert isinstance(profiles["legalhk-1"], FrontierLegalProblem)
    assert profiles["legalhk-1"].retrieval_summary.startswith("disputed loan")


def test_frontier_profiles_reject_outcome_leakage(tmp_path):
    path = tmp_path / "profiles.jsonl"
    path.write_text(
        json.dumps({**_profile(), "gold_answer": "support"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="gold_answer"):
        load_frontier_profiles(path, required_case_ids={"legalhk-1"})


def test_frontier_profiles_require_complete_case_coverage(tmp_path):
    path = tmp_path / "profiles.jsonl"
    path.write_text(json.dumps(_profile()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing"):
        load_frontier_profiles(
            path, required_case_ids={"legalhk-1", "legalhk-2"}
        )


def test_frontier_profiles_reject_unknown_fact_ids(tmp_path):
    path = tmp_path / "profiles.jsonl"
    path.write_text(
        json.dumps({**_profile(), "material_fact_ids": ["F1", "F99"]}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="F99"):
        load_frontier_profiles(
            path,
            required_case_ids={"legalhk-1"},
            valid_fact_ids_by_case={"legalhk-1": {"F1", "F2"}},
        )
