import json
from pathlib import Path

from legal_pilot.runner import (
    _normalize_case_state_payload,
    _normalize_direct_payload,
    _normalize_final_analysis_payload,
    _recover_case_state_schema_envelope,
)
from legal_pilot.models import NormalizedCase


def test_direct_normalization_drops_obsolete_task_answer():
    payload, repairs = _normalize_direct_payload(
        {
            "final_decision": "support",
            "task_answer": [{"asset": "car", "answer": "$1,000"}],
            "final_rationale": "Applied the supplied rule.",
        }
    )

    assert "task_answer" not in payload
    assert repairs == ["obsolete_task_answer_removed"]


def test_direct_normalization_does_not_invent_answer_from_pure_schema():
    payload, repairs = _normalize_direct_payload(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "DirectAnalysis",
            "type": "object",
            "properties": {
                "final_decision": {"type": "string"},
                "final_rationale": {"type": "string"},
            },
        }
    )

    assert "final_decision" not in payload
    assert repairs == []


def test_flattens_nested_final_analysis_envelope_and_repairs_key_typo():
    payload, repairs = _normalize_final_analysis_payload(
        {
            "issue_conclusions": [
                {
                    "issue_id": "I1",
                    "conclusion": "satisfied",
                    "supporting_fact_ids": ["F1"],
                    "oposing_fact_ids": [],
                    "explanation": "Supported.",
                },
                {
                    "issue_conclusions": [],
                    "final_decision": "support",
                    "task_answer": ["A", "B"],
                    "final_rationale": "Overall result.",
                },
            ]
        }
    )

    assert payload["final_decision"] == "support"
    assert "task_answer" not in payload
    assert payload["issue_conclusions"][0]["opposing_fact_ids"] == []
    assert len(payload["issue_conclusions"]) == 1
    assert "nested_final_analysis_flattened" in repairs
    assert "obsolete_task_answer_removed" in repairs
    assert "schema_key_typo: oposing_fact_ids -> opposing_fact_ids" in repairs


def test_normalizes_case_state_defense_objects_and_punctuated_key():
    payload, repairs = _normalize_case_state_payload(
        {
            "claims": ["claim"],
            "requested_remedies": ["remedy"],
            "issues": [
                {
                    "issue_id": "I1",
                    "issue": "Issue",
                    "rule_or_test": "Rule",
                    "burden_on": "plaintiff",
                    "elements": [
                        {
                            "element_id": "E1",
                            "element": "Element",
                            "supporting_fact_ids": [],
                            "opforcing_fact_ids": ["F2"],
                            "missing_information": [],
                            "status": "unresolved",
                        }
                    ],
                    "defenses[": [{"defense": "Limitations"}],
                }
            ],
        }
    )

    issue = payload["issues"][0]
    assert issue["defenses"] == ["Limitations"]
    assert issue["elements"][0]["opposing_fact_ids"] == ["F2"]
    assert "defense_object_serialized_to_string" in repairs


def test_case_state_normalization_removes_schema_metadata():
    payload, repairs = _normalize_case_state_payload(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "CaseState",
            "claims": ["claim"],
            "requested_remedies": ["remedy"],
            "issues": [],
        }
    )

    assert "$schema" not in payload
    assert "title" not in payload
    assert repairs == [
        "schema_metadata_removed: $schema",
        "schema_metadata_removed: title",
    ]


def test_case_state_repairs_concatenated_remedy_and_issues_key():
    payload, repairs = _normalize_case_state_payload(
        {
            "claims": ["Fraud claim"],
            'requested_remedies["Damages"],"issues': [
                {
                    "issue_id": "I1",
                    "issue": "Fraud",
                    "rule_or_test": "Fraud elements",
                    "burden_on": "plaintiff",
                    "elements": [],
                    "defenses": [],
                }
            ],
        }
    )

    assert payload["requested_remedies"] == ["Damages"]
    assert payload["issues"][0]["issue_id"] == "I1"
    assert "concatenated_requested_remedies_issues_key_repaired" in repairs


def test_case_state_drops_orphan_issue_and_fills_missing_remedy():
    payload, repairs = _normalize_case_state_payload(
        {
            "claims": ["Claim"],
            "issues": [
                {
                    "issue_id": "I1",
                    "issue": "Issue",
                    "rule_or_test": "Rule",
                    "burden_on": "plaintiff",
                    "elements": [],
                    "defenses": [],
                },
                {"defenses": ["Limitations"]},
            ],
        }
    )

    assert payload["requested_remedies"] == ["Claim"]
    assert len(payload["issues"]) == 1
    assert "requested_remedies_copied_from_claims" in repairs
    assert "orphan_issue_removed" in repairs


def test_final_analysis_fills_only_missing_summary_field():
    payload, repairs = _normalize_final_analysis_payload(
        {
            "issue_conclusions": [],
            "final_decision": "reject",
        }
    )

    assert payload["final_rationale"] == "See issue conclusions."
    assert repairs == ["missing_final_rationale_filled"]


def test_case_state_unwraps_text_objects_in_string_lists():
    payload, repairs = _normalize_case_state_payload(
        {
            "claims": [{"text": "Repayment claim"}],
            "requested_remedies": [{"text": "Judgment for repayment"}],
            "issues": [],
        }
    )

    assert payload["claims"] == ["Repayment claim"]
    assert payload["requested_remedies"] == ["Judgment for repayment"]
    assert repairs == [
        "claims_text_object_unwrapped",
        "requested_remedies_text_object_unwrapped",
    ]


def test_final_analysis_unwraps_fact_ids_and_promotes_issue_level_summary():
    payload, repairs = _normalize_final_analysis_payload(
        {
            "issue_conclusions": [
                {
                    "issue_id": "I1",
                    "conclusion": "not_satisfied",
                    "supporting_fact_ids": [],
                    "opposing_fact_ids": [{"fact_id": "F1"}],
                    "explanation": "Not proved.",
                    'F,"final_decision': "reject",
                    "final_rationale": "The first issue fails.",
                },
                {
                    "issue_id": "I2",
                    "conclusion": "defeated",
                    "supporting_fact_ids": ["F2"],
                    "opposing_fact_ids": [],
                    "explanation": "A defense succeeds.",
                    "final_decision": "reject",
                    "final_rationale": "The defense independently defeats the claim.",
                },
            ]
        }
    )

    assert payload["issue_conclusions"][0]["opposing_fact_ids"] == ["F1"]
    assert payload["final_decision"] == "reject"
    assert payload["final_rationale"] == (
        "The first issue fails. The defense independently defeats the claim."
    )
    assert all(
        "final_decision" not in key and key != "final_rationale"
        for issue in payload["issue_conclusions"]
        for key in issue
    )
    assert "fact_id_object_unwrapped" in repairs
    assert "issue_level_final_decision_promoted" in repairs
    assert "issue_level_final_rationales_combined" in repairs


def test_final_analysis_fills_missing_required_issue_scaffolding():
    payload, repairs = _normalize_final_analysis_payload(
        {
            "issue_conclusions": [
                {
                    "conclusion": "not-satisfied",
                    "supporting_fact_ids": None,
                    "opposing_fact_ids": "F1",
                }
            ],
            "final_decision": "reject",
            "final_rationale": "The claim is not proved.",
        }
    )

    issue = payload["issue_conclusions"][0]
    assert issue == {
        "issue_id": "I1",
        "conclusion": "unresolved",
        "supporting_fact_ids": [],
        "opposing_fact_ids": ["F1"],
        "explanation": "No issue-level explanation supplied.",
    }
    assert "issue_id_missing_filled" in repairs
    assert "issue_conclusion_invalid_filled" in repairs
    assert "issue_explanation_missing_filled" in repairs
    assert "supporting_fact_ids_missing_filled" in repairs
    assert "opposing_fact_ids_wrapped_as_array" in repairs


def test_final_analysis_folds_extra_fields_into_text_fields():
    payload, repairs = _normalize_final_analysis_payload(
        {
            "issue_conclusions": [
                {
                    "issue_id": "I1",
                    "conclusion": "satisfied",
                    "supporting_fact_ids": [],
                    "opposing_fact_ids": [],
                    "explanation": "The issue is satisfied.",
                    "burden_notes": {"burden": "plaintiff"},
                }
            ],
            "final_decision": "support",
            "final_rationale": "The claim succeeds.",
            "overall_notes": {"confidence": "medium"},
        }
    )

    issue = payload["issue_conclusions"][0]
    assert "burden_notes" not in issue
    assert "overall_notes" not in payload
    assert "Additional structured notes" in issue["explanation"]
    assert "Additional structured notes" in payload["final_rationale"]
    assert "issue_conclusion_extra_fields_folded_into_explanation" in repairs
    assert "final_analysis_extra_fields_folded_into_final_rationale" in repairs


def test_recovers_substantive_issues_from_schema_envelope_using_case_input():
    case = NormalizedCase(
        dataset="legalhk",
        case_id="legalhk-1",
        claim="Damages for breach",
        requested_remedy="Damages for breach",
        facts={"F1": "Fact"},
        gold_answer="support",
    )
    payload = {
        "type": "object",
        "properties": {
            "claims": {"type": "array"},
            "requested_remedies": {"type": "array"},
            "issues": [
                {
                    "issue_id": "I1",
                    "issue": "Breach",
                    "rule_or_test": "Contract test",
                    "burden_on": "plaintiff",
                    "elements": [],
                    "defenses": [],
                }
            ],
        },
    }

    recovered, repairs = _recover_case_state_schema_envelope(payload, case)

    assert recovered["claims"] == ["Damages for breach"]
    assert recovered["requested_remedies"] == ["Damages for breach"]
    assert recovered["issues"][0]["issue_id"] == "I1"
    assert repairs == ["case_state_schema_envelope_unwrapped"]


def test_analysis_schemas_do_not_expose_task_answer():
    root = Path(__file__).parents[1]
    for name in ("direct_analysis.json", "final_analysis.json"):
        schema = json.loads((root / "schemas" / name).read_text(encoding="utf-8"))
        assert "task_answer" not in schema["properties"]
        assert "task_answer" not in schema["required"]
        assert set(schema["properties"]["final_decision"]["enum"]) == {
            "support",
            "reject",
            "mixed",
            "unresolved",
        }


def test_principal_prompts_are_binary_and_hide_reference_authorities():
    root = Path(__file__).parents[1]
    for name in ("direct.txt", "structured.txt", "state.txt", "state_analysis.txt"):
        prompt = (root / "prompts" / name).read_text(encoding="utf-8").lower()
        assert "task_answer" not in prompt
        assert "non-binary" not in prompt
        assert "{authorities}" not in prompt
