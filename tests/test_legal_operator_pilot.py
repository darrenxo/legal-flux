import json

from legal_pilot.legal_operator_pilot import (
    _case_prompt_payload,
    _issue_execution_order,
    _normalize_final_decision_payload,
    _normalize_issue_graph_payload,
    _normalize_issue_finding_payload,
    retrieve_operator_for_issue,
)
from legal_pilot.models import (
    LegalIssueFinding,
    LegalIssueNode,
    LegalOperator,
    NormalizedCase,
)


def _case() -> NormalizedCase:
    return NormalizedCase(
        dataset="legalhk",
        case_id="legalhk-pilot-test",
        claim="The plaintiff seeks repayment of the alleged debt.",
        requested_remedy="Order repayment of the alleged debt.",
        parties=["Plaintiff: P", "Defendant: D"],
        facts={
            "F1": "P advanced money to D.",
            "F2": "D says the money was a gift.",
        },
        authorities="A supplied statutory provision.",
        gold_answer="GOLD_LEAK_SENTINEL",
        reference_issues=["REFERENCE_ISSUE_LEAK_SENTINEL"],
        metadata={
            "selection_split": "trajectory_dev",
            "private_marker": "METADATA_LEAK_SENTINEL",
        },
    )


def _operator(
    operator_id: str,
    operator_name: str,
    issue_type: str,
) -> LegalOperator:
    return LegalOperator(
        operator_id=operator_id,
        operator_name=operator_name,
        issue_types=[issue_type],
        knowledge_tags=[issue_type, "legal_reasoning"],
        description=f"Resolve a {issue_type} issue.",
        application_scenario=f"Use when the case raises a {issue_type} question.",
        reasoning_flow=[
            "Identify the governing standard and material facts.",
            "Apply the standard to the competing positions.",
        ],
        example_application="Apply the operation to a synthetic dispute.",
    )


def test_issue_graph_normalization_caps_nodes_and_drops_unknown_references():
    case = _case()
    authorities = {"A1": "A supplied statutory provision."}
    payload = {
        "graph_analysis": "Resolve the procedural gate before the merits.",
        "root_claim": case.claim,
        "issues": [
            {
                "issue_id": "I1",
                "parent_id": "ROOT",
                "issue_type": "claim_elements",
                "issue_question": "Whether the alleged debt is enforceable.",
                "party_positions": [],
                "fact_ids": ["F1", "F99", "F1"],
                "authority_ids": ["A1", "A99"],
            },
            {
                "issue_id": "I2",
                "parent_id": "I1",
                "issue_type": "evidence",
                "issue_question": "Whether the payment was a loan or a gift.",
                "party_positions": [],
                "fact_ids": ["F2"],
                "authority_ids": [],
            },
            {
                "issue_id": "I3",
                "parent_id": "I1",
                "issue_type": "remedy",
                "issue_question": "What remedy follows.",
                "party_positions": [],
                "fact_ids": ["F1"],
                "authority_ids": [],
            },
        ],
    }

    normalized, repairs = _normalize_issue_graph_payload(
        payload,
        case=case,
        authorities=authorities,
        max_issues=2,
    )

    assert [item["issue_id"] for item in normalized["issues"]] == ["I1", "I2"]
    assert [item["parent_id"] for item in normalized["issues"]] == ["ROOT", "I1"]
    assert normalized["issues"][0]["fact_ids"] == ["F1"]
    assert normalized["issues"][0]["authority_ids"] == ["A1"]
    assert repairs


def test_issue_execution_order_runs_children_before_parent():
    parent = LegalIssueNode(
        issue_id="I1",
        parent_id="ROOT",
        issue_type="claim_elements",
        issue_question="Whether the claim is established.",
    )
    child = LegalIssueNode(
        issue_id="I2",
        parent_id="I1",
        issue_type="evidence",
        issue_question="Whether the decisive factual allegation is proved.",
    )
    grandchild = LegalIssueNode(
        issue_id="I3",
        parent_id="I2",
        issue_type="authority",
        issue_question="Which supplied authority governs that allegation.",
    )

    ordered = _issue_execution_order([parent, grandchild, child])

    assert [issue.issue_id for issue in ordered] == ["I3", "I2", "I1"]


def test_operator_retrieval_filters_by_issue_type_before_similarity():
    class RecordingBackend:
        def __init__(self):
            self.documents = []

        def similarities(self, query, documents):
            del query
            self.documents = list(documents)
            return [1.0 for _ in documents]

    issue = LegalIssueNode(
        issue_id="I1",
        parent_id="ROOT",
        issue_type="evidence",
        issue_question="Whether F1 is sufficient to discharge the burden of proof.",
        fact_ids=["F1"],
    )
    evidence = _operator("LO-EVIDENCE", "Test Evidential Sufficiency", "evidence")
    remedy = _operator("LO-REMEDY", "Select an Available Remedy", "remedy")
    backend = RecordingBackend()

    result = retrieve_operator_for_issue(
        issue,
        [remedy, evidence],
        similarity_backend=backend,
    )

    assert result["operator"].operator_id == "LO-EVIDENCE"
    assert len(backend.documents) == 1
    assert "Test Evidential Sufficiency" in backend.documents[0]
    assert "Select an Available Remedy" not in backend.documents[0]


def test_case_prompt_payload_never_exposes_gold_or_reference_annotations():
    case = _case()
    authorities = {"A1": "A supplied statutory provision."}

    payload = _case_prompt_payload(case, authorities)
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["claim"] == case.claim
    assert payload["facts"] == case.facts
    assert payload["authorities"] == authorities
    assert "gold_answer" not in payload
    assert "reference_issues" not in payload
    assert "reference_state" not in payload
    assert "metadata" not in payload
    assert "GOLD_LEAK_SENTINEL" not in serialized
    assert "REFERENCE_ISSUE_LEAK_SENTINEL" not in serialized
    assert "METADATA_LEAK_SENTINEL" not in serialized


def test_issue_finding_uses_only_assigned_evidence_and_all_children():
    case = _case()
    issue = LegalIssueNode(
        issue_id="I1",
        parent_id="ROOT",
        issue_type="claim_elements",
        issue_question="Whether the debt claim is established.",
        fact_ids=["F1"],
        authority_ids=["A1"],
    )
    payload = {
        "issue_id": "I1",
        "analysis_for_claim": "F1 supports the claim.",
        "analysis_against_claim": "F2 opposes it.",
        "resolution": "Resolve the assigned issue.",
        "conclusion": "supports_claim",
        "supporting_fact_ids": ["F1", "F2"],
        "opposing_fact_ids": ["F2"],
        "cited_authority_ids": ["A1", "A2"],
        "relied_on_child_issue_ids": ["I2"],
    }

    normalized, repairs = _normalize_issue_finding_payload(
        payload,
        issue=issue,
        case=case,
        authorities={"A1": "Supplied authority", "A2": "Other authority"},
        child_issue_ids={"I2", "I3"},
    )

    assert normalized["supporting_fact_ids"] == ["F1"]
    assert normalized["opposing_fact_ids"] == []
    assert normalized["cited_authority_ids"] == ["A1"]
    assert normalized["relied_on_child_issue_ids"] == ["I2", "I3"]
    assert repairs


def test_final_decision_is_aligned_to_resolved_root_finding():
    root = LegalIssueFinding(
        issue_id="I1",
        analysis_for_claim="",
        analysis_against_claim="",
        resolution="The challenged decision should stand.",
        conclusion="opposes_claim",
    )

    normalized, repairs = _normalize_final_decision_payload(
        {
            "final_rationale": "The challenged decision should stand.",
            "dispositive_issue_ids": ["I2"],
            "final_decision": "support",
        },
        issue_ids={"I1", "I2"},
        root_finding=root,
    )

    assert normalized["final_decision"] == "reject"
    assert normalized["dispositive_issue_ids"] == ["I1"]
    assert normalized["final_rationale"] == "The challenged decision should stand."
    assert "final_decision_aligned_to_root_finding" in repairs
