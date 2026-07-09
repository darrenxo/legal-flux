from legal_pilot.models import CaseState, Element, Issue
from legal_pilot.validation import validate_case_state


def valid_state() -> CaseState:
    return CaseState(
        claims=["breach of contract"],
        requested_remedies=["damages"],
        issues=[
            Issue(
                issue_id="I1",
                issue="Was a contract formed?",
                rule_or_test="offer, acceptance, consideration",
                burden_on="plaintiff",
                elements=[
                    Element(
                        element_id="E1",
                        element="acceptance",
                        supporting_fact_ids=["F1"],
                        opposing_fact_ids=[],
                        missing_information=[],
                    )
                ],
                defenses=[],
            )
        ],
    )


def test_valid_case_state_passes():
    result = validate_case_state(
        valid_state(), valid_fact_ids={"F1", "F2"}, known_parties={"P", "D"}
    )
    assert result.valid
    assert result.errors == []


def test_invalid_fact_reference_and_duplicate_ids_are_reported():
    state = valid_state()
    state.issues.append(state.issues[0].model_copy(deep=True))
    state.issues[1].elements[0].supporting_fact_ids = ["F999"]
    result = validate_case_state(
        state, valid_fact_ids={"F1"}, known_parties={"P", "D"}
    )
    codes = {error.code for error in result.errors}
    assert "duplicate_issue_id" in codes
    assert "unknown_fact_id" in codes


def test_preanalysis_status_must_remain_unresolved():
    state = valid_state()
    state.issues[0].elements[0].status = "satisfied"
    result = validate_case_state(
        state, valid_fact_ids={"F1"}, known_parties={"P", "D"}
    )
    assert any(error.code == "premature_status" for error in result.errors)

