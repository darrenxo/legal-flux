from legal_pilot.input_leakage_audit import (
    InputLeakageAudit,
    render_input_leakage_prompt,
)


def test_leakage_prompt_is_condition_blind_and_defines_current_dispute():
    prompt = render_input_leakage_prompt(
        claim="The plaintiff seeks damages.",
        facts={"F1": "The parties signed a contract."},
    )

    assert "The plaintiff seeks damages." in prompt
    assert "F1: The parties signed a contract." in prompt
    assert "gold" not in prompt.lower()
    assert "support" not in prompt.lower()
    assert "reject" not in prompt.lower()
    assert "current dispute" in prompt.lower()
    assert "properly effected" in prompt.lower()
    assert "not established" in prompt.lower()
    assert "conservative" in prompt.lower()


def test_leakage_audit_schema_accepts_clean_result():
    result = InputLeakageAudit(
        risk="clean",
        current_outcome_disclosed=False,
        judicial_evaluation_present=False,
        suspect_snippets=[],
        rationale="Only party allegations and objective events are provided.",
    )

    assert result.risk == "clean"
