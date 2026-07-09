from legal_pilot.runner import _repair_case_state_shape


def test_repairs_only_high_confidence_case_state_key_typo():
    payload = {
        "issues": [
            {
                "elements": [
                    {
                        "element_id": "E1",
                        "element": "test",
                        "supporting_fact_ids": [],
                        "opensing_fact_ids": ["F1"],
                        "missing_information": [],
                        "status": "unresolved",
                    }
                ]
            }
        ]
    }

    repaired, actions = _repair_case_state_shape(payload)

    element = repaired["issues"][0]["elements"][0]
    assert element["opposing_fact_ids"] == ["F1"]
    assert "opensing_fact_ids" not in element
    assert actions == ["schema_key_typo: opensing_fact_ids -> opposing_fact_ids"]
