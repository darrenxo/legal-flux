from legal_pilot.audit_selection import balanced_select


def test_balanced_selection_is_deterministic_and_spans_conditions():
    rows = []
    for condition in ["direct", "structured", "typed", "validated"]:
        for index in range(10):
            rows.append(
                {
                    "dataset": "openexempt" if index % 2 == 0 else "legalhk",
                    "condition": condition,
                    "case_id": f"{condition}-{index}",
                    "status": "ok",
                }
            )
    first = balanced_select(rows, limit=12, seed=42)
    second = balanced_select(rows, limit=12, seed=42)
    assert first == second
    assert {row["condition"] for row in first} == {
        "direct",
        "structured",
        "typed",
        "validated",
    }

