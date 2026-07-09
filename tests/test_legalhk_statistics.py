from legal_pilot.reporting import paired_condition_comparisons


def test_paired_comparison_reports_difference_and_mcnemar_counts():
    rows = []
    outcomes = [
        ("c1", 1.0, 1.0),
        ("c2", 0.0, 1.0),
        ("c3", 1.0, 1.0),
        ("c4", 0.0, 0.0),
    ]
    for case_id, structured, validated in outcomes:
        rows.extend(
            [
                {
                    "dataset": "legalhk",
                    "case_id": case_id,
                    "variant_id": "original",
                    "condition": "structured",
                    "status": "ok",
                    "answer_correct": structured,
                },
                {
                    "dataset": "legalhk",
                    "case_id": case_id,
                    "variant_id": "original",
                    "condition": "validated",
                    "status": "ok",
                    "answer_correct": validated,
                },
            ]
        )

    comparison = paired_condition_comparisons(
        rows, baseline="structured", seed=20260619, samples=500
    )
    row = comparison.query("condition == 'validated'").iloc[0]

    assert row["paired_n"] == 4
    assert row["accuracy_difference"] == 0.25
    assert row["baseline_only_correct"] == 0
    assert row["condition_only_correct"] == 1
    assert 0.0 <= row["mcnemar_exact_p"] <= 1.0
