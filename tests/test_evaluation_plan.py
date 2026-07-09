import json

from legal_pilot.evaluation import _filter_to_run_plan


def test_filter_to_run_plan_excludes_stale_generation_rows(tmp_path):
    (tmp_path / "run_plan.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {"run_hash": "current-a"},
                    {"run_hash": "current-b"},
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {"run_hash": "current-a"},
        {"run_hash": "stale"},
        {"run_hash": "current-b"},
    ]

    filtered = _filter_to_run_plan(rows, tmp_path)

    assert [row["run_hash"] for row in filtered] == [
        "current-a",
        "current-b",
    ]
