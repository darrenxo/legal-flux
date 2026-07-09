import pandas as pd
import matplotlib

from legal_pilot.reporting import (
    _audit_summary,
    _blind_reannotation,
    _condition_summary,
    _load_audit_records,
    _oracle_gap,
    _recommend,
    _select_manual_case_keys,
)


def test_reporting_uses_headless_matplotlib_backend():
    assert matplotlib.get_backend().lower() == "agg"


def test_condition_summary_includes_failures_in_itt_accuracy():
    rows = [
        {
            "dataset": "legalhk",
            "condition": "validated",
            "status": "ok",
            "answer_correct": 1.0,
            "valid_fact_reference_rate": 1.0,
        },
        {
            "dataset": "legalhk",
            "condition": "validated",
            "status": "error",
        },
    ]

    summary = _condition_summary(rows)
    row = summary.iloc[0]

    assert row["planned_n"] == 2
    assert row["usable_n"] == 1
    assert row["failure_rate"] == 0.5
    assert row["answer_accuracy_conditional"] == 1.0
    assert row["answer_accuracy_itt"] == 0.5


def test_condition_summary_reports_binary_macro_f1_and_class_recall():
    rows = [
        {
            "dataset": "legalhk",
            "condition": "structured",
            "status": "ok",
            "gold_answer": "support",
            "prediction": "support",
            "answer_correct": 1.0,
        },
        {
            "dataset": "legalhk",
            "condition": "structured",
            "status": "ok",
            "gold_answer": "reject",
            "prediction": "support",
            "answer_correct": 0.0,
        },
    ]

    summary = _condition_summary(rows)
    row = summary.iloc[0]

    assert row["macro_f1_itt"] == 1 / 3
    assert row["support_recall_itt"] == 1.0
    assert row["reject_recall_itt"] == 0.0


def test_oracle_gap_uses_exact_oracle_case_subset():
    rows = []
    outcomes = {
        "case-a": {
            "structured": 0.0,
            "typed": 1.0,
            "validated": 0.0,
            "oracle": 1.0,
        },
        "case-b": {
            "structured": 0.0,
            "typed": 0.0,
            "validated": 0.0,
            "oracle": 1.0,
        },
        "case-outside": {
            "structured": 1.0,
            "typed": 1.0,
            "validated": 1.0,
        },
    }
    for case_id, conditions in outcomes.items():
        for condition, correct in conditions.items():
            rows.append(
                {
                    "dataset": "legalhk",
                    "case_id": case_id,
                    "variant_id": "original",
                    "condition": condition,
                    "status": "ok",
                    "answer_correct": correct,
                }
            )

    gap = _oracle_gap(rows).iloc[0]

    assert gap["oracle_subset_n"] == 2
    assert gap["structured_accuracy_itt"] == 0.0
    assert gap["typed_accuracy_itt"] == 0.5
    assert gap["oracle_accuracy_itt"] == 1.0
    assert gap["automatic_oracle_gain_recovery"] == 0.5


def test_manual_selection_is_case_level_and_dataset_balanced():
    rows = []
    for dataset, count in (("openexempt", 30), ("legalhk", 24)):
        for index in range(count):
            for condition in ("direct", "structured", "typed", "validated"):
                rows.append(
                    {
                        "dataset": dataset,
                        "case_id": f"{dataset}-{index}",
                        "variant_id": "original",
                        "condition": condition,
                        "status": "ok",
                        "metadata": {
                            "suite": f"suite-{index % 3}",
                            "lawsuit_type": f"type-{index % 3}",
                        },
                    }
                )

    selected = _select_manual_case_keys(rows, count=40, seed=7)

    frame = pd.DataFrame(selected)
    assert len(selected) == 40
    assert frame.groupby("dataset").size().to_dict() == {
        "legalhk": 20,
        "openexempt": 20,
    }
    assert not frame.duplicated(["dataset", "case_id", "variant_id"]).any()


def test_reannotation_uses_fresh_opaque_output_ids():
    item = {
        "review_case_id": "M001",
        "outputs": [
            {"output_id": "original-a", "status": "ok"},
            {"output_id": "original-b", "status": "ok"},
        ],
    }

    duplicate, mappings = _blind_reannotation(item, index=1, seed=7)

    assert duplicate["review_case_id"] == "R001"
    assert {row["source_output_id"] for row in mappings} == {
        "original-a",
        "original-b",
    }
    assert {
        output["output_id"] for output in duplicate["outputs"]
    }.isdisjoint({"original-a", "original-b"})


def test_loads_successful_audits_from_api_and_local_ledgers(tmp_path):
    (tmp_path / "audits.jsonl").write_text(
        '{"run_hash":"a","status":"ok","audit":{"issue_coverage":4}}\n',
        encoding="utf-8",
    )
    (tmp_path / "audits_local_gpt_oss_20b.jsonl").write_text(
        "\n".join(
            [
                '{"run_hash":"b","status":"error","audit":null}',
                '{"run_hash":"b","status":"ok","audit":{"issue_coverage":3}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _load_audit_records(tmp_path)

    assert {row["run_hash"] for row in rows} == {"a", "b"}


def test_load_audits_filters_outputs_not_in_current_selection(tmp_path):
    (tmp_path / "audit_selection.jsonl").write_text(
        '{"run_hash":"generation-current"}\n',
        encoding="utf-8",
    )
    (tmp_path / "audits_local_gpt_oss_20b.jsonl").write_text(
        "\n".join(
            [
                '{"run_hash":"audit-current","generation_run_hash":"generation-current","status":"ok","audit":{"issue_coverage":4}}',
                '{"run_hash":"audit-stale","generation_run_hash":"generation-stale","status":"ok","audit":{"issue_coverage":1}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _load_audit_records(tmp_path)

    assert [row["run_hash"] for row in rows] == ["audit-current"]


def test_audit_summary_groups_scores_and_first_errors():
    rows = [
        {
            "dataset": "legalhk",
            "condition": "validated",
            "audit_model": "gpt-oss:20b",
            "audit": {
                "issue_coverage": 4,
                "rule_fit": 3,
                "factual_grounding": 4,
                "defense_coverage": 2,
                "burden_correctness": 3,
                "final_decision_consistency": 4,
                "first_error": "none",
            },
        }
    ]

    summary = _audit_summary(rows)

    assert summary.iloc[0]["n"] == 1
    assert summary.iloc[0]["factual_grounding"] == 4
    assert summary.iloc[0]["none_rate"] == 1


def test_recommendation_does_not_scale_when_audit_quality_declines():
    summary = pd.DataFrame(
        [
            {
                "dataset": "legalhk",
                "condition": "structured",
                "answer_accuracy_itt": 0.16,
                "valid_fact_reference_rate": 0.99,
                "failure_rate": 0.08,
            },
            {
                "dataset": "legalhk",
                "condition": "validated",
                "answer_accuracy_itt": 0.21,
                "valid_fact_reference_rate": 1.0,
                "failure_rate": 0.17,
            },
            {
                "dataset": "openexempt",
                "condition": "structured",
                "answer_accuracy_itt": 0.0,
                "valid_fact_reference_rate": 0.80,
                "failure_rate": 0.07,
            },
            {
                "dataset": "openexempt",
                "condition": "validated",
                "answer_accuracy_itt": 0.0,
                "valid_fact_reference_rate": 0.98,
                "failure_rate": 0.10,
            },
        ]
    )
    audit_summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "condition": condition,
                "factual_grounding": grounding,
                "issue_coverage": coverage,
            }
            for dataset in ("legalhk", "openexempt")
            for condition, grounding, coverage in (
                ("structured", 1.5, 1.8),
                ("validated", 1.0, 1.2),
            )
        ]
    )

    recommendation = _recommend(
        summary, pd.DataFrame(), audit_summary=audit_summary
    )

    assert recommendation.startswith("DO NOT SCALE")


def test_recommendation_accepts_legalhk_only_summary():
    summary = pd.DataFrame(
        [
            {
                "dataset": "legalhk",
                "condition": "structured",
                "answer_accuracy_itt": 0.40,
                "conclusion_with_fact_rate": 0.70,
                "failure_rate": 0.05,
            },
            {
                "dataset": "legalhk",
                "condition": "validated",
                "answer_accuracy_itt": 0.50,
                "conclusion_with_fact_rate": 0.85,
                "failure_rate": 0.05,
            },
        ]
    )

    recommendation = _recommend(summary, pd.DataFrame())

    assert not recommendation.startswith("INCOMPLETE")
    assert "OpenExempt" not in recommendation
