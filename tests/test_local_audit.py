from legal_pilot.audit import _build_chatgpt_audit_items, _batch_items
from legal_pilot.models import NormalizedCase


def test_chatgpt_export_is_condition_blind_and_complete():
    case = NormalizedCase(
        dataset="legalhk",
        case_id="c1",
        claim="Should relief be granted?",
        facts={"F1": "A fact."},
        gold_answer="support",
        reference_issues=["Liability"],
    )
    selected = [
        {
            "run_hash": "run-1",
            "dataset": "legalhk",
            "case_id": "c1",
            "variant_id": "original",
            "condition": "validated",
            "parsed_json": {"final_decision": "support"},
        }
    ]

    items, key = _build_chatgpt_audit_items(
        selected,
        {("legalhk", "c1", "original"): case},
        seed=7,
    )

    assert len(items) == 1
    assert "condition" not in items[0]
    assert items[0]["generated_output"] == {"final_decision": "support"}
    assert key[0]["condition"] == "validated"
    assert key[0]["audit_id"] == items[0]["audit_id"]


def test_chatgpt_export_batches_without_dropping_items():
    items = [{"audit_id": f"A{i:03d}"} for i in range(23)]

    batches = _batch_items(items, size=10)

    assert [len(batch) for batch in batches] == [10, 10, 3]
    assert [item for batch in batches for item in batch] == items
