from pathlib import Path

import pandas as pd

from legal_pilot.data_prep import _dataset_notes, prepare_legalhk
from legal_pilot.legalhk_selection import strict_evaluation_reasons


def _legalhk_frame() -> pd.DataFrame:
    rows = []
    for outcome in ("support", "reject"):
        for index in range(40):
            rows.append(
                {
                    "plaintiff": f"Plaintiff {outcome} {index}",
                    "defendant": f"Defendant {outcome} {index}",
                    "plaintiff_claim": f"Damages for breach {outcome} {index}",
                    "more_facts": (
                        f"The parties made agreement {index}. "
                        f"Payment event {index} occurred."
                    ),
                    "related_laws": "Contract law principles.",
                    "issues": "Whether a binding agreement existed.",
                    "support&reject": outcome,
                    "lawsuit_type": f"contract-{index % 4}",
                    "court_reasoning": "Reference reasoning must not be stored.",
                    "judgment_decision": f"Decision marker {outcome} {index}.",
                }
            )
    rows.append(
        {
            **rows[0],
            "plaintiff": "Leaky plaintiff",
            "more_facts": "The court held the defendant liable.",
        }
    )
    return pd.DataFrame(rows)


def test_prepare_legalhk_creates_disjoint_balanced_screened_splits(tmp_path: Path):
    parquet_path = tmp_path / "legalhk" / "train.parquet"
    parquet_path.parent.mkdir(parents=True)
    _legalhk_frame().to_parquet(parquet_path)

    cases, selection, review_rows = prepare_legalhk(
        tmp_path,
        smoke_count=5,
        evaluation_count=64,
        seed=20260619,
        max_characters=48000,
        ngram_size=6,
        overlap_threshold=0.12,
        smoke_case_ids=[
            "legalhk-0",
            "legalhk-1",
            "legalhk-2",
            "legalhk-40",
            "legalhk-41",
        ],
        excluded_evaluation_case_ids=["legalhk-3"],
    )

    smoke = [case for case in cases if case.metadata["selection_split"] == "smoke"]
    evaluation = [
        case for case in cases if case.metadata["selection_split"] == "evaluation"
    ]
    assert len(smoke) == 5
    assert [case.case_id for case in smoke] == [
        "legalhk-0",
        "legalhk-1",
        "legalhk-2",
        "legalhk-40",
        "legalhk-41",
    ]
    assert len(evaluation) == 64
    assert {case.case_id for case in smoke}.isdisjoint(
        case.case_id for case in evaluation
    )
    assert "legalhk-3" not in {case.case_id for case in evaluation}
    assert pd.Series([case.gold_answer for case in evaluation]).value_counts().to_dict() == {
        "support": 32,
        "reject": 32,
    }
    assert all(case.authorities is None for case in cases)
    assert all(
        strict_evaluation_reasons(" ".join(case.facts.values())) == []
        for case in evaluation
    )
    assert all("reference_reasoning" not in case.metadata for case in cases)
    assert all("judgment_decision" not in case.metadata for case in cases)
    assert selection["excluded_reasons"]["explicit_court_outcome"] >= 1
    assert selection["strict_evaluation_pool_rows"] >= 64
    assert selection["manually_excluded_evaluation_cases"] == 1
    assert selection["evaluation_review_status"] == "local_audit_pending"
    assert all(
        not {
            "gold_answer",
            "court_reasoning",
            "judgment_decision",
            "support&reject",
        }
        & set(row)
        for row in review_rows
    )


def test_legalhk_only_notes_do_not_describe_openexempt():
    notes = _dataset_notes(["legalhk"])

    assert all("OpenExempt" not in note for note in notes)
    assert any("LegalHK" in note for note in notes)
