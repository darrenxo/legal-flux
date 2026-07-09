from pathlib import Path

from legal_pilot.config import load_config
from legal_pilot.jobs import build_jobs
from legal_pilot.models import CaseState, Element, Issue, NormalizedCase


def _case(index: int, split: str) -> NormalizedCase:
    state = CaseState(
        claims=["Claim"],
        requested_remedies=["Remedy"],
        issues=[
            Issue(
                issue_id="I1",
                issue="Issue",
                rule_or_test="Rule",
                burden_on="plaintiff",
                elements=[
                    Element(
                        element_id="E1",
                        element="Element",
                        supporting_fact_ids=["F1"],
                    )
                ],
            )
        ],
    )
    return NormalizedCase(
        dataset="legalhk",
        case_id=f"legalhk-{index}",
        claim="Claim",
        facts={"F1": "Fact"},
        gold_answer="support" if index % 2 else "reject",
        reference_state=state,
        metadata={"selection_split": split},
    )


def test_job_counts_and_split_isolation():
    config = load_config(
        Path(__file__).parents[1] / "configs" / "legalhk_only.yaml"
    )
    evaluation_cases = sorted(
        [_case(index + 5, "evaluation") for index in range(64)],
        key=lambda case: case.gold_answer,
        reverse=True,
    )
    cases = [_case(index, "smoke") for index in range(5)] + evaluation_cases

    smoke_jobs = build_jobs(cases, config, smoke=True)
    main_jobs = build_jobs(cases, config, smoke=False)

    assert len(smoke_jobs) == 20
    assert {
        job["case"].metadata["selection_split"] for job in smoke_jobs
    } == {"smoke"}
    assert len(main_jobs) == 316
    assert {
        job["case"].metadata["selection_split"] for job in main_jobs
    } == {"evaluation"}
    assert sum(job["condition"] == "oracle" for job in main_jobs) == 24
    assert sum(job["condition"] == "sampling_control" for job in main_jobs) == 36

    oracle_labels = [
        job["case"].gold_answer
        for job in main_jobs
        if job["condition"] == "oracle"
    ]
    sampling_cases = {
        job["case"].case_id: job["case"].gold_answer
        for job in main_jobs
        if job["condition"] == "sampling_control"
    }
    assert oracle_labels.count("support") == 12
    assert oracle_labels.count("reject") == 12
    assert list(sampling_cases.values()).count("support") == 6
    assert list(sampling_cases.values()).count("reject") == 6
