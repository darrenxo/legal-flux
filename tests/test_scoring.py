from legal_pilot.models import FinalAnalysis, IssueConclusion, NormalizedCase
from legal_pilot.scoring import answers_exactly_match, score_record


def test_scoring_detects_unknown_fact_and_correct_answer():
    case = NormalizedCase(
        dataset="legalhk",
        case_id="c1",
        variant_id="original",
        claim="Support the plaintiff's claim.",
        facts={"F1": "The debtor lived in Arizona."},
        gold_answer="support",
        metadata={},
    )
    analysis = FinalAnalysis(
        issue_conclusions=[
            IssueConclusion(
                issue_id="I1",
                conclusion="satisfied",
                supporting_fact_ids=["F1", "F9"],
                opposing_fact_ids=[],
                explanation="Arizona applies.",
            )
        ],
        final_decision="support",
        final_rationale="Arizona applies.",
    )
    scores = score_record(case, analysis)
    assert scores["answer_correct"] == 1.0
    assert scores["binary_prediction_valid"] == 1.0
    assert scores["conclusion_with_fact_rate"] == 1.0
    assert scores["valid_fact_reference_rate"] == 0.5
    assert scores["unknown_fact_reference_count"] == 1


def test_unresolved_without_fact_support_is_incorrect():
    case = NormalizedCase(
        dataset="legalhk",
        case_id="c2",
        claim="Support the plaintiff's claim.",
        facts={"F1": "A fact."},
        gold_answer="reject",
    )
    analysis = FinalAnalysis(
        issue_conclusions=[
            IssueConclusion(
                issue_id="I1",
                conclusion="unresolved",
                supporting_fact_ids=[],
                opposing_fact_ids=[],
                explanation="Insufficient.",
            )
        ],
        final_decision="unresolved",
        final_rationale="Insufficient.",
    )

    scores = score_record(case, analysis)

    assert scores["answer_correct"] == 0.0
    assert scores["binary_prediction_valid"] == 0.0
    assert scores["conclusion_with_fact_rate"] == 0.0


def test_irac_reasoning_metrics_do_not_claim_issue_id_coverage():
    case = NormalizedCase(
        dataset="legalhk",
        case_id="c3",
        claim="Support the plaintiff's claim.",
        facts={"F1": "A fact."},
        gold_answer="support",
        reference_issues=["Whether the claim is established."],
    )
    analysis = FinalAnalysis(
        irac_reasoning="Issue: claim. Rule: supplied rule. Application: F1. Conclusion: support.",
        final_decision="support",
    )

    scores = score_record(case, analysis)

    assert scores["irac_reasoning_present"] == 1.0
    assert scores["irac_reasoning_characters"] == len(analysis.irac_reasoning)
    assert scores["issue_coverage_proxy"] is None
    assert scores["unresolved_issue_rate"] is None


def test_exact_match_accepts_equivalent_python_and_json_structures():
    gold = "{'Wisconsin': 0, 'Federal': 0}"
    prediction = '{"Federal":0,"Wisconsin":0}'

    assert answers_exactly_match(prediction, gold)
