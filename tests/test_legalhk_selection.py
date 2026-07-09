import pandas as pd

from legal_pilot.legalhk_selection import (
    explicit_leakage_reasons,
    is_civil_legalhk_row,
    select_legalhk_splits,
    strict_evaluation_reasons,
)


def test_explicit_holding_and_credibility_language_are_rejected():
    text = (
        "The court held the defendant liable. "
        "The plaintiff was a truthful and reliable witness."
    )

    reasons = explicit_leakage_reasons(
        text,
        judgment_decision="The defendant is liable.",
        ngram_size=6,
        overlap_threshold=0.12,
    )

    assert "explicit_court_outcome" in reasons
    assert "credibility_finding" in reasons


def test_neutral_event_facts_pass():
    reasons = explicit_leakage_reasons(
        "The parties signed a lease in 2018. Rent was unpaid for three months.",
        judgment_decision="The claim is dismissed.",
        ngram_size=6,
        overlap_threshold=0.12,
    )

    assert reasons == []


def test_prior_judgment_and_judicial_credibility_language_are_rejected():
    texts = {
        "judgment_entered": (
            "Interlocutory judgment on liability was entered against the employer."
        ),
        "admitted_liability": (
            "The respondents admitted liability and judgment was entered "
            "in favour of the applicant."
        ),
        "judicial_credibility": (
            "The court preferred the evidence of Mr Shum and found him honest."
        ),
        "appeal_disposition": "The appeal was dismissed and a retrial was ordered.",
        "costs_awarded": "Leave to defend was granted and costs were awarded.",
    }

    for expected_reason, text in texts.items():
        reasons = explicit_leakage_reasons(
            text,
            judgment_decision="Unrelated decision wording.",
            ngram_size=6,
            overlap_threshold=0.12,
        )
        assert expected_reason in reasons


def test_evaluative_judgment_summaries_are_rejected():
    texts = [
        "There was no evidence to support the alleged connection.",
        "The plaintiff failed to provide evidence during discovery.",
        "The court received applications and made orders for costs.",
        "The medical evidence did not support the respondents' objections.",
        "The plaintiff's allegations were vague and imprecise.",
    ]

    for text in texts:
        reasons = explicit_leakage_reasons(
            text,
            judgment_decision="Unrelated decision wording.",
            ngram_size=6,
            overlap_threshold=0.12,
        )
        assert "evaluative_conclusion" in reasons


def test_legal_entitlement_and_credibility_conclusions_are_rejected():
    texts = [
        "The applicant is only entitled to sick leave up to September.",
        "Miss Lee's evidence was found to be more credible than the plaintiff's.",
        "Doctors found that the applicant had not been honest about his symptoms.",
        "The video clearly demonstrated exaggeration.",
    ]

    for text in texts:
        reasons = explicit_leakage_reasons(
            text,
            judgment_decision="Unrelated decision wording.",
            ngram_size=6,
            overlap_threshold=0.12,
        )
        assert "legal_or_credibility_conclusion" in reasons


def test_party_specific_legal_evaluations_are_rejected():
    texts = [
        "The third defendant's contentions were inconsistent with the rules.",
        "The plaintiff is equally responsible for not joining a necessary party.",
    ]

    for text in texts:
        reasons = explicit_leakage_reasons(
            text,
            judgment_decision="Unrelated decision wording.",
            ngram_size=6,
            overlap_threshold=0.12,
        )
        assert "party_evaluation" in reasons


def test_strict_evaluation_gate_rejects_judicial_result_language():
    assert strict_evaluation_reasons(
        "The Court granted summary judgment after finding the evidence credible."
    ) == ["strict_judicial_or_evaluative_language"]
    assert strict_evaluation_reasons(
        "The parties signed an agreement and payment was not made."
    ) == []


def test_hksar_and_sentencing_rows_are_not_civil():
    assert not is_civil_legalhk_row(
        plaintiff="HKSAR",
        lawsuit_type="criminal case",
        claim="trafficking in a dangerous drug",
    )
    assert is_civil_legalhk_row(
        plaintiff="Alice",
        lawsuit_type="negligence claim",
        claim="damages for vehicle repair",
    )


def test_immigration_refugee_habeas_and_criminal_review_are_not_civil():
    for lawsuit_type, claim in (
        ("Habeas Corpus", "release from immigration detention"),
        ("non-refoulement claim", "risk of persecution in Bangladesh"),
        ("immigration appeal", "review of a removal order"),
        ("appeal by way of case stated", "review of a criminal conviction"),
    ):
        assert not is_civil_legalhk_row(
            plaintiff="Applicant",
            lawsuit_type=lawsuit_type,
            claim=claim,
        )


def test_selection_is_balanced_deterministic_and_disjoint():
    rows = []
    for outcome in ("support", "reject"):
        for index in range(40):
            rows.append(
                {
                    "support&reject": outcome,
                    "issues": "Issue one\nIssue two" if index % 2 else "Issue one",
                    "more_facts": "Fact text " * (20 + index),
                    "lawsuit_type": f"type-{index % 4}",
                    "has_defense": bool(index % 2),
                }
            )
    frame = pd.DataFrame(rows)

    smoke, evaluation = select_legalhk_splits(
        frame,
        smoke_count=5,
        evaluation_count=64,
        seed=20260619,
    )
    smoke_again, evaluation_again = select_legalhk_splits(
        frame,
        smoke_count=5,
        evaluation_count=64,
        seed=20260619,
    )

    assert len(smoke) == 5
    assert len(evaluation) == 64
    assert set(smoke.index).isdisjoint(evaluation.index)
    assert evaluation["support&reject"].value_counts().to_dict() == {
        "support": 32,
        "reject": 32,
    }
    assert smoke.index.tolist() == smoke_again.index.tolist()
    assert evaluation.index.tolist() == evaluation_again.index.tolist()


def test_selection_accepts_manually_reviewed_smoke_indices():
    rows = []
    for outcome in ("support", "reject"):
        for index in range(40):
            rows.append(
                {
                    "support&reject": outcome,
                    "issues": "Issue",
                    "more_facts": "Neutral facts " * (20 + index),
                    "lawsuit_type": "contract",
                    "has_defense": False,
                }
            )
    frame = pd.DataFrame(rows)
    reviewed = [0, 1, 2, 40, 41]

    smoke, evaluation = select_legalhk_splits(
        frame,
        smoke_count=5,
        evaluation_count=64,
        seed=20260619,
        smoke_indices=reviewed,
    )

    assert smoke.index.tolist() == reviewed
    assert set(smoke.index).isdisjoint(evaluation.index)
    assert evaluation["support&reject"].value_counts().to_dict() == {
        "support": 32,
        "reject": 32,
    }
