from __future__ import annotations

from pathlib import Path

from legal_pilot.bot import (
    BOT_CONDITIONS,
    TemplateBuffer,
    build_bot_plan,
    generic_template,
    seed_templates,
    select_bot_cases,
    should_update_buffer,
)
from legal_pilot.config import load_config
from legal_pilot.models import (
    BufferUpdateEvent,
    DistilledLegalProblem,
    LegalThoughtTemplate,
    NormalizedCase,
)


def _case(index: int, answer: str) -> NormalizedCase:
    return NormalizedCase(
        dataset="legalhk",
        case_id=f"legalhk-{index}",
        claim="The plaintiff seeks judgment.",
        requested_remedy="Damages",
        facts={"F1": f"Fact {index}"},
        gold_answer=answer,
        metadata={
            "selection_split": "evaluation",
            "lawsuit_type": f"type-{index % 4}",
        },
    )


def _template(template_id: str = "debt") -> LegalThoughtTemplate:
    return LegalThoughtTemplate(
        template_id=template_id,
        name="Unpaid debt claim",
        description="Evaluate whether an unpaid sum is presently due.",
        applicability_cues=["unpaid debt", "loan", "invoice", "sum due"],
        reasoning_steps=[
            "Identify the alleged obligation.",
            "Check whether payment became due.",
            "Test payment, discharge, limitation, and proof defenses.",
            "Resolve the requested remedy.",
        ],
        required_checks=["obligation", "due date", "payment", "proof"],
        contraindications=["possession-only dispute"],
        provenance_case_ids=[],
        version=1,
    )


def test_distilled_problem_and_template_models_are_strict():
    problem = DistilledLegalProblem(
        claim_type="debt",
        remedy_type="money judgment",
        lawsuit_type="commercial",
        material_factual_pattern=["loan advanced", "payment disputed"],
        issue_families=["existence of debt", "payment"],
        defenses_or_counterarguments=["repayment"],
        evidence_posture="conflicting",
        retrieval_query="commercial debt loan payment disputed",
    )

    assert problem.retrieval_query.startswith("commercial debt")
    assert _template().version == 1


def test_seeded_buffer_retrieves_domain_template_and_falls_back_when_weak():
    buffer = TemplateBuffer([_template()])

    strong = buffer.retrieve("unpaid debt loan sum due", threshold=0.15)
    weak = buffer.retrieve("judicial review immigration public law", threshold=0.95)

    assert strong.template.template_id == "debt"
    assert strong.best_candidate_template_id == "debt"
    assert strong.used_fallback is False
    assert strong.similarity >= 0.15
    assert weak.template.template_id == generic_template().template_id
    assert weak.best_candidate_template_id == "debt"
    assert weak.used_fallback is True


def test_buffer_manager_adds_then_merges_and_events_replay():
    buffer = TemplateBuffer([])
    candidate = _template()

    new_event = buffer.apply_candidate(
        candidate,
        source_case_id="legalhk-1",
        merge_threshold=0.80,
    )
    merge_event = buffer.apply_candidate(
        candidate.model_copy(update={"provenance_case_ids": []}),
        source_case_id="legalhk-2",
        merge_threshold=0.80,
    )

    assert new_event.action == "new"
    assert merge_event.action == "merge"
    assert len(buffer.templates) == 1
    assert buffer.templates[0].version == 2
    assert buffer.templates[0].provenance_case_ids == [
        "legalhk-1",
        "legalhk-2",
    ]

    replayed = TemplateBuffer.replay([], [new_event, merge_event])
    assert replayed.model_dump() == buffer.model_dump()


def test_rejected_buffer_event_does_not_change_state():
    buffer = TemplateBuffer([_template()])
    event = BufferUpdateEvent(
        action="reject",
        source_case_id="legalhk-3",
        target_template_id=None,
        template=None,
        rationale="Prediction was incorrect.",
    )

    replayed = TemplateBuffer.replay(buffer.templates, [event])

    assert replayed.model_dump() == buffer.model_dump()


def test_bot_split_is_balanced_disjoint_and_deterministic():
    config = load_config(
        Path(__file__).parents[1] / "configs" / "legal_bot.yaml"
    )
    cases = [
        *[_case(index, "support") for index in range(32)],
        *[_case(index + 32, "reject") for index in range(32)],
    ]

    first = select_bot_cases(cases, config, smoke=False)
    second = select_bot_cases(cases, config, smoke=False)

    assert [case.case_id for case in first.adaptation] == [
        case.case_id for case in second.adaptation
    ]
    assert len(first.adaptation) == 32
    assert len(first.holdout) == 32
    assert {case.case_id for case in first.adaptation}.isdisjoint(
        {case.case_id for case in first.holdout}
    )
    assert [case.gold_answer for case in first.adaptation].count("support") == 16
    assert [case.gold_answer for case in first.holdout].count("reject") == 16


def test_update_gate_is_predict_then_update_and_never_uses_holdout():
    assert should_update_buffer(
        phase="adaptation",
        answer_correct=True,
        manager_enabled=True,
        used_fallback=True,
        similarity=0.0,
        novelty_threshold=0.65,
    )
    assert not should_update_buffer(
        phase="adaptation",
        answer_correct=False,
        manager_enabled=True,
        used_fallback=True,
        similarity=0.0,
        novelty_threshold=0.65,
    )
    assert not should_update_buffer(
        phase="holdout",
        answer_correct=True,
        manager_enabled=True,
        used_fallback=True,
        similarity=0.0,
        novelty_threshold=0.65,
    )
    assert not should_update_buffer(
        phase="adaptation",
        answer_correct=True,
        manager_enabled=False,
        used_fallback=True,
        similarity=0.0,
        novelty_threshold=0.65,
    )
    assert not should_update_buffer(
        phase="adaptation",
        answer_correct=True,
        manager_enabled=True,
        used_fallback=False,
        similarity=0.90,
        novelty_threshold=0.65,
    )


def test_bot_plan_has_component_ablations_and_freezes_holdout_updates():
    config = load_config(
        Path(__file__).parents[1] / "configs" / "legal_bot.yaml"
    )
    cases = [
        *[_case(index, "support") for index in range(32)],
        *[_case(index + 32, "reject") for index in range(32)],
    ]

    plan = build_bot_plan(cases, config, smoke=False)

    assert set(BOT_CONDITIONS) == {
        "direct",
        "bot_full",
        "bot_no_distiller",
        "bot_no_buffer",
        "bot_no_manager",
        "bot_generic_init",
    }
    assert len(plan) == 64 * len(BOT_CONDITIONS)
    assert sum(item.phase == "adaptation" for item in plan) == 32 * len(
        BOT_CONDITIONS
    )
    assert all(
        not item.allow_update for item in plan if item.phase == "holdout"
    )
    assert any(
        item.allow_update and item.condition == "bot_full" for item in plan
    )
    assert all(
        not item.allow_update
        for item in plan
        if item.condition in {"direct", "bot_no_buffer", "bot_no_manager"}
    )


def test_seed_templates_are_legal_structures_not_substantive_rules():
    seeds = seed_templates()

    assert len(seeds) >= 6
    assert len({template.template_id for template in seeds}) == len(seeds)
    joined = " ".join(
        step for template in seeds for step in template.reasoning_steps
    ).lower()
    assert "section " not in joined
    assert "cap." not in joined


def test_configured_tfidf_threshold_retrieves_clear_legal_seed():
    config = load_config(
        Path(__file__).parents[1] / "configs" / "legal_bot.yaml"
    )
    buffer = TemplateBuffer(seed_templates())

    result = buffer.retrieve(
        "landlord possession tenancy own use notice",
        threshold=config["bot"]["retrieval_threshold"],
    )

    assert result.used_fallback is False
    assert result.template.template_id == "property_possession"
