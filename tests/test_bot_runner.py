from __future__ import annotations

from pathlib import Path

from legal_pilot.bot import TemplateBuffer
from legal_pilot.bot_runner import (
    _execute_bot_case,
    restore_condition_buffer,
    run_bot_generation,
    sanitize_template_candidate,
)
from legal_pilot.bot_freeze import bot_main_plan_hash, bot_workflow_components
from legal_pilot.clients import ModelResponse
from legal_pilot.config import load_config
from legal_pilot.models import BufferUpdateEvent, LegalThoughtTemplate, NormalizedCase
from legal_pilot.__main__ import build_parser


def _case(index: int, answer: str = "support") -> NormalizedCase:
    return NormalizedCase(
        dataset="legalhk",
        case_id=f"legalhk-{index}",
        claim="The plaintiff seeks repayment of a loan.",
        requested_remedy="Money judgment",
        parties=["Plaintiff", "Defendant"],
        facts={
            "F1": "The plaintiff advanced money to the defendant.",
            "F2": "The repayment date passed without payment.",
        },
        gold_answer=answer,
        metadata={
            "selection_split": "evaluation",
            "lawsuit_type": "debt",
        },
    )


def _response(payload: dict) -> ModelResponse:
    return ModelResponse(
        raw_text="{}",
        parsed=payload,
        elapsed_seconds=0.1,
        prompt_tokens=10,
        output_tokens=5,
        metadata={},
    )


class SequenceClient:
    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        return self.responses.pop(0)


def _candidate() -> dict:
    return {
        "template_id": "loan_repayment",
        "name": "Loan repayment",
        "description": "Evaluate an alleged loan and non-payment.",
        "applicability_cues": ["loan", "repayment", "money due"],
        "reasoning_steps": [
            "Identify proof of the loan.",
            "Check whether repayment became due.",
            "Evaluate payment and proof defenses.",
            "Resolve the money remedy.",
        ],
        "required_checks": ["loan", "due date", "payment", "proof"],
        "contraindications": [],
        "provenance_case_ids": [],
        "version": 1,
    }


def test_bot_dry_run_uses_existing_cases_without_contacting_ollama(monkeypatch):
    config = load_config(
        Path(__file__).parents[1] / "configs" / "legal_bot.yaml"
    )
    cases = [
        *[_case(index, "support") for index in range(32)],
        *[_case(index + 32, "reject") for index in range(32)],
    ]
    monkeypatch.setattr("legal_pilot.bot_runner.load_cases", lambda _: cases)

    result = run_bot_generation(config, smoke=False, dry_run=True)

    assert result["jobs"] == 384
    assert result["adaptation_jobs"] == 192
    assert result["holdout_jobs"] == 192
    assert result["conditions"] == 6


def test_full_bot_distills_retrieves_analyzes_then_updates():
    config = load_config(
        Path(__file__).parents[1] / "configs" / "legal_bot.yaml"
    )
    client = SequenceClient(
        [
            _response(
                {
                    "claim_type": "debt",
                    "remedy_type": "money judgment",
                    "lawsuit_type": "debt",
                    "material_factual_pattern": [
                        "loan advanced",
                        "repayment overdue",
                    ],
                    "issue_families": ["loan", "non-payment"],
                    "defenses_or_counterarguments": [],
                    "evidence_posture": "supporting",
                    "retrieval_query": "debt loan repayment overdue",
                }
            ),
            _response(
                {
                    "issue_conclusions": [
                        {
                            "issue_id": "I1",
                            "conclusion": "satisfied",
                            "supporting_fact_ids": ["F1", "F2"],
                            "opposing_fact_ids": [],
                            "explanation": "The supplied facts show an overdue loan.",
                        }
                    ],
                    "final_decision": "support",
                    "final_rationale": "The overdue loan is supported by F1 and F2.",
                }
            ),
            _response(_candidate()),
        ]
    )
    buffer = TemplateBuffer([])

    analysis, trace, event = _execute_bot_case(
        client,
        config,
        _case(1),
        condition="bot_generic_init",
        phase="adaptation",
        buffer=buffer,
    )

    assert analysis.final_decision == "support"
    assert len(client.prompts) == 3
    assert trace["calls"] == 3
    assert trace["distilled_problem"]["claim_type"] == "debt"
    assert trace["retrieval"]["used_fallback"] is True
    assert event.action == "new"
    assert len(buffer.templates) == 1
    assert "gold_answer" not in "\n".join(client.prompts).lower()


def test_holdout_never_calls_template_distillation_or_updates():
    config = load_config(
        Path(__file__).parents[1] / "configs" / "legal_bot.yaml"
    )
    client = SequenceClient(
        [
            _response(
                {
                    "claim_type": "debt",
                    "remedy_type": "money judgment",
                    "lawsuit_type": "debt",
                    "material_factual_pattern": ["loan advanced"],
                    "issue_families": ["loan"],
                    "defenses_or_counterarguments": [],
                    "evidence_posture": "supporting",
                    "retrieval_query": "debt loan",
                }
            ),
            _response(
                {
                    "issue_conclusions": [],
                    "final_decision": "support",
                    "final_rationale": "Supported.",
                }
            ),
        ]
    )
    buffer = TemplateBuffer([])

    _, trace, event = _execute_bot_case(
        client,
        config,
        _case(2),
        condition="bot_generic_init",
        phase="holdout",
        buffer=buffer,
    )

    assert len(client.prompts) == 2
    assert trace["calls"] == 2
    assert event.action == "reject"
    assert len(buffer.templates) == 0


def test_resume_replays_only_recorded_buffer_events():
    template = LegalThoughtTemplate.model_validate(_candidate())
    event = BufferUpdateEvent(
        action="new",
        source_case_id="legalhk-1",
        target_template_id=template.template_id,
        template=template,
        rationale="new",
    )
    records = [
        {
            "condition": "bot_generic_init",
            "status": "ok",
            "stream_index": 0,
            "buffer_update": event.model_dump(mode="json"),
        },
        {
            "condition": "bot_generic_init",
            "status": "error",
            "stream_index": 1,
            "buffer_update": None,
        },
    ]

    restored = restore_condition_buffer("bot_generic_init", records)

    assert [item.template_id for item in restored.templates] == [
        "loan_repayment"
    ]


def test_resume_ignores_events_from_an_older_workflow_hash():
    template = LegalThoughtTemplate.model_validate(_candidate())
    event = BufferUpdateEvent(
        action="new",
        source_case_id="legalhk-1",
        target_template_id=template.template_id,
        template=template,
        rationale="new",
    )
    records = [
        {
            "condition": "bot_generic_init",
            "status": "ok",
            "stream_index": 0,
            "workflow_hash": "old",
            "buffer_update": event.model_dump(mode="json"),
        }
    ]

    restored = restore_condition_buffer(
        "bot_generic_init", records, workflow_hash="current"
    )

    assert restored.templates == []


def test_cli_exposes_bot_workflow_commands():
    parser = build_parser()

    for command in (
        "bot-smoke",
        "bot-freeze",
        "bot-generate",
        "bot-score",
        "bot-report",
    ):
        args = parser.parse_args(
            ["--config", "configs/legal_bot.yaml", command]
        )
        assert args.command == command

    smoke_score = parser.parse_args(
        [
            "--config",
            "configs/legal_bot.yaml",
            "bot-score",
            "--smoke",
        ]
    )
    assert smoke_score.smoke is True


def test_workflow_hash_components_include_shared_generation_contracts():
    config = load_config(
        Path(__file__).parents[1] / "configs" / "legal_bot.yaml"
    )

    components = bot_workflow_components(config)

    assert "direct.txt" in components["prompts"]
    assert "bot/distill.txt" in components["prompts"]
    assert "direct_analysis.json" in components["schemas"]
    assert "final_analysis.json" in components["schemas"]
    assert "bot/legal_thought_template.json" in components["schemas"]


def test_main_plan_hash_changes_when_selected_case_ids_change():
    config = load_config(
        Path(__file__).parents[1] / "configs" / "legal_bot.yaml"
    )
    cases = [
        *[_case(index, "support") for index in range(32)],
        *[_case(index + 32, "reject") for index in range(32)],
    ]

    first = bot_main_plan_hash(config, cases)
    changed = list(cases)
    changed[0] = changed[0].model_copy(update={"case_id": "legalhk-new"})
    second = bot_main_plan_hash(config, changed)

    assert first != second


def test_template_sanitizer_removes_case_specific_text_and_numbers():
    candidate = LegalThoughtTemplate.model_validate(
        {
            **_candidate(),
            "description": (
                "Plaintiff says F1 proves a $500 payment made in 2020, "
                "so support the claim."
            ),
        }
    )

    cleaned = sanitize_template_candidate(candidate, _case(1))

    description = cleaned.description.lower()
    assert "plaintiff" not in description
    assert "f1" not in description
    assert "500" not in description
    assert "2020" not in description
    assert "support" not in description
