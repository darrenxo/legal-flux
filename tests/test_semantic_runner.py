from __future__ import annotations

from pathlib import Path

from legal_pilot.__main__ import build_parser
from legal_pilot.bot import CONDITION_SPECS, TemplateBuffer, build_bot_plan
from legal_pilot.bot_runner import _execute_bot_case, restore_condition_buffer
from legal_pilot.bot_freeze import bot_run_hash
from legal_pilot.clients import ModelResponse
from legal_pilot.config import load_config
from legal_pilot.embeddings import FixedEmbeddingBackend
from legal_pilot.models import FrontierLegalProblem, NormalizedCase


def _case(index: int, answer: str = "support") -> NormalizedCase:
    return NormalizedCase(
        dataset="legalhk",
        case_id=f"legalhk-{index}",
        claim="The plaintiff seeks repayment of a loan.",
        requested_remedy="Money judgment",
        parties=["Plaintiff", "Defendant"],
        facts={"F1": "Money was advanced.", "F2": "Repayment is disputed."},
        gold_answer=answer,
        metadata={"selection_split": "evaluation", "lawsuit_type": "Debt"},
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
        self.prompts = []

    def generate(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        return self.responses.pop(0)


def _analysis() -> dict:
    return {
        "issue_conclusions": [],
        "final_decision": "support",
        "final_rationale": "The supplied facts support repayment.",
    }


def _candidate() -> dict:
    return {
        "template_id": "loan_repayment",
        "name": "Loan repayment",
        "description": "Evaluate an alleged loan and non-payment.",
        "applicability_cues": ["loan", "repayment", "money due"],
        "reasoning_steps": [
            "Identify proof of the loan.",
            "Check whether repayment became due.",
        ],
        "required_checks": ["loan", "due date", "payment"],
        "contraindications": [],
        "provenance_case_ids": [],
        "version": 1,
    }


def _frontier_profile() -> FrontierLegalProblem:
    return FrontierLegalProblem(
        case_id="legalhk-1",
        procedural_posture="Civil repayment claim",
        claim_and_remedy="Repayment of an alleged loan",
        material_fact_ids=["F1", "F2"],
        dispositive_questions=["Whether the transfer was a loan"],
        evidence_conflicts=["Repayment obligation is disputed"],
        missing_information=[],
        retrieval_summary="disputed loan repayment obligation",
    )


def test_semantic_conditions_are_registered_and_plan_is_isolated():
    config = load_config(
        Path(__file__).parents[1] / "configs" / "legal_bot_semantic.yaml"
    )
    cases = [
        *[_case(index, "support") for index in range(32)],
        *[_case(index + 32, "reject") for index in range(32)],
    ]

    plan = build_bot_plan(cases, config, smoke=False)

    assert {
        "semantic_qwen_fixed",
        "semantic_qwen_dynamic",
        "semantic_raw_fixed",
    }.issubset(CONDITION_SPECS)
    assert len(plan) == 64 * 4
    assert {item.condition for item in plan} == {
        "direct",
        "semantic_qwen_fixed",
        "semantic_qwen_dynamic",
        "semantic_raw_fixed",
    }


def test_frontier_profile_skips_problem_distiller_call():
    client = SequenceClient([_response(_analysis())])
    buffer = TemplateBuffer(
        [],
        similarity_backend=FixedEmbeddingBackend({}),
    )

    analysis, trace, event = _execute_bot_case(
        client,
        load_config(
            Path(__file__).parents[1] / "configs" / "legal_bot_frontier.yaml"
        ),
        _case(1),
        condition="semantic_frontier_generic",
        phase="holdout",
        buffer=buffer,
        frontier_profile=_frontier_profile(),
    )

    assert analysis.final_decision == "support"
    assert trace["calls"] == 1
    assert trace["distilled_problem"]["procedural_posture"] == "Civil repayment claim"
    assert event.action == "reject"


def test_semantic_dynamic_condition_uses_append_only_update():
    candidate = _candidate()
    backend = FixedEmbeddingBackend(
        {
            "disputed loan repayment obligation": [1.0, 0.0],
            "loan repayment Loan repayment Evaluate an alleged loan and "
            "non-payment. loan repayment money due Identify proof of the loan. "
            "Check whether repayment became due. loan due date payment": [0.0, 1.0],
        }
    )
    client = SequenceClient([_response(_analysis()), _response(candidate)])
    buffer = TemplateBuffer([], similarity_backend=backend)

    _, _, event = _execute_bot_case(
        client,
        load_config(
            Path(__file__).parents[1] / "configs" / "legal_bot_frontier.yaml"
        ),
        _case(1),
        condition="semantic_frontier_dynamic",
        phase="adaptation",
        buffer=buffer,
        frontier_profile=_frontier_profile(),
    )

    assert event.action == "new"
    assert len(buffer.templates) == 1


def test_cli_exposes_semantic_and_frontier_setup_commands():
    parser = build_parser()

    assert parser.parse_args(
        ["--config", "configs/legal_bot_semantic.yaml", "bot-embedding-check"]
    ).command == "bot-embedding-check"
    assert parser.parse_args(
        ["--config", "configs/legal_bot_frontier.yaml", "bot-export-frontier"]
    ).command == "bot-export-frontier"
    args = parser.parse_args(
        [
            "--config",
            "configs/legal_bot_frontier.yaml",
            "bot-import-frontier",
            "--input",
            "profiles.jsonl",
        ]
    )
    assert args.input == "profiles.jsonl"


def test_restored_buffer_keeps_semantic_backend():
    backend = FixedEmbeddingBackend({})

    buffer = restore_condition_buffer(
        "semantic_qwen_fixed",
        [],
        similarity_backend=backend,
    )

    assert buffer.similarity_backend is backend


def test_run_hash_includes_embedding_model_digest():
    first = bot_run_hash(
        case_id="legalhk-1",
        condition="semantic_qwen_fixed",
        phase="holdout",
        stream_index=1,
        model_digest="generator",
        embedding_digest="embed-a",
        workflow_hash="workflow",
        seed=7,
    )
    second = bot_run_hash(
        case_id="legalhk-1",
        condition="semantic_qwen_fixed",
        phase="holdout",
        stream_index=1,
        model_digest="generator",
        embedding_digest="embed-b",
        workflow_hash="workflow",
        seed=7,
    )

    assert first != second
