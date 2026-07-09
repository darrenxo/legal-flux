from __future__ import annotations

from pathlib import Path

import pytest

from legal_pilot.__main__ import build_parser
from legal_pilot.clients import ModelResponse
from legal_pilot.config import load_config
from legal_pilot.legal_flux import (
    fixed_trajectory_plan,
    load_template_pool,
    retrieve_template_for_abstract_step,
    sanitize_flux_template,
    template_pool_hash,
    validate_template_pool,
)
from legal_pilot.legal_flux_chatgpt import export_legal_flux_chatgpt_batches
from legal_pilot.legal_flux_runner import (
    _execute_flux_case,
    _normalize_plan_payload,
    _normalize_review_payload,
    _normalize_step_artifact_payload,
    _renumber_remaining_steps,
    flux_run_hash,
)
from legal_pilot.embeddings import FixedEmbeddingBackend
from legal_pilot.models import LegalFluxAbstractStep, LegalFluxPlanStep
from legal_pilot.legal_flux_setup import import_legal_flux_templates
from legal_pilot.io_utils import read_jsonl, write_jsonl
from legal_pilot.models import LegalFluxTemplate, NormalizedCase


def _case(split: str = "smoke") -> NormalizedCase:
    return NormalizedCase(
        dataset="legalhk",
        case_id="legalhk-100",
        claim="The plaintiff seeks repayment of a disputed loan.",
        requested_remedy="Money judgment",
        parties=["Plaintiff: A", "Defendant: B"],
        facts={
            "F1": "The plaintiff advanced money to the defendant.",
            "F2": "The defendant disputes that repayment is due.",
        },
        authorities=None,
        gold_answer="support",
        reference_issues=["Whether the defendant must repay the money."],
        metadata={"selection_split": split, "lawsuit_type": "Debt"},
    )


def _template(template_id: str, name: str, *tags: str) -> LegalFluxTemplate:
    return LegalFluxTemplate(
        template_id=template_id,
        template_name=name,
        knowledge_tags=list(tags) or ["debt", "rule_application"],
        description=f"Use {name} for a reusable legal reasoning step.",
        application_scenario=f"Cases needing {name}.",
        reasoning_flow=[
            "Identify the legally material question.",
            "Match supplied facts to the selected question.",
        ],
        example_application="Apply the template to a generic disputed obligation.",
    )


def _profiled_case(
    index: int,
    *,
    family: str,
    demand: str,
    split: str = "template_source",
) -> NormalizedCase:
    return NormalizedCase(
        dataset="legalhk",
        case_id=f"legalhk-{index}",
        claim=f"The plaintiff seeks civil relief in case {index}.",
        requested_remedy="Civil relief",
        parties=["Plaintiff", "Defendant"],
        facts={"F1": f"Material fact for case {index}."},
        authorities="Applicable principles.",
        gold_answer="support" if index % 2 else "reject",
        reference_issues=["Whether civil relief should be granted."],
        metadata={
            "selection_split": split,
            "lawsuit_type": family,
            "legal_flux_profile": {
                "template_families": family,
                "reasoning_demands": f"{demand}|supplied_rule_extraction",
                "trajectory_signature": (
                    "case_profile > issue_confirmation > rule_extraction > "
                    f"domain_template:{family} > final_decision"
                ),
            },
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
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _analysis() -> dict:
    return {
        "issue_conclusions": [
            {
                "issue_id": "I1",
                "conclusion": "satisfied",
                "supporting_fact_ids": ["F1"],
                "opposing_fact_ids": ["F2"],
                "explanation": "The transfer supports a repayment obligation.",
            }
        ],
        "final_decision": "support",
        "final_rationale": "The supplied facts support repayment despite the dispute.",
    }


def test_template_pool_validation_and_sanitization():
    thin = _template("LF001", "Thin", "only_one")
    with pytest.raises(ValueError):
        validate_template_pool([thin])

    first = _template("LF001", "Debt support F1 1000", "debt", "support")
    duplicate = _template("LF001", "Duplicate", "debt", "evidence")
    with pytest.raises(ValueError):
        validate_template_pool([first, duplicate])

    cleaned = sanitize_flux_template(first, forbidden_terms=["legalhk-100"])

    assert "support" not in cleaned.template_name.lower()
    assert "F1" not in cleaned.template_name
    assert "1000" not in cleaned.template_name


def test_import_templates_writes_sanitized_pool(tmp_path: Path):
    source = tmp_path / "incoming.jsonl"
    output = tmp_path / "pool.jsonl"
    write_jsonl(
        source,
        [
            _template(
                "LF001",
                "Debt support F1 1000",
                "debt",
                "rule_application",
            ).model_dump(mode="json"),
            _template(
                "LF002",
                "Evidence burden",
                "evidence",
                "burden",
            ).model_dump(mode="json"),
        ],
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    config["legal_flux"] = {
        **config["legal_flux"],
        "template_pool_file": str(output),
    }

    result = import_legal_flux_templates(config, input_path=source)
    templates = load_template_pool(output)

    assert result["templates"] == 2
    assert output.exists()
    assert "support" not in templates[0].template_name.lower()


def test_fixed_plan_uses_pool_and_run_hash_depends_on_template_pool():
    templates = [
        _template("LF001", "Debt obligation", "debt", "contract"),
        _template("LF002", "Evidence burden", "evidence", "burden"),
    ]
    plan = fixed_trajectory_plan(_case(), templates, max_steps=2)

    assert 1 <= len(plan.planned_steps) <= 2
    assert {step.template_id for step in plan.planned_steps}.issubset(
        {"LF001", "LF002"}
    )
    first = flux_run_hash(
        _case(),
        condition="flux_fixed",
        phase="smoke",
        model_digest="model",
        workflow_hash="workflow",
        template_hash=template_pool_hash(templates),
        seed=7,
    )
    second = flux_run_hash(
        _case(),
        condition="flux_fixed",
        phase="smoke",
        model_digest="model",
        workflow_hash="workflow",
        template_hash="different",
        seed=7,
    )
    assert first != second


def test_rf_retrieval_uses_unique_exact_tag_match_before_similarity():
    target = _template(
        "LF001",
        "Employment status",
        "employment_compensation",
        "workplace_safety",
    )
    distractor = _template("LF002", "Contract formation", "contract_formation")
    step = LegalFluxAbstractStep(
        step_id="S1",
        step_name="Work injury employment status",
        template_tags=["workplace safety"],
        purpose="Determine whether employment/workplace rules are triggered.",
    )

    result = retrieve_template_for_abstract_step(step, [target, distractor])

    assert result["template"].template_id == "LF001"
    assert result["retrieval_mode"] == "exact_unique"


def test_rf_retrieval_uses_embedding_for_ambiguous_exact_matches():
    query_text = (
        "Step: Possession and equitable property claim\n"
        "Tags: property\n"
        "Purpose: Decide whether property possession or beneficial ownership controls."
    )
    possession_doc = (
        "Factual Possession of Land property possession "
        "Use Factual Possession of Land for a reusable legal reasoning step. "
        "Cases needing Factual Possession of Land. Identify the legally material "
        "question. Match supplied facts to the selected question."
    )
    trust_doc = (
        "Beneficial Ownership and Trust Formation property trust "
        "Use Beneficial Ownership and Trust Formation for a reusable legal reasoning "
        "step. Cases needing Beneficial Ownership and Trust Formation. Identify the "
        "legally material question. Match supplied facts to the selected question."
    )
    backend = FixedEmbeddingBackend(
        {
            query_text: [1.0, 0.0],
            possession_doc: [0.2, 0.8],
            trust_doc: [0.95, 0.05],
        }
    )
    step = LegalFluxAbstractStep(
        step_id="S1",
        step_name="Possession and equitable property claim",
        template_tags=["property"],
        purpose="Decide whether property possession or beneficial ownership controls.",
    )

    result = retrieve_template_for_abstract_step(
        step,
        [
            _template("LF067", "Factual Possession of Land", "property", "possession"),
            _template(
                "LF076",
                "Beneficial Ownership and Trust Formation",
                "property",
                "trust",
            ),
        ],
        similarity_backend=backend,
    )

    assert result["template"].template_id == "LF076"
    assert result["retrieval_mode"] == "embedding_ambiguous_exact"


def test_rf_exact_retrieval_ignores_generic_step_name_words():
    step = LegalFluxAbstractStep(
        step_id="S1",
        step_name="Property claim control analysis",
        template_tags=["property"],
        purpose="Resolve the property issue.",
    )

    result = retrieve_template_for_abstract_step(
        step,
        [
            _template("LF001", "Generic Claim Review", "procedure"),
            _template("LF002", "Property Possession", "property"),
        ],
    )

    assert result["template"].template_id == "LF002"
    assert result["retrieval_mode"] == "exact_unique"


def test_flux_output_normalizers_repair_common_local_model_shape_errors():
    plan, plan_repairs = _normalize_plan_payload(
        {
            "case_profile": "profile",
            "planned_steps": [
                {
                    "step_id": 1,
                    "template_id": "LF001",
                    "purpose": "Use a template.",
                    "expected_artifact": "Artifact.",
                },
                {
                    "step_3": {
                        "step_id": "3",
                        "template_id": "LF003",
                        "purpose": "Nested wrapped step.",
                        "expected_artifact": "Nested artifact.",
                    }
                }
            ],
            "planning_rationale": "rationale",
            "diagnostic_notes": {"why": "extra plan metadata"},
        },
        max_steps=4,
    )
    artifact, artifact_repairs = _normalize_step_artifact_payload(
        {
            "step_id": "wrong",
            "template_id": "wrong",
            "instantiated_result": "result",
            "material_fact_ids": None,
            "issue_ids": None,
            "confidence": "Medium",
            "needs_revision": None,
            "revision_reason": None,
        },
        LegalFluxPlanStep(
            step_id="S1",
            template_id="LF001",
            purpose="Use a template.",
            expected_artifact="Artifact.",
        ),
    )

    assert plan["planned_steps"][0]["step_id"] == "S1"
    assert plan["planned_steps"][1]["step_id"] == "S3"
    assert "step_3" not in plan["planned_steps"][1]
    assert "Additional structured notes" in plan["planning_rationale"]
    assert "plan_step_id_coerced_to_string" in plan_repairs
    assert "plan_extra_fields_folded_into_planning_rationale" in plan_repairs
    assert artifact["template_id"] == "LF001"
    assert artifact["revision_reason"] == ""
    assert artifact["material_fact_ids"] == []
    assert artifact["confidence"] == "medium"
    assert "revision_reason_null_filled" in artifact_repairs


def test_flux_artifact_and_review_normalizers_repair_nested_list_items():
    artifact, artifact_repairs = _normalize_step_artifact_payload(
        {
            "step_id": "S1",
            "template_id": "LF001",
            "instantiated_result": "result",
            "material_fact_ids": [{"fact_id": "F1", "text": "Material fact."}],
            "issue_ids": [{"id": "I1", "description": "Dispositive issue."}],
            "confidence": "high",
            "needs_revision": False,
            "revision_reason": "",
            "burden_allocation": {"primary_claim_burden": "plaintiff"},
        },
        LegalFluxPlanStep(
            step_id="S1",
            template_id="LF001",
            purpose="Use a template.",
            expected_artifact="Artifact.",
        ),
    )
    review, review_repairs = _normalize_review_payload(
        {
            "decision": "continue",
            "rationale": "Continue with the remaining useful steps.",
            "revised_remaining_steps": [
                {
                    "step_id": 2,
                    "template_id": "LF002",
                    "purpose": "Check evidence.",
                    "expected_artifact": "Evidence artifact.",
                },
                ["step_id"],
                {
                    "step_3": {
                        "step_id": "3",
                        "template_id": "LF003",
                        "purpose": "Nested step.",
                        "expected_artifact": "Nested artifact.",
                    }
                },
            ],
            "trajectory_quality": {"coverage": "partial"},
        }
    )

    assert artifact["material_fact_ids"] == ["F1"]
    assert artifact["issue_ids"] == ["I1"]
    assert "burden_allocation" not in artifact
    assert "Additional structured notes" in artifact["instantiated_result"]
    assert "material_fact_ids_object_unwrapped" in artifact_repairs
    assert "issue_ids_object_unwrapped" in artifact_repairs
    assert "step_extra_fields_folded_into_result" in artifact_repairs
    assert [step["step_id"] for step in review["revised_remaining_steps"]] == [
        "S2",
        "S3",
    ]
    assert "review_invalid_revised_step_removed" in review_repairs
    assert "review_extra_fields_folded_into_rationale" in review_repairs
    assert "Additional structured notes" in review["rationale"]
    assert "step_3" not in review["revised_remaining_steps"][1]


def test_revised_remaining_steps_continue_after_executed_artifacts():
    steps = [
        LegalFluxPlanStep(
            step_id="S1",
            template_id="LF001",
            purpose="First revised step.",
            expected_artifact="Artifact.",
        ),
        LegalFluxPlanStep(
            step_id="S2",
            template_id="LF002",
            purpose="Second revised step.",
            expected_artifact="Artifact.",
        ),
    ]

    renumbered = _renumber_remaining_steps(steps, start_index=3)

    assert [step.step_id for step in renumbered] == ["S3", "S4"]


def test_adaptive_flux_executes_plan_review_and_finalize():
    templates = [
        _template("LF001", "Debt obligation", "debt", "contract"),
        _template("LF002", "Evidence burden", "evidence", "burden"),
    ]
    client = SequenceClient(
        [
            _response(
                {
                    "case_profile": "debt dispute",
                    "planned_steps": [
                        {
                            "step_id": "S1",
                            "template_id": "LF001",
                            "purpose": "Check repayment obligation.",
                            "expected_artifact": "Repayment finding.",
                        },
                        {
                            "step_id": "S2",
                            "template_id": "LF002",
                            "purpose": "Check evidential posture.",
                            "expected_artifact": "Evidence finding.",
                        },
                    ],
                    "planning_rationale": "Debt case with a factual dispute.",
                }
            ),
            _response(
                {
                    "step_id": "S1",
                    "template_id": "LF001",
                    "instantiated_result": "F1 supports an advance; F2 creates a dispute.",
                    "material_fact_ids": ["F1", "F2"],
                    "issue_ids": ["I1"],
                    "confidence": "medium",
                    "needs_revision": False,
                    "revision_reason": "",
                }
            ),
            _response(
                {
                    "decision": "stop",
                    "rationale": "The first artifact is enough for final synthesis.",
                    "revised_remaining_steps": [],
                }
            ),
            _response(_analysis()),
        ]
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")

    analysis, trace = _execute_flux_case(
        client,
        config,
        _case(),
        templates=templates,
        condition="flux_adaptive",
    )

    assert analysis.final_decision == "support"
    assert trace["calls"] == 4
    assert len(trace["executed_steps"]) == 1
    assert trace["trajectory_reviews"][0]["decision"] == "stop"
    assert client.calls[-1]["schema"]["properties"]["final_decision"]["enum"] == [
        "support",
        "reject",
    ]


def test_rf_style_flux_plans_abstract_steps_and_answers_from_review():
    templates = [
        _template(
            "LF025",
            "Test for Summary Disposition or a Genuine Triable Issue",
            "summary_judgment",
            "triable_issue",
        ),
        _template("LF002", "Evidence burden", "evidence", "burden"),
    ]
    client = SequenceClient(
        [
            _response(
                {
                    "case_profile": "summary judgment debt dispute",
                    "planned_steps": [
                        {
                            "step_id": "S1",
                            "step_name": "Summary judgment triable issue screen",
                            "template_tags": ["summary_judgment", "triable_issue"],
                            "purpose": "Check whether the defense raises a genuine triable issue.",
                        }
                    ],
                    "planning_rationale": "The procedural threshold controls the outcome.",
                }
            ),
            _response(
                {
                    "step_id": "S1",
                    "template_id": "LF025",
                    "instantiated_result": "F1 supports the debt and F2 does not create a genuine triable issue.",
                    "material_fact_ids": ["F1", "F2"],
                    "issue_ids": ["I1"],
                    "confidence": "high",
                    "needs_revision": False,
                    "revision_reason": "",
                }
            ),
            _response(
                {
                    "decision": "final_answer",
                    "rationale": "The executed artifact is enough to decide the case.",
                    "revised_remaining_steps": [],
                    "final_decision": "support",
                    "final_rationale": "The debt claim is supported because no genuine triable issue remains.",
                }
            ),
        ]
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")

    analysis, trace = _execute_flux_case(
        client,
        config,
        _case(),
        templates=templates,
        condition="flux_rf_style",
    )

    assert analysis.final_decision == "support"
    assert trace["calls"] == 3
    assert trace["retrieved_template_ids"] == ["LF025"]
    assert trace["trajectory_reviews"][0]["decision"] == "final_answer"
    assert "TEMPLATE CATALOG" not in client.prompts[0]
    assert "finalize" not in trace["prompt_hashes"]


def test_rf_style_flux_requests_final_answer_when_steps_are_exhausted():
    templates = [
        _template("LF001", "Debt obligation", "debt_payment", "burden_of_proof"),
    ]
    client = SequenceClient(
        [
            _response(
                {
                    "case_profile": "debt dispute",
                    "planned_steps": [
                        {
                            "step_id": "S1",
                            "step_name": "Debt entitlement check",
                            "template_tags": ["debt_payment"],
                            "purpose": "Check whether repayment is due.",
                        }
                    ],
                    "planning_rationale": "Debt entitlement is dispositive.",
                }
            ),
            _response(
                {
                    "step_id": "S1",
                    "template_id": "LF001",
                    "instantiated_result": "F1 supports repayment and F2 does not defeat it.",
                    "material_fact_ids": ["F1", "F2"],
                    "issue_ids": ["I1"],
                    "confidence": "high",
                    "needs_revision": False,
                    "revision_reason": "",
                }
            ),
            _response(
                {
                    "decision": "continue",
                    "rationale": "Continue.",
                    "revised_remaining_steps": [],
                    "final_decision": None,
                    "final_rationale": "",
                }
            ),
            _response(
                {
                    "decision": "final_answer",
                    "rationale": "No remaining step is needed.",
                    "revised_remaining_steps": [],
                    "final_decision": "support",
                    "final_rationale": "The repayment claim is supported by the executed artifact.",
                }
            ),
        ]
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")

    analysis, trace = _execute_flux_case(
        client,
        config,
        _case(),
        templates=templates,
        condition="flux_rf_style",
    )

    assert analysis.final_decision == "support"
    assert trace["calls"] == 4
    assert "No remaining abstract steps are available" in client.prompts[-1]
    assert client.calls[-1]["schema"]["properties"]["decision"]["enum"] == [
        "final_answer"
    ]


def test_no_review_adaptive_flux_executes_initial_plan_without_reviews():
    templates = [
        _template("LF001", "Debt obligation", "debt", "contract"),
        _template("LF002", "Evidence burden", "evidence", "burden"),
    ]
    client = SequenceClient(
        [
            _response(
                {
                    "case_profile": "debt dispute",
                    "planned_steps": [
                        {
                            "step_id": "S1",
                            "template_id": "LF001",
                            "purpose": "Check repayment obligation.",
                            "expected_artifact": "Repayment finding.",
                        },
                        {
                            "step_id": "S2",
                            "template_id": "LF002",
                            "purpose": "Check evidential posture.",
                            "expected_artifact": "Evidence finding.",
                        },
                    ],
                    "planning_rationale": "Debt case with a factual dispute.",
                }
            ),
            _response(
                {
                    "step_id": "S1",
                    "template_id": "LF001",
                    "instantiated_result": "F1 supports an advance; F2 creates a dispute.",
                    "material_fact_ids": ["F1", "F2"],
                    "issue_ids": ["I1"],
                    "confidence": "medium",
                    "needs_revision": False,
                    "revision_reason": "",
                }
            ),
            _response(
                {
                    "step_id": "S2",
                    "template_id": "LF002",
                    "instantiated_result": "The evidence remains disputed but usable.",
                    "material_fact_ids": ["F1", "F2"],
                    "issue_ids": ["I1"],
                    "confidence": "medium",
                    "needs_revision": False,
                    "revision_reason": "",
                }
            ),
            _response(_analysis()),
        ]
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")

    analysis, trace = _execute_flux_case(
        client,
        config,
        _case(),
        templates=templates,
        condition="flux_adaptive_no_review",
    )

    assert analysis.final_decision == "support"
    assert trace["calls"] == 4
    assert len(trace["executed_steps"]) == 2
    assert trace["trajectory_reviews"] == []
    assert not any("CURRENT REMAINING STEPS:" in prompt for prompt in client.prompts)


def test_chatgpt_batch_export_writes_clustered_workflow(tmp_path: Path):
    processed = tmp_path / "data" / "processed" / "legal_flux"
    cases = [
        *[
            _profiled_case(
                index,
                family="contract_performance",
                demand="defense_or_counterargument_check",
            )
            for index in range(1, 7)
        ],
        *[
            _profiled_case(
                index,
                family="procedure_appeal",
                demand="procedural_threshold_check",
            )
            for index in range(7, 13)
        ],
        _profiled_case(
            999,
            family="debt_payment",
            demand="focused_issue_resolution",
            split="final_test",
        ),
    ]
    write_jsonl(processed / "cases.jsonl", [case.model_dump(mode="json") for case in cases])
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    config["_project_root"] = str(tmp_path)
    config["paths"] = {
        **config["paths"],
        "processed_dir": "data/processed/legal_flux",
    }
    config["legal_flux"] = {
        **config["legal_flux"],
        "chatgpt_batch_dir": str(
            tmp_path
            / "reports"
            / "legal_flux"
            / "template_distillation"
            / "chatgpt_batches"
        ),
        "chatgpt_homogeneous_batches": 2,
        "chatgpt_mixed_batches": 1,
        "chatgpt_cases_per_batch": 4,
    }

    result = export_legal_flux_chatgpt_batches(config)
    manifest = tmp_path / "reports" / "legal_flux" / "template_distillation" / "chatgpt_batches" / "batch_manifest.json"
    rows = read_jsonl(
        next((tmp_path / "reports" / "legal_flux" / "template_distillation" / "chatgpt_batches" / "01_homogeneous_batches").glob("*.jsonl"))
    )

    assert result["homogeneous_batches"] == 2
    assert result["mixed_batches"] == 1
    assert manifest.exists()
    assert "gold_answer" not in rows[0]
    assert rows[0]["case_id"] != "legalhk-999"
    assert (manifest.parent / "prompts" / "02_merge_deduplicate_templates.md").exists()
    assert (manifest.parent / "prompts" / "03_coverage_audit_and_gap_fill.md").exists()


def test_cli_exposes_legal_flux_commands():
    parser = build_parser()

    assert parser.parse_args(
        ["--config", "configs/legal_flux.yaml", "flux-prepare"]
    ).command == "flux-prepare"
    assert parser.parse_args(
        ["--config", "configs/legal_flux.yaml", "flux-export-chatgpt-batches"]
    ).command == "flux-export-chatgpt-batches"
    assert parser.parse_args(
        ["--config", "configs/legal_flux.yaml", "flux-smoke", "--dry-run"]
    ).dry_run
    args = parser.parse_args(
        [
            "--config",
            "configs/legal_flux.yaml",
            "flux-generate",
            "--phase",
            "final-test",
            "--dry-run",
        ]
    )
    assert args.phase == "final-test"
    assert args.dry_run
