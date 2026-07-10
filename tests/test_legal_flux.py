from __future__ import annotations

from pathlib import Path

import pytest

from legal_pilot.__main__ import build_parser
from legal_pilot.clients import ModelResponse
from legal_pilot.config import load_config
from legal_pilot.embeddings import FixedEmbeddingBackend
from legal_pilot.io_utils import read_jsonl, write_jsonl
from legal_pilot.legal_flux import (
    legal_flux_workflow_components,
    load_template_pool,
    retrieve_template_for_abstract_step,
    sanitize_flux_template,
    template_pool_hash,
    validate_template_pool,
)
from legal_pilot.legal_flux_chatgpt import export_legal_flux_chatgpt_batches
from legal_pilot.legal_flux_runner import (
    _execute_rf_style_case,
    _normalize_rf_review_payload,
    _normalize_step_artifact_payload,
    flux_run_hash,
)
from legal_pilot.legal_flux_setup import import_legal_flux_templates
from legal_pilot.models import (
    LegalFluxAbstractStep,
    LegalFluxPlanStep,
    LegalFluxTemplate,
    NormalizedCase,
)


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


def test_template_pool_validation_sanitization_and_import(tmp_path: Path):
    thin = _template("LF001", "Thin", "only_one")
    with pytest.raises(ValueError):
        validate_template_pool([thin])

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
    config["legal_flux"] = {**config["legal_flux"], "template_pool_file": str(output)}

    result = import_legal_flux_templates(config, input_path=source)
    templates = load_template_pool(output)
    cleaned = sanitize_flux_template(templates[0])

    assert result["templates"] == 2
    assert output.exists()
    assert "support" not in cleaned.template_name.lower()
    assert "F1" not in cleaned.template_name


def test_rf_retrieval_uses_exact_terms_then_embedding_and_excludes_repeats():
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
        {query_text: [1.0, 0.0], possession_doc: [0.2, 0.8], trust_doc: [0.95, 0.05]}
    )
    ambiguous = LegalFluxAbstractStep(
        step_id="S1",
        step_name="Possession and equitable property claim",
        template_tags=["property"],
        purpose="Decide whether property possession or beneficial ownership controls.",
    )

    result = retrieve_template_for_abstract_step(
        ambiguous,
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

    repeated = retrieve_template_for_abstract_step(
        LegalFluxAbstractStep(
            step_id="S2",
            step_name="Debt repayment check",
            template_tags=["debt_payment"],
            purpose="Do not repeat the already selected debt template.",
        ),
        [
            _template("LF001", "Debt entitlement", "debt_payment"),
            _template("LF002", "Payment maturity", "debt_payment"),
        ],
        exclude_template_ids={"LF001"},
    )
    assert repeated["template"].template_id == "LF002"


def test_rf_style_flux_plans_retrieves_executes_and_answers_from_review():
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
    case = _case().model_copy(
        update={
            "authorities": "Related law: Contract principle supplied for this task.",
            "metadata": {
                "selection_split": "smoke",
                "lawsuit_type": "Debt",
                "relevant_cases": "Relevant case: Example v Debtor.",
                "legal_flux_profile": {
                    "template_families": "HEURISTIC_FAMILY_SHOULD_NOT_LEAK",
                    "reasoning_demands": "HEURISTIC_DEMAND_SHOULD_NOT_LEAK",
                    "trajectory_signature": "HEURISTIC_TRAJECTORY_SHOULD_NOT_LEAK",
                },
            },
        }
    )

    analysis, trace = _execute_rf_style_case(client, config, case, templates=templates)

    planner_prompt = client.prompts[0]
    assert analysis.final_decision == "support"
    assert trace["calls"] == 3
    assert trace["retrieved_template_ids"] == ["LF025"]
    assert trace["selected_templates"][0]["retrieval_mode"] == "exact_unique"
    assert "TEMPLATE CATALOG" not in planner_prompt
    assert "TEMPLATE TAG EXAMPLES" in planner_prompt
    assert "AUTHORITY CONTEXT" in planner_prompt
    assert "HEURISTIC_FAMILY_SHOULD_NOT_LEAK" not in planner_prompt


def test_rf_style_forces_final_answer_after_exhausting_steps():
    templates = [_template("LF001", "Debt obligation", "debt_payment", "burden")]
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
                    "final_rationale": "The repayment claim is supported.",
                }
            ),
        ]
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")

    analysis, trace = _execute_rf_style_case(client, config, _case(), templates=templates)

    assert analysis.final_decision == "support"
    assert trace["calls"] == 4
    assert "No remaining abstract steps are available" in client.prompts[-1]
    assert client.calls[-1]["schema"]["properties"]["decision"]["enum"] == [
        "final_answer"
    ]


def test_output_normalizers_repair_common_rf_shapes():
    artifact, artifact_repairs = _normalize_step_artifact_payload(
        {
            "step_id": "wrong",
            "template_id": "wrong",
            "instantiated_result": "result",
            "material_fact_ids": [{"fact_id": "F1", "text": "Material fact."}],
            "issue_ids": [{"id": "I1", "description": "Dispositive issue."}],
            "confidence": "Medium",
            "needs_revision": None,
            "revision_reason": None,
            "burden_allocation": {"primary_claim_burden": "plaintiff"},
        },
        LegalFluxPlanStep(
            step_id="S1",
            template_id="LF001",
            purpose="Use a template.",
            expected_artifact="Artifact.",
        ),
    )
    review, review_repairs = _normalize_rf_review_payload(
        {
            "decision": "FINAL_ANSWER",
            "rationale": "Done.",
            "revised_remaining_steps": None,
            "final_decision": "Support",
            "final_rationale": None,
            "extra": {"note": "fold me"},
        }
    )

    assert artifact["step_id"] == "S1"
    assert artifact["template_id"] == "LF001"
    assert artifact["material_fact_ids"] == ["F1"]
    assert artifact["issue_ids"] == ["I1"]
    assert artifact["confidence"] == "medium"
    assert "Additional structured notes" in artifact["instantiated_result"]
    assert "material_fact_ids_object_unwrapped" in artifact_repairs
    assert review["decision"] == "final_answer"
    assert review["final_decision"] == "support"
    assert "rf_review_extra_fields_folded_into_rationale" in review_repairs


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
    config["paths"] = {**config["paths"], "processed_dir": "data/processed/legal_flux"}
    config["legal_flux"] = {
        **config["legal_flux"],
        "chatgpt_batch_dir": str(
            tmp_path / "reports" / "legal_flux" / "template_distillation" / "chatgpt_batches"
        ),
        "chatgpt_homogeneous_batches": 2,
        "chatgpt_mixed_batches": 1,
        "chatgpt_cases_per_batch": 4,
    }

    result = export_legal_flux_chatgpt_batches(config)
    manifest = (
        tmp_path
        / "reports"
        / "legal_flux"
        / "template_distillation"
        / "chatgpt_batches"
        / "batch_manifest.json"
    )
    rows = read_jsonl(
        next(
            (
                tmp_path
                / "reports"
                / "legal_flux"
                / "template_distillation"
                / "chatgpt_batches"
                / "01_homogeneous_batches"
            ).glob("*.jsonl")
        )
    )

    assert result["homogeneous_batches"] == 2
    assert result["mixed_batches"] == 1
    assert manifest.exists()
    assert "gold_answer" not in rows[0]
    assert rows[0]["case_id"] != "legalhk-999"
    assert (manifest.parent / "prompts" / "02_merge_deduplicate_templates.md").exists()


def test_cli_and_workflow_hash_only_expose_current_legal_flux_surface():
    parser = build_parser()

    assert parser.parse_args(
        ["--config", "configs/legal_flux.yaml", "flux-prepare"]
    ).command == "flux-prepare"
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
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "configs/legal_flux.yaml", "unknown-command"])

    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    components = legal_flux_workflow_components(config)

    assert "legal_flux/rf_plan.txt" in components["prompts"]
    assert "legal_flux/plan.txt" not in components["prompts"]
    assert "legal_flux_abstract_plan.json" in components["schemas"]
    assert "legal_flux_trajectory_plan.json" not in components["schemas"]
    assert flux_run_hash(
        _case(),
        condition="flux_rf_style",
        phase="smoke",
        model_digest="model",
        workflow_hash="workflow",
        template_hash=template_pool_hash([_template("LF001", "Debt", "debt", "rule")]),
        seed=7,
    )
