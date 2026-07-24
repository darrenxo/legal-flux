from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_pilot.__main__ import build_parser
from legal_pilot.adaptive_profiles import profile_row
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
from legal_pilot.legal_flux_deepseek import run_deepseek_template_workflow
from legal_pilot.legal_flux_gemini import run_gemini_template_workflow
from legal_pilot.legal_flux_runner import (
    _execute_rf_style_case,
    _normalize_rf_review_payload,
    _normalize_step_artifact_payload,
    flux_run_hash,
)
from legal_pilot.legal_flux_setup import import_legal_flux_templates
from legal_pilot.legal_flux_training import (
    export_template_structure_sft,
    export_trajectory_dpo,
)
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


def test_adaptive_profile_regexes_match_legal_word_families():
    profile = profile_row(
        {
            "plaintiff_claim": (
                "The plaintiff alleges repudiation of the agreement, tenant "
                "possession issues, injuries, liquidation, insolvency, and "
                "credible defenses requiring discretionary declarations. The "
                "case also mentions criminal charges, a non-refoulement claim, "
                "and judicial review of a tribunal decision."
            ),
            "lawsuit_type": "Contract and insolvency",
            "more_facts": "The defendant repudiated the lease and raised defences.",
            "issues": "Whether the repudiatory breach caused injuries.",
            "related_laws": "",
            "relevant_cases": "",
        }
    )
    families = set(profile["template_families"].split("|"))
    demands = set(profile["reasoning_demands"].split("|"))

    assert "contract_performance" in families
    assert "property_possession" in families
    assert "tort_negligence_damage" in families
    assert "company_insolvency" in families
    assert "criminal_procedure" in families
    assert "immigration_non_refoulement" in families
    assert "public_law_judicial_review" in families
    assert "evidence_and_burden_assessment" in demands
    assert "defense_or_counterargument_check" in demands
    assert "remedy_discretion_check" in demands


class SequenceClient:
    def __init__(self, responses: list[ModelResponse]):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeTemplateApiClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> dict:
        self.messages.append(messages)
        return {
            "content": self.responses.pop(0),
            "metadata": {"response_id": f"fake-{len(self.messages)}"},
        }


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
    legal_number = sanitize_flux_template(
        _template(
            "LF003",
            "Summary Judgment under Order 14",
            "order_14",
            "100%",
        )
    )

    assert result["templates"] == 2
    assert output.exists()
    assert "support" not in cleaned.template_name.lower()
    assert "F1" not in cleaned.template_name
    assert legal_number.template_name == "Summary Judgment under Order 14"
    assert legal_number.knowledge_tags[0] == "order_14"
    assert legal_number.knowledge_tags[1] == "case-specific value"


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
    assert "legal_flux_profile" not in rows[0]
    assert rows[0]["case_id"] != "legalhk-999"
    assert (manifest.parent / "prompts" / "02_merge_deduplicate_templates.md").exists()


def test_deepseek_template_workflow_generates_candidates_merge_and_audit(tmp_path: Path):
    batch_root = tmp_path / "batches"
    batch_dir = batch_root / "01_homogeneous_batches"
    prompts_dir = batch_root / "prompts"
    batch_dir.mkdir(parents=True)
    prompts_dir.mkdir(parents=True)
    batch_path = batch_dir / "homogeneous_001_contract.jsonl"
    write_jsonl(
        batch_path,
        [
            {
                "case_id": "legalhk-1",
                "claim": "Repayment under a disputed agreement.",
                "facts": {"F1": "The defendant received money."},
            }
        ],
    )
    schema_path = batch_root / "legal_flux_template.schema.json"
    schema_path.write_text(
        json.dumps(LegalFluxTemplate.model_json_schema(), ensure_ascii=False),
        encoding="utf-8",
    )
    (batch_root / "coverage_summary.json").write_text("{}", encoding="utf-8")
    (batch_root / "batch_manifest.json").write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "batch_id": "homogeneous_001",
                        "kind": "homogeneous",
                        "label": "contract",
                        "path": str(batch_path),
                        "case_count": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (prompts_dir / "01_generate_candidate_templates.md").write_text(
        "Generate candidate templates.", encoding="utf-8"
    )
    (prompts_dir / "02_merge_deduplicate_templates.md").write_text(
        "Merge candidate templates.", encoding="utf-8"
    )
    (prompts_dir / "03_coverage_audit_and_gap_fill.md").write_text(
        "Audit coverage.", encoding="utf-8"
    )
    candidate = _template(
        "CAND_homogeneous_001_01",
        "Agreement Entitlement Check",
        "contract",
        "entitlement",
    ).model_dump(mode="json")
    merged = [
        _template("LF001", "Agreement Entitlement Check", "contract", "entitlement").model_dump(mode="json"),
        _template("LF002", "Evidence Burden Check", "evidence", "burden").model_dump(mode="json"),
    ]
    client = FakeTemplateApiClient(
        [
            json.dumps(candidate, ensure_ascii=False),
            "\n".join(json.dumps(row, ensure_ascii=False) for row in merged),
            "No important gaps remain.",
        ]
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    config["legal_flux"] = {
        **config["legal_flux"],
        "chatgpt_batch_dir": str(batch_root),
        "deepseek_template_dir": str(tmp_path / "deepseek_api"),
    }

    result = run_deepseek_template_workflow(
        config,
        stage="all",
        force=True,
        client=client,
    )
    output_root = tmp_path / "deepseek_api"

    assert result["candidates"]["records"][0]["template_count"] == 1
    assert result["merge"]["template_count"] == 2
    assert result["audit"]["gap_fill_count"] == 0
    assert (output_root / "03_candidate_templates" / "homogeneous_001_candidates.jsonl").exists()
    assert (output_root / "legal_flux_templates_deepseek_merged.jsonl").exists()
    assert "BATCH_ID" in client.messages[0][-1]["content"]


def test_gemini_template_workflow_generates_candidates_merge_and_audit(tmp_path: Path):
    batch_root = tmp_path / "batches"
    batch_dir = batch_root / "01_homogeneous_batches"
    prompts_dir = batch_root / "prompts"
    batch_dir.mkdir(parents=True)
    prompts_dir.mkdir(parents=True)
    batch_path = batch_dir / "homogeneous_001_contract.jsonl"
    write_jsonl(
        batch_path,
        [
            {
                "case_id": "legalhk-1",
                "claim": "Repayment under a disputed agreement.",
                "facts": {"F1": "The defendant received money."},
            }
        ],
    )
    schema_path = batch_root / "legal_flux_template.schema.json"
    schema_path.write_text(
        json.dumps(LegalFluxTemplate.model_json_schema(), ensure_ascii=False),
        encoding="utf-8",
    )
    (batch_root / "coverage_summary.json").write_text("{}", encoding="utf-8")
    (batch_root / "batch_manifest.json").write_text(
        json.dumps(
            {
                "batches": [
                    {
                        "batch_id": "homogeneous_001",
                        "kind": "homogeneous",
                        "label": "contract",
                        "path": str(batch_path),
                        "case_count": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (prompts_dir / "01_generate_candidate_templates.md").write_text(
        "Generate candidate templates.", encoding="utf-8"
    )
    (prompts_dir / "02_merge_deduplicate_templates.md").write_text(
        "Merge candidate templates.", encoding="utf-8"
    )
    (prompts_dir / "03_coverage_audit_and_gap_fill.md").write_text(
        "Audit coverage.", encoding="utf-8"
    )
    candidate = _template(
        "CAND_homogeneous_001_01",
        "Agreement Entitlement Check",
        "contract",
        "entitlement",
    ).model_dump(mode="json")
    merged = [
        _template("LF001", "Agreement Entitlement Check", "contract", "entitlement").model_dump(mode="json"),
        _template("LF002", "Evidence Burden Check", "evidence", "burden").model_dump(mode="json"),
    ]
    client = FakeTemplateApiClient(
        [
            json.dumps(candidate, ensure_ascii=False),
            "\n".join(json.dumps(row, ensure_ascii=False) for row in merged),
            "No important gaps remain.",
        ]
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    config["legal_flux"] = {
        **config["legal_flux"],
        "chatgpt_batch_dir": str(batch_root),
        "gemini_template_dir": str(tmp_path / "gemini_api"),
    }

    result = run_gemini_template_workflow(
        config,
        stage="all",
        force=True,
        client=client,
    )
    output_root = tmp_path / "gemini_api"

    assert result["candidates"]["records"][0]["template_count"] == 1
    assert result["merge"]["template_count"] == 2
    assert result["audit"]["gap_fill_count"] == 0
    assert (
        output_root / "03_candidate_templates" / "homogeneous_001_candidates.jsonl"
    ).exists()
    assert (output_root / "legal_flux_templates_gemini_merged.jsonl").exists()
    assert "BATCH_ID" in client.messages[0][-1]["content"]


def test_template_structure_sft_export_uses_template_name_and_tags(tmp_path: Path):
    pool = tmp_path / "templates.jsonl"
    write_jsonl(
        pool,
        [
            _template("LF001", "Debt entitlement", "debt", "rule_application").model_dump(mode="json"),
            _template("LF002", "Procedural gate", "procedure", "threshold").model_dump(mode="json"),
        ],
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    config["_project_root"] = str(tmp_path)
    config["paths"] = {
        **config["paths"],
        "processed_dir": str(tmp_path / "processed"),
    }
    config["legal_flux"] = {
        **config["legal_flux"],
        "template_pool_file": str(pool),
    }

    result = export_template_structure_sft(config)
    rows = read_jsonl(Path(result["output_path"]))

    assert result["templates"] == 2
    assert rows[0]["task"] == "template_structure_sft"
    assert "Template name: Debt entitlement" in rows[0]["messages"][1]["content"]
    assert "application_scenario" in rows[0]["messages"][2]["content"]


def test_trajectory_dpo_export_pairs_planner_train_samples(tmp_path: Path):
    pool = tmp_path / "templates.jsonl"
    processed = tmp_path / "processed"
    runs = tmp_path / "runs"
    write_jsonl(
        pool,
        [
            _template("LF001", "Debt entitlement", "debt", "rule_application").model_dump(mode="json"),
            _template("LF002", "Procedural gate", "procedure", "threshold").model_dump(mode="json"),
        ],
    )
    case = _case(split="planner_train")
    write_jsonl(processed / "cases.jsonl", [case.model_dump(mode="json")])
    plan_good = {
        "case_profile": "good debt profile",
        "planned_steps": [
            {
                "step_id": "S1",
                "step_name": "Debt entitlement",
                "template_tags": ["debt", "rule_application"],
                "purpose": "Resolve whether repayment is due.",
            }
        ],
        "planning_rationale": "The entitlement step is dispositive.",
    }
    plan_bad = {
        "case_profile": "bad unrelated profile",
        "planned_steps": [
            {
                "step_id": "S1",
                "step_name": "Procedural gate",
                "template_tags": ["procedure", "threshold"],
                "purpose": "Screen for a procedural issue not shown by the facts.",
            }
        ],
        "planning_rationale": "This over-focuses on procedure.",
    }
    write_jsonl(
        runs / "planner_train" / "scored.jsonl",
        [
            {
                "run_hash": "chosen",
                "status": "ok",
                "condition": "flux_rf_style",
                "dataset": case.dataset,
                "case_id": case.case_id,
                "variant_id": case.variant_id,
                "sample_index": 0,
                "trajectory_plan": plan_good,
                "executed_steps": [{"step_id": "S1"}],
                "retrieved_template_ids": ["LF001"],
                "trajectory_length": 1,
                "schema_errors": [],
                "answer_correct": True,
                "binary_prediction_valid": True,
            },
            {
                "run_hash": "rejected",
                "status": "ok",
                "condition": "flux_rf_style",
                "dataset": case.dataset,
                "case_id": case.case_id,
                "variant_id": case.variant_id,
                "sample_index": 1,
                "trajectory_plan": plan_bad,
                "executed_steps": [{"step_id": "S1"}, {"step_id": "S2"}],
                "retrieved_template_ids": [],
                "trajectory_length": 2,
                "schema_errors": ["schema repair needed"],
                "answer_correct": False,
                "binary_prediction_valid": True,
            },
        ],
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    config["_project_root"] = str(tmp_path)
    config["paths"] = {
        **config["paths"],
        "processed_dir": str(processed),
        "runs_dir": str(runs),
        "prompts_dir": str(Path(__file__).parents[1] / "prompts"),
    }
    config["legal_flux"] = {
        **config["legal_flux"],
        "template_pool_file": str(pool),
        "max_steps": 4,
    }

    result = export_trajectory_dpo(config)
    rows = read_jsonl(Path(result["output_path"]))

    assert result["pairs"] == 1
    assert "good debt profile" in rows[0]["chosen"]
    assert "bad unrelated profile" in rows[0]["rejected"]
    assert rows[0]["chosen_reward"] > rows[0]["rejected_reward"]


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
    train_args = parser.parse_args(
        [
            "--config",
            "configs/legal_flux.yaml",
            "flux-generate",
            "--phase",
            "planner-train",
            "--samples",
            "3",
        ]
    )
    assert train_args.phase == "planner-train"
    assert train_args.samples == 3
    assert parser.parse_args(
        ["--config", "configs/legal_flux.yaml", "flux-export-template-sft"]
    ).command == "flux-export-template-sft"
    assert parser.parse_args(
        ["--config", "configs/legal_flux.yaml", "flux-export-template-batches"]
    ).command == "flux-export-template-batches"
    assert parser.parse_args(
        ["--config", "configs/legal_flux.yaml", "flux-export-trajectory-dpo"]
    ).command == "flux-export-trajectory-dpo"
    gemini_args = parser.parse_args(
        [
            "--config",
            "configs/legal_flux.yaml",
            "flux-gemini-templates",
            "--stage",
            "all",
            "--limit",
            "2",
            "--dry-run",
        ]
    )
    assert gemini_args.command == "flux-gemini-templates"
    assert gemini_args.stage == "all"
    assert gemini_args.limit == 2
    assert gemini_args.dry_run
    deepseek_args = parser.parse_args(
        [
            "--config",
            "configs/legal_flux.yaml",
            "flux-deepseek-templates",
            "--stage",
            "all",
            "--limit",
            "2",
            "--dry-run",
        ]
    )
    assert deepseek_args.command == "flux-deepseek-templates"
    assert deepseek_args.stage == "all"
    assert deepseek_args.limit == 2
    assert deepseek_args.dry_run
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
