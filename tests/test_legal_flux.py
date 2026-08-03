from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
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
from legal_pilot.legal_flux_dpo import build_dpo_data
from legal_pilot.legal_flux_gemini import run_gemini_template_workflow
from legal_pilot.legal_flux_evaluation import _aggregate_frame
from legal_pilot.legal_flux_runner import (
    _execute_rf_style_case,
    _normalize_rf_review_payload,
    _normalize_step_artifact_payload,
    _print_generation_progress,
    _select_generation_shard,
    flux_run_hash,
)
from legal_pilot.legal_flux_setup import import_legal_flux_templates
from legal_pilot.legal_flux_sft import (
    _load_text_tokenizer,
    _template_lora_config_kwargs,
    _template_sft_config_kwargs,
    _validate_constructor_kwargs,
    export_trajectory_dev_tune_subset,
    summarize_sft_checkpoint_grid,
    template_sft_settings,
    train_template_structure_sft,
)
from legal_pilot.legal_flux_training import (
    _dpo_pair_from_xsim_group,
    export_template_structure_sft,
    export_trajectory_dpo,
)
from legal_pilot.legal_flux_xsim import build_xsim, load_xsim_neighbors
from legal_pilot.models import (
    LegalFluxAbstractStep,
    LegalFluxPlanStep,
    LegalFluxTemplate,
    NormalizedCase,
)
from legal_pilot.runner import _execute_condition


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


def test_generation_progress_is_periodic_and_flushable(capsys: pytest.CaptureFixture[str]):
    _print_generation_progress(completed=24, skipped=0, errors=0, total=100)
    assert capsys.readouterr().out == ""

    _print_generation_progress(completed=25, skipped=0, errors=1, total=100)
    assert "25/100 jobs" in capsys.readouterr().out

    _print_generation_progress(completed=99, skipped=1, errors=1, total=100)
    assert "100/100 jobs" in capsys.readouterr().out


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


def test_structured_baseline_uses_concise_irac_then_decision_contract():
    client = SequenceClient(
        [
            _response(
                {
                    "irac_reasoning": (
                        "Issue: whether repayment is due. Rule: an enforceable debt "
                        "supports repayment. Application: F1 supports the debt and F2 "
                        "does not defeat it. Conclusion: the claim is supported."
                    ),
                    "final_decision": "support",
                }
            )
        ]
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")

    analysis, _ = _execute_condition(
        client,
        config,
        _case(),
        "structured",
        temperature=0.0,
        seed=1,
    )

    assert analysis.final_decision == "support"
    assert analysis.irac_reasoning.startswith("Issue:")
    assert analysis.final_rationale == ""
    assert client.calls[0]["schema"]["required"] == [
        "irac_reasoning",
        "final_decision",
    ]
    assert set(client.calls[0]["schema"]["properties"]) == {
        "irac_reasoning",
        "final_decision",
    }
    assert client.calls[0]["max_tokens"] == 1600
    assert "Keep\nirac_reasoning concise" in client.prompts[0]


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


class FakeDenseEncoder:
    model_name = "fake-bge-m3"

    def encode(self, texts: list[str], *, batch_size: int):
        vectors = {
            "Case 0": [1.0, 0.0, 0.0],
            "Case 1": [0.9, 0.1, 0.0],
            "Case 2": [0.8, 0.2, 0.0],
            "Case 3": [0.7, 0.3, 0.0],
        }
        return [
            next(vector for marker, vector in vectors.items() if marker in text)
            for text in texts
        ]


class FakeReranker:
    model_name = "fake-reranker"

    def score(self, pairs: list[tuple[str, str]], *, batch_size: int):
        candidate_scores = {
            "Case 1": 0.7,
            "Case 2": 0.2,
            "Case 3": 0.9,
        }
        return [
            next(
                score
                for marker, score in candidate_scores.items()
                if marker in candidate
            )
            for _, candidate in pairs
        ]


class FakeDpoPipelineClient:
    def __init__(self, *, invalid_profiles: set[str] | None = None):
        self.plan_index = 0
        self.calls: list[dict] = []
        self.invalid_profiles = invalid_profiles or set()

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        title = kwargs["schema"]["title"]
        if title == "LegalFluxAbstractPlan":
            index = self.plan_index
            self.plan_index += 1
            return _response(
                {
                    "planning_analysis": f"profile {index}; rationale {index}",
                    "planned_steps": [
                        {
                            "step_id": "S1",
                            "step_name": "Debt entitlement",
                            "step_description": f"Resolve debt variant {index}.",
                            "template_tags": ["debt", "rule_application"],
                        }
                    ],
                }
            )
        if title == "LegalFluxStepArtifact":
            return _response(
                {
                    "step_id": "S1",
                    "template_id": "LF001",
                    "instantiated_result": "The supplied facts resolve the claim.",
                    "material_fact_ids": ["F1"],
                    "issue_ids": [],
                    "confidence": "high",
                    "needs_revision": False,
                    "revision_reason": "",
                }
            )
        if title == "LegalFluxRfFinalReview":
            prompt = kwargs["prompt"]
            target_is_reject = "Claim for case 101" in prompt
            invalid_profile = next(
                (profile for profile in self.invalid_profiles if profile in prompt),
                None,
            )
            if invalid_profile is not None:
                decision = "undetermined"
            elif "profile 0" in prompt:
                decision = "reject" if target_is_reject else "support"
            elif "profile 1" in prompt:
                decision = "support" if target_is_reject else "reject"
            elif "profile 2" in prompt:
                decision = "support"
            else:
                decision = "reject"
            return _response(
                {
                    "final_rationale": "The artifacts support this binary result.",
                    "final_decision": decision,
                }
            )
        raise AssertionError(title)


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
        step_description="Determine whether employment/workplace rules are triggered.",
        template_tags=["workplace safety"],
    )

    result = retrieve_template_for_abstract_step(step, [target, distractor])

    assert result["template"].template_id == "LF001"
    assert result["retrieval_mode"] == "exact_tag_unique"

    exact_name = retrieve_template_for_abstract_step(
        LegalFluxAbstractStep(
            step_id="S1",
            step_name="Employment status",
            step_description="Resolve the employment relationship.",
            template_tags=["contract_formation"],
        ),
        [target, distractor],
    )
    assert exact_name["template"].template_id == "LF001"
    assert exact_name["retrieval_mode"] == "exact_name"

    query_text = (
        "Step: Possession and equitable property claim\n"
        "Description: Decide whether property possession or beneficial ownership controls.\n"
        "Tags: property"
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
        step_description="Decide whether property possession or beneficial ownership controls.",
        template_tags=["property"],
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
    assert result["retrieval_mode"] == "embedding_tag_overlap"

    repeated = retrieve_template_for_abstract_step(
        LegalFluxAbstractStep(
            step_id="S2",
            step_name="Debt repayment check",
            step_description="Do not repeat the already selected debt template.",
            template_tags=["debt_payment"],
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
                    "planning_analysis": (
                        "A summary-judgment debt dispute turns on the procedural threshold."
                    ),
                    "planned_steps": [
                        {
                            "step_id": "S1",
                            "step_name": "Summary judgment triable issue screen",
                            "step_description": "Check whether the defense raises a genuine triable issue.",
                            "template_tags": ["summary_judgment", "triable_issue"],
                        },
                        {
                            "step_id": "S2",
                            "step_name": "Evidence burden check",
                            "step_description": "Resolve any evidentiary gap if one remains.",
                            "template_tags": ["evidence", "burden"],
                        }
                    ],
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
                    "review_analysis": "The first artifact is sufficient to resolve the claim.",
                    "decision": "final_answer",
                    "revised_remaining_steps": [],
                    "final_rationale": "The debt claim is supported because no genuine triable issue remains.",
                    "final_decision": "support",
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
    assert trace["selected_templates"][0]["retrieval_mode"] == "exact_tag_unique"
    plan_schema = client.calls[0]["schema"]
    assert plan_schema["required"] == ["planning_analysis", "planned_steps"]
    assert plan_schema["properties"]["planned_steps"]["items"]["required"] == [
        "step_id",
        "step_name",
        "step_description",
        "template_tags",
    ]
    assert [call["max_tokens"] for call in client.calls] == [1400, 1400, 1000]
    assert client.calls[-1]["schema"]["required"] == [
        "review_analysis",
        "decision",
        "revised_remaining_steps",
        "final_rationale",
        "final_decision",
    ]
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
                    "planning_analysis": "The debt entitlement is dispositive.",
                    "planned_steps": [
                        {
                            "step_id": "S1",
                            "step_name": "Debt entitlement check",
                            "step_description": "Check whether repayment is due.",
                            "template_tags": ["debt_payment"],
                        }
                    ],
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
                    "final_rationale": "The repayment claim is supported.",
                    "final_decision": "support",
                }
            ),
        ]
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")

    analysis, trace = _execute_rf_style_case(client, config, _case(), templates=templates)

    assert analysis.final_decision == "support"
    assert trace["calls"] == 3
    assert "No remaining abstract steps are available" in client.prompts[-1]
    assert client.calls[-1]["schema"]["required"] == [
        "final_rationale",
        "final_decision",
    ]
    assert set(client.calls[-1]["schema"]["properties"]) == {
        "final_rationale",
        "final_decision",
    }


def test_output_normalizers_repair_common_rf_shapes():
    legacy_step = LegalFluxAbstractStep.model_validate(
        {
            "step_id": "S1",
            "step_name": "Legacy step",
            "template_tags": ["legacy"],
            "purpose": "Legacy description.",
        }
    )
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
    assert legacy_step.step_description == "Legacy description."
    assert "purpose" not in legacy_step.model_dump(mode="json")
    assert artifact["template_id"] == "LF001"
    assert artifact["material_fact_ids"] == ["F1"]
    assert artifact["issue_ids"] == ["I1"]
    assert artifact["confidence"] == "medium"
    assert "Additional structured notes" in artifact["instantiated_result"]
    assert "material_fact_ids_object_unwrapped" in artifact_repairs
    assert review["decision"] == "final_answer"
    assert review["final_decision"] == "support"
    assert review["review_analysis"].startswith("Done.")
    assert "rf_review_legacy_rationale_renamed" in review_repairs
    assert "rf_review_extra_fields_folded_into_review_analysis" in review_repairs


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
    assert "Template ID:" not in rows[0]["messages"][1]["content"]
    assert '"scope"' in rows[0]["messages"][2]["content"]
    assert "reasoning_flow" not in rows[0]["messages"][2]["content"]
    assert rows[0]["prompt"] == rows[0]["messages"][:2]
    assert rows[0]["completion"] == rows[0]["messages"][2:]


def test_template_structure_sft_dry_run_uses_full_template_library(tmp_path: Path):
    pool = tmp_path / "templates.jsonl"
    write_jsonl(
        pool,
        [
            _template(
                f"LF{index:03d}", f"Template {index}", "tag", "legal_reasoning"
            ).model_dump(mode="json")
            for index in range(10)
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

    result = train_template_structure_sft(
        config,
        dry_run=True,
        learning_rate=5e-5,
        output_dir="runs/sft/lr-5e-5",
    )

    assert result["train_examples"] == 10
    assert result["eval_examples"] == 0
    assert result["estimated_optimizer_steps"] == 6
    assert result["settings"]["num_train_epochs"] == 6
    assert result["settings"]["learning_rate"] == 0.00005
    assert result["settings"]["save_total_limit"] == 6
    assert Path(result["output_dir"]).parts[-3:] == ("runs", "sft", "lr-5e-5")


def test_template_sft_passes_trust_remote_code_to_model_loading():
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    settings = template_sft_settings(config)
    settings["trust_remote_code"] = True
    model_dtype = object()

    kwargs = _template_sft_config_kwargs(
        settings,
        model_dtype=model_dtype,
        use_bf16=True,
        use_fp16=False,
        has_eval=False,
    )

    assert "trust_remote_code" not in kwargs
    assert kwargs["model_init_kwargs"]["trust_remote_code"] is True
    assert kwargs["model_init_kwargs"]["dtype"] is model_dtype


def test_template_sft_reports_unsupported_runtime_config_fields():
    class MinimalConfig:
        def __init__(self, output_dir: str):
            self.output_dir = output_dir

    with pytest.raises(RuntimeError, match="completion_only_loss"):
        _validate_constructor_kwargs(
            MinimalConfig,
            {"output_dir": "checkpoint", "completion_only_loss": True},
            component="test config",
        )


def test_template_sft_uses_text_tokenizer_instead_of_multimodal_processor():
    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

    class FakeAutoTokenizer:
        call = None

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            cls.call = (model_name, kwargs)
            return FakeTokenizer()

    settings = {
        "model_name_or_path": "Qwen/Qwen3.5-9B",
        "trust_remote_code": False,
    }

    tokenizer = _load_text_tokenizer(FakeAutoTokenizer, settings)

    assert FakeAutoTokenizer.call == (
        "Qwen/Qwen3.5-9B",
        {"trust_remote_code": False},
    )
    assert tokenizer.pad_token == "<eos>"


def test_template_sft_builds_expected_lora_config():
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    settings = template_sft_settings(config)

    kwargs = _template_lora_config_kwargs(settings)

    assert kwargs["r"] == 32
    assert kwargs["target_modules"] == "all-linear"
    assert kwargs["exclude_modules"] == [
        "visual",
        "vision_model",
        "vision_tower",
        "merger",
    ]
    assert kwargs["task_type"] == "CAUSAL_LM"


def test_work_root_environment_redirects_generated_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    work_root = tmp_path / "cluster-work"
    monkeypatch.setenv("LEGAL_FLUX_WORK_ROOT", str(work_root))

    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")

    assert Path(config["paths"]["processed_dir"]) == (
        work_root / "data" / "processed" / "legal_flux"
    )
    assert Path(config["paths"]["runs_dir"]) == work_root / "runs" / "legal_flux"
    assert Path(config["paths"]["reports_dir"]) == (
        work_root / "reports" / "legal_flux"
    )
    assert Path(config["training"]["template_sft"]["output_dir"]) == (
        work_root
        / "runs"
        / "legal_flux"
        / "training"
        / "template_structure_sft"
    )


def test_dev_tune_subset_is_fixed_and_stratified(tmp_path: Path):
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    config["_project_root"] = str(tmp_path)
    config["paths"] = {
        **config["paths"],
        "processed_dir": str(tmp_path / "processed"),
    }
    cases = []
    for index in range(20):
        case = _case("trajectory_dev").model_copy(
            update={
                "case_id": f"case-{index}",
                "gold_answer": "support" if index % 2 else "reject",
                "metadata": {
                    "selection_split": "trajectory_dev",
                    "broad_domain": "procedure" if index % 3 else "contract",
                    "authority_bucket": "laws_only" if index % 4 else "none",
                },
            }
        )
        cases.append(case.model_dump(mode="json"))
    processed_dir = Path(config["paths"]["processed_dir"])
    write_jsonl(processed_dir / "cases.jsonl", cases)

    first = export_trajectory_dev_tune_subset(config, count=8)
    second = export_trajectory_dev_tune_subset(config, count=8)

    assert first["case_ids"] == second["case_ids"]
    assert len(first["case_ids"]) == 8
    assert sum(first["label_counts"].values()) == 8


def test_aggregate_frame_reports_binary_f1_metrics():
    frame = pd.DataFrame(
        [
            {
                "dataset": "legalhk",
                "condition": "flux_rf_style",
                "gold_answer": "support",
                "prediction": "support",
                "answer_correct": 1.0,
                "calls": 4,
            },
            {
                "dataset": "legalhk",
                "condition": "flux_rf_style",
                "gold_answer": "support",
                "prediction": "reject",
                "answer_correct": 0.0,
                "calls": 4,
            },
            {
                "dataset": "legalhk",
                "condition": "flux_rf_style",
                "gold_answer": "reject",
                "prediction": "reject",
                "answer_correct": 1.0,
                "calls": 4,
            },
        ]
    )

    row = _aggregate_frame(frame).iloc[0]

    assert row["answer_correct"] == pytest.approx(2 / 3)
    assert row["weighted_f1"] == pytest.approx(2 / 3)
    assert row["macro_f1"] == pytest.approx(2 / 3)


def test_sft_grid_summary_uses_weighted_f1_as_accuracy_tie_break(tmp_path: Path):
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    config["_project_root"] = str(tmp_path)
    config["paths"] = {
        **config["paths"],
        "runs_dir": str(tmp_path / "runs"),
        "reports_dir": str(tmp_path / "reports"),
    }
    for tag, weighted_f1 in (
        ("sft-screen-a", 0.70),
        ("sft-screen-b", 0.75),
    ):
        run_dir = tmp_path / "runs" / "trajectory_dev" / "experiments" / tag
        run_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "dataset": "legalhk",
                    "condition": "flux_rf_style",
                    "answer_correct": 0.8,
                    "weighted_f1": weighted_f1,
                    "calls": 4,
                }
            ]
        ).to_csv(run_dir / "aggregate.csv", index=False)

    result = summarize_sft_checkpoint_grid(
        config,
        phase="trajectory-dev",
        prefix="sft-screen-",
    )

    assert result["best_run_tag"] == "sft-screen-b"


def test_generation_sharding_keeps_conditions_for_each_case_together():
    cases = [_case().model_copy(update={"case_id": f"case-{index}"}) for index in range(5)]
    jobs = [
        {"case": case, "condition": condition}
        for case in cases
        for condition in ("direct", "structured", "flux_rf_style")
    ]

    shards = [
        _select_generation_shard(jobs, num_shards=2, shard_index=index)
        for index in range(2)
    ]

    assert sum(len(shard) for shard in shards) == len(jobs)
    for shard in shards:
        by_case: dict[str, set[str]] = {}
        for job in shard:
            by_case.setdefault(job["case"].case_id, set()).add(job["condition"])
        assert all(
            conditions == {"direct", "structured", "flux_rf_style"}
            for conditions in by_case.values()
        )


def test_xsim_dense_retrieval_and_cross_encoder_reranking(tmp_path: Path):
    processed = tmp_path / "processed"
    cases = [
        _case(split="planner_train").model_copy(
            update={
                "case_id": f"legalhk-{index}",
                "claim": f"Case {index} claim",
                "facts": {"F1": f"Case {index} facts"},
            }
        )
        for index in range(4)
    ]
    write_jsonl(
        processed / "cases.jsonl",
        [case.model_dump(mode="json") for case in cases],
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    config["_project_root"] = str(tmp_path)
    config["paths"] = {
        **config["paths"],
        "processed_dir": str(processed),
    }
    config["xsim"] = {
        **config["xsim"],
        "dense_model": "fake-bge-m3",
        "reranker_model": "fake-reranker",
        "dense_top_k": 3,
        "final_top_k": 2,
    }

    result = build_xsim(
        config,
        case_limit=1,
        dense_encoder=FakeDenseEncoder(),
        reranker=FakeReranker(),
    )
    neighbors = load_xsim_neighbors(config)

    assert result["dense_rows_added"] == 1
    assert result["reranked_rows_added"] == 1
    assert neighbors["legalhk-0"] == ["legalhk-0", "legalhk-3", "legalhk-1"]


def test_dpo_candidate_sampling_creates_four_seeded_plans(tmp_path: Path):
    processed = tmp_path / "processed"
    pool = tmp_path / "templates.jsonl"
    cases = [
        _case(split="planner_train").model_copy(
            update={"case_id": f"legalhk-{100 + index}"}
        )
        for index in range(3)
    ]
    write_jsonl(
        processed / "cases.jsonl",
        [case.model_dump(mode="json") for case in cases],
    )
    write_jsonl(
        processed / "xsim" / "xsim_neighbors.jsonl",
        [
            {
                "anchor_case_id": cases[0].case_id,
                "selected_neighbors": [],
                "x_sim_case_ids": [case.case_id for case in cases],
            }
        ],
    )
    write_jsonl(
        pool,
        [
            _template(
                "LF001", "Debt entitlement", "debt", "rule_application"
            ).model_dump(mode="json"),
        ],
    )
    responses = [
        _response(
            {
                "planning_analysis": f"profile {index}; rationale {index}",
                "planned_steps": [
                    {
                        "step_id": "S1",
                        "step_name": "Debt entitlement",
                        "step_description": f"Resolve entitlement variant {index}.",
                        "template_tags": ["debt", "rule_application"],
                    }
                ],
            }
        )
        for index in range(4)
    ]
    client = SequenceClient(responses)
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    config["_project_root"] = str(tmp_path)
    config["paths"] = {
        **config["paths"],
        "processed_dir": str(processed),
        "prompts_dir": str(Path(__file__).parents[1] / "prompts"),
        "schemas_dir": str(Path(__file__).parents[1] / "schemas"),
    }
    config["legal_flux"] = {
        **config["legal_flux"],
        "template_pool_file": str(pool),
    }

    result = build_dpo_data(
        config,
        stage="sample",
        case_limit=1,
        client=client,
        similarity_backend=FixedEmbeddingBackend({}),
    )
    rows = read_jsonl(Path(result["candidates_path"]))

    assert result["sample_records_added"] == 4
    assert len(rows) == 4
    assert [row["sample_index"] for row in rows] == [0, 1, 2, 3]
    assert [call["seed"] for call in client.calls] == [
        config["dpo"]["seed"] + index for index in range(4)
    ]
    assert all(
        call["temperature"] == config["dpo"]["planner_temperature"]
        for call in client.calls
    )


def test_trajectory_dpo_export_uses_three_case_xsim_accuracy(tmp_path: Path):
    pool = tmp_path / "templates.jsonl"
    processed = tmp_path / "processed"
    write_jsonl(
        pool,
        [
            _template("LF001", "Debt entitlement", "debt", "rule_application").model_dump(mode="json"),
            _template("LF002", "Procedural gate", "procedure", "threshold").model_dump(mode="json"),
        ],
    )
    cases = [
        _case(split="planner_train").model_copy(
            update={
                "case_id": f"legalhk-{100 + index}",
                "gold_answer": answer,
                "claim": f"Claim for case {100 + index}",
            }
        )
        for index, answer in enumerate(["support", "reject", "support"])
    ]
    write_jsonl(
        processed / "cases.jsonl",
        [case.model_dump(mode="json") for case in cases],
    )
    plan_good = {
        "planning_analysis": "good debt profile; the entitlement step is dispositive.",
        "planned_steps": [
            {
                "step_id": "S1",
                "step_name": "Debt entitlement",
                "step_description": "Resolve whether repayment is due.",
                "template_tags": ["debt", "rule_application"],
            }
        ],
    }
    plan_bad = {
        "planning_analysis": "bad unrelated profile; this over-focuses on procedure.",
        "planned_steps": [
            {
                "step_id": "S1",
                "step_name": "Procedural gate",
                "step_description": "Screen for a procedural issue not shown by the facts.",
                "template_tags": ["procedure", "threshold"],
            }
        ],
    }
    plans = [
        plan_good,
        plan_bad,
        {**plan_good, "planning_analysis": "second good profile"},
        {**plan_bad, "planning_analysis": "second bad profile"},
    ]
    candidates = [
        {
            "run_hash": f"candidate-{index}",
            "candidate_id": f"candidate-{index}",
            "status": "ok",
            "anchor_case_id": cases[0].case_id,
            "sample_index": index,
            "trajectory_plan": plan,
            "retrieved_template_ids": ["LF001" if index in {0, 2} else "LF002"],
            "retrieval_trace": [
                {
                    "retrieval_mode": (
                        "exact_name" if index in {1, 2} else "embedding_full_pool"
                    ),
                    "similarity": 1.0 if index in {1, 2} else 0.6,
                }
            ],
        }
        for index, plan in enumerate(plans)
    ]
    training_dir = processed / "planner_training"
    write_jsonl(
        training_dir / "trajectory_candidates.jsonl",
        candidates,
    )
    correctness = {
        "candidate-0": [True, True, True],
        "candidate-1": [False, False, False],
        "candidate-2": [True, True, True],
        "candidate-3": [False, False, False],
    }
    evaluations = []
    for candidate_id, values in correctness.items():
        for target, correct in zip(cases, values, strict=True):
            evaluations.append(
                {
                    "run_hash": f"{candidate_id}-{target.case_id}",
                    "status": "ok",
                    "candidate_id": candidate_id,
                    "anchor_case_id": cases[0].case_id,
                    "target_case_id": target.case_id,
                    "answer_correct": correct,
                }
            )
    write_jsonl(
        training_dir / "trajectory_evaluations.jsonl",
        evaluations,
    )
    write_jsonl(
        processed / "xsim" / "xsim_neighbors.jsonl",
        [
            {
                "anchor_case_id": cases[0].case_id,
                "selected_neighbors": [],
                "x_sim_case_ids": [case.case_id for case in cases],
            }
        ],
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    config["_project_root"] = str(tmp_path)
    config["paths"] = {
        **config["paths"],
        "processed_dir": str(processed),
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
    assert "second good profile" in rows[0]["chosen"]
    assert "second bad profile" in rows[0]["rejected"]
    assert rows[0]["chosen_reward"] == 1.0
    assert rows[0]["rejected_reward"] == 0.0
    assert rows[0]["chosen_candidate_id"] == "candidate-2"
    assert rows[0]["rejected_candidate_id"] == "candidate-3"
    assert rows[0]["metadata"]["chosen_tie_break"]["mean_retrieval_similarity"] == 1.0
    assert rows[0]["metadata"]["rejected_tie_break"]["mean_retrieval_similarity"] == 0.6
    assert rows[0]["metadata"]["x_sim_case_ids"] == [
        case.case_id for case in cases
    ]


def test_trajectory_dpo_skips_group_when_all_accuracy_scores_tie():
    case = _case(split="planner_train")
    plan = {
        "planning_analysis": "Debt dispute; use the debt template.",
        "planned_steps": [
            {
                "step_id": "S1",
                "step_name": "Debt entitlement",
                "step_description": "Resolve repayment.",
                "template_tags": ["debt", "rule_application"],
            }
        ],
    }
    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "sample_index": index,
            "trajectory_plan": plan,
            "retrieval_trace": [
                {
                    "retrieval_mode": (
                        "exact_name" if index == 0 else "embedding_full_pool"
                    ),
                    "similarity": 1.0 if index == 0 else 0.5,
                }
            ],
        }
        for index in range(2)
    ]
    xsim_case_ids = [case.case_id, "legalhk-101", "legalhk-102"]
    evaluations = {
        candidate["candidate_id"]: [
            {
                "anchor_case_id": case.case_id,
                "target_case_id": target_id,
                "answer_correct": correct,
            }
            for target_id, correct in zip(
                xsim_case_ids,
                [True, False, True],
                strict=True,
            )
        ]
        for candidate in candidates
    }

    pair = _dpo_pair_from_xsim_group(
        {},
        case.case_id,
        candidates,
        case,
        xsim_case_ids,
        evaluations,
        "",
    )

    assert pair is None


def test_dpo_pipeline_executes_fixed_trajectory_on_all_xsim_cases(tmp_path: Path):
    processed = tmp_path / "processed"
    pool = tmp_path / "templates.jsonl"
    cases = [
        _case(split="planner_train").model_copy(
            update={
                "case_id": f"legalhk-{100 + index}",
                "gold_answer": answer,
                "claim": f"Claim for case {100 + index}",
            }
        )
        for index, answer in enumerate(["support", "reject", "support"])
    ]
    write_jsonl(
        processed / "cases.jsonl",
        [case.model_dump(mode="json") for case in cases],
    )
    write_jsonl(
        processed / "xsim" / "xsim_neighbors.jsonl",
        [
            {
                "anchor_case_id": cases[0].case_id,
                "selected_neighbors": [],
                "x_sim_case_ids": [case.case_id for case in cases],
            }
        ],
    )
    write_jsonl(
        pool,
        [
            _template(
                "LF001", "Debt entitlement", "debt", "rule_application"
            ).model_dump(mode="json"),
        ],
    )
    config = load_config(Path(__file__).parents[1] / "configs" / "legal_flux.yaml")
    config["_project_root"] = str(tmp_path)
    config["paths"] = {
        **config["paths"],
        "processed_dir": str(processed),
        "prompts_dir": str(Path(__file__).parents[1] / "prompts"),
        "schemas_dir": str(Path(__file__).parents[1] / "schemas"),
    }
    config["legal_flux"] = {
        **config["legal_flux"],
        "template_pool_file": str(pool),
    }
    client = FakeDpoPipelineClient(invalid_profiles={"profile 3"})

    result = build_dpo_data(
        config,
        stage="all",
        case_limit=1,
        client=client,
        similarity_backend=FixedEmbeddingBackend({}),
    )
    evaluations = read_jsonl(Path(result["evaluations_path"]))
    export = export_trajectory_dpo(config)
    pairs = read_jsonl(Path(export["output_path"]))

    assert result["sample_records_added"] == 4
    assert result["evaluation_records_added"] == 12
    assert result["invalid_answer_records"] == 3
    assert len(evaluations) == 12
    assert all(row["retrieved_template_ids"] == ["LF001"] for row in evaluations)
    invalid = [row for row in evaluations if not row["answer_valid"]]
    assert len(invalid) == 3
    assert all(row["status"] == "ok" for row in invalid)
    assert all(row["answer_correct"] is False for row in invalid)
    assert export["pairs"] == 1
    assert pairs[0]["chosen_reward"] == 1.0
    assert pairs[0]["rejected_reward"] == 0.0


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
            "--num-shards",
            "8",
            "--shard-index",
            "3",
            "--conditions",
            "direct",
            "structured",
            "--run-tag",
            "baseline-check",
            "--case-ids-file",
            "dev_ids.json",
            "--fail-on-errors",
        ]
    )
    assert args.phase == "final-test"
    assert args.dry_run
    assert args.num_shards == 8
    assert args.shard_index == 3
    assert args.conditions == ["direct", "structured"]
    assert args.run_tag == "baseline-check"
    assert args.case_ids_file == "dev_ids.json"
    assert args.fail_on_errors
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
        [
            "--config",
            "configs/legal_flux.yaml",
            "flux-train-template-sft",
            "--dry-run",
            "--learning-rate",
            "5e-5",
            "--num-train-epochs",
            "6",
            "--output-dir",
            "runs/sft/lr-5e-5",
        ]
    ).learning_rate == 5e-5
    assert parser.parse_args(
        [
            "--config",
            "configs/legal_flux.yaml",
            "flux-export-dev-tune",
            "--count",
            "256",
        ]
    ).count == 256
    xsim_args = parser.parse_args(
        [
            "--config",
            "configs/legal_flux.yaml",
            "flux-build-xsim",
            "--stage",
            "rerank",
            "--case-limit",
            "10",
        ]
    )
    assert xsim_args.command == "flux-build-xsim"
    assert xsim_args.stage == "rerank"
    assert xsim_args.case_limit == 10
    dpo_args = parser.parse_args(
        [
            "--config",
            "configs/legal_flux.yaml",
            "flux-build-dpo-data",
            "--stage",
            "sample",
            "--case-limit",
            "10",
        ]
    )
    assert dpo_args.command == "flux-build-dpo-data"
    assert dpo_args.stage == "sample"
    assert dpo_args.case_limit == 10
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
