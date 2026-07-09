from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Burden = Literal["plaintiff", "defendant", "unclear"]
ElementStatus = Literal["unresolved", "satisfied", "not_satisfied", "defeated"]
IssueConclusionValue = Literal[
    "satisfied", "not_satisfied", "defeated", "unresolved"
]
FinalDecisionValue = Literal["support", "reject", "mixed", "unresolved"]
BinaryFinalDecisionValue = Literal["support", "reject"]
BufferUpdateAction = Literal["new", "merge", "reject"]
FluxReviewDecision = Literal["continue", "revise", "stop"]
FluxRfReviewDecision = Literal["continue", "revise", "final_answer"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Element(StrictModel):
    element_id: str
    element: str
    supporting_fact_ids: list[str] = Field(default_factory=list)
    opposing_fact_ids: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    status: ElementStatus = "unresolved"


class Issue(StrictModel):
    issue_id: str
    issue: str
    rule_or_test: str
    burden_on: Burden
    elements: list[Element]
    defenses: list[str] = Field(default_factory=list)


class CaseState(StrictModel):
    claims: list[str]
    requested_remedies: list[str]
    issues: list[Issue]


class IssueConclusion(StrictModel):
    issue_id: str
    conclusion: IssueConclusionValue
    supporting_fact_ids: list[str] = Field(default_factory=list)
    opposing_fact_ids: list[str] = Field(default_factory=list)
    explanation: str


class FinalAnalysis(StrictModel):
    issue_conclusions: list[IssueConclusion]
    final_decision: FinalDecisionValue
    final_rationale: str


class DirectAnalysis(StrictModel):
    final_decision: FinalDecisionValue
    final_rationale: str


class DistilledLegalProblem(StrictModel):
    claim_type: str
    remedy_type: str
    lawsuit_type: str
    material_factual_pattern: list[str] = Field(default_factory=list)
    issue_families: list[str] = Field(default_factory=list)
    defenses_or_counterarguments: list[str] = Field(default_factory=list)
    evidence_posture: Literal[
        "supporting", "opposing", "conflicting", "insufficient", "unclear"
    ]
    retrieval_query: str


class FrontierLegalProblem(StrictModel):
    case_id: str
    procedural_posture: str
    claim_and_remedy: str
    material_fact_ids: list[str]
    dispositive_questions: list[str]
    evidence_conflicts: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    retrieval_summary: str


class LegalThoughtTemplate(StrictModel):
    template_id: str
    name: str
    description: str
    applicability_cues: list[str]
    reasoning_steps: list[str]
    required_checks: list[str]
    contraindications: list[str] = Field(default_factory=list)
    provenance_case_ids: list[str] = Field(default_factory=list)
    version: int = Field(default=1, ge=1)


class TemplateRetrieval(StrictModel):
    template: LegalThoughtTemplate
    similarity: float = Field(ge=0.0, le=1.0)
    used_fallback: bool
    best_candidate_template_id: str | None = None


class BufferUpdateEvent(StrictModel):
    action: BufferUpdateAction
    source_case_id: str
    target_template_id: str | None
    template: LegalThoughtTemplate | None
    rationale: str


class LegalFluxTemplate(StrictModel):
    template_id: str
    template_name: str
    knowledge_tags: list[str]
    description: str
    application_scenario: str
    reasoning_flow: list[str]
    example_application: str


class LegalFluxPlanStep(StrictModel):
    step_id: str
    template_id: str
    purpose: str
    expected_artifact: str


class LegalFluxAbstractStep(StrictModel):
    step_id: str
    step_name: str
    template_tags: list[str] = Field(default_factory=list)
    purpose: str


class LegalFluxAbstractPlan(StrictModel):
    case_profile: str
    planned_steps: list[LegalFluxAbstractStep]
    planning_rationale: str


class LegalFluxTrajectoryPlan(StrictModel):
    case_profile: str
    planned_steps: list[LegalFluxPlanStep]
    planning_rationale: str


class LegalFluxStepArtifact(StrictModel):
    step_id: str
    template_id: str
    instantiated_result: str
    material_fact_ids: list[str] = Field(default_factory=list)
    issue_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    needs_revision: bool
    revision_reason: str


class LegalFluxTrajectoryReview(StrictModel):
    decision: FluxReviewDecision
    rationale: str
    revised_remaining_steps: list[LegalFluxPlanStep] = Field(default_factory=list)


class LegalFluxRfReview(StrictModel):
    decision: FluxRfReviewDecision
    rationale: str
    revised_remaining_steps: list[LegalFluxAbstractStep] = Field(default_factory=list)
    final_decision: BinaryFinalDecisionValue | None = None
    final_rationale: str = ""


class NormalizedCase(StrictModel):
    dataset: Literal["openexempt", "legalhk"]
    case_id: str
    variant_id: str = "original"
    pair_id: str | None = None
    perturbation_kind: str | None = None
    claim: str
    requested_remedy: str | None = None
    parties: list[str] = Field(default_factory=list)
    facts: dict[str, str]
    authorities: str | None = None
    gold_answer: str
    reference_issues: list[str] = Field(default_factory=list)
    reference_state: CaseState | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationErrorItem(StrictModel):
    code: str
    message: str
    path: str | None = None


class ValidationResult(StrictModel):
    valid: bool
    errors: list[ValidationErrorItem] = Field(default_factory=list)


class AuditResult(StrictModel):
    issue_coverage: int = Field(ge=0, le=4)
    rule_fit: int = Field(ge=0, le=4)
    factual_grounding: int = Field(ge=0, le=4)
    defense_coverage: int = Field(ge=0, le=4)
    burden_correctness: int = Field(ge=0, le=4)
    final_decision_consistency: int = Field(ge=0, le=4)
    first_error: Literal[
        "none",
        "input_insufficiency_or_leakage",
        "case_representation",
        "rule_or_authority",
        "fact_to_element_application",
        "burden_defense_or_counterargument",
        "issue_composition_or_final_judgment",
    ]
    downstream_symptoms: list[str] = Field(default_factory=list)
    rationale: str
