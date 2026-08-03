from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Burden = Literal["plaintiff", "defendant", "unclear"]
ElementStatus = Literal["unresolved", "satisfied", "not_satisfied", "defeated"]
IssueConclusionValue = Literal["satisfied", "not_satisfied", "defeated", "unresolved"]
FinalDecisionValue = Literal["support", "reject", "mixed", "unresolved"]
BinaryFinalDecisionValue = Literal["support", "reject"]
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
    irac_reasoning: str = ""
    issue_conclusions: list[IssueConclusion] = Field(default_factory=list)
    final_decision: FinalDecisionValue
    final_rationale: str = ""


class DirectAnalysis(StrictModel):
    final_decision: FinalDecisionValue
    final_rationale: str


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
    step_description: str
    template_tags: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_purpose(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "purpose" not in value:
            return value
        migrated = dict(value)
        if "step_description" not in migrated:
            migrated["step_description"] = migrated["purpose"]
        migrated.pop("purpose", None)
        return migrated


class LegalFluxAbstractPlan(StrictModel):
    planning_analysis: str
    planned_steps: list[LegalFluxAbstractStep]

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_analysis(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "planning_analysis" in value:
            return value
        migrated = dict(value)
        parts = [
            str(migrated.pop(key)).strip()
            for key in ("case_profile", "planning_rationale")
            if migrated.get(key) not in (None, "")
        ]
        if parts:
            migrated["planning_analysis"] = " ".join(parts)
        return migrated


class LegalFluxStepArtifact(StrictModel):
    step_id: str
    template_id: str
    instantiated_result: str = Field(max_length=1800)
    material_fact_ids: list[str] = Field(default_factory=list)
    issue_ids: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"]
    needs_revision: bool
    revision_reason: str


class LegalFluxRfReview(StrictModel):
    review_analysis: str = ""
    decision: FluxRfReviewDecision = "final_answer"
    revised_remaining_steps: list[LegalFluxAbstractStep] = Field(default_factory=list)
    final_rationale: str = ""
    final_decision: BinaryFinalDecisionValue | None = None


class NormalizedCase(StrictModel):
    dataset: Literal["legalhk"]
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
