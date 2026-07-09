from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from .embeddings import SimilarityBackend, TfidfSimilarityBackend
from .models import (
    BufferUpdateEvent,
    LegalThoughtTemplate,
    NormalizedCase,
    TemplateRetrieval,
)


BOT_CONDITIONS = (
    "direct",
    "bot_full",
    "bot_no_distiller",
    "bot_no_buffer",
    "bot_no_manager",
    "bot_generic_init",
)


@dataclass(frozen=True)
class BotCaseSplit:
    adaptation: list[NormalizedCase]
    holdout: list[NormalizedCase]


@dataclass(frozen=True)
class BotPlanItem:
    case: NormalizedCase
    condition: str
    phase: Literal["adaptation", "holdout"]
    stream_index: int
    allow_update: bool


@dataclass(frozen=True)
class BotConditionSpec:
    use_distiller: bool
    use_buffer: bool
    use_manager: bool
    use_legal_seeds: bool
    profile_source: Literal["qwen", "raw", "frontier", "none"] = "qwen"


CONDITION_SPECS = {
    "direct": BotConditionSpec(False, False, False, False, "none"),
    "bot_full": BotConditionSpec(True, True, True, True),
    "bot_no_distiller": BotConditionSpec(False, True, True, True, "raw"),
    "bot_no_buffer": BotConditionSpec(True, False, False, False),
    "bot_no_manager": BotConditionSpec(True, True, False, True),
    "bot_generic_init": BotConditionSpec(True, True, True, False),
    "semantic_qwen_fixed": BotConditionSpec(True, True, False, True),
    "semantic_qwen_dynamic": BotConditionSpec(True, True, True, True),
    "semantic_raw_fixed": BotConditionSpec(False, True, False, True, "raw"),
    "semantic_frontier_generic": BotConditionSpec(
        False, False, False, False, "frontier"
    ),
    "semantic_frontier_fixed": BotConditionSpec(
        False, True, False, True, "frontier"
    ),
    "semantic_frontier_dynamic": BotConditionSpec(
        False, True, True, True, "frontier"
    ),
}


def generic_template() -> LegalThoughtTemplate:
    return LegalThoughtTemplate(
        template_id="generic_legal_reasoning",
        name="Generic dispositive-issue analysis",
        description=(
            "A fallback structure for legal claims that do not match a more "
            "specific reusable thought-template."
        ),
        applicability_cues=["legal claim", "facts", "remedy", "support reject"],
        reasoning_steps=[
            "Identify the dispositive issue or issues raised by the claim.",
            "Map only supplied facts to the requirements or decision criteria.",
            "Check defenses, counterarguments, burdens, and evidence gaps.",
            "Resolve each material issue and compose the final decision.",
        ],
        required_checks=[
            "claim and remedy",
            "supporting and opposing facts",
            "defenses or evidence gaps",
            "issue-to-decision consistency",
        ],
        contraindications=[],
        provenance_case_ids=[],
        version=1,
    )


def seed_templates() -> list[LegalThoughtTemplate]:
    definitions = [
        (
            "debt_payment",
            "Debt or unpaid-sum claim",
            "Evaluate whether a monetary obligation exists, became due, and remains unpaid.",
            ["debt", "loan", "invoice", "unpaid sum", "money due", "cheque"],
            [
                "Identify the alleged source and amount of the obligation.",
                "Check formation, performance, maturity, and proof of non-payment.",
                "Test payment, discharge, limitation, set-off, and proof defenses.",
                "Connect the surviving obligation to the requested money remedy.",
            ],
            ["obligation", "amount", "due date", "payment status", "defenses"],
            ["possession-only dispute"],
        ),
        (
            "contract_performance",
            "Contract formation and performance",
            "Analyze formation, operative terms, breach, defenses, and remedy.",
            ["contract", "agreement", "breach", "performance", "termination"],
            [
                "Identify the alleged agreement and material obligations.",
                "Check formation and the terms supported by the supplied facts.",
                "Evaluate performance, breach, termination, and contractual defenses.",
                "Assess whether the requested remedy follows from the breach found.",
            ],
            ["formation", "terms", "performance", "breach", "remedy"],
            ["claim depends solely on tort duties"],
        ),
        (
            "property_possession",
            "Property, tenancy, or possession",
            "Analyze entitlement to occupy, possess, recover, or retain property.",
            ["property", "possession", "tenancy", "lease", "landlord", "occupier"],
            [
                "Identify the property interest and the claimed right to possession.",
                "Check the source, duration, and termination of any tenancy or licence.",
                "Evaluate notice, payment, consent, abandonment, and competing entitlement.",
                "Match the established entitlement to possession or related relief.",
            ],
            ["property identity", "source of possession", "termination", "notice", "relief"],
            ["pure money claim with no possessory issue"],
        ),
        (
            "specific_relief",
            "Specific or equitable relief",
            "Test entitlement to non-monetary relief and practical bars to granting it.",
            ["specific performance", "injunction", "declaration", "equitable relief"],
            [
                "Identify the precise conduct or legal relationship the order would compel.",
                "Check the underlying substantive right and adequacy of ordinary relief.",
                "Evaluate discretion, delay, impossibility, hardship, and clean-hands concerns.",
                "Ensure the proposed order is sufficiently definite and supported.",
            ],
            ["underlying right", "adequacy", "discretionary bars", "order specificity"],
            ["routine liquidated debt adequately resolved by damages"],
        ),
        (
            "negligence_damage",
            "Negligence or damage claim",
            "Analyze duty or responsibility, breach, causation, loss, and defenses.",
            ["negligence", "injury", "damage", "duty", "causation", "loss"],
            [
                "Identify the alleged responsibility and protected interest.",
                "Check breach against the conduct described in the supplied facts.",
                "Trace factual and legal causation to a proved loss.",
                "Evaluate contributory conduct, remoteness, mitigation, and proof gaps.",
            ],
            ["responsibility", "breach", "causation", "loss", "defenses"],
            ["claim turns only on an express payment promise"],
        ),
        (
            "procedure_appeal",
            "Procedural application or appeal",
            "Analyze the governing procedural threshold, timing, prejudice, and disposition.",
            ["appeal", "leave", "extension", "set aside", "procedure", "jurisdiction"],
            [
                "Identify the procedural order requested and its threshold.",
                "Check jurisdiction, standing, timing, preservation, and compliance.",
                "Evaluate explanation, merits, prejudice, finality, and proportionality.",
                "Resolve the application without treating procedural allegations as merits facts.",
            ],
            ["jurisdiction", "standing", "timing", "threshold", "prejudice"],
            ["ordinary merits claim with no procedural application"],
        ),
        (
            "evidence_insufficiency",
            "Evidence conflict or insufficiency",
            "Resolve a claim whose decisive element is unsupported, contradicted, or uncertain.",
            ["insufficient evidence", "conflicting evidence", "proof", "credibility", "burden"],
            [
                "Identify which party bears the burden on the decisive issue.",
                "Separate supplied evidence from allegations and missing information.",
                "Compare supporting and opposing facts without inventing a tie-breaker.",
                "Treat an unmet burden as insufficiency and propagate it to the decision.",
            ],
            ["burden", "supporting proof", "opposing proof", "missing evidence"],
            [],
        ),
        (
            "mixed_defenses",
            "Multiple issues and defenses",
            "Compose a result across several claims, issues, or independent defenses.",
            ["multiple issues", "defense", "counterclaim", "alternative claim", "mixed"],
            [
                "List only issues capable of changing the requested disposition.",
                "Analyze each issue against its own facts, burden, and defenses.",
                "Keep independent issues separate and identify dependencies.",
                "Compose the final result from the issue outcomes without contradiction.",
            ],
            ["dispositive issues", "dependencies", "defenses", "composition"],
            ["single uncontested issue"],
        ),
    ]
    return [
        LegalThoughtTemplate(
            template_id=template_id,
            name=name,
            description=description,
            applicability_cues=cues,
            reasoning_steps=steps,
            required_checks=checks,
            contraindications=contraindications,
            provenance_case_ids=[],
            version=1,
        )
        for (
            template_id,
            name,
            description,
            cues,
            steps,
            checks,
            contraindications,
        ) in definitions
    ]


class TemplateBuffer:
    def __init__(
        self,
        templates: Iterable[LegalThoughtTemplate],
        *,
        similarity_backend: SimilarityBackend | None = None,
    ):
        self.templates = [
            template.model_copy(deep=True) for template in templates
        ]
        self.similarity_backend = (
            similarity_backend or TfidfSimilarityBackend()
        )
        self._assert_unique_ids()

    def _assert_unique_ids(self) -> None:
        ids = [template.template_id for template in self.templates]
        if len(ids) != len(set(ids)):
            raise ValueError("Thought-template IDs must be unique.")

    def model_dump(self) -> dict[str, Any]:
        return {
            "templates": [
                template.model_dump(mode="json") for template in self.templates
            ]
        }

    def retrieve(self, query: str, *, threshold: float) -> TemplateRetrieval:
        if not self.templates:
            return TemplateRetrieval(
                template=generic_template(),
                similarity=0.0,
                used_fallback=True,
                best_candidate_template_id=None,
            )
        documents = [_template_document(template) for template in self.templates]
        similarities = self.similarity_backend.similarities(query, documents)
        best_index = max(range(len(similarities)), key=similarities.__getitem__)
        score = float(similarities[best_index])
        if score < threshold:
            return TemplateRetrieval(
                template=generic_template(),
                similarity=score,
                used_fallback=True,
                best_candidate_template_id=self.templates[
                    best_index
                ].template_id,
            )
        return TemplateRetrieval(
            template=self.templates[best_index],
            similarity=score,
            used_fallback=False,
            best_candidate_template_id=self.templates[best_index].template_id,
        )

    def apply_candidate(
        self,
        candidate: LegalThoughtTemplate,
        *,
        source_case_id: str,
        merge_threshold: float,
    ) -> BufferUpdateEvent:
        sanitized = _with_provenance(candidate, source_case_id)
        if not self.templates:
            self.templates.append(sanitized)
            return BufferUpdateEvent(
                action="new",
                source_case_id=source_case_id,
                target_template_id=sanitized.template_id,
                template=sanitized,
                rationale="No existing template was available to merge.",
            )
        similarities = self.similarity_backend.similarities(
            _template_document(sanitized),
            [_template_document(template) for template in self.templates],
        )
        best_index = max(range(len(similarities)), key=similarities.__getitem__)
        score = float(similarities[best_index])
        if score >= merge_threshold:
            current = self.templates[best_index]
            merged = _merge_templates(current, sanitized, source_case_id)
            self.templates[best_index] = merged
            return BufferUpdateEvent(
                action="merge",
                source_case_id=source_case_id,
                target_template_id=current.template_id,
                template=merged,
                rationale=f"Candidate similarity {score:.3f} met merge threshold.",
            )
        existing_ids = {template.template_id for template in self.templates}
        if sanitized.template_id in existing_ids:
            sanitized = sanitized.model_copy(
                update={
                    "template_id": _unique_template_id(
                        sanitized.template_id, source_case_id, existing_ids
                    )
                }
            )
        self.templates.append(sanitized)
        return BufferUpdateEvent(
            action="new",
            source_case_id=source_case_id,
            target_template_id=sanitized.template_id,
            template=sanitized,
            rationale=f"Candidate similarity {score:.3f} was below merge threshold.",
        )

    def apply_candidate_append_only(
        self,
        candidate: LegalThoughtTemplate,
        *,
        source_case_id: str,
        novelty_threshold: float,
    ) -> BufferUpdateEvent:
        sanitized = _with_provenance(candidate, source_case_id)
        if self.templates:
            similarities = self.similarity_backend.similarities(
                _template_document(sanitized),
                [_template_document(template) for template in self.templates],
            )
            best_index = max(
                range(len(similarities)), key=similarities.__getitem__
            )
            score = float(similarities[best_index])
            if score >= novelty_threshold:
                return BufferUpdateEvent(
                    action="reject",
                    source_case_id=source_case_id,
                    target_template_id=self.templates[
                        best_index
                    ].template_id,
                    template=None,
                    rationale=(
                        f"Candidate similarity {score:.3f} met the redundancy "
                        "threshold; append-only manager made no change."
                    ),
                )
        else:
            score = 0.0
        existing_ids = {template.template_id for template in self.templates}
        if sanitized.template_id in existing_ids:
            sanitized = sanitized.model_copy(
                update={
                    "template_id": _unique_template_id(
                        sanitized.template_id, source_case_id, existing_ids
                    )
                }
            )
        self.templates.append(sanitized)
        return BufferUpdateEvent(
            action="new",
            source_case_id=source_case_id,
            target_template_id=sanitized.template_id,
            template=sanitized,
            rationale=(
                f"Candidate maximum similarity {score:.3f} was below the "
                "novelty threshold."
            ),
        )

    @classmethod
    def replay(
        cls,
        seeds: Iterable[LegalThoughtTemplate],
        events: Iterable[BufferUpdateEvent],
        *,
        similarity_backend: SimilarityBackend | None = None,
    ) -> "TemplateBuffer":
        buffer = cls(seeds, similarity_backend=similarity_backend)
        for event in events:
            if event.action == "reject":
                continue
            if event.template is None:
                raise ValueError("New and merge events require a template.")
            if event.action == "new":
                buffer.templates.append(event.template.model_copy(deep=True))
            elif event.action == "merge":
                matches = [
                    index
                    for index, template in enumerate(buffer.templates)
                    if template.template_id == event.target_template_id
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Cannot replay merge for {event.target_template_id!r}."
                    )
                buffer.templates[matches[0]] = event.template.model_copy(deep=True)
            buffer._assert_unique_ids()
        return buffer


def select_bot_cases(
    cases: list[NormalizedCase],
    config: dict[str, Any],
    *,
    smoke: bool,
) -> BotCaseSplit:
    split_name = "smoke" if smoke else "evaluation"
    candidates = [
        case
        for case in cases
        if case.dataset == "legalhk"
        and case.metadata.get("selection_split") == split_name
    ]
    if smoke:
        count = min(config["bot"]["smoke_cases"], len(candidates))
        return BotCaseSplit(
            adaptation=_balanced_order(
                candidates, count=count, seed=config["project"]["seed"]
            ),
            holdout=[],
        )
    adaptation_count = config["bot"]["adaptation_cases"]
    holdout_count = config["bot"]["holdout_cases"]
    ordered = _balanced_order(
        candidates,
        count=adaptation_count + holdout_count,
        seed=config["project"]["seed"],
    )
    if len(ordered) < adaptation_count + holdout_count:
        raise ValueError("Not enough LegalHK evaluation cases for the BoT split.")
    adaptation = _take_balanced(
        ordered, adaptation_count, seed=config["project"]["seed"] + 1
    )
    adaptation_ids = {case.case_id for case in adaptation}
    remaining = [case for case in ordered if case.case_id not in adaptation_ids]
    holdout = _take_balanced(
        remaining, holdout_count, seed=config["project"]["seed"] + 2
    )
    return BotCaseSplit(adaptation=adaptation, holdout=holdout)


def build_bot_plan(
    cases: list[NormalizedCase],
    config: dict[str, Any],
    *,
    smoke: bool,
) -> list[BotPlanItem]:
    selected = select_bot_cases(cases, config, smoke=smoke)
    conditions = tuple(config["bot"].get("conditions", BOT_CONDITIONS))
    unknown = set(conditions) - set(CONDITION_SPECS)
    if unknown:
        raise ValueError(f"Unknown BoT conditions: {sorted(unknown)}")
    plan: list[BotPlanItem] = []
    for condition in conditions:
        spec = CONDITION_SPECS[condition]
        for index, case in enumerate(selected.adaptation):
            plan.append(
                BotPlanItem(
                    case=case,
                    condition=condition,
                    phase="adaptation",
                    stream_index=index,
                    allow_update=spec.use_manager,
                )
            )
        for offset, case in enumerate(selected.holdout):
            plan.append(
                BotPlanItem(
                    case=case,
                    condition=condition,
                    phase="holdout",
                    stream_index=len(selected.adaptation) + offset,
                    allow_update=False,
                )
            )
    return plan


def should_update_buffer(
    *,
    phase: str,
    answer_correct: bool,
    manager_enabled: bool,
    used_fallback: bool,
    similarity: float,
    novelty_threshold: float,
) -> bool:
    return (
        phase == "adaptation"
        and answer_correct
        and manager_enabled
        and (used_fallback or similarity < novelty_threshold)
    )


def raw_case_query(case: NormalizedCase) -> str:
    lawsuit_type = str(case.metadata.get("lawsuit_type", ""))
    return " ".join(
        [
            lawsuit_type,
            case.claim,
            case.requested_remedy or "",
            *case.facts.values(),
        ]
    )


def _balanced_order(
    cases: list[NormalizedCase], *, count: int, seed: int
) -> list[NormalizedCase]:
    by_answer: dict[str, list[NormalizedCase]] = {}
    for case in cases:
        by_answer.setdefault(str(case.gold_answer), []).append(case)
    rng = random.Random(seed)
    for group in by_answer.values():
        group.sort(key=lambda case: case.case_id)
        rng.shuffle(group)
    labels = sorted(by_answer)
    result: list[NormalizedCase] = []
    while len(result) < count:
        added = False
        for label in labels:
            if by_answer[label] and len(result) < count:
                result.append(by_answer[label].pop())
                added = True
        if not added:
            break
    return result


def _take_balanced(
    cases: list[NormalizedCase], count: int, *, seed: int
) -> list[NormalizedCase]:
    return _balanced_order(list(cases), count=count, seed=seed)


def _template_document(template: LegalThoughtTemplate) -> str:
    return " ".join(
        [
            template.name,
            template.description,
            *template.applicability_cues,
            *template.reasoning_steps,
            *template.required_checks,
            *template.contraindications,
        ]
    )


def _merge_templates(
    current: LegalThoughtTemplate,
    candidate: LegalThoughtTemplate,
    source_case_id: str,
) -> LegalThoughtTemplate:
    return current.model_copy(
        update={
            "description": (
                current.description
                if len(current.description) >= len(candidate.description)
                else candidate.description
            ),
            "applicability_cues": _union(
                current.applicability_cues, candidate.applicability_cues
            ),
            "reasoning_steps": _union(
                current.reasoning_steps, candidate.reasoning_steps
            ),
            "required_checks": _union(
                current.required_checks, candidate.required_checks
            ),
            "contraindications": _union(
                current.contraindications, candidate.contraindications
            ),
            "provenance_case_ids": _union(
                current.provenance_case_ids,
                [*candidate.provenance_case_ids, source_case_id],
            ),
            "version": current.version + 1,
        }
    )


def _with_provenance(
    template: LegalThoughtTemplate, source_case_id: str
) -> LegalThoughtTemplate:
    return template.model_copy(
        update={
            "template_id": _slug(template.template_id or template.name),
            "provenance_case_ids": _union(
                template.provenance_case_ids, [source_case_id]
            ),
            "version": max(template.version, 1),
        }
    )


def _union(first: Iterable[str], second: Iterable[str]) -> list[str]:
    return list(dict.fromkeys([*first, *second]))


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "legal_template"


def _unique_template_id(
    base: str, source_case_id: str, existing_ids: set[str]
) -> str:
    suffix = _slug(source_case_id)
    candidate = f"{base}_{suffix}"
    index = 2
    while candidate in existing_ids:
        candidate = f"{base}_{suffix}_{index}"
        index += 1
    return candidate
