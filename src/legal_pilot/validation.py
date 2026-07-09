from __future__ import annotations

from collections import Counter

from .models import (
    CaseState,
    ValidationErrorItem,
    ValidationResult,
)


def validate_case_state(
    state: CaseState,
    *,
    valid_fact_ids: set[str],
    known_parties: set[str] | None = None,
) -> ValidationResult:
    errors: list[ValidationErrorItem] = []

    if not state.claims:
        errors.append(
            ValidationErrorItem(code="missing_claim", message="At least one claim is required.")
        )
    if not state.issues:
        errors.append(
            ValidationErrorItem(code="missing_issue", message="At least one issue is required.")
        )

    issue_ids = [issue.issue_id for issue in state.issues]
    for issue_id, count in Counter(issue_ids).items():
        if count > 1:
            errors.append(
                ValidationErrorItem(
                    code="duplicate_issue_id",
                    message=f"Issue ID {issue_id!r} appears {count} times.",
                )
            )

    element_ids: list[str] = []
    for issue_index, issue in enumerate(state.issues):
        if not issue.elements:
            errors.append(
                ValidationErrorItem(
                    code="missing_element",
                    message=f"Issue {issue.issue_id!r} has no element or decision criterion.",
                    path=f"issues.{issue_index}.elements",
                )
            )
        for element_index, element in enumerate(issue.elements):
            element_ids.append(element.element_id)
            if element.status != "unresolved":
                errors.append(
                    ValidationErrorItem(
                        code="premature_status",
                        message="Pre-analysis element status must be unresolved.",
                        path=f"issues.{issue_index}.elements.{element_index}.status",
                    )
                )
            for field_name, fact_ids in (
                ("supporting_fact_ids", element.supporting_fact_ids),
                ("opposing_fact_ids", element.opposing_fact_ids),
            ):
                for fact_id in fact_ids:
                    if fact_id not in valid_fact_ids:
                        errors.append(
                            ValidationErrorItem(
                                code="unknown_fact_id",
                                message=f"Unknown fact reference {fact_id!r}.",
                                path=(
                                    f"issues.{issue_index}.elements."
                                    f"{element_index}.{field_name}"
                                ),
                            )
                        )

    for element_id, count in Counter(element_ids).items():
        if count > 1:
            errors.append(
                ValidationErrorItem(
                    code="duplicate_element_id",
                    message=f"Element ID {element_id!r} appears {count} times.",
                )
            )

    if known_parties:
        joined = " ".join(state.claims + [issue.issue for issue in state.issues]).lower()
        # This conservative check only flags explicit party-like labels in brackets.
        for token in _bracketed_party_tokens(joined):
            if token not in {party.lower() for party in known_parties}:
                errors.append(
                    ValidationErrorItem(
                        code="unknown_party",
                        message=f"Unknown party label {token!r}.",
                    )
                )

    return ValidationResult(valid=not errors, errors=errors)


def _bracketed_party_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    start = 0
    while True:
        left = text.find("[party:", start)
        if left < 0:
            return tokens
        right = text.find("]", left)
        if right < 0:
            return tokens
        tokens.add(text[left + len("[party:") : right].strip())
        start = right + 1

