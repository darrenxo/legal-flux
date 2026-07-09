from __future__ import annotations

from pathlib import Path

from .io_utils import read_jsonl
from .models import FrontierLegalProblem, NormalizedCase


def build_frontier_inputs(
    cases: list[NormalizedCase],
) -> list[dict]:
    return [
        {
            "case_id": case.case_id,
            "claim": case.claim,
            "requested_remedy": case.requested_remedy,
            "parties": case.parties,
            "facts": case.facts,
            "lawsuit_type": case.metadata.get("lawsuit_type", ""),
        }
        for case in cases
    ]


def load_frontier_profiles(
    path: Path,
    *,
    required_case_ids: set[str],
    valid_fact_ids_by_case: dict[str, set[str]] | None = None,
) -> dict[str, FrontierLegalProblem]:
    profiles: dict[str, FrontierLegalProblem] = {}
    for row in read_jsonl(path):
        profile = FrontierLegalProblem.model_validate(row)
        if profile.case_id in profiles:
            raise ValueError(
                f"Duplicate frontier profile for {profile.case_id}."
            )
        if valid_fact_ids_by_case is not None:
            valid_ids = valid_fact_ids_by_case.get(profile.case_id, set())
            unknown = sorted(set(profile.material_fact_ids) - valid_ids)
            if unknown:
                raise ValueError(
                    f"Frontier profile {profile.case_id} references unknown "
                    f"fact IDs: {', '.join(unknown)}"
                )
        profiles[profile.case_id] = profile
    missing = sorted(required_case_ids - profiles.keys())
    if missing:
        raise ValueError(
            f"Frontier profiles are missing {len(missing)} cases: "
            + ", ".join(missing[:10])
        )
    return profiles
