from __future__ import annotations

import random
from typing import Any

from .models import NormalizedCase


PRINCIPAL_CONDITIONS = ["direct", "structured", "typed", "validated"]


def _balanced_subset(
    cases: list[NormalizedCase], count: int, *, seed: int
) -> list[NormalizedCase]:
    by_answer: dict[str, list[NormalizedCase]] = {}
    for case in cases:
        by_answer.setdefault(str(case.gold_answer), []).append(case)
    rng = random.Random(seed)
    for group in by_answer.values():
        rng.shuffle(group)
    labels = sorted(by_answer)
    selected: list[NormalizedCase] = []
    while len(selected) < count:
        added = False
        for label in labels:
            group = by_answer[label]
            if group and len(selected) < count:
                selected.append(group.pop())
                added = True
        if not added:
            break
    return selected


def build_jobs(
    cases: list[NormalizedCase],
    config: dict[str, Any],
    *,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    split_cases = [
        case
        for case in cases
        if case.metadata.get("selection_split") == (
            "smoke" if smoke else "evaluation"
        )
    ]
    selected = split_cases
    if not split_cases and smoke:
        oe = [case for case in cases if case.dataset == "openexempt"][
            : config["data"]["smoke_openexempt_cases"]
        ]
        hk = [case for case in cases if case.dataset == "legalhk"][
            : config["data"]["smoke_legalhk_cases"]
        ]
        selected = oe + hk
    elif not split_cases:
        selected = cases

    jobs = [
        {
            "case": case,
            "condition": condition,
            "sample_index": 0,
            "temperature": config["model"]["temperature"],
            "seed": config["model"]["seed"],
        }
        for case in selected
        for condition in PRINCIPAL_CONDITIONS
    ]
    if not smoke:
        if split_cases:
            hk = selected
            oracle_count = config["data"]["oracle_cases"]
            sampling_count = config["data"]["sampling_control_cases"]
            sampling_repeats = config["data"]["sampling_control_repeats"]
            oracle_cases = _balanced_subset(
                hk, oracle_count, seed=config["project"]["seed"]
            )
            sampling_cases = _balanced_subset(
                hk, sampling_count, seed=config["project"]["seed"] + 1
            )
        else:
            oe_original = [
                case
                for case in cases
                if case.dataset == "openexempt" and case.variant_id == "original"
            ]
            hk = [case for case in cases if case.dataset == "legalhk"]
            oracle_cases = (
                oe_original[: config["data"]["oracle_openexempt_cases"]]
                + hk[: config["data"]["oracle_legalhk_cases"]]
            )
            sampling_cases = oe_original[
                : config["data"]["sampling_control_cases"]
            ]
            sampling_repeats = config["data"]["sampling_control_repeats"]
        for case in oracle_cases:
            jobs.append(
                {
                    "case": case,
                    "condition": "oracle",
                    "sample_index": 0,
                    "temperature": 0.0,
                    "seed": config["model"]["seed"],
                }
            )
        for case in sampling_cases:
            for sample_index in range(sampling_repeats):
                jobs.append(
                    {
                        "case": case,
                        "condition": "sampling_control",
                        "sample_index": sample_index,
                        "temperature": 0.7,
                        "seed": config["model"]["seed"] + sample_index + 1,
                    }
                )
    random.Random(config["project"]["seed"]).shuffle(jobs)
    return jobs
