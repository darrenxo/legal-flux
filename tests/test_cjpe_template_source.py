from __future__ import annotations

import pandas as pd
import numpy as np

from legal_pilot.cjpe_template_source import (
    _apportion_counts,
    _proportional_family_sample,
)
from legal_pilot.legal_benchmark_data import BenchmarkCase
from legal_pilot.legal_flux_chatgpt import _split_batches_to_prompt_budget


def test_apportion_counts_uses_largest_remainders() -> None:
    assert _apportion_counts({0: 70, 1: 20, 2: 10}, 11) == {
        0: 8,
        1: 2,
        2: 1,
    }


def test_family_sample_is_exact_proportional_stratified_and_deterministic() -> None:
    rows = []
    for family, size in ((0, 70), (1, 20), (2, 10)):
        for index in range(size):
            rows.append(
                {
                    "source_id": f"{1980 + index % 40}_{family}_{index}",
                    "semantic_family": family,
                    "gold_label": "accepted" if index % 2 else "rejected",
                    "decade": f"{1980 + (index % 4) * 10}s",
                }
            )
    frame = pd.DataFrame(rows)

    first, quotas = _proportional_family_sample(frame, target_count=10, seed=17)
    second, _ = _proportional_family_sample(frame, target_count=10, seed=17)

    selected = frame[frame["source_id"].isin(first)]
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert quotas == {0: 7, 1: 2, 2: 1}
    assert selected["semantic_family"].value_counts().to_dict() == quotas


def test_semantic_batch_is_split_below_safe_prompt_budget() -> None:
    cases = [
        BenchmarkCase(
            dataset="il_tur_cjpe",
            case_id=f"case-{index}",
            source_split="aligned_multi_train_10pct",
            input_text="reasoning " * 55,
            gold_label="accepted" if index % 2 else "rejected",
            labels=["rejected", "accepted"],
            label_descriptions={
                "rejected": "appeal rejected",
                "accepted": "appeal accepted",
            },
            task_instruction="Predict the outcome.",
        )
        for index in range(6)
    ]
    human_outputs = {
        case.case_id: {"full_text": case.input_text} for case in cases
    }
    embeddings = {
        case.case_id: np.asarray([1.0, float(index % 2)], dtype=np.float32)
        for index, case in enumerate(cases)
    }
    batch = {
        "label": "other_uncertain__semantic_cluster",
        "coarse_legal_family": "other_uncertain",
        "cases": cases,
        "semantic_coherence": {},
    }

    split = _split_batches_to_prompt_budget(
        [batch],
        embeddings_by_case_id=embeddings,
        human_outputs=human_outputs,
        max_batch_characters=2200,
        minimum_support_cases=3,
    )

    assert len(split) == 2
    assert all(part["token_budget_split"] for part in split)
    assert all(len(part["cases"]) == 3 for part in split)
    assert all(part["batch_content_characters"] <= 2200 for part in split)
    assert {
        case.case_id for part in split for case in part["cases"]
    } == {case.case_id for case in cases}
