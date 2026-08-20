from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_pilot.io_utils import write_jsonl
from legal_pilot.legal_benchmark_data import (
    BenchmarkCase,
    _balanced_sample,
    _binary_judgment_label,
    _hugging_face_token,
    select_benchmark_cases,
)
from legal_pilot.legal_benchmark_runner import (
    _aggregate_scores,
    _paired_comparisons,
    _response_schema,
    _truncate_input,
    render_benchmark_prompt,
)


def _case(case_id: str, label: str, *, split: str = "dev") -> BenchmarkCase:
    return BenchmarkCase(
        dataset="realistic_ljp_facts",
        case_id=case_id,
        source_split=split,
        input_text=f"Facts for {case_id}",
        gold_label=label,
        labels=["rejected", "accepted"],
        label_descriptions={
            "rejected": "the appeal is rejected",
            "accepted": "the appeal is accepted",
        },
        task_instruction="Predict the appeal outcome.",
    )


def _config(root: Path) -> dict:
    return {
        "_project_root": str(root),
        "project": {"seed": 17},
        "model": {
            "name": "test-model",
            "context_length": 16384,
            "temperature": 0.0,
            "seed": 17,
        },
        "paths": {"prompts_dir": "prompts"},
        "benchmarks": {
            "max_input_characters": 20,
            "input_truncation": "head",
            "paths": {
                "processed_dir": "processed",
                "raw_dir": "raw",
                "runs_dir": "runs",
            },
            "datasets": {
                "realistic_ljp_facts": {
                    "pilot_split": "dev",
                    "pilot_size": 2,
                    "full_split": "test",
                }
            },
        },
    }


def test_binary_judgment_label_mapping() -> None:
    assert _binary_judgment_label(0) == "rejected"
    assert _binary_judgment_label(1) == "accepted"
    assert _binary_judgment_label("REJECTED") == "rejected"
    assert _binary_judgment_label("allowed") == "accepted"
    with pytest.raises(ValueError, match="Unknown binary judgment label"):
        _binary_judgment_label("mixed")


def test_hugging_face_token_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "fallback-token")
    assert _hugging_face_token() == "test-token"


def test_balanced_sample_is_deterministic_and_balanced() -> None:
    cases = [
        *[_case(f"r-{index}", "rejected") for index in range(6)],
        *[_case(f"a-{index}", "accepted") for index in range(6)],
    ]
    first = _balanced_sample(cases, count=6, seed=9)
    second = _balanced_sample(cases, count=6, seed=9)
    assert [case.case_id for case in first] == [case.case_id for case in second]
    assert [case.gold_label for case in first].count("accepted") == 3
    assert [case.gold_label for case in first].count("rejected") == 3


def test_select_cases_uses_pilot_ids_and_full_test_split(tmp_path: Path) -> None:
    config = _config(tmp_path)
    directory = tmp_path / "processed" / "realistic_ljp_facts"
    cases = [
        _case("dev-a", "accepted"),
        _case("dev-r", "rejected"),
        _case("test-a", "accepted", split="test"),
    ]
    write_jsonl(directory / "cases.jsonl", [case.model_dump() for case in cases])
    (directory / "pilot_case_ids.json").write_text(
        json.dumps(["dev-r", "dev-a"]),
        encoding="utf-8",
    )

    pilot = select_benchmark_cases(
        config,
        datasets=["realistic_ljp_facts"],
        subset="pilot",
    )
    full = select_benchmark_cases(
        config,
        datasets=["realistic_ljp_facts"],
        subset="full",
    )
    assert [case.case_id for case in pilot] == ["dev-r", "dev-a"]
    assert [case.case_id for case in full] == ["test-a"]


def test_prompt_rendering_records_head_truncation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "benchmark_direct.txt").write_text(
        "{task_instruction}\n{labels}\n{case_text}",
        encoding="utf-8",
    )
    case = _case("long", "accepted").model_copy(
        update={"input_text": "0123456789" * 3}
    )
    prompt, metadata = render_benchmark_prompt(config, case, "direct")
    assert prompt.endswith("01234567890123456789")
    assert metadata["truncated"] is True
    assert metadata["original_characters"] == 30
    assert metadata["used_characters"] == 20


def test_annocaselaw_prompt_and_schema_allow_all_three_outcomes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "benchmark_direct.txt").write_text(
        "{task_instruction}\n{labels}\n{case_text}",
        encoding="utf-8",
    )
    case = BenchmarkCase(
        dataset="annocaselaw",
        case_id="anno-three-labels",
        source_split="all",
        input_text="Facts and procedural history.",
        gold_label="mixed",
        labels=["affirm", "reverse", "mixed"],
        label_descriptions={
            "affirm": "affirm in full",
            "reverse": "reverse in full",
            "mixed": "affirm some material parts and reverse others",
        },
        task_instruction="Predict the appellate outcome.",
    )

    prompt, _ = render_benchmark_prompt(config, case, "direct")
    schema = _response_schema(case.labels, "direct")

    assert all(f'- "{label}":' in prompt for label in case.labels)
    assert schema["properties"]["final_decision"]["enum"] == case.labels


def test_head_tail_truncation_is_explicit() -> None:
    text, metadata = _truncate_input(
        "0123456789" * 4,
        max_characters=20,
        strategy="head_tail",
    )
    assert text.startswith("012345678901234")
    assert text.endswith("56789")
    assert "middle omitted" in text
    assert metadata["truncated"] is True


def test_aggregate_and_paired_metrics() -> None:
    rows = []
    gold = ["accepted", "accepted", "rejected", "rejected"]
    direct = ["accepted", "rejected", "rejected", "accepted"]
    structured = ["accepted", "accepted", "rejected", "rejected"]
    for condition, predictions in (("direct", direct), ("structured", structured)):
        for index, (truth, prediction) in enumerate(zip(gold, predictions, strict=True)):
            rows.append(
                {
                    "dataset": "realistic_ljp_facts",
                    "case_id": f"case-{index}",
                    "condition": condition,
                    "gold_label": truth,
                    "prediction": prediction,
                    "answer_correct": int(truth == prediction),
                    "labels": ["rejected", "accepted"],
                    "elapsed_seconds": 1.0,
                    "prompt_tokens": 10,
                    "output_tokens": 2,
                    "input": {"truncated": False},
                }
            )
    aggregate, matrices = _aggregate_scores(rows)
    paired = _paired_comparisons(rows)
    direct_row = aggregate[aggregate["condition"] == "direct"].iloc[0]
    assert direct_row["accuracy"] == pytest.approx(0.5)
    assert matrices["realistic_ljp_facts/direct"]["matrix"] == [[1, 1], [1, 1]]
    assert paired.iloc[0]["structured_minus_direct"] == pytest.approx(0.5)
