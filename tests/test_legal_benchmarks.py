from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_pilot.config import load_config
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
    _benchmark_flux_case,
    _benchmark_label_from_flux_decision,
    _paired_comparisons,
    _response_schema,
    _truncate_input,
    render_benchmark_prompt,
)
from legal_pilot.legal_flux_runner import (
    _rf_decision_label_instruction,
    _rf_review_output_requirement,
)
from legal_pilot.prompting import render_prompt


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


def test_realistic_ljp_case_adapts_to_legal_flux_without_exposing_gold() -> None:
    config = _config(Path("."))
    config["benchmarks"]["max_input_characters"] = 100
    case = _case("test-case", "accepted", split="test").model_copy(
        update={"input_text": "First fact.\nSecond fact."}
    )

    normalized, metadata = _benchmark_flux_case(config, case)

    assert normalized.dataset == "realistic_ljp_facts"
    assert normalized.facts == {"case_text": "First fact.\nSecond fact."}
    assert normalized.gold_answer == "accepted"
    assert "accepted" not in normalized.claim.lower()
    assert metadata["input_format"] == "whole_fact_text"
    assert _benchmark_label_from_flux_decision("support") == "accepted"
    assert _benchmark_label_from_flux_decision("reject") == "rejected"
    assert _benchmark_label_from_flux_decision("accepted") == "accepted"
    assert _benchmark_label_from_flux_decision("rejected") == "rejected"


def test_realistic_ljp_rf_prompts_receive_only_whole_fact_text() -> None:
    config = load_config(
        Path(__file__).parents[1]
        / "configs"
        / "legal_benchmarks_cjpe_flux.cluster.yaml"
    )
    case, _ = _benchmark_flux_case(
        config,
        _case("ljp-prompt", "accepted", split="test").model_copy(
            update={"input_text": "First fact.\nSecond fact."}
        ),
    )
    planner, _ = render_prompt(config, "legal_flux/rf_plan", case, max_steps=4)
    executor, _ = render_prompt(
        config,
        "legal_flux/instantiate",
        case,
        prior_artifacts=[],
        trajectory_step={"step_id": "S1"},
        selected_template={"template_id": "LF001"},
    )
    reviewer, _ = render_prompt(
        config,
        "legal_flux/rf_review",
        case,
        review_output_requirement="Return the final outcome.",
        executed_trajectory=[],
        remaining_steps=[],
    )

    for prompt in (planner, executor, reviewer):
        assert "First fact.\nSecond fact." in prompt
        assert "PLAINTIFF'S CLAIM:" not in prompt
        assert "PARTIES:" not in prompt
        assert "F1:" not in prompt
    assert "Indian Supreme Court case" in planner
    assert "F-numbered facts" not in executor
    assert "F-numbered facts" not in reviewer
    assert "do not decide the overall outcome" in executor
    label_instruction = _rf_decision_label_instruction(case)
    assert "what the appeal or petition" in " ".join(planner.split())
    assert "predicted Supreme Court disposition" in label_instruction
    assert "that requested relief" in label_instruction
    assert "plaintiff's claim" not in label_instruction
    assert "accepted means" in label_instruction
    assert "rejected means" in label_instruction
    assert "support means" not in label_instruction

    review_requirement = _rf_review_output_requirement(
        case,
        remaining_step_limit=3,
        max_steps=4,
        force_final_answer=False,
    )
    assert 'final_decision must be exactly "accepted" or "rejected"' in (
        review_requirement
    )
    assert "accepted means" in review_requirement
    assert "rejected means" in review_requirement
    assert '"support" or "reject"' not in review_requirement

    forced_requirement = _rf_review_output_requirement(
        case,
        remaining_step_limit=0,
        max_steps=4,
        force_final_answer=True,
    )
    assert forced_requirement.startswith("No remaining abstract steps are available.")
    assert 'final_decision must be exactly "accepted" or "rejected"' in (
        forced_requirement
    )


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
