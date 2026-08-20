from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import resolve_project_path
from .io_utils import atomic_write_json, read_jsonl, write_jsonl


ANN_CASELAW_REPOSITORY = "https://github.com/anonymouspolar1/annocaselaw.git"
REALISTIC_LJP_URL = (
    "https://huggingface.co/datasets/L-NLProc/Realistic_LJP_Facts/"
    "resolve/main/Realistic_LJP_Facts.csv"
)
IL_TUR_BASE_URL = (
    "https://huggingface.co/datasets/Exploration-Lab/IL-TUR/resolve/script"
)
SUPPORTED_BENCHMARKS = (
    "annocaselaw",
    "realistic_ljp_facts",
    "il_tur_cjpe",
)


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    case_id: str
    source_split: str
    input_text: str
    gold_label: str
    labels: list[str] = Field(min_length=2)
    label_descriptions: dict[str, str]
    task_instruction: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_labels(self) -> "BenchmarkCase":
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Benchmark labels must be unique.")
        if self.gold_label not in self.labels:
            raise ValueError(
                f"Gold label {self.gold_label!r} is not in {self.labels!r}."
            )
        if set(self.label_descriptions) != set(self.labels):
            raise ValueError("Every benchmark label requires one description.")
        if not self.input_text.strip():
            raise ValueError("Benchmark input text cannot be empty.")
        return self


class DatasetAccessError(RuntimeError):
    pass


def prepare_legal_benchmarks(
    config: dict[str, Any],
    *,
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    selected = _validated_dataset_names(datasets)
    raw_root = benchmark_path(config, "raw_dir")
    processed_root = benchmark_path(config, "processed_dir")
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for dataset in selected:
        cases, source = _prepare_dataset(dataset, raw_root)
        _validate_unique_cases(cases)
        dataset_dir = processed_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(
            dataset_dir / "cases.jsonl",
            [case.model_dump(mode="json") for case in cases],
        )
        pilot_cases = _select_pilot_cases(cases, config, dataset)
        pilot_ids = [case.case_id for case in pilot_cases]
        atomic_write_json(dataset_dir / "pilot_case_ids.json", pilot_ids)
        manifest = _dataset_manifest(
            dataset,
            cases,
            pilot_cases,
            source=source,
        )
        atomic_write_json(dataset_dir / "prepare_manifest.json", manifest)
        results[dataset] = manifest

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "processed_root": str(processed_root),
        "datasets": results,
    }


def load_benchmark_cases(
    config: dict[str, Any],
    dataset: str,
) -> list[BenchmarkCase]:
    _validated_dataset_names([dataset])
    path = benchmark_path(config, "processed_dir") / dataset / "cases.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared benchmark cases not found for {dataset}: {path}. "
            f"Run benchmark-prepare --datasets {dataset} first."
        )
    return [BenchmarkCase.model_validate(row) for row in read_jsonl(path)]


def select_benchmark_cases(
    config: dict[str, Any],
    *,
    datasets: list[str] | None,
    subset: str,
    case_limit: int | None = None,
) -> list[BenchmarkCase]:
    if subset not in {"pilot", "full"}:
        raise ValueError("Benchmark subset must be 'pilot' or 'full'.")
    if case_limit is not None and case_limit < 1:
        raise ValueError("case_limit must be at least 1.")
    selected: list[BenchmarkCase] = []
    for dataset in _validated_dataset_names(datasets):
        cases = load_benchmark_cases(config, dataset)
        if subset == "pilot":
            ids_path = (
                benchmark_path(config, "processed_dir")
                / dataset
                / "pilot_case_ids.json"
            )
            if not ids_path.exists():
                raise FileNotFoundError(
                    f"Pilot IDs not found for {dataset}; rerun benchmark-prepare."
                )
            pilot_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            by_id = {case.case_id: case for case in cases}
            missing = [case_id for case_id in pilot_ids if case_id not in by_id]
            if missing:
                raise ValueError(
                    f"Pilot file for {dataset} contains unknown IDs: {missing[:5]}"
                )
            chosen = [by_id[case_id] for case_id in pilot_ids]
        else:
            full_split = str(
                config["benchmarks"]["datasets"][dataset].get(
                    "full_split",
                    "all",
                )
            )
            chosen = [
                case
                for case in cases
                if full_split == "all" or case.source_split == full_split
            ]
        if not chosen:
            raise ValueError(f"No {subset} cases selected for {dataset}.")
        if case_limit is not None:
            chosen = chosen[:case_limit]
        selected.extend(chosen)
    return selected


def benchmark_path(config: dict[str, Any], key: str) -> Path:
    try:
        value = config["benchmarks"]["paths"][key]
    except KeyError as exc:
        raise KeyError(f"Missing benchmarks.paths.{key} in config.") from exc
    return resolve_project_path(config, value)


def _prepare_dataset(
    dataset: str,
    raw_root: Path,
) -> tuple[list[BenchmarkCase], dict[str, Any]]:
    if dataset == "annocaselaw":
        return _prepare_annocaselaw(raw_root)
    if dataset == "realistic_ljp_facts":
        return _prepare_realistic_ljp(raw_root)
    if dataset == "il_tur_cjpe":
        return _prepare_il_tur_cjpe(raw_root)
    raise AssertionError(dataset)


def _prepare_annocaselaw(
    raw_root: Path,
) -> tuple[list[BenchmarkCase], dict[str, Any]]:
    repository = raw_root / "annocaselaw"
    clean_csv = repository / "data" / "csvs" / "clean_df.csv"
    if not clean_csv.exists():
        if repository.exists() and any(repository.iterdir()):
            raise RuntimeError(
                f"AnnoCaseLaw directory exists but is incomplete: {repository}"
            )
        repository.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                ANN_CASELAW_REPOSITORY,
                str(repository),
            ],
            check=True,
        )

    frame = pd.read_csv(clean_csv)
    cases_dir = repository / "data" / "cases_json"
    labels = ["affirm", "reverse", "mixed"]
    descriptions = {
        "affirm": "affirm the lower-court judgment in full",
        "reverse": "reverse or vacate the lower-court judgment in full",
        "mixed": "affirm some material parts and reverse or vacate others",
    }
    cases: list[BenchmarkCase] = []
    for row in frame.itertuples(index=False):
        source_path = cases_dir / str(row.file_name)
        payload = json.loads(source_path.read_text(encoding="utf-8-sig"))
        annotations = payload.get("annotations") or {}
        facts = _clean_text_list(annotations.get("Facts"))
        procedure = _clean_text_list(annotations.get("Procedural History"))
        if not facts or not procedure:
            raise ValueError(
                f"AnnoCaseLaw case lacks Task 1a input components: {source_path.name}"
            )
        text = (
            "FACTS OF THE CASE:\n"
            + "\n".join(facts)
            + "\n\nPROCEDURAL HISTORY:\n"
            + "\n".join(procedure)
        )
        cases.append(
            BenchmarkCase(
                dataset="annocaselaw",
                case_id=f"annocaselaw-{int(row.case_id):04d}",
                source_split="all",
                input_text=text,
                gold_label=str(row.outcome).strip().lower(),
                labels=labels,
                label_descriptions=descriptions,
                task_instruction=(
                    "Predict the U.S. appellate outcome in this civil-negligence "
                    "case from the expert-annotated facts and procedural history."
                ),
                metadata={
                    "input_variant": "task_1a_facts_plus_procedural_history",
                    "source_file": source_path.name,
                    "original_characters": len(text),
                    "source_url": ANN_CASELAW_REPOSITORY,
                    "license": "MIT",
                },
            )
        )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return cases, {
        "repository": ANN_CASELAW_REPOSITORY,
        "commit": commit,
        "clean_csv_sha256": _sha256_file(clean_csv),
    }


def _prepare_realistic_ljp(
    raw_root: Path,
) -> tuple[list[BenchmarkCase], dict[str, Any]]:
    csv_path = raw_root / "realistic_ljp_facts" / "Realistic_LJP_Facts.csv"
    _download_file(REALISTIC_LJP_URL, csv_path)
    frame = pd.read_csv(csv_path)
    required = {"text", "label", "split", "name"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Realistic_LJP_Facts is missing columns: {sorted(required - set(frame))}"
        )
    labels = ["rejected", "accepted"]
    descriptions = {
        "rejected": "the appeal or petition is rejected/dismissed",
        "accepted": "the appeal or petition is accepted/allowed",
    }
    cases: list[BenchmarkCase] = []
    for row in frame.itertuples(index=False):
        split = str(row.split).strip().lower()
        if split not in {"dev", "test"}:
            continue
        text = str(row.text).strip()
        if not text:
            raise ValueError(f"Empty Realistic_LJP_Facts input: {row.name}")
        cases.append(
            BenchmarkCase(
                dataset="realistic_ljp_facts",
                case_id=f"realistic-ljp-{str(row.name).strip()}",
                source_split=split,
                input_text=text,
                gold_label=_binary_judgment_label(row.label),
                labels=labels,
                label_descriptions=descriptions,
                task_instruction=(
                    "Predict whether the Indian court accepts or rejects the "
                    "appeal or petition using the supplied pre-decision fact section."
                ),
                metadata={
                    "input_variant": "facts",
                    "source_name": str(row.name),
                    "original_characters": len(text),
                    "source_url": (
                        "https://huggingface.co/datasets/"
                        "L-NLProc/Realistic_LJP_Facts"
                    ),
                    "license": "Apache-2.0",
                },
            )
        )
    return cases, {
        "url": REALISTIC_LJP_URL,
        "csv_sha256": _sha256_file(csv_path),
        "raw_rows": int(len(frame)),
        "prepared_splits": ["dev", "test"],
    }


def _prepare_il_tur_cjpe(
    raw_root: Path,
) -> tuple[list[BenchmarkCase], dict[str, Any]]:
    token = _hugging_face_token()
    if not token:
        raise DatasetAccessError(
            "IL-TUR files are gated. Log in at "
            "https://huggingface.co/datasets/Exploration-Lab/IL-TUR, accept "
            "the conditions, then run `hf auth login` or set HF_TOKEN to a read "
            "token before rerunning benchmark-prepare --datasets il_tur_cjpe."
        )
    dataset_dir = raw_root / "il_tur_cjpe"
    files = {
        "multi_dev": "cjpe/multi_dev.jsonl",
        "test": "cjpe/test.jsonl",
    }
    local_paths: dict[str, Path] = {}
    for split, relative in files.items():
        destination = dataset_dir / f"{split}.jsonl"
        _download_file(
            f"{IL_TUR_BASE_URL}/{relative}",
            destination,
            headers={"Authorization": f"Bearer {token}"},
            gated_name="IL-TUR",
        )
        local_paths[split] = destination

    labels = ["rejected", "accepted"]
    descriptions = {
        "rejected": "the appeal is rejected/dismissed",
        "accepted": "at least one appeal is accepted/allowed",
    }
    cases: list[BenchmarkCase] = []
    for split, path in local_paths.items():
        for row in read_jsonl(path):
            text_value = row.get("text")
            text = (
                "\n".join(str(value) for value in text_value)
                if isinstance(text_value, list)
                else str(text_value or "")
            ).strip()
            if not text:
                raise ValueError(f"Empty IL-TUR CJPE input in {path}: {row.get('id')}")
            source_id = str(row.get("id") or "").strip()
            if not source_id:
                raise ValueError(f"IL-TUR CJPE row lacks id in {path}.")
            cases.append(
                BenchmarkCase(
                    dataset="il_tur_cjpe",
                    case_id=f"il-tur-cjpe-{source_id}",
                    source_split=split,
                    input_text=text,
                    gold_label=_binary_judgment_label(row.get("label")),
                    labels=labels,
                    label_descriptions=descriptions,
                    task_instruction=(
                        "Predict whether the Indian Supreme Court accepts or "
                        "rejects the appeal from the supplied case document."
                    ),
                    metadata={
                        "input_variant": "official_cjpe_document",
                        "source_id": source_id,
                        "original_characters": len(text),
                        "source_url": (
                            "https://huggingface.co/datasets/Exploration-Lab/IL-TUR"
                        ),
                        "license": "CC-BY-NC-SA-4.0",
                    },
                )
            )
    return cases, {
        "repository": "Exploration-Lab/IL-TUR",
        "revision": "script",
        "files": {
            split: {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for split, path in local_paths.items()
        },
    }


def _hugging_face_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except (ImportError, OSError):
        return None


def _select_pilot_cases(
    cases: list[BenchmarkCase],
    config: dict[str, Any],
    dataset: str,
) -> list[BenchmarkCase]:
    dataset_config = config["benchmarks"]["datasets"][dataset]
    split = str(dataset_config.get("pilot_split", "all"))
    pool = [
        case for case in cases if split == "all" or case.source_split == split
    ]
    if not pool:
        raise ValueError(f"No cases in configured pilot split {split!r} for {dataset}.")
    size_value = dataset_config.get("pilot_size")
    if size_value in (None, "all"):
        return sorted(pool, key=lambda case: case.case_id)
    size = int(size_value)
    seed = int(config["project"]["seed"])
    return _balanced_sample(pool, count=size, seed=seed)


def _balanced_sample(
    cases: list[BenchmarkCase],
    *,
    count: int,
    seed: int,
) -> list[BenchmarkCase]:
    if count < 1 or count > len(cases):
        raise ValueError(f"Requested {count} pilot rows from a pool of {len(cases)}.")
    rng = random.Random(seed)
    grouped: dict[str, list[BenchmarkCase]] = {}
    for case in cases:
        grouped.setdefault(case.gold_label, []).append(case)
    labels = sorted(grouped)
    for values in grouped.values():
        values.sort(key=lambda case: case.case_id)
        rng.shuffle(values)
    base, remainder = divmod(count, len(labels))
    targets = {
        label: base + (1 if index < remainder else 0)
        for index, label in enumerate(labels)
    }
    selected: list[BenchmarkCase] = []
    for label in labels:
        take = min(targets[label], len(grouped[label]))
        selected.extend(grouped[label][:take])
    if len(selected) < count:
        chosen_ids = {case.case_id for case in selected}
        remaining = [case for case in cases if case.case_id not in chosen_ids]
        remaining.sort(key=lambda case: case.case_id)
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    if len(selected) != count:
        raise ValueError("Could not draw the requested balanced benchmark pilot.")
    rng.shuffle(selected)
    return selected


def _dataset_manifest(
    dataset: str,
    cases: list[BenchmarkCase],
    pilot_cases: list[BenchmarkCase],
    *,
    source: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "cases": len(cases),
        "splits": _nested_counts(cases),
        "pilot_cases": len(pilot_cases),
        "pilot_split_counts": _nested_counts(pilot_cases),
        "pilot_case_ids_sha256": _sha256_strings(
            case.case_id for case in pilot_cases
        ),
        "source": source,
    }


def _nested_counts(cases: Iterable[BenchmarkCase]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for case in cases:
        counts.setdefault(case.source_split, Counter())[case.gold_label] += 1
    return {
        split: dict(sorted(label_counts.items()))
        for split, label_counts in sorted(counts.items())
    }


def _download_file(
    url: str,
    destination: Path,
    *,
    headers: dict[str, str] | None = None,
    gated_name: str | None = None,
) -> None:
    if destination.exists() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        with httpx.stream(
            "GET",
            url,
            headers=headers,
            follow_redirects=True,
            timeout=600,
        ) as response:
            if gated_name and response.status_code in {401, 403}:
                raise DatasetAccessError(
                    f"{gated_name} denied the download. Confirm that your HF "
                    "account accepted the dataset conditions and that HF_TOKEN "
                    "is a valid read token for that account."
                )
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        partial.replace(destination)
    finally:
        if partial.exists() and not destination.exists():
            partial.unlink()


def _binary_judgment_label(value: Any) -> str:
    if isinstance(value, bool):
        return "accepted" if value else "rejected"
    if isinstance(value, (int, float)) and not pd.isna(value):
        integer = int(value)
        if float(value) == integer and integer in {0, 1}:
            return "accepted" if integer == 1 else "rejected"
    text = str(value).strip().lower()
    mapping = {
        "0": "rejected",
        "1": "accepted",
        "reject": "rejected",
        "rejected": "rejected",
        "dismissed": "rejected",
        "accept": "accepted",
        "accepted": "accepted",
        "allowed": "accepted",
    }
    if text not in mapping:
        raise ValueError(f"Unknown binary judgment label: {value!r}")
    return mapping[text]


def _clean_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _validated_dataset_names(datasets: list[str] | None) -> list[str]:
    selected = list(datasets or SUPPORTED_BENCHMARKS)
    unknown = sorted(set(selected) - set(SUPPORTED_BENCHMARKS))
    if unknown:
        raise ValueError(f"Unsupported legal benchmarks: {unknown}")
    if not selected:
        raise ValueError("At least one legal benchmark is required.")
    return list(dict.fromkeys(selected))


def _validate_unique_cases(cases: list[BenchmarkCase]) -> None:
    ids = [case.case_id for case in cases]
    duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate benchmark case IDs: {duplicates[:5]}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
