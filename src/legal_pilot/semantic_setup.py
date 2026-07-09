from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .embeddings import OllamaEmbeddingBackend, TfidfSimilarityBackend
from .frontier_profiles import build_frontier_inputs, load_frontier_profiles
from .io_utils import write_jsonl
from .runner import load_cases


def resolve_project_file(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_similarity_backend(config: dict[str, Any]):
    backend = config["bot"].get("retrieval_backend", "tfidf")
    if backend == "tfidf":
        return TfidfSimilarityBackend()
    if backend != "ollama_embedding":
        raise ValueError(f"Unknown retrieval backend: {backend}")
    return OllamaEmbeddingBackend(
        base_url=config["bot"].get(
            "embedding_base_url", config["model"]["base_url"]
        ),
        model=config["bot"]["embedding_model"],
        cache_path=resolve_project_file(
            config["bot"]["embedding_cache_file"]
        ),
        timeout_seconds=config["bot"].get(
            "embedding_timeout_seconds",
            config["model"]["timeout_seconds"],
        ),
    )


def embedding_runtime_info(config: dict[str, Any]) -> dict[str, Any]:
    backend = build_similarity_backend(config)
    try:
        if not isinstance(backend, OllamaEmbeddingBackend):
            return {
                "backend": backend.name,
                "model": None,
                "digest": "builtin-tfidf",
            }
        info = backend.model_info()
        if not info:
            raise RuntimeError(
                f"Embedding model {backend.model!r} is not installed in Ollama."
            )
        return {
            "backend": backend.name,
            "model": backend.model,
            "digest": info.get("digest", "unknown"),
            "size": info.get("size"),
        }
    finally:
        close = getattr(backend, "close", None)
        if close:
            close()


def embedding_check(config: dict[str, Any]) -> dict[str, Any]:
    backend = build_similarity_backend(config)
    if not isinstance(backend, OllamaEmbeddingBackend):
        return {"backend": backend.name, "available": True}
    try:
        info = backend.model_info()
        if not info:
            raise RuntimeError(
                f"Embedding model {backend.model!r} is not installed in Ollama."
            )
        vectors = backend.embed(
            [
                "contract breach and unpaid debt",
                "failure to repay a contractual loan",
                "landlord possession of leased property",
            ]
        )
        related = float(vectors[0] @ vectors[1])
        unrelated = float(vectors[0] @ vectors[2])
        return {
            "backend": backend.name,
            "model": backend.model,
            "model_digest": info.get("digest"),
            "dimension": len(vectors[0]),
            "related_similarity": related,
            "unrelated_similarity": unrelated,
            "cache_path": str(backend.cache_path),
        }
    finally:
        backend.close()


def export_frontier_inputs(config: dict[str, Any]) -> dict[str, Any]:
    cases = [
        case
        for case in load_cases(config)
        if case.metadata.get("selection_split") in {"smoke", "evaluation"}
    ]
    output_dir = resolve_project_file(config["bot"]["frontier_export_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    packet = output_dir / "frontier_distillation_input.jsonl"
    write_jsonl(packet, build_frontier_inputs(cases))
    schema_path = output_dir / "frontier_profile.schema.json"
    from .models import FrontierLegalProblem

    schema_path.write_text(
        json.dumps(
            FrontierLegalProblem.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    instructions = output_dir / "INSTRUCTIONS.md"
    instructions.write_text(
        _frontier_instructions(len(cases)),
        encoding="utf-8",
    )
    return {
        "cases": len(cases),
        "packet": str(packet),
        "schema": str(schema_path),
        "instructions": str(instructions),
    }


def import_frontier_profiles(
    config: dict[str, Any], *, input_path: str | Path
) -> dict[str, Any]:
    cases = [
        case
        for case in load_cases(config)
        if case.metadata.get("selection_split") in {"smoke", "evaluation"}
    ]
    required = {case.case_id for case in cases}
    profiles = load_frontier_profiles(
        resolve_project_file(input_path),
        required_case_ids=required,
        valid_fact_ids_by_case={
            case.case_id: set(case.facts) for case in cases
        },
    )
    output = resolve_project_file(config["bot"]["frontier_profiles_file"])
    write_jsonl(
        output,
        [
            profiles[case_id].model_dump(mode="json")
            for case_id in sorted(profiles)
        ],
    )
    return {"profiles": len(profiles), "path": str(output)}


def _frontier_instructions(case_count: int) -> str:
    return f"""# Blinded frontier problem distillation

Process all {case_count} JSONL cases in a fresh Temporary Chat or Codex thread.
The packet contains no outcome labels or judgment decisions.

For each case, return one JSON object matching `frontier_profile.schema.json`.
Use only the supplied claim, requested remedy, parties, lawsuit type, and
F-numbered facts.

Requirements:

- Do not decide support or reject.
- Do not invent legal rules, defenses, authorities, or missing facts.
- `material_fact_ids` must contain only supplied F-numbers.
- State at most four genuinely dispositive questions.
- Keep `retrieval_summary` abstract: omit names, dates, amounts, citations,
  outcome language, and case-specific identifiers.
- Return JSONL in the same case order with no prose outside the records.
"""
