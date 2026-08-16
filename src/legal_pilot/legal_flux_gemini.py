from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from json_repair import repair_json

from .io_utils import read_jsonl, sha256_text, write_jsonl
from .legal_flux import (
    resolve_project_file,
    sanitize_flux_template,
    validate_template_pool,
)
from .legal_flux_chatgpt import TemplateBatchEncoder
from .legal_flux_deepseek import (
    TemplateApiClient,
    _batch_file_path,
    _messages,
    _read_text,
    _write_stage_manifest,
)
from .legal_flux_xsim import SentenceTransformerDenseEncoder
from .models import (
    LegalFluxCandidate,
    LegalFluxCandidateDraft,
    LegalFluxCandidateResponse,
    LegalFluxConsolidatedTemplateDraft,
    LegalFluxConsolidationResponse,
    LegalFluxGapAuditResponse,
    LegalFluxTemplate,
)


class GeminiTemplateClient:
    def __init__(self, config: dict[str, Any]):
        gemini_config = config.get("gemini", {})
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or gemini_config.get("project")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION") or gemini_config.get(
            "location", "global"
        )
        credentials_file = gemini_config.get("application_credentials_file")
        if credentials_file and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_file)
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        if not project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is not set and gemini.project is empty."
            )

        from google import genai
        from google.genai import types

        self.model = str(gemini_config.get("model", "gemini-3.1-pro-preview"))
        self.seed = int(config["project"]["seed"])
        temperature = gemini_config.get("temperature")
        self.temperature = float(temperature) if temperature is not None else None
        self.thinking_level = gemini_config.get("thinking_level")
        self.include_thoughts = bool(gemini_config.get("include_thoughts", False))
        self.max_output_tokens = int(gemini_config.get("max_output_tokens", 24000))
        self.client = genai.Client(
            vertexai=True,
            project=str(project),
            location=str(location),
            http_options=types.HttpOptions(
                api_version=str(gemini_config.get("api_version", "v1")),
                timeout=int(float(gemini_config.get("timeout_seconds", 600)) * 1000),
            ),
        )
        self.types = types

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        system_instruction = "\n\n".join(
            message["content"] for message in messages if message["role"] == "system"
        )
        contents = "\n\n".join(
            message["content"] for message in messages if message["role"] != "system"
        )
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction or None,
            "max_output_tokens": self.max_output_tokens,
            "seed": self.seed,
        }
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = response_schema
        if self.temperature is not None:
            config_kwargs["temperature"] = self.temperature
        if self.thinking_level:
            config_kwargs["thinking_config"] = self.types.ThinkingConfig(
                thinking_level=str(self.thinking_level),
                include_thoughts=self.include_thoughts,
            )
        config = self.types.GenerateContentConfig(**config_kwargs)
        start = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        usage = getattr(response, "usage_metadata", None)
        return {
            "content": getattr(response, "text", "") or "",
            "metadata": {
                "model": self.model,
                "seed": self.seed,
                "elapsed_seconds": time.perf_counter() - start,
                "prompt_tokens": _usage_attr(usage, "prompt_token_count"),
                "completion_tokens": _usage_attr(usage, "candidates_token_count"),
                "total_tokens": _usage_attr(usage, "total_token_count"),
                "thought_tokens": _usage_attr(usage, "thoughts_token_count"),
            },
        }

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        content = "\n\n".join(message["content"] for message in messages)
        response = self.client.models.count_tokens(model=self.model, contents=content)
        return int(getattr(response, "total_tokens", 0))


def run_gemini_template_workflow(
    config: dict[str, Any],
    *,
    stage: str = "candidates",
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    client: TemplateApiClient | None = None,
    dense_encoder: TemplateBatchEncoder | None = None,
) -> dict[str, Any]:
    normalized_stage = stage.replace("-", "_")
    if normalized_stage == "all":
        result: dict[str, Any] = {
            "stage": "all",
            "candidates": generate_gemini_template_candidates(
                config, limit=limit, force=force, dry_run=dry_run, client=client
            ),
        }
        if dry_run:
            result["merge"] = _dry_stage_or_blocked(
                lambda: generate_gemini_template_merge(
                    config, force=force, dry_run=True, client=client
                ),
                stage="merge",
            )
            result["audit"] = _dry_stage_or_blocked(
                lambda: generate_gemini_template_audit(
                    config, limit=limit, force=force, dry_run=True, client=client
                ),
                stage="audit",
            )
            return result
        result["merge"] = generate_gemini_template_merge(
            config, force=force, dry_run=dry_run, client=client
        )
        result["audit"] = generate_gemini_template_audit(
            config, limit=limit, force=force, dry_run=dry_run, client=client
        )
        return result
    if normalized_stage == "candidates":
        return generate_gemini_template_candidates(
            config, limit=limit, force=force, dry_run=dry_run, client=client
        )
    if normalized_stage == "merge":
        return generate_gemini_template_merge(
            config, force=force, dry_run=dry_run, client=client
        )
    if normalized_stage == "audit":
        return generate_gemini_template_audit(
            config, limit=limit, force=force, dry_run=dry_run, client=client
        )
    if normalized_stage == "similarity_audit":
        return generate_gemini_similarity_audit(
            config,
            dry_run=dry_run,
            dense_encoder=dense_encoder,
        )
    raise ValueError(
        "stage must be one of: candidates, merge, audit, similarity_audit, all."
    )


def _dry_stage_or_blocked(call: Any, *, stage: str) -> dict[str, Any]:
    try:
        return call()
    except RuntimeError as exc:
        return {"stage": stage, "dry_run": True, "blocked": str(exc)}


def generate_gemini_template_candidates(
    config: dict[str, Any],
    *,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    client: TemplateApiClient | None = None,
) -> dict[str, Any]:
    batch_root = _gemini_batch_root(config)
    prompt = _read_text(batch_root / "prompts" / "01_generate_candidate_templates.md")
    schema = _read_json_schema(
        batch_root / "legal_flux_candidate_response.schema.json"
    )
    manifest = _batch_manifest(batch_root)
    batches = manifest.get("batches", [])
    if limit is not None:
        batches = batches[:limit]
    output_dir = _gemini_root(config) / "03_candidate_templates"
    raw_dir = output_dir / "raw"
    planned = [
        _candidate_plan_row(batch_root, batch, prompt)
        for batch in batches
    ]
    if dry_run:
        return {
            "stage": "candidates",
            "dry_run": True,
            "planned_calls": len(planned),
            "output_dir": str(output_dir),
            "calls": planned,
        }
    client = client or GeminiTemplateClient(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for batch in batches:
        batch_id = str(batch["batch_id"])
        output_path = output_dir / f"{batch_id}_candidates.jsonl"
        raw_path = raw_dir / f"{batch_id}_raw.txt"
        if output_path.exists() and not force:
            records.append(
                {
                    "batch_id": batch_id,
                    "status": "skipped_existing",
                    "output_path": str(output_path),
                    "template_count": len(read_jsonl(output_path)),
                }
            )
            continue
        messages = _candidate_messages(batch_root, batch, prompt)
        prompt_tokens = _preflight_prompt_tokens(config, client, messages)
        response = _complete(client, messages, response_schema=schema)
        raw_text = str(response["content"])
        raw_path.write_text(raw_text, encoding="utf-8")
        parsed = _parse_model(raw_text, LegalFluxCandidateResponse)
        candidates = _validated_candidates(
            parsed.candidates,
            prefix=f"CAND_{batch_id}",
            allowed_case_ids=set(batch.get("case_ids", [])),
            minimum_support_cases=int(manifest.get("minimum_support_cases", 3)),
            maximum_candidates=int(manifest.get("max_candidates_per_batch", 5)),
            source=f"Gemini candidate response for {batch_id}",
        )
        write_jsonl(
            output_path,
            [candidate.model_dump(mode="json") for candidate in candidates],
        )
        records.append(
            {
                "batch_id": batch_id,
                "status": "ok",
                "output_path": str(output_path),
                "raw_path": str(raw_path),
                "template_count": len(candidates),
                "metadata": response.get("metadata", {}),
                "preflight_prompt_tokens": prompt_tokens,
                "prompt_sha256": sha256_text(messages[-1]["content"]),
                "output_sha256": sha256_text(output_path.read_text(encoding="utf-8")),
            }
        )
    return _write_stage_manifest(
        _gemini_root(config),
        "candidates",
        {
            "stage": "candidates",
            "output_dir": str(output_dir),
            "batch_count": len(batches),
            "records": records,
        },
    )


def generate_gemini_template_merge(
    config: dict[str, Any],
    *,
    force: bool = False,
    dry_run: bool = False,
    client: TemplateApiClient | None = None,
) -> dict[str, Any]:
    batch_root = _gemini_batch_root(config)
    output_root = _gemini_root(config)
    candidate_dir = output_root / "03_candidate_templates"
    output_path = output_root / "legal_flux_templates_gemini_merged.jsonl"
    raw_path = output_root / "legal_flux_templates_gemini_merged_raw.txt"
    candidate_paths = sorted(candidate_dir.glob("*_candidates.jsonl"))
    if not candidate_paths:
        raise RuntimeError("No Gemini candidate files found. Run candidates stage first.")
    expected_batch_ids = {
        str(batch["batch_id"]) for batch in _batch_manifest(batch_root).get("batches", [])
    }
    available_batch_ids = {
        path.name.removesuffix("_candidates.jsonl") for path in candidate_paths
    }
    missing_batch_ids = sorted(expected_batch_ids - available_batch_ids)
    unexpected_batch_ids = sorted(available_batch_ids - expected_batch_ids)
    if missing_batch_ids or unexpected_batch_ids:
        details = []
        if missing_batch_ids:
            details.append(f"missing batches: {missing_batch_ids[:10]}")
        if unexpected_batch_ids:
            details.append(f"unexpected batches: {unexpected_batch_ids[:10]}")
        raise RuntimeError(
            "Gemini consolidation requires exactly one candidate file for every "
            "current source batch; " + "; ".join(details)
        )
    prompt_text = _merge_prompt_text(batch_root, candidate_paths)
    schema = _read_json_schema(
        batch_root / "legal_flux_consolidation_response.schema.json"
    )
    if dry_run:
        return {
            "stage": "merge",
            "dry_run": True,
            "candidate_files": len(candidate_paths),
            "prompt_characters": len(prompt_text),
            "output_path": str(output_path),
        }
    if output_path.exists() and not force:
        return {
            "stage": "merge",
            "status": "skipped_existing",
            "output_path": str(output_path),
            "template_count": len(read_jsonl(output_path)),
        }
    client = client or GeminiTemplateClient(config)
    output_root.mkdir(parents=True, exist_ok=True)
    messages = _messages(prompt_text)
    prompt_tokens = _preflight_prompt_tokens(config, client, messages)
    response = _complete(client, messages, response_schema=schema)
    raw_text = str(response["content"])
    raw_path.write_text(raw_text, encoding="utf-8")
    parsed = _parse_model(raw_text, LegalFluxConsolidationResponse)
    candidates = [row for path in candidate_paths for row in read_jsonl(path)]
    templates, lineage = _finalize_consolidated_templates(
        parsed.templates,
        allowed_source_ids={str(row["candidate_id"]) for row in candidates},
        source="Gemini merge response",
    )
    validate_template_pool(templates)
    write_jsonl(output_path, [template.model_dump(mode="json") for template in templates])
    lineage_path = output_root / "legal_flux_templates_gemini_merged_lineage.json"
    lineage_path.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return _write_stage_manifest(
        output_root,
        "merge",
        {
            "stage": "merge",
            "status": "ok",
            "candidate_files": len(candidate_paths),
            "template_count": len(templates),
            "output_path": str(output_path),
            "raw_path": str(raw_path),
            "lineage_path": str(lineage_path),
            "metadata": response.get("metadata", {}),
            "preflight_prompt_tokens": prompt_tokens,
            "prompt_sha256": sha256_text(prompt_text),
            "output_sha256": sha256_text(output_path.read_text(encoding="utf-8")),
        },
    )


def generate_gemini_template_audit(
    config: dict[str, Any],
    *,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    client: TemplateApiClient | None = None,
) -> dict[str, Any]:
    batch_root = _gemini_batch_root(config)
    output_root = _gemini_root(config)
    pool_path = output_root / "legal_flux_templates_gemini_merged.jsonl"
    if not pool_path.exists():
        raise RuntimeError("No merged Gemini template pool found. Run merge stage first.")
    manifest = _batch_manifest(batch_root)
    all_batches = list(manifest.get("batches", []))
    batches = all_batches[:limit] if limit is not None else all_batches
    prompt = _read_text(batch_root / "prompts" / "03_coverage_audit_and_gap_fill.md")
    schema = _read_json_schema(batch_root / "legal_flux_gap_audit_response.schema.json")
    audit_dir = output_root / "04_gap_audits"
    raw_dir = audit_dir / "raw"
    planned = [
        {
            "batch_id": batch["batch_id"],
            "case_count": batch.get("case_count"),
            "prompt_characters": len(
                _gap_audit_prompt_text(batch_root, batch, pool_path, prompt)
            ),
        }
        for batch in batches
    ]
    if dry_run:
        return {
            "stage": "audit",
            "dry_run": True,
            "planned_calls": len(planned),
            "calls": planned,
            "audit_dir": str(audit_dir),
            "will_adjudicate": limit is None,
        }
    client = client or GeminiTemplateClient(config)
    audit_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for batch in batches:
        batch_id = str(batch["batch_id"])
        gap_path = audit_dir / f"{batch_id}_gap_candidates.jsonl"
        coverage_path = audit_dir / f"{batch_id}_coverage.json"
        raw_path = raw_dir / f"{batch_id}_raw.txt"
        if gap_path.exists() and coverage_path.exists() and not force:
            records.append(
                {
                    "batch_id": batch_id,
                    "status": "skipped_existing",
                    "gap_count": len(read_jsonl(gap_path)),
                    "gap_path": str(gap_path),
                }
            )
            continue
        prompt_text = _gap_audit_prompt_text(batch_root, batch, pool_path, prompt)
        messages = _messages(prompt_text)
        prompt_tokens = _preflight_prompt_tokens(config, client, messages)
        response = _complete(client, messages, response_schema=schema)
        raw_text = str(response["content"])
        raw_path.write_text(raw_text, encoding="utf-8")
        parsed = _parse_model(raw_text, LegalFluxGapAuditResponse)
        gaps = _validated_candidates(
            parsed.gap_candidates,
            prefix=f"GAP_{batch_id}",
            allowed_case_ids=set(batch.get("case_ids", [])),
            minimum_support_cases=int(manifest.get("minimum_support_cases", 3)),
            maximum_candidates=int(manifest.get("max_candidates_per_batch", 5)),
            source=f"Gemini gap audit response for {batch_id}",
        )
        write_jsonl(gap_path, [gap.model_dump(mode="json") for gap in gaps])
        coverage_path.write_text(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "coverage_analysis": parsed.coverage_analysis,
                    "gap_candidate_ids": [gap.candidate_id for gap in gaps],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        records.append(
            {
                "batch_id": batch_id,
                "status": "ok",
                "gap_count": len(gaps),
                "gap_path": str(gap_path),
                "coverage_path": str(coverage_path),
                "raw_path": str(raw_path),
                "metadata": response.get("metadata", {}),
                "preflight_prompt_tokens": prompt_tokens,
                "prompt_sha256": sha256_text(prompt_text),
            }
        )

    final_path: Path | None = None
    adjudication: dict[str, Any] | None = None
    if limit is None:
        final_path, adjudication = _adjudicate_gap_candidates(
            config,
            client=client,
            batch_root=batch_root,
            output_root=output_root,
            pool_path=pool_path,
            batches=all_batches,
            force=force,
        )
    return _write_stage_manifest(
        output_root,
        "audit",
        {
            "stage": "audit",
            "status": "ok" if limit is None else "partial",
            "batch_count": len(batches),
            "records": records,
            "gap_fill_count": sum(int(record.get("gap_count", 0)) for record in records),
            "final_pool_path": str(final_path) if final_path else None,
            "adjudication": adjudication,
        },
    )


def generate_gemini_similarity_audit(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    dense_encoder: TemplateBatchEncoder | None = None,
) -> dict[str, Any]:
    output_root = _gemini_root(config)
    final_path = output_root / "legal_flux_templates_gemini_final.jsonl"
    merged_path = output_root / "legal_flux_templates_gemini_merged.jsonl"
    pool_path = final_path if final_path.exists() else merged_path
    if not pool_path.exists():
        raise RuntimeError("No Gemini template pool found for similarity audit.")
    templates = [LegalFluxTemplate.model_validate(row) for row in read_jsonl(pool_path)]
    flux_config = config["legal_flux"]
    review_threshold = float(
        flux_config.get("template_similarity_review_threshold", 0.90)
    )
    duplicate_threshold = float(
        flux_config.get("template_similarity_likely_duplicate_threshold", 0.95)
    )
    if not 0 <= review_threshold <= duplicate_threshold <= 1:
        raise ValueError("Template similarity thresholds must satisfy 0 <= review <= duplicate <= 1.")
    report_path = output_root / "template_similarity_audit.json"
    if dry_run:
        return {
            "stage": "similarity_audit",
            "dry_run": True,
            "template_count": len(templates),
            "pool_path": str(pool_path),
            "report_path": str(report_path),
            "review_threshold": review_threshold,
            "likely_duplicate_threshold": duplicate_threshold,
        }
    if dense_encoder is None:
        dense_encoder = SentenceTransformerDenseEncoder(
            str(flux_config.get("template_batch_embedding_model", "BAAI/bge-m3")),
            device=flux_config.get("template_batch_device") or None,
            max_length=int(flux_config.get("template_batch_embedding_max_length", 8192)),
        )
    texts = [_template_similarity_text(template) for template in templates]
    embeddings = np.asarray(
        dense_encoder.encode(
            texts,
            batch_size=int(flux_config.get("template_batch_embedding_batch_size", 8)),
        ),
        dtype=np.float32,
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    similarity = embeddings @ embeddings.T
    flags = []
    for left in range(len(templates)):
        for right in range(left + 1, len(templates)):
            score = float(similarity[left, right])
            if score < review_threshold:
                continue
            flags.append(
                {
                    "left_template_id": templates[left].template_id,
                    "left_template_name": templates[left].template_name,
                    "right_template_id": templates[right].template_id,
                    "right_template_name": templates[right].template_name,
                    "cosine_similarity": round(score, 6),
                    "severity": (
                        "likely_duplicate"
                        if score >= duplicate_threshold
                        else "manual_review"
                    ),
                }
            )
    flags.sort(key=lambda row: row["cosine_similarity"], reverse=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "similarity_audit",
        "pool_path": str(pool_path),
        "template_count": len(templates),
        "embedding_model": dense_encoder.model_name,
        "review_threshold": review_threshold,
        "likely_duplicate_threshold": duplicate_threshold,
        "flag_count": len(flags),
        "likely_duplicate_count": sum(
            row["severity"] == "likely_duplicate" for row in flags
        ),
        "flags": flags,
        "note": "Flags are diagnostic only; no template is removed automatically.",
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**report, "report_path": str(report_path)}


def _batch_manifest(batch_root: Path) -> dict[str, Any]:
    path = batch_root / "batch_manifest.json"
    if not path.exists():
        raise RuntimeError("Batch manifest not found. Run flux-export-gemini-batches first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_plan_row(
    batch_root: Path,
    batch: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    messages = _candidate_messages(batch_root, batch, prompt)
    return {
        "batch_id": batch["batch_id"],
        "kind": batch.get("kind"),
        "case_count": batch.get("case_count"),
        "prompt_characters": len(messages[-1]["content"]),
    }


def _candidate_messages(
    batch_root: Path,
    batch: dict[str, Any],
    prompt: str,
) -> list[dict[str, str]]:
    batch_text = _read_text(_batch_file_path(batch, batch_root))
    return _messages(
        f"""{prompt}

BATCH_ID:
{batch["batch_id"]}

BATCH LABEL:
{batch.get("label", "")}

SUPPLIED SOURCE CASES:
```jsonl
{batch_text}
```
"""
    )


def _merge_prompt_text(batch_root: Path, candidate_paths: list[Path]) -> str:
    prompt = _read_text(batch_root / "prompts" / "02_merge_deduplicate_templates.md")
    coverage = _read_text(batch_root / "coverage_summary.json")
    sections = [
        f"## {path.name}\n\n```jsonl\n{_read_text(path)}\n```"
        for path in candidate_paths
    ]
    return f"""{prompt}

SOURCE COVERAGE SUMMARY:
```json
{coverage}
```

COMPLETE CANDIDATE SET:
{chr(10).join(sections)}
"""


def _gap_audit_prompt_text(
    batch_root: Path,
    batch: dict[str, Any],
    pool_path: Path,
    prompt: str,
) -> str:
    batch_text = _read_text(_batch_file_path(batch, batch_root))
    pool_text = _read_text(pool_path)
    return f"""{prompt}

BATCH_ID:
{batch["batch_id"]}

BATCH LABEL:
{batch.get("label", "")}

CURRENT CONSOLIDATED TEMPLATE LIBRARY:
```jsonl
{pool_text}
```

SUPPLIED SOURCE CASES:
```jsonl
{batch_text}
```
"""


def _adjudicate_gap_candidates(
    config: dict[str, Any],
    *,
    client: TemplateApiClient,
    batch_root: Path,
    output_root: Path,
    pool_path: Path,
    batches: list[dict[str, Any]],
    force: bool,
) -> tuple[Path, dict[str, Any]]:
    audit_dir = output_root / "04_gap_audits"
    gap_paths = [
        audit_dir / f"{batch['batch_id']}_gap_candidates.jsonl"
        for batch in batches
    ]
    missing = [str(path) for path in gap_paths if not path.exists()]
    if missing:
        raise RuntimeError(
            "Gap adjudication requires every batch audit. Missing: "
            + ", ".join(missing[:5])
        )
    gap_rows = [row for path in gap_paths for row in read_jsonl(path)]
    final_path = output_root / "legal_flux_templates_gemini_final.jsonl"
    if not gap_rows:
        shutil.copyfile(pool_path, final_path)
        return final_path, {
            "status": "copied_merged_pool_no_gaps",
            "gap_candidate_count": 0,
            "template_count": len(read_jsonl(final_path)),
        }
    if final_path.exists() and not force:
        return final_path, {
            "status": "skipped_existing",
            "gap_candidate_count": len(gap_rows),
            "template_count": len(read_jsonl(final_path)),
        }
    prompt = f"""You are performing the final global adjudication of proposed gaps in
a consolidated LegalFlux template library.

The current library has already passed global consolidation. Change it only
where a gap candidate adds a genuinely missing, reusable legal reasoning
operation at a middle-to-high abstraction level. Reject or merge any gap
candidate that is already covered, subsumed, weakly supported, overly broad,
overly specific, or directionally tied to an outcome. There is no target final
template count.

Return the complete resulting library as one JSON object matching the
consolidation output schema. Use current LF template IDs and GAP candidate IDs
in source_candidate_ids for provenance. Template IDs will be reassigned
deterministically afterward.

CURRENT LIBRARY:
```jsonl
{_read_text(pool_path)}
```

PROPOSED GAP CANDIDATES:
```jsonl
{chr(10).join(json.dumps(row, ensure_ascii=False) for row in gap_rows)}
```
"""
    schema = _read_json_schema(
        batch_root / "legal_flux_consolidation_response.schema.json"
    )
    messages = _messages(prompt)
    prompt_tokens = _preflight_prompt_tokens(config, client, messages)
    response = _complete(client, messages, response_schema=schema)
    raw_text = str(response["content"])
    raw_path = output_root / "legal_flux_gap_adjudication_raw.txt"
    raw_path.write_text(raw_text, encoding="utf-8")
    parsed = _parse_model(raw_text, LegalFluxConsolidationResponse)
    allowed_source_ids = {
        str(row["template_id"]) for row in read_jsonl(pool_path)
    } | {str(row["candidate_id"]) for row in gap_rows}
    templates, lineage = _finalize_consolidated_templates(
        parsed.templates,
        allowed_source_ids=allowed_source_ids,
        source="Gemini gap adjudication response",
    )
    validate_template_pool(templates)
    write_jsonl(final_path, [template.model_dump(mode="json") for template in templates])
    lineage_path = output_root / "legal_flux_templates_gemini_final_lineage.json"
    lineage_path.write_text(
        json.dumps(lineage, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return final_path, {
        "status": "ok",
        "gap_candidate_count": len(gap_rows),
        "template_count": len(templates),
        "raw_path": str(raw_path),
        "lineage_path": str(lineage_path),
        "preflight_prompt_tokens": prompt_tokens,
        "metadata": response.get("metadata", {}),
    }


def _validated_candidates(
    drafts: list[LegalFluxCandidateDraft],
    *,
    prefix: str,
    allowed_case_ids: set[str],
    minimum_support_cases: int,
    maximum_candidates: int,
    source: str,
) -> list[LegalFluxCandidate]:
    if len(drafts) > maximum_candidates:
        raise ValueError(
            f"{source} returned {len(drafts)} candidates; maximum is {maximum_candidates}."
        )
    candidates = []
    for index, draft in enumerate(drafts, start=1):
        support_ids = list(dict.fromkeys(draft.supporting_case_ids))
        unknown = sorted(set(support_ids) - allowed_case_ids)
        if unknown:
            raise ValueError(f"{source} cited unknown supporting cases: {unknown}.")
        if len(support_ids) < minimum_support_cases:
            raise ValueError(
                f"{source} candidate {index} has {len(support_ids)} supporting cases; "
                f"minimum is {minimum_support_cases}."
            )
        if draft.support_count != len(support_ids):
            raise ValueError(
                f"{source} candidate {index} support_count does not match "
                "supporting_case_ids."
            )
        candidates.append(
            LegalFluxCandidate(
                candidate_id=f"{prefix}_{index:02d}",
                **draft.model_dump(mode="python", exclude={"supporting_case_ids"}),
                supporting_case_ids=support_ids,
            )
        )
    return candidates


def _finalize_consolidated_templates(
    drafts: list[LegalFluxConsolidatedTemplateDraft],
    *,
    allowed_source_ids: set[str],
    source: str,
) -> tuple[list[LegalFluxTemplate], dict[str, list[str]]]:
    templates = []
    lineage: dict[str, list[str]] = {}
    used_source_ids: set[str] = set()
    for index, draft in enumerate(drafts, start=1):
        source_ids = list(dict.fromkeys(draft.source_candidate_ids))
        unknown = sorted(set(source_ids) - allowed_source_ids)
        if unknown:
            raise ValueError(f"{source} cited unknown candidate IDs: {unknown}.")
        repeated = sorted(set(source_ids) & used_source_ids)
        if repeated:
            raise ValueError(f"{source} reused candidate IDs across templates: {repeated}.")
        used_source_ids.update(source_ids)
        template_id = f"LF{index:03d}"
        template = sanitize_flux_template(
            LegalFluxTemplate(
                template_id=template_id,
                **draft.model_dump(mode="python", exclude={"source_candidate_ids"}),
            )
        )
        templates.append(template)
        lineage[template_id] = source_ids
    if not templates:
        raise ValueError(f"{source} did not retain any templates.")
    return templates, lineage


def _parse_model(text: str, model_type: Any) -> Any:
    try:
        return model_type.model_validate_json(text)
    except Exception:
        repaired = repair_json(text, return_objects=True)
        return model_type.model_validate(repaired)


def _complete(
    client: TemplateApiClient,
    messages: list[dict[str, str]],
    *,
    response_schema: dict[str, Any],
) -> dict[str, Any]:
    try:
        return client.complete(messages, response_schema=response_schema)  # type: ignore[call-arg]
    except TypeError as exc:
        if "response_schema" not in str(exc):
            raise
        return client.complete(messages)


def _preflight_prompt_tokens(
    config: dict[str, Any],
    client: TemplateApiClient,
    messages: list[dict[str, str]],
) -> int | None:
    counter = getattr(client, "count_tokens", None)
    if counter is None:
        return None
    tokens = int(counter(messages))
    maximum = int(config.get("gemini", {}).get("max_input_tokens", 900000))
    if tokens > maximum:
        raise ValueError(
            f"Gemini prompt has {tokens} tokens, exceeding configured maximum {maximum}."
        )
    return tokens


def _read_json_schema(path: Path) -> dict[str, Any]:
    return json.loads(_read_text(path))


def _template_similarity_text(template: LegalFluxTemplate) -> str:
    return "\n".join(
        [
            template.template_name,
            " ".join(template.knowledge_tags),
            template.description,
            template.application_scenario,
            *template.reasoning_flow,
        ]
    )


def _gemini_batch_root(config: dict[str, Any]) -> Path:
    return resolve_project_file(
        config["legal_flux"].get(
            "gemini_batch_dir",
            config["legal_flux"].get(
                "chatgpt_batch_dir",
                "reports/legal_flux/template_distillation/gemini31_pro_batches",
            ),
        )
    )


def _gemini_root(config: dict[str, Any]) -> Path:
    return resolve_project_file(
        config["legal_flux"].get(
            "gemini_template_dir",
            "reports/legal_flux/template_distillation/gemini31_pro_api",
        )
    )


def _usage_attr(usage: Any, name: str) -> int | None:
    if usage is None:
        return None
    return getattr(usage, name, None)
