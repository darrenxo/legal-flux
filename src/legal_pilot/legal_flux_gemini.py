from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, sha256_text, write_jsonl
from .legal_flux import resolve_project_file, validate_template_pool
from .legal_flux_deepseek import (
    TemplateApiClient,
    _audit_prompt_text,
    _batch_manifest,
    _batch_root,
    _candidate_messages,
    _candidate_plan_row,
    _coerce_templates,
    _extract_json_payloads,
    _merge_prompt_text,
    _messages,
    _read_text,
    _write_stage_manifest,
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

        self.model = str(gemini_config.get("model", "gemini-3.5-flash"))
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

    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        system_instruction = "\n\n".join(
            message["content"] for message in messages if message["role"] == "system"
        )
        contents = "\n\n".join(
            message["content"] for message in messages if message["role"] != "system"
        )
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction or None,
            "max_output_tokens": self.max_output_tokens,
        }
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
                "elapsed_seconds": time.perf_counter() - start,
                "prompt_tokens": _usage_attr(usage, "prompt_token_count"),
                "completion_tokens": _usage_attr(usage, "candidates_token_count"),
                "total_tokens": _usage_attr(usage, "total_token_count"),
            },
        }


def run_gemini_template_workflow(
    config: dict[str, Any],
    *,
    stage: str = "candidates",
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    client: TemplateApiClient | None = None,
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
                    config, force=force, dry_run=True, client=client
                ),
                stage="audit",
            )
            return result
        result["merge"] = generate_gemini_template_merge(
            config, force=force, dry_run=dry_run, client=client
        )
        result["audit"] = generate_gemini_template_audit(
            config, force=force, dry_run=dry_run, client=client
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
            config, force=force, dry_run=dry_run, client=client
        )
    raise ValueError("stage must be one of: candidates, merge, audit, all.")


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
    batch_root = _batch_root(config)
    prompt = _read_text(batch_root / "prompts" / "01_generate_candidate_templates.md")
    schema_text = _read_text(batch_root / "legal_flux_template.schema.json")
    manifest = _batch_manifest(batch_root)
    batches = manifest.get("batches", [])
    if limit is not None:
        batches = batches[:limit]
    output_dir = _gemini_root(config) / "03_candidate_templates"
    raw_dir = output_dir / "raw"
    planned = [
        _candidate_plan_row(batch_root, batch, prompt, schema_text)
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
        messages = _candidate_messages(batch_root, batch, prompt, schema_text)
        response = client.complete(messages)
        raw_text = str(response["content"])
        raw_path.write_text(raw_text, encoding="utf-8")
        templates = _coerce_templates(
            _extract_json_payloads(raw_text),
            source=f"Gemini candidate response for {batch_id}",
            sanitize=False,
        )
        write_jsonl(
            output_path,
            [template.model_dump(mode="json") for template in templates],
        )
        records.append(
            {
                "batch_id": batch_id,
                "status": "ok",
                "output_path": str(output_path),
                "raw_path": str(raw_path),
                "template_count": len(templates),
                "metadata": response.get("metadata", {}),
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
    batch_root = _batch_root(config)
    output_root = _gemini_root(config)
    candidate_dir = output_root / "03_candidate_templates"
    output_path = output_root / "legal_flux_templates_gemini_merged.jsonl"
    raw_path = output_root / "legal_flux_templates_gemini_merged_raw.txt"
    candidate_paths = sorted(candidate_dir.glob("*_candidates.jsonl"))
    if not candidate_paths:
        raise RuntimeError("No Gemini candidate files found. Run candidates stage first.")
    prompt_text = _merge_prompt_text(batch_root, candidate_paths)
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
    response = client.complete(_messages(prompt_text))
    raw_text = str(response["content"])
    raw_path.write_text(raw_text, encoding="utf-8")
    templates = _coerce_templates(
        _extract_json_payloads(raw_text),
        source="Gemini merge response",
        sanitize=True,
    )
    validate_template_pool(templates)
    write_jsonl(output_path, [template.model_dump(mode="json") for template in templates])
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
            "metadata": response.get("metadata", {}),
            "prompt_sha256": sha256_text(prompt_text),
            "output_sha256": sha256_text(output_path.read_text(encoding="utf-8")),
        },
    )


def generate_gemini_template_audit(
    config: dict[str, Any],
    *,
    force: bool = False,
    dry_run: bool = False,
    client: TemplateApiClient | None = None,
) -> dict[str, Any]:
    batch_root = _batch_root(config)
    output_root = _gemini_root(config)
    pool_path = output_root / "legal_flux_templates_gemini_merged.jsonl"
    raw_path = output_root / "legal_flux_coverage_audit_gemini.md"
    gap_path = output_root / "legal_flux_gap_fill_gemini.jsonl"
    if not pool_path.exists():
        raise RuntimeError("No merged Gemini template pool found. Run merge stage first.")
    prompt_text = _audit_prompt_text(batch_root, pool_path)
    if dry_run:
        return {
            "stage": "audit",
            "dry_run": True,
            "prompt_characters": len(prompt_text),
            "audit_path": str(raw_path),
            "gap_fill_path": str(gap_path),
        }
    if raw_path.exists() and not force:
        return {
            "stage": "audit",
            "status": "skipped_existing",
            "audit_path": str(raw_path),
            "gap_fill_path": str(gap_path) if gap_path.exists() else None,
        }
    client = client or GeminiTemplateClient(config)
    response = client.complete(_messages(prompt_text))
    raw_text = str(response["content"])
    raw_path.write_text(raw_text, encoding="utf-8")
    gap_templates = _coerce_templates(
        _extract_json_payloads(raw_text),
        source="Gemini audit gap-fill response",
        sanitize=True,
        allow_empty=True,
    )
    write_jsonl(gap_path, [template.model_dump(mode="json") for template in gap_templates])
    return _write_stage_manifest(
        output_root,
        "audit",
        {
            "stage": "audit",
            "status": "ok",
            "audit_path": str(raw_path),
            "gap_fill_path": str(gap_path),
            "gap_fill_count": len(gap_templates),
            "metadata": response.get("metadata", {}),
            "prompt_sha256": sha256_text(prompt_text),
            "audit_sha256": sha256_text(raw_text),
        },
    )


def _gemini_root(config: dict[str, Any]) -> Path:
    return resolve_project_file(
        config["legal_flux"].get(
            "gemini_template_dir",
            "reports/legal_flux/template_distillation/gemini_api",
        )
    )


def _usage_attr(usage: Any, name: str) -> int | None:
    if usage is None:
        return None
    return getattr(usage, name, None)
