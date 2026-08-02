from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import httpx
from json_repair import repair_json
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


T = TypeVar("T", bound=BaseModel)


@dataclass
class ModelResponse:
    raw_text: str
    parsed: dict[str, Any] | None
    elapsed_seconds: float
    prompt_tokens: int | None
    output_tokens: int | None
    metadata: dict[str, Any]


class GenerationResponseError(ValueError):
    def __init__(self, message: str, *, raw_text: str, payload: dict[str, Any]):
        super().__init__(message)
        self.raw_text = raw_text
        self.payload = payload


class GenerationClient(Protocol):
    def close(self) -> None:
        ...

    def model_info(self, model_name: str) -> dict[str, Any] | None:
        ...

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int,
        context_length: int,
        max_tokens: int,
        think: bool | str = False,
    ) -> ModelResponse:
        ...


class OllamaResponseError(GenerationResponseError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: int = 600):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        self.client.close()

    def model_info(self, model_name: str) -> dict[str, Any] | None:
        response = self.client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        for model in response.json().get("models", []):
            if model.get("name") == model_name or model.get("model") == model_name:
                return model
        return None

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, ValueError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int,
        context_length: int,
        max_tokens: int,
        think: bool | str = False,
    ) -> ModelResponse:
        start = time.perf_counter()
        schema_instruction = (
            "\n\nOUTPUT CONTRACT:\nReturn only one JSON object matching this JSON "
            "Schema exactly. Do not use Markdown fences or add fields.\n"
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
        response = self.client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt + schema_instruction}
                ],
                "stream": False,
                "think": think,
                "format": schema,
                "options": {
                    "temperature": temperature,
                    "seed": seed,
                    "num_ctx": context_length,
                    "num_predict": max_tokens,
                },
                "keep_alive": "30m",
            },
        )
        response.raise_for_status()
        payload = response.json()
        raw_text = payload["message"]["content"]
        thinking = payload.get("message", {}).get("thinking") or ""
        repaired = False
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            try:
                parsed = repair_json(raw_text, return_objects=True)
                repaired = True
            except Exception as repair_exc:
                raise OllamaResponseError(
                    f"Invalid Ollama JSON content: {exc}",
                    raw_text=raw_text,
                    payload=payload,
                ) from repair_exc
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                merged: dict[str, Any] = {}
                for item in parsed:
                    merged.update(item)
                parsed = merged
                repaired = True
                payload["_codex_repair_merged_object_list"] = True
            if not isinstance(parsed, dict):
                raise OllamaResponseError(
                    f"JSON repair did not produce an object: {exc}",
                    raw_text=raw_text,
                    payload=payload,
                ) from exc
        return ModelResponse(
            raw_text=raw_text,
            parsed=parsed,
            elapsed_seconds=time.perf_counter() - start,
            prompt_tokens=payload.get("prompt_eval_count"),
            output_tokens=payload.get("eval_count"),
            metadata={
                "total_duration": payload.get("total_duration"),
                "load_duration": payload.get("load_duration"),
                "done_reason": payload.get("done_reason"),
                "json_repair_applied": repaired,
                "json_repair_merged_object_list": bool(
                    payload.get("_codex_repair_merged_object_list")
                ),
                "thinking_characters": len(thinking),
                "thinking_sha256": (
                    hashlib.sha256(thinking.encode("utf-8")).hexdigest()
                    if thinking
                    else None
                ),
            },
        )


class OpenAICompatibleClient:
    """Client for local vLLM/SGLang/Transformers OpenAI-compatible servers."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: int = 600,
        *,
        api_key: str = "EMPTY",
        extra_body: dict[str, Any] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.extra_body = dict(extra_body or {})
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def close(self) -> None:
        self.client.close()

    def model_info(self, model_name: str) -> dict[str, Any] | None:
        response = self.client.get("/models")
        response.raise_for_status()
        for model in response.json().get("data", []):
            if model.get("id") == model_name:
                return {
                    **model,
                    "name": model_name,
                    "digest": str(model.get("revision") or model_name),
                }
        return None

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, ValueError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int,
        context_length: int,
        max_tokens: int,
        think: bool | str = False,
    ) -> ModelResponse:
        del context_length, think
        start = time.perf_counter()
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "legalflux_response",
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        payload.update(self.extra_body)
        response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        response_payload = response.json()
        choices = response_payload.get("choices") or []
        if not choices:
            raise GenerationResponseError(
                "OpenAI-compatible response contained no choices.",
                raw_text="",
                payload=response_payload,
            )
        message = choices[0].get("message") or {}
        raw_text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        if not raw_text.strip():
            raise GenerationResponseError(
                "OpenAI-compatible response contained no visible JSON content "
                f"(finish_reason={choices[0].get('finish_reason')!r}, "
                f"reasoning_characters={len(reasoning)}).",
                raw_text=raw_text,
                payload=response_payload,
            )
        parsed, repaired, merged = _parse_json_object(
            raw_text,
            payload=response_payload,
            source="OpenAI-compatible",
        )
        usage = response_payload.get("usage") or {}
        return ModelResponse(
            raw_text=raw_text,
            parsed=parsed,
            elapsed_seconds=time.perf_counter() - start,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            metadata={
                "finish_reason": choices[0].get("finish_reason"),
                "json_repair_applied": repaired,
                "json_repair_merged_object_list": merged,
                "thinking_characters": len(reasoning),
                "thinking_sha256": (
                    hashlib.sha256(reasoning.encode("utf-8")).hexdigest()
                    if reasoning
                    else None
                ),
            },
        )


def build_generation_client(config: dict[str, Any]) -> GenerationClient:
    model_config = config["model"]
    provider = str(model_config.get("provider", "ollama"))
    if provider == "ollama":
        return OllamaClient(
            model_config["base_url"],
            model_config.get("timeout_seconds", 600),
        )
    if provider == "openai_compatible":
        api_key_env = str(model_config.get("api_key_env", "OPENAI_API_KEY"))
        api_key = os.environ.get(api_key_env, "EMPTY")
        return OpenAICompatibleClient(
            model_config["base_url"],
            model_config.get("timeout_seconds", 600),
            api_key=api_key,
            extra_body=model_config.get("extra_body"),
        )
    raise ValueError(f"Unsupported generation provider: {provider}")


def _parse_json_object(
    raw_text: str,
    *,
    payload: dict[str, Any],
    source: str,
) -> tuple[dict[str, Any], bool, bool]:
    repaired = False
    merged = False
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        try:
            parsed = repair_json(raw_text, return_objects=True)
            repaired = True
        except Exception as repair_exc:
            raise GenerationResponseError(
                f"Invalid {source} JSON content: {exc}",
                raw_text=raw_text,
                payload=payload,
            ) from repair_exc
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        combined: dict[str, Any] = {}
        for item in parsed:
            combined.update(item)
        parsed = combined
        repaired = True
        merged = True
    if not isinstance(parsed, dict):
        raise GenerationResponseError(
            f"{source} JSON did not produce an object.",
            raw_text=raw_text,
            payload=payload,
        )
    return parsed, repaired, merged


class OpenAIAuditClient:
    def __init__(self, model: str, reasoning_effort: str):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.reasoning_effort = reasoning_effort

    def audit(self, prompt: str, output_type: type[T]) -> tuple[T, dict[str, Any]]:
        start = time.perf_counter()
        response = self.client.responses.parse(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            reasoning={"effort": self.reasoning_effort},
            text_format=output_type,
        )
        parsed = response.output_parsed
        usage = getattr(response, "usage", None)
        metadata = {
            "response_id": response.id,
            "elapsed_seconds": time.perf_counter() - start,
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }
        return parsed, metadata
