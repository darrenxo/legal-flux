import json

import httpx

from legal_pilot.clients import OllamaClient


def test_ollama_generation_disables_hidden_thinking():
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": '{"answer":"ok"}'},
                "prompt_eval_count": 10,
                "eval_count": 5,
                "done_reason": "stop",
            },
        )

    client = OllamaClient("http://test")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        response = client.generate(
            model="qwen3.5:9b",
            prompt="test",
            schema={"type": "object"},
            temperature=0,
            seed=1,
            context_length=128,
            max_tokens=32,
        )
    finally:
        client.close()

    assert captured["think"] is False
    assert "OUTPUT CONTRACT" in captured["messages"][0]["content"]
    assert '"type":"object"' in captured["messages"][0]["content"]
    assert response.parsed == {"answer": "ok"}


def test_ollama_generation_repairs_nearly_valid_json():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": '{"answer":"ok" "reason":"missing comma"}',
                },
                "done_reason": "stop",
            },
        )

    client = OllamaClient("http://test")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        response = client.generate(
            model="qwen3.5:9b",
            prompt="test",
            schema={"type": "object"},
            temperature=0,
            seed=1,
            context_length=128,
            max_tokens=32,
        )
    finally:
        client.close()

    assert response.parsed == {"answer": "ok", "reason": "missing comma"}
    assert response.metadata["json_repair_applied"] is True


def test_ollama_generation_merges_repaired_object_fragments():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": '{"answer":"ok"},{"reason":"split object"}',
                },
                "done_reason": "stop",
            },
        )

    client = OllamaClient("http://test")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        response = client.generate(
            model="qwen3.5:9b",
            prompt="test",
            schema={"type": "object"},
            temperature=0,
            seed=1,
            context_length=128,
            max_tokens=32,
        )
    finally:
        client.close()

    assert response.parsed == {"answer": "ok", "reason": "split object"}
    assert response.metadata["json_repair_applied"] is True
    assert response.metadata["json_repair_merged_object_list"] is True


def test_ollama_generation_supports_reasoning_level_without_storing_trace():
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "thinking": "private reasoning trace",
                    "content": '{"answer":"ok"}',
                },
                "done_reason": "stop",
            },
        )

    client = OllamaClient("http://test")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        response = client.generate(
            model="gpt-oss:20b",
            prompt="test",
            schema={"type": "object"},
            temperature=0,
            seed=1,
            context_length=128,
            max_tokens=32,
            think="medium",
        )
    finally:
        client.close()

    assert captured["think"] == "medium"
    assert response.metadata["thinking_characters"] == 23
    assert response.metadata["thinking_sha256"]
    assert "private reasoning trace" not in json.dumps(response.metadata)
