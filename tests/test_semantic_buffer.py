from __future__ import annotations

import json

import httpx

from legal_pilot.bot import TemplateBuffer
from legal_pilot.embeddings import (
    FixedEmbeddingBackend,
    OllamaEmbeddingBackend,
)
from legal_pilot.models import LegalThoughtTemplate


def _template(template_id: str, cue: str) -> LegalThoughtTemplate:
    return LegalThoughtTemplate(
        template_id=template_id,
        name=f"{cue} analysis",
        description=f"Analyze legal disputes involving {cue}.",
        applicability_cues=[cue],
        reasoning_steps=[
            f"Identify the relevant {cue} question.",
            "Apply supplied facts and resolve the requested remedy.",
        ],
        required_checks=[cue, "evidence"],
        contraindications=[],
        provenance_case_ids=[],
        version=1,
    )


def test_semantic_buffer_retrieves_by_embedding_not_lexical_overlap():
    backend = FixedEmbeddingBackend(
        {
            "employment status and control": [1.0, 0.0],
            "employment analysis Analyze legal disputes involving employment. "
            "employment Identify the relevant employment question. "
            "Apply supplied facts and resolve the requested remedy. "
            "employment evidence": [0.95, 0.05],
            "contract analysis Analyze legal disputes involving contract. "
            "contract Identify the relevant contract question. "
            "Apply supplied facts and resolve the requested remedy. "
            "contract evidence": [0.0, 1.0],
        }
    )
    buffer = TemplateBuffer(
        [_template("employment", "employment"), _template("contract", "contract")],
        similarity_backend=backend,
    )

    result = buffer.retrieve(
        "employment status and control", threshold=0.60
    )

    assert result.used_fallback is False
    assert result.template.template_id == "employment"
    assert result.similarity > 0.90


def test_append_only_manager_rejects_redundant_candidate_without_merging():
    existing = _template("contract", "contract")
    candidate = _template("contract_variant", "contract variation")
    backend = FixedEmbeddingBackend(
        {
            "contract analysis Analyze legal disputes involving contract. "
            "contract Identify the relevant contract question. "
            "Apply supplied facts and resolve the requested remedy. "
            "contract evidence": [1.0, 0.0],
            "contract variation analysis Analyze legal disputes involving "
            "contract variation. contract variation Identify the relevant "
            "contract variation question. Apply supplied facts and resolve "
            "the requested remedy. contract variation evidence": [0.98, 0.02],
        }
    )
    buffer = TemplateBuffer([existing], similarity_backend=backend)

    event = buffer.apply_candidate_append_only(
        candidate,
        source_case_id="legalhk-1",
        novelty_threshold=0.60,
    )

    assert event.action == "reject"
    assert event.target_template_id == "contract"
    assert len(buffer.templates) == 1
    assert buffer.templates[0].model_dump() == existing.model_dump()


def test_append_only_manager_adds_semantically_novel_candidate():
    existing = _template("contract", "contract")
    candidate = _template("employment", "employment")
    backend = FixedEmbeddingBackend(
        {
            "contract analysis Analyze legal disputes involving contract. "
            "contract Identify the relevant contract question. "
            "Apply supplied facts and resolve the requested remedy. "
            "contract evidence": [1.0, 0.0],
            "employment analysis Analyze legal disputes involving employment. "
            "employment Identify the relevant employment question. "
            "Apply supplied facts and resolve the requested remedy. "
            "employment evidence": [0.0, 1.0],
        }
    )
    buffer = TemplateBuffer([existing], similarity_backend=backend)

    event = buffer.apply_candidate_append_only(
        candidate,
        source_case_id="legalhk-2",
        novelty_threshold=0.60,
    )

    assert event.action == "new"
    assert len(buffer.templates) == 2
    assert buffer.templates[-1].provenance_case_ids == ["legalhk-2"]


def test_ollama_embedding_backend_batches_and_caches(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "bge-m3:latest",
                            "model": "bge-m3:latest",
                            "digest": "embed-digest",
                        }
                    ]
                },
            )
        payload = json.loads(request.content)
        calls.append(payload["input"])
        vectors = {
            "query": [1.0, 0.0],
            "near": [0.8, 0.2],
            "far": [0.0, 1.0],
        }
        return httpx.Response(
            200,
            json={"embeddings": [vectors[text] for text in payload["input"]]},
        )

    client = httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    backend = OllamaEmbeddingBackend(
        base_url="http://test",
        model="bge-m3:latest",
        cache_path=tmp_path / "embeddings.json",
        http_client=client,
    )

    assert backend.model_info()["digest"] == "embed-digest"
    first = backend.similarities("query", ["near", "far"])
    second = backend.similarities("query", ["near", "far"])
    backend.close()

    assert first[0] > first[1]
    assert second == first
    assert calls == [["query", "near", "far"]]
    cached = json.loads((tmp_path / "embeddings.json").read_text())
    assert cached["model"] == "bge-m3:latest"
    assert len(cached["vectors"]) == 3
