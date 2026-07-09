from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .io_utils import atomic_write_json, sha256_text


class SimilarityBackend(Protocol):
    name: str

    def similarities(self, query: str, documents: list[str]) -> list[float]:
        ...


class TfidfSimilarityBackend:
    name = "tfidf"

    def similarities(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            strip_accents="unicode",
        )
        matrix = vectorizer.fit_transform([query, *documents])
        return cosine_similarity(matrix[0:1], matrix[1:]).ravel().tolist()


class FixedEmbeddingBackend:
    """Deterministic test backend with caller-supplied vectors."""

    name = "fixed"

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = {
            text: _normalized(np.asarray(vector, dtype=float))
            for text, vector in vectors.items()
        }

    def similarities(self, query: str, documents: list[str]) -> list[float]:
        query_vector = self.vectors[query]
        return [
            float(np.dot(query_vector, self.vectors[document]))
            for document in documents
        ]


class OllamaEmbeddingBackend:
    name = "ollama_embedding"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        cache_path: Path,
        timeout_seconds: int = 600,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.cache_path = Path(cache_path)
        self.client = http_client or httpx.Client(
            base_url=self.base_url, timeout=timeout_seconds
        )
        self._vectors = self._load_cache()

    def close(self) -> None:
        self.client.close()

    def model_info(self) -> dict | None:
        response = self.client.get("/api/tags")
        response.raise_for_status()
        requested = self.model.removesuffix(":latest")
        for item in response.json().get("models", []):
            names = {
                str(item.get("name", "")),
                str(item.get("model", "")),
            }
            bases = {name.removesuffix(":latest") for name in names}
            if self.model in names or requested in bases:
                return item
        return None

    def similarities(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        vectors = self.embed([query, *documents])
        query_vector = vectors[0]
        return [
            float(np.dot(query_vector, document))
            for document in vectors[1:]
        ]

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        missing = [
            text for text in dict.fromkeys(texts) if self._key(text) not in self._vectors
        ]
        if missing:
            response = self.client.post(
                "/api/embed",
                json={"model": self.model, "input": missing},
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings", [])
            if len(embeddings) != len(missing):
                raise ValueError(
                    "Ollama embedding response count does not match input count."
                )
            for text, vector in zip(missing, embeddings, strict=True):
                self._vectors[self._key(text)] = _normalized(
                    np.asarray(vector, dtype=float)
                )
            self._save_cache()
        return [self._vectors[self._key(text)] for text in texts]

    def _key(self, text: str) -> str:
        return sha256_text(text)

    def _load_cache(self) -> dict[str, np.ndarray]:
        if not self.cache_path.exists():
            return {}
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if payload.get("model") != self.model:
            return {}
        return {
            key: _normalized(np.asarray(vector, dtype=float))
            for key, vector in payload.get("vectors", {}).items()
        }

    def _save_cache(self) -> None:
        atomic_write_json(
            self.cache_path,
            {
                "model": self.model,
                "vectors": {
                    key: vector.tolist()
                    for key, vector in sorted(self._vectors.items())
                },
            },
        )


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Embedding vectors must be non-zero.")
    return vector / norm
