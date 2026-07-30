from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import resolve_path
from .io_utils import atomic_write_json, canonical_json, read_jsonl, sha256_text
from .models import NormalizedCase
from .runner import load_cases


class DenseTextEncoder(Protocol):
    model_name: str

    def encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        ...


class PairReranker(Protocol):
    model_name: str

    def score(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int,
    ) -> list[float]:
        ...


class SentenceTransformerDenseEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        max_length: int = 8192,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "X_sim construction requires the retrieval dependencies. "
                "Install the project with `pip install -e .[retrieval]`."
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length

    def encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )


class BgeCrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        max_length: int = 8192,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "X_sim construction requires the retrieval dependencies. "
                "Install the project with `pip install -e .[retrieval]`."
            ) from exc
        self.model_name = model_name
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
            num_labels=1,
        )

    def score(
        self,
        pairs: list[tuple[str, str]],
        *,
        batch_size: int,
    ) -> list[float]:
        values = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(values, dtype=np.float32).reshape(-1).tolist()


def build_xsim(
    config: dict[str, Any],
    *,
    stage: str = "all",
    case_limit: int | None = None,
    force: bool = False,
    dense_encoder: DenseTextEncoder | None = None,
    reranker: PairReranker | None = None,
) -> dict[str, Any]:
    if stage not in {"dense", "rerank", "all"}:
        raise ValueError(f"Unsupported X_sim stage: {stage}")
    settings = _xsim_settings(config)
    cases = _planner_train_cases(config)
    if len(cases) < 3:
        raise ValueError("X_sim construction requires at least three planner_train cases.")
    anchors = cases[:case_limit] if case_limit is not None else cases
    output_dir = _xsim_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    dense_path = output_dir / settings["dense_candidates_file"]
    neighbors_path = output_dir / settings["neighbors_file"]
    embedding_path = output_dir / settings["embeddings_file"]
    case_index_path = output_dir / settings["case_index_file"]
    manifest_path = output_dir / settings["manifest_file"]

    case_rows = [
        {
            "case_id": case.case_id,
            "text": xsim_case_text(case),
        }
        for case in cases
    ]
    corpus_hash = sha256_text(canonical_json(case_rows))
    manifest_key = {
        "corpus_hash": corpus_hash,
        "dense_model": settings["dense_model"],
        "reranker_model": settings["reranker_model"],
        "dense_top_k": settings["dense_top_k"],
        "final_top_k": settings["final_top_k"],
        "max_length": settings["max_length"],
    }
    _guard_existing_manifest(manifest_path, manifest_key, force=force)
    if force:
        for path in (dense_path, neighbors_path, embedding_path, case_index_path):
            path.unlink(missing_ok=True)

    dense_completed = 0
    rerank_completed = 0
    if stage in {"dense", "all"}:
        if dense_encoder is None:
            dense_encoder = SentenceTransformerDenseEncoder(
                settings["dense_model"],
                device=settings["device"],
                max_length=settings["max_length"],
            )
        embeddings = _load_or_encode_embeddings(
            case_rows,
            embedding_path=embedding_path,
            case_index_path=case_index_path,
            encoder=dense_encoder,
            batch_size=settings["dense_batch_size"],
        )
        dense_completed = _write_dense_candidates(
            anchors=anchors,
            cases=cases,
            embeddings=embeddings,
            output_path=dense_path,
            top_k=settings["dense_top_k"],
        )

    if stage in {"rerank", "all"}:
        if not dense_path.exists():
            raise RuntimeError(
                f"Dense candidates do not exist at {dense_path}. Run stage `dense` first."
            )
        if reranker is None:
            reranker = BgeCrossEncoderReranker(
                settings["reranker_model"],
                device=settings["device"],
                max_length=settings["max_length"],
            )
        rerank_completed = _write_reranked_neighbors(
            anchors=anchors,
            cases=cases,
            dense_path=dense_path,
            output_path=neighbors_path,
            reranker=reranker,
            batch_size=settings["reranker_batch_size"],
            final_top_k=settings["final_top_k"],
        )

    manifest = {
        **manifest_key,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "planner_train_cases": len(cases),
        "requested_anchors": len(anchors),
        "dense_rows_added": dense_completed,
        "reranked_rows_added": rerank_completed,
        "embedding_path": str(embedding_path),
        "case_index_path": str(case_index_path),
        "dense_candidates_path": str(dense_path),
        "neighbors_path": str(neighbors_path),
        "fields": ["claim", "facts", "authorities", "relevant_cases"],
        "selection": (
            "BGE-M3 dense top-k over planner_train, followed by "
            "bge-reranker-v2-m3 cross-encoder top-2. Only the anchor itself "
            "is excluded."
        ),
    }
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path), "stage": stage}


def load_xsim_neighbors(
    config: dict[str, Any],
) -> dict[str, list[str]]:
    settings = _xsim_settings(config)
    path = _xsim_dir(config) / settings["neighbors_file"]
    if not path.exists():
        raise RuntimeError(
            f"X_sim neighbor file does not exist at {path}. Run `flux-build-xsim`."
        )
    result: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        anchor = str(row.get("anchor_case_id", ""))
        ids = [str(value) for value in row.get("x_sim_case_ids", [])]
        if not anchor or len(ids) != settings["final_top_k"] + 1 or ids[0] != anchor:
            raise ValueError(f"Invalid X_sim row for anchor {anchor!r}.")
        result[anchor] = ids
    return result


def xsim_case_text(case: NormalizedCase) -> str:
    facts = "\n".join(
        f"{fact_id}: {text}" for fact_id, text in case.facts.items()
    )
    authorities = (case.authorities or "").strip() or "None supplied."
    relevant_cases = (
        str(case.metadata.get("relevant_cases") or "").strip() or "None supplied."
    )
    return (
        f"[CLAIM]\n{case.claim.strip()}\n\n"
        f"[FACTS]\n{facts}\n\n"
        f"[RELATED LAWS]\n{authorities}\n\n"
        f"[RELEVANT CASES]\n{relevant_cases}"
    )


def _planner_train_cases(config: dict[str, Any]) -> list[NormalizedCase]:
    return [
        case
        for case in load_cases(config)
        if case.metadata.get("selection_split") == "planner_train"
    ]


def _xsim_settings(config: dict[str, Any]) -> dict[str, Any]:
    values = config.get("xsim", {})
    return {
        "dense_model": values.get("dense_model", "BAAI/bge-m3"),
        "reranker_model": values.get(
            "reranker_model", "BAAI/bge-reranker-v2-m3"
        ),
        "device": values.get("device") or None,
        "max_length": int(values.get("max_length", 8192)),
        "dense_batch_size": int(values.get("dense_batch_size", 8)),
        "reranker_batch_size": int(values.get("reranker_batch_size", 8)),
        "dense_top_k": int(values.get("dense_top_k", 50)),
        "final_top_k": int(values.get("final_top_k", 2)),
        "embeddings_file": values.get(
            "embeddings_file", "planner_train_bge_m3_embeddings.npy"
        ),
        "case_index_file": values.get(
            "case_index_file", "planner_train_embedding_cases.json"
        ),
        "dense_candidates_file": values.get(
            "dense_candidates_file", "xsim_dense_top50.jsonl"
        ),
        "neighbors_file": values.get("neighbors_file", "xsim_neighbors.jsonl"),
        "manifest_file": values.get("manifest_file", "xsim_manifest.json"),
    }


def _xsim_dir(config: dict[str, Any]) -> Path:
    return resolve_path(config, "processed_dir") / "xsim"


def _guard_existing_manifest(
    manifest_path: Path,
    expected: dict[str, Any],
    *,
    force: bool,
) -> None:
    if force or not manifest_path.exists():
        return
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = {
        key: (existing.get(key), value)
        for key, value in expected.items()
        if existing.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Existing X_sim artifacts were built with different settings. "
            f"Use --force to rebuild them. Mismatches: {mismatches}"
        )


def _load_or_encode_embeddings(
    case_rows: list[dict[str, str]],
    *,
    embedding_path: Path,
    case_index_path: Path,
    encoder: DenseTextEncoder,
    batch_size: int,
) -> np.ndarray:
    case_ids = [row["case_id"] for row in case_rows]
    if embedding_path.exists() and case_index_path.exists():
        existing_ids = json.loads(case_index_path.read_text(encoding="utf-8"))
        if existing_ids != case_ids:
            raise RuntimeError("Cached X_sim embedding case order does not match.")
        embeddings = np.load(embedding_path)
        if embeddings.shape[0] != len(case_ids):
            raise RuntimeError("Cached X_sim embedding row count does not match.")
        return _normalize_rows(np.asarray(embeddings, dtype=np.float32))

    embeddings = np.asarray(
        encoder.encode(
            [row["text"] for row in case_rows],
            batch_size=batch_size,
        ),
        dtype=np.float32,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(case_rows):
        raise ValueError("Dense encoder returned an invalid embedding matrix.")
    embeddings = _normalize_rows(embeddings)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, embeddings)
    atomic_write_json(case_index_path, case_ids)
    return embeddings


def _write_dense_candidates(
    *,
    anchors: list[NormalizedCase],
    cases: list[NormalizedCase],
    embeddings: np.ndarray,
    output_path: Path,
    top_k: int,
) -> int:
    if top_k < 1 or top_k >= len(cases):
        raise ValueError("dense_top_k must be between 1 and planner_train size - 1.")
    case_index = {case.case_id: index for index, case in enumerate(cases)}
    completed = {
        str(row.get("anchor_case_id"))
        for row in read_jsonl(output_path)
        if row.get("anchor_case_id")
    }
    added = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for anchor in anchors:
            if anchor.case_id in completed:
                continue
            anchor_index = case_index[anchor.case_id]
            scores = embeddings @ embeddings[anchor_index]
            scores[anchor_index] = -np.inf
            candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
            candidate_indices = candidate_indices[
                np.argsort(scores[candidate_indices])[::-1]
            ]
            row = {
                "anchor_case_id": anchor.case_id,
                "candidates": [
                    {
                        "case_id": cases[int(index)].case_id,
                        "dense_rank": rank,
                        "dense_score": float(scores[int(index)]),
                    }
                    for rank, index in enumerate(candidate_indices, start=1)
                ],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            added += 1
    return added


def _write_reranked_neighbors(
    *,
    anchors: list[NormalizedCase],
    cases: list[NormalizedCase],
    dense_path: Path,
    output_path: Path,
    reranker: PairReranker,
    batch_size: int,
    final_top_k: int,
) -> int:
    dense_rows = {
        str(row["anchor_case_id"]): row for row in read_jsonl(dense_path)
    }
    case_by_id = {case.case_id: case for case in cases}
    completed = {
        str(row.get("anchor_case_id"))
        for row in read_jsonl(output_path)
        if row.get("anchor_case_id")
    }
    added = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for anchor in anchors:
            if anchor.case_id in completed:
                continue
            dense_row = dense_rows.get(anchor.case_id)
            if dense_row is None:
                raise RuntimeError(
                    f"Dense candidates missing for anchor {anchor.case_id}."
                )
            candidates = dense_row["candidates"]
            anchor_text = xsim_case_text(anchor)
            pairs = [
                (anchor_text, xsim_case_text(case_by_id[item["case_id"]]))
                for item in candidates
            ]
            scores = reranker.score(pairs, batch_size=batch_size)
            if len(scores) != len(candidates):
                raise ValueError("Cross-encoder score count does not match candidates.")
            ranked = sorted(
                (
                    {
                        **candidate,
                        "reranker_score": float(score),
                    }
                    for candidate, score in zip(candidates, scores, strict=True)
                ),
                key=lambda item: item["reranker_score"],
                reverse=True,
            )
            selected = [
                {**item, "reranker_rank": rank}
                for rank, item in enumerate(ranked[:final_top_k], start=1)
            ]
            row = {
                "anchor_case_id": anchor.case_id,
                "selected_neighbors": selected,
                "x_sim_case_ids": [
                    anchor.case_id,
                    *[item["case_id"] for item in selected],
                ],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            added += 1
    return added


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Dense encoder returned a zero vector.")
    return matrix / norms
