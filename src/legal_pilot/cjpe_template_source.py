from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import BisectingKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

from .config import resolve_project_path
from .io_utils import atomic_write_json, write_jsonl
from .legal_benchmark_data import (
    REALISTIC_LJP_URL,
    BenchmarkCase,
    DatasetAccessError,
    _binary_judgment_label,
    _download_file,
    _hugging_face_token,
    _sha256_file,
    benchmark_path,
)


IL_TUR_DATASET = "Exploration-Lab/IL-TUR"
IL_TUR_PARQUET_ENDPOINT = "https://datasets-server.huggingface.co/parquet"
TEMPLATE_SOURCE_SPLIT = "aligned_multi_train_10pct"


def prepare_cjpe_template_source(config: dict[str, Any]) -> dict[str, Any]:
    """Select a reproducible, distribution-preserving CJPE template source."""
    flux_config = config["legal_flux"]
    fraction = float(flux_config.get("template_source_fraction", 0.10))
    family_count = int(flux_config.get("template_source_family_clusters", 48))
    max_features = int(
        flux_config.get("template_source_tfidf_max_features", 40000)
    )
    svd_components = int(
        flux_config.get("template_source_svd_components", 128)
    )
    max_case_characters = int(
        flux_config.get("template_source_max_case_characters", 300000)
    )
    seed = int(config["project"]["seed"])
    if not 0.0 < fraction < 1.0:
        raise ValueError("template_source_fraction must be between 0 and 1.")
    if family_count < 2:
        raise ValueError("template_source_family_clusters must be at least 2.")

    token = _hugging_face_token()
    if not token:
        raise DatasetAccessError(
            "IL-TUR is gated. Accept its conditions and run `hf auth login` "
            "before preparing the CJPE template source."
        )
    headers = {"Authorization": f"Bearer {token}"}
    raw_root = benchmark_path(config, "raw_dir")
    facts_path = (
        raw_root / "realistic_ljp_facts" / "Realistic_LJP_Facts.csv"
    )
    _download_file(REALISTIC_LJP_URL, facts_path)
    parquet_paths = _download_multi_train_parquet(raw_root, headers=headers)
    cjpe_index = _cjpe_index(parquet_paths)

    facts = pd.read_csv(
        facts_path,
        usecols=["text", "label", "split", "name"],
    )
    facts = facts[facts["split"].astype(str).str.lower().eq("train")].copy()
    facts["source_id"] = facts["name"].map(_source_id)
    if facts["source_id"].duplicated().any():
        duplicates = facts.loc[
            facts["source_id"].duplicated(keep=False), "source_id"
        ].tolist()
        raise ValueError(f"Duplicate Realistic LJP training IDs: {duplicates[:5]}")
    facts["gold_label"] = facts["label"].map(_binary_judgment_label)
    facts["decade"] = facts["source_id"].map(_case_decade)

    aligned = facts[facts["source_id"].isin(cjpe_index)].copy()
    aligned_count = len(aligned)
    target_count = int(round(aligned_count * fraction))
    eligible = aligned[
        aligned["source_id"].map(
            lambda source_id: 0
            < int(cjpe_index[source_id]["characters"])
            <= max_case_characters
        )
    ].reset_index(drop=True)
    excluded_for_length = aligned_count - len(eligible)
    if len(eligible) < target_count:
        raise ValueError(
            f"Only {len(eligible)} aligned cases meet the per-case size limit, "
            f"but {target_count} are required."
        )
    if family_count > len(eligible):
        raise ValueError("More semantic families requested than eligible cases.")

    family_labels, family_profiles, selection_model = _semantic_families(
        eligible["text"].fillna("").astype(str).tolist(),
        case_ids=eligible["source_id"].tolist(),
        family_count=family_count,
        max_features=max_features,
        svd_components=svd_components,
        seed=seed,
    )
    eligible["semantic_family"] = family_labels
    selected_ids, family_quotas = _proportional_family_sample(
        eligible,
        target_count=target_count,
        seed=seed,
    )
    selected_set = set(selected_ids)
    selected_facts = eligible[eligible["source_id"].isin(selected_set)].copy()
    facts_by_id = selected_facts.set_index("source_id").to_dict(orient="index")
    cases = _selected_cjpe_cases(
        parquet_paths,
        selected_ids=selected_set,
        facts_by_id=facts_by_id,
    )
    cases.sort(
        key=lambda case: (
            int(case.metadata["selection_semantic_family"]),
            case.case_id,
        )
    )
    if len(cases) != target_count:
        found = {str(case.metadata["source_id"]) for case in cases}
        missing = sorted(selected_set - found)
        raise ValueError(f"Selected CJPE cases missing from Parquet: {missing[:5]}")

    output_path = resolve_project_path(
        config,
        flux_config.get(
            "template_source_file",
            "data/processed/legal_benchmarks/il_tur_cjpe/"
            "template_source_aligned_train_10pct.jsonl",
        ),
    )
    manifest_path = resolve_project_path(
        config,
        flux_config.get(
            "template_source_selection_manifest",
            "data/processed/legal_benchmarks/il_tur_cjpe/"
            "template_source_aligned_train_10pct_manifest.json",
        ),
    )
    ids_path = output_path.with_name(output_path.stem + "_case_ids.json")
    write_jsonl(
        output_path,
        [case.model_dump(mode="json") for case in cases],
    )
    atomic_write_json(ids_path, [case.case_id for case in cases])

    selected_frame = eligible[eligible["source_id"].isin(selected_set)]
    full_family_counts = Counter(int(value) for value in family_labels)
    selected_family_counts = Counter(
        int(value) for value in selected_frame["semantic_family"]
    )
    profile_by_family = {
        int(profile["semantic_family"]): profile for profile in family_profiles
    }
    family_rows = []
    for family in sorted(full_family_counts):
        full_count = full_family_counts[family]
        selected_count = selected_family_counts[family]
        profile = profile_by_family[family]
        family_rows.append(
            {
                **profile,
                "training_cases": full_count,
                "selected_cases": selected_count,
                "training_share": full_count / len(eligible),
                "selected_share": selected_count / target_count,
                "selection_rate": selected_count / full_count,
                "quota": family_quotas[family],
            }
        )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": IL_TUR_DATASET,
        "source_split": "multi_train",
        "selection_split": TEMPLATE_SOURCE_SPLIT,
        "selection_fraction": fraction,
        "aligned_training_cases": aligned_count,
        "eligible_aligned_cases": len(eligible),
        "excluded_above_case_character_limit": excluded_for_length,
        "max_case_characters": max_case_characters,
        "selected_cases": len(cases),
        "actual_selection_fraction": len(cases) / aligned_count,
        "selection_seed": seed,
        "selection_method": (
            "TF-IDF + truncated SVD + largest-cluster bisecting KMeans; "
            "largest-remainder proportional quota per semantic family; "
            "label-and-decade-stratified deterministic sampling within family"
        ),
        "selection_model": selection_model,
        "families": family_rows,
        "label_distribution": _distribution_comparison(
            eligible["gold_label"], selected_frame["gold_label"]
        ),
        "decade_distribution": _distribution_comparison(
            eligible["decade"], selected_frame["decade"]
        ),
        "output_file": str(output_path),
        "case_ids_file": str(ids_path),
        "output_sha256": _sha256_file(output_path),
        "source_files": {
            "realistic_ljp_facts": {
                "path": str(facts_path),
                "sha256": _sha256_file(facts_path),
            },
            "cjpe_multi_train_parquet": [
                {"path": str(path), "sha256": _sha256_file(path)}
                for path in parquet_paths
            ],
        },
    }
    atomic_write_json(manifest_path, manifest)
    return {
        "aligned_training_cases": aligned_count,
        "selected_cases": len(cases),
        "semantic_families": family_count,
        "output_file": str(output_path),
        "case_ids_file": str(ids_path),
        "manifest": str(manifest_path),
    }


def _download_multi_train_parquet(
    raw_root: Path,
    *,
    headers: dict[str, str],
) -> list[Path]:
    response = httpx.get(
        IL_TUR_PARQUET_ENDPOINT,
        params={"dataset": IL_TUR_DATASET},
        headers=headers,
        follow_redirects=True,
        timeout=60,
    )
    if response.status_code in {401, 403}:
        raise DatasetAccessError(
            "IL-TUR denied the Parquet listing. Confirm gated access and HF login."
        )
    response.raise_for_status()
    files = [
        item
        for item in response.json().get("parquet_files", [])
        if item.get("config") == "cjpe" and item.get("split") == "multi_train"
    ]
    if not files:
        raise DatasetAccessError("No CJPE multi_train Parquet shards were listed.")
    destinations = []
    parquet_dir = raw_root / "il_tur_cjpe" / "parquet"
    for index, item in enumerate(sorted(files, key=lambda row: row["filename"])):
        destination = parquet_dir / f"multi_train_{index:04d}.parquet"
        _download_file(
            str(item["url"]),
            destination,
            headers=headers,
            gated_name="IL-TUR",
        )
        destinations.append(destination)
    return destinations


def _cjpe_index(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=["id", "text", "label"],
            batch_size=256,
        ):
            values = batch.to_pydict()
            for source_id, text, label in zip(
                values["id"], values["text"], values["label"], strict=True
            ):
                source_id = str(source_id).strip()
                if source_id in result:
                    raise ValueError(f"Duplicate CJPE multi_train ID: {source_id}")
                result[source_id] = {
                    "characters": len(str(text or "")),
                    "label": _binary_judgment_label(label),
                }
    return result


def _semantic_families(
    texts: list[str],
    *,
    case_ids: list[str],
    family_count: int,
    max_features: int,
    svd_components: int,
    seed: int,
) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.98,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    component_count = min(svd_components, matrix.shape[0] - 1, matrix.shape[1] - 1)
    if component_count < 2:
        raise ValueError("Not enough rows or TF-IDF features for semantic selection.")
    svd = TruncatedSVD(
        n_components=component_count,
        n_iter=7,
        random_state=seed,
    )
    dense = Normalizer(copy=False).fit_transform(svd.fit_transform(matrix))
    clusterer = BisectingKMeans(
        n_clusters=family_count,
        random_state=seed,
        n_init=3,
        bisecting_strategy="largest_cluster",
    )
    labels = clusterer.fit_predict(dense).astype(int)
    vocabulary = np.asarray(vectorizer.get_feature_names_out())
    term_centers = svd.inverse_transform(clusterer.cluster_centers_)
    profiles = []
    for family in range(family_count):
        indices = np.flatnonzero(labels == family)
        center = np.asarray(clusterer.cluster_centers_[family], dtype=np.float32)
        norm = float(np.linalg.norm(center)) or 1.0
        similarities = dense[indices] @ (center / norm)
        representative_order = indices[
            np.argsort(np.asarray(similarities).ravel())[-5:][::-1]
        ]
        top_terms = vocabulary[
            np.argsort(term_centers[family])[-12:][::-1]
        ].tolist()
        profiles.append(
            {
                "semantic_family": family,
                "top_terms": top_terms,
                "representative_case_ids": [
                    case_ids[index] for index in representative_order
                ],
            }
        )
    return (
        labels.tolist(),
        profiles,
        {
            "family_count": family_count,
            "tfidf_features": int(matrix.shape[1]),
            "svd_components": component_count,
            "svd_explained_variance": float(svd.explained_variance_ratio_.sum()),
            "clustering": "BisectingKMeans(largest_cluster)",
        },
    )


def _proportional_family_sample(
    frame: pd.DataFrame,
    *,
    target_count: int,
    seed: int,
) -> tuple[list[str], dict[int, int]]:
    family_sizes = Counter(int(value) for value in frame["semantic_family"])
    quotas = _apportion_counts(family_sizes, target_count)
    selected: list[str] = []
    for family in sorted(family_sizes):
        family_rows = frame[frame["semantic_family"].eq(family)]
        strata: dict[str, list[str]] = defaultdict(list)
        for row in family_rows.itertuples(index=False):
            strata[f"{row.gold_label}|{row.decade}"].append(str(row.source_id))
        stratum_quotas = _apportion_counts(
            {key: len(ids) for key, ids in strata.items()},
            quotas[family],
        )
        for key in sorted(strata):
            ranked = sorted(
                strata[key],
                key=lambda case_id: (_stable_rank(seed, case_id), case_id),
            )
            selected.extend(ranked[: stratum_quotas[key]])
    if len(selected) != target_count or len(set(selected)) != target_count:
        raise ValueError("Proportional CJPE selection did not produce unique target IDs.")
    return selected, quotas


def _apportion_counts(
    sizes: dict[Any, int] | Counter[Any],
    target: int,
) -> dict[Any, int]:
    if target < 0 or target > sum(sizes.values()):
        raise ValueError("Target must be between zero and the available population.")
    total = sum(sizes.values())
    if not sizes:
        return {}
    exact = {key: target * size / total for key, size in sizes.items()}
    result = {key: min(size, math.floor(exact[key])) for key, size in sizes.items()}
    remaining = target - sum(result.values())
    order = sorted(
        sizes,
        key=lambda key: (
            -(exact[key] - math.floor(exact[key])),
            str(key),
        ),
    )
    for key in order:
        if not remaining:
            break
        if result[key] < sizes[key]:
            result[key] += 1
            remaining -= 1
    if remaining:
        raise ValueError("Could not apportion the full selection target.")
    return result


def _selected_cjpe_cases(
    paths: list[Path],
    *,
    selected_ids: set[str],
    facts_by_id: dict[str, dict[str, Any]],
) -> list[BenchmarkCase]:
    labels = ["rejected", "accepted"]
    descriptions = {
        "rejected": "the appeal is rejected/dismissed",
        "accepted": "at least one appeal is accepted/allowed",
    }
    cases = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=["id", "text", "label"],
            batch_size=128,
        ):
            values = batch.to_pydict()
            for source_id, text, label in zip(
                values["id"], values["text"], values["label"], strict=True
            ):
                source_id = str(source_id).strip()
                if source_id not in selected_ids:
                    continue
                facts = facts_by_id[source_id]
                gold_label = _binary_judgment_label(label)
                if gold_label != str(facts["gold_label"]):
                    raise ValueError(
                        f"CJPE/LJP label mismatch for aligned case {source_id}."
                    )
                full_text = str(text or "").strip()
                cases.append(
                    BenchmarkCase(
                        dataset="il_tur_cjpe",
                        case_id=f"il-tur-cjpe-{source_id}",
                        source_split=TEMPLATE_SOURCE_SPLIT,
                        input_text=full_text,
                        gold_label=gold_label,
                        labels=labels,
                        label_descriptions=descriptions,
                        task_instruction=(
                            "Predict whether the Indian Supreme Court accepts or "
                            "rejects the appeal from the supplied case document."
                        ),
                        metadata={
                            "input_variant": "official_cjpe_document",
                            "source_id": source_id,
                            "original_source_split": "multi_train",
                            "selection_split": TEMPLATE_SOURCE_SPLIT,
                            "selection_semantic_family": int(
                                facts["semantic_family"]
                            ),
                            "ljp_source_name": str(facts["name"]),
                            "facts_characters": len(str(facts["text"])),
                            "original_characters": len(full_text),
                            "source_url": (
                                "https://huggingface.co/datasets/"
                                "Exploration-Lab/IL-TUR"
                            ),
                            "license": "CC-BY-NC-SA-4.0",
                        },
                    )
                )
    return cases


def _distribution_comparison(
    population: pd.Series,
    selected: pd.Series,
) -> dict[str, Any]:
    population_counts = Counter(str(value) for value in population)
    selected_counts = Counter(str(value) for value in selected)
    population_total = len(population)
    selected_total = len(selected)
    return {
        key: {
            "training_cases": population_counts[key],
            "selected_cases": selected_counts[key],
            "training_share": population_counts[key] / population_total,
            "selected_share": selected_counts[key] / selected_total,
        }
        for key in sorted(population_counts)
    }


def _source_id(value: Any) -> str:
    return re.sub(r"\.txt$", "", str(value).strip(), flags=re.IGNORECASE)


def _case_decade(source_id: str) -> str:
    match = re.match(r"^(\d{4})_", source_id)
    return f"{int(match.group(1)) // 10 * 10}s" if match else "unknown"


def _stable_rank(seed: int, case_id: str) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()
