from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import resolve_path
from .io_utils import canonical_json, sha256_text, write_jsonl
from .legal_flux import resolve_project_file
from .legalhk_data import LEGALHK_PARQUET_URL, download_file, legalhk_index
from .models import (
    LegalFluxCandidateResponse,
    LegalFluxConsolidationResponse,
    LegalFluxGapAuditResponse,
    LegalFluxTemplate,
    NormalizedCase,
)
from .runner import load_cases
from .legal_flux_xsim import SentenceTransformerDenseEncoder


DEMAND_PRIORITY = [
    "procedural_threshold_check",
    "evidence_and_burden_assessment",
    "defense_or_counterargument_check",
    "precedent_or_analogy_handling",
    "remedy_discretion_check",
    "multi_issue_composition",
    "long_fact_filtering",
    "supplied_rule_extraction",
    "rule_recall_or_doctrine_identification",
    "issue_spotting_gap",
    "dual_issue_resolution",
    "focused_issue_resolution",
]

COARSE_LEGAL_FAMILY_BY_BROAD_DOMAIN = {
    "criminal": "criminal",
    "immigration_public": "immigration",
    "public_law": "public_administrative",
    "procedure_appeal": "civil_private",
    "company_insolvency": "civil_private",
    "contract_debt": "civil_private",
    "property_land": "civil_private",
    "family_trust_probate": "civil_private",
    "tort_damages": "civil_private",
    "employment_labor": "civil_private",
    "other_legal": "other_uncertain",
}


class TemplateBatchEncoder(Protocol):
    model_name: str

    def encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        ...


def export_legal_flux_chatgpt_batches(
    config: dict[str, Any],
    *,
    dense_encoder: TemplateBatchEncoder | None = None,
) -> dict[str, Any]:
    cases = [
        case
        for case in load_cases(config)
        if case.metadata.get("selection_split") == "template_source"
    ]
    if not cases:
        raise RuntimeError("No LegalFlux template-source cases found. Run flux-prepare first.")
    flux_config = config["legal_flux"]
    output_dir = resolve_project_file(
        flux_config.get(
            "gemini_batch_dir",
            flux_config.get(
                "chatgpt_batch_dir",
                "reports/legal_flux/template_distillation/gemini31_pro_batches",
            ),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = output_dir / "01_semantic_family_batches"
    prompts_dir = output_dir / "prompts"
    for directory in (batch_dir, prompts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _clear_generated_files(batch_dir, patterns=("*.jsonl",))
    _clear_generated_files(prompts_dir, patterns=("*.md",))
    _clear_generated_files(output_dir, patterns=("*.json", "*.md"))

    human_outputs = _template_source_human_outputs(config, cases)
    min_cases = int(flux_config.get("template_batch_min_cases", 24))
    target_cases = int(flux_config.get("template_batch_target_cases", 30))
    max_cases = int(flux_config.get("template_batch_max_cases", 36))
    if not 1 <= min_cases <= target_cases <= max_cases:
        raise ValueError(
            "Template batch sizes must satisfy 1 <= min <= target <= max."
        )
    if dense_encoder is None:
        dense_encoder = SentenceTransformerDenseEncoder(
            str(flux_config.get("template_batch_embedding_model", "BAAI/bge-m3")),
            device=flux_config.get("template_batch_device") or None,
            max_length=int(flux_config.get("template_batch_embedding_max_length", 8192)),
        )
    semantic_batches = _build_semantic_batches(
        cases,
        human_outputs=human_outputs,
        min_cases=min_cases,
        target_cases=target_cases,
        max_cases=max_cases,
        seed=config["project"]["seed"],
        encoder=dense_encoder,
        batch_size=int(flux_config.get("template_batch_embedding_batch_size", 8)),
        cache_dir=output_dir / "00_semantic_clustering",
    )
    manifest_batches = []
    for index, batch in enumerate(semantic_batches, start=1):
        path = batch_dir / _batch_filename("semantic_family", index, batch["label"])
        write_jsonl(
            path,
            [_case_record(case, human_outputs) for case in batch["cases"]],
        )
        manifest_batches.append(
            _batch_manifest_row("semantic_family", index, path, batch)
        )

    schema_path = output_dir / "legal_flux_template.schema.json"
    schema_path.write_text(
        json.dumps(LegalFluxTemplate.model_json_schema(), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    candidate_schema_path = output_dir / "legal_flux_candidate_response.schema.json"
    candidate_schema_path.write_text(
        json.dumps(
            LegalFluxCandidateResponse.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    consolidation_schema_path = output_dir / "legal_flux_consolidation_response.schema.json"
    consolidation_schema_path.write_text(
        json.dumps(
            LegalFluxConsolidationResponse.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    gap_schema_path = output_dir / "legal_flux_gap_audit_response.schema.json"
    gap_schema_path.write_text(
        json.dumps(
            LegalFluxGapAuditResponse.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    coverage = _coverage_summary(cases, manifest_batches)
    max_candidates = int(flux_config.get("template_batch_max_candidates", 5))
    minimum_support_cases = int(
        flux_config.get("template_batch_minimum_support_cases", 3)
    )
    _write_prompts(
        prompts_dir,
        coverage,
        max_candidates=max_candidates,
        minimum_support_cases=minimum_support_cases,
    )
    coverage["candidate_prompt_size_estimates"] = _candidate_prompt_size_estimates(
        manifest_batches=manifest_batches,
        schema_path=candidate_schema_path,
        candidate_prompt_path=prompts_dir / "01_generate_candidate_templates.md",
    )
    (output_dir / "coverage_summary.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(_readme(), encoding="utf-8")
    batch_case_ids = [
        case_id
        for batch in manifest_batches
        for case_id in batch.get("case_ids", [])
    ]
    unique_batched_case_ids = set(batch_case_ids)
    source_case_ids = {case.case_id for case in cases}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "template_source_cases": len(cases),
        "semantic_family_batches": len(semantic_batches),
        "batch_size_bounds": {
            "minimum": min_cases,
            "target": target_cases,
            "maximum": max_cases,
        },
        "max_candidates_per_batch": max_candidates,
        "minimum_support_cases": minimum_support_cases,
        "clustering_model": dense_encoder.model_name,
        "clustering_method": (
            "coarse legal-family partition, equal-weight normalized case/reasoning "
            "view concatenation, seeded KMeans, deterministic similarity rebalancing"
        ),
        "clustering_seed": int(config["project"]["seed"]),
        "coarse_legal_family_counts": dict(
            Counter(_coarse_legal_family(case) for case in cases).most_common()
        ),
        "coarse_legal_family_batch_counts": dict(
            Counter(batch["coarse_legal_family"] for batch in manifest_batches).most_common()
        ),
        "coarse_legal_family_mapping": COARSE_LEGAL_FAMILY_BY_BROAD_DOMAIN,
        "clustering_views": {
            "case_view": [
                "lawsuit_type",
                "claim_and_remedy",
                "issues",
                "authorities",
                "relevant_cases",
                "facts_verbatim",
            ],
            "reasoning_view": ["court_reasoning", "judgment_decision"],
            "combination": "L2-normalize each view, concatenate equally, L2-normalize",
        },
        "court_reasoning_included": bool(
            flux_config.get("template_include_court_reasoning", True)
        ),
        "judgment_decision_included": bool(
            flux_config.get("template_include_judgment_decision", True)
        ),
        "verbatim_facts_included": bool(
            flux_config.get("template_include_verbatim_facts", True)
        ),
        "unique_batched_source_cases": len(unique_batched_case_ids & source_case_ids),
        "source_case_coverage_rate": (
            len(unique_batched_case_ids & source_case_ids) / len(source_case_ids)
            if source_case_ids
            else 0.0
        ),
        "duplicate_case_appearances": len(batch_case_ids) - len(set(batch_case_ids)),
        "unbatched_source_case_ids": sorted(source_case_ids - unique_batched_case_ids),
        "output_dir": str(output_dir),
        "schema": str(schema_path),
        "candidate_schema": str(candidate_schema_path),
        "consolidation_schema": str(consolidation_schema_path),
        "gap_audit_schema": str(gap_schema_path),
        "readme": str(readme_path),
        "coverage_summary": str(output_dir / "coverage_summary.json"),
        "batches": manifest_batches,
    }
    manifest["manifest_hash"] = sha256_text(canonical_json(manifest_batches))
    (output_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "template_source_cases": len(cases),
        "semantic_family_batches": len(semantic_batches),
        "batch_manifest": str(output_dir / "batch_manifest.json"),
        "coverage_summary": str(output_dir / "coverage_summary.json"),
        "prompts_dir": str(prompts_dir),
        "schema": str(schema_path),
        "candidate_schema": str(candidate_schema_path),
        "readme": str(readme_path),
    }


def _template_source_human_outputs(
    config: dict[str, Any],
    cases: list[NormalizedCase],
) -> dict[str, dict[str, str]]:
    flux_config = config["legal_flux"]
    include_reasoning = bool(
        flux_config.get("template_include_court_reasoning", True)
    )
    include_decision = bool(
        flux_config.get("template_include_judgment_decision", True)
    )
    include_verbatim_facts = bool(
        flux_config.get("template_include_verbatim_facts", True)
    )
    if not include_reasoning and not include_decision and not include_verbatim_facts:
        return {case.case_id: {} for case in cases}

    requested_fields: list[tuple[str, str]] = []
    if include_verbatim_facts:
        requested_fields.append(("facts_verbatim", "more_facts"))
    if include_reasoning:
        requested_fields.append(("court_reasoning", "court_reasoning"))
    if include_decision:
        requested_fields.append(("judgment_decision", "judgment_decision"))
    from_metadata = {
        case.case_id: {
            output_field: str(case.metadata.get(output_field, ""))
            for output_field, _ in requested_fields
        }
        for case in cases
    }
    if all(
        all(output_field in case.metadata for output_field, _ in requested_fields)
        for case in cases
    ):
        return from_metadata

    raw_dir = resolve_path(config, "raw_dir")
    parquet_path = raw_dir / "legalhk" / "train.parquet"
    download_file(LEGALHK_PARQUET_URL, parquet_path)
    raw_columns = list(dict.fromkeys(raw_field for _, raw_field in requested_fields))
    frame = pd.read_parquet(parquet_path, columns=raw_columns).fillna("")
    outputs: dict[str, dict[str, str]] = {}
    for case in cases:
        index = legalhk_index(case.case_id)
        if index not in frame.index:
            raise ValueError(f"Raw LegalHK row {index} is missing for {case.case_id}.")
        outputs[case.case_id] = {
            output_field: str(frame.at[index, raw_field])
            for output_field, raw_field in requested_fields
        }
    return outputs


def _build_semantic_batches(
    cases: list[NormalizedCase],
    *,
    human_outputs: dict[str, dict[str, str]],
    min_cases: int,
    target_cases: int,
    max_cases: int,
    seed: int,
    encoder: TemplateBatchEncoder,
    batch_size: int,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    cases = sorted(cases, key=lambda case: case.case_id)
    case_view_texts = [
        _template_case_view_text(case, human_outputs.get(case.case_id, {}))
        for case in cases
    ]
    reasoning_view_texts = [
        _template_reasoning_view_text(case, human_outputs.get(case.case_id, {}))
        for case in cases
    ]
    case_embeddings = _load_or_encode_template_embeddings(
        cases=cases,
        texts=case_view_texts,
        encoder=encoder,
        batch_size=batch_size,
        cache_dir=cache_dir,
        cache_key="case_view",
    )
    reasoning_embeddings = _load_or_encode_template_embeddings(
        cases=cases,
        texts=reasoning_view_texts,
        encoder=encoder,
        batch_size=batch_size,
        cache_dir=cache_dir,
        cache_key="reasoning_view",
    )
    embeddings = _combine_dual_view_embeddings(
        case_embeddings,
        reasoning_embeddings,
    )
    family_indices: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        family_indices[_coarse_legal_family(case)].append(index)

    batches: list[dict[str, Any]] = []
    for family in sorted(family_indices):
        indices = family_indices[family]
        selected_cases = [cases[index] for index in indices]
        selected_embeddings = embeddings[np.asarray(indices)]
        batches.extend(
            _cluster_legal_family(
                selected_cases,
                embeddings=selected_embeddings,
                family=family,
                min_cases=min_cases,
                target_cases=target_cases,
                max_cases=max_cases,
                seed=seed,
            )
        )
    batches.sort(key=lambda batch: (batch["coarse_legal_family"], batch["cases"][0].case_id))
    return batches


def _cluster_legal_family(
    cases: list[NormalizedCase],
    *,
    embeddings: np.ndarray,
    family: str,
    min_cases: int,
    target_cases: int,
    max_cases: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(cases) <= max_cases:
        return [_semantic_family_batch(cases, embeddings=embeddings, family=family)]
    lower_clusters = math.ceil(len(cases) / max_cases)
    upper_clusters = len(cases) // min_cases
    if lower_clusters > upper_clusters:
        raise ValueError(
            f"Coarse legal family {family!r} has {len(cases)} cases, which cannot "
            f"be partitioned within batch bounds {min_cases}-{max_cases}."
        )
    cluster_count = min(
        max(round(len(cases) / target_cases), lower_clusters),
        upper_clusters,
    )
    kmeans = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings).tolist()
    labels = _rebalance_cluster_labels(
        labels,
        embeddings=embeddings,
        centers=np.asarray(kmeans.cluster_centers_, dtype=np.float32),
        min_cases=min_cases,
        max_cases=max_cases,
    )
    clusters: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        clusters[label].append(index)
    return [
        _semantic_family_batch(
            [cases[index] for index in indices],
            embeddings=embeddings[np.asarray(indices)],
            family=family,
        )
        for _, indices in sorted(
            clusters.items(),
            key=lambda item: min(cases[index].case_id for index in item[1]),
        )
    ]


def _semantic_family_batch(
    cases: list[NormalizedCase],
    *,
    embeddings: np.ndarray,
    family: str,
) -> dict[str, Any]:
    cases = sorted(cases, key=lambda case: case.case_id)
    centroid = _normalize_embedding_rows(np.mean(embeddings, axis=0, keepdims=True))[0]
    similarities = np.asarray(embeddings @ centroid, dtype=np.float32)
    return {
        "label": f"{family}__semantic_cluster",
        "coarse_legal_family": family,
        "group_key": {"mode": "semantic_within_coarse_legal_family", "family": family},
        "cases": cases,
        "semantic_coherence": {
            "mean_cosine_to_centroid": round(float(np.mean(similarities)), 6),
            "p10_cosine_to_centroid": round(float(np.quantile(similarities, 0.10)), 6),
            "minimum_cosine_to_centroid": round(float(np.min(similarities)), 6),
        },
    }


def _load_or_encode_template_embeddings(
    *,
    cases: list[NormalizedCase],
    texts: list[str],
    encoder: TemplateBatchEncoder,
    batch_size: int,
    cache_dir: Path,
    cache_key: str,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = cache_dir / f"{cache_key}_embeddings.npy"
    manifest_path = cache_dir / f"{cache_key}_manifest.json"
    expected = {
        "case_ids": [case.case_id for case in cases],
        "corpus_hash": sha256_text(canonical_json(texts)),
        "model": encoder.model_name,
    }
    if embedding_path.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in expected.items()):
            return _normalize_embedding_rows(np.load(embedding_path))
    embeddings = np.asarray(
        encoder.encode(texts, batch_size=batch_size),
        dtype=np.float32,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(cases):
        raise ValueError("Template batch encoder returned an invalid embedding matrix.")
    embeddings = _normalize_embedding_rows(embeddings)
    np.save(embedding_path, embeddings)
    manifest_path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return embeddings


def _normalize_embedding_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def _combine_dual_view_embeddings(
    case_embeddings: np.ndarray,
    reasoning_embeddings: np.ndarray,
) -> np.ndarray:
    case_embeddings = _normalize_embedding_rows(case_embeddings)
    reasoning_embeddings = _normalize_embedding_rows(reasoning_embeddings)
    if case_embeddings.shape[0] != reasoning_embeddings.shape[0]:
        raise ValueError("Case-view and reasoning-view embeddings must have equal row counts.")
    return _normalize_embedding_rows(
        np.concatenate([case_embeddings, reasoning_embeddings], axis=1)
    )


def _rebalance_cluster_labels(
    labels: list[int],
    *,
    embeddings: np.ndarray,
    centers: np.ndarray,
    min_cases: int,
    max_cases: int,
) -> list[int]:
    labels = list(labels)
    similarities = embeddings @ _normalize_embedding_rows(centers).T

    def members(label: int) -> list[int]:
        return [index for index, value in enumerate(labels) if value == label]

    cluster_ids = list(range(centers.shape[0]))
    while True:
        oversized = [label for label in cluster_ids if len(members(label)) > max_cases]
        if not oversized:
            break
        source = max(oversized, key=lambda label: len(members(label)))
        destinations = [
            label for label in cluster_ids if label != source and len(members(label)) < max_cases
        ]
        if not destinations:
            raise ValueError("Could not rebalance oversized semantic template batches.")
        move = min(
            (
                (similarities[index, source] - similarities[index, destination], index, destination)
                for index in members(source)
                for destination in destinations
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        labels[move[1]] = move[2]

    while True:
        undersized = [label for label in cluster_ids if len(members(label)) < min_cases]
        if not undersized:
            break
        destination = min(undersized, key=lambda label: (len(members(label)), label))
        donors = [label for label in cluster_ids if len(members(label)) > min_cases]
        if not donors:
            raise ValueError("Could not rebalance undersized semantic template batches.")
        move = min(
            (
                (similarities[index, source] - similarities[index, destination], index, source)
                for source in donors
                for index in members(source)
            ),
            key=lambda item: (item[0], item[1], item[2]),
        )
        labels[move[1]] = destination
    return labels


def _template_case_view_text(
    case: NormalizedCase,
    human_output: dict[str, str],
) -> str:
    facts = human_output.get("facts_verbatim") or "\n".join(case.facts.values())
    return (
        f"[LAWSUIT TYPE]\n{case.metadata.get('lawsuit_type', '')}\n\n"
        f"[CLAIM AND REMEDY]\n{case.claim}\n{case.requested_remedy or ''}\n\n"
        f"[ISSUES]\n{' '.join(case.reference_issues)}\n\n"
        f"[AUTHORITIES]\n{case.authorities or ''}\n\n"
        f"[RELEVANT CASES]\n{case.metadata.get('relevant_cases', '')}\n\n"
        f"[FACTS]\n{facts}"
    )


def _template_reasoning_view_text(
    case: NormalizedCase,
    human_output: dict[str, str],
) -> str:
    return (
        f"[COURT REASONING]\n{human_output.get('court_reasoning', '')}\n\n"
        f"[JUDGMENT DECISION]\n{human_output.get('judgment_decision', '')}"
    )


def _batch_manifest_row(
    kind: str,
    index: int,
    path: Path,
    batch: dict[str, Any],
) -> dict[str, Any]:
    cases = batch["cases"]
    coarse_family_counts = Counter(_coarse_legal_family(case) for case in cases)
    return {
        "batch_id": f"{kind}_{index:03d}",
        "kind": kind,
        "label": batch["label"],
        "path": str(path),
        "case_count": len(cases),
        "case_ids": [case.case_id for case in cases],
        "coarse_legal_family": batch["coarse_legal_family"],
        "coarse_legal_family_counts": dict(coarse_family_counts.most_common()),
        "coarse_legal_family_purity": (
            max(coarse_family_counts.values()) / len(cases) if cases else 0.0
        ),
        "semantic_coherence": batch["semantic_coherence"],
        "primary_family_counts": dict(Counter(_primary_family(case) for case in cases).most_common()),
        "demand_focus_counts": dict(Counter(_demand_focus(case) for case in cases).most_common()),
        "broad_domain_counts": dict(
            Counter(
                str(case.metadata.get("broad_domain", ""))
                for case in cases
            ).most_common()
        ),
        "authority_bucket_counts": dict(
            Counter(
                str(case.metadata.get("authority_bucket", ""))
                for case in cases
            ).most_common()
        ),
        "lawsuit_type_top10": dict(
            Counter(str(case.metadata.get("lawsuit_type", "")) for case in cases).most_common(10)
        ),
        "trajectory_prefix_count": len({_trajectory_prefix(case) for case in cases}),
        "file_sha256": sha256_text(path.read_text(encoding="utf-8")),
    }


def _case_record(
    case: NormalizedCase,
    human_outputs: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    record = {
        "case_id": case.case_id,
        "claim": case.claim,
        "requested_remedy": case.requested_remedy,
        "parties": case.parties,
        "facts": case.facts,
        "facts_verbatim": (human_outputs or {}).get(case.case_id, {}).get(
            "facts_verbatim"
        ),
        "lawsuit_type": case.metadata.get("lawsuit_type"),
        "reference_issues": case.reference_issues,
        "authorities": case.authorities,
        "relevant_cases": case.metadata.get("relevant_cases"),
    }
    record.update((human_outputs or {}).get(case.case_id, {}))
    return record


def _coverage_summary(
    cases: list[NormalizedCase],
    manifest_batches: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "template_source_cases": len(cases),
        "coarse_legal_family_counts": dict(
            Counter(_coarse_legal_family(case) for case in cases).most_common()
        ),
        "coarse_legal_family_batch_counts": dict(
            Counter(
                batch["coarse_legal_family"] for batch in manifest_batches
            ).most_common()
        ),
        "primary_family_counts": dict(Counter(_primary_family(case) for case in cases).most_common()),
        "demand_focus_counts": dict(Counter(_demand_focus(case) for case in cases).most_common()),
        "all_reasoning_demand_counts": dict(_all_demand_counts(cases).most_common()),
        "trajectory_prefix_counts_top50": dict(
            Counter(_trajectory_prefix(case) for case in cases).most_common(50)
        ),
        "batch_count": len(manifest_batches),
        "batched_case_ids": len(
            {
                case_id
                for batch in manifest_batches
                for case_id in batch.get("case_ids", [])
            }
        ),
        "batch_kind_counts": dict(Counter(batch["kind"] for batch in manifest_batches)),
    }


def _candidate_prompt_size_estimates(
    *,
    manifest_batches: list[dict[str, Any]],
    schema_path: Path,
    candidate_prompt_path: Path,
) -> dict[str, Any]:
    schema_text = schema_path.read_text(encoding="utf-8")
    prompt_text = candidate_prompt_path.read_text(encoding="utf-8")
    rows = []
    for batch in manifest_batches:
        batch_path = Path(batch["path"])
        char_count = (
            len(prompt_text)
            + len(schema_text)
            + len(batch_path.read_text(encoding="utf-8"))
        )
        rows.append(
            {
                "batch_id": batch["batch_id"],
                "label": batch["label"],
                "kind": batch["kind"],
                "characters": char_count,
                "tokens_at_3_5_chars_per_token": _ceil_div(char_count, 3.5),
                "tokens_at_4_chars_per_token": _ceil_div(char_count, 4.0),
            }
        )
    char_counts = [row["characters"] for row in rows]
    tokens_35 = [row["tokens_at_3_5_chars_per_token"] for row in rows]
    tokens_4 = [row["tokens_at_4_chars_per_token"] for row in rows]
    return {
        "per_batch_count": len(rows),
        "character_stats": _numeric_stats(char_counts),
        "token_estimates": {
            "chars_per_token_3_5": _numeric_stats(tokens_35),
            "chars_per_token_4": _numeric_stats(tokens_4),
        },
        "largest_batches": sorted(
            rows,
            key=lambda row: row["characters"],
            reverse=True,
        )[:5],
    }


def _numeric_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0.0, "max": 0, "total": 0}
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        median: float | int = ordered[midpoint]
    else:
        median = (ordered[midpoint - 1] + ordered[midpoint]) / 2
    return {
        "min": min(ordered),
        "median": median,
        "mean": sum(ordered) / len(ordered),
        "max": max(ordered),
        "total": sum(ordered),
    }


def _ceil_div(value: int, divisor: float) -> int:
    return int((value + divisor - 1) // divisor)


def _all_demand_counts(cases: list[NormalizedCase]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for case in cases:
        counter.update(_split_profile(case, "reasoning_demands"))
    return counter


def _primary_family(case: NormalizedCase) -> str:
    return (_split_profile(case, "template_families") or ["general_legal_reasoning"])[0]


def _coarse_legal_family(case: NormalizedCase) -> str:
    broad_domain = str(case.metadata.get("broad_domain") or "").strip()
    return COARSE_LEGAL_FAMILY_BY_BROAD_DOMAIN.get(
        broad_domain,
        "other_uncertain",
    )


def _demand_focus(case: NormalizedCase) -> str:
    demands = set(_split_profile(case, "reasoning_demands"))
    for demand in DEMAND_PRIORITY:
        if demand in demands:
            return demand
    return "general_resolution"


def _trajectory_prefix(case: NormalizedCase) -> str:
    profile = case.metadata.get("legal_flux_profile") or {}
    signature = str(profile.get("trajectory_signature") or "")
    parts = [part.strip() for part in signature.split(" > ") if part.strip()]
    return " > ".join(parts[:5]) if parts else "unknown"


def _split_profile(case: NormalizedCase, key: str) -> list[str]:
    profile = case.metadata.get("legal_flux_profile") or {}
    value = str(profile.get(key) or "")
    return [part for part in value.split("|") if part]


def _batch_filename(kind: str, index: int, label: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", label).strip("_").lower()
    return f"{kind}_{index:03d}_{safe[:80]}.jsonl"


def _clear_generated_files(directory: Path, *, patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()


def _write_prompts(
    prompts_dir: Path,
    coverage: dict[str, Any],
    *,
    max_candidates: int,
    minimum_support_cases: int,
) -> None:
    (prompts_dir / "01_generate_candidate_templates.md").write_text(
        _candidate_generation_prompt(
            max_candidates=max_candidates,
            minimum_support_cases=minimum_support_cases,
        ),
        encoding="utf-8",
    )
    (prompts_dir / "02_merge_deduplicate_templates.md").write_text(
        _merge_prompt(),
        encoding="utf-8",
    )
    (prompts_dir / "03_coverage_audit_and_gap_fill.md").write_text(
        _coverage_audit_prompt(coverage),
        encoding="utf-8",
    )


def _candidate_generation_prompt(
    *,
    max_candidates: int,
    minimum_support_cases: int,
) -> str:
    return f"""You are constructing a compact, high-quality library of reusable legal
reasoning templates for all-domain Hong Kong court cases.

Analyze the supplied batch and return at most {max_candidates} candidate
templates. Zero candidates is valid and preferable to creating a weak,
duplicative, overly broad, or overly specific template.

A candidate is eligible only when:

1. At least {minimum_support_cases} supplied cases exhibit the same underlying
   legal reasoning operation.
2. The supporting cases include at least two meaningfully different factual or
   procedural manifestations of that operation.
3. The template would provide useful guidance for unseen cases of similar type.
4. Its application can be described with clear positive triggers and meaningful
   boundaries.
5. Its reasoning flow performs a distinct legal operation rather than merely
   restating a legal topic.

TARGET ABSTRACTION:

A good template is a legal reasoning operation with a middle-to-high abstraction
level, such as a procedural gate, allocation of burden, structured evidential
assessment, authority-synthesis method, multi-factor legal test, defense
analysis, remedy selection, or appellate review operation.

Reject a candidate if it is:

- so general that it merely says to identify issues, apply law to facts,
  consider evidence, or reach a conclusion;
- tied to one case, one unusual fact pattern, one party, one outcome, or a
  narrowly worded procedural event;
- primarily a summary of substantive law rather than an executable reasoning
  procedure;
- directionally framed to produce a particular support/reject result;
- substantially equivalent to another candidate in this batch;
- wholly contained within another candidate that performs the same reasoning
  operation.

For every candidate:

- Use a concise, retrieval-friendly template_name.
- Use a concise set of normalized knowledge_tags.
- Make description explain the reusable reasoning operation.
- Make application_scenario state when to use it and, if applicable and helpful,
  when not to use it.
- Give at least 2 ordered, operational reasoning_flow steps.
- Use a synthetic example_application that does not reproduce a source case.
- Record the supporting_case_ids and explain the shared_pattern.
- Make scope_exclusions identify contexts where the operation should not be used.
- Make support_count equal the number of distinct supporting_case_ids.
- Do not copy party names, dates, amounts, citations, outcomes, or F-numbered
  facts.

Before returning, compare all proposed candidates with one another. Merge
near-duplicates and remove any candidate subsumed by another.

Return one JSON object matching the candidate-template output schema.
"""


def _merge_prompt() -> str:
    return """You are consolidating candidate LegalFlux templates into one compact,
high-quality library for all-domain Hong Kong court cases.

Review the complete candidate set together. Keep only reusable, distinct legal
reasoning operations at a middle-to-high abstraction level. There is no required
or preferred final template count: retain every candidate that independently
meets the quality standard, and do not add or remove templates to reach a quota.

Merge candidates that perform substantially the same reasoning operation.
Remove candidates that are overly broad, overly specific, directionally tied to
an outcome, weakly supported, wholly subsumed by another template, or primarily
summaries of substantive law rather than executable reasoning procedures.

For each retained template:

- use a concise retrieval-friendly name and normalized tags;
- state clear positive application conditions and meaningful boundaries;
- provide an ordered operational reasoning flow;
- use a synthetic example that does not reproduce a source case;
- record every source_candidate_id that was retained or merged into it;
- do not copy source case IDs, parties, dates, amounts, citations, outcomes, or
  F-numbered facts into the executable template fields.

Return one JSON object matching the consolidation output schema. Template IDs
will be assigned deterministically after this consolidation call.
"""


def _coverage_audit_prompt(coverage: dict[str, Any]) -> str:
    return f"""You are auditing one original source-case batch against the current
consolidated LegalFlux template library.

Inspect every supplied case, including its court reasoning and judgment
decision. Identify whether the library covers the recurring legal reasoning
operations actually exhibited by this batch. An individual uncovered case is
not a library gap. Propose a gap candidate only when the same missing operation
is supported by the configured minimum number of supplied cases and satisfies
the same abstraction, reuse, manifestation, and boundary requirements as the
initial candidate stage.

Do not restate a legal topic, reproduce a source outcome, or propose a candidate
already covered or subsumed by the current library. Zero gap candidates is
valid and preferable to a weak addition. Gap candidates are proposals only and
will undergo a separate global adjudication before entering the final library.

Return one JSON object matching the gap-audit output schema.

Aggregate source coverage metadata:

```json
{json.dumps(coverage, ensure_ascii=False, indent=2)}
```
"""


def _readme() -> str:
    return """# Gemini 3.1 Pro LegalFlux Template-Pool Workflow

This folder supports the automated template-pool construction workflow for
Gemini 3.1 Pro on Vertex AI. The same artifacts can still be inspected manually before
spending API credit.

## Pass 1: Candidate templates

For each file in `01_semantic_family_batches`, send:

- one batch JSONL file
- `legal_flux_candidate_response.schema.json`
- `prompts/01_generate_candidate_templates.md`

Save the returned candidate JSONL files under the API output folder.

## Pass 2: Merge and deduplicate

After candidate templates are generated, send:

- all candidate-template JSONL files
- `batch_manifest.json`
- `coverage_summary.json`
- `legal_flux_template.schema.json`
- `prompts/02_merge_deduplicate_templates.md`

Ask for one globally consolidated pool with no required template count.

## Pass 3: Coverage audit

Audit every original batch against the consolidated pool. Gap candidates are
globally adjudicated before a final pool is written. Then import the final pool:

```powershell
python -m legal_pilot --config configs\\legal_flux.yaml flux-import-templates --input path\\to\\final_templates.jsonl
```

These batches come only from the `template_source` split. Do not include
planner-train, trajectory-dev, or final-test cases in the template-pool creation
step.
"""
