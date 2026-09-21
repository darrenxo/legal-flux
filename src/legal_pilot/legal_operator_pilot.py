from __future__ import annotations

import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .clients import GenerationClient, GenerationResponseError, build_generation_client
from .config import resolve_path, resolve_project_path
from .embeddings import OllamaEmbeddingBackend, SimilarityBackend, TfidfSimilarityBackend
from .io_utils import canonical_json, latest_by_run_hash, read_jsonl, sha256_text, write_jsonl
from .ledger import JsonlLedger, make_run_hash
from .legal_flux_evaluation import _aggregate_frame
from .legal_flux_runner import _validated_run_tag
from .models import (
    FinalAnalysis,
    IssueConclusion,
    LegalIssueFinding,
    LegalIssueGraph,
    LegalIssueNode,
    LegalOperator,
    LegalOperatorFinalDecision,
    NormalizedCase,
)
from .prompting import load_prompt
from .runner import _load_schema, _response_trace, load_cases
from .scoring import score_record


ISSUE_TYPES = {
    "procedural_gate",
    "claim_elements",
    "legal_interpretation",
    "authority",
    "evidence",
    "defense",
    "causation",
    "loss_quantum",
    "discretion",
    "appellate_review",
    "remedy",
}


def run_legal_operator_pilot(
    config: dict[str, Any],
    *,
    phase: str = "smoke",
    case_limit: int | None = None,
    run_tag: str | None = None,
    dry_run: bool = False,
    fail_on_errors: bool = False,
) -> dict[str, Any]:
    normalized_phase = phase.replace("-", "_")
    if normalized_phase not in {"smoke", "trajectory_dev"}:
        raise ValueError("Operator pilot phase must be smoke or trajectory_dev.")
    normalized_tag = _validated_run_tag(run_tag) or "issue-graph-operator-local-v1"
    cases = [
        case
        for case in load_cases(config)
        if case.metadata.get("selection_split") == normalized_phase
    ]
    if case_limit is not None:
        if case_limit < 1:
            raise ValueError("case_limit must be at least 1.")
        cases = cases[:case_limit]
    if not cases:
        raise ValueError(f"No cases found for operator pilot phase {normalized_phase}.")

    operators = load_operator_pool(config)
    run_dir = (
        resolve_path(config, "runs_dir")
        / "operator_pilot"
        / normalized_phase
        / "experiments"
        / normalized_tag
    )
    workflow_hash = operator_pilot_workflow_hash(config, operators)
    planned = [
        {
            "run_hash": operator_pilot_run_hash(
                case,
                phase=normalized_phase,
                workflow_hash=workflow_hash,
                seed=int(config["model"]["seed"]),
            ),
            "case_id": case.case_id,
            "phase": normalized_phase,
        }
        for case in cases
    ]
    if dry_run:
        return {
            "phase": normalized_phase,
            "cases": len(cases),
            "operator_count": len(operators),
            "run_dir": str(run_dir),
            "run_tag": normalized_tag,
            "workflow_hash": workflow_hash,
            "dry_run": True,
        }

    client = build_generation_client(config)
    model_name = str(config["model"]["name"])
    model_info = client.model_info(model_name)
    if model_info is None:
        client.close()
        raise RuntimeError(
            f"Model {model_name!r} is not exposed at {config['model']['base_url']}."
        )
    model_digest = str(model_info.get("digest") or model_name)
    similarity_backend = _build_operator_similarity_backend(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_plan.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "phase": normalized_phase,
                "run_tag": normalized_tag,
                "condition": "operator_issue_graph",
                "model": model_name,
                "model_digest": model_digest,
                "workflow_hash": workflow_hash,
                "operator_pool_hash": operator_pool_hash(operators),
                "jobs": planned,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = JsonlLedger(run_dir / "generations.jsonl")
    completed = skipped = errors = 0
    try:
        for index, (case, plan) in enumerate(zip(cases, planned, strict=True), start=1):
            if ledger.contains(plan["run_hash"]):
                skipped += 1
                continue
            base = {
                "run_hash": plan["run_hash"],
                "dataset": case.dataset,
                "case_id": case.case_id,
                "variant_id": case.variant_id,
                "condition": "operator_issue_graph",
                "phase": normalized_phase,
                "model_name": model_name,
                "model_digest": model_digest,
                "workflow_hash": workflow_hash,
                "operator_pool_hash": operator_pool_hash(operators),
                "seed": int(config["model"]["seed"]),
                "gold_answer": case.gold_answer,
                "metadata": case.metadata,
            }
            try:
                analysis, trace = execute_operator_issue_graph_case(
                    client,
                    config,
                    case,
                    operators=operators,
                    similarity_backend=similarity_backend,
                )
                record = {
                    **base,
                    "status": "ok",
                    "parsed_json": analysis.model_dump(mode="json", exclude_defaults=True),
                    **trace,
                }
            except Exception as exc:
                errors += 1
                record = {
                    **base,
                    "status": "error",
                    "parsed_json": None,
                    "raw_response": (
                        exc.raw_text
                        if isinstance(exc, GenerationResponseError)
                        else None
                    ),
                    "issue_graph": None,
                    "selected_operators": None,
                    "issue_findings": None,
                    "prompt_hashes": {},
                    "elapsed_seconds": None,
                    "prompt_tokens": None,
                    "output_tokens": None,
                    "calls": 0,
                    "repair_actions": [],
                    "schema_errors": [str(exc)],
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }
            ledger.append(record)
            completed += 1
            print(
                f"Operator pilot progress: {index}/{len(cases)}; "
                f"completed={completed}, skipped={skipped}, errors={errors}",
                flush=True,
            )
    finally:
        client.close()
        if hasattr(similarity_backend, "close"):
            similarity_backend.close()

    summary = score_legal_operator_pilot(config, run_dir=run_dir)
    result = {
        **summary,
        "completed": completed,
        "skipped": skipped,
        "generation_errors": errors,
        "model_digest": model_digest,
        "workflow_hash": workflow_hash,
        "operator_pool_hash": operator_pool_hash(operators),
    }
    if fail_on_errors and errors:
        raise RuntimeError(
            f"Operator pilot recorded {errors} error(s) under {run_dir}."
        )
    return result


def execute_operator_issue_graph_case(
    client: GenerationClient,
    config: dict[str, Any],
    case: NormalizedCase,
    *,
    operators: list[LegalOperator],
    similarity_backend: SimilarityBackend | None = None,
) -> tuple[FinalAnalysis, dict[str, Any]]:
    settings = config.get("operator_pilot", {})
    max_issues = min(4, max(1, int(settings.get("max_issues", 4))))
    max_depth = min(2, max(1, int(settings.get("max_depth", 2))))
    authorities = _number_authorities(case)
    common = {
        "model": config["model"]["name"],
        "temperature": config["model"]["temperature"],
        "seed": config["model"]["seed"],
        "context_length": config["model"]["context_length"],
    }
    schema_dir = resolve_path(config, "schemas_dir")
    raw_parts: list[str] = []
    prompt_hashes: dict[str, str] = {}
    elapsed = 0.0
    prompt_tokens = 0
    output_tokens = 0
    calls = 0
    repairs: list[str] = []
    schema_errors: list[str] = []

    graph_prompt, graph_hash = _render_operator_prompt(
        config,
        "legal_operator/issue_graph",
        case,
        authorities,
        max_issues=max_issues,
        max_depth=max_depth,
    )
    graph_schema = _load_schema(schema_dir / "legal_operator_issue_graph.json")
    graph_schema["properties"]["issues"]["maxItems"] = max_issues
    response = client.generate(
        prompt=graph_prompt,
        schema=graph_schema,
        max_tokens=int(settings.get("graph_max_tokens", 1200)),
        **common,
    )
    normalized_graph, graph_repairs = _normalize_issue_graph_payload(
        response.parsed,
        case=case,
        authorities=authorities,
        max_issues=max_issues,
        max_depth=max_depth,
    )
    graph = LegalIssueGraph.model_validate(normalized_graph)
    repairs.extend(graph_repairs)
    trace = _response_trace(response)
    repairs.extend(trace["repair_actions"])
    schema_errors.extend(trace["schema_errors"])
    raw_parts.append(response.raw_text)
    prompt_hashes["issue_graph"] = graph_hash
    elapsed += response.elapsed_seconds
    prompt_tokens += response.prompt_tokens or 0
    output_tokens += response.output_tokens or 0
    calls += 1

    findings: dict[str, LegalIssueFinding] = {}
    selected_operators: list[dict[str, Any]] = []
    for issue in _issue_execution_order(graph.issues):
        retrieval = retrieve_operator_for_issue(
            issue,
            operators,
            similarity_backend=similarity_backend,
        )
        operator = retrieval["operator"]
        child_findings = [
            findings[child.issue_id]
            for child in graph.issues
            if child.parent_id == issue.issue_id and child.issue_id in findings
        ]
        selection = {
            "issue_id": issue.issue_id,
            "operator_id": operator.operator_id,
            "operator_name": operator.operator_name,
            "retrieval_mode": retrieval["retrieval_mode"],
            "similarity": retrieval["similarity"],
            "candidate_operator_ids": retrieval["candidate_operator_ids"],
        }
        selected_operators.append(selection)
        assigned_facts = {
            fact_id: case.facts[fact_id]
            for fact_id in issue.fact_ids
            if fact_id in case.facts
        }
        assigned_authorities = {
            authority_id: authorities[authority_id]
            for authority_id in issue.authority_ids
            if authority_id in authorities
        }
        issue_prompt, issue_hash = _render_operator_prompt(
            config,
            "legal_operator/issue_execute",
            case,
            authorities,
            facts=(
                "\n".join(
                    f"{key}: {value}" for key, value in assigned_facts.items()
                )
                or "No direct facts assigned; reason only from child findings."
            ),
            numbered_authorities=(
                "\n".join(
                    f"{key}: {value}"
                    for key, value in assigned_authorities.items()
                )
                or "No direct authorities assigned; do not invent a rule."
            ),
            issue_graph=graph.model_dump(mode="json"),
            current_issue=issue.model_dump(mode="json"),
            child_findings=[item.model_dump(mode="json") for item in child_findings],
            selected_operator=operator.model_dump(mode="json"),
        )
        issue_schema = _load_schema(
            schema_dir / "legal_operator_issue_finding.json"
        )
        issue_schema["properties"]["issue_id"] = {"const": issue.issue_id}
        response = client.generate(
            prompt=issue_prompt,
            schema=issue_schema,
            max_tokens=int(settings.get("issue_max_tokens", 700)),
            **common,
        )
        normalized_finding, finding_repairs = _normalize_issue_finding_payload(
            response.parsed,
            issue=issue,
            case=case,
            authorities=authorities,
            child_issue_ids={item.issue_id for item in child_findings},
        )
        finding = LegalIssueFinding.model_validate(normalized_finding)
        findings[issue.issue_id] = finding
        repairs.extend(finding_repairs)
        trace = _response_trace(response)
        repairs.extend(trace["repair_actions"])
        schema_errors.extend(trace["schema_errors"])
        raw_parts.append(response.raw_text)
        prompt_hashes[f"issue_{issue.issue_id}"] = issue_hash
        elapsed += response.elapsed_seconds
        prompt_tokens += response.prompt_tokens or 0
        output_tokens += response.output_tokens or 0
        calls += 1

    ordered_findings = [findings[issue.issue_id] for issue in graph.issues]
    final_prompt, final_hash = _render_operator_prompt(
        config,
        "legal_operator/final_synthesis",
        case,
        authorities,
        issue_graph=graph.model_dump(mode="json"),
        issue_findings=[item.model_dump(mode="json") for item in ordered_findings],
    )
    final_schema = _load_schema(schema_dir / "legal_operator_final_decision.json")
    final_schema["properties"]["dispositive_issue_ids"]["items"] = {
        "enum": [issue.issue_id for issue in graph.issues]
    }
    response = client.generate(
        prompt=final_prompt,
        schema=final_schema,
        max_tokens=int(settings.get("final_max_tokens", 600)),
        **common,
    )
    normalized_final, final_repairs = _normalize_final_decision_payload(
        response.parsed,
        issue_ids={issue.issue_id for issue in graph.issues},
        root_finding=findings[_root_issue(graph.issues).issue_id],
    )
    final = LegalOperatorFinalDecision.model_validate(normalized_final)
    repairs.extend(final_repairs)
    trace = _response_trace(response)
    repairs.extend(trace["repair_actions"])
    schema_errors.extend(trace["schema_errors"])
    raw_parts.append(response.raw_text)
    prompt_hashes["final_synthesis"] = final_hash
    elapsed += response.elapsed_seconds
    prompt_tokens += response.prompt_tokens or 0
    output_tokens += response.output_tokens or 0
    calls += 1

    conclusion_map = {
        "supports_claim": "satisfied",
        "opposes_claim": "not_satisfied",
        "mixed": "unresolved",
        "unresolved": "unresolved",
    }
    analysis = FinalAnalysis(
        issue_conclusions=[
            IssueConclusion(
                issue_id=finding.issue_id,
                conclusion=conclusion_map[finding.conclusion],
                supporting_fact_ids=finding.supporting_fact_ids,
                opposing_fact_ids=finding.opposing_fact_ids,
                explanation=finding.resolution,
            )
            for finding in ordered_findings
        ],
        final_decision=final.final_decision,
        final_rationale=final.final_rationale,
    )
    return analysis, {
        "raw_response": "\n---CALL---\n".join(raw_parts),
        "issue_graph": graph.model_dump(mode="json"),
        "selected_operators": selected_operators,
        "issue_findings": [item.model_dump(mode="json") for item in ordered_findings],
        "dispositive_issue_ids": final.dispositive_issue_ids,
        "prompt_hashes": prompt_hashes,
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "calls": calls,
        "repair_actions": list(dict.fromkeys(repairs)),
        "schema_errors": list(dict.fromkeys(schema_errors)),
    }


def load_operator_pool(config_or_path: dict[str, Any] | str | Path) -> list[LegalOperator]:
    if isinstance(config_or_path, dict):
        configured = config_or_path.get("operator_pilot", {}).get(
            "operator_pool_file", "templates/legal_operators_v0.jsonl"
        )
        path = resolve_project_path(config_or_path, configured)
    else:
        path = Path(config_or_path)
    operators = [LegalOperator.model_validate(row) for row in read_jsonl(path)]
    if not operators:
        raise ValueError("Legal operator pool is empty.")
    ids = [item.operator_id for item in operators]
    if len(ids) != len(set(ids)):
        raise ValueError("Legal operator pool contains duplicate operator IDs.")
    return operators


def operator_pool_hash(operators: list[LegalOperator]) -> str:
    return sha256_text(
        canonical_json([item.model_dump(mode="json") for item in operators])
    )


def retrieve_operator_for_issue(
    issue: LegalIssueNode,
    operators: list[LegalOperator],
    *,
    similarity_backend: SimilarityBackend | None = None,
) -> dict[str, Any]:
    candidates = [item for item in operators if issue.issue_type in item.issue_types]
    filtered = bool(candidates)
    candidates = candidates or operators
    query = "\n".join(
        [
            f"Issue type: {issue.issue_type}",
            f"Question: {issue.issue_question}",
            "Positions: "
            + " ".join(item.position for item in issue.party_positions),
        ]
    )
    documents = [_operator_document(item) for item in candidates]
    backend = similarity_backend or TfidfSimilarityBackend()
    scores = backend.similarities(query, documents)
    winner_index = max(range(len(candidates)), key=lambda index: scores[index])
    mode_prefix = "embedding" if similarity_backend is not None else "tfidf"
    return {
        "operator": candidates[winner_index],
        "similarity": float(scores[winner_index]),
        "retrieval_mode": (
            f"{mode_prefix}_type_filtered" if filtered else f"{mode_prefix}_full_pool"
        ),
        "candidate_operator_ids": [item.operator_id for item in candidates],
    }


def _operator_document(operator: LegalOperator) -> str:
    return " ".join(
        [
            operator.operator_name,
            " ".join(operator.issue_types),
            " ".join(operator.knowledge_tags),
            operator.description,
            operator.application_scenario,
            " ".join(operator.reasoning_flow),
        ]
    )


def _normalize_issue_graph_payload(
    payload: dict[str, Any] | None,
    *,
    case: NormalizedCase,
    authorities: dict[str, str],
    max_issues: int,
    max_depth: int = 2,
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        raise ValueError("Issue-graph generation did not return a JSON object.")
    actions: list[str] = []
    raw_issues = payload.get("issues")
    if not isinstance(raw_issues, list) or not raw_issues:
        raise ValueError("Issue graph must contain at least one issue.")
    if len(raw_issues) > max_issues:
        actions.append("issue_graph_truncated_to_max_issues")
    raw_issues = [item for item in raw_issues[:max_issues] if isinstance(item, dict)]
    if not raw_issues:
        raise ValueError("Issue graph contains no valid issue objects.")

    old_to_new: dict[str, str] = {}
    for index, item in enumerate(raw_issues, start=1):
        old_to_new.setdefault(str(item.get("issue_id") or f"I{index}"), f"I{index}")
    normalized: list[dict[str, Any]] = []
    depth_by_id: dict[str, int] = {}
    for index, item in enumerate(raw_issues, start=1):
        issue_id = f"I{index}"
        if item.get("issue_id") != issue_id:
            actions.append("issue_ids_renumbered")
        raw_parent = str(item.get("parent_id") or "ROOT")
        parent_id = old_to_new.get(raw_parent, raw_parent)
        existing_ids = {row["issue_id"] for row in normalized}
        if issue_id == "I1":
            if parent_id != "ROOT":
                actions.append("I1_parent_forced_to_root")
            parent_id = "ROOT"
        elif parent_id == "ROOT":
            parent_id = "I1"
            actions.append(f"{issue_id}_flat_parent_attached_to_I1")
        elif parent_id not in existing_ids:
            parent_id = "I1"
            actions.append(f"{issue_id}_invalid_parent_replaced")
        parent_depth = 0 if parent_id == "ROOT" else depth_by_id[parent_id]
        if parent_depth + 1 > max_depth:
            parent_id = "I1"
            parent_depth = depth_by_id["I1"]
            actions.append(f"{issue_id}_depth_capped")

        issue_type = str(item.get("issue_type") or "claim_elements")
        if issue_type not in ISSUE_TYPES:
            issue_type = "claim_elements"
            actions.append(f"{issue_id}_issue_type_defaulted")
        question = str(item.get("issue_question") or "").strip()
        if not question:
            question = "Whether the supplied plaintiff's claim is legally established."
            actions.append(f"{issue_id}_question_defaulted")
        positions = []
        for position in item.get("party_positions") or []:
            if not isinstance(position, dict):
                continue
            party = str(position.get("party") or "").strip()
            text = str(position.get("position") or "").strip()
            if party and text:
                positions.append({"party": party, "position": text})
            if len(positions) == 2:
                break
        fact_ids = _valid_unique_ids(item.get("fact_ids"), set(case.facts))
        authority_ids = _valid_unique_ids(
            item.get("authority_ids"), set(authorities)
        )
        if len(fact_ids) != len(_string_list(item.get("fact_ids"))):
            actions.append(f"{issue_id}_invalid_fact_ids_removed")
        if len(authority_ids) != len(_string_list(item.get("authority_ids"))):
            actions.append(f"{issue_id}_invalid_authority_ids_removed")
        normalized.append(
            {
                "issue_id": issue_id,
                "parent_id": parent_id,
                "issue_type": issue_type,
                "issue_question": question,
                "party_positions": positions,
                "fact_ids": fact_ids,
                "authority_ids": authority_ids,
            }
        )
        depth_by_id[issue_id] = parent_depth + 1
    root_claim = str(payload.get("root_claim") or "").strip()
    if root_claim != case.claim:
        actions.append("root_claim_forced_to_supplied_claim")
    return {
        "graph_analysis": str(payload.get("graph_analysis") or "").strip(),
        "root_claim": case.claim,
        "issues": normalized,
    }, list(dict.fromkeys(actions))


def _issue_execution_order(issues: list[LegalIssueNode]) -> list[LegalIssueNode]:
    by_id = {issue.issue_id: issue for issue in issues}
    depths: dict[str, int] = {}

    def depth(issue_id: str, visiting: set[str]) -> int:
        if issue_id in depths:
            return depths[issue_id]
        if issue_id in visiting:
            raise ValueError("Issue graph contains a cycle.")
        issue = by_id[issue_id]
        if issue.parent_id == "ROOT" or issue.parent_id not in by_id:
            value = 1
        else:
            value = 1 + depth(issue.parent_id, visiting | {issue_id})
        depths[issue_id] = value
        return value

    positions = {issue.issue_id: index for index, issue in enumerate(issues)}
    return sorted(
        issues,
        key=lambda issue: (-depth(issue.issue_id, set()), positions[issue.issue_id]),
    )


def _root_issue(issues: list[LegalIssueNode]) -> LegalIssueNode:
    roots = [issue for issue in issues if issue.parent_id == "ROOT"]
    if len(roots) != 1:
        raise ValueError(
            f"Issue graph must contain exactly one top-level issue; found {len(roots)}."
        )
    return roots[0]


def _normalize_issue_finding_payload(
    payload: dict[str, Any] | None,
    *,
    issue: LegalIssueNode,
    case: NormalizedCase,
    authorities: dict[str, str],
    child_issue_ids: set[str],
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        raise ValueError(f"Issue {issue.issue_id} did not return a JSON object.")
    result = dict(payload)
    actions: list[str] = []
    if result.get("issue_id") != issue.issue_id:
        result["issue_id"] = issue.issue_id
        actions.append(f"{issue.issue_id}_finding_id_forced")
    for key in ("analysis_for_claim", "analysis_against_claim", "resolution"):
        value = result.get(key)
        if not isinstance(value, str):
            result[key] = "" if value is None else str(value)
            actions.append(f"{issue.issue_id}_{key}_normalized")
    if result.get("conclusion") not in {
        "supports_claim",
        "opposes_claim",
        "mixed",
        "unresolved",
    }:
        result["conclusion"] = "unresolved"
        actions.append(f"{issue.issue_id}_conclusion_defaulted")
    allowed = {
        "supporting_fact_ids": set(issue.fact_ids),
        "opposing_fact_ids": set(issue.fact_ids),
        "cited_authority_ids": set(issue.authority_ids),
        "relied_on_child_issue_ids": child_issue_ids,
    }
    for key, valid in allowed.items():
        original = _string_list(result.get(key))
        result[key] = _valid_unique_ids(original, valid)
        if len(original) != len(result[key]):
            actions.append(f"{issue.issue_id}_{key}_filtered")
    if child_issue_ids:
        required_children = sorted(
            child_issue_ids,
            key=lambda value: int(value[1:]),
        )
        if result["relied_on_child_issue_ids"] != required_children:
            result["relied_on_child_issue_ids"] = required_children
            actions.append(f"{issue.issue_id}_child_dependencies_completed")
    accepted = {
        "issue_id",
        "analysis_for_claim",
        "analysis_against_claim",
        "resolution",
        "conclusion",
        *allowed,
    }
    for key in list(result):
        if key not in accepted:
            result.pop(key)
            actions.append(f"{issue.issue_id}_extra_fields_removed")
    return result, list(dict.fromkeys(actions))


def _normalize_final_decision_payload(
    payload: dict[str, Any] | None,
    *,
    issue_ids: set[str],
    root_finding: LegalIssueFinding | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(payload, dict):
        raise ValueError("Final synthesis did not return a JSON object.")
    result = dict(payload)
    actions: list[str] = []
    ids = _valid_unique_ids(result.get("dispositive_issue_ids"), issue_ids)
    if not ids:
        ids = sorted(issue_ids, key=lambda value: int(value[1:]))[:1]
        actions.append("missing_dispositive_issue_id_filled")
    result["dispositive_issue_ids"] = ids
    rationale = result.get("final_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        result["final_rationale"] = "The disposition follows from the resolved issues."
        actions.append("missing_final_rationale_filled")
    if result.get("final_decision") not in {"support", "reject"}:
        raise ValueError("Final synthesis did not return support or reject.")
    if root_finding is not None:
        expected = {
            "supports_claim": "support",
            "opposes_claim": "reject",
        }.get(root_finding.conclusion)
        if expected is not None and result["final_decision"] != expected:
            result["final_decision"] = expected
            result["final_rationale"] = root_finding.resolution
            actions.append("final_decision_aligned_to_root_finding")
            actions.append("final_rationale_aligned_to_root_finding")
        if root_finding.issue_id not in result["dispositive_issue_ids"]:
            result["dispositive_issue_ids"] = [root_finding.issue_id]
            actions.append("dispositive_issue_aligned_to_root_finding")
    return {
        "final_rationale": result["final_rationale"],
        "dispositive_issue_ids": result["dispositive_issue_ids"],
        "final_decision": result["final_decision"],
    }, actions


def _case_prompt_payload(
    case: NormalizedCase,
    authorities: dict[str, str],
) -> dict[str, Any]:
    return {
        "claim": case.claim,
        "parties": list(case.parties),
        "facts": dict(case.facts),
        "authorities": dict(authorities),
    }


def _render_operator_prompt(
    config: dict[str, Any],
    name: str,
    case: NormalizedCase,
    authorities: dict[str, str],
    **extra: Any,
) -> tuple[str, str]:
    payload = _case_prompt_payload(case, authorities)
    values: dict[str, Any] = {
        "claim": payload["claim"],
        "parties": "\n".join(payload["parties"]) or "Not separately specified.",
        "facts": "\n".join(
            f"{key}: {value}" for key, value in payload["facts"].items()
        ),
        "numbered_authorities": "\n".join(
            f"{key}: {value}" for key, value in payload["authorities"].items()
        )
        or "No authorities supplied.",
    }
    values.update(extra)
    rendered = {
        key: (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, indent=2)
        )
        for key, value in values.items()
    }
    prompt = load_prompt(config, name).format(**rendered)
    return prompt, sha256_text(prompt)


def _number_authorities(case: NormalizedCase) -> dict[str, str]:
    fragments: list[str] = []
    for value in (case.authorities, case.metadata.get("relevant_cases")):
        text = str(value or "").strip()
        if not text:
            continue
        parts = [
            part.strip(" \t-*;")
            for part in re.split(r"[\r\n]+", text)
            if part.strip(" \t-*;")
        ]
        fragments.extend(parts or [text])
    unique = list(dict.fromkeys(fragments))
    return {f"A{index}": value for index, value in enumerate(unique, start=1)}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _valid_unique_ids(value: Any, valid: set[str]) -> list[str]:
    return list(dict.fromkeys(item for item in _string_list(value) if item in valid))


def _build_operator_similarity_backend(config: dict[str, Any]) -> SimilarityBackend:
    settings = config.get("operator_pilot", {})
    backend = str(settings.get("retrieval_backend", "ollama_embedding"))
    if backend == "tfidf":
        return TfidfSimilarityBackend()
    if backend != "ollama_embedding":
        raise ValueError(f"Unknown operator retrieval backend: {backend}")
    cache_path = resolve_project_path(
        config,
        settings.get(
            "embedding_cache_file",
            "data/processed/legal_flux/operator_embeddings_bge_m3.json",
        ),
    )
    return OllamaEmbeddingBackend(
        base_url=settings.get("embedding_base_url", config["model"]["base_url"]),
        model=settings.get("embedding_model", "bge-m3:latest"),
        cache_path=cache_path,
        timeout_seconds=int(settings.get("embedding_timeout_seconds", 600)),
    )


def operator_pilot_run_hash(
    case: NormalizedCase,
    *,
    phase: str,
    workflow_hash: str,
    seed: int,
) -> str:
    return make_run_hash(
        dataset=case.dataset,
        case_id=case.case_id,
        variant_id=case.variant_id,
        condition="operator_issue_graph",
        phase=phase,
        workflow_hash=workflow_hash,
        seed=seed,
    )


def operator_pilot_workflow_hash(
    config: dict[str, Any], operators: list[LegalOperator]
) -> str:
    project_root = Path(config["_project_root"])
    prompt_root = resolve_path(config, "prompts_dir")
    schema_root = resolve_path(config, "schemas_dir")
    files = [
        prompt_root / "legal_operator" / "issue_graph.txt",
        prompt_root / "legal_operator" / "issue_execute.txt",
        prompt_root / "legal_operator" / "final_synthesis.txt",
        schema_root / "legal_operator_issue_graph.json",
        schema_root / "legal_operator_issue_finding.json",
        schema_root / "legal_operator_final_decision.json",
        project_root / "src" / "legal_pilot" / "legal_operator_pilot.py",
    ]
    return sha256_text(
        canonical_json(
            {
                "operator_pilot": config.get("operator_pilot", {}),
                "model": {
                    key: config["model"].get(key)
                    for key in (
                        "provider",
                        "name",
                        "context_length",
                        "temperature",
                        "seed",
                    )
                },
                "operator_pool_hash": operator_pool_hash(operators),
                "files": {
                    str(path.relative_to(project_root)): sha256_text(
                        path.read_text(encoding="utf-8")
                    )
                    for path in files
                },
            }
        )
    )


def score_legal_operator_pilot(
    config: dict[str, Any], *, run_dir: Path
) -> dict[str, Any]:
    rows = latest_by_run_hash(read_jsonl(run_dir / "generations.jsonl"))
    plan_path = run_dir / "run_plan.json"
    if plan_path.exists():
        allowed = {
            item["run_hash"]
            for item in json.loads(plan_path.read_text(encoding="utf-8")).get(
                "jobs", []
            )
        }
        rows = [row for row in rows if row.get("run_hash") in allowed]
    cases = {
        (case.dataset, case.case_id, case.variant_id): case
        for case in load_cases(config)
    }
    scored: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        if row.get("status") == "ok":
            case = cases[(row["dataset"], row["case_id"], row["variant_id"])]
            try:
                analysis = FinalAnalysis.model_validate(row["parsed_json"])
                value.update(score_record(case, analysis))
                value["prediction"] = analysis.final_decision
                value["trajectory_length"] = len(row.get("issue_findings") or [])
                value["review_count"] = 0
            except Exception as exc:
                value["status"] = "score_error"
                value["score_error"] = str(exc)
        scored.append(value)
    write_jsonl(run_dir / "scored.jsonl", scored)
    ok = [row for row in scored if row.get("status") == "ok"]
    frame = pd.DataFrame(ok)
    aggregate = _aggregate_frame(frame)
    if not aggregate.empty:
        trajectory = (
            frame.groupby(["dataset", "condition"], dropna=False)[
                ["trajectory_length", "review_count"]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        aggregate = aggregate.merge(
            trajectory, on=["dataset", "condition"], how="left"
        )
    aggregate.to_csv(run_dir / "aggregate.csv", index=False)
    summary = {
        "run_dir": str(run_dir),
        "records": len(scored),
        "ok_records": len(ok),
        "error_records": len(scored) - len(ok),
        "aggregate_path": str(run_dir / "aggregate.csv"),
        "scored_path": str(run_dir / "scored.jsonl"),
    }
    (run_dir / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
