"""Candidate-only artifact serialization and public run summaries."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from neo4j_graphrag.exceptions import LLMGenerationError

from ..artifacts import sha256_path
from .contract import (
    CANDIDATE_RUN_VERSION,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
)
from ..llm_extraction import EvidenceChunk, atomic_write_json
from .validation import _hold


def write_candidate_artifacts(
    output_dir: Path,
    *,
    schema: Mapping[str, Any],
    schema_path: Path,
    chunk: EvidenceChunk,
    run_id: str,
    source_manifest_sha256: str,
    nodes: Sequence[Mapping[str, Any]],
    relationships: Sequence[Mapping[str, Any]],
    holds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write candidate-only artifacts and review outcomes, never model response text."""
    base = {
        "schema_id": schema["schema_id"],
        "schema_version": schema["schema_version"],
        "status": "candidate-only",
        "publication_status": "HOLD",
        "approved": 0,
        "run_id": run_id,
        "source": {"chunk_id": chunk.chunk_id, "chunk_sha256": chunk.chunk_sha256},
    }
    node_doc = {**base, "nodes": list(nodes)}
    relation_doc = {**base, "relationships": list(relationships)}
    graph_doc = {**base, "nodes": list(nodes), "relationships": list(relationships)}
    review_doc = {
        "schema_version": "candidate-graph-review-queue/v0.2",
        "status": "candidate-only",
        "publication_status": "HOLD",
        "approved": 0,
        "items": list(holds),
        "counts": {
            "review_required": sum(item.get("status") == "REVIEW_REQUIRED" for item in holds),
            "rejected": sum(item.get("status") == "REJECTED" for item in holds),
        },
    }
    atomic_write_json(output_dir / "candidate-nodes.json", node_doc)
    atomic_write_json(output_dir / "candidate-relations.json", relation_doc)
    atomic_write_json(output_dir / "graph.json", graph_doc)
    atomic_write_json(output_dir / "review-queue.json", review_doc)
    artifact_names = ("candidate-nodes.json", "candidate-relations.json", "graph.json", "review-queue.json")
    manifest = {
        "schema_version": CANDIDATE_RUN_VERSION,
        "run_id": run_id,
        "status": "candidate-only",
        "approved": 0,
        "provider": "deepseek",
        "model": DEEPSEEK_MODEL,
        "configuration": {
            "base_url": DEEPSEEK_BASE_URL,
            "temperature": 0,
            "response_format": "json_object",
            "thinking": "disabled",
            "trust_env": False,
            "graph_builder": "LLMEntityRelationExtractor",
            "database_write": False,
        },
        "input": {
            "chunk_id": chunk.chunk_id,
            "chunk_sha256": chunk.chunk_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "candidate_schema_sha256": sha256_path(schema_path),
        },
        "counts": {
            "nodes": len(nodes),
            "relationships": len(relationships),
            "review_required": sum(item.get("status") == "REVIEW_REQUIRED" for item in holds),
            "rejected": sum(item.get("status") == "REJECTED" for item in holds),
        },
        "artifacts": {name: sha256_path(output_dir / name) for name in artifact_names},
    }
    atomic_write_json(output_dir / "run-manifest.json", manifest)
    return manifest


def candidate_summary(
    *, chunk: EvidenceChunk, nodes: Sequence[Mapping[str, Any]], relationships: Sequence[Mapping[str, Any]], holds: Sequence[Mapping[str, Any]], output_dir: Path
) -> dict[str, Any]:
    node_by_key = {item["candidate_key"]: item for item in nodes}
    review_count = sum(item.get("status") == "REVIEW_REQUIRED" for item in holds)
    rejected_count = sum(item.get("status") == "REJECTED" for item in holds)
    return {
        "model": DEEPSEEK_MODEL,
        "chunk_id": chunk.chunk_id,
        "node_count": len(nodes),
        "relationship_count": len(relationships),
        "review_count": review_count,
        "rejected_count": rejected_count,
        "hold_count": len(holds),
        "output_dir": str(output_dir),
        "nodes": [
            {
                "candidate_key": item["candidate_key"],
                "entity_type": item["entity_type"],
                **({"rule_expression": item["rule_expression"], "rule_name": item["rule_name"]}
                   if item["entity_type"] == "RuleDefinition"
                   else {"mention": item["mention"]}),
            }
            for item in nodes
        ],
        "relationships": [
            {
                "relation_type": item["relation_type"],
                "source": _candidate_display(node_by_key[item["source_candidate_key"]]),
                "target": _candidate_display(node_by_key[item["target_candidate_key"]]),
                "generation": item["generation"],
            }
            for item in relationships
        ],
    }


def _candidate_display(node: Mapping[str, Any]) -> str:
    return str(node.get("mention") or node.get("rule_expression") or node["candidate_key"])


def _public_candidate_nodes(nodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in node.items() if not key.startswith("_")}
        for node in nodes
    ]


def _model_phase_failure_hold(
    *, stage: str, phase: str, error: LLMGenerationError, response_diagnostics: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Write response shape only; never retain model text, keys, or environment."""
    observed = response_diagnostics[-1] if response_diagnostics else {}
    return _hold(
        stage,
        0,
        f"{phase}_model_response_invalid",
        {
            "parse_phase": phase,
            "reason_code": "llm_generation_error",
            "exception_type": type(error).__name__,
            "response_sha256": observed.get("response_sha256"),
            "json_top_level_fields": observed.get("json_top_level_fields", []),
            "json_top_level_field_types": observed.get("json_top_level_field_types", {}),
            "missing_fields": observed.get("missing_fields", ["nodes", "relationships", "properties"]),
            "response_shape_reason_code": observed.get("reason_code"),
        },
    )
