"""Candidate graph extraction orchestration and command-line entry helpers."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor, OnError
from neo4j_graphrag.experimental.components.types import Neo4jGraph, TextChunk, TextChunks

from ..artifacts import sha256_path
from .artifacts import (
    _candidate_display,
    _public_candidate_nodes,
    candidate_summary,
    write_candidate_artifacts,
    _model_phase_failure_hold,
)
from .client import DeepSeekGraphBuilderClient, create_deepseek_graph_builder
from .contract import (
    BUSINESS_NODE_TYPES,
    DEFAULT_CHUNK_ID,
    DEFAULT_CHUNK_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCHEMA_PATH,
    DEEPSEEK_MODEL,
    NODE_PROMPT_TEMPLATE,
    ORDINARY_RELATION_PROMPT_TEMPLATE,
    RULE_EDGE_PROMPT_TEMPLATE,
    RULE_NODE_PROMPT_TEMPLATE,
    RULE_EDGE_TYPES,
    ORDINARY_RELATION_TYPES,
    SMOKE_TEXT,
    GraphBuilderConfigurationError,
)
from .schema import _extract_graph, build_graphrag_schema, load_candidate_graph_schema
from .validation import (
    _catalog_for_prompt,
    _hold,
    normalize_candidate_nodes,
    normalize_candidate_relationships,
)
from ..llm_extraction import EvidenceChunk, load_chunk_manifest


async def run_candidate_graph(
    client: DeepSeekGraphBuilderClient,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    output_dir: Path,
    source_manifest_sha256: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run entity, rule, ordinary relation, and rule-edge phases for one chunk."""
    input_text = chunk.text
    effective_run_id = run_id or output_dir.name
    entity_response_diagnostics: list[dict[str, Any]] = []
    try:
        entity_graph = await _extract_graph(
            client,
            chunk=chunk,
            graph_schema=build_graphrag_schema(
                schema, relation_types=(), node_types=sorted(BUSINESS_NODE_TYPES)
            ),
            prompt_template=NODE_PROMPT_TEMPLATE,
            examples="{}",
            input_text=input_text,
            response_diagnostics=entity_response_diagnostics,
        )
        entity_nodes, entity_holds = normalize_candidate_nodes(
            entity_graph, chunk=chunk, schema=schema, allowed_node_types=BUSINESS_NODE_TYPES
        )
    except LLMGenerationError as error:
        entity_nodes = []
        entity_holds = [_model_phase_failure_hold(
            stage="entity", phase="entity_phase", error=error,
            response_diagnostics=entity_response_diagnostics,
        )]
        write_candidate_artifacts(
            output_dir,
            schema=schema,
            schema_path=schema_path,
            chunk=chunk,
            run_id=effective_run_id,
            source_manifest_sha256=source_manifest_sha256,
            nodes=[],
            relationships=[],
            holds=entity_holds,
        )
        return candidate_summary(
            chunk=chunk, nodes=[], relationships=[], holds=entity_holds, output_dir=output_dir
        )
    rule_response_diagnostics: list[dict[str, Any]] = []
    try:
        rule_graph = await _extract_graph(
            client,
            chunk=chunk,
            graph_schema=build_graphrag_schema(schema, relation_types=(), node_types=("RuleDefinition",)),
            prompt_template=RULE_NODE_PROMPT_TEMPLATE,
            examples=_catalog_for_prompt(entity_nodes),
            input_text=input_text,
            response_diagnostics=rule_response_diagnostics,
        )
        rule_nodes, rule_holds = normalize_candidate_nodes(
            rule_graph, chunk=chunk, schema=schema, allowed_node_types=("RuleDefinition",)
        )
    except LLMGenerationError as error:
        rule_nodes = []
        rule_holds = [_model_phase_failure_hold(
            stage="rule", phase="rule_phase", error=error, response_diagnostics=rule_response_diagnostics
        )]
    nodes = [*entity_nodes, *rule_nodes]
    ordinary_response_diagnostics: list[dict[str, Any]] = []
    try:
        ordinary_relation_graph = await _extract_graph(
            client,
            chunk=chunk,
            graph_schema=build_graphrag_schema(
                schema, relation_types=sorted(ORDINARY_RELATION_TYPES), node_types=sorted(BUSINESS_NODE_TYPES)
            ),
            prompt_template=ORDINARY_RELATION_PROMPT_TEMPLATE,
            examples=_catalog_for_prompt(entity_nodes),
            input_text=input_text,
            response_diagnostics=ordinary_response_diagnostics,
        )
        ordinary_relationships, ordinary_holds = normalize_candidate_relationships(
            ordinary_relation_graph,
            chunk=chunk,
            schema=schema,
            nodes=nodes,
            allowed_relation_types=sorted(ORDINARY_RELATION_TYPES),
            include_deterministic_state=True,
            validate_rule_structures=False,
        )
    except LLMGenerationError as error:
        ordinary_relationships, ordinary_holds = normalize_candidate_relationships(
            Neo4jGraph(),
            chunk=chunk,
            schema=schema,
            nodes=nodes,
            allowed_relation_types=sorted(ORDINARY_RELATION_TYPES),
            include_deterministic_state=True,
            validate_rule_structures=False,
        )
        ordinary_holds.append(_model_phase_failure_hold(
            stage="relation", phase="ordinary_relation_phase", error=error,
            response_diagnostics=ordinary_response_diagnostics,
        ))
    rule_edge_response_diagnostics: list[dict[str, Any]] = []
    if rule_nodes:
        try:
            rule_edge_graph = await _extract_graph(
                client,
                chunk=chunk,
                graph_schema=build_graphrag_schema(schema, relation_types=sorted(RULE_EDGE_TYPES)),
                prompt_template=RULE_EDGE_PROMPT_TEMPLATE,
                examples=_catalog_for_prompt(nodes),
                input_text=input_text,
                response_diagnostics=rule_edge_response_diagnostics,
            )
            rule_relationships, rule_edge_holds, invalid_rule_keys = normalize_candidate_relationships(
                rule_edge_graph,
                chunk=chunk,
                schema=schema,
                nodes=nodes,
                allowed_relation_types=sorted(RULE_EDGE_TYPES),
                include_deterministic_state=False,
                validate_rule_structures=True,
                return_invalid_rule_keys=True,
            )
        except LLMGenerationError as error:
            rule_relationships, rule_edge_holds, invalid_rule_keys = normalize_candidate_relationships(
                Neo4jGraph(),
                chunk=chunk,
                schema=schema,
                nodes=nodes,
                allowed_relation_types=sorted(RULE_EDGE_TYPES),
                include_deterministic_state=False,
                validate_rule_structures=True,
                return_invalid_rule_keys=True,
            )
            rule_edge_holds.append(_model_phase_failure_hold(
                stage="rule", phase="rule_edge_phase", error=error,
                response_diagnostics=rule_edge_response_diagnostics,
            ))
    else:
        rule_relationships, rule_edge_holds, invalid_rule_keys = [], [], set()
    relationships = [*ordinary_relationships, *rule_relationships]
    nodes = _public_candidate_nodes(
        [node for node in nodes if node["candidate_key"] not in invalid_rule_keys]
    )
    holds = [*entity_holds, *rule_holds, *ordinary_holds, *rule_edge_holds]
    write_candidate_artifacts(
        output_dir,
        schema=schema,
        schema_path=schema_path,
        chunk=chunk,
        run_id=effective_run_id,
        source_manifest_sha256=source_manifest_sha256,
        nodes=nodes,
        relationships=relationships,
        holds=holds,
    )
    return candidate_summary(
        chunk=chunk, nodes=nodes, relationships=relationships, holds=holds, output_dir=output_dir
    )


async def run_candidate_block(
    client: DeepSeekGraphBuilderClient,
    *,
    chunk_id: str = DEFAULT_CHUNK_ID,
    manifest_path: Path = DEFAULT_CHUNK_MANIFEST,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Load one canonical block into a fresh candidate-only run directory."""
    schema = load_candidate_graph_schema(schema_path)
    _manifest, chunks = load_chunk_manifest(manifest_path)
    selected = next((item for item in chunks if item.chunk_id == chunk_id), None)
    if selected is None:
        raise GraphBuilderConfigurationError(f"chunk_id is not in the canonical manifest: {chunk_id}")
    effective_run_id = run_id or _default_run_id(chunk_id)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", effective_run_id):
        raise GraphBuilderConfigurationError("run_id contains unsupported characters")
    run_output_dir = output_dir / effective_run_id
    if (run_output_dir / "run-manifest.json").exists():
        raise GraphBuilderConfigurationError(f"run_id already exists: {effective_run_id}")
    return await run_candidate_graph(
        client,
        chunk=selected,
        schema=schema,
        schema_path=schema_path,
        output_dir=run_output_dir,
        source_manifest_sha256=sha256_path(manifest_path),
        run_id=effective_run_id,
    )


def _default_run_id(chunk_id: str) -> str:
    chunk_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", chunk_id).strip("-")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{chunk_slug}-{timestamp}"


async def run_smoke(client: DeepSeekGraphBuilderClient) -> dict[str, int | str]:
    """Run one fixed non-patient sentence through Graph Builder in memory."""
    extractor = LLMEntityRelationExtractor(
        llm=client.llm,
        create_lexical_graph=False,
        on_error=OnError.RAISE,
        max_concurrency=1,
        use_structured_output=False,
    )
    graph = await extractor.run(
        chunks=TextChunks(chunks=[TextChunk(text=SMOKE_TEXT, index=0)])
    )
    return {
        "model": DEEPSEEK_MODEL,
        "node_count": len(graph.nodes),
        "relationship_count": len(graph.relationships),
    }


async def _run_smoke_main() -> dict[str, int | str]:
    client = create_deepseek_graph_builder()
    try:
        return await run_smoke(client)
    finally:
        await client.aclose()


async def _run_candidate_main(args: argparse.Namespace) -> dict[str, Any]:
    client = create_deepseek_graph_builder()
    try:
        return await run_candidate_block(
            client,
            chunk_id=args.chunk_id,
            manifest_path=args.manifest,
            schema_path=args.schema,
            output_dir=args.output,
            run_id=args.run_id,
        )
    finally:
        await client.aclose()
