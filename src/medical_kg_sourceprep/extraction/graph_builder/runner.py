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
    ENTITY_DISCOVERY_EXAMPLES,
    ENTITY_DISCOVERY_PROMPT_TEMPLATE,
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
    CandidateNormalization,
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

    async def extract_with_retry(
        *, graph_schema: Any, prompt_template: str, examples: str, diagnostics: list[dict[str, Any]],
    ) -> tuple[Neo4jGraph | None, LLMGenerationError | None, int]:
        """每阶段总共尝试两次；失败响应只保存安全形状诊断。"""
        last_error: LLMGenerationError | None = None
        for attempt in range(1, 3):
            try:
                return await _extract_graph(
                    client, chunk=chunk, graph_schema=graph_schema, prompt_template=prompt_template,
                    examples=examples, input_text=input_text, response_diagnostics=diagnostics,
                ), None, attempt
            except LLMGenerationError as error:
                last_error = error
        return None, last_error, 2

    judge_drafts: list[dict[str, Any]] = []
    entity_response_diagnostics: list[dict[str, Any]] = []
    entity_graph, entity_error, entity_attempts = await extract_with_retry(
        graph_schema=build_graphrag_schema(
            schema,
            relation_types=(),
            node_types=sorted(BUSINESS_NODE_TYPES),
            node_property_names=("mention", "extraction_reason"),
        ),
        prompt_template=ENTITY_DISCOVERY_PROMPT_TEMPLATE,
        examples=ENTITY_DISCOVERY_EXAMPLES,
        diagnostics=entity_response_diagnostics,
    )
    if entity_error is not None:
        entity_nodes = []
        entity_holds = [_model_phase_failure_hold(
            stage="entity", phase="entity_phase", error=entity_error,
            response_diagnostics=entity_response_diagnostics, attempts=entity_attempts,
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
            holds=entity_holds, judge_drafts=judge_drafts,
        )
        return candidate_summary(
            chunk=chunk, nodes=[], relationships=[], holds=entity_holds, output_dir=output_dir, judge_drafts=judge_drafts,
        )
    entity_result = normalize_candidate_nodes(
        entity_graph or Neo4jGraph(),
        chunk=chunk,
        schema=schema,
        allowed_node_types=BUSINESS_NODE_TYPES,
        derive_entity_provenance=True,
    )
    entity_nodes, entity_holds = entity_result
    judge_drafts.extend(entity_result.judge_drafts)
    rule_response_diagnostics: list[dict[str, Any]] = []
    rule_graph, rule_error, rule_attempts = await extract_with_retry(
        graph_schema=build_graphrag_schema(schema, relation_types=(), node_types=("RuleDefinition",)),
        prompt_template=RULE_NODE_PROMPT_TEMPLATE, examples=_catalog_for_prompt(entity_nodes), diagnostics=rule_response_diagnostics,
    )
    if rule_error is not None:
        rule_nodes = []
        rule_holds = [_model_phase_failure_hold(
            stage="rule", phase="rule_phase", error=rule_error, response_diagnostics=rule_response_diagnostics,
            attempts=rule_attempts,
        )]
    else:
        rule_result = normalize_candidate_nodes(
            rule_graph or Neo4jGraph(), chunk=chunk, schema=schema, allowed_node_types=("RuleDefinition",)
        )
        rule_nodes, rule_holds = rule_result
        judge_drafts.extend(rule_result.judge_drafts)
    nodes = [*entity_nodes, *rule_nodes]
    frozen_nodes = [item for item in nodes if item.get("extraction_status") == "VALID"]
    ordinary_response_diagnostics: list[dict[str, Any]] = []
    ordinary_relation_graph, ordinary_error, ordinary_attempts = await extract_with_retry(
        graph_schema=build_graphrag_schema(
            schema, relation_types=sorted(ORDINARY_RELATION_TYPES), node_types=sorted(BUSINESS_NODE_TYPES),
        ),
        prompt_template=ORDINARY_RELATION_PROMPT_TEMPLATE, examples=_catalog_for_prompt(entity_nodes),
        diagnostics=ordinary_response_diagnostics,
    )
    if ordinary_error is not None:
        ordinary_result = normalize_candidate_relationships(
            Neo4jGraph(),
            chunk=chunk,
            schema=schema,
            nodes=frozen_nodes,
            allowed_relation_types=sorted(ORDINARY_RELATION_TYPES),
            include_deterministic_state=True,
            validate_rule_structures=False,
        )
        ordinary_result.review_items.append(_model_phase_failure_hold(
            stage="relation", phase="ordinary_relation_phase", error=ordinary_error,
            response_diagnostics=ordinary_response_diagnostics, attempts=ordinary_attempts,
        ))
    else:
        ordinary_result = normalize_candidate_relationships(
            ordinary_relation_graph or Neo4jGraph(), chunk=chunk, schema=schema, nodes=frozen_nodes,
            allowed_relation_types=sorted(ORDINARY_RELATION_TYPES), include_deterministic_state=True,
            validate_rule_structures=False,
        )
    ordinary_relationships, ordinary_holds = ordinary_result
    judge_drafts.extend(ordinary_result.judge_drafts)
    rule_edge_response_diagnostics: list[dict[str, Any]] = []
    if any(item.get("entity_type") == "RuleDefinition" for item in frozen_nodes):
        rule_edge_graph, rule_edge_error, rule_edge_attempts = await extract_with_retry(
            graph_schema=build_graphrag_schema(schema, relation_types=sorted(RULE_EDGE_TYPES)),
            prompt_template=RULE_EDGE_PROMPT_TEMPLATE, examples=_catalog_for_prompt(frozen_nodes), diagnostics=rule_edge_response_diagnostics,
        )
        if rule_edge_error is not None:
            rule_result = normalize_candidate_relationships(
                Neo4jGraph(),
                chunk=chunk,
                schema=schema,
                nodes=frozen_nodes,
                allowed_relation_types=sorted(RULE_EDGE_TYPES),
                include_deterministic_state=False,
                validate_rule_structures=True,
                return_invalid_rule_keys=True,
            )
            rule_result.review_items.append(_model_phase_failure_hold(
                stage="rule", phase="rule_edge_phase", error=rule_edge_error,
                response_diagnostics=rule_edge_response_diagnostics, attempts=rule_edge_attempts,
            ))
        else:
            rule_result = normalize_candidate_relationships(
                rule_edge_graph or Neo4jGraph(), chunk=chunk, schema=schema, nodes=frozen_nodes,
                allowed_relation_types=sorted(RULE_EDGE_TYPES), include_deterministic_state=False,
                validate_rule_structures=True, return_invalid_rule_keys=True,
            )
        rule_relationships, rule_edge_holds = rule_result
        invalid_rule_keys = rule_result.invalid_rule_keys
        judge_drafts.extend(rule_result.judge_drafts)
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
        holds=holds, judge_drafts=judge_drafts,
    )
    return candidate_summary(
        chunk=chunk, nodes=nodes, relationships=relationships, holds=holds, output_dir=output_dir,
        judge_drafts=judge_drafts,
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
