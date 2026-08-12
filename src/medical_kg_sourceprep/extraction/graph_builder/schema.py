"""Schema loading and GraphRAG extraction adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.experimental.components.schema import GraphSchema
from neo4j_graphrag.experimental.components.types import Neo4jGraph, TextChunk, TextChunks

from .client import DeepSeekGraphBuilderClient, _GraphRagIdCompletingLLM
from .contract import (
    DEFAULT_SCHEMA_PATH,
    TRIAL_NODE_TYPES,
    TRIAL_RELATION_TYPES,
    GraphBuilderConfigurationError,
)
from ..llm_extraction import EvidenceChunk


def load_candidate_graph_schema(path: Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    """Load the candidate schema and reject an incomplete local contract."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GraphBuilderConfigurationError(f"candidate graph schema is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise GraphBuilderConfigurationError("candidate graph schema must be an object")
    node_types = value.get("node_types")
    relation_types = value.get("relationship_types")
    node_names = {
        item.get("name")
        for item in node_types
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    } if isinstance(node_types, list) else set()
    relation_names = {
        item.get("type")
        for item in relation_types
        if isinstance(item, Mapping) and isinstance(item.get("type"), str)
    } if isinstance(relation_types, list) else set()
    if value.get("schema_id") != "medical-report-candidate-graph":
        raise GraphBuilderConfigurationError("candidate graph schema_id is unsupported")
    if not TRIAL_NODE_TYPES <= node_names or not TRIAL_RELATION_TYPES <= relation_names:
        raise GraphBuilderConfigurationError("candidate graph schema lacks required trial types")
    return value


def _as_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    return ()


def _relation_endpoint_pairs(schema: Mapping[str, Any], relation_type: str) -> tuple[tuple[str, str], ...]:
    for item in schema.get("relationship_types", []):
        if not isinstance(item, Mapping) or item.get("type") != relation_type:
            continue
        pairs = []
        for endpoint in item.get("allowed_endpoints", []):
            if not isinstance(endpoint, Mapping):
                continue
            for source in _as_names(endpoint.get("source")):
                for target in _as_names(endpoint.get("target")):
                    pairs.append((source, target))
        return tuple(pairs)
    return ()


def build_graphrag_schema(
    schema: Mapping[str, Any], *, relation_types: Sequence[str], node_types: Sequence[str] = TRIAL_NODE_TYPES
) -> GraphSchema:
    """Convert the JSON contract into the GraphRAG schema supplied to the model."""
    node_definitions = []
    for item in schema["node_types"]:
        if not isinstance(item, Mapping) or item.get("name") not in node_types:
            continue
        node_definitions.append(
            {
                "label": item["name"],
                "description": item.get("description", ""),
                "properties": [
                    {"name": "mention", "type": "STRING"},
                    {"name": "canonical_name_candidate", "type": "STRING"},
                    {"name": "exact_quote", "type": "STRING"},
                    {"name": "exact_quote_occurrence_index", "type": "INTEGER"},
                    {"name": "mention_occurrence_index", "type": "INTEGER"},
                    {"name": "source_char_start", "type": "INTEGER"},
                    {"name": "source_char_end", "type": "INTEGER"},
                    {"name": "bound_indicator_mention", "type": "STRING"},
                    {"name": "rule_stage_candidate", "type": "STRING"},
                    {"name": "rule_expression", "type": "STRING"},
                    {"name": "rule_name", "type": "STRING"},
                    {"name": "rule_evidence_json", "type": "STRING"},
                ],
                "additional_properties": False,
            }
        )
    relationship_definitions = [
        {
            "label": relation_type,
            "properties": [
                {"name": "exact_quote", "type": "STRING"},
                {"name": "exact_quote_occurrence_index", "type": "INTEGER"},
                {"name": "source_char_start", "type": "INTEGER"},
                {"name": "source_char_end", "type": "INTEGER"},
                {"name": "relation_cue", "type": "STRING"},
                {"name": "rule_evidence_role", "type": "STRING"},
            ],
            "additional_properties": False,
        }
        for relation_type in relation_types
    ]
    patterns = [
        (source, relation_type, target)
        for relation_type in relation_types
        for source, target in _relation_endpoint_pairs(schema, relation_type)
        if source in TRIAL_NODE_TYPES and target in TRIAL_NODE_TYPES
    ]
    return GraphSchema(
        node_types=node_definitions,
        relationship_types=relationship_definitions,
        patterns=patterns,
        additional_node_types=False,
        additional_relationship_types=False,
        additional_patterns=False,
    )


async def _extract_graph(
    client: DeepSeekGraphBuilderClient,
    *,
    chunk: EvidenceChunk,
    graph_schema: GraphSchema,
    prompt_template: str,
    examples: str,
    input_text: str,
    response_diagnostics: list[dict[str, Any]] | None = None,
) -> Neo4jGraph:
    llm = _GraphRagIdCompletingLLM(client.llm)
    extractor = LLMEntityRelationExtractor(
        llm=llm,
        prompt_template=prompt_template,
        create_lexical_graph=False,
        on_error=OnError.RAISE,
        max_concurrency=1,
        use_structured_output=False,
    )
    text_chunk = TextChunk(text=input_text, index=0, uid=chunk.chunk_id)
    try:
        return await extractor.run(
            chunks=TextChunks(chunks=[text_chunk]), schema=graph_schema, examples=examples
        )
    finally:
        if response_diagnostics is not None and llm.last_response_diagnostic:
            response_diagnostics.append(llm.last_response_diagnostic)
