"""Compatibility entrypoint for provenance-validated graph builder candidates.

Implementation is separated by responsibility: contract, client, schema,
validation, artifact output, and run orchestration. This module intentionally
retains the original import and console-script surface.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import OpenAILLM

from .graph_builder.artifacts import (
    _candidate_display,
    _model_phase_failure_hold,
    _public_candidate_nodes,
    candidate_summary,
    write_candidate_artifacts,
)
from .graph_builder.client import (
    DeepSeekGraphBuilderClient,
    _GraphRagIdCompletingLLM,
    _response_shape_diagnostic,
    load_deepseek_api_key,
)
from .graph_builder.client import create_deepseek_graph_builder as _create_client
from .graph_builder.contract import *  # noqa: F403
from .graph_builder.contract import GraphBuilderConfigurationError
from .graph_builder.runner import run_candidate_block, run_candidate_graph, run_smoke
from .graph_builder.schema import _extract_graph, build_graphrag_schema, load_candidate_graph_schema
from .graph_builder.validation import (
    CandidateNormalization,
    _candidate_key,
    _catalog_for_prompt,
    _hold,
    _relation_key,
    _rule_candidate_key,
    _rule_evidence_ref,
    _source_ref,
    _source_refs_for_mention,
    _table_state_candidate_key,
    deterministic_state_relations,
    normalize_candidate_nodes,
    normalize_candidate_relationships,
)


def create_deepseek_graph_builder(
    *, env: Mapping[str, str] | None = None
) -> DeepSeekGraphBuilderClient:
    """Create the client while preserving patch points used by existing callers."""
    return _create_client(
        env=env,
        http_client_factory=httpx.AsyncClient,
        llm_factory=OpenAILLM,
        api_key_loader=load_deepseek_api_key,
    )


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


def main() -> int:
    """Run the bounded, in-memory DeepSeek Graph Builder smoke test."""
    parser = argparse.ArgumentParser(description="Run DeepSeek Graph Builder smoke test")
    parser.parse_args()
    try:
        summary = asyncio.run(_run_smoke_main())
    except GraphBuilderConfigurationError as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def candidate_block_main() -> int:
    """Run the provenance-validated single-block candidate graph trial."""
    parser = argparse.ArgumentParser(description="Run the single-block candidate graph trial")
    parser.add_argument("--chunk-id", default=DEFAULT_CHUNK_ID)  # noqa: F405
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CHUNK_MANIFEST)  # noqa: F405
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)  # noqa: F405
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)  # noqa: F405
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        summary = asyncio.run(_run_candidate_main(args))
    except GraphBuilderConfigurationError as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
