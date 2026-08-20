"""从第一章原文和冻结 v0.8 规范实体目录抽取规则节点。"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[4]))
    __package__ = "medical_kg_sourceprep.extraction.graph_builder.runner"

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from ...llm_extraction import EvidenceChunk, atomic_write_json, load_chunk_manifest
from ..client import create_deepseek_graph_builder
from ..contract import (
    DEFAULT_CHUNK_MANIFEST,
    DEFAULT_SCHEMA_PATH,
    DEEPSEEK_MODEL,
    EXCLUSION_RULE_PROMPT_TEMPLATE,
    EXCLUSION_RULE_PROMPT_VERSION,
    PROJECT_ROOT,
    RULE_NODE_PROMPT_TEMPLATE,
    RULE_NODE_PROMPT_VERSION,
    GraphBuilderConfigurationError,
)
from ..rule_gate import partition_invalid_rules
from ..schema import load_candidate_graph_schema
from ..validation import build_rule_relationships_from_definitions
from .chapter_candidate_graph import DEFAULT_CANONICAL_PATH, _git_commit, _sha256, load_frozen_catalog
from .rule_semantic_gate_evaluation import _extract_rule_nodes


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runtime/candidates/chapter-01/rules-v0.8"


def _json_int(value: object) -> int:
    """只接纳 checkpoint 中实际可解释为整数的计数。"""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def build_canonical_endpoints_by_chunk(
    *,
    canonical: Mapping[str, Any],
    mentions: Sequence[Mapping[str, Any]],
    entities: Sequence[Mapping[str, Any]],
    chunks_by_id: Mapping[str, EvidenceChunk],
) -> dict[str, list[dict[str, Any]]]:
    """为每个 chunk 建立去重后的 canonical 规则端点目录。"""
    mention_by_id = {str(item["mention_id"]): item for item in mentions}
    entity_by_id = {str(item["canonical_id"]): item for item in entities}
    refs_by_chunk_entity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for link in canonical.get("mention_to_canonical_links", []):
        if not isinstance(link, Mapping):
            continue
        mention = mention_by_id[str(link["mention_id"])]
        canonical_id = str(link["canonical_id"])
        chunk_id = str(mention["chunk_id"])
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise GraphBuilderConfigurationError("chapter_rule_chunk_unknown")
        start, end, quote = mention.get("start"), mention.get("end"), mention.get("exact_quote")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not isinstance(quote, str)
            or chunk.text[start:end] != quote
        ):
            raise GraphBuilderConfigurationError("chapter_rule_mention_replay_failed")
        refs_by_chunk_entity[(chunk_id, canonical_id)].append({
            "chunk_id": chunk_id,
            "chunk_sha256": chunk.chunk_sha256,
            "mention_char_start": start,
            "mention_char_end": end,
            "exact_quote": quote,
            "mention_id": mention["mention_id"],
            "derivation": link.get("derivation"),
        })

    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (chunk_id, canonical_id), refs in refs_by_chunk_entity.items():
        entity = entity_by_id[canonical_id]
        refs.sort(key=lambda item: (item["mention_char_start"], item["mention_char_end"]))
        canonical_name = str(entity["canonical_name"])
        result[chunk_id].append({
            "candidate_key": canonical_id,
            "canonical_id": canonical_id,
            "entity_type": entity["entity_type"],
            # 规则结构直接使用规范名称，避免同一外层 mention 展开多个内层实体时歧义。
            "mention": canonical_name,
            "canonical_name_candidate": canonical_name,
            "source_ref": refs[0],
            "source_refs": refs,
            "extraction_status": "VALID",
            "review_status": entity.get("review_status", "PENDING"),
            "publication_status": "HOLD",
        })

    # 规则可引用753目录中的全局规范实体；当规范名在当前块逐字出现但mention层漏映射时，
    # 只补一个可回放的块内引用，不创建新实体或同义词。
    present_ids = {
        (chunk_id, str(item["canonical_id"]))
        for chunk_id, items in result.items()
        for item in items
    }
    for chunk_id, chunk in chunks_by_id.items():
        for entity in entities:
            canonical_id = str(entity["canonical_id"])
            canonical_name = str(entity["canonical_name"])
            if (chunk_id, canonical_id) in present_ids or not canonical_name:
                continue
            start = chunk.text.find(canonical_name)
            if start < 0:
                continue
            ref = {
                "chunk_id": chunk_id,
                "chunk_sha256": chunk.chunk_sha256,
                "mention_char_start": start,
                "mention_char_end": start + len(canonical_name),
                "exact_quote": canonical_name,
                "mention_id": None,
                "derivation": "GLOBAL_CANONICAL_EXACT_SURFACE",
            }
            result[chunk_id].append({
                "candidate_key": canonical_id,
                "canonical_id": canonical_id,
                "entity_type": entity["entity_type"],
                "mention": canonical_name,
                "canonical_name_candidate": canonical_name,
                "source_ref": ref,
                "source_refs": [ref],
                "extraction_status": "VALID",
                "review_status": entity.get("review_status", "PENDING"),
                "publication_status": "HOLD",
            })
    for nodes in result.values():
        nodes.sort(key=lambda item: (item["entity_type"], item["mention"], item["canonical_id"]))
    return dict(result)


def _canonical_rule_graph(
    *,
    schema: Mapping[str, Any],
    business_nodes: Sequence[Mapping[str, Any]],
    rule_nodes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    combined = [*map(dict, business_nodes), *map(dict, rule_nodes)]
    normalized = build_rule_relationships_from_definitions(schema=schema, nodes=combined)
    invalid = normalized.invalid_rule_keys
    partial_rule_keys = {
        str(
            item["target_candidate_key"]
            if item.get("relation_type") == "RULE_INPUT"
            else item["source_candidate_key"]
        )
        for item in normalized.accepted
        if item.get("extraction_status") != "VALID"
    }
    edge_counts: dict[str, int] = defaultdict(int)
    for item in normalized.accepted:
        rule_key = (
            item.get("target_candidate_key")
            if item.get("relation_type") == "RULE_INPUT"
            else item.get("source_candidate_key")
        )
        if isinstance(rule_key, str):
            edge_counts[rule_key] += 1
    incomplete_rule_keys = {
        str(item["candidate_key"])
        for item in rule_nodes
        if edge_counts.get(str(item["candidate_key"]), 0)
        != (
            len(item.get("rule_inputs", []))
            + len(item.get("rule_outputs", []))
            + len(item.get("rule_excluded_outputs", []))
        )
    }
    rejected_keys = invalid | partial_rule_keys | incomplete_rule_keys
    retained_rules = [
        dict(item) for item in rule_nodes if item.get("candidate_key") not in rejected_keys
    ]
    retained_keys = {str(item["candidate_key"]) for item in retained_rules}
    retained_edges = [
        dict(item)
        for item in normalized.accepted
        if (
            item.get("target_candidate_key") in retained_keys
            if item.get("relation_type") == "RULE_INPUT"
            else item.get("source_candidate_key") in retained_keys
        )
    ]
    return retained_rules, retained_edges, list(normalized.review_items)


def _edges_for_rules(
    relationships: Sequence[Mapping[str, Any]], rules: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """移除 checkpoint 中被结构门禁淘汰的规则所留下的孤立边。"""
    rule_keys = {str(item["candidate_key"]) for item in rules}
    return [
        dict(item)
        for item in relationships
        if (
            item.get("target_candidate_key") in rule_keys
            if item.get("relation_type") == "RULE_INPUT"
            else item.get("source_candidate_key") in rule_keys
        )
    ]


async def run_chapter_rule_nodes(
    *,
    canonical_path: Path = DEFAULT_CANONICAL_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    plan_only: bool = False,
    chunk_ids: Sequence[str] | None = None,
    exclusions_only: bool = False,
) -> dict[str, Any]:
    """逐 chunk 抽取规则，输出可恢复的 candidate-only/HOLD 工件。"""
    started_at = datetime.now().astimezone()
    started_clock = perf_counter()
    manifest, chunks = load_chunk_manifest(DEFAULT_CHUNK_MANIFEST)
    all_chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if chunk_ids is not None:
        requested = set(chunk_ids)
        known = set(all_chunks_by_id)
        unknown = requested - known
        if unknown:
            raise GraphBuilderConfigurationError(
                f"chapter_rule_chunk_unknown:{','.join(sorted(unknown))}"
            )
        chunks = [chunk for chunk in chunks if chunk.chunk_id in requested]
    canonical, mentions, entities = load_frozen_catalog(canonical_path)
    endpoints_by_chunk = build_canonical_endpoints_by_chunk(
        canonical=canonical,
        mentions=mentions,
        entities=entities,
        chunks_by_id=all_chunks_by_id,
    )
    schema = load_candidate_graph_schema(DEFAULT_SCHEMA_PATH)
    plan = {
        "schema_version": "chapter-rule-extraction-plan/v0.1",
        "status": "candidate-only",
        "publication_status": "HOLD",
        "gold_exposed_to_model": False,
        "judge": "not_run",
        "chapter_id": manifest.get("chapter_id"),
        "chunk_count": len(chunks),
        "canonical_entity_count": len(entities),
        "mention_count": len(mentions),
        "planned_model_calls": len(chunks),
        "endpoint_counts_by_chunk": {
            chunk.chunk_id: len(endpoints_by_chunk.get(chunk.chunk_id, [])) for chunk in chunks
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "extraction-plan.json", plan)
    if plan_only:
        return plan

    client = create_deepseek_graph_builder()
    all_rules: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []
    all_reviews: list[dict[str, Any]] = []
    chunk_records: dict[str, Any] = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    actual_calls = 0
    try:
        for index, chunk in enumerate(chunks, start=1):
            slug = "-".join(chunk.chunk_id.rsplit(":", 2)[-2:])
            chunk_root = output_root / "chunks" / slug
            checkpoint = chunk_root / "rules.json"
            if checkpoint.is_file():
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                checkpoint_rules = payload.get("rules", [])
                checkpoint_edges = _edges_for_rules(
                    payload.get("relationships", []), checkpoint_rules
                )
                if checkpoint_edges != payload.get("relationships", []):
                    payload["relationships"] = checkpoint_edges
                    payload.setdefault("summary", {})["relationship_count"] = len(checkpoint_edges)
                    atomic_write_json(checkpoint, payload)
                all_rules.extend(checkpoint_rules)
                all_edges.extend(checkpoint_edges)
                all_reviews.extend(
                    {"chunk_id": chunk.chunk_id, **dict(item)}
                    for item in payload.get("review_items", [])
                    if isinstance(item, Mapping)
                )
                model_record = payload.get("model_record", {})
                if isinstance(model_record, Mapping):
                    actual_calls += _json_int(model_record.get("attempts", 0))
                    checkpoint_usage = model_record.get("usage", {})
                    if isinstance(checkpoint_usage, Mapping):
                        for key in usage:
                            usage[key] += _json_int(checkpoint_usage.get(key, 0))
                chunk_records[chunk.chunk_id] = payload.get("summary", {})
                print(f"[{index}/{len(chunks)}] {chunk.chunk_id}: 使用已完成 checkpoint", flush=True)
                continue
            endpoints = endpoints_by_chunk.get(chunk.chunk_id, [])
            print(f"[{index}/{len(chunks)}] {chunk.chunk_id}: {len(endpoints)} 个 canonical 端点", flush=True)
            extracted, model_record = await _extract_rule_nodes(
                client,
                chunk=chunk,
                schema=schema,
                business_nodes=endpoints,
                prompt_template=(
                    EXCLUSION_RULE_PROMPT_TEMPLATE
                    if exclusions_only else RULE_NODE_PROMPT_TEMPLATE
                ),
            )
            gated, gate_rejections = partition_invalid_rules(
                extracted, strict_graph_shapes=True
            )
            rules, edges, edge_reviews = _canonical_rule_graph(
                schema=schema, business_nodes=endpoints, rule_nodes=gated
            )
            reviews = [
                *model_record.get("normalization_reviews", []),
                *gate_rejections,
                *edge_reviews,
            ]
            for item in rules:
                item["publication_status"] = "HOLD"
            for item in edges:
                item["publication_status"] = "HOLD"
            for key in usage:
                usage[key] += int(model_record.get("usage", {}).get(key, 0))
            actual_calls += int(model_record.get("attempts", 0))
            summary = {
                "endpoint_count": len(endpoints),
                "proposed_count": model_record.get("proposed_count", 0),
                "rule_count": len(rules),
                "relationship_count": len(edges),
                "review_count": len(reviews),
                "model_error": model_record.get("model_error"),
                "attempts": model_record.get("attempts", 0),
            }
            atomic_write_json(checkpoint, {
                "chunk_id": chunk.chunk_id,
                "chunk_sha256": chunk.chunk_sha256,
                "status": "candidate-only",
                "publication_status": "HOLD",
                "rules": rules,
                "relationships": edges,
                "review_items": reviews,
                "model_record": model_record,
                "summary": summary,
            })
            all_rules.extend(rules)
            all_edges.extend(edges)
            all_reviews.extend({"chunk_id": chunk.chunk_id, **dict(item)} for item in reviews)
            chunk_records[chunk.chunk_id] = summary
    finally:
        await client.aclose()

    canonical_nodes = [{
        **dict(entity),
        "candidate_key": entity["canonical_id"],
        "publication_status": "HOLD",
    } for entity in entities]
    graph = {
        "schema_version": "chapter-candidate-rule-graph/v0.8",
        "status": "candidate-only",
        "publication_status": "HOLD",
        "nodes": [*canonical_nodes, *all_rules],
        "relationships": all_edges,
        "rules": all_rules,
    }
    atomic_write_json(output_root / "rules.json", {
        "schema_version": "chapter-rule-candidates/v0.8",
        "status": "candidate-only",
        "publication_status": "HOLD",
        "rules": all_rules,
    })
    atomic_write_json(output_root / "relationships.json", {
        "schema_version": "chapter-rule-relationships/v0.8",
        "status": "candidate-only",
        "publication_status": "HOLD",
        "relationships": all_edges,
    })
    atomic_write_json(output_root / "review-queue.json", {"items": all_reviews})
    atomic_write_json(output_root / "graph.json", graph)
    run_manifest = {
        **plan,
        "schema_version": "chapter-rule-extraction-run/v0.1",
        "model": DEEPSEEK_MODEL,
        "prompt_version": (
            EXCLUSION_RULE_PROMPT_VERSION if exclusions_only else RULE_NODE_PROMPT_VERSION
        ),
        "rule_count": len(all_rules),
        "relationship_count": len(all_edges),
        "review_item_count": len(all_reviews),
        "chunks": chunk_records,
        "execution": {
            "model_calls": actual_calls,
            "usage": usage,
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now().astimezone().isoformat(),
            "duration_seconds": round(perf_counter() - started_clock, 3),
        },
        "reproducibility": {
            "source_commit": _git_commit(),
            "canonical_path": str(canonical_path),
            "canonical_sha256": _sha256(canonical_path),
            "chunk_manifest": str(DEFAULT_CHUNK_MANIFEST),
            "chunk_manifest_sha256": _sha256(DEFAULT_CHUNK_MANIFEST),
            "schema_path": str(DEFAULT_SCHEMA_PATH),
            "schema_sha256": _sha256(DEFAULT_SCHEMA_PATH),
            "prompt_sha256": hashlib.sha256(
                (
                    EXCLUSION_RULE_PROMPT_TEMPLATE
                    if exclusions_only else RULE_NODE_PROMPT_TEMPLATE
                ).encode()
            ).hexdigest(),
        },
        "boundary": (
            "Rules were discovered from Chapter 01 canonical chunks and may bind only to frozen v0.8 "
            "canonical entities. Gold and Judge were not used; all outputs remain candidate-only/HOLD."
        ),
    }
    atomic_write_json(output_root / "run-manifest.json", run_manifest)
    return run_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从第一章原文和 v0.8 规范实体抽取规则节点")
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--chunk-id", action="append", dest="chunk_ids")
    parser.add_argument("--exclusions-only", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(run_chapter_rule_nodes(
        canonical_path=args.canonical,
        output_root=args.output_root,
        plan_only=args.plan_only,
        chunk_ids=args.chunk_ids,
        exclusions_only=args.exclusions_only,
    ))
    print(json.dumps({
        "canonical_entity_count": report["canonical_entity_count"],
        "chunk_count": report["chunk_count"],
        "rule_count": report.get("rule_count"),
        "relationship_count": report.get("relationship_count"),
        "review_item_count": report.get("review_item_count"),
        "publication_status": report["publication_status"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
