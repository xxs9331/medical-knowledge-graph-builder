"""从冻结的第一章规范实体生成关系候选图，不读取评测金标或调用 Judge。"""

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
import math
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from ...llm_extraction import EvidenceChunk, atomic_write_json, load_chunk_manifest
from ..client import (
    create_deepseek_graph_builder,
    create_luna_graph_builder,
    create_opencode_luna_graph_builder,
    create_qwen_flash_graph_builder,
)
from ..contract import (
    DEFAULT_CHUNK_MANIFEST,
    DEFAULT_SCHEMA_PATH,
    DEEPSEEK_MODEL,
    ORDINARY_RELATION_TYPES,
    PROJECT_ROOT,
    STATE_RELATION_TYPES,
    GraphBuilderConfigurationError,
)
from ..evaluation.artifacts import load_json_object
from ..relation_classifier import (
    RELATION_ROUTE_DEFINITIONS,
    build_relation_pairs,
    classify_relationships_one_stage,
    classify_relationships_two_stage,
    classify_relationships_verified,
    classify_relationships_routed,
    classify_relationships_evidence_grouped,
)
from ..schema import load_candidate_graph_schema
from ..validation import normalize_candidate_relationships


DEFAULT_CANONICAL_PATH = (
    PROJECT_ROOT / "evaluation/chapter-01/chapter-01-canonical-entities-v0.8.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "runtime/candidates/chapter-01/knowledge-graph-v0.8"
)
CLASSIFIED_RELATION_TYPES = frozenset(ORDINARY_RELATION_TYPES) | frozenset(
    STATE_RELATION_TYPES
)
LUNA_MODEL = "gpt-5.6-luna"
LUNA_REASONING_EFFORT = "high"
OPENCODE_LUNA_MODEL = "gpt-5.6-luna"
QWEN_FLASH_MODEL = "qwen-flash"

# 嵌套实体层会把内层实体的证据回链到外层 mention，用于保留可追溯的展开来源。
# 这类回链不代表外层整段文字可以直接作为该内层实体的关系端点；否则一个长 mention
# 会同时成为多个不相干实体的端点，候选对会发生语义漂移。关系阶段只采用下面这些
# 可直接解释为该位置实体的映射，完整映射仍原样写入 evidence-links.json 供审计使用。
RELATION_ENDPOINT_DERIVATIONS = frozenset({
    "DIRECT_MENTION",
    "PARENTHETICAL_ALIAS",
    "COORDINATION_EXPANSION",
    "CONTEXT_NORMALIZATION",
    "NESTED_INDICATOR_BASE",
    "TABLE_THRESHOLD_DERIVATION",
})
_ASCII_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]")


def _is_relation_endpoint_span(text: str, start: int, end: int, mention: str) -> bool:
    """拒绝被更长 ASCII 缩写包住的伪 mention，例如把 MCHC 中的 MCH 当端点。"""
    if _ASCII_TOKEN_PATTERN.search(mention) is None:
        return True
    left = text[start - 1] if start else ""
    right = text[end] if end < len(text) else ""
    return not (
        (left.isascii() and left.isalnum())
        or (right.isascii() and right.isalnum())
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_mentions(path: Path) -> list[dict[str, Any]]:
    payload = load_json_object(path)
    mentions: list[dict[str, Any]] = []
    for case in payload.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        case_mentions = case.get("mentions")
        if not isinstance(case_mentions, list):
            continue
        mentions.extend(
            dict(item)
            for item in case_mentions
            if isinstance(item, Mapping)
        )
    return mentions


def load_frozen_catalog(
    canonical_path: Path = DEFAULT_CANONICAL_PATH,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """读取并交叉校验 canonical、mention 和映射三层冻结数据。"""
    canonical = load_json_object(canonical_path)
    source_mentions = canonical.get("source_mentions")
    if not isinstance(source_mentions, str) or not source_mentions:
        raise GraphBuilderConfigurationError("chapter_graph_source_mentions_missing")
    mention_path = Path(source_mentions)
    if not mention_path.is_absolute():
        mention_path = PROJECT_ROOT / mention_path
    mentions = _load_mentions(mention_path)
    entities = [
        dict(item)
        for item in canonical.get("canonical_entities", [])
        if isinstance(item, Mapping)
    ]
    links = [
        dict(item)
        for item in canonical.get("mention_to_canonical_links", [])
        if isinstance(item, Mapping)
    ]
    mention_ids = {item.get("mention_id") for item in mentions}
    entity_ids = {item.get("canonical_id") for item in entities}
    if len(mention_ids) != len(mentions) or None in mention_ids:
        raise GraphBuilderConfigurationError("chapter_graph_mention_identity_invalid")
    if len(entity_ids) != len(entities) or None in entity_ids:
        raise GraphBuilderConfigurationError("chapter_graph_entity_identity_invalid")
    if any(link.get("mention_id") not in mention_ids for link in links):
        raise GraphBuilderConfigurationError("chapter_graph_link_mention_unknown")
    if any(link.get("canonical_id") not in entity_ids for link in links):
        raise GraphBuilderConfigurationError("chapter_graph_link_entity_unknown")
    canonical["resolved_source_mentions_path"] = str(mention_path)
    return canonical, mentions, entities


def build_frozen_nodes_by_chunk(
    *,
    canonical: Mapping[str, Any],
    mentions: Sequence[Mapping[str, Any]],
    entities: Sequence[Mapping[str, Any]],
    chunks_by_id: Mapping[str, EvidenceChunk],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """把规范实体映射为逐 mention 的关系端点，同时保留最终规范节点。"""
    mention_by_id = {str(item["mention_id"]): item for item in mentions}
    entity_by_id = {str(item["canonical_id"]): item for item in entities}
    nodes_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_links: list[dict[str, Any]] = []
    for link in canonical.get("mention_to_canonical_links", []):
        if not isinstance(link, Mapping):
            continue
        mention_id = str(link["mention_id"])
        canonical_id = str(link["canonical_id"])
        mention = mention_by_id[mention_id]
        entity = entity_by_id[canonical_id]
        chunk_id = str(mention["chunk_id"])
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise GraphBuilderConfigurationError("chapter_graph_chunk_unknown")
        start, end = mention.get("start"), mention.get("end")
        exact_quote = mention.get("exact_quote")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not isinstance(exact_quote, str)
            or chunk.text[start:end] != exact_quote
        ):
            raise GraphBuilderConfigurationError("chapter_graph_mention_replay_failed")
        candidate_key = f"{mention_id}:{canonical_id}"
        source_ref = {
            "chunk_id": chunk_id,
            "chunk_sha256": chunk.chunk_sha256,
            "mention_char_start": start,
            "mention_char_end": end,
            "exact_quote": exact_quote,
        }
        evidence_links.append({
            "mention_id": mention_id,
            "canonical_id": canonical_id,
            "derivation": link.get("derivation"),
            "source_ref": source_ref,
        })
        if link.get("derivation") not in RELATION_ENDPOINT_DERIVATIONS:
            continue
        if not _is_relation_endpoint_span(chunk.text, start, end, exact_quote):
            continue
        nodes_by_chunk[chunk_id].append({
            "candidate_key": candidate_key,
            "canonical_id": canonical_id,
            "canonical_name": entity["canonical_name"],
            "aliases": entity.get("aliases", []),
            "entity_type": entity["entity_type"],
            "mention": exact_quote,
            "source_ref": source_ref,
            "extraction_status": "VALID",
            "review_status": entity.get("review_status", "PENDING"),
            "publication_status": "HOLD",
        })
    for nodes in nodes_by_chunk.values():
        nodes.sort(key=lambda item: (
            item["source_ref"]["mention_char_start"],
            item["source_ref"]["mention_char_end"],
            item["canonical_id"],
        ))
    canonical_nodes = [
        {
            **dict(entity),
            "candidate_key": entity["canonical_id"],
            "publication_status": "HOLD",
        }
        for entity in entities
    ]
    return dict(nodes_by_chunk), evidence_links


def _canonicalize_relationships(
    relationships: Sequence[Mapping[str, Any]],
    endpoint_nodes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """把逐 mention 关系折叠到规范实体端点，并按类型、端点和证据去重。"""
    canonical_by_key = {
        str(node["candidate_key"]): str(node["canonical_id"])
        for node in endpoint_nodes
    }
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, int, int]] = set()
    for relation in relationships:
        source_id = canonical_by_key[str(relation["source_candidate_key"])]
        target_id = canonical_by_key[str(relation["target_candidate_key"])]
        if source_id == target_id:
            continue
        source_ref = relation.get("source_ref", {})
        identity = (
            str(relation["relation_type"]),
            source_id,
            target_id,
            str(source_ref.get("chunk_id", "")),
            int(source_ref.get("char_start", source_ref.get("mention_char_start", -1))),
            int(source_ref.get("char_end", source_ref.get("mention_char_end", -1))),
        )
        if identity in seen:
            continue
        seen.add(identity)
        digest = hashlib.sha256("\x1f".join(map(str, identity)).encode()).hexdigest()[:24]
        result.append({
            **dict(relation),
            "candidate_key": f"relation:{digest}",
            "source_candidate_key": source_id,
            "target_candidate_key": target_id,
        })
    return result


async def run_chapter_candidate_graph(
    *,
    canonical_path: Path = DEFAULT_CANONICAL_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    batch_size: int = 16,
    max_concurrency: int = 4,
    provider: str = "deepseek",
    classification_strategy: str = "one-stage",
    chunk_ids: Sequence[str] | None = None,
    plan_only: bool = False,
) -> dict[str, Any]:
    """运行第一章冻结实体关系分类并生成候选图工件。"""
    if batch_size < 1:
        raise GraphBuilderConfigurationError("chapter_graph_batch_size_invalid")
    if max_concurrency < 1:
        raise GraphBuilderConfigurationError("chapter_graph_concurrency_invalid")
    if provider not in {"deepseek", "luna", "opencode-go", "qwen-flash"}:
        raise GraphBuilderConfigurationError("chapter_graph_provider_invalid")
    if classification_strategy not in {
        "one-stage", "two-stage", "verified", "routed", "evidence-grouped"
    }:
        raise GraphBuilderConfigurationError("chapter_graph_classification_strategy_invalid")
    model_name = {
        "deepseek": DEEPSEEK_MODEL,
        "luna": LUNA_MODEL,
        "opencode-go": OPENCODE_LUNA_MODEL,
        "qwen-flash": QWEN_FLASH_MODEL,
    }[provider]
    reasoning_effort = (
        LUNA_REASONING_EFFORT if provider in {"luna", "opencode-go"} else None
    )
    resuming = output_root.exists() and any(output_root.iterdir())
    if resuming and not (output_root / "candidate-plan.json").is_file():
        raise GraphBuilderConfigurationError(f"chapter_graph_output_not_resumable:{output_root}")
    started_at = datetime.now().astimezone()
    started_clock = perf_counter()
    manifest, chunks = load_chunk_manifest(DEFAULT_CHUNK_MANIFEST)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if chunk_ids is not None:
        requested = set(chunk_ids)
        known = {chunk.chunk_id for chunk in chunks}
        if not requested or not requested <= known:
            raise GraphBuilderConfigurationError("chapter_graph_chunk_selection_invalid")
        chunks = [chunk for chunk in chunks if chunk.chunk_id in requested]
    canonical, mentions, entities = load_frozen_catalog(canonical_path)
    nodes_by_chunk, evidence_links = build_frozen_nodes_by_chunk(
        canonical=canonical,
        mentions=mentions,
        entities=entities,
        chunks_by_id=chunks_by_id,
    )
    selected_chunk_ids = {chunk.chunk_id for chunk in chunks}
    selected_evidence_links = [
        link
        for link in evidence_links
        if link.get("source_ref", {}).get("chunk_id") in selected_chunk_ids
    ]
    schema = load_candidate_graph_schema(DEFAULT_SCHEMA_PATH)
    pair_counts = {}
    planned_calls_by_chunk: dict[str, int] = {}
    for chunk in chunks:
        if classification_strategy == "routed":
            route_pair_counts = [
                len(build_relation_pairs(
                    chunk=chunk,
                    schema=schema,
                    nodes=nodes_by_chunk.get(chunk.chunk_id, []),
                    allowed_relation_types=CLASSIFIED_RELATION_TYPES & relation_types,
                ))
                for _route_name, relation_types in RELATION_ROUTE_DEFINITIONS
            ]
            pair_counts[chunk.chunk_id] = sum(route_pair_counts)
            planned_calls_by_chunk[chunk.chunk_id] = sum(
                math.ceil(count / batch_size) for count in route_pair_counts
            )
        else:
            pairs = build_relation_pairs(
                chunk=chunk,
                schema=schema,
                nodes=nodes_by_chunk.get(chunk.chunk_id, []),
                allowed_relation_types=CLASSIFIED_RELATION_TYPES,
            )
            pair_counts[chunk.chunk_id] = len(pairs)
            if classification_strategy == "evidence-grouped":
                group_sizes: dict[tuple[int, int], int] = {}
                for pair in pairs:
                    key = (pair.evidence_start, pair.evidence_end)
                    group_sizes[key] = group_sizes.get(key, 0) + 1
                planned_calls_by_chunk[chunk.chunk_id] = sum(
                    math.ceil(size / batch_size) for size in group_sizes.values()
                )
            else:
                planned_calls_by_chunk[chunk.chunk_id] = math.ceil(len(pairs) / batch_size)
    planned_calls = sum(planned_calls_by_chunk.values())
    plan = {
        "schema_version": "chapter-candidate-graph-plan/v0.1",
        "status": "candidate-only",
        "publication_status": "HOLD",
        "gold_exposed_to_model": False,
        "judge": "not_run",
        "chapter_id": manifest.get("chapter_id"),
        "chunk_count": len(chunks),
        "canonical_entity_count": len(entities),
        "mention_count": len(mentions),
        "evidence_link_count": len(selected_evidence_links),
        "candidate_pair_count": sum(pair_counts.values()),
        "planned_model_calls": planned_calls,
        "batch_size": batch_size,
        "max_concurrency": max_concurrency,
        "provider": provider,
        "model": model_name,
        "reasoning_effort": reasoning_effort,
        "classification_strategy": classification_strategy,
        "selected_chunk_ids": [chunk.chunk_id for chunk in chunks],
        "pair_counts_by_chunk": pair_counts,
        "planned_calls_by_chunk": planned_calls_by_chunk,
    }
    if not resuming:
        atomic_write_json(output_root / "candidate-plan.json", plan)
    nodes_payload = {
        "schema_version": "chapter-canonical-nodes/v0.1",
        "status": "candidate-only",
        "publication_status": "HOLD",
        "nodes": [
            {
                **dict(entity),
                "candidate_key": entity["canonical_id"],
                "publication_status": "HOLD",
            }
            for entity in entities
        ],
    }
    if not resuming:
        atomic_write_json(output_root / "nodes.json", nodes_payload)
        atomic_write_json(output_root / "evidence-links.json", {
        "schema_version": "chapter-entity-evidence-links/v0.1",
        "links": selected_evidence_links,
        })
    if plan_only:
        return plan

    if provider == "deepseek":
        client = create_deepseek_graph_builder()
    elif provider == "luna":
        client = create_luna_graph_builder(
            model_name=model_name,
            reasoning_effort=LUNA_REASONING_EFFORT,
        )
    elif provider == "opencode-go":
        client = create_opencode_luna_graph_builder(
            model_name=model_name,
            reasoning_effort=LUNA_REASONING_EFFORT,
        )
    else:
        client = create_qwen_flash_graph_builder(model_name=model_name)
    all_relationships: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    chunk_records: dict[str, Any] = {}
    actual_model_calls = 0
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_chunk(
        index: int, chunk: EvidenceChunk
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], int]:
        endpoint_nodes = nodes_by_chunk.get(chunk.chunk_id, [])
        pair_count = pair_counts[chunk.chunk_id]
        chunk_root = output_root / "chunks" / "-".join(chunk.chunk_id.rsplit(":", 2)[-2:])
        completed_path = chunk_root / "relationships.json"
        audit_path = chunk_root / "classification-audit.json"
        if completed_path.is_file() and audit_path.is_file():
            completed = load_json_object(completed_path)
            audit = load_json_object(audit_path)
            canonical_relations = [
                dict(item) for item in completed.get("relationships", [])
                if isinstance(item, Mapping)
            ]
            saved_review_items = [
                dict(item) for item in completed.get("review_items", [])
                if isinstance(item, Mapping)
            ]
            print(f"[{index}/{len(chunks)}] {chunk.chunk_id}: 使用已完成 checkpoint", flush=True)
            return canonical_relations, saved_review_items, {
                "endpoint_count": len(endpoint_nodes),
                "candidate_pair_count": pair_count,
                "classified_count": audit.get("classified_count", 0),
                "accepted_count": len(canonical_relations),
                "review_count": len(saved_review_items),
                "resumed_from_checkpoint": True,
            }, int(audit.get("model_call_count", 0))
        async with semaphore:
            print(
                f"[{index}/{len(chunks)}] {chunk.chunk_id}: "
                f"{len(endpoint_nodes)} 个端点，{pair_count} 个候选对",
                flush=True,
            )
            classifier = {
                "one-stage": classify_relationships_one_stage,
                "two-stage": classify_relationships_two_stage,
                "verified": classify_relationships_verified,
                "routed": classify_relationships_routed,
                "evidence-grouped": classify_relationships_evidence_grouped,
            }[classification_strategy]
            raw_graph, audit = await classifier(
                client,
                chunk=chunk,
                schema=schema,
                nodes=endpoint_nodes,
                allowed_relation_types=CLASSIFIED_RELATION_TYPES,
                batch_size=batch_size,
            )
            normalized = normalize_candidate_relationships(
                raw_graph,
                chunk=chunk,
                schema=schema,
                nodes=endpoint_nodes,
                allowed_relation_types=CLASSIFIED_RELATION_TYPES,
                validate_rule_structures=False,
            )
            canonical_relations = _canonicalize_relationships(
                normalized.accepted,
                endpoint_nodes,
            )
            saved_review_items = [dict(item) for item in normalized.review_items]
            atomic_write_json(chunk_root / "classification-audit.json", audit)
            atomic_write_json(chunk_root / "relationships.json", {
                "relationships": canonical_relations,
                "review_items": saved_review_items,
            })
            return canonical_relations, saved_review_items, {
                "endpoint_count": len(endpoint_nodes),
                "candidate_pair_count": pair_count,
                "classified_count": audit["classified_count"],
                "accepted_count": len(canonical_relations),
                "review_count": len(saved_review_items),
                "resumed_from_checkpoint": False,
            }, int(audit.get("model_call_count", 0))

    try:
        results = await asyncio.gather(*(
            process_chunk(index, chunk)
            for index, chunk in enumerate(chunks, start=1)
        ))
        for chunk, result in zip(chunks, results, strict=True):
            canonical_relations, saved_review_items, record, model_calls = result
            all_relationships.extend(canonical_relations)
            review_items.extend(
                {"chunk_id": chunk.chunk_id, **item} for item in saved_review_items
            )
            chunk_records[chunk.chunk_id] = record
            actual_model_calls += model_calls
    finally:
        await client.aclose()

    unique_relationships: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for relationship in all_relationships:
        key = str(relationship["candidate_key"])
        if key not in seen_keys:
            seen_keys.add(key)
            unique_relationships.append(relationship)
    canonical_nodes = load_json_object(output_root / "nodes.json")["nodes"]
    graph = {
        "schema_version": "chapter-candidate-knowledge-graph/v0.8",
        "status": "candidate-only",
        "publication_status": "HOLD",
        "nodes": canonical_nodes,
        "relationships": unique_relationships,
        "rules": [],
    }
    atomic_write_json(output_root / "relationships.json", {
        "schema_version": "chapter-candidate-relationships/v0.1",
        "status": "candidate-only",
        "publication_status": "HOLD",
        "relationships": unique_relationships,
    })
    atomic_write_json(output_root / "review-queue.json", {
        "schema_version": "chapter-candidate-review-queue/v0.1",
        "items": review_items,
    })
    atomic_write_json(output_root / "graph.json", graph)
    manifest_record = {
        **plan,
        "schema_version": "chapter-candidate-graph-run/v0.1",
        "model": model_name,
        "reasoning_effort": reasoning_effort,
        "prompt_version": f"{classification_strategy}-structure-relation-classification/v0.1",
        "relationship_count": len(unique_relationships),
        "review_item_count": len(review_items),
        "rules_included": False,
        "chunks": chunk_records,
        "execution": {
            "model_calls": actual_model_calls,
            "initial_planned_model_calls": planned_calls,
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now().astimezone().isoformat(),
            "duration_seconds": round(perf_counter() - started_clock, 3),
        },
        "reproducibility": {
            "source_commit": _git_commit(),
            "canonical_path": str(canonical_path),
            "canonical_sha256": _sha256(canonical_path),
            "mention_path": canonical["resolved_source_mentions_path"],
            "mention_sha256": _sha256(Path(canonical["resolved_source_mentions_path"])),
            "chunk_manifest": str(DEFAULT_CHUNK_MANIFEST),
            "chunk_manifest_sha256": _sha256(DEFAULT_CHUNK_MANIFEST),
            "schema_path": str(DEFAULT_SCHEMA_PATH),
            "schema_sha256": _sha256(DEFAULT_SCHEMA_PATH),
        },
        "boundary": (
            "Frozen v0.8 entities and local structure pairs only. Gold and Judge were not used. "
            "Cross-chunk relations and rules are not included; all outputs remain HOLD."
        ),
    }
    atomic_write_json(output_root / "run-manifest.json", manifest_record)
    return manifest_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成第一章 v0.8 冻结实体候选知识图谱")
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument(
        "--provider",
        choices=("deepseek", "luna", "opencode-go", "qwen-flash"),
        default="deepseek",
    )
    parser.add_argument(
        "--classification-strategy",
        choices=("one-stage", "two-stage", "verified", "routed", "evidence-grouped"),
        default="one-stage",
    )
    parser.add_argument("--chunk-id", action="append", dest="chunk_ids")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(run_chapter_candidate_graph(
        canonical_path=args.canonical,
        output_root=args.output_root,
        batch_size=args.batch_size,
        max_concurrency=args.max_concurrency,
        provider=args.provider,
        classification_strategy=args.classification_strategy,
        chunk_ids=args.chunk_ids,
        plan_only=args.plan_only,
    ))
    print(json.dumps({
        "canonical_entity_count": report["canonical_entity_count"],
        "mention_count": report["mention_count"],
        "candidate_pair_count": report["candidate_pair_count"],
        "planned_model_calls": report["planned_model_calls"],
        "relationship_count": report.get("relationship_count"),
        "publication_status": report["publication_status"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
