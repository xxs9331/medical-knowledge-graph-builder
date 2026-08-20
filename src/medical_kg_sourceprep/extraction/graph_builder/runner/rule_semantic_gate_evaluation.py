"""在冻结业务实体上运行规则语义门控实验。"""

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.experimental.components.types import Neo4jGraph

from ...artifacts import sha256_path
from ...llm_extraction import EvidenceChunk, atomic_write_json, load_chunk_manifest
from ..client import DeepSeekGraphBuilderClient, create_deepseek_graph_builder
from ..contract import (
    DEFAULT_CHUNK_MANIFEST,
    DEFAULT_SCHEMA_PATH,
    DEEPSEEK_MODEL,
    PROJECT_ROOT,
    RULE_NODE_PROMPT_TEMPLATE,
    RULE_NODE_PROMPT_VERSION,
    GraphBuilderConfigurationError,
)
from ..evaluation.artifacts import load_json_object
from ..evaluation.scoring import merge_candidate_graphs, score_candidate_graph
from ..rule_gate import partition_invalid_rules
from ..schema import _extract_graph, build_graphrag_schema, load_candidate_graph_schema
from ..validation import (
    _catalog_for_prompt,
    build_rule_relationships_from_definitions,
    normalize_candidate_nodes,
)


DEFAULT_GOLD = PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json"
BASELINE_ROOT = (
    PROJECT_ROOT / "runtime/evaluations/typical-cases/structured-rules-v0.10-full"
)
BASELINE_REPORT = BASELINE_ROOT / "evaluation-result-standard-prf1.json"
DEFAULT_PROJECT_ROOT = PROJECT_ROOT / "runtime/evaluations/rule-semantic-gate-r1"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _string_list(value: object) -> list[str]:
    """从动态 JSON 字段中安全读取字符串数组。"""
    return [str(item) for item in value] if isinstance(value, list) else []


def _baseline_graph_path(chunk_id: str) -> Path:
    """根据 chunk ID 解析 A 组冻结候选图。"""
    parts = chunk_id.split(":")
    if len(parts) < 2:
        raise GraphBuilderConfigurationError(f"invalid_chunk_id:{chunk_id}")
    return BASELINE_ROOT / "chunks" / f"{parts[-2]}-{parts[-1]}" / "candidate-graph/graph.json"


def _rule_metrics(case_results: Sequence[Mapping[str, Any]], phase: str) -> dict[str, Any]:
    """汇总规则 TP/FP/FN 和标准监督 P/R/F1。"""
    tp = sum(int(item[phase]["tp"]) for item in case_results)
    fp = sum(int(item[phase]["fp"]) for item in case_results)
    fn = sum(int(item[phase]["fn"]) for item in case_results)
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_total": tp + fp,
        "gold_total": tp + fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "precision_percent": round(precision * 100, 2),
        "recall_percent": round(recall * 100, 2),
        "f1_percent": round(f1 * 100, 2),
    }


async def _extract_rule_nodes(
    client: DeepSeekGraphBuilderClient,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    business_nodes: Sequence[Mapping[str, Any]],
    prompt_template: str = RULE_NODE_PROMPT_TEMPLATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """基于冻结实体目录调用一次规则阶段，格式失败时最多重试一次。"""
    diagnostics: list[dict[str, Any]] = []
    graph_schema = build_graphrag_schema(
        schema, relation_types=(), node_types=("RuleDefinition",)
    )
    last_error: str | None = None
    graph: Neo4jGraph | None = None
    normalization = None
    attempts = 0
    endpoint_names = {
        str(item["candidate_key"]): str(item["mention"])
        for item in business_nodes
        if isinstance(item.get("candidate_key"), str)
        and isinstance(item.get("mention"), str)
    }
    for attempts in range(1, 3):
        try:
            active_prompt = prompt_template
            if attempts > 1:
                active_prompt += """

CORRECTION FOR RETRY: The previous candidate failed deterministic validation.
For every RuleDefinition, include exactly the declared rule fields, especially
rule_evidence_json with a verbatim quote from Input text. Use exact catalog
mention strings in endpoint arrays, GRAPH_COMPOSITE as the stage, ALL or
ALL_SAME_WINDOW as logic, and include top-level relationships as an empty array.
"""
            graph = await _extract_graph(
                client,
                chunk=chunk,
                graph_schema=graph_schema,
                prompt_template=active_prompt,
                examples=_catalog_for_prompt(business_nodes),
                input_text=chunk.text,
                response_diagnostics=diagnostics,
            )
            for node in graph.nodes:
                if node.label != "RuleDefinition":
                    continue
                for field in (
                    "rule_inputs_json", "rule_outputs_json", "rule_excluded_outputs_json",
                ):
                    raw = node.properties.get(field)
                    if not isinstance(raw, str):
                        continue
                    try:
                        values = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(values, list):
                        continue
                    normalized = [endpoint_names.get(str(value), value) for value in values]
                    node.properties[field] = json.dumps(
                        normalized, ensure_ascii=False, separators=(",", ":")
                    )
            normalization = normalize_candidate_nodes(
                graph,
                chunk=chunk,
                schema=schema,
                allowed_node_types=("RuleDefinition",),
                allowed_rule_stages=("GRAPH_COMPOSITE",),
                allowed_rule_logics=("ALL", "ALL_SAME_WINDOW"),
            )
            if normalization.accepted or not graph.nodes:
                last_error = None
                break
            last_error = "candidate_normalization_failed"
        except LLMGenerationError as error:
            last_error = type(error).__name__
    if normalization is None:
        normalization = normalize_candidate_nodes(
            graph or Neo4jGraph(),
            chunk=chunk,
            schema=schema,
            allowed_node_types=("RuleDefinition",),
            allowed_rule_stages=("GRAPH_COMPOSITE",),
            allowed_rule_logics=("ALL", "ALL_SAME_WINDOW"),
        )
    accepted = [
        dict(item) for item in normalization.accepted
        if item.get("extraction_status") == "VALID"
    ]
    usage = {
        key: sum(
            int(item.get("usage", {}).get(key, 0))
            for item in diagnostics
            if isinstance(item.get("usage"), Mapping)
        )
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    return accepted, {
        "attempts": attempts,
        "model_error": last_error,
        "proposed_count": len(graph.nodes) if graph is not None else 0,
        "accepted_count": len(accepted),
        "review_count": len(normalization.review_items),
        "judge_draft_count": len(normalization.judge_drafts),
        "normalization_reviews": list(normalization.review_items),
        "usage": usage,
        "response_diagnostics": diagnostics,
    }


def _rule_graph(
    *,
    schema: Mapping[str, Any],
    business_nodes: Sequence[Mapping[str, Any]],
    rule_nodes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """将规则节点确定性投影为 RULE_INPUT/RULE_OUTPUT 子图。"""
    combined_nodes = [*map(dict, business_nodes), *map(dict, rule_nodes)]
    relation_result = build_rule_relationships_from_definitions(
        schema=schema, nodes=combined_nodes
    )
    invalid_keys = relation_result.invalid_rule_keys
    public_rules = [
        node for node in rule_nodes if node.get("candidate_key") not in invalid_keys
    ]
    return {
        "nodes": [*map(dict, business_nodes), *map(dict, public_rules)],
        "relationships": list(relation_result.accepted),
    }, {
        "invalid_rule_keys": sorted(invalid_keys),
        "rule_edge_review_count": len(relation_result.review_items),
    }


async def run_rule_semantic_gate_evaluation(
    client: DeepSeekGraphBuilderClient,
    *,
    repeat_index: int,
    output_root: Path,
    gold_path: Path = DEFAULT_GOLD,
    manifest_path: Path = DEFAULT_CHUNK_MANIFEST,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> dict[str, Any]:
    """运行一次 R1/R2 规则实验；全部模型调用结束后才加载规则金标。"""
    if repeat_index not in {1, 2, 3}:
        raise GraphBuilderConfigurationError("repeat_index_must_be_1_2_or_3")
    locator_document = load_json_object(gold_path)
    case_locators = [
        {
            "case_id": str(item["case_id"]),
            "chunk_ids": _string_list(item.get("chunk_ids", [])),
        }
        for item in locator_document.get("cases", [])
        if isinstance(item, Mapping) and isinstance(item.get("case_id"), str)
    ]
    del locator_document
    selected_chunk_ids = list(dict.fromkeys(
        chunk_id for case in case_locators for chunk_id in case["chunk_ids"]
    ))
    _manifest, chunks = load_chunk_manifest(manifest_path)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    schema = load_candidate_graph_schema(schema_path)

    r1_graphs: dict[str, dict[str, list[Any]]] = {}
    r2_graphs: dict[str, dict[str, list[Any]]] = {}
    chunk_records: dict[str, dict[str, Any]] = {}
    for index, chunk_id in enumerate(selected_chunk_ids, start=1):
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise GraphBuilderConfigurationError(f"chunk_not_found:{chunk_id}")
        baseline_path = _baseline_graph_path(chunk_id)
        baseline_graph = load_json_object(baseline_path)
        business_nodes = [
            item for item in baseline_graph.get("nodes", [])
            if isinstance(item, Mapping)
            and item.get("entity_type") != "RuleDefinition"
            and item.get("extraction_status") == "VALID"
        ]
        print(f"[{index}/{len(selected_chunk_ids)}] {chunk_id} frozen_entities={len(business_nodes)}")
        rule_nodes, model_record = await _extract_rule_nodes(
            client, chunk=chunk, schema=schema, business_nodes=business_nodes
        )
        gated_nodes, gate_rejections = partition_invalid_rules(rule_nodes)
        r1_graph, r1_structure = _rule_graph(
            schema=schema, business_nodes=business_nodes, rule_nodes=rule_nodes
        )
        r2_graph, r2_structure = _rule_graph(
            schema=schema, business_nodes=business_nodes, rule_nodes=gated_nodes
        )
        r1_graphs[chunk_id] = r1_graph
        r2_graphs[chunk_id] = r2_graph
        chunk_dir = output_root / "chunks" / f"{chunk_id.split(':')[-2]}-{chunk_id.split(':')[-1]}"
        atomic_write_json(chunk_dir / "r1-rule-graph.json", r1_graph)
        atomic_write_json(chunk_dir / "r2-rule-graph.json", r2_graph)
        chunk_records[chunk_id] = {
            "baseline_graph": {
                "path": str(baseline_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_path(baseline_path),
            },
            "frozen_entity_count": len(business_nodes),
            "model": model_record,
            "r1": r1_structure,
            "r2": {
                **r2_structure,
                "gate_rejection_count": len(gate_rejections),
                "gate_rejections": gate_rejections,
            },
            "artifacts": {
                "r1_graph": str(chunk_dir / "r1-rule-graph.json"),
                "r2_graph": str(chunk_dir / "r2-rule-graph.json"),
            },
        }

    # 模型调用已经全部结束。此后才重新加载完整金标并进入确定性评分。
    gold = load_json_object(gold_path)
    cases = [item for item in gold.get("cases", []) if isinstance(item, Mapping)]
    case_results: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        chunk_ids = _string_list(case.get("chunk_ids", []))
        source_text = "\n\n".join(chunks_by_id[chunk_id].text for chunk_id in chunk_ids)
        r1_score = score_candidate_graph(
            merge_candidate_graphs(r1_graphs[chunk_id] for chunk_id in chunk_ids),
            case,
            source_text=source_text,
        )["rules"]
        r2_score = score_candidate_graph(
            merge_candidate_graphs(r2_graphs[chunk_id] for chunk_id in chunk_ids),
            case,
            source_text=source_text,
        )["rules"]
        case_results.append({"case_id": case_id, "r1": r1_score, "r2": r2_score})

    baseline_report = load_json_object(BASELINE_REPORT)
    baseline_metrics = baseline_report["prf1"]["categories"]["rules"]
    counts: dict[str, int] = {
        "case_count": len(cases),
        "unique_chunk_count": len(selected_chunk_ids),
        "model_calls": sum(int(item["model"]["attempts"]) for item in chunk_records.values()),
        "failed_chunks": sum(
            1 for item in chunk_records.values() if item["model"]["model_error"] is not None
        ),
        "r2_gate_rejections": sum(
            int(item["r2"]["gate_rejection_count"]) for item in chunk_records.values()
        ),
    }
    result: dict[str, Any] = {
        "schema_version": "rule-semantic-gate-evaluation/v0.1",
        "status": "evaluation-only",
        "publication_status": "HOLD",
        "gold_exposed_to_model": False,
        "repeat_index": repeat_index,
        "configuration": {
            "model": DEEPSEEK_MODEL,
            "temperature": 0,
            "thinking": "disabled",
            "prompt_version": RULE_NODE_PROMPT_VERSION,
            "prompt_sha256": _sha256_text(RULE_NODE_PROMPT_TEMPLATE),
            "r2_gates": [
                "reject_when_rule_inputs_intersect_rule_outputs",
                "reject_when_rule_output_is_explicit_example",
            ],
        },
        "inputs": {
            "gold": {"path": str(gold_path), "sha256": sha256_path(gold_path)},
            "manifest": {"path": str(manifest_path), "sha256": sha256_path(manifest_path)},
            "schema": {"path": str(schema_path), "sha256": sha256_path(schema_path)},
        },
        "counts": counts,
        "metrics": {
            "A": baseline_metrics,
            "R1": _rule_metrics(case_results, "r1"),
            "R2": _rule_metrics(case_results, "r2"),
        },
        "cases": case_results,
        "chunks": chunk_records,
        "boundary": (
            "Ten HUMAN_REVIEW_REQUIRED development rules only; candidate-only/HOLD; "
            "no Neo4j publication or executor-rule claim."
        ),
    }
    atomic_write_json(output_root / "evaluation-result.json", result)
    atomic_write_json(output_root / "run-manifest.json", {
        "schema_version": "rule-semantic-gate-run/v0.1",
        "status": "completed" if counts["failed_chunks"] == 0 else "partial",
        "publication_status": "HOLD",
        "repeat_index": repeat_index,
        "configuration": result["configuration"],
        "inputs": result["inputs"],
        "counts": result["counts"],
    })
    return result


async def _main(args: argparse.Namespace) -> None:
    client = create_deepseek_graph_builder()
    try:
        result = await run_rule_semantic_gate_evaluation(
            client,
            repeat_index=args.repeat_index,
            output_root=args.output_root,
            gold_path=args.gold,
            manifest_path=args.manifest,
            schema_path=args.schema,
        )
    finally:
        await client.aclose()
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行冻结实体上的规则语义门控实验。")
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CHUNK_MANIFEST)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    asyncio.run(_main(parser.parse_args()))
