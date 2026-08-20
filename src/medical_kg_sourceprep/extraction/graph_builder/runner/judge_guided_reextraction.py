"""携带审查反馈的二次抽取、两轮并集与提升效果对比。

本模块负责首次抽取、候选 Judge、遗漏审查、携带反馈的二次抽取、两轮并集和
人工典型案例评分。模型调用阶段只读取原文、Schema、候选图和审查建议；人工金标
只在候选工件全部生成后参与评分。
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...artifacts import sha256_path
from ...llm_extraction import EvidenceChunk, atomic_write_json, load_chunk_manifest
from ..candidate_graph import run_candidate_graph
from ..contract import (
    DEFAULT_CHUNK_MANIFEST,
    DEFAULT_SCHEMA_PATH,
    GraphBuilderConfigurationError,
)
from ..evaluation.artifacts import (
    artifact_matches_graph,
    first_extraction_is_usable,
    load_json_object,
    second_extraction_is_usable,
)
from ..evaluation.aggregation import aggregate_case_scores, aggregate_supervised_prf1
from ..evaluation.coverage import audit_extraction_coverage
from ..evaluation.scoring import merge_candidate_graphs, score_candidate_graph
from ..judge import judge_candidate_graph
from ..schema import load_candidate_graph_schema


def compact_candidate_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    """保留二次抽取所需身份和语义字段，省略原文中已有的冗长证据。"""
    node_fields = {
        "candidate_key", "entity_type", "mention", "rule_stage_candidate",
        "rule_inputs", "rule_outputs", "rule_expression", "rule_name", "extraction_status",
    }
    relation_fields = {
        "candidate_key", "relation_type", "source_candidate_key", "target_candidate_key",
        "extraction_status",
    }
    # 原文和 Schema 会单独放入二次提示词，因此这里不重复携带证据全文和派生字段，
    # 避免上下文过长，也避免模型把旧证据位置误当成新的抽取结论。
    return {
        "nodes": [
            {key: value for key, value in item.items() if key in node_fields}
            for item in graph.get("nodes", []) if isinstance(item, dict)
        ],
        "relationships": [
            {key: value for key, value in item.items() if key in relation_fields}
            for item in graph.get("relationships", []) if isinstance(item, dict)
        ],
    }


def build_revision_context(
    judge: Mapping[str, Any], coverage: Mapping[str, Any], first_graph: Mapping[str, Any]
) -> str:
    """构造二次抽取反馈，不包含金标或冗长的 SUPPORTED 理由。"""
    # SUPPORTED 项不需要模型再次处理；只传入错误、修复和无法判断的项目，降低噪声。
    actionable_results = [
        item for item in judge.get("results", [])
        if isinstance(item, dict) and item.get("verdict") != "SUPPORTED"
    ]
    judge_input = judge.get("input")
    if not isinstance(judge_input, Mapping) or not isinstance(judge_input.get("graph_sha256"), str):
        raise GraphBuilderConfigurationError("judge_input_binding_invalid")
    # 此对象有意不接受 gold_case 参数，从接口层阻止人工答案进入二次抽取提示词。
    return json.dumps({
        "previous_graph_sha256": judge_input["graph_sha256"],
        "first_candidate_graph": compact_candidate_graph(first_graph),
        "judge_counts": judge.get("counts", {}),
        "judge_actionable_results": actionable_results,
        "coverage_missing_items": coverage.get("missing_items", []),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


async def run_reextraction_chunk(
    client: Any,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    manifest_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    """对一个 chunk 执行首次抽取、审查、二次抽取并写出两轮并集。"""
    slug = "-".join(chunk.chunk_id.rsplit(":", 2)[-2:])
    first_dir = output_root / "first-extraction"
    second_dir = output_root / "second-extraction"
    judge_path = output_root / "judge-result.json"
    coverage_path = output_root / "coverage-audit.json"
    state_path = output_root / "experiment-state.json"
    union_path = output_root / "union-graph.json"

    # 第一轮必须四个模型阶段都成功，才能作为 Judge 和二次抽取的稳定基线。
    if not first_extraction_is_usable(first_dir):
        await run_candidate_graph(
            client, chunk=chunk, schema=schema, output_dir=first_dir,
            source_manifest_sha256=manifest_sha256, run_id=f"{slug}-first",
        )
    if not first_extraction_is_usable(first_dir):
        raise GraphBuilderConfigurationError(f"first_extraction_unusable:{chunk.chunk_id}")

    first_graph_path = first_dir / "graph.json"
    first_graph = load_json_object(first_graph_path)
    graph_sha256 = hashlib.sha256(first_graph_path.read_bytes()).hexdigest()
    # 正确性 Judge 检查已有候选；遗漏审查寻找未被抽出的候选，两者职责互补。
    # 两类审查工件均绑定第一轮图哈希，第一轮变化后不会错误复用旧建议。
    judge = artifact_matches_graph(
        judge_path, graph_sha256, hash_path=("input", "graph_sha256")
    )
    if judge is None:
        judge = await judge_candidate_graph(
            client, graph_path=first_graph_path, chunks=[chunk], schema=schema,
            output_path=judge_path, case_id=f"TYPICAL:{chunk.chunk_id}",
        )
    coverage = artifact_matches_graph(
        coverage_path, graph_sha256, hash_path=("input_graph_sha256",)
    )
    if coverage is None:
        coverage = await audit_extraction_coverage(
            client, chunk=chunk, schema=schema, graph_path=first_graph_path,
            output_path=coverage_path,
        )

    # 二次抽取同时接收第一轮精简图、Judge 可执行建议和遗漏建议，但不接收金标。
    revision_context = build_revision_context(judge, coverage, first_graph)
    revision_sha256 = hashlib.sha256(revision_context.encode()).hexdigest()
    state = load_json_object(state_path) if state_path.is_file() else {}
    # 状态文件绑定反馈内容的哈希；反馈变化或实体基础阶段失败时重新执行第二轮。
    if (
        state.get("revision_context_sha256") != revision_sha256
        or not second_extraction_is_usable(second_dir)
    ):
        await run_candidate_graph(
            client, chunk=chunk, schema=schema, output_dir=second_dir,
            source_manifest_sha256=manifest_sha256, run_id=f"{slug}-second",
            revision_context=revision_context,
        )
        if not second_extraction_is_usable(second_dir):
            raise GraphBuilderConfigurationError(f"second_extraction_unusable:{chunk.chunk_id}")
        atomic_write_json(state_path, {
            "schema_version": "judge-reextraction-chunk-state/v0.1",
            "chunk_id": chunk.chunk_id,
            "first_graph_sha256": graph_sha256,
            "revision_context_sha256": revision_sha256,
        })

    second_graph = load_json_object(second_dir / "graph.json")
    # 最终实验结果采用稳定 candidate_key 去重后的集合并集。并集保留两轮候选，
    # 不代表这些候选已获批准，后续仍需独立 Judge 或人工审核。
    union_graph = {
        "schema_version": "candidate-graph-union/v0.1",
        "status": "candidate-only",
        "publication_status": "HOLD",
        "approved": 0,
        "first_graph_sha256": graph_sha256,
        "revision_context_sha256": revision_sha256,
        **merge_candidate_graphs([first_graph, second_graph]),
    }
    atomic_write_json(union_path, union_graph)
    return {
        "chunk_id": chunk.chunk_id,
        "source_text": chunk.text,
        "first_graph": first_graph,
        "second_graph": second_graph,
        "union_graph": union_graph,
        "artifacts": {
            "first_graph": str(first_graph_path),
            "judge_result": str(judge_path),
            "coverage_audit": str(coverage_path),
            "second_graph": str(second_dir / "graph.json"),
            "union_graph": str(union_path),
        },
    }


async def run_typical_cases_experiment(
    client: Any,
    *,
    gold_path: Path,
    output_root: Path,
    chunk_manifest_path: Path = DEFAULT_CHUNK_MANIFEST,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    case_ids: set[str] | None = None,
    chunk_output_overrides: Mapping[str, Path] | None = None,
    comparison_filename: str = "comparison-all.json",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """运行人工典型案例实验；金标只在全部候选生成后用于评分。"""
    # 此处先读取金标仅用于确定案例及其 chunk 范围。下面所有模型调用只接收 chunk、
    # Schema、候选图和审查建议；case 的实体、关系、规则答案不会传给模型函数。
    dataset = load_json_object(gold_path)
    if dataset.get("status") != "HUMAN_VALIDATED":
        raise GraphBuilderConfigurationError("experiment_gold_is_not_human_validated")
    cases = [
        item for item in dataset.get("cases", [])
        if isinstance(item, dict) and (case_ids is None or item.get("case_id") in case_ids)
    ]
    if not cases:
        raise GraphBuilderConfigurationError("experiment_cases_missing")

    _manifest, chunks = load_chunk_manifest(chunk_manifest_path)
    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    # 多个案例可能复用同一个 chunk。按 chunk 去重执行昂贵的模型流程，再按案例组合评分。
    case_ids_by_chunk: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        for chunk_id in case.get("chunk_ids", []):
            if not isinstance(chunk_id, str) or chunk_id not in chunk_lookup:
                raise GraphBuilderConfigurationError(f"experiment_chunk_missing:{chunk_id}")
            case_ids_by_chunk[chunk_id].append(str(case["case_id"]))

    schema = load_candidate_graph_schema(schema_path)
    manifest_sha256 = sha256_path(chunk_manifest_path)
    output_overrides = chunk_output_overrides or {}
    chunk_results: dict[str, dict[str, Any]] = {}
    for index, chunk_id in enumerate(case_ids_by_chunk, start=1):
        if progress is not None:
            progress(
                f"[{index}/{len(case_ids_by_chunk)}] {chunk_id} "
                f"cases={','.join(case_ids_by_chunk[chunk_id])}"
            )
        chunk_output = output_overrides.get(
            chunk_id, output_root / "chunks" / "-".join(chunk_id.rsplit(":", 2)[-2:])
        )
        chunk_results[chunk_id] = await run_reextraction_chunk(
            client, chunk=chunk_lookup[chunk_id], schema=schema,
            manifest_sha256=manifest_sha256, output_root=chunk_output,
        )

    # 数据隔离线：只有所有候选工件生成完成后，才在这里使用人工答案计算分数。
    case_results: list[dict[str, Any]] = []
    for case in cases:
        selected = [chunk_results[chunk_id] for chunk_id in case["chunk_ids"]]
        source_text = "\n\n".join(item["source_text"] for item in selected)
        # 分别保留首次、二次单轮和两轮并集分数，既能观察反馈收益，也能发现退化。
        phase_scores = {
            phase: score_candidate_graph(
                merge_candidate_graphs(item[f"{phase}_graph"] for item in selected),
                case,
                source_text=source_text,
            )
            for phase in ("first", "second", "union")
        }
        first_score = phase_scores["first"]["challenge"]["score"]
        case_results.append({
            "case_id": case["case_id"],
            "chunk_ids": case["chunk_ids"],
            **phase_scores,
            "delta": round(phase_scores["second"]["challenge"]["score"] - first_score, 6),
            "union_delta": round(phase_scores["union"]["challenge"]["score"] - first_score, 6),
        })

    aggregates = {
        phase: {
            **aggregate_case_scores(case_results, phase),
            "prf1": aggregate_supervised_prf1(case_results, phase),
        }
        for phase in ("first", "second", "union")
    }
    result = {
        "schema_version": "judge-reextraction-typical-cases/v0.2",
        "gold_status": "HUMAN_VALIDATED",
        "gold_exposed_to_models": False,
        "case_count": len(case_results),
        "unique_chunk_count": len(chunk_results),
        **aggregates,
        "delta": {
            "micro_score": round(
                aggregates["second"]["micro"]["score"]
                - aggregates["first"]["micro"]["score"], 6
            ),
            "macro_score": round(
                aggregates["second"]["macro"]["score"]
                - aggregates["first"]["macro"]["score"], 6
            ),
            "union_micro_score": round(
                aggregates["union"]["micro"]["score"]
                - aggregates["first"]["micro"]["score"], 6
            ),
            "union_macro_score": round(
                aggregates["union"]["macro"]["score"]
                - aggregates["first"]["macro"]["score"], 6
            ),
        },
        "cases": case_results,
        "chunk_artifacts": {
            chunk_id: item["artifacts"] for chunk_id, item in chunk_results.items()
        },
    }
    atomic_write_json(output_root / comparison_filename, result)
    return result


def comparison_summary(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """裁剪批量结果为命令行适合展示的摘要。"""
    return {
        "case_count": comparison["case_count"],
        "unique_chunk_count": comparison["unique_chunk_count"],
        "first": comparison["first"],
        "second": comparison["second"],
        "union": comparison["union"],
        "delta": comparison["delta"],
        "case_scores": [{
            "case_id": item["case_id"],
            "first": item["first"]["challenge"]["score_percent"],
            "second": item["second"]["challenge"]["score_percent"],
            "union": item["union"]["challenge"]["score_percent"],
            "delta": round(item["delta"] * 100, 2),
            "union_delta": round(item["union_delta"] * 100, 2),
        } for item in comparison["cases"]],
    }
