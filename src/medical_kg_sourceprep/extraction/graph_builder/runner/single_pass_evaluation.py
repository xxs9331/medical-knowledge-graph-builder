"""EvidenceChunk 到候选图、无监督 Judge 与人工金标评分的单轮主链路。

本模块只执行一次候选图抽取，不使用 Judge 建议进行二次抽取。生成的同一份
``graph.json`` 分别进入两条评测支路：LLM Judge 在不知道金标的情况下审查候选
语义，确定性评分函数则在所有模型调用结束后与人工金标比较。
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...artifacts import sha256_path
from ...llm_extraction import EvidenceChunk, atomic_write_json, load_chunk_manifest
from ..contract import (
    DEFAULT_CHUNK_MANIFEST,
    DEFAULT_SCHEMA_PATH,
    GraphBuilderConfigurationError,
)
from ..evaluation.artifacts import (
    artifact_matches_graph,
    first_extraction_is_usable,
    load_json_object,
)
from ..evaluation.scoring import merge_candidate_graphs, score_candidate_graph
from ..judge import judge_candidate_graph
from ..schema import load_candidate_graph_schema
from .candidate_graph import run_candidate_graph
from .score_aggregation import aggregate_case_scores


def aggregate_judge_results(
    judge_documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """汇总所有 chunk 的无监督 Judge 判定，不把判定分布冒充金标准确率。"""
    # 固定保留全部合法 verdict，即使某类数量为零，报告结构也不会随本次结果变化。
    verdicts = ("SUPPORTED", "UNSUPPORTED", "REPAIR", "ABSTAIN")
    # 每个候选只属于一个 chunk Judge 工件，因此直接跨工件累计不会重复计算候选。
    counts = {
        verdict: sum(
            1
            for document in judge_documents
            for item in document.get("results", [])
            if isinstance(item, Mapping) and item.get("verdict") == verdict
        )
        for verdict in verdicts
    }
    total = sum(counts.values())
    # rates 只描述 Judge 自己的判定分布。没有人工标签参与，不能称为准确率或召回率。
    return {
        "reviewed_candidates": total,
        "counts": counts,
        "rates": {
            verdict: round(count / total, 6) if total else 0.0
            for verdict, count in counts.items()
        },
        "note": "Judge 判定分布是无监督质量审查结果，不等同于人工金标准确率。",
    }


async def run_evaluation_chunk(
    client: Any,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    manifest_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    """将一个真实 chunk 抽取为候选图，并对同一候选图执行无监督 Judge。"""
    # chunk_id 末两段足以在本实验目录中形成稳定且可读的子目录名。
    slug = "-".join(chunk.chunk_id.rsplit(":", 2)[-2:])
    graph_dir = output_root / "candidate-graph"
    graph_path = graph_dir / "graph.json"
    judge_path = output_root / "judge-result.json"

    # 已存在且四个抽取阶段都成功的候选图可以断点复用；不完整工件必须重新抽取。
    if not first_extraction_is_usable(graph_dir):
        await run_candidate_graph(
            client,
            chunk=chunk,
            schema=schema,
            output_dir=graph_dir,
            source_manifest_sha256=manifest_sha256,
            run_id=f"{slug}-evaluation",
        )
    # 第二次检查针对刚生成的工件，防止模型阶段失败后继续拿空图做 Judge 和评分。
    if not first_extraction_is_usable(graph_dir):
        raise GraphBuilderConfigurationError(f"evaluation_extraction_unusable:{chunk.chunk_id}")

    graph = load_json_object(graph_path)
    # Judge 工件必须绑定 graph.json 的精确字节哈希。候选图发生任何变化时，旧 Judge
    # 结果都不能复用，必须针对新图重新审查。
    graph_sha256 = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    judge = artifact_matches_graph(
        judge_path, graph_sha256, hash_path=("input", "graph_sha256")
    )
    if judge is None:
        judge = await judge_candidate_graph(
            client,
            graph_path=graph_path,
            chunks=[chunk],
            schema=schema,
            output_path=judge_path,
            case_id=f"EVALUATION:{chunk.chunk_id}",
        )

    # 返回内存数据供案例级流程组合，同时保留磁盘路径供最终报告追溯。
    return {
        "chunk_id": chunk.chunk_id,
        "source_text": chunk.text,
        "graph": graph,
        "judge": judge,
        "artifacts": {"graph": str(graph_path), "judge_result": str(judge_path)},
    }


async def run_typical_cases_evaluation(
    client: Any,
    *,
    gold_path: Path,
    output_root: Path,
    chunk_manifest_path: Path = DEFAULT_CHUNK_MANIFEST,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    case_ids: set[str] | None = None,
    report_filename: str = "evaluation-result.json",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """运行 chunk 到候选图、无监督 Judge 和人工金标评分的单轮主链路。"""
    # 金标文件在模型调用前只用于选择 case 和定位其 chunk。case 中的实体、关系、规则
    # 答案不会传入 run_evaluation_chunk()、run_candidate_graph() 或 Judge。
    dataset = load_json_object(gold_path)
    if dataset.get("status") != "HUMAN_VALIDATED":
        raise GraphBuilderConfigurationError("evaluation_gold_is_not_human_validated")
    cases = [
        item for item in dataset.get("cases", [])
        if isinstance(item, dict) and (case_ids is None or item.get("case_id") in case_ids)
    ]
    if not cases:
        raise GraphBuilderConfigurationError("evaluation_cases_missing")

    # manifest 是规范原文入口。测试案例引用的每个 chunk_id 都必须能在其中找到。
    _manifest, chunks = load_chunk_manifest(chunk_manifest_path)
    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    # 一个 chunk 可能同时服务多个案例，例如同一段原文可以测试不同抽取目标。
    # 先按 chunk 去重，避免对相同原文重复支付抽取和 Judge 调用成本。
    case_ids_by_chunk: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        for chunk_id in case.get("chunk_ids", []):
            if not isinstance(chunk_id, str) or chunk_id not in chunk_lookup:
                raise GraphBuilderConfigurationError(f"evaluation_chunk_missing:{chunk_id}")
            case_ids_by_chunk[chunk_id].append(str(case["case_id"]))

    # 所有 chunk 使用同一版本 Schema 和同一份 manifest 哈希，保证候选合同一致。
    schema = load_candidate_graph_schema(schema_path)
    manifest_sha256 = sha256_path(chunk_manifest_path)
    chunk_results: dict[str, dict[str, Any]] = {}
    # 第一阶段：逐个唯一 chunk 生成候选图，并立即对该图执行无监督 Judge。
    for index, chunk_id in enumerate(case_ids_by_chunk, start=1):
        if progress is not None:
            progress(
                f"[{index}/{len(case_ids_by_chunk)}] {chunk_id} "
                f"cases={','.join(case_ids_by_chunk[chunk_id])}"
            )
        chunk_output = output_root / "chunks" / "-".join(chunk_id.rsplit(":", 2)[-2:])
        chunk_results[chunk_id] = await run_evaluation_chunk(
            client,
            chunk=chunk_lookup[chunk_id],
            schema=schema,
            manifest_sha256=manifest_sha256,
            output_root=chunk_output,
        )

    # 第二阶段：模型调用至此已经全部完成。人工答案只从这里开始用于确定性评分，
    # 从执行顺序上保证抽取模型和 Judge 都无法看到金标内容。
    case_results: list[dict[str, Any]] = []
    for case in cases:
        # 一个案例可以跨多个 chunk。先按稳定候选键合并相关图，再对完整案例评分。
        selected = [chunk_results[chunk_id] for chunk_id in case["chunk_ids"]]
        score = score_candidate_graph(
            merge_candidate_graphs(item["graph"] for item in selected),
            case,
            source_text="\n\n".join(item["source_text"] for item in selected),
        )
        case_results.append({
            "case_id": case["case_id"],
            "chunk_ids": case["chunk_ids"],
            "score": score,
        })

    # 最终报告并列保存无监督审查和有监督评分，不把两个来源不同的指标混成一个分数。
    # 本报告只用于评测，候选图和 Judge 的 SUPPORTED 都不会改变 HOLD 发布状态。
    report = {
        "schema_version": "candidate-graph-evaluation/v0.1",
        "status": "evaluation-only",
        "publication_status": "HOLD",
        "gold_status": "HUMAN_VALIDATED",
        "gold_exposed_to_models": False,
        "case_count": len(case_results),
        "unique_chunk_count": len(chunk_results),
        "unsupervised_judge": aggregate_judge_results([
            item["judge"] for item in chunk_results.values()
        ]),
        "supervised_gold": aggregate_case_scores(case_results, "score"),
        "cases": case_results,
        "chunk_artifacts": {
            chunk_id: item["artifacts"] for chunk_id, item in chunk_results.items()
        },
    }
    # 原子写入避免中断时留下半个 JSON 报告。
    atomic_write_json(output_root / report_filename, report)
    return report


def evaluation_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """生成单轮主链路适合命令行展示的摘要。"""
    # 摘要省略候选明细和工件路径；完整可审计信息仍保留在 evaluation-result.json。
    return {
        "case_count": report["case_count"],
        "unique_chunk_count": report["unique_chunk_count"],
        "unsupervised_judge": report["unsupervised_judge"],
        "supervised_gold": report["supervised_gold"],
        "case_scores": [{
            "case_id": item["case_id"],
            "score_percent": item["score"]["challenge"]["score_percent"],
        } for item in report["cases"]],
    }
