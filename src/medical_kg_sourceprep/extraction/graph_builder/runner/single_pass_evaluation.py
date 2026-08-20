"""EvidenceChunk 到候选图和人工金标评分的单轮主链路。

本模块只执行一次候选图抽取，不使用 Judge 建议进行二次抽取。生成的同一份
``graph.json`` 进入确定性金标评分；调用方也可以选择额外运行无监督 Judge，
但 Judge 不是主链路必需步骤，也不会改变候选图。
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from neo4j_graphrag.exceptions import LLMGenerationError

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
from ..candidate_graph import extract_cross_chunk_relationships, run_candidate_graph
from ..evaluation.aggregation import aggregate_case_scores, aggregate_supervised_prf1
from ..evaluation.scoring import merge_candidate_graphs, score_candidate_graph
from ..judge import judge_candidate_graph
from ..schema import load_candidate_graph_schema
from ..trace import JsonlTrace, NULL_TRACE, TraceRecorder


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


def _cross_chunk_failure_graph(
    nodes: Sequence[Mapping[str, Any]], error: LLMGenerationError
) -> dict[str, Any]:
    """把跨 chunk 模型格式失败保存为可审计空候选，不中断其他 case。"""
    return {
        "schema_version": "candidate-cross-chunk-graph/v0.1",
        "status": "candidate-only",
        "publication_status": "HOLD",
        "approved": 0,
        "extraction_status": "FAILED",
        "nodes": [dict(item) for item in nodes],
        "relationships": [],
        "review_items": [{
            "stage": "cross_chunk_relation",
            "status": "REVIEW_REQUIRED",
            "reason_code": "cross_chunk_relation_phase_model_response_invalid",
            "error_type": type(error).__name__,
        }],
        "judge_drafts": [],
    }
async def run_evaluation_chunk(
    client: Any,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    manifest_sha256: str,
    output_root: Path,
    relation_extraction_mode: str = "generative",
    run_judge: bool = True,
    trace: TraceRecorder = NULL_TRACE,
) -> dict[str, Any]:
    """将一个真实 chunk 抽取为候选图，并按配置选择是否执行 Judge。"""
    trace.record("chunk/start", chunk_id=chunk.chunk_id)
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
            relation_extraction_mode=relation_extraction_mode,
            trace=trace,
        )
    else:
        trace.record(
            "candidate-graph/reused",
            chunk_id=chunk.chunk_id,
            graph_path=graph_path,
        )
    # 单个 chunk 的模型格式失败不能中断整章评测。失败图保留 review-queue 作为证据，
    # 并以空候选进入监督评分；报告显式记录 FAILED，不能把它冒充成功的空抽取。
    extraction_usable = first_extraction_is_usable(graph_dir)
    graph = load_json_object(graph_path)
    judge: Mapping[str, Any] | None = None
    if run_judge and extraction_usable:
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
                trace=trace,
            )
        else:
            trace.record(
                "judge/reused",
                chunk_id=chunk.chunk_id,
                output_path=judge_path,
                graph_sha256=graph_sha256,
            )
    else:
        trace.record(
            "judge/skipped",
            chunk_id=chunk.chunk_id,
            reason=("disabled_by_workflow" if not run_judge else "extraction_failed"),
        )

    # 返回内存数据供案例级流程组合，同时保留磁盘路径供最终报告追溯。
    result = {
        "chunk_id": chunk.chunk_id,
        "source_text": chunk.text,
        "graph": graph,
        "judge": judge,
        "extraction_status": "SUCCESS" if extraction_usable else "FAILED",
        "artifacts": {
            "graph": str(graph_path),
            **({"judge_result": str(judge_path)} if run_judge else {}),
        },
    }
    trace.record(
        "chunk/end",
        chunk_id=chunk.chunk_id,
        status="success",
        graph_path=graph_path,
        judge_path=judge_path if run_judge else None,
    )
    return result


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
    trace: TraceRecorder | None = None,
    relation_extraction_mode: str = "generative",
    allow_review_required_gold: bool = False,
    run_judge: bool = True,
) -> dict[str, Any]:
    """运行 chunk 到候选图和人工金标评分，可选无监督 Judge。"""
    # 默认每次运行创建独立事件文件，避免断点复跑时把两个 run 的 seq 混入同一账本。
    created_trace: JsonlTrace | None = None
    if trace is None:
        trace_run_id = str(uuid.uuid4())
        created_trace = JsonlTrace(
            output_root / "trace" / f"{trace_run_id}.jsonl",
            run_id=trace_run_id,
        )
        effective_trace: TraceRecorder = created_trace
    else:
        effective_trace = trace
    effective_trace.record(
        "run/start",
        workflow="single_pass_evaluation",
        gold_path=gold_path,
        chunk_manifest_path=chunk_manifest_path,
        schema_path=schema_path,
        relation_extraction_mode=relation_extraction_mode,
        run_judge=run_judge,
    )

    # 金标文件在模型调用前只用于选择 case 和定位其 chunk。case 中的实体、关系、规则
    # 答案不会传入 run_evaluation_chunk()、run_candidate_graph() 或 Judge。
    dataset = load_json_object(gold_path)
    if dataset.get("status") != "HUMAN_VALIDATED" and not (
        allow_review_required_gold and dataset.get("status") == "HUMAN_REVIEW_REQUIRED"
    ):
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
    # 第一阶段：逐个唯一 chunk 生成候选图；Judge 是显式可选支路。
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
            relation_extraction_mode=relation_extraction_mode,
            run_judge=run_judge,
            trace=effective_trace,
        )

    # 多 chunk 案例增加一次案例级普通关系抽取。这里只使用 case_id 及其 chunk 范围来
    # 组织输入，不读取该案例的实体、关系或规则金标。输出证据逐项绑定真实 chunk。
    cross_chunk_results: dict[str, dict[str, Any]] = {}
    for case in cases:
        chunk_ids = [str(item) for item in case.get("chunk_ids", [])]
        if len(chunk_ids) < 2:
            continue
        case_id = str(case["case_id"])
        case_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", case_id).strip("-")
        selected = [chunk_results[chunk_id] for chunk_id in chunk_ids]
        merged = merge_candidate_graphs(item["graph"] for item in selected)
        case_output = output_root / "cases" / case_slug
        graph_path = case_output / "cross-chunk-graph.json"
        judge_path = case_output / "cross-chunk-judge.json"
        if graph_path.is_file():
            graph_document = load_json_object(graph_path)
            effective_trace.record(
                "cross-chunk/reused",
                case_id=case_id,
                graph_path=graph_path,
                extraction_status=graph_document.get("extraction_status", "SUCCESS"),
            )
        else:
            try:
                cross_result = await extract_cross_chunk_relationships(
                    client,
                    chunks=[chunk_lookup[chunk_id] for chunk_id in chunk_ids],
                    schema=schema,
                    nodes=[item for item in merged["nodes"] if isinstance(item, Mapping)],
                    trace=effective_trace,
                )
                graph_document = {
                    "schema_version": "candidate-cross-chunk-graph/v0.1",
                    "status": "candidate-only",
                    "publication_status": "HOLD",
                    "approved": 0,
                    "extraction_status": "SUCCESS",
                    # 节点只作为 Judge 的端点上下文，新增候选只有 relationships。
                    "nodes": merged["nodes"],
                    "relationships": cross_result.accepted,
                    "review_items": cross_result.review_items,
                    "judge_drafts": cross_result.judge_drafts,
                }
            except LLMGenerationError as error:
                graph_document = _cross_chunk_failure_graph(
                    [item for item in merged["nodes"] if isinstance(item, Mapping)], error
                )
                effective_trace.record(
                    "cross-chunk/failed",
                    case_id=case_id,
                    error_type=type(error).__name__,
                    reason_code="cross_chunk_relation_phase_model_response_invalid",
                )
            atomic_write_json(graph_path, graph_document)
        cross_judge: Mapping[str, Any] | None = None
        if run_judge and graph_document.get("extraction_status") != "FAILED":
            cross_judge = await judge_candidate_graph(
                client,
                graph_path=graph_path,
                chunks=[chunk_lookup[chunk_id] for chunk_id in chunk_ids],
                schema=schema,
                output_path=judge_path,
                case_id=f"{case_id}:CROSS_CHUNK",
                trace=effective_trace,
                item_kinds={"relationship"},
            )
        else:
            effective_trace.record(
                "judge/skipped",
                case_id=case_id,
                stage="cross_chunk",
                reason="disabled_by_workflow",
            )
        cross_chunk_results[case_id] = {
            "graph": graph_document,
            "judge": cross_judge,
            "extraction_status": graph_document.get("extraction_status", "SUCCESS"),
            "artifacts": {
                "graph": str(graph_path),
                **({"judge_result": str(judge_path)} if run_judge else {}),
            },
        }

    # 第二阶段：模型调用至此已经全部完成。人工答案只从这里开始用于确定性评分，
    # 从执行顺序上保证抽取模型和 Judge 都无法看到金标内容。
    case_results: list[dict[str, Any]] = []
    for case in cases:
        # 一个案例可以跨多个 chunk。先按稳定候选键合并相关图，再对完整案例评分。
        selected = [chunk_results[chunk_id] for chunk_id in case["chunk_ids"]]
        candidate_graphs = [item["graph"] for item in selected]
        cross_chunk = cross_chunk_results.get(str(case["case_id"]))
        if cross_chunk is not None:
            candidate_graphs.append(cross_chunk["graph"])
        score = score_candidate_graph(
            merge_candidate_graphs(candidate_graphs),
            case,
            source_text="\n\n".join(item["source_text"] for item in selected),
        )
        # 金标只在模型调用全部完成后进入本事件；Trace 只保存指标，不复制目标明细。
        effective_trace.record(
            "scoring/case",
            case_id=case["case_id"],
            chunk_ids=case["chunk_ids"],
            score_percent=score["challenge"]["score_percent"],
            satisfied_constraints=score["challenge"]["satisfied_constraints"],
            total_constraints=score["challenge"]["total_constraints"],
            entity_matched=score["entities"]["matched"],
            entity_total=score["entities"]["target_total"],
            relationship_matched=score["relationships"]["matched"],
            relationship_total=score["relationships"]["target_total"],
            rule_matched=score["rules"]["matched"],
            rule_total=score["rules"]["target_total"],
            forbidden_violations=score["forbidden"]["violations"],
            forbidden_total=score["forbidden"]["target_total"],
        )
        case_results.append({
            "case_id": case["case_id"],
            "chunk_ids": case["chunk_ids"],
            "score": score,
            **({"cross_chunk_artifacts": cross_chunk["artifacts"]} if cross_chunk is not None else {}),
        })

    # 最终报告并列保存无监督审查和有监督评分，不把两个来源不同的指标混成一个分数。
    # 本报告只用于评测，候选图和 Judge 的 SUPPORTED 都不会改变 HOLD 发布状态。
    judge_summary = (
        aggregate_judge_results([
            judge
            for item in (*chunk_results.values(), *cross_chunk_results.values())
            if isinstance((judge := item.get("judge")), Mapping)
        ])
        if run_judge
        else {
            "status": "not_run",
            "reviewed_candidates": 0,
            "counts": {},
            "rates": {},
            "note": "本工作流未调用 Judge；候选直接进入本地金标评分。",
        }
    )
    supervised_gold = aggregate_case_scores(case_results, "score")
    prf1 = aggregate_supervised_prf1(case_results, "score")
    effective_trace.record(
        "scoring/aggregate",
        case_count=len(case_results),
        micro_score_percent=supervised_gold["micro"]["score_percent"],
        macro_score_percent=supervised_gold["macro"]["score_percent"],
        categories=supervised_gold["categories"],
        precision=prf1["precision"],
        recall=prf1["recall"],
        f1=prf1["f1"],
    )
    report = {
        "schema_version": "candidate-graph-evaluation/v0.2",
        "status": "evaluation-only",
        "publication_status": "HOLD",
        "gold_status": dataset.get("status"),
        "gold_annotation_method": dataset.get("annotation_method"),
        "gold_scope_contract": dataset.get("scope_contract"),
        "gold_exposed_to_models": False,
        "configuration": {
            "relation_extraction_mode": relation_extraction_mode,
            "cross_chunk_relation_mode": "generative",
            "judge": "enabled" if run_judge else "not_run",
        },
        "case_count": len(case_results),
        "unique_chunk_count": len(chunk_results),
        "failed_chunk_ids": [
            chunk_id for chunk_id, item in chunk_results.items()
            if item["extraction_status"] == "FAILED"
        ],
        "unsupervised_judge": judge_summary,
        "supervised_gold": supervised_gold,
        "prf1": prf1,
        "cases": case_results,
        "chunk_artifacts": {
            chunk_id: item["artifacts"] for chunk_id, item in chunk_results.items()
        },
        "trace": {
            "run_id": effective_trace.run_id,
            **({"events": str(created_trace.path)} if created_trace is not None else {}),
        },
    }
    # 原子写入避免中断时留下半个 JSON 报告。
    atomic_write_json(output_root / report_filename, report)
    effective_trace.record(
        "run/end",
        workflow="single_pass_evaluation",
        status="success",
        case_count=len(case_results),
        unique_chunk_count=len(chunk_results),
        report_path=output_root / report_filename,
    )
    return report


def evaluation_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """生成单轮主链路适合命令行展示的摘要。"""
    # 摘要省略候选明细和工件路径；完整可审计信息仍保留在 evaluation-result.json。
    return {
        "case_count": report["case_count"],
        "unique_chunk_count": report["unique_chunk_count"],
        "failed_chunk_ids": report.get("failed_chunk_ids", []),
        "unsupervised_judge": report["unsupervised_judge"],
        "supervised_gold": report["supervised_gold"],
        "prf1": report["prf1"],
        "case_scores": [{
            "case_id": item["case_id"],
            "score_percent": item["score"]["challenge"]["score_percent"],
        } for item in report["cases"]],
    }
