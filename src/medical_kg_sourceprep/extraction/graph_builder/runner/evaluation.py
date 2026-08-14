"""EvidenceChunk 到候选图、无监督 Judge 与人工金标评分的单轮主链路。"""

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
from .common import aggregate_case_scores
from .extraction import run_candidate_graph


def aggregate_judge_results(
    judge_documents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """汇总所有 chunk 的无监督 Judge 判定，不把判定分布冒充金标准确率。"""
    verdicts = ("SUPPORTED", "UNSUPPORTED", "REPAIR", "ABSTAIN")
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
    slug = "-".join(chunk.chunk_id.rsplit(":", 2)[-2:])
    graph_dir = output_root / "candidate-graph"
    graph_path = graph_dir / "graph.json"
    judge_path = output_root / "judge-result.json"

    if not first_extraction_is_usable(graph_dir):
        await run_candidate_graph(
            client,
            chunk=chunk,
            schema=schema,
            output_dir=graph_dir,
            source_manifest_sha256=manifest_sha256,
            run_id=f"{slug}-evaluation",
        )
    if not first_extraction_is_usable(graph_dir):
        raise GraphBuilderConfigurationError(f"evaluation_extraction_unusable:{chunk.chunk_id}")

    graph = load_json_object(graph_path)
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
    dataset = load_json_object(gold_path)
    if dataset.get("status") != "HUMAN_VALIDATED":
        raise GraphBuilderConfigurationError("evaluation_gold_is_not_human_validated")
    cases = [
        item for item in dataset.get("cases", [])
        if isinstance(item, dict) and (case_ids is None or item.get("case_id") in case_ids)
    ]
    if not cases:
        raise GraphBuilderConfigurationError("evaluation_cases_missing")

    _manifest, chunks = load_chunk_manifest(chunk_manifest_path)
    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    case_ids_by_chunk: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        for chunk_id in case.get("chunk_ids", []):
            if not isinstance(chunk_id, str) or chunk_id not in chunk_lookup:
                raise GraphBuilderConfigurationError(f"evaluation_chunk_missing:{chunk_id}")
            case_ids_by_chunk[chunk_id].append(str(case["case_id"]))

    schema = load_candidate_graph_schema(schema_path)
    manifest_sha256 = sha256_path(chunk_manifest_path)
    chunk_results: dict[str, dict[str, Any]] = {}
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

    # 模型调用至此已经全部完成。人工答案只从这里开始用于确定性评分。
    case_results: list[dict[str, Any]] = []
    for case in cases:
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
    atomic_write_json(output_root / report_filename, report)
    return report


def evaluation_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """生成单轮主链路适合命令行展示的摘要。"""
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
