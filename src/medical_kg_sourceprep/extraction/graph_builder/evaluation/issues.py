"""把确定性评测结果整理成可供人工和 Sol 使用的问题工件。

本模块只做问题归并，不调用模型、不修改候选图，也不把问题自动解释成修复方案。
每个问题保留运行、来源和评分工件引用，后续优化模块可以据此构造失败证据包。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _severity_for_count(count: int, *, critical: bool = False) -> str:
    """按错误规模给出稳定的诊断严重度；不代表医学风险等级。"""
    if critical:
        return "critical"
    if count >= 20:
        return "high"
    if count >= 5:
        return "medium"
    return "low"


def _issue(
    *, issue_id: str, run_id: str, stage: str, category: str, severity: str,
    summary: str, affected_ids: list[str], evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "issue_id": issue_id,
        "run_id": run_id,
        "severity": severity,
        "stage": stage,
        "category": category,
        "affected_ids": affected_ids,
        "source_chunks": list(evidence.get("source_chunks", [])),
        "deterministic_evidence": dict(evidence),
        "metric_evidence": {},
        "deepeval_scores": {},
        "deepeval_reasons": [],
        "first_divergence_hint": stage,
        "cluster_id": f"cluster:{category}",
        "status": "OPEN",
        "summary": summary,
    }


def build_evaluation_issues(
    *, score: Mapping[str, Any], run_manifest: Mapping[str, Any] | None = None,
    review_queue: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """从评分、运行清单和硬校验队列生成版本化问题列表。"""
    run_id = str(
        (run_manifest or {}).get("run_id")
        or (run_manifest or {}).get("output_root")
        or score.get("prediction_path")
        or "unknown-run"
    )
    issues: list[dict[str, Any]] = []
    by_type = score.get("by_relation_type", {})
    if isinstance(by_type, Mapping):
        for relation_type, metrics in by_type.items():
            if not isinstance(metrics, Mapping):
                continue
            fn = int(metrics.get("fn", 0))
            fp = int(metrics.get("fp", 0))
            if fn:
                issues.append(_issue(
                    issue_id=f"issue:{run_id}:relation-recall:{relation_type}",
                    run_id=run_id, stage="evaluation", category="relation_recall_gap",
                    severity=_severity_for_count(fn),
                    summary=f"关系类型 {relation_type} 漏抽 {fn} 条，召回率为 {metrics.get('recall_percent', 0)}%。",
                    affected_ids=[str(relation_type)],
                    evidence={"relation_type": relation_type, "fn": fn, "recall": metrics.get("recall")},
                ))
            if fp:
                issues.append(_issue(
                    issue_id=f"issue:{run_id}:relation-precision:{relation_type}",
                    run_id=run_id, stage="evaluation", category="relation_over_generation",
                    severity=_severity_for_count(fp),
                    summary=f"关系类型 {relation_type} 误抽 {fp} 条，精确率为 {metrics.get('precision_percent', 0)}%。",
                    affected_ids=[str(relation_type)],
                    evidence={"relation_type": relation_type, "fp": fp, "precision": metrics.get("precision")},
                ))

    diagnostics = score.get("false_positive_diagnostics", {})
    if isinstance(diagnostics, Mapping):
        for category, count_value in diagnostics.items():
            count = int(count_value)
            if count <= 0:
                continue
            issues.append(_issue(
                issue_id=f"issue:{run_id}:diagnostic:{category}",
                run_id=run_id, stage="evaluation", category=str(category).lower(),
                severity=_severity_for_count(count),
                summary=f"确定性评测发现 {category} {count} 条。",
                affected_ids=[], evidence={"count": count, "diagnostic": category},
            ))

    if isinstance(review_queue, Mapping):
        items = review_queue.get("items", [])
        if isinstance(items, list):
            grouped: dict[str, list[Mapping[str, Any]]] = {}
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                code = str(item.get("reason_code", "UNKNOWN_REVIEW"))
                grouped.setdefault(code, []).append(item)
            for code, grouped_items in grouped.items():
                issues.append(_issue(
                    issue_id=f"issue:{run_id}:validation:{code}",
                    run_id=run_id, stage="validation", category="deterministic_validation",
                    severity=_severity_for_count(len(grouped_items), critical="invalid" in code.lower()),
                    summary=f"硬校验队列包含 {len(grouped_items)} 条 {code}。",
                    affected_ids=[str(item.get("candidate_key", "")) for item in grouped_items],
                    evidence={"reason_code": code, "count": len(grouped_items)},
                ))

    issues.sort(key=lambda item: (_SEVERITY_ORDER.get(item["severity"], 99), item["issue_id"]))
    counts: dict[str, int] = {}
    for item in issues:
        counts[item["severity"]] = counts.get(item["severity"], 0) + 1
    return {
        "schema_version": "graph-builder-evaluation-issues/v0.1",
        "status": "OPEN" if issues else "NO_ISSUES",
        "publication_status": "HOLD",
        "run_id": run_id,
        "issue_count": len(issues),
        "counts_by_severity": dict(sorted(counts.items())),
        "source_score_schema": score.get("schema_version"),
        "issues": issues,
        "boundary": "确定性问题归并；不调用 DeepEval，不修改候选图，不自动生成补丁。",
    }


def build_issues_from_paths(
    *, score_path: Path, manifest_path: Path | None = None,
    review_queue_path: Path | None = None,
) -> dict[str, Any]:
    """从 JSON 工件路径构建问题报告，供 CLI 和批处理 runner 复用。"""
    import json

    score = json.loads(score_path.read_text(encoding="utf-8"))
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path and manifest_path.is_file() else None
    )
    review_queue = (
        json.loads(review_queue_path.read_text(encoding="utf-8"))
        if review_queue_path and review_queue_path.is_file() else None
    )
    return build_evaluation_issues(score=score, run_manifest=manifest, review_queue=review_queue)
