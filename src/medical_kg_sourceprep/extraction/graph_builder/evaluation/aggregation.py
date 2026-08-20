"""单轮评测与二次抽取实验共用的确定性汇总函数。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contract import GraphBuilderConfigurationError


def aggregate_case_scores(
    case_results: Sequence[Mapping[str, Any]], phase: str
) -> dict[str, Any]:
    """汇总一个评分阶段的 micro、macro 和分类型目标覆盖率。"""
    if not case_results:
        raise GraphBuilderConfigurationError("experiment_cases_missing")
    # micro 按全部约束汇总，约束较多的案例权重更大；macro 对每个案例等权平均。
    satisfied = sum(item[phase]["challenge"]["satisfied_constraints"] for item in case_results)
    total = sum(item[phase]["challenge"]["total_constraints"] for item in case_results)
    macro = sum(item[phase]["challenge"]["score"] for item in case_results) / len(case_results)
    categories: dict[str, Any] = {}
    for category in ("entities", "relationships", "rules"):
        matched = sum(item[phase][category]["matched"] for item in case_results)
        target_total = sum(item[phase][category]["target_total"] for item in case_results)
        categories[category] = {
            "matched": matched,
            "target_total": target_total,
            "coverage": round(matched / target_total, 6) if target_total else 1.0,
        }
    forbidden_total = sum(item[phase]["forbidden"]["target_total"] for item in case_results)
    forbidden_violations = sum(item[phase]["forbidden"]["violations"] for item in case_results)
    categories["forbidden"] = {
        "violations": forbidden_violations,
        "target_total": forbidden_total,
        "avoidance": round((forbidden_total - forbidden_violations) / forbidden_total, 6)
        if forbidden_total else 1.0,
    }
    return {
        "micro": {
            "satisfied_constraints": satisfied,
            "total_constraints": total,
            "score": round(satisfied / total, 6) if total else 1.0,
            "score_percent": round(100 * satisfied / total, 2) if total else 100.0,
        },
        "macro": {
            "case_count": len(case_results),
            "score": round(macro, 6),
            "score_percent": round(100 * macro, 2),
        },
        "categories": categories,
    }


def _prf1_from_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """把跨案例累加后的 TP/FP/FN 转为标准监督指标。"""
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
        "precision_percent": round(100 * precision, 2),
        "recall_percent": round(100 * recall, 2),
        "f1_percent": round(100 * f1, 2),
    }


def aggregate_supervised_prf1(
    case_results: Sequence[Mapping[str, Any]], phase: str
) -> dict[str, Any]:
    """按实体、普通关系和规则分别汇总标准监督 P/R/F1。"""
    if not case_results:
        raise GraphBuilderConfigurationError("experiment_cases_missing")
    categories = {
        category: _prf1_from_counts(
            sum(int(item[phase][category]["tp"]) for item in case_results),
            sum(int(item[phase][category]["fp"]) for item in case_results),
            sum(int(item[phase][category]["fn"]) for item in case_results),
        )
        for category in ("entities", "relationships", "rules")
    }
    overall = _prf1_from_counts(
        sum(item["tp"] for item in categories.values()),
        sum(item["fp"] for item in categories.values()),
        sum(item["fn"] for item in categories.values()),
    )
    return {
        **overall,
        "graph": dict(overall),
        "categories": categories,
        "contract": (
            "scoped_closed_world_supervised: P=TP/(TP+FP), R=TP/(TP+FN)"
        ),
        "standard_supervised_prf1": True,
    }
