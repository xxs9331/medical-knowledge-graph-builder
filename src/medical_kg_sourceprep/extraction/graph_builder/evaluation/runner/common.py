"""单轮评测与二次抽取实验共用的确定性汇总函数。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...contract import GraphBuilderConfigurationError


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
