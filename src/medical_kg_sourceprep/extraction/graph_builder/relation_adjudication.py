"""对多路关系候选做统一的 LLM 裁决与只读派生。

本模块不包含具体疾病、指标、句式或测试案例规则。它只负责把指定证据范围内的
旧关系与新提案组成候选并集，再根据独立 Judge 的逐候选结论生成派生候选图。
原始候选图和 Judge 工件均不修改，派生图仍保持 ``candidate-only/HOLD``。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .contract import GraphBuilderConfigurationError
from .evaluation.scoring import merge_candidate_graphs


RULE_RELATION_TYPES = frozenset({"RULE_INPUT", "RULE_OUTPUT"})


def mask_text_outside_ranges(
    text: str, evidence_ranges: Sequence[tuple[int, int]]
) -> str:
    """用等长空格隐藏范围外正文，同时保持换行和字符坐标不变。"""
    if not evidence_ranges or any(
        start < 0 or start >= end or end > len(text) for start, end in evidence_ranges
    ):
        raise GraphBuilderConfigurationError("relation_adjudication_evidence_ranges_invalid")
    visible = [False] * len(text)
    for start, end in evidence_ranges:
        visible[start:end] = [True] * (end - start)
    return "".join(
        character if keep or character in "\r\n" else " "
        for character, keep in zip(text, visible, strict=True)
    )


def _relation_spans(relationship: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    """读取已经通过来源校验的关系证据范围。"""
    references = relationship.get("relation_evidence_refs")
    if not isinstance(references, list):
        source_ref = relationship.get("source_ref")
        references = [source_ref] if isinstance(source_ref, Mapping) else []
    spans: list[tuple[int, int]] = []
    for reference in references:
        if not isinstance(reference, Mapping):
            continue
        start, end = reference.get("char_start"), reference.get("char_end")
        if (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and start < end
        ):
            spans.append((start, end))
    return tuple(spans)


def _overlaps_any(
    relationship: Mapping[str, Any], evidence_ranges: Sequence[tuple[int, int]]
) -> bool:
    return any(
        max(relation_start, range_start) < min(relation_end, range_end)
        for relation_start, relation_end in _relation_spans(relationship)
        for range_start, range_end in evidence_ranges
    )


def _deduplicate_relationships(
    relationships: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """同端点、同类型关系只保留首次出现的可追溯候选。"""
    unique: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for relationship in relationships:
        identity = (
            str(relationship.get("source_candidate_key", "")),
            str(relationship.get("relation_type", "")),
            str(relationship.get("target_candidate_key", "")),
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(relationship)
    return unique


def build_relationship_candidate_union(
    *,
    baseline: Mapping[str, Any],
    proposal_graphs: Iterable[Mapping[str, Any]],
    evidence_ranges: Sequence[tuple[int, int]],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """构造待 Judge 的范围内候选并集，以及不参加裁决的保留子图。"""
    if not evidence_ranges or any(start >= end for start, end in evidence_ranges):
        raise GraphBuilderConfigurationError("relation_adjudication_evidence_ranges_invalid")
    proposals = list(proposal_graphs)
    union = merge_candidate_graphs((baseline, *proposals))
    judged_relationships = _deduplicate_relationships(
        relationship
        for relationship in union.get("relationships", [])
        if isinstance(relationship, Mapping)
        and relationship.get("relation_type") not in RULE_RELATION_TYPES
        and _overlaps_any(relationship, evidence_ranges)
    )
    preserved_relationships = [
        relationship
        for relationship in baseline.get("relationships", [])
        if isinstance(relationship, Mapping)
        and (
            relationship.get("relation_type") in RULE_RELATION_TYPES
            or not _overlaps_any(relationship, evidence_ranges)
        )
    ]
    judge_graph = {
        "nodes": list(union.get("nodes", [])),
        "relationships": judged_relationships,
    }
    preserved_graph = {
        "nodes": list(union.get("nodes", [])),
        "relationships": preserved_relationships,
    }
    return judge_graph, preserved_graph


def apply_relationship_judgments(
    *,
    judge_graph: Mapping[str, Any],
    preserved_graph: Mapping[str, Any],
    judge_document: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """只保留 Judge 明确支持的普通关系，不自动执行修复建议。"""
    relationships = [
        item for item in judge_graph.get("relationships", []) if isinstance(item, Mapping)
    ]
    verdict_by_id: dict[str, str] = {}
    for result in judge_document.get("results", []):
        if not isinstance(result, Mapping):
            continue
        judge_item_id = result.get("judge_item_id")
        verdict = result.get("verdict")
        if isinstance(judge_item_id, str) and isinstance(verdict, str):
            verdict_by_id[judge_item_id] = verdict

    expected_ids = {
        f"relationship:{relationship['candidate_key']}"
        for relationship in relationships
        if isinstance(relationship.get("candidate_key"), str)
    }
    if set(verdict_by_id) != expected_ids:
        raise GraphBuilderConfigurationError("relation_adjudication_results_incomplete")

    supported = [
        relationship
        for relationship in relationships
        if verdict_by_id[f"relationship:{relationship['candidate_key']}"] == "SUPPORTED"
    ]
    merged = merge_candidate_graphs((
        preserved_graph,
        {"nodes": judge_graph.get("nodes", []), "relationships": supported},
    ))
    graph: dict[str, Any] = {
        **merged,
        "schema_version": "relationship-adjudicated-candidate-graph/v0.1",
        "status": "candidate-only",
        "publication_status": "HOLD",
    }
    counts = {
        "input_candidates": len(relationships),
        "supported": len(supported),
        "unsupported": sum(verdict == "UNSUPPORTED" for verdict in verdict_by_id.values()),
        "repair": sum(verdict == "REPAIR" for verdict in verdict_by_id.values()),
        "abstain": sum(verdict == "ABSTAIN" for verdict in verdict_by_id.values()),
    }
    return graph, counts
