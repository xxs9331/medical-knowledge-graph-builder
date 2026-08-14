"""按典型案例的目标项与禁止项合同评估候选图。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from typing import Any


def _metrics(predicted: set[tuple[Any, ...]], expected: set[tuple[Any, ...]]) -> dict[str, Any]:
    """计算穷举集合常用指标；当前典型案例主评分不直接使用假阳性。"""
    true_positive = predicted & expected
    false_positive = predicted - expected
    false_negative = expected - predicted
    tp, fp, fn = len(true_positive), len(false_positive), len(false_negative)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not expected else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6),
        "true_positive": sorted(true_positive),
        "false_positive": sorted(false_positive),
        "false_negative": sorted(false_negative),
    }


def _rule_endpoints(
    rule_key: str, relationships: Iterable[Mapping[str, Any]], mentions: Mapping[str, str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """沿 RULE_INPUT/RULE_OUTPUT 方向还原一个规则节点的输入和输出 mention。"""
    inputs: list[str] = []
    outputs: list[str] = []
    for relation in relationships:
        relation_type = relation.get("relation_type")
        source_key = relation.get("source_candidate_key")
        target_key = relation.get("target_candidate_key")
        if relation_type == "RULE_INPUT" and target_key == rule_key and source_key in mentions:
            inputs.append(mentions[str(source_key)])
        elif relation_type == "RULE_OUTPUT" and source_key == rule_key and target_key in mentions:
            outputs.append(mentions[str(target_key)])
    return tuple(sorted(set(inputs))), tuple(sorted(set(outputs)))


def project_candidate_graph(graph: Mapping[str, Any]) -> dict[str, set[tuple[Any, ...]]]:
    """使用可读 mention 将候选图投影为与 typical-cases 金标一致的集合。"""
    nodes = [item for item in graph.get("nodes", []) if isinstance(item, Mapping)]
    relationships = [
        item for item in graph.get("relationships", []) if isinstance(item, Mapping)
    ]
    # 评分目标使用人可读 mention，先建立 candidate_key 到 mention 的端点映射。
    mentions = {
        str(node["candidate_key"]): str(node["mention"])
        for node in nodes
        if isinstance(node.get("candidate_key"), str) and isinstance(node.get("mention"), str)
    }
    entities = {
        (str(node["entity_type"]), str(node["mention"]))
        for node in nodes
        if node.get("entity_type") != "RuleDefinition"
        and isinstance(node.get("entity_type"), str)
        and isinstance(node.get("mention"), str)
    }
    # 规则边由规则评分单独处理，不能混入普通关系覆盖率。
    ordinary_relationships = {
        (mentions[str(item["source_candidate_key"])], str(item["relation_type"]),
         mentions[str(item["target_candidate_key"])])
        for item in relationships
        if item.get("relation_type") not in {"RULE_INPUT", "RULE_OUTPUT"}
        and str(item.get("source_candidate_key")) in mentions
        and str(item.get("target_candidate_key")) in mentions
    }
    rules: set[tuple[Any, ...]] = set()
    for node in nodes:
        if node.get("entity_type") != "RuleDefinition" or not isinstance(node.get("candidate_key"), str):
            continue
        inputs, outputs = _rule_endpoints(str(node["candidate_key"]), relationships, mentions)
        # 当前候选合同没有独立 logic 字段；规则名只用于诊断，不能臆造金标 logic。
        rules.add((str(node.get("rule_stage_candidate", "UNKNOWN")), inputs, outputs))
    return {"entities": entities, "relationships": ordinary_relationships, "rules": rules}


def merge_candidate_graphs(graphs: Iterable[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """按稳定候选键合并多个候选图，可用于跨 chunk 或多轮结果并集。"""
    nodes: list[Any] = []
    relationships: list[Any] = []
    seen_node_keys: set[str] = set()
    seen_relationship_keys: set[str] = set()
    # 保留最先出现的候选版本。稳定键相同表示同一候选，不能因重复出现增加得分。
    for graph in graphs:
        for item in graph.get("nodes", []):
            if not isinstance(item, Mapping):
                continue
            key = item.get("candidate_key")
            if isinstance(key, str) and key in seen_node_keys:
                continue
            nodes.append(item)
            if isinstance(key, str):
                seen_node_keys.add(key)
        for item in graph.get("relationships", []):
            if not isinstance(item, Mapping):
                continue
            key = item.get("candidate_key")
            if isinstance(key, str) and key in seen_relationship_keys:
                continue
            relationships.append(item)
            if isinstance(key, str):
                seen_relationship_keys.add(key)
    return {"nodes": nodes, "relationships": relationships}


def _equivalent_mention(left: str, right: str, source_text: str) -> bool:
    """只接受完全相同或原文明示的“全称（缩写）”等价，不使用外部词典。"""
    if left == right:
        return True
    for full_name, abbreviation in ((left, right), (right, left)):
        pattern = rf"{re.escape(full_name)}\s*[（(]\s*{re.escape(abbreviation)}\s*[）)]"
        if re.search(pattern, source_text):
            return True
    return False


def _target_metrics(expected: list[Any], matched_indexes: set[int]) -> dict[str, Any]:
    """按人工标注目标计算覆盖率，并保留命中与遗漏明细。"""
    total = len(expected)
    matched = [item for index, item in enumerate(expected) if index in matched_indexes]
    missed = [item for index, item in enumerate(expected) if index not in matched_indexes]
    return {
        "target_total": total,
        "matched": len(matched),
        "missed": len(missed),
        "coverage": round(len(matched) / total, 6) if total else 1.0,
        "matched_targets": matched,
        "missed_targets": missed,
    }


def _rule_records(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    """将图中的 RuleDefinition 与规则边整理为便于匹配的记录。"""
    nodes = [item for item in graph.get("nodes", []) if isinstance(item, Mapping)]
    relationships = [
        item for item in graph.get("relationships", []) if isinstance(item, Mapping)
    ]
    mentions = {
        str(node["candidate_key"]): str(node["mention"])
        for node in nodes
        if isinstance(node.get("candidate_key"), str) and isinstance(node.get("mention"), str)
    }
    records = []
    for node in nodes:
        rule_key = node.get("candidate_key")
        if node.get("entity_type") != "RuleDefinition" or not isinstance(rule_key, str):
            continue
        inputs, outputs = _rule_endpoints(rule_key, relationships, mentions)
        # 公式角色来自可回放证据，不根据规则名称猜测逻辑类型。
        evidence = node.get("rule_evidence_refs", [])
        quotes = [
            str(item["exact_quote"])
            for item in evidence
            if isinstance(item, Mapping) and isinstance(item.get("exact_quote"), str)
        ] if isinstance(evidence, list) else []
        roles = {
            str(item["role"]).lower()
            for item in evidence
            if isinstance(item, Mapping) and isinstance(item.get("role"), str)
        } if isinstance(evidence, list) else set()
        records.append({
            "candidate_key": rule_key,
            "stage": str(node.get("rule_stage_candidate", "UNKNOWN")),
            "inputs": inputs,
            "outputs": outputs,
            "evidence_text": "\n".join(quotes),
            "logic": "FORMULA" if "formula" in roles else None,
        })
    return records


def _rule_matches(
    candidate: Mapping[str, Any], expected: Mapping[str, Any], source_text: str,
    manual_review_notes: tuple[str, ...],
) -> bool:
    """检查一个规则候选是否完整覆盖指定金标规则。"""
    if candidate.get("stage") != expected.get("rule_stage"):
        return False
    expected_logic = expected.get("logic")
    if expected_logic == "FORMULA" and candidate.get("logic") != "FORMULA":
        return False
    candidate_inputs = tuple(str(item) for item in candidate.get("inputs", ()))
    candidate_outputs = tuple(str(item) for item in candidate.get("outputs", ()))
    evidence_text = str(candidate.get("evidence_text", ""))

    def endpoint_is_supported(target: str, endpoints: tuple[str, ...]) -> bool:
        return any(_equivalent_mention(target, endpoint, source_text) for endpoint in endpoints)

    if not all(endpoint_is_supported(str(item), candidate_outputs) for item in expected.get("outputs", [])):
        return False
    # 当前图合同不把参考量和运行时参数强制建成 RULE_INPUT；逐字公式证据同样可证明输入覆盖。
    def input_is_supported(target: str) -> bool:
        if endpoint_is_supported(target, candidate_inputs) or target in evidence_text:
            return True
        # 人工金标可以显式裁定规范源 OCR；抽取候选仍必须忠实保留原始坏字符。
        return target in source_text and any(
            "OCR" in note and target in note for note in manual_review_notes
        )

    return all(input_is_supported(str(item)) for item in expected.get("inputs", []))


def score_candidate_graph(
    graph: Mapping[str, Any], gold_case: Mapping[str, Any], *, source_text: str = ""
) -> dict[str, Any]:
    """评分非穷举典型案例；未标注候选不冒充假阳性，金标不进入模型上下文。"""
    projected = project_candidate_graph(graph)
    expected_entities = [tuple(item) for item in gold_case.get("entities", [])]
    # 典型案例是目标清单而非穷举语料，只判断每个目标是否至少被一个候选覆盖。
    matched_entities = {
        index
        for index, (entity_type, mention) in enumerate(expected_entities)
        if any(
            predicted_type == entity_type
            and _equivalent_mention(str(predicted_mention), str(mention), source_text)
            for predicted_type, predicted_mention in projected["entities"]
        )
    }

    expected_relationships = [tuple(item) for item in gold_case.get("relationships", [])]
    matched_relationships = {
        index
        for index, (source, relation_type, target) in enumerate(expected_relationships)
        if any(
            predicted_type == relation_type
            and _equivalent_mention(str(predicted_source), str(source), source_text)
            and _equivalent_mention(str(predicted_target), str(target), source_text)
            for predicted_source, predicted_type, predicted_target in projected["relationships"]
        )
    }

    expected_rules = [item for item in gold_case.get("rules", []) if isinstance(item, Mapping)]
    manual_review_notes = tuple(
        str(item) for item in gold_case.get("review_notes", []) if isinstance(item, str)
    )
    candidate_rules = _rule_records(graph)
    matched_rules: set[int] = set()
    used_candidates: set[int] = set()
    # 一条候选规则最多匹配一条金标规则，避免重复计分。
    for expected_index, expected_rule in enumerate(expected_rules):
        for candidate_index, candidate_rule in enumerate(candidate_rules):
            if candidate_index not in used_candidates and _rule_matches(
                candidate_rule, expected_rule, source_text, manual_review_notes
            ):
                matched_rules.add(expected_index)
                used_candidates.add(candidate_index)
                break

    forbidden = [tuple(item) for item in gold_case.get("must_not_extract", [])]
    violated_forbidden = [
        item for item in forbidden
        if any(
            predicted_type == item[1]
            and _equivalent_mention(str(predicted_source), str(item[0]), source_text)
            and _equivalent_mention(str(predicted_target), str(item[2]), source_text)
            for predicted_source, predicted_type, predicted_target in projected["relationships"]
        )
    ]

    scores: dict[str, Any] = {
        "entities": _target_metrics(expected_entities, matched_entities),
        "relationships": _target_metrics(expected_relationships, matched_relationships),
        "rules": _target_metrics(expected_rules, matched_rules),
    }
    # 综合分把每个正向目标和每个禁止项视为一条等权约束：
    # 命中正向目标得 1 分，成功避开禁止项也得 1 分。
    positive_total = sum(scores[name]["target_total"] for name in scores)
    positive_matched = sum(scores[name]["matched"] for name in scores)
    constraint_total = positive_total + len(forbidden)
    satisfied = positive_matched + len(forbidden) - len(violated_forbidden)
    scores["forbidden"] = {
        "target_total": len(forbidden),
        "violations": len(violated_forbidden),
        "avoidance": round((len(forbidden) - len(violated_forbidden)) / len(forbidden), 6)
        if forbidden else 1.0,
        "violated_targets": violated_forbidden,
    }
    scores["challenge"] = {
        "satisfied_constraints": satisfied,
        "total_constraints": constraint_total,
        "score": round(satisfied / constraint_total, 6) if constraint_total else 1.0,
        "score_percent": round(100 * satisfied / constraint_total, 2) if constraint_total else 100.0,
    }
    # 未标注候选只记录数量，不作为假阳性扣分；其正确性由独立 Judge 负责。
    scores["unscored_candidates"] = {
        "entities": len(projected["entities"]),
        "relationships": len(projected["relationships"]),
        "rules": len(candidate_rules),
        "note": "典型案例不是穷举标注；候选正确性由独立 Judge 评估。",
    }
    scores["scoring_contract"] = (
        "target coverage + forbidden avoidance; source-explicit full-name/abbreviation aliases; "
        "formula parameters may be verified from replayable rule evidence; explicit human OCR "
        "adjudications in review_notes are applied only during scoring"
    )
    scores["manual_adjudications"] = list(manual_review_notes)
    return scores
