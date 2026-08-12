"""候选记录的审查状态、拒绝原因和安全摘要。

本模块不判断医学语义。它接收其他校验模块给出的结果，把“为什么不能直接采用”
整理成稳定、可审计且不含原始模型响应的 review queue 记录。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

if __package__ in {None, ""}:
    # 允许直接运行本文件查看底部示例；正常作为包导入时仍走下面的相对导入。
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from medical_kg_sourceprep.extraction.graph_builder.contract import GraphBuilderConfigurationError
else:
    from ..contract import GraphBuilderConfigurationError


_JUDGE_TEXT_LIMIT = 2_000


def _bounded_value(value: Any) -> Any:
    """裁剪 Judge 草稿中的自由文本，不保留模型响应之外的任意字段。"""
    if isinstance(value, str):
        return value[:_JUDGE_TEXT_LIMIT]
    if isinstance(value, list):
        return [_bounded_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key)[:120]: _bounded_value(item) for key, item in value.items()}
    return value if value is None or isinstance(value, (bool, int, float)) else str(value)[:_JUDGE_TEXT_LIMIT]


def _judge_draft(
    stage: str,
    index: int,
    reason_code: str,
    candidate_draft: Mapping[str, Any],
) -> dict[str, Any]:
    """生成给未来独立 Judge 的单条候选草稿。

    草稿不属于候选图，故没有伪造的 ``source_ref`` 或 ``candidate_key``。它只保存
    单条输出本身的有限字段和稳定原因码；完整原文由运行工件中的 chunk 引用提供。
    """
    normalized = _bounded_value(candidate_draft)
    identity = hashlib.sha256(
        json.dumps(
            {"stage": stage, "index": index, "reason_code": reason_code, "candidate_draft": normalized},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode()
    ).hexdigest()[:24]
    return {
        "judge_id": f"judge:{identity}",
        "stage": stage,
        "model_item_index": index,
        "judge_status": "PENDING",
        "reason_code": reason_code,
        "candidate_draft": normalized,
    }


def _node_judge_draft(node: Any) -> dict[str, Any] | None:
    """提取可由 Judge 重映射或补证据的最小节点草稿。"""
    properties = getattr(node, "properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    label = getattr(node, "label", "")
    mention = properties.get("mention")
    expression = properties.get("rule_expression")
    if label == "RuleDefinition":
        has_minimum_identity = isinstance(expression, str) and bool(expression)
    else:
        has_minimum_identity = isinstance(mention, str) and bool(mention)
    if not (isinstance(label, str) and label and has_minimum_identity):
        return None
    allowed = (
        "mention", "extraction_reason", "canonical_name_candidate", "exact_quote", "exact_quote_occurrence_index",
        "mention_occurrence_index", "source_char_start", "source_char_end",
        "bound_indicator_mention", "rule_stage_candidate", "rule_expression", "rule_name",
        "rule_evidence_json", "table_state_evidence_json",
    )
    return {"kind": "node", "label": label, "properties": {key: properties[key] for key in allowed if key in properties}}


def _relationship_judge_draft(relationship: Any) -> dict[str, Any] | None:
    """提取有类型和两个端点标识的最小关系草稿。"""
    relation_type = getattr(relationship, "type", "")
    source = getattr(relationship, "start_node_id", "")
    target = getattr(relationship, "end_node_id", "")
    if not all(isinstance(value, str) and value for value in (relation_type, source, target)):
        return None
    properties = getattr(relationship, "properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    allowed = (
        "exact_quote", "exact_quote_occurrence_index", "source_char_start", "source_char_end",
        "relation_cue", "rule_evidence_role",
    )
    return {
        "kind": "relationship", "relation_type": relation_type,
        "start_node_id": source, "end_node_id": target,
        "properties": {key: properties[key] for key in allowed if key in properties},
    }


def _review_item(
    stage: str,
    index: int,
    status: str,
    reason_code: str,
    summary: Mapping[str, Any],
    *,
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """生成可审计的审查项，不保存模型原始响应。

    参数含义：
    - stage：问题发生在哪个抽取阶段，例如 entity、rule、relation；
    - index：该阶段模型返回数组中的位置，供定位同一次响应的候选；
    - status：只能是彻底拒绝的 REJECTED，或保留待人工复核的 REVIEW_REQUIRED；
    - reason_code：稳定的机器可读原因码；
    - summary：仅保留限长的候选摘要，不能放入完整模型响应；
    - warnings：可回放但不完整时附加的警告列表。
    """
    # 审查队列不是通用状态机，只有这两类问题记录允许进入队列。
    if status not in {"REVIEW_REQUIRED", "REJECTED"}:
        raise GraphBuilderConfigurationError("review_status_invalid")
    # 同一问题在相同阶段、位置、原因和摘要下得到相同 ID，便于重跑时去重和追踪。
    # sort_keys 和 set 去重保证字典键顺序、重复 warning 不会影响 ID。
    identity = hashlib.sha256(
        json.dumps(
            {
                "stage": stage,
                "index": index,
                "status": status,
                "reason_code": reason_code,
                "summary": summary,
                "warnings": sorted(set(warnings)),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:24]
    return {
        # hold 前缀明确表示该记录不是已经发布的医学知识或最终图元素。
        "review_id": f"hold:{identity}",
        "stage": stage,
        "status": status,
        "reason_code": reason_code,
        "candidate_summary": dict(summary),
        **({"warnings": sorted(set(warnings))} if warnings else {}),
    }


def _hold(stage: str, index: int, reason_code: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    """将不可接纳的候选转换为 REJECTED 审查项。

    适用于来源无法回放、类型非法、JSON 结构损坏等硬错误。原候选不会写入图，
    但拒绝原因会进入 review queue，避免失败静默丢失。
    """
    return _review_item(stage, index, "REJECTED", reason_code, summary)


def _mark_partial(record: Mapping[str, Any], *warnings: str) -> None:
    """保留可回放但不完整的候选，并禁止其自动发布。

    PARTIAL 与 REJECTED 的区别：PARTIAL 的来源和基本结构仍可回放，只是端点、
    规则输入输出或绑定不完整，因而可以留给后续模型评测或人工审核；REJECTED
    则不保留为候选节点/关系。
    """
    if not isinstance(record, dict):
        return
    # 防御性处理旧数据或异常数据：warnings 不是列表时重新建立列表，避免污染候选记录。
    current = record.setdefault("warnings", [])
    if not isinstance(current, list):
        current = []
        record["warnings"] = current
    for warning in warnings:
        if warning not in current:
            current.append(warning)
    # 无论 warning 来自哪个校验器，PARTIAL 一律要求人工复核，发布状态仍由调用方保持 HOLD。
    record["extraction_status"] = "PARTIAL"
    record["review_status"] = "REVIEW_REQUIRED"


def _node_summary(node: Any) -> dict[str, Any]:
    """生成限长节点摘要，供 review queue 使用。

    审查项只需让人识别失败候选，不能复制完整 prompt、表格或模型原始 JSON；因此
    所有自由文本均截断。RuleDefinition 没有业务实体 mention，改用规则阶段和表达式摘要。
    """
    properties = getattr(node, "properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    if getattr(node, "label", "") == "RuleDefinition":
        return {
            "label": "RuleDefinition",
            "rule_stage_candidate": properties.get("rule_stage_candidate"),
            "rule_expression": str(properties.get("rule_expression", ""))[:160],
            "rule_name": str(properties.get("rule_name", ""))[:160],
        }
    return {
        "label": str(getattr(node, "label", ""))[:80],
        "mention": str(properties.get("mention", ""))[:160],
        "canonical_name_candidate": str(properties.get("canonical_name_candidate", ""))[:160],
    }


def _relationship_summary(relationship: Any) -> dict[str, Any]:
    """生成限长关系摘要，避免将原始模型响应写入审查项。

    这里保留关系类型、两个模型端点 ID 和 cue，足以定位问题；不写入完整 exact_quote，
    因为原文证据应在候选或来源包中按 source_ref 单独管理。
    """
    properties = getattr(relationship, "properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    return {
        "relation_type": str(getattr(relationship, "type", ""))[:80],
        "start_node_id": str(getattr(relationship, "start_node_id", ""))[:160],
        "end_node_id": str(getattr(relationship, "end_node_id", ""))[:160],
        "relation_cue": str(properties.get("relation_cue", ""))[:80],
    }


if __name__ == "__main__":
    # 此处只用于手动观察本模块的输入和输出；不调用模型，也不读写文件。
    from types import SimpleNamespace

    node = SimpleNamespace(
        # 模型临时 ID 只在本次响应中有意义，不能成为 Judge 队列里的稳定身份。
        id="node-17",
        label="UnknownType",
        properties={
            "mention": "血清铁降低",
            "canonical_name_candidate": "血清铁降低",
            "exact_quote": "血清铁降低提示缺铁性贫血。",
            # 同一句引语多次出现时，用下标标明本候选引用的是第几次出现。
            "exact_quote_occurrence_index": 0,
            # 以下是模型可能擅自输出的自评分和解释，均不属于 Schema 证据。
            "model_confidence": 0.94,
            "model_explanation": "根据医学知识推断该状态与缺铁性贫血有关。",
        },
    )
    relationship = SimpleNamespace(
        type="INDICATES",
        start_node_id="candidate:serum-iron-low",
        end_node_id="candidate:iron-deficiency-anemia",
        properties={
            "exact_quote": "血清铁降低提示缺铁性贫血。",
            "relation_cue": "提示",
            "exact_quote_occurrence_index": 0,
        },
    )
    partial_record = {
        "candidate_key": "candidate:state",
        "entity_type": "IndicatorState",
        "mention": "血清铁降低",
        "extraction_status": "VALID",
        "review_status": "PENDING",
        "publication_status": "HOLD",
    }
    print("模型原始节点（包含不受合同约束的字段）\n", node)
    node_draft = _node_judge_draft(node)
    print("送入 Judge 的节点草稿（仅保留身份和证据字段）\n", node_draft)
    relationship_draft = _relationship_judge_draft(relationship)
    assert node_draft is not None and relationship_draft is not None

    examples = {
        "输入节点": {"label": node.label, "properties": node.properties},
        "节点摘要": _node_summary(node),
        "节点 Judge 草稿": _judge_draft("entity", 0, "entity_type_not_enabled_for_trial", node_draft),
        "输入关系": {
            "type": relationship.type,
            "start_node_id": relationship.start_node_id,
            "end_node_id": relationship.end_node_id,
            "properties": relationship.properties,
        },
        "关系摘要": _relationship_summary(relationship),
        "关系 Judge 草稿": _judge_draft(
            "relation", 1, "relation_endpoint_not_from_frozen_catalog", relationship_draft,
        ),
        "PARTIAL 前": dict(partial_record),
    }
    _mark_partial(partial_record, "RULE_INPUTS_INCOMPLETE", "OUTPUT_ENTITY_UNRESOLVED")
    examples["PARTIAL 后"] = partial_record
    examples["REJECTED 审查项"] = _hold(
        "rule", 2, "rule_expression_missing", {"label": "RuleDefinition", "rule_expression": ""},
    )
    examples["REVIEW_REQUIRED 审查项"] = _review_item(
        "relation", 3, "REVIEW_REQUIRED", "relation_may_be_joint_condition",
        _relationship_summary(relationship), warnings=("RELATION_MAY_BE_JOINT_CONDITION",),
    )
    # print(json.dumps(examples, ensure_ascii=False, indent=2, sort_keys=True))
