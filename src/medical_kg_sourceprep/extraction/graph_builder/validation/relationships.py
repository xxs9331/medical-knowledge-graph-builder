"""普通关系、规则边和 RuleDefinition 图结构的本地确定性校验。

本模块不判断医学语义是否正确，只回答候选能否被稳定保存和逐字回放：

- 模型是否在关系阶段错误创建了节点；
- 关系类型和两个端点是否来自当前阶段及冻结目录；
- 引语或规则证据角色是否能回到当前 chunk；
- Schema 端点类型不匹配时是否需要保留为 ``PARTIAL``；
- 同一响应是否重复输出同一条关系；
- RuleDefinition 的输入输出边是否与其结构化表达式一致。

关系方向、直接性、是否跨越中间机制以及联合条件语义不在这里裁决，后续交给
LLM Judge。所有本地接纳结果仍为 ``publication_status=HOLD``，不会直接发布。
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
import re
from typing import Any

from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jNode

if __package__ in {None, ""}:
    # 允许直接执行本文件观察底部离线 demo；正常作为包导入时不进入该分支。
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from medical_kg_sourceprep.extraction.graph_builder.contract import (
        MODEL_RELATION_TYPES,
        GraphBuilderConfigurationError,
    )
    from medical_kg_sourceprep.extraction.graph_builder.schema import _relation_endpoint_pairs
    from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk
    from medical_kg_sourceprep.extraction.graph_builder.validation.provenance import (
        _relation_key,
        _multi_evidence_relation_key,
        _parse_cross_chunk_relation_evidence,
        _rule_expression_endpoints,
        _source_ref,
    )
    from medical_kg_sourceprep.extraction.graph_builder.validation.result import CandidateNormalization
    from medical_kg_sourceprep.extraction.graph_builder.validation.review import (
        _hold,
        _judge_draft,
        _mark_partial,
        _node_judge_draft,
        _node_summary,
        _relationship_judge_draft,
        _relationship_summary,
        _review_item,
    )
else:
    from ..contract import MODEL_RELATION_TYPES, GraphBuilderConfigurationError
    from ..schema import _relation_endpoint_pairs
    from ...llm_extraction import EvidenceChunk
    from .provenance import (
        _multi_evidence_relation_key,
        _parse_cross_chunk_relation_evidence,
        _relation_key,
        _rule_expression_endpoints,
        _source_ref,
    )
    from .result import CandidateNormalization
    from .review import (
        _hold,
        _judge_draft,
        _mark_partial,
        _node_judge_draft,
        _node_summary,
        _relationship_judge_draft,
        _relationship_summary,
        _review_item,
    )


def _strip_chunk_prefix(value: str, chunk_id: str) -> str | None:
    """移除 GraphRAG 加入的当前 chunk 前缀，拒绝其他 chunk 的端点。"""
    prefix = f"{chunk_id}:"
    return value[len(prefix):] if value.startswith(prefix) else None


def _has_allowed_endpoints(
    schema: Mapping[str, Any], relation_type: str, source: Mapping[str, Any], target: Mapping[str, Any]
) -> bool:
    """核对源节点和目标节点的实体类型是否属于 Schema 允许组合。"""
    return (source["entity_type"], target["entity_type"]) in _relation_endpoint_pairs(schema, relation_type)


def _rule_relation_source_ref(
    relationship: Any, relation_type: str, source: Mapping[str, Any], target: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """为规则边取得 RuleDefinition 中已经验证过的对应证据。

    普通关系返回 ``None``，继续走普通引语定位。``RULE_INPUT`` 的规则节点位于
    target，``RULE_OUTPUT`` 的规则节点位于 source。模型只能选择规则节点已有的
    ``rule_evidence_role``，不能为规则边重新提供或编造一段证据。
    """
    if relation_type not in {"RULE_INPUT", "RULE_OUTPUT"}:
        return None
    # RULE_INPUT: 业务实体 -> 规则；RULE_OUTPUT: 规则 -> 业务实体。
    rule = target if relation_type == "RULE_INPUT" else source
    if rule.get("entity_type") != "RuleDefinition":
        raise GraphBuilderConfigurationError("rule_relation_definition_missing")
    role = relationship.properties.get("rule_evidence_role")
    if not isinstance(role, str):
        raise GraphBuilderConfigurationError("rule_relation_evidence_role_missing")
    # 返回相同角色下已经完成原文回放的 source_ref。
    for evidence_ref in rule.get("rule_evidence_refs", []):
        if evidence_ref.get("role") == role:
            return evidence_ref
    raise GraphBuilderConfigurationError("rule_relation_evidence_role_unknown")


def _has_state_source_refs(
    relationship: Any,
    *,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, list[Mapping[str, Any]] | None]:
    """为 HAS_STATE 选择普通引语或派生状态已经验证过的证据锚点。

    普通状态词如“血清铁降低”可直接出现在关系引语中；表格箭头派生状态则没有
    连续的状态词，不能要求模型伪造 exact_quote。后者复用目标状态节点已回放的
    table_header/table_row，并要求源指标名称至少出现在其中一个锚点内。
    """
    properties = relationship.properties
    exact_quote = properties.get("exact_quote")
    derived_refs = target.get("table_state_evidence_refs")
    if not isinstance(derived_refs, list):
        derived_refs = target.get("derived_entity_evidence_refs")
    # 有普通引语时交给通用 _source_ref；没有已回放派生证据也无法走特殊通道。
    if exact_quote not in (None, "") or not isinstance(derived_refs, list):
        return None, None
    if target.get("entity_type") != "IndicatorState":
        return None, None
    if not all(isinstance(ref, Mapping) for ref in derived_refs):
        raise GraphBuilderConfigurationError("has_state_derived_evidence_invalid")
    quotes = [ref.get("exact_quote") for ref in derived_refs]
    if not any(isinstance(quote, str) and source["mention"] in quote for quote in quotes):
        raise GraphBuilderConfigurationError("has_state_derived_evidence_lacks_indicator")
    source_ref = target.get("source_ref")
    if not isinstance(source_ref, Mapping):
        raise GraphBuilderConfigurationError("has_state_derived_evidence_missing")
    return source_ref, list(derived_refs)


def normalize_candidate_relationships(
    graph: Neo4jGraph,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    allowed_relation_types: Collection[str] = MODEL_RELATION_TYPES,
    validate_rule_structures: bool = True,
) -> CandidateNormalization:
    """接纳模型提出的可回放关系，并将无法入图的最小关系分流给 Judge。

    包括 ``HAS_STATE`` 在内的所有边均由关系阶段模型输出。本地只验证关系类型、
    冻结端点组合和原文证据，不再根据状态名称自动推断指标归属。
    """
    # 冻结目录索引：后续所有模型端点都必须在这里命中。
    node_by_key = {item["candidate_key"]: item for item in nodes}
    # accepted：可保存的 VALID/PARTIAL 关系。
    relations: list[dict[str, Any]] = []
    # review_items：失败、重复或需要复核的结构化审查记录。
    holds: list[dict[str, Any]] = []
    # judge_drafts：具备最小关系身份、但无法进入候选图的未来 Judge 输入。
    judge_drafts: list[dict[str, Any]] = []
    # 同一模型响应内按稳定 relation key 去重。
    seen_keys: set[str] = set()
    # 先暂存普通关系和规则边；规则边可在最后进行跨边结构检查。
    model_relations: list[tuple[int, dict[str, Any]]] = []

    # 关系阶段不允许新增节点，关系必须引用先前冻结的 candidate_key。
    for index, node in enumerate(graph.nodes):
        draft = _node_judge_draft(node)
        if draft is None:
            # 连最小节点身份都没有，只能记录 REJECTED 审查项。
            holds.append(_hold("relation", index, "relation_phase_node_not_allowed", _node_summary(node)))
        else:
            # 有最小身份则记录阶段错位，并留给未来 Judge 判断是否需要重路由。
            judge_drafts.append(_judge_draft("relation", index, "relation_phase_node_not_allowed", draft))
            holds.append(_review_item("relation", index, "REVIEW_REQUIRED", "relation_phase_node_not_allowed", _node_summary(node)))
    for index, relationship in enumerate(graph.relationships):
        # summary 只保存定位问题所需的有限字段，不复制完整模型响应。
        summary = _relationship_summary(relationship)
        try:
            relation_type = relationship.type
            # 未知类型或属于其他阶段的关系不能进入当前候选图。
            if relation_type not in allowed_relation_types:
                raise GraphBuilderConfigurationError("relation_type_not_enabled_for_trial")
            # GraphRAG 输出端点通常形如 <chunk_id>:<candidate_key>，先去掉命名空间。
            source_key = _strip_chunk_prefix(relationship.start_node_id, chunk.chunk_id)
            target_key = _strip_chunk_prefix(relationship.end_node_id, chunk.chunk_id)
            if not source_key or not target_key:
                summary["missing_endpoint_candidate_keys"] = [
                    value for value in (relationship.start_node_id, relationship.end_node_id)
                    if _strip_chunk_prefix(value, chunk.chunk_id) is None
                ]
                raise GraphBuilderConfigurationError("relation_endpoint_not_from_frozen_catalog")
            source = node_by_key.get(source_key)
            target = node_by_key.get(target_key)
            # 前缀正确仍不代表候选存在；必须实际命中冻结目录。
            if source is None or target is None:
                summary["missing_endpoint_candidate_keys"] = [
                    key for key in (source_key, target_key) if key not in node_by_key
                ]
                for endpoint in (source, target):
                    if endpoint is not None and endpoint.get("entity_type") == "RuleDefinition":
                        _mark_partial(endpoint, "RULE_ENDPOINT_UNRESOLVED")
                raise GraphBuilderConfigurationError("relation_endpoint_not_from_frozen_catalog")

            properties = relationship.properties
            # 证据来源分三种：规则已有证据、表格状态双锚点、普通逐字引语。
            rule_source_ref = _rule_relation_source_ref(relationship, relation_type, source, target)
            table_state_source_ref: list[Mapping[str, Any]] | None = None
            if rule_source_ref is not None:
                # 规则边直接复用规则节点已验证的证据，不重新定位模型引语。
                source_ref = rule_source_ref
            elif relation_type == "HAS_STATE":
                source_ref, table_state_source_ref = _has_state_source_refs(
                    relationship, source=source, target=target
                )
                if source_ref is None:
                    # 非表格状态必须像普通关系一样提供包含两个端点的逐字引语。
                    source_ref = _source_ref(
                        chunk,
                        source["mention"],
                        properties.get("exact_quote"),
                        exact_quote_occurrence_index=properties.get("exact_quote_occurrence_index"),
                        source_char_start=properties.get("source_char_start"),
                        source_char_end=properties.get("source_char_end"),
                    )
            else:
                # 普通关系由代码定位 exact_quote；模型坐标仅作为兼容输入，不是信任来源。
                source_ref = _source_ref(
                    chunk,
                    source["mention"],
                    properties.get("exact_quote"),
                    exact_quote_occurrence_index=properties.get("exact_quote_occurrence_index"),
                    source_char_start=properties.get("source_char_start"),
                    source_char_end=properties.get("source_char_end"),
                )
            # 普通关系的同一段引语必须同时包含源端点和目标端点。源端点已由
            # _source_ref 验证；此处补验目标端点。规则边和表格状态使用各自证据规则。
            if rule_source_ref is None and table_state_source_ref is None and target["mention"] not in source_ref["exact_quote"]:
                raise GraphBuilderConfigurationError("relation_quote_lacks_endpoint")

            # warnings 只用于可保留但不完整的 PARTIAL，不用于医学语义评分。
            warnings: list[str] = []
            reasons: list[str] = []
            # 自环或 Schema 类型组合不匹配仍有完整端点和证据，所以不送 Judge，
            # 而是保留为 PARTIAL，等待后续 Schema 重映射或人工复核。
            if source_key == target_key or not _has_allowed_endpoints(schema, relation_type, source, target):
                warnings.append("RELATION_ENDPOINT_TYPE_INVALID")
                reasons.append("relation_endpoint_type_invalid")
                for endpoint in (source, target):
                    if endpoint.get("entity_type") == "RuleDefinition":
                        _mark_partial(endpoint, "RULE_ENDPOINT_TYPE_INVALID")
            # 关系类型、方向、直接性和联合条件都属于语义问题。本地只保存完整引语，
            # 不再要求或校验单词级触发表达，后续 LLM Judge 直接阅读整段证据。
            # 关系身份由类型、两个冻结端点和已回放证据共同决定。
            candidate_key = _relation_key(relation_type, source_key, target_key, source_ref)
            if candidate_key in seen_keys:
                raise GraphBuilderConfigurationError("duplicate_relation")
            seen_keys.add(candidate_key)
            # VALID 仅表示本地结构与证据通过，不代表语义已由 Judge 确认。
            record: dict[str, Any] = {
                "candidate_key": candidate_key,
                "relation_type": relation_type,
                "source_candidate_key": source_key,
                "target_candidate_key": target_key,
                "source_ref": source_ref,
                "generation": "model_candidate",
                "extraction_status": "VALID",
                "review_status": "PENDING",
                "publication_status": "HOLD",
            }
            if table_state_source_ref is not None:
                # 关系自身保留同一组证据锚点，避免只有状态节点知道派生依据。
                evidence_field = (
                    "table_state_evidence_refs"
                    if target.get("table_state_evidence_refs")
                    else "derived_entity_evidence_refs"
                )
                record[evidence_field] = table_state_source_ref
            if warnings:
                # 结构可保存但端点类型不符合当前 Schema 时降为 PARTIAL，并留下审查项。
                _mark_partial(record, *warnings)
                holds.append(_review_item(
                    "relation", index, "REVIEW_REQUIRED", reasons[0], summary, warnings=warnings,
                ))
            model_relations.append((index, record))
        except GraphBuilderConfigurationError as error:
            # 无法进入候选图时，具备“类型 + 两个端点”的最小关系送 Judge 队列；
            # 重复项不送 Judge，避免对同一内容重复评分。
            draft = _relationship_judge_draft(relationship)
            if draft is None or str(error) == "duplicate_relation":
                holds.append(_hold("relation", index, str(error), summary))
            else:
                judge_drafts.append(_judge_draft("relation", index, str(error), draft))
                holds.append(_review_item("relation", index, "REVIEW_REQUIRED", str(error), summary))

    # 普通关系阶段传 validate_rule_structures=False，直接保留前面的分流结果；
    # 规则边阶段传 True，再按整个 RuleDefinition 子图统一检查输入输出完整性。
    invalid_rule_keys: set[str] = set()
    if validate_rule_structures:
        valid_rule_edges, composite_holds, invalid_rule_keys = _validate_composite_structures(
            model_relations, node_by_key=node_by_key
        )
        relations.extend(valid_rule_edges)
        holds.extend(composite_holds)
    else:
        relations.extend(record for _index, record in model_relations)
    result = CandidateNormalization(
        accepted=relations,
        review_items=holds,
        judge_drafts=judge_drafts,
        invalid_rule_keys=invalid_rule_keys,
    )
    return result


def _resolve_frozen_endpoint(value: Any, node_by_key: Mapping[str, Mapping[str, Any]]) -> str:
    """接受原始 candidate_key 或 GraphRAG 自动增加前缀后的端点 ID。"""
    if not isinstance(value, str) or not value:
        raise GraphBuilderConfigurationError("relation_endpoint_invalid")
    matches = [key for key in node_by_key if value == key or value.endswith(f":{key}")]
    if len(matches) != 1:
        raise GraphBuilderConfigurationError("relation_endpoint_not_frozen")
    return matches[0]


def normalize_cross_chunk_relationships(
    graph: Neo4jGraph,
    *,
    chunks: Sequence[EvidenceChunk],
    schema: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    allowed_relation_types: Collection[str],
) -> CandidateNormalization:
    """校验案例级跨 chunk 普通关系及其分块证据数组。

    与单 chunk 校验不同，这里不要求伪造一个同时包含两端的连续引语。模型必须分别
    提供至少两个真实 chunk 的证据；端点名称可在后续 chunk 中重复出现，因此跨块性
    由证据数组而非节点主锚点决定。医学方向和直接性仍交给 Judge。
    """
    node_by_key = {
        str(item["candidate_key"]): item
        for item in nodes
        if isinstance(item.get("candidate_key"), str)
    }
    relations: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    judge_drafts: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for index, node in enumerate(graph.nodes):
        review_items.append(_hold("cross_chunk_relation", index, "relation_phase_created_node", _node_summary(node)))
    for index, relationship in enumerate(graph.relationships):
        summary = _relationship_summary(relationship)
        try:
            relation_type = relationship.type
            if relation_type not in allowed_relation_types:
                raise GraphBuilderConfigurationError("relation_type_not_enabled_for_trial")
            source_key = _resolve_frozen_endpoint(relationship.start_node_id, node_by_key)
            target_key = _resolve_frozen_endpoint(relationship.end_node_id, node_by_key)
            source = node_by_key[source_key]
            target = node_by_key[target_key]
            evidence_refs = _parse_cross_chunk_relation_evidence(
                chunks, value=relationship.properties.get("relation_evidence_json")
            )
            quotes = [str(ref["exact_quote"]) for ref in evidence_refs]
            if not any(str(source.get("mention", "")) in quote for quote in quotes):
                raise GraphBuilderConfigurationError("relation_evidence_lacks_source_endpoint")
            if not any(str(target.get("mention", "")) in quote for quote in quotes):
                raise GraphBuilderConfigurationError("relation_evidence_lacks_target_endpoint")
            candidate_key = _multi_evidence_relation_key(
                relation_type, source_key, target_key, evidence_refs
            )
            if candidate_key in seen_keys:
                raise GraphBuilderConfigurationError("duplicate_relation")
            seen_keys.add(candidate_key)
            record: dict[str, Any] = {
                "candidate_key": candidate_key,
                "relation_type": relation_type,
                "source_candidate_key": source_key,
                "target_candidate_key": target_key,
                "source_ref": evidence_refs[0],
                "relation_evidence_refs": evidence_refs,
                "generation": "cross_chunk_model_candidate",
                "extraction_status": "VALID",
                "review_status": "PENDING",
                "publication_status": "HOLD",
            }
            if source_key == target_key or not _has_allowed_endpoints(
                schema, relation_type, source, target
            ):
                _mark_partial(record, "RELATION_ENDPOINT_TYPE_INVALID")
                review_items.append(_review_item(
                    "cross_chunk_relation", index, "REVIEW_REQUIRED",
                    "relation_endpoint_type_invalid", summary,
                    warnings=["RELATION_ENDPOINT_TYPE_INVALID"],
                ))
            relations.append(record)
        except GraphBuilderConfigurationError as error:
            draft = _relationship_judge_draft(relationship)
            if draft is None or str(error) == "duplicate_relation":
                review_items.append(_hold("cross_chunk_relation", index, str(error), summary))
            else:
                judge_drafts.append(_judge_draft(
                    "cross_chunk_relation", index, str(error), draft
                ))
                review_items.append(_review_item(
                    "cross_chunk_relation", index, "REVIEW_REQUIRED", str(error), summary
                ))
    return CandidateNormalization(
        accepted=relations,
        review_items=review_items,
        judge_drafts=judge_drafts,
    )


def _rule_endpoint_for_evidence(
    mention: str,
    *,
    business_nodes: Sequence[Mapping[str, Any]],
    evidence_refs: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """按 mention 找冻结端点；允许唯一的空白差异，重名时再按证据位置消歧。"""
    candidates = [node for node in business_nodes if node.get("mention") == mention]
    if not candidates:
        normalized = re.sub(r"\s+", "", mention)
        candidates = [
            node for node in business_nodes
            if re.sub(r"\s+", "", str(node.get("mention", ""))) == normalized
        ]
    if len(candidates) == 1:
        return candidates[0]
    evidence_quotes = [str(ref.get("exact_quote", "")) for ref in evidence_refs]
    aligned = []
    for node in candidates:
        source_ref = node.get("source_ref")
        if not isinstance(source_ref, Mapping):
            continue
        node_quote = str(source_ref.get("exact_quote", ""))
        if node_quote and any(
            node_quote in evidence_quote or evidence_quote in node_quote
            for evidence_quote in evidence_quotes if evidence_quote
        ):
            aligned.append(node)
    return aligned[0] if len(aligned) == 1 else None


def _rule_definition_endpoints(
    rule: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """优先读取结构化端点数组，缺失时兼容历史表达式。"""
    raw_inputs = rule.get("rule_inputs")
    raw_outputs = rule.get("rule_outputs")
    raw_excluded_outputs = rule.get("rule_excluded_outputs", [])
    if (
        isinstance(raw_inputs, Sequence) and not isinstance(raw_inputs, (str, bytes))
        and isinstance(raw_outputs, Sequence) and not isinstance(raw_outputs, (str, bytes))
        and isinstance(raw_excluded_outputs, Sequence)
        and not isinstance(raw_excluded_outputs, (str, bytes))
        and all(isinstance(item, str) for item in (*raw_inputs, *raw_outputs, *raw_excluded_outputs))
    ):
        return (
            tuple(str(item) for item in raw_outputs),
            tuple(str(item) for item in raw_excluded_outputs),
            tuple(str(item) for item in raw_inputs),
        )
    outputs, inputs = _rule_expression_endpoints(str(rule.get("rule_expression", "")))
    return outputs, (), inputs


def build_rule_relationships_from_definitions(
    *, schema: Mapping[str, Any], nodes: Sequence[Mapping[str, Any]],
) -> CandidateNormalization:
    """从已校验 RuleDefinition 结构化端点确定性生成规则输入输出边。"""
    business_nodes = [
        node for node in nodes
        if node.get("entity_type") != "RuleDefinition" and node.get("extraction_status") == "VALID"
    ]
    rule_nodes = [
        node for node in nodes
        if node.get("entity_type") == "RuleDefinition" and node.get("extraction_status") == "VALID"
    ]
    relation_items: list[tuple[int, dict[str, Any]]] = []
    review_items: list[dict[str, Any]] = []
    relation_index = 0
    for rule in rule_nodes:
        rule_key = str(rule["candidate_key"])
        outputs, excluded_outputs, inputs = _rule_definition_endpoints(rule)
        evidence_refs = [
            ref for ref in rule.get("rule_evidence_refs", []) if isinstance(ref, Mapping)
        ]
        if not evidence_refs:
            continue
        source_ref = evidence_refs[0]
        role = str(source_ref.get("role", "rule_evidence"))
        endpoint_groups = (
            ("RULE_INPUT", inputs, None),
            ("RULE_OUTPUT", outputs, "ASSERTED"),
            ("RULE_OUTPUT", excluded_outputs, "EXCLUDED"),
        )
        for relation_type, mentions, output_semantics in endpoint_groups:
            for mention in mentions:
                endpoint = _rule_endpoint_for_evidence(
                    mention, business_nodes=business_nodes, evidence_refs=evidence_refs
                )
                # PREPROCESS 允许公式参数、单位和阈值不建业务节点；组合规则和输出缺失
                # 会在统一结构校验中形成明确的 PARTIAL 原因。
                if endpoint is None:
                    continue
                endpoint_key = str(endpoint["candidate_key"])
                if relation_type == "RULE_INPUT":
                    source_key, target_key = endpoint_key, rule_key
                    source, target = endpoint, rule
                else:
                    source_key, target_key = rule_key, endpoint_key
                    source, target = rule, endpoint
                record: dict[str, Any] = {
                    "candidate_key": _relation_key(
                        relation_type, source_key, target_key, source_ref
                    ),
                    "relation_type": relation_type,
                    "source_candidate_key": source_key,
                    "target_candidate_key": target_key,
                    "source_ref": source_ref,
                    "rule_evidence_role": role,
                    "generation": "deterministic_rule_structure",
                    "extraction_status": "VALID",
                    "review_status": "PENDING",
                    "publication_status": "HOLD",
                }
                if output_semantics is not None:
                    record["output_semantics"] = output_semantics
                if not _has_allowed_endpoints(schema, relation_type, source, target):
                    _mark_partial(record, "RELATION_ENDPOINT_TYPE_INVALID")
                    review_items.append(_review_item(
                        "rule", relation_index, "REVIEW_REQUIRED",
                        "relation_endpoint_type_invalid", {
                            "relation_type": relation_type,
                            "source_candidate_key": source_key,
                            "target_candidate_key": target_key,
                        }, warnings=["RELATION_ENDPOINT_TYPE_INVALID"],
                    ))
                relation_items.append((relation_index, record))
                relation_index += 1
    node_by_key = {str(node["candidate_key"]): node for node in nodes}
    relationships, structure_reviews, _invalid_rule_keys = _validate_composite_structures(
        relation_items, node_by_key=node_by_key
    )
    return CandidateNormalization(
        accepted=relationships,
        review_items=[*review_items, *structure_reviews],
        judge_drafts=[],
    )


def _validate_composite_structures(
    relation_items: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    node_by_key: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """检查规则表达式、规则边和冻结业务端点是否一致。

    该检查只用于 ``RULE_INPUT``/``RULE_OUTPUT`` 子图。缺边、端点未冻结或表达式与
    实际边不一致时，规则节点及相关边保留为 ``PARTIAL``，不直接删除，也不在这里
    判断规则医学语义是否正确。
    """
    # 先为每个冻结 RuleDefinition 建立分组，即使它没有任何边也必须被检查到。
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {
        key: []
        for key, node in node_by_key.items()
        if node.get("entity_type") == "RuleDefinition"
    }
    for index, record in relation_items:
        relation_type = record["relation_type"]
        # 根据边方向找到所属规则：RULE_INPUT 的规则在目标端，RULE_OUTPUT 在源端。
        rule_key = record["target_candidate_key"] if relation_type == "RULE_INPUT" else record["source_candidate_key"]
        if relation_type in {"RULE_INPUT", "RULE_OUTPUT"} and node_by_key.get(rule_key, {}).get(
            "entity_type"
        ) == "RuleDefinition":
            grouped.setdefault(rule_key, []).append((index, record))

    review_items: list[dict[str, Any]] = []
    # 内部原因码映射为候选工件中稳定、面向审查者的 warning。
    warning_map = {
        "rule_inputs_missing": "INPUT_ENTITY_UNRESOLVED",
        "rule_output_missing": "OUTPUT_ENTITY_UNRESOLVED",
        "composite_rule_inputs_incomplete": "RULE_INPUTS_INCOMPLETE",
        "rule_expression_input_not_frozen": "INPUT_ENTITY_UNRESOLVED",
        "rule_expression_output_not_frozen": "OUTPUT_ENTITY_UNRESOLVED",
        "rule_expression_endpoints_mismatch": "RULE_EXPRESSION_ENDPOINTS_MISMATCH",
    }
    for rule_key, items in grouped.items():
        rule = node_by_key[rule_key]
        # 分离实际输入边和输出边，并按 candidate_key 去重计数。
        inputs = [record for _index, record in items if record["relation_type"] == "RULE_INPUT"]
        outputs = [record for _index, record in items if record["relation_type"] == "RULE_OUTPUT"]
        reasons: list[str] = []
        distinct_inputs = {record["source_candidate_key"] for record in inputs}
        distinct_outputs = {record["target_candidate_key"] for record in outputs}
        if not distinct_inputs:
            reasons.append("rule_inputs_missing")
        if not distinct_outputs:
            reasons.append("rule_output_missing")
        stage = rule.get("rule_stage_candidate")
        # 完整组合或排除条件可以由一个冻结端点承载；图规则仍必须至少有一个业务输入。
        if stage == "GRAPH_COMPOSITE" and not distinct_inputs:
            reasons.append("composite_rule_inputs_incomplete")
        # 读取结构化期望端点，再与实际规则边逐项比较；旧表达式仅作为历史回放兜底。
        expected_outputs, expected_excluded_outputs, expected_inputs_tuple = (
            _rule_definition_endpoints(rule)
        )
        expected_all_outputs = (*expected_outputs, *expected_excluded_outputs)
        input_mentions = {
            str(node_by_key[record["source_candidate_key"]].get("mention", "")) for record in inputs
        }
        output_mentions = {
            str(node_by_key[record["target_candidate_key"]].get("mention", "")) for record in outputs
        }
        frozen_business_mentions = {
            str(node.get("mention", ""))
            for node in node_by_key.values()
            if node.get("entity_type") != "RuleDefinition"
        }
        normalize_endpoint = lambda value: re.sub(r"\s+", "", value)
        frozen_normalized = {normalize_endpoint(item) for item in frozen_business_mentions}
        missing_inputs: set[str] = set()
        if stage == "PREPROCESS":
            # 公式中的常量、单位和参考量可能不在冻结业务目录，只要求已冻结输入建边。
            expected_inputs = {
                item for item in expected_inputs_tuple
                if normalize_endpoint(item) in frozen_normalized
            }
        else:
            # 组合规则的业务输入必须全部已经冻结，否则规则结构不完整。
            expected_inputs = set(expected_inputs_tuple)
            missing_inputs = {
                item for item in expected_inputs
                if normalize_endpoint(item) not in frozen_normalized
            }
            if missing_inputs:
                reasons.append("rule_expression_input_not_frozen")
        missing_outputs = {
            item for item in expected_all_outputs
            if normalize_endpoint(item) not in frozen_normalized
        }
        if missing_outputs:
            reasons.append("rule_expression_output_not_frozen")
        if (
            {normalize_endpoint(item) for item in expected_inputs}
            != {normalize_endpoint(item) for item in input_mentions}
            or {normalize_endpoint(item) for item in expected_all_outputs}
            != {normalize_endpoint(item) for item in output_mentions}
        ):
            reasons.append("rule_expression_endpoints_mismatch")
        if reasons:
            # 任一结构问题都会同时标记规则节点和它已有的规则边，避免下游误用半张子图。
            warnings = tuple(sorted({warning_map[reason] for reason in reasons}))
            _mark_partial(rule, *warnings)
            for _index, record in items:
                _mark_partial(record, *warnings)
            review_items.append(_review_item(
                "rule",
                int(rule.get("_model_node_index", -1)),
                "REVIEW_REQUIRED",
                "rule_structure_incomplete",
                {
                    "rule_candidate_key": rule_key,
                    "rule_inputs": list(expected_inputs_tuple),
                    "rule_outputs": list(expected_outputs),
                    "rule_excluded_outputs": list(expected_excluded_outputs),
                    "reasons": reasons,
                    "missing_business_inputs": sorted(
                        missing_inputs
                    ) if stage != "PREPROCESS" else [],
                    "missing_business_outputs": sorted(missing_outputs),
                },
                warnings=warnings,
            ))
    # 当前宽松策略保留所有规则边；invalid_rule_keys 为空，问题通过 PARTIAL 和审查项表达。
    return [dict(record) for _index, record in relation_items], review_items, set()


if __name__ == "__main__":
    # 离线关系硬校验 demo：读取真实 chunk，使用此前真实实体模型结果和真实关系
    # 模型结果，不创建客户端、不调用大模型、不写候选工件或 Neo4j。
    # 可自行打印 entity_result、entity_nodes、relationship_graph 或 relationship_result。
    from neo4j_graphrag.experimental.components.types import Neo4jRelationship

    from medical_kg_sourceprep.extraction.graph_builder.contract import (
        BUSINESS_NODE_TYPES,
        DEFAULT_CHUNK_MANIFEST,
        DEFAULT_SCHEMA_PATH,
        ORDINARY_RELATION_TYPES,
    )
    from medical_kg_sourceprep.extraction.graph_builder.schema import load_candidate_graph_schema
    from medical_kg_sourceprep.extraction.graph_builder.validation.nodes import normalize_candidate_nodes
    from medical_kg_sourceprep.extraction.llm_extraction import load_chunk_manifest

    chunk_id = "clinical-hematology:chapter-01:0012:0000"
    schema = load_candidate_graph_schema(DEFAULT_SCHEMA_PATH)
    _manifest, chunks = load_chunk_manifest(DEFAULT_CHUNK_MANIFEST)
    chunk = next(item for item in chunks if item.chunk_id == chunk_id)

    # 该 chunk 此前真实实体模型返回的轻量语义字段。重复项原样保留，让节点硬校验
    # 先生成稳定 candidate_key 并去重，再为关系硬校验提供冻结端点。
    raw_entity_nodes = [
        ("ClinicalContext", "严重的肝病", "原文明示的疾病背景，影响检验结果解释。"),
        ("ClinicalContext", "营养不良", "原文明示的临床背景，影响检验结果解释。"),
        ("LabIndicator", "转铁蛋白", "原文明示的检验指标。"),
        ("IndicatorState", "转铁蛋白合成减少", "原文明示该指标的减少状态。"),
        ("ClinicalContext", "肾病综合征", "原文明示的疾病背景，影响检验结果解释。"),
        ("ClinicalContext", "大量蛋白质从尿液丢失", "原文明示的病理机制背景。"),
        ("LabIndicator", "转铁蛋白", "原文明示的检验指标。"),
        ("IndicatorState", "转铁蛋白减少", "原文明示该指标的减少状态。"),
        ("LabIndicator", "总铁结合力", "原文明示的检验指标。"),
        ("LabIndicator", "血清铁", "原文明示的检验指标。"),
        ("LabIndicator", "总铁结合力", "原文明示的检验指标。"),
        ("LabIndicator", "亚铁嗪显色法", "原文明示的检验方法，作为检验指标背景。"),
    ]
    entity_graph = Neo4jGraph(nodes=[
        Neo4jNode(
            id=f"recorded-entity-{index}",
            label=label,
            properties={"mention": mention, "extraction_reason": reason},
        )
        for index, (label, mention, reason) in enumerate(raw_entity_nodes)
    ])
    entity_result = normalize_candidate_nodes(
        entity_graph,
        chunk=chunk,
        schema=schema,
        allowed_node_types=BUSINESS_NODE_TYPES,
        derive_entity_provenance=True,
    )
    entity_nodes = [
        node for node in entity_result.accepted if node.get("extraction_status") == "VALID"
    ]
    key_by_mention = {node["mention"]: node["candidate_key"] for node in entity_nodes}

    # 以下四条是该 chunk 最近一次真实普通关系模型输出。这里重新包装为 Neo4jGraph，
    # 只演示模型返回后 normalize_candidate_relationships() 的本地处理。
    first_quote = "严重的肝病、营养不良使转铁蛋白合成减少。"
    second_quote = "肾病综合征时, 大量蛋白质从尿液丢失, 使转铁蛋白减少。"
    recorded_relationships = [
        ("严重的肝病", "转铁蛋白合成减少", first_quote),
        ("营养不良", "转铁蛋白合成减少", first_quote),
        ("肾病综合征", "转铁蛋白减少", second_quote),
        ("大量蛋白质从尿液丢失", "转铁蛋白减少", second_quote),
    ]
    relationship_graph = Neo4jGraph(relationships=[
        Neo4jRelationship(
            start_node_id=f"{chunk.chunk_id}:{key_by_mention[source_mention]}",
            end_node_id=f"{chunk.chunk_id}:{key_by_mention[target_mention]}",
            type="CAUSES",
            properties={"exact_quote": quote},
        )
        for source_mention, target_mention, quote in recorded_relationships
    ])

    # 结果分为 accepted、review_items 和 judge_drafts。当前宽松策略下，语义方向、
    # 中间机制和联合条件不由本地判断；只要结构与证据可回放，就保留 VALID/HOLD。
    relationship_result = normalize_candidate_relationships(
        relationship_graph,
        chunk=chunk,
        schema=schema,
        nodes=entity_nodes,
        allowed_relation_types=sorted(ORDINARY_RELATION_TYPES),
        validate_rule_structures=False,
    )
    print(relationship_result)
