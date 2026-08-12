"""普通关系、规则边和 RuleDefinition 图结构的本地校验。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from neo4j_graphrag.experimental.components.types import Neo4jGraph

from ..contract import MODEL_RELATION_TYPES, GraphBuilderConfigurationError
from ..schema import _relation_endpoint_pairs
from ...llm_extraction import EvidenceChunk
from .provenance import _relation_key, _rule_expression_endpoints, _source_ref
from .result import CandidateNormalization
from .review import (
    _hold,
    _judge_draft,
    _mark_partial,
    _node_summary,
    _relationship_judge_draft,
    _relationship_summary,
    _review_item,
)


def _strip_chunk_prefix(value: str, chunk_id: str) -> str | None:
    """只接受当前 chunk 命名空间下的候选键。"""
    prefix = f"{chunk_id}:"
    return value[len(prefix):] if value.startswith(prefix) else None


def _has_allowed_endpoints(
    schema: Mapping[str, Any], relation_type: str, source: Mapping[str, Any], target: Mapping[str, Any]
) -> bool:
    """核对关系端点类型是否在 Schema 的允许组合中。"""
    return (source["entity_type"], target["entity_type"]) in _relation_endpoint_pairs(schema, relation_type)


def deterministic_state_relations(nodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """仅对已本地绑定的状态生成 HAS_STATE，不让模型自行猜测该边。"""
    relations = []
    for node in nodes:
        if node["entity_type"] != "IndicatorState" or "bound_indicator_candidate_key" not in node:
            continue
        source_ref = node["source_ref"]
        relations.append(
            {
                "candidate_key": _relation_key(
                    "HAS_STATE", node["bound_indicator_candidate_key"], node["candidate_key"], source_ref
                ),
                "relation_type": "HAS_STATE",
                "source_candidate_key": node["bound_indicator_candidate_key"],
                "target_candidate_key": node["candidate_key"],
                "source_ref": source_ref,
                "generation": "deterministic_state_binding",
                "extraction_status": "VALID",
                "review_status": "PENDING",
                "publication_status": "HOLD",
            }
        )
    return relations


def _rule_relation_source_ref(
    relationship: Any, relation_type: str, source: Mapping[str, Any], target: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """规则边只能引用所属 RuleDefinition 已回放的证据角色。"""
    if relation_type not in {"RULE_INPUT", "RULE_OUTPUT"}:
        return None
    rule = target if relation_type == "RULE_INPUT" else source
    if rule.get("entity_type") != "RuleDefinition":
        raise GraphBuilderConfigurationError("rule_relation_definition_missing")
    role = relationship.properties.get("rule_evidence_role")
    if not isinstance(role, str):
        raise GraphBuilderConfigurationError("rule_relation_evidence_role_missing")
    for evidence_ref in rule.get("rule_evidence_refs", []):
        if evidence_ref.get("role") == role:
            return evidence_ref
    raise GraphBuilderConfigurationError("rule_relation_evidence_role_unknown")


def normalize_candidate_relationships(
    graph: Neo4jGraph,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    allowed_relation_types: Sequence[str] = MODEL_RELATION_TYPES,
    include_deterministic_state: bool = True,
    validate_rule_structures: bool = True,
    return_invalid_rule_keys: bool = False,
) -> CandidateNormalization:
    """接纳可回放关系，并将无法入图的最小关系分流给 Judge。"""
    node_by_key = {item["candidate_key"]: item for item in nodes}
    relations = deterministic_state_relations(nodes) if include_deterministic_state else []
    holds: list[dict[str, Any]] = []
    judge_drafts: list[dict[str, Any]] = []
    seen_keys = {item["candidate_key"] for item in relations}
    model_relations: list[tuple[int, dict[str, Any]]] = []

    # 关系阶段不允许新增节点，关系必须引用先前冻结的 candidate_key。
    for index, node in enumerate(graph.nodes):
        from .review import _node_judge_draft
        draft = _node_judge_draft(node)
        if draft is None:
            holds.append(_hold("relation", index, "relation_phase_node_not_allowed", _node_summary(node)))
        else:
            judge_drafts.append(_judge_draft("relation", index, "relation_phase_node_not_allowed", draft))
            holds.append(_review_item("relation", index, "REVIEW_REQUIRED", "relation_phase_node_not_allowed", _node_summary(node)))
    for index, relationship in enumerate(graph.relationships):
        summary = _relationship_summary(relationship)
        try:
            relation_type = relationship.type
            if relation_type not in allowed_relation_types:
                raise GraphBuilderConfigurationError("relation_type_not_enabled_for_trial")
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
            if source is None or target is None:
                summary["missing_endpoint_candidate_keys"] = [
                    key for key in (source_key, target_key) if key not in node_by_key
                ]
                for endpoint in (source, target):
                    if endpoint is not None and endpoint.get("entity_type") == "RuleDefinition":
                        _mark_partial(endpoint, "RULE_ENDPOINT_UNRESOLVED")
                raise GraphBuilderConfigurationError("relation_endpoint_not_from_frozen_catalog")

            properties = relationship.properties
            rule_source_ref = _rule_relation_source_ref(relationship, relation_type, source, target)
            source_ref = rule_source_ref if rule_source_ref is not None else _source_ref(
                chunk,
                source["mention"],
                properties.get("exact_quote"),
                exact_quote_occurrence_index=properties.get("exact_quote_occurrence_index"),
                source_char_start=properties.get("source_char_start"),
                source_char_end=properties.get("source_char_end"),
            )
            if rule_source_ref is None and target["mention"] not in source_ref["exact_quote"]:
                raise GraphBuilderConfigurationError("relation_quote_lacks_endpoint")

            warnings: list[str] = []
            reasons: list[str] = []
            if source_key == target_key or not _has_allowed_endpoints(schema, relation_type, source, target):
                warnings.append("RELATION_ENDPOINT_TYPE_INVALID")
                reasons.append("relation_endpoint_type_invalid")
                for endpoint in (source, target):
                    if endpoint.get("entity_type") == "RuleDefinition":
                        _mark_partial(endpoint, "RULE_ENDPOINT_TYPE_INVALID")
            if relation_type not in {"HAS_METRIC", "RULE_INPUT", "RULE_OUTPUT"}:
                cue = properties.get("relation_cue")
                if not isinstance(cue, str) or not cue.strip():
                    raise GraphBuilderConfigurationError("relation_cue_invalid")
                if cue not in source_ref["exact_quote"]:
                    raise GraphBuilderConfigurationError("relation_cue_not_in_exact_quote")
                source_start = source_ref["exact_quote"].find(source["mention"])
                target_start = source_ref["exact_quote"].find(target["mention"])
                cue_start = source_ref["exact_quote"].find(cue)
                if relation_type in {"CAUSES", "INDICATES", "IS_A"} and not (
                    source_start < cue_start < target_start
                ):
                    warnings.append("RELATION_DIRECTION_UNCERTAIN")
                    reasons.append("relation_direction_not_verbatim")
                quote_states = {
                    item["candidate_key"]
                    for item in node_by_key.values()
                    if item["entity_type"] == "IndicatorState" and item["mention"] in source_ref["exact_quote"]
                }
                if len(quote_states) >= 2:
                    warnings.append("RELATION_MAY_BE_JOINT_CONDITION")
                    reasons.append("relation_may_be_joint_condition")
            else:
                cue = None
            candidate_key = _relation_key(relation_type, source_key, target_key, source_ref)
            if candidate_key in seen_keys:
                raise GraphBuilderConfigurationError("duplicate_relation")
            seen_keys.add(candidate_key)
            record = {
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
            if cue is not None:
                record["relation_cue"] = cue
            if warnings:
                _mark_partial(record, *warnings)
                holds.append(_review_item(
                    "relation", index, "REVIEW_REQUIRED", reasons[0], summary, warnings=warnings,
                ))
            model_relations.append((index, record))
        except GraphBuilderConfigurationError as error:
            draft = _relationship_judge_draft(relationship)
            if draft is None or str(error) == "duplicate_relation":
                holds.append(_hold("relation", index, str(error), summary))
            else:
                judge_drafts.append(_judge_draft("relation", index, str(error), draft))
                holds.append(_review_item("relation", index, "REVIEW_REQUIRED", str(error), summary))

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


def _validate_composite_structures(
    relation_items: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    node_by_key: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """检查规则表达式、规则边和冻结业务端点是否一致；不完整候选保留为 PARTIAL。"""
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {
        key: []
        for key, node in node_by_key.items()
        if node.get("entity_type") == "RuleDefinition"
    }
    for index, record in relation_items:
        relation_type = record["relation_type"]
        rule_key = record["target_candidate_key"] if relation_type == "RULE_INPUT" else record["source_candidate_key"]
        if relation_type in {"RULE_INPUT", "RULE_OUTPUT"} and node_by_key.get(rule_key, {}).get(
            "entity_type"
        ) == "RuleDefinition":
            grouped.setdefault(rule_key, []).append((index, record))

    review_items: list[dict[str, Any]] = []
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
        if stage == "GRAPH_COMPOSITE" and len(distinct_inputs) < 2:
            reasons.append("composite_rule_inputs_incomplete")
        expression_outputs, expression_inputs = _rule_expression_endpoints(rule["rule_expression"])
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
        if stage == "PREPROCESS":
            expected_inputs = set(expression_inputs) & frozen_business_mentions
        else:
            expected_inputs = set(expression_inputs)
            missing_inputs = expected_inputs - frozen_business_mentions
            if missing_inputs:
                reasons.append("rule_expression_input_not_frozen")
        missing_outputs = set(expression_outputs) - frozen_business_mentions
        if missing_outputs:
            reasons.append("rule_expression_output_not_frozen")
        if expected_inputs != input_mentions or set(expression_outputs) != output_mentions:
            reasons.append("rule_expression_endpoints_mismatch")
        if reasons:
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
                    "rule_expression": rule["rule_expression"],
                    "reasons": reasons,
                    "missing_business_inputs": sorted(
                        set(expression_inputs) - frozen_business_mentions
                    ) if stage != "PREPROCESS" else [],
                    "missing_business_outputs": sorted(missing_outputs),
                },
                warnings=warnings,
            ))
    return [dict(record) for _index, record in relation_items], review_items, set()
