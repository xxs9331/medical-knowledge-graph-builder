"""业务实体和 RuleDefinition 候选节点的本地接纳逻辑。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from neo4j_graphrag.experimental.components.types import Neo4jGraph

from ..contract import TRIAL_NODE_TYPES, GraphBuilderConfigurationError
from ...llm_extraction import EvidenceChunk
from .provenance import (
    _candidate_key,
    _normalize_rule_expression,
    _parse_rule_evidence,
    _parse_table_state_evidence,
    _rule_candidate_key,
    _source_ref,
    _table_state_candidate_key,
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


def normalize_candidate_nodes(
    graph: Neo4jGraph,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    allowed_node_types: Sequence[str] = TRIAL_NODE_TYPES,
) -> CandidateNormalization:
    """接纳节点或将其分流为图内 PARTIAL、Judge 草稿、重复审计项。"""
    del schema  # Schema 已在加载时验证；当前阶段只使用允许节点类型这个切片。
    accepted: list[dict[str, Any]] = []
    pending_states: list[tuple[int, dict[str, Any], str]] = []
    pending_rules: list[tuple[int, str, str, str, list[dict[str, Any]], tuple[str, ...], Any]] = []
    holds: list[dict[str, Any]] = []
    judge_drafts: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    # 第一轮只验证每个模型节点自身；跨节点绑定在后续统一处理。
    for index, node in enumerate(graph.nodes):
        summary = _node_summary(node)
        try:
            entity_type = node.label
            properties = node.properties
            if entity_type not in allowed_node_types:
                raise GraphBuilderConfigurationError("entity_type_not_enabled_for_trial")
            if entity_type == "RuleDefinition":
                if any(properties.get(field) not in (None, "") for field in (
                    "mention", "canonical_name_candidate", "exact_quote"
                )):
                    raise GraphBuilderConfigurationError("rule_definition_uses_business_fields")
                raw_rule_stage = properties.get("rule_stage_candidate")
                rule_warnings: list[str] = []
                if raw_rule_stage in {"PREPROCESS", "GRAPH_COMPOSITE", "UNKNOWN"}:
                    rule_stage = raw_rule_stage
                else:
                    rule_stage = "UNKNOWN"
                    rule_warnings.append("RULE_STAGE_UNKNOWN")
                expression, _output, expression_name = _normalize_rule_expression(
                    properties.get("rule_expression")
                )
                rule_name = properties.get("rule_name")
                if not isinstance(rule_name, str) or not rule_name.strip():
                    rule_name = f"来源规则:{expression_name}"
                    rule_warnings.append("RULE_NAME_FALLBACK")
                evidence_refs = _parse_rule_evidence(chunk, properties.get("rule_evidence_json"))
                pending_rules.append((
                    index,
                    rule_stage,
                    expression,
                    rule_name.strip(),
                    evidence_refs,
                    tuple(rule_warnings),
                    raw_rule_stage,
                ))
                continue

            mention = properties.get("mention")
            canonical = properties.get("canonical_name_candidate")
            if not isinstance(mention, str) or not mention:
                raise GraphBuilderConfigurationError("mention_missing")
            if not isinstance(canonical, str) or not canonical:
                raise GraphBuilderConfigurationError("canonical_name_missing")
            table_state_evidence = properties.get("table_state_evidence_json")
            if table_state_evidence is not None:
                if entity_type != "IndicatorState":
                    raise GraphBuilderConfigurationError("table_state_evidence_requires_indicator_state")
                if properties.get("exact_quote") not in (None, ""):
                    raise GraphBuilderConfigurationError("table_state_evidence_uses_exact_quote")
                source_ref, table_state_evidence_refs = _parse_table_state_evidence(
                    chunk, value=table_state_evidence
                )
                candidate_key = _table_state_candidate_key(
                    chunk=chunk, mention=mention, evidence_refs=table_state_evidence_refs
                )
            else:
                source_ref = _source_ref(
                    chunk,
                    mention,
                    properties.get("exact_quote"),
                    exact_quote_occurrence_index=properties.get("exact_quote_occurrence_index"),
                    mention_occurrence_index=properties.get("mention_occurrence_index"),
                    source_char_start=properties.get("source_char_start"),
                    source_char_end=properties.get("source_char_end"),
                )
                if canonical not in source_ref["exact_quote"]:
                    raise GraphBuilderConfigurationError("canonical_name_not_in_exact_quote")
                table_state_evidence_refs = []
                candidate_key = _candidate_key(entity_type, mention, source_ref)
            if candidate_key in seen_keys:
                raise GraphBuilderConfigurationError("duplicate_candidate")
            seen_keys.add(candidate_key)
            record = {
                "candidate_key": candidate_key,
                "entity_type": entity_type,
                "mention": mention,
                "canonical_name_candidate": canonical,
                "source_ref": source_ref,
                "extraction_status": "VALID",
                "review_status": "PENDING",
                "publication_status": "HOLD",
            }
            if table_state_evidence_refs:
                record["table_state_evidence_refs"] = table_state_evidence_refs
            if entity_type == "IndicatorState":
                binding = properties.get("bound_indicator_mention")
                if not isinstance(binding, str) or not binding:
                    _mark_partial(record, "STATE_INDICATOR_UNRESOLVED")
                    accepted.append(record)
                    holds.append(_review_item(
                        "entity", index, "REVIEW_REQUIRED", "state_indicator_binding_missing",
                        {"mention": record["mention"]}, warnings=("STATE_INDICATOR_UNRESOLVED",),
                    ))
                else:
                    pending_states.append((index, record, binding))
            else:
                accepted.append(record)
        except GraphBuilderConfigurationError as error:
            draft = _node_judge_draft(node)
            if draft is None or str(error) in {"duplicate_candidate"}:
                holds.append(_hold("entity", index, str(error), summary))
            else:
                judge_drafts.append(_judge_draft("entity", index, str(error), draft))
                holds.append(_review_item(
                    "entity", index, "REVIEW_REQUIRED", str(error), summary,
                ))

    # IndicatorState 只能绑定到同一次响应中唯一的 LabIndicator；无法唯一绑定时保留为 PARTIAL。
    indicators = [item for item in accepted if item["entity_type"] == "LabIndicator"]
    for index, record, binding in pending_states:
        matches = [item for item in indicators if item["mention"] == binding]
        if len(matches) != 1:
            _mark_partial(record, "STATE_INDICATOR_UNRESOLVED")
            accepted.append(record)
            holds.append(_review_item(
                "entity",
                index,
                "REVIEW_REQUIRED",
                "state_indicator_binding_not_unique",
                {"mention": record["mention"], "bound_indicator_mention": binding},
                warnings=("STATE_INDICATOR_UNRESOLVED",),
            ))
            continue
        record["bound_indicator_candidate_key"] = matches[0]["candidate_key"]
        accepted.append(record)

    # 规则节点与业务实体不同：规则身份由表达式与证据位置共同决定。
    for index, rule_stage, expression, rule_name, evidence_refs, rule_warnings, raw_rule_stage in pending_rules:
        candidate_key = _rule_candidate_key(
            chunk=chunk,
            rule_stage=rule_stage,
            rule_expression=expression,
            rule_evidence_refs=evidence_refs,
        )
        if candidate_key in seen_keys:
            holds.append(_hold(
                "entity", index, "duplicate_rule_identity",
                {"rule_candidate_key": candidate_key, "rule_expression": expression},
            ))
            continue
        seen_keys.add(candidate_key)
        record = {
            "candidate_key": candidate_key,
            "rule_candidate_key": candidate_key,
            "entity_type": "RuleDefinition",
            "rule_expression": expression,
            "rule_name": rule_name,
            "rule_stage_candidate": rule_stage,
            "rule_evidence_refs": evidence_refs,
            "extraction_status": "VALID",
            "review_status": "PENDING",
            "publication_status": "HOLD",
            "_model_node_index": index,
        }
        if rule_stage == "UNKNOWN" or rule_warnings:
            warnings = tuple(sorted(set((*rule_warnings, "RULE_STAGE_UNKNOWN" if rule_stage == "UNKNOWN" else ""))))
            warnings = tuple(warning for warning in warnings if warning)
            _mark_partial(record, *warnings)
            holds.append(_review_item(
                "rule",
                index,
                "REVIEW_REQUIRED",
                "rule_stage_unknown" if raw_rule_stage == "UNKNOWN" else "rule_stage_candidate_invalid",
                {
                    "rule_candidate_key": candidate_key,
                    "rule_expression": expression,
                    "rule_stage_candidate": raw_rule_stage,
                },
                warnings=warnings,
            ))
        accepted.append(record)

    # 阶段错位关系不自动搬运，但最小关系草稿可由未来 Judge 重路由。
    for index, relationship in enumerate(graph.relationships):
        draft = _relationship_judge_draft(relationship)
        if draft is None:
            holds.append(_hold(
                "entity", index, "entity_phase_relationship_not_allowed", _relationship_summary(relationship)
            ))
        else:
            judge_drafts.append(_judge_draft("entity", index, "entity_phase_relationship_not_allowed", draft))
            holds.append(_review_item(
                "entity", index, "REVIEW_REQUIRED", "entity_phase_relationship_not_allowed",
                _relationship_summary(relationship),
            ))
    return CandidateNormalization(accepted=accepted, review_items=holds, judge_drafts=judge_drafts)


def _catalog_for_prompt(nodes: Sequence[Mapping[str, Any]]) -> str:
    """把已接纳节点裁剪为后续模型阶段唯一可引用的冻结目录。"""
    import json

    catalog = []
    for item in nodes:
        # PARTIAL 仍供审计与 Judge 使用，但不能成为后续模型阶段的冻结端点。
        if item.get("extraction_status") != "VALID":
            continue
        entry = {"candidate_key": item["candidate_key"], "entity_type": item["entity_type"]}
        if item["entity_type"] == "RuleDefinition":
            entry.update({
                "rule_expression": item["rule_expression"],
                "rule_name": item["rule_name"],
                "rule_evidence_roles": [ref["role"] for ref in item["rule_evidence_refs"]],
            })
        else:
            entry.update({
                "mention": item["mention"],
                "canonical_name_candidate": item["canonical_name_candidate"],
            })
        catalog.append(entry)
    return json.dumps({"frozen_candidate_catalog": catalog}, ensure_ascii=False, sort_keys=True)
