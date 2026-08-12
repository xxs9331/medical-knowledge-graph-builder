"""Local provenance replay, candidate normalization, and rule-structure validation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from neo4j_graphrag.experimental.components.types import Neo4jGraph

from .contract import (
    CANDIDATE_RUN_VERSION,
    MODEL_RELATION_TYPES,
    RULE_EXPRESSION_PATTERN,
    TRIAL_NODE_TYPES,
    GraphBuilderConfigurationError,
)
from .schema import _relation_endpoint_pairs
from ..llm_extraction import EvidenceChunk


def _review_item(
    stage: str,
    index: int,
    status: str,
    reason_code: str,
    summary: Mapping[str, Any],
    *,
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    if status not in {"REVIEW_REQUIRED", "REJECTED"}:
        raise GraphBuilderConfigurationError("review_status_invalid")
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
        "review_id": f"hold:{identity}",
        "stage": stage,
        "status": status,
        "reason_code": reason_code,
        "candidate_summary": dict(summary),
        **({"warnings": sorted(set(warnings))} if warnings else {}),
    }


def _hold(stage: str, index: int, reason_code: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for records that cannot be retained as candidates."""
    return _review_item(stage, index, "REJECTED", reason_code, summary)


def _mark_partial(record: Mapping[str, Any], *warnings: str) -> None:
    """Retain incomplete candidate material while blocking publication and requesting review."""
    if not isinstance(record, dict):
        return
    current = record.setdefault("warnings", [])
    if not isinstance(current, list):
        current = []
        record["warnings"] = current
    for warning in warnings:
        if warning not in current:
            current.append(warning)
    record["extraction_status"] = "PARTIAL"
    record["review_status"] = "REVIEW_REQUIRED"


def _node_summary(node: Any) -> dict[str, Any]:
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
    properties = getattr(relationship, "properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    return {
        "relation_type": str(getattr(relationship, "type", ""))[:80],
        "start_node_id": str(getattr(relationship, "start_node_id", ""))[:160],
        "end_node_id": str(getattr(relationship, "end_node_id", ""))[:160],
        "relation_cue": str(properties.get("relation_cue", ""))[:80],
    }


def _is_source_offset(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _occurrence_starts(text: str, needle: str, *, start: int = 0, end: int | None = None) -> list[int]:
    if not needle:
        return []
    limit = len(text) if end is None else end
    starts: list[int] = []
    position = text.find(needle, start, limit)
    while position >= 0:
        starts.append(position)
        position = text.find(needle, position + len(needle), limit)
    return starts


def _replay_source_span(
    chunk: EvidenceChunk,
    exact_quote: str,
    *,
    exact_quote_occurrence_index: Any = None,
    source_char_start: Any = None,
    source_char_end: Any = None,
) -> tuple[int, int]:
    """Resolve a quote to one exact source span without interpreting its content."""
    has_start = source_char_start is not None
    has_end = source_char_end is not None
    if has_start != has_end:
        raise GraphBuilderConfigurationError("quote_position_incomplete")
    if has_start:
        if (
            _is_source_offset(source_char_start)
            and _is_source_offset(source_char_end)
            and 0 <= source_char_start < source_char_end <= len(chunk.text)
            and chunk.text[source_char_start:source_char_end] == exact_quote
        ):
            return source_char_start, source_char_end
    starts = _occurrence_starts(chunk.text, exact_quote)
    if not starts:
        raise GraphBuilderConfigurationError("quote_absent_or_ambiguous")
    if len(starts) == 1:
        return starts[0], starts[0] + len(exact_quote)
    if not _is_source_offset(exact_quote_occurrence_index):
        raise GraphBuilderConfigurationError("quote_absent_or_ambiguous")
    if not 0 <= exact_quote_occurrence_index < len(starts):
        raise GraphBuilderConfigurationError("quote_occurrence_index_invalid")
    start = starts[exact_quote_occurrence_index]
    return start, start + len(exact_quote)


def _source_ref(
    chunk: EvidenceChunk,
    mention: str,
    exact_quote: Any,
    *,
    exact_quote_occurrence_index: Any = None,
    mention_occurrence_index: Any = None,
    source_char_start: Any = None,
    source_char_end: Any = None,
) -> dict[str, Any]:
    if not isinstance(exact_quote, str) or not exact_quote:
        raise GraphBuilderConfigurationError("exact_quote is missing")
    try:
        quote_start, quote_end = _replay_source_span(
            chunk,
            exact_quote,
            exact_quote_occurrence_index=exact_quote_occurrence_index,
            source_char_start=source_char_start,
            source_char_end=source_char_end,
        )
    except GraphBuilderConfigurationError as error:
        raise GraphBuilderConfigurationError(f"source_ref_{error}") from error
    mention_starts = _occurrence_starts(chunk.text, mention, start=quote_start, end=quote_end)
    if not mention_starts:
        raise GraphBuilderConfigurationError("mention_not_unique_in_exact_quote")
    if len(mention_starts) == 1:
        mention_start = mention_starts[0]
    elif _is_source_offset(mention_occurrence_index) and 0 <= mention_occurrence_index < len(mention_starts):
        mention_start = mention_starts[mention_occurrence_index]
    else:
        raise GraphBuilderConfigurationError("mention_not_unique_in_exact_quote")
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_sha256": chunk.chunk_sha256,
        "exact_quote": exact_quote,
        "char_start": quote_start,
        "char_end": quote_end,
        "mention_char_start": mention_start,
        "mention_char_end": mention_start + len(mention),
    }


def _candidate_key(entity_type: str, mention: str, source_ref: Mapping[str, Any]) -> str:
    mention_start = source_ref.get("mention_char_start")
    if not _is_source_offset(mention_start):
        mention_start = source_ref["char_start"] + source_ref["exact_quote"].find(mention)
    raw = (
        f"{CANDIDATE_RUN_VERSION}:{entity_type}:{source_ref['chunk_id']}:"
        f"{mention_start}:{mention_start + len(mention)}"
    )
    return f"candidate:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def _table_state_candidate_key(
    *, chunk: EvidenceChunk, mention: str, evidence_refs: Sequence[Mapping[str, Any]]
) -> str:
    """Identify a model-only table state by its raw header and row anchors."""
    raw = json.dumps(
        {
            "version": CANDIDATE_RUN_VERSION,
            "entity_type": "IndicatorState",
            "chunk_id": chunk.chunk_id,
            "mention": mention,
            "evidence_positions": [
                {"role": ref["role"], "char_start": ref["char_start"], "char_end": ref["char_end"]}
                for ref in evidence_refs
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"candidate:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def _parse_table_state_evidence(
    chunk: EvidenceChunk, *, value: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replay raw table anchors without interpreting their medical semantics."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise GraphBuilderConfigurationError("table_state_evidence_json_invalid") from error
    if not isinstance(value, Mapping):
        raise GraphBuilderConfigurationError("table_state_evidence_missing")
    refs = []
    for role, key in (("table_header", "header_exact_quote"), ("table_row", "row_exact_quote")):
        quote = value.get(key)
        if not isinstance(quote, str) or not quote:
            raise GraphBuilderConfigurationError("table_state_evidence_quote_missing")
        try:
            start, end = _replay_source_span(
                chunk,
                quote,
                exact_quote_occurrence_index=value.get(f"{role}_occurrence_index"),
                source_char_start=value.get(f"{role}_char_start"),
                source_char_end=value.get(f"{role}_char_end"),
            )
        except GraphBuilderConfigurationError as error:
            raise GraphBuilderConfigurationError(f"table_state_evidence_{error}") from error
        refs.append({
            "role": role,
            "chunk_id": chunk.chunk_id,
            "chunk_sha256": chunk.chunk_sha256,
            "exact_quote": quote,
            "char_start": start,
            "char_end": end,
        })
    # Keep the row as the primary locator; the companion header remains in table_state_evidence_refs.
    row = refs[1]
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_sha256": chunk.chunk_sha256,
        "exact_quote": row["exact_quote"],
        "char_start": row["char_start"],
        "char_end": row["char_end"],
    }, refs


def _rule_evidence_ref(
    chunk: EvidenceChunk,
    *,
    role: Any,
    exact_quote: Any,
    exact_quote_occurrence_index: Any = None,
    source_char_start: Any = None,
    source_char_end: Any = None,
) -> dict[str, Any]:
    """Replay one model-selected rule evidence span without interpreting its role."""
    if not isinstance(role, str) or not role.strip():
        raise GraphBuilderConfigurationError("rule_evidence_role_invalid")
    if not isinstance(exact_quote, str) or not exact_quote:
        raise GraphBuilderConfigurationError("rule_evidence_quote_missing")
    try:
        quote_start, quote_end = _replay_source_span(
            chunk,
            exact_quote,
            exact_quote_occurrence_index=exact_quote_occurrence_index,
            source_char_start=source_char_start,
            source_char_end=source_char_end,
        )
    except GraphBuilderConfigurationError as error:
        raise GraphBuilderConfigurationError(f"rule_evidence_{error}") from error
    return {
        "role": role,
        "chunk_id": chunk.chunk_id,
        "chunk_sha256": chunk.chunk_sha256,
        "exact_quote": exact_quote,
        "char_start": quote_start,
        "char_end": quote_end,
    }


def _normalize_rule_expression(value: Any) -> tuple[str, str, str]:
    if not isinstance(value, str) or not value.strip():
        raise GraphBuilderConfigurationError("rule_expression_missing")
    expression = re.sub(r"\s+", "", value)
    matched = RULE_EXPRESSION_PATTERN.fullmatch(expression)
    if matched is None or not matched.group("inputs"):
        raise GraphBuilderConfigurationError("rule_expression_invalid")
    return expression, matched.group("output"), matched.group("name")


def _rule_expression_endpoints(expression: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    matched = RULE_EXPRESSION_PATTERN.fullmatch(expression)
    if matched is None:
        raise GraphBuilderConfigurationError("rule_expression_invalid")
    output = matched.group("output")
    if output.startswith("[") and output.endswith("]"):
        outputs = tuple(item for item in output[1:-1].split(",") if item)
    else:
        outputs = (output,)
    inputs = tuple(item for item in matched.group("inputs").split(",") if item)
    if not outputs or not inputs:
        raise GraphBuilderConfigurationError("rule_expression_invalid")
    return outputs, inputs


def _parse_rule_evidence(chunk: EvidenceChunk, value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise GraphBuilderConfigurationError("rule_evidence_json_invalid") from error
    if not isinstance(value, list) or not value:
        raise GraphBuilderConfigurationError("rule_evidence_missing")
    refs: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise GraphBuilderConfigurationError("rule_evidence_item_invalid")
        role = item.get("role")
        if not isinstance(role, str) or not role.strip():
            raise GraphBuilderConfigurationError("rule_evidence_role_invalid")
        refs.append(_rule_evidence_ref(
            chunk,
            role=role,
            exact_quote=item.get("exact_quote"),
            exact_quote_occurrence_index=item.get("exact_quote_occurrence_index"),
            source_char_start=item.get("source_char_start"),
            source_char_end=item.get("source_char_end"),
        ))
    return refs


def _rule_candidate_key(
    *, chunk: EvidenceChunk, rule_stage: str, rule_expression: str, rule_evidence_refs: Sequence[Mapping[str, Any]]
) -> str:
    evidence_positions = [
        {"role": item["role"], "char_start": item["char_start"], "char_end": item["char_end"]}
        for item in rule_evidence_refs
    ]
    raw = json.dumps(
        {
            "version": CANDIDATE_RUN_VERSION,
            "chunk_id": chunk.chunk_id,
            "rule_stage": rule_stage,
            "rule_expression": rule_expression,
            "rule_evidence_positions": evidence_positions,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"candidate:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def _relation_key(relation_type: str, source_key: str, target_key: str, source_ref: Mapping[str, Any]) -> str:
    raw = f"{CANDIDATE_RUN_VERSION}:{relation_type}:{source_key}:{target_key}:{source_ref['chunk_id']}:{source_ref['char_start']}:{source_ref['char_end']}"
    return f"relation:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def normalize_candidate_nodes(
    graph: Neo4jGraph,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    allowed_node_types: Sequence[str] = TRIAL_NODE_TYPES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate candidate nodes and locally bind each IndicatorState."""
    del schema  # The complete schema was checked on load; this phase uses its trial slice.
    accepted: list[dict[str, Any]] = []
    pending_states: list[tuple[int, dict[str, Any], str]] = []
    pending_rules: list[tuple[int, str, str, str, list[dict[str, Any]], tuple[str, ...], Any]] = []
    holds: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

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
                    raise GraphBuilderConfigurationError("state_indicator_binding_missing")
                pending_states.append((index, record, binding))
            else:
                accepted.append(record)
        except GraphBuilderConfigurationError as error:
            holds.append(_hold("entity", index, str(error), summary))

    indicators = [item for item in accepted if item["entity_type"] == "LabIndicator"]
    for index, record, binding in pending_states:
        matches = [item for item in indicators if item["mention"] == binding]
        if len(matches) != 1:
            _mark_partial(record, "STATE_INDICATOR_UNRESOLVED")
            accepted.append(record)
            holds.append(
                _review_item(
                    "entity",
                    index,
                    "REVIEW_REQUIRED",
                    "state_indicator_binding_not_unique",
                    {"mention": record["mention"], "bound_indicator_mention": binding},
                    warnings=("STATE_INDICATOR_UNRESOLVED",),
                )
            )
            continue
        record["bound_indicator_candidate_key"] = matches[0]["candidate_key"]
        accepted.append(record)

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

    for index, relationship in enumerate(graph.relationships):
        holds.append(
            _hold("entity", index, "entity_phase_relationship_not_allowed", _relationship_summary(relationship))
        )
    return accepted, holds


def _strip_chunk_prefix(value: str, chunk_id: str) -> str | None:
    prefix = f"{chunk_id}:"
    return value[len(prefix):] if value.startswith(prefix) else None


def _has_allowed_endpoints(
    schema: Mapping[str, Any], relation_type: str, source: Mapping[str, Any], target: Mapping[str, Any]
) -> bool:
    return (source["entity_type"], target["entity_type"]) in _relation_endpoint_pairs(schema, relation_type)


def deterministic_state_relations(nodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create RC-02 only after an IndicatorState has a locally verified binding."""
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
    """Bind a rule edge only to an already replayed, role-labelled rule span."""
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Validate one relation stage against the immutable local node catalog."""
    node_by_key = {item["candidate_key"]: item for item in nodes}
    relations = deterministic_state_relations(nodes) if include_deterministic_state else []
    holds: list[dict[str, Any]] = []
    seen_keys = {item["candidate_key"] for item in relations}
    model_relations: list[tuple[int, dict[str, Any]]] = []

    for index, node in enumerate(graph.nodes):
        holds.append(_hold("relation", index, "relation_phase_node_not_allowed", _node_summary(node)))
    for index, relationship in enumerate(graph.relationships):
        summary = _relationship_summary(relationship)
        try:
            relation_type = relationship.type
            if relation_type not in allowed_relation_types:
                raise GraphBuilderConfigurationError("relation_type_not_enabled_for_trial")
            source_key = _strip_chunk_prefix(relationship.start_node_id, chunk.chunk_id)
            target_key = _strip_chunk_prefix(relationship.end_node_id, chunk.chunk_id)
            if not source_key or not target_key:
                missing = [
                    value for value in (relationship.start_node_id, relationship.end_node_id)
                    if _strip_chunk_prefix(value, chunk.chunk_id) is None
                ]
                summary["missing_endpoint_candidate_keys"] = missing
                raise GraphBuilderConfigurationError("relation_endpoint_not_from_frozen_catalog")
            source = node_by_key.get(source_key)
            target = node_by_key.get(target_key)
            if source is None or target is None:
                summary["missing_endpoint_candidate_keys"] = [
                    key for key in (source_key, target_key) if key not in node_by_key
                ]
                for node in (source, target):
                    if node is not None and node.get("entity_type") == "RuleDefinition":
                        _mark_partial(node, "RULE_ENDPOINT_UNRESOLVED")
                raise GraphBuilderConfigurationError("relation_endpoint_not_from_frozen_catalog")
            if source_key == target_key or not _has_allowed_endpoints(schema, relation_type, source, target):
                for node in (source, target):
                    if node.get("entity_type") == "RuleDefinition":
                        _mark_partial(node, "RULE_ENDPOINT_TYPE_INVALID")
                raise GraphBuilderConfigurationError("relation_endpoint_type_invalid")
            properties = relationship.properties
            rule_source_ref = _rule_relation_source_ref(relationship, relation_type, source, target)
            source_ref = (
                rule_source_ref
                if rule_source_ref is not None
                else _source_ref(
                    chunk,
                    source["mention"],
                    properties.get("exact_quote"),
                    exact_quote_occurrence_index=properties.get("exact_quote_occurrence_index"),
                    source_char_start=properties.get("source_char_start"),
                    source_char_end=properties.get("source_char_end"),
                )
            )
            if rule_source_ref is None and target["mention"] not in source_ref["exact_quote"]:
                raise GraphBuilderConfigurationError("relation_quote_lacks_endpoint")
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
                    raise GraphBuilderConfigurationError("relation_direction_not_verbatim")
                between_source_and_cue = source_ref["exact_quote"][
                    source_start + len(source["mention"]):cue_start
                ]
                quote_states = {
                    item["candidate_key"]
                    for item in node_by_key.values()
                    if item["entity_type"] == "IndicatorState"
                    and item["mention"] in source_ref["exact_quote"]
                }
                if len(quote_states) >= 2:
                    raise GraphBuilderConfigurationError("relation_may_be_joint_condition")
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
            model_relations.append((index, record))
        except GraphBuilderConfigurationError as error:
            holds.append(_hold("relation", index, str(error), summary))
    invalid_rule_keys: set[str] = set()
    if validate_rule_structures:
        valid_rule_edges, composite_holds, invalid_rule_keys = _validate_composite_structures(
            model_relations, node_by_key=node_by_key
        )
        relations.extend(valid_rule_edges)
        holds.extend(composite_holds)
    else:
        relations.extend(record for _index, record in model_relations)
    if return_invalid_rule_keys:
        return relations, holds, invalid_rule_keys
    return relations, holds


def _validate_composite_structures(
    relation_items: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    node_by_key: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Classify incomplete rule structures for review without discarding candidates."""
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {
        key: []
        for key, node in node_by_key.items()
        if node.get("entity_type") == "RuleDefinition"
    }
    for index, record in relation_items:
        relation_type = record["relation_type"]
        rule_key = (
            record["target_candidate_key"]
            if relation_type == "RULE_INPUT"
            else record["source_candidate_key"]
        )
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


def _catalog_for_prompt(nodes: Sequence[Mapping[str, Any]]) -> str:
    catalog = []
    for item in nodes:
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
