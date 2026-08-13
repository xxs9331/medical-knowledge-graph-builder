"""原文证据回放、规则证据解析和候选身份生成。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..contract import CANDIDATE_RUN_VERSION, RULE_EXPRESSION_PATTERN, GraphBuilderConfigurationError
from ...llm_extraction import EvidenceChunk


def _is_source_offset(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _occurrence_starts(text: str, needle: str, *, start: int = 0, end: int | None = None) -> list[int]:
    """返回文本中所有匹配位置，用于处理重复引语。"""
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
    """把模型给出的引语定位到唯一、连续且逐字相同的原文区间。"""
    has_start = source_char_start is not None
    has_end = source_char_end is not None
    if has_start != has_end:
        raise GraphBuilderConfigurationError("quote_position_incomplete")
    if has_start and (
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
    """创建普通实体或关系使用的来源引用，并定位 mention 在引语中的位置。"""
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


def _source_refs_for_mention(chunk: EvidenceChunk, mention: str) -> list[dict[str, Any]]:
    """由代码为逐字出现的实体生成全部可回放来源。

    轻量实体发现阶段只让模型做类型和名称判断。此函数以每次 ``mention`` 出现所在
    的非空文本行作为最小可回放原文单元，并由已有 ``_source_ref`` 复验其字符位置。
    它不解释表格箭头等非连续语义；这类名称无法逐字找到时必须交给 Judge。
    """
    if not isinstance(mention, str) or not mention:
        raise GraphBuilderConfigurationError("mention_missing")
    starts = _occurrence_starts(chunk.text, mention)
    if not starts:
        raise GraphBuilderConfigurationError("semantic_mention_not_in_source")
    refs: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int, int]] = set()
    for mention_start in starts:
        line_start = chunk.text.rfind("\n", 0, mention_start) + 1
        line_end = chunk.text.find("\n", mention_start)
        if line_end < 0:
            line_end = len(chunk.text)
        quote = chunk.text[line_start:line_end]
        if not quote:
            raise GraphBuilderConfigurationError("semantic_mention_source_unit_missing")
        mention_starts = _occurrence_starts(chunk.text, mention, start=line_start, end=line_end)
        occurrence_index = mention_starts.index(mention_start)
        ref = _source_ref(
            chunk,
            mention,
            quote,
            source_char_start=line_start,
            source_char_end=line_end,
            mention_occurrence_index=occurrence_index,
        )
        identity = (ref["char_start"], ref["char_end"], ref["mention_char_start"])
        if identity not in seen_spans:
            seen_spans.add(identity)
            refs.append(ref)
    return refs


def _candidate_key(entity_type: str, mention: str, source_ref: Mapping[str, Any]) -> str:
    """以类型、chunk 和 mention 原文位置生成稳定候选键。"""
    raw_mention_start = source_ref.get("mention_char_start")
    if isinstance(raw_mention_start, int) and not isinstance(raw_mention_start, bool):
        mention_start = raw_mention_start
    else:
        mention_start = int(source_ref["char_start"]) + str(source_ref["exact_quote"]).find(mention)
    raw = f"{CANDIDATE_RUN_VERSION}:{entity_type}:{source_ref['chunk_id']}:{mention_start}:{mention_start + len(mention)}"
    return f"candidate:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def _table_state_candidate_key(
    *, chunk: EvidenceChunk, mention: str, evidence_refs: Sequence[Mapping[str, Any]]
) -> str:
    """为模型从表格语义得到的状态，以表头/行原文位置生成稳定键。"""
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
    """回放模型提交的表头和表格行；不解释箭头或单元格医学语义。"""
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
    """回放一条规则证据；role 仅用于定位，不在这里裁决语义。"""
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
    """规范规则表达式的空白，并只验证固定的 r=A(a,b) 外形。"""
    if not isinstance(value, str) or not value.strip():
        raise GraphBuilderConfigurationError("rule_expression_missing")
    expression = re.sub(r"\s+", "", value)
    matched = RULE_EXPRESSION_PATTERN.fullmatch(expression)
    if matched is None or not matched.group("inputs"):
        raise GraphBuilderConfigurationError("rule_expression_invalid")
    return expression, matched.group("output"), matched.group("name")


def _rule_expression_endpoints(expression: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """从已通过形状检查的表达式中拆出输出端点和输入端点。"""
    matched = RULE_EXPRESSION_PATTERN.fullmatch(expression)
    if matched is None:
        raise GraphBuilderConfigurationError("rule_expression_invalid")
    output = matched.group("output")
    outputs = tuple(item for item in output[1:-1].split(",") if item) if output.startswith("[") and output.endswith("]") else (output,)
    inputs = tuple(item for item in matched.group("inputs").split(",") if item)
    if not outputs or not inputs:
        raise GraphBuilderConfigurationError("rule_expression_invalid")
    return outputs, inputs


def _parse_rule_evidence(chunk: EvidenceChunk, value: Any) -> list[dict[str, Any]]:
    """将模型返回的规则证据 JSON 转为本地可回放引用。"""
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
        refs.append(_rule_evidence_ref(
            chunk,
            role=item.get("role"),
            exact_quote=item.get("exact_quote"),
            exact_quote_occurrence_index=item.get("exact_quote_occurrence_index"),
            source_char_start=item.get("source_char_start"),
            source_char_end=item.get("source_char_end"),
        ))
    return refs


def _rule_candidate_key(
    *, chunk: EvidenceChunk, rule_stage: str, rule_expression: str, rule_evidence_refs: Sequence[Mapping[str, Any]]
) -> str:
    """以规则阶段、表达式和证据位置生成规则候选键。"""
    raw = json.dumps(
        {
            "version": CANDIDATE_RUN_VERSION,
            "chunk_id": chunk.chunk_id,
            "rule_stage": rule_stage,
            "rule_expression": rule_expression,
            "rule_evidence_positions": [
                {"role": item["role"], "char_start": item["char_start"], "char_end": item["char_end"]}
                for item in rule_evidence_refs
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"candidate:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def _relation_key(relation_type: str, source_key: str, target_key: str, source_ref: Mapping[str, Any]) -> str:
    """以关系端点和独立证据位置生成稳定关系键。"""
    raw = f"{CANDIDATE_RUN_VERSION}:{relation_type}:{source_key}:{target_key}:{source_ref['chunk_id']}:{source_ref['char_start']}:{source_ref['char_end']}"
    return f"relation:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"
