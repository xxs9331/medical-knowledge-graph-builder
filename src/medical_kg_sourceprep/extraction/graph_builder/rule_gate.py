"""规则候选的确定性结构门控。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any


def _is_explicit_example(output: str, evidence_text: str) -> bool:
    """判断输出 mention 是否在证据中被显式的举例标记引出。"""
    marker = r"(?:如|例如)\s*[：:,，、（(]?\s*"
    english_marker = r"(?:e\.g\.)\s*[：:,，、（(]?\s*"
    return re.search(f"(?:{marker}|{english_marker}){re.escape(output)}", evidence_text) is not None


def _inputs_are_explicit_examples(inputs: set[str], evidence_text: str) -> bool:
    """拒绝把 ``如/例如`` 后列举的多个替代项合并为 ALL 输入。"""
    markers = list(re.finditer(r"(?:如|例如|e\.g\.)\s*[：:,，、（(]?\s*", evidence_text))
    if not markers or len(inputs) < 2:
        return False
    marker_end = markers[-1].end()
    positions = [evidence_text.find(item) for item in inputs]
    return all(position >= marker_end for position in positions)


def _inputs_are_alternatives(inputs: set[str], evidence_text: str) -> bool:
    """识别输入跨度之间明确的析取词，或整列条件均可独立触发的表述。"""
    if len(inputs) < 2:
        return False
    positions = sorted(
        (evidence_text.find(item), evidence_text.find(item) + len(item))
        for item in inputs
        if evidence_text.find(item) >= 0
    )
    if len(positions) == len(inputs):
        span = evidence_text[positions[0][0]:positions[-1][1]]
        if re.search(r"(?:或|或者|任一|任意一种)", span):
            return True
    return re.search(r"等\s*(?:均|都)?\s*可", evidence_text) is not None


def _is_threshold_rule(inputs: set[str], evidence_text: str) -> bool:
    """阈值、参考区间和比较判断属于 PREPROCESS，而非图语义规则。"""
    text = "\n".join([*sorted(inputs), evidence_text])
    return re.search(r"(?:<=|>=|<|>|≤|≥|低于|高于|不超过|不少于)\s*\d", text) is not None


def _merges_separate_evidence_items(rule: Mapping[str, Any]) -> bool:
    refs = rule.get("rule_evidence_refs")
    if not isinstance(refs, list) or len(refs) < 2:
        return False
    roles = {
        str(item.get("role", "")).lower()
        for item in refs
        if isinstance(item, Mapping)
    }
    return not roles <= {"table_header", "table_row"}


def _has_supported_graph_shape(
    rule: Mapping[str, Any], inputs: set[str], outputs: set[str],
    excluded_outputs: set[str], evidence_text: str,
) -> bool:
    """只接纳当前图规则合同中可确定识别的四种来源形状。"""
    refs = rule.get("rule_evidence_refs")
    roles = {
        str(item.get("role", "")).lower()
        for item in refs
        if isinstance(item, Mapping)
    } if isinstance(refs, list) else set()
    if {"table_header", "table_row"} <= roles:
        return True
    if excluded_outputs:
        return re.search(r"(?:排除|不能|不可能|无|阴性|不支持|不科学)", evidence_text) is not None
    logic = rule.get("rule_logic_candidate")
    if logic == "ALL_SAME_WINDOW" and re.search(r"(?:同时|同一时间|同一窗口|同步)", evidence_text):
        return True
    colon = re.search(r"[：:]", evidence_text)
    if colon is None:
        return False
    prefix = re.sub(r"\s+", "", evidence_text[:colon.start()])
    suffix = re.sub(r"\s+", "", evidence_text[colon.end():])

    def endpoint_is_supported(endpoint: str) -> bool:
        normalized = re.sub(r"\s+", "", endpoint)
        if normalized in suffix:
            return True
        state_match = re.fullmatch(
            r"(.+?)(持续下降|持续升高|正常|增大|减小|升高|降低|阳性|阴性)",
            normalized,
        )
        return (
            state_match is not None
            and state_match.group(1) in suffix
            and state_match.group(2) in suffix
        )

    return (
        any(re.sub(r"\s+", "", output) in prefix for output in outputs)
        and all(endpoint_is_supported(input_) for input_ in inputs)
    )


def partition_invalid_rules(
    rules: Sequence[Mapping[str, Any]],
    *,
    strict_graph_shapes: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """拒绝可由来源结构确定的非规则候选，其余语义判断留给 Judge。"""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for rule in rules:
        value = dict(rule)
        raw_inputs = value.get("rule_inputs", [])
        raw_outputs = value.get("rule_outputs", [])
        raw_excluded_outputs = value.get("rule_excluded_outputs", [])
        inputs = {
            str(item).strip() for item in raw_inputs
            if isinstance(item, str) and item.strip()
        } if isinstance(raw_inputs, list) else set()
        outputs = {
            str(item).strip() for item in raw_outputs
            if isinstance(item, str) and item.strip()
        } if isinstance(raw_outputs, list) else set()
        excluded_outputs = {
            str(item).strip() for item in raw_excluded_outputs
            if isinstance(item, str) and item.strip()
        } if isinstance(raw_excluded_outputs, list) else set()
        overlap = sorted(inputs & (outputs | excluded_outputs))
        if overlap:
            rejected.append({
                "candidate_key": value.get("candidate_key"),
                "reason_code": "rule_input_output_overlap",
                "overlap_mentions": overlap,
            })
            continue
        evidence_text = "\n".join(
            str(item.get("exact_quote", ""))
            for item in value.get("rule_evidence_refs", [])
            if isinstance(item, Mapping)
        ) if isinstance(value.get("rule_evidence_refs"), list) else ""
        example_outputs = sorted(
            output for output in outputs if _is_explicit_example(output, evidence_text)
        )
        if example_outputs:
            rejected.append({
                "candidate_key": value.get("candidate_key"),
                "reason_code": "rule_output_is_explicit_example",
                "example_outputs": example_outputs,
            })
            continue
        if _is_threshold_rule(inputs, evidence_text):
            rejected.append({
                "candidate_key": value.get("candidate_key"),
                "reason_code": "rule_threshold_belongs_to_preprocess",
            })
            continue
        if len(inputs) == 1 and not excluded_outputs:
            rejected.append({
                "candidate_key": value.get("candidate_key"),
                "reason_code": "rule_single_input_not_explicit_exclusion",
            })
            continue
        if _inputs_are_explicit_examples(inputs, evidence_text):
            rejected.append({
                "candidate_key": value.get("candidate_key"),
                "reason_code": "rule_inputs_are_explicit_examples",
                "example_inputs": sorted(inputs),
            })
            continue
        if _inputs_are_alternatives(inputs, evidence_text):
            rejected.append({
                "candidate_key": value.get("candidate_key"),
                "reason_code": "rule_inputs_are_alternatives",
                "alternative_inputs": sorted(inputs),
            })
            continue
        if _merges_separate_evidence_items(value):
            rejected.append({
                "candidate_key": value.get("candidate_key"),
                "reason_code": "rule_cross_evidence_merge",
            })
            continue
        if strict_graph_shapes and not _has_supported_graph_shape(
            value, inputs, outputs, excluded_outputs, evidence_text
        ):
            rejected.append({
                "candidate_key": value.get("candidate_key"),
                "reason_code": "rule_source_shape_not_graph_composite",
            })
            continue
        accepted.append(value)
    return accepted, rejected
