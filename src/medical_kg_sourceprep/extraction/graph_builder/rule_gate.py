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


def partition_invalid_rules(
    rules: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """拒绝自指规则和把显式举例误作结论的规则，其余候选保持原样。"""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for rule in rules:
        value = dict(rule)
        raw_inputs = value.get("rule_inputs", [])
        raw_outputs = value.get("rule_outputs", [])
        inputs = {
            str(item).strip() for item in raw_inputs
            if isinstance(item, str) and item.strip()
        } if isinstance(raw_inputs, list) else set()
        outputs = {
            str(item).strip() for item in raw_outputs
            if isinstance(item, str) and item.strip()
        } if isinstance(raw_outputs, list) else set()
        overlap = sorted(inputs & outputs)
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
        accepted.append(value)
    return accepted, rejected
