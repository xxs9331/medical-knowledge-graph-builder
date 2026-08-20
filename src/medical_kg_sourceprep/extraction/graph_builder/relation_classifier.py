"""基于冻结实体目录的关系分类实验。

保留早期的二阶段分类器用于对照，并提供结构感知的单阶段闭集分类器。后者先根据
节点已经回放的证据锚点和文档结构生成候选实体对，再让模型一次选择“无关系”或
Schema 允许的关系类型与方向。模型不能创建实体，最终候选仍交给既有关系硬校验器。
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jRelationship
from neo4j_graphrag.exceptions import LLMGenerationError

from ..llm_extraction import EvidenceChunk
from .contract import GraphBuilderConfigurationError
from .joint_extraction import EvidenceUnit, _minimal_relation_quote, build_evidence_units
from .schema import _relation_endpoint_pairs
from .trace import NULL_TRACE, TraceRecorder


RELATION_SEMANTICS = {
    "HAS_STATE": "检验指标具有原文明确表达的正常、升高、降低或异常状态。",
    "CAUSES": "源实体是目标实体在原文中直接表达的原因、机制或导致因素。",
    "INDICATES": "源指标、状态或临床语境在原文中直接提示目标疾病或临床语境。",
    "ASSOCIATED_WITH": (
        "原文使用‘相关、有关、关联、联系’等明确关系谓词连接两个实体，且没有更具体的"
        "因果、提示、包含、状态或层级关系；必须返回包含该谓词的逐字 trigger_quote。"
        "全称与缩写、同一公式中的变量、共同出现、并列、同表出现、共享父项和检测使用"
        "某材料均不属于该关系。"
    ),
    "IS_A": "源实体是目标实体的下位类型或实例，原文直接表达这种分类关系。",
    "HAS_METRIC": "源检验组合明确包含目标子组合或指标，或源复合指标明确包含目标子指标。",
}


_ASSOCIATION_TRIGGER_PATTERN = re.compile(r"相关|有关|关联|联系")
_CAUSAL_TRIGGER_PATTERN = re.compile(r"引起|导致|所致|由于|造成|使")
_INDICATION_TRIGGER_PATTERN = re.compile(r"提示|表明|支持|考虑|可见于|诊断意义")
_METRIC_TRIGGER_PATTERN = re.compile(r"包括|包含|以|表示|分为|由[^。；\n<]{0,30}组成")
_HIERARCHY_TRIGGER_PATTERN = re.compile(r"属于|分为|分型|分类为|类型")
_CLAUSE_BOUNDARY_PATTERN = re.compile(r"[。！？；\n]|</td>", re.IGNORECASE)
_STATE_INDICATOR_SUFFIX_PATTERN = re.compile(
    r"(?:正常|异常|升高|降低|增高|减低|增大|减小|减少|增多|阳性|阴性|延长|缩短).*$"
)

# 每条路由只暴露一种正类和 NO_RELATION，避免模型把相近标签互相替代。
# 泛化关联必须经过独立的显式触发词通道，不进入这里的常规关系分类。
RELATION_ROUTE_DEFINITIONS: tuple[tuple[str, frozenset[str]], ...] = (
    ("has_state", frozenset({"HAS_STATE"})),
    ("has_metric", frozenset({"HAS_METRIC"})),
    ("is_a", frozenset({"IS_A"})),
    ("causes", frozenset({"CAUSES"})),
    ("indicates", frozenset({"INDICATES"})),
)

RELATION_DECISION_RULES = {
    "HAS_STATE": (
        "只判断检验指标是否直接具有目标状态；状态必须在语义上修饰该指标。"
        "疾病、分类或同表项目不能因为相邻而具有该状态。"
    ),
    "HAS_METRIC": (
        "只判断检验组合或复合指标是否明确包含目标指标；同表出现、共同用于诊断或共享标题不等于包含。"
    ),
    "IS_A": (
        "只判断下位概念到上位类别的分类关系。表格分类行中的示例疾病可以属于该行类别；"
        "并列疾病、病因与结果、缩写与全称都不是上下位关系。"
    ),
    "CAUSES": (
        "只接受原文直接表达‘引起、导致、由于、所致、造成’等因果语义，方向必须从原因到结果；"
        "提示、诊断意义、同表列举或类别示例不是因果。"
    ),
    "INDICATES": (
        "只接受观察结果、指标或状态直接‘提示、表明、支持、考虑’某发现，或原文明示具有诊断意义；"
        "‘引起、导致、由于、所致、造成’属于因果而不是提示，类别与示例、并列疾病也不是提示。"
    ),
}


def _has_explicit_association_trigger(evidence: str, trigger_quote: object) -> bool:
    """只接受证据中逐字出现且带明确关联谓词的触发片段。"""
    return (
        isinstance(trigger_quote, str)
        and bool(trigger_quote.strip())
        and trigger_quote in evidence
        and _ASSOCIATION_TRIGGER_PATTERN.search(trigger_quote) is not None
    )


def _has_verbatim_trigger(evidence: str, trigger_quote: object) -> bool:
    """所有正例都必须给出自己的非空逐字证据片段。"""
    return (
        isinstance(trigger_quote, str)
        and bool(trigger_quote.strip())
        and trigger_quote in evidence
    )


def _normalized_endpoint_text(value: str) -> str:
    """用于比较缩写和状态前缀，保留中文与字母数字，忽略排版符号。"""
    return "".join(character for character in value.casefold() if character.isalnum())


def _is_state_for_indicator(
    indicator: Mapping[str, Any], state: Mapping[str, Any]
) -> bool:
    """状态名必须以指标名或其原文别名开头，禁止 MCV 指向 RDW 正常等错配。"""
    state_name = state.get("canonical_name", state.get("mention"))
    if not isinstance(state_name, str) or not state_name:
        return False
    state_base = _STATE_INDICATOR_SUFFIX_PATTERN.sub("", state_name)
    normalized_state_base = _normalized_endpoint_text(state_base)
    if not normalized_state_base:
        return False
    forms = [indicator.get("canonical_name"), indicator.get("mention")]
    aliases = indicator.get("aliases")
    if isinstance(aliases, list):
        forms.extend(aliases)
    return any(
        isinstance(form, str)
        and bool(form)
        and normalized_state_base == _normalized_endpoint_text(form)
        for form in forms
    )


def _cue_clause_contains_both_endpoints(
    evidence: str,
    *,
    left_mention: str,
    right_mention: str,
    trigger_pattern: re.Pattern[str],
) -> bool:
    """两个端点和关系触发词必须位于同一最小分句或同一表格单元格。"""
    for trigger in trigger_pattern.finditer(evidence):
        start = max(
            (boundary.end() for boundary in _CLAUSE_BOUNDARY_PATTERN.finditer(evidence, 0, trigger.start())),
            default=0,
        )
        following = _CLAUSE_BOUNDARY_PATTERN.search(evidence, trigger.end())
        end = following.start() if following is not None else len(evidence)
        clause = evidence[start:end]
        if left_mention in clause and right_mention in clause:
            return True
    return False


def _cue_clause_has_endpoints_on_opposite_sides(
    evidence: str,
    *,
    left_mention: str,
    right_mention: str,
    trigger_pattern: re.Pattern[str],
) -> bool:
    """因果和提示的端点必须分居谓词两侧，不能把同侧并列原因或结论互连。"""
    for trigger in trigger_pattern.finditer(evidence):
        start = max(
            (boundary.end() for boundary in _CLAUSE_BOUNDARY_PATTERN.finditer(evidence, 0, trigger.start())),
            default=0,
        )
        following = _CLAUSE_BOUNDARY_PATTERN.search(evidence, trigger.end())
        end = following.start() if following is not None else len(evidence)
        clause = evidence[start:end]
        trigger_start = trigger.start() - start
        trigger_end = trigger.end() - start
        left_before = any(match.end() <= trigger_start for match in re.finditer(re.escape(left_mention), clause))
        left_after = any(match.start() >= trigger_end for match in re.finditer(re.escape(left_mention), clause))
        right_before = any(match.end() <= trigger_start for match in re.finditer(re.escape(right_mention), clause))
        right_after = any(match.start() >= trigger_end for match in re.finditer(re.escape(right_mention), clause))
        if (left_before and right_after) or (right_before and left_after):
            return True
    return False


def _direct_relation_option_is_eligible(
    *,
    relation_type: str,
    direction: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    evidence: str,
) -> bool:
    """在模型前排除表格共现和端点错配，只保留具有对应结构信号的候选。"""
    left_mention = left.get("mention")
    right_mention = right.get("mention")
    if not isinstance(left_mention, str) or not isinstance(right_mention, str):
        return False
    if relation_type == "HAS_STATE":
        indicator, state = (left, right) if direction == "LEFT_TO_RIGHT" else (right, left)
        return _is_state_for_indicator(indicator, state)
    trigger_pattern = {
        "CAUSES": _CAUSAL_TRIGGER_PATTERN,
        "INDICATES": _INDICATION_TRIGGER_PATTERN,
        "HAS_METRIC": _METRIC_TRIGGER_PATTERN,
        "IS_A": _HIERARCHY_TRIGGER_PATTERN,
    }.get(relation_type)
    if trigger_pattern is None:
        return False
    predicate = (
        _cue_clause_has_endpoints_on_opposite_sides
        if relation_type in {"CAUSES", "INDICATES"}
        else _cue_clause_contains_both_endpoints
    )
    return predicate(
        evidence,
        left_mention=left_mention,
        right_mention=right_mention,
        trigger_pattern=trigger_pattern,
    )


def _lexical_is_a_options(
    *,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    endpoint_pairs: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    """从同一原文中的复合疾病名恢复严格的词面下位关系候选。"""
    options: list[tuple[str, str]] = []
    for source, target, direction in (
        (left, right, "LEFT_TO_RIGHT"),
        (right, left, "RIGHT_TO_LEFT"),
    ):
        source_type = source.get("entity_type")
        target_type = target.get("entity_type")
        source_name = source.get("canonical_name")
        target_name = target.get("canonical_name")
        if (
            not isinstance(source_type, str)
            or not isinstance(target_type, str)
            or (source_type, target_type) not in endpoint_pairs
            or source_type != "Disease"
            or target_type != "Disease"
            or not isinstance(source_name, str)
            or not isinstance(target_name, str)
            or len(target_name) < 2
            or not source_name.endswith(target_name)
            or len(source_name) - len(target_name) < 2
        ):
            continue
        options.append(("IS_A", direction))
    return options


def _smallest_unit_for_spans(
    units: Sequence[EvidenceUnit], spans: Sequence[tuple[int, int]]
) -> EvidenceUnit | None:
    """词面下位关系仅使用子类名称所在的最小可回放结构单元。"""
    candidates = [
        unit
        for unit in units
        if unit.kind in {"line", "table_row"}
        and any(_span_inside_unit(span, unit) for span in spans)
    ]
    return min(candidates, key=lambda unit: (unit.end - unit.start, unit.start, unit.kind)) if candidates else None


@dataclass(frozen=True, slots=True)
class RelationPair:
    """一个可分类实体对及其可回放的最小原文窗口。"""

    pair_id: str
    left_key: str
    left_type: str
    left_mention: str
    left_canonical_name: str
    right_key: str
    right_type: str
    right_mention: str
    right_canonical_name: str
    evidence_start: int
    evidence_end: int
    evidence_text: str
    options: tuple[tuple[str, str], ...]


def _mention_positions(text: str, mention: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(re.escape(mention), text)]


def _node_reference_spans(
    node: Mapping[str, Any], *, chunk_id: str, text: str
) -> list[tuple[int, int]]:
    """读取节点已通过硬校验的证据位置，缺失时才回退到 mention 搜索。"""
    references: list[Mapping[str, Any]] = []
    for key in ("source_refs", "table_state_evidence_refs", "derived_entity_evidence_refs"):
        value = node.get(key)
        if isinstance(value, list):
            references.extend(item for item in value if isinstance(item, Mapping))
    source_ref = node.get("source_ref")
    if isinstance(source_ref, Mapping):
        references.append(source_ref)

    spans: list[tuple[int, int]] = []
    for reference in references:
        if reference.get("chunk_id") not in {None, chunk_id}:
            continue
        start = reference.get("mention_char_start", reference.get("char_start"))
        end = reference.get("mention_char_end", reference.get("char_end"))
        if (
            isinstance(start, int) and not isinstance(start, bool)
            and isinstance(end, int) and not isinstance(end, bool)
            and 0 <= start < end <= len(text)
        ):
            spans.append((start, end))
    if spans:
        return list(dict.fromkeys(spans))
    mention = node.get("mention")
    return _mention_positions(text, mention) if isinstance(mention, str) and mention else []


def _structure_units(text: str) -> list[EvidenceUnit]:
    """补充整表上下文，使表头和数据行上的两个证据锚点能进入同一候选单元。"""
    units = list(build_evidence_units(text))
    table_index = 1
    for match in re.finditer(r"<table>.*?</table>", text, re.DOTALL):
        units.append(EvidenceUnit(
            unit_id=f"relation-table-{table_index:04d}",
            kind="table_context",
            start=match.start(),
            end=match.end(),
            text=match.group(),
        ))
        table_index += 1
    # 原始 EvidenceUnit 只在段落空行满足特定形式时生成 list_context。关系候选
    # 还需兼容紧邻排版，因此按行识别“(n) 标题 + 1) 子项”连续结构。
    lines = list(re.finditer(r"[^\n]+", text))
    list_index = 1
    for index, heading in enumerate(lines):
        if re.fullmatch(r"\s*\(\d+\)\s*.+", heading.group()) is None:
            continue
        children: list[re.Match[str]] = []
        for line in lines[index + 1 :]:
            if re.match(r"\s*\d+\)", line.group()):
                children.append(line)
                continue
            if children:
                break
        if not children:
            continue
        start, end = heading.start(), children[-1].end()
        if not any(
            unit.kind == "list_context" and unit.start == start and unit.end == end
            for unit in units
        ):
            units.append(EvidenceUnit(
                unit_id=f"relation-list-{list_index:04d}",
                kind="list_context",
                start=start,
                end=end,
                text=text[start:end],
            ))
            list_index += 1
    return units


def _span_inside_unit(span: tuple[int, int], unit: EvidenceUnit) -> bool:
    """判断证据锚点是否完整位于一个结构单元。"""
    start, end = span
    return unit.start <= start < end <= unit.end


def _direct_list_windows(
    unit: EvidenceUnit,
) -> list[tuple[EvidenceUnit, tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """把列表上下文拆成“标题 + 一个直接子项标题”的局部窗口。

    子项冒号后的举例和原因仍可在同一行内互相形成候选，但不会直接连接列表总标题，
    从候选层面避免跳过原文层级。
    """
    lines = list(re.finditer(r"[^\n]+", unit.text))
    if len(lines) < 2:
        return []
    heading = lines[0]
    heading_span = (unit.start + heading.start(), unit.start + heading.end())
    windows: list[
        tuple[EvidenceUnit, tuple[int, int], tuple[int, int], tuple[int, int]]
    ] = []
    for index, line in enumerate(lines[1:], start=1):
        child = re.match(r"\s*(?:\d+\)|[①-⑳])\s*([^:：\n]+)[:：]", line.group())
        if child is None:
            continue
        child_span = (
            unit.start + line.start() + child.start(1),
            unit.start + line.start() + child.end(1),
        )
        line_span = (unit.start + line.start(), unit.start + line.end())
        window_end = unit.start + line.end()
        windows.append((
            EvidenceUnit(
                unit_id=f"{unit.unit_id}-direct-{index:02d}",
                kind="list_direct_child",
                start=heading_span[0],
                end=window_end,
                text=unit.text[heading.start() : line.end()],
            ),
            heading_span,
            child_span,
            line_span,
        ))
    return windows


def _smallest_shared_structure(
    units: Sequence[EvidenceUnit],
    left_spans: Sequence[tuple[int, int]],
    right_spans: Sequence[tuple[int, int]],
    *,
    text: str,
    left_mention: str,
    right_mention: str,
) -> EvidenceUnit | None:
    """选择同一原子单元或列表直接父子窗口，不使用整段落两两配对。"""
    candidates = [
        unit
        for unit in units
        if unit.kind in {"line", "table_row"}
        and any(_span_inside_unit(span, unit) for span in left_spans)
        and any(_span_inside_unit(span, unit) for span in right_spans)
    ]
    if candidates:
        return min(candidates, key=lambda unit: (unit.end - unit.start, unit.start, unit.kind))

    # 列表跨行候选只允许总标题连接每一行冒号前的直接子项标题。这里重新查找
    # mention 的实际位置，避免旧工件中整行 source_ref 把孙级举例误当成直接子项。
    left_mentions = _mention_positions(text, left_mention)
    right_mentions = _mention_positions(text, right_mention)
    for unit in units:
        if unit.kind != "list_context":
            continue
        for window, heading_span, child_span, line_span in _direct_list_windows(unit):
            left_heading = any(
                heading_span[0] <= start < end <= heading_span[1]
                for start, end in left_mentions
            )
            right_heading = any(
                heading_span[0] <= start < end <= heading_span[1]
                for start, end in right_mentions
            )
            left_child = any(
                child_span[0] <= start < end <= child_span[1]
                for start, end in left_mentions
            )
            right_child = any(
                child_span[0] <= start < end <= child_span[1]
                for start, end in right_mentions
            )
            if (left_heading and right_child) or (right_heading and left_child):
                return window
            # 冒号前的词既可能是真实语义父项，也可能只是“其他”一类排版分组。
            # 候选层无法可靠裁决其语义，因此保留总标题到当前直接子项行内实体的
            # 候选；是否跨越真实中间节点由闭集分类提示词判断。
            left_in_line = any(
                line_span[0] <= start < end <= line_span[1]
                for start, end in left_mentions
            )
            right_in_line = any(
                line_span[0] <= start < end <= line_span[1]
                for start, end in right_mentions
            )
            if (left_heading and right_in_line) or (right_heading and left_in_line):
                return window
    return None


def build_relation_pairs(
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    allowed_relation_types: Collection[str],
) -> list[RelationPair]:
    """按 Schema 端点约束枚举共享句子、列表或表格结构的无序实体对。"""
    endpoint_pairs = {
        relation_type: set(_relation_endpoint_pairs(schema, relation_type))
        for relation_type in allowed_relation_types
    }
    structure_units = _structure_units(chunk.text)
    spans_by_key = {
        str(node.get("candidate_key")): _node_reference_spans(
            node, chunk_id=chunk.chunk_id, text=chunk.text
        )
        for node in nodes
        if isinstance(node.get("candidate_key"), str)
    }
    pairs: list[RelationPair] = []
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            left_type, right_type = left.get("entity_type"), right.get("entity_type")
            left_mention, right_mention = left.get("mention"), right.get("mention")
            left_key, right_key = left.get("candidate_key"), right.get("candidate_key")
            left_canonical_name = left.get("canonical_name", left_mention)
            right_canonical_name = right.get("canonical_name", right_mention)
            if not isinstance(left_type, str) or not left_type:
                continue
            if not isinstance(right_type, str) or not right_type:
                continue
            if not isinstance(left_mention, str) or not left_mention:
                continue
            if not isinstance(right_mention, str) or not right_mention:
                continue
            if not isinstance(left_key, str) or not left_key:
                continue
            if not isinstance(right_key, str) or not right_key:
                continue
            if not isinstance(left_canonical_name, str) or not left_canonical_name:
                continue
            if not isinstance(right_canonical_name, str) or not right_canonical_name:
                continue
            # 同一规范实体可能在同一结构单元出现多个 mention。它们是同一个节点的
            # 多条证据，不应送给模型判断自关系。
            if left.get("canonical_id") is not None and (
                left.get("canonical_id") == right.get("canonical_id")
            ):
                continue
            left_spans = spans_by_key.get(left_key, ())
            right_spans = spans_by_key.get(right_key, ())
            if left_mention == right_mention and set(left_spans) & set(right_spans):
                continue
            # 指标-状态对只由 HAS_STATE 专用阶段处理。普通阶段若仅凭
            # ASSOCIATED_WITH 再处理一次，会把同一语义错误降级成泛化关联边。
            if "HAS_STATE" not in allowed_relation_types and {
                left_type, right_type
            } == {"LabIndicator", "IndicatorState"}:
                continue
            options: list[tuple[str, str]] = []
            for relation_type in sorted(allowed_relation_types):
                allowed = endpoint_pairs[relation_type]
                if (left_type, right_type) in allowed:
                    options.append((relation_type, "LEFT_TO_RIGHT"))
                if (right_type, left_type) in allowed:
                    options.append((relation_type, "RIGHT_TO_LEFT"))
            if not options:
                continue
            lexical_options = (
                _lexical_is_a_options(
                    left=left,
                    right=right,
                    endpoint_pairs=endpoint_pairs.get("IS_A", set()),
                )
                if "IS_A" in allowed_relation_types
                else []
            )
            unit = _smallest_shared_structure(
                structure_units,
                left_spans,
                right_spans,
                text=chunk.text,
                left_mention=left_mention,
                right_mention=right_mention,
            )
            if unit is None:
                if not lexical_options:
                    continue
                child_spans = (
                    left_spans
                    if any(direction == "LEFT_TO_RIGHT" for _type, direction in lexical_options)
                    else right_spans
                )
                unit = _smallest_unit_for_spans(structure_units, child_spans)
                if unit is None:
                    continue
            start, end, evidence_text = unit.start, unit.end, unit.text
            endpoint_options = list(lexical_options)
            endpoint_options.extend(
                (relation_type, direction)
                for relation_type, direction in options
                if (relation_type, direction) not in lexical_options
                if _direct_relation_option_is_eligible(
                    relation_type=relation_type,
                    direction=direction,
                    left=left,
                    right=right,
                    evidence=evidence_text,
                )
            )
            if not endpoint_options:
                continue
            pairs.append(RelationPair(
                pair_id=f"pair-{len(pairs):04d}",
                left_key=left_key, left_type=left_type, left_mention=left_mention,
                left_canonical_name=left_canonical_name,
                right_key=right_key, right_type=right_type, right_mention=right_mention,
                right_canonical_name=right_canonical_name,
                evidence_start=start, evidence_end=end, evidence_text=evidence_text,
                options=tuple(endpoint_options),
            ))
    return pairs


def _relationship_from_classification(
    *, chunk: EvidenceChunk, pair: RelationPair, relation_type: str, direction: str
) -> Neo4jRelationship | None:
    """把闭集标签转换为候选边；普通关系必须能生成包含双端点的逐字引语。"""
    if direction == "LEFT_TO_RIGHT":
        source_key, target_key = pair.left_key, pair.right_key
        source_mention, target_mention = pair.left_mention, pair.right_mention
    else:
        source_key, target_key = pair.right_key, pair.left_key
        source_mention, target_mention = pair.right_mention, pair.left_mention
    properties: dict[str, Any] = {}
    try:
        quote_start, quote_end, quote = _minimal_relation_quote(
            chunk.text,
            start=pair.evidence_start,
            end=pair.evidence_end,
            source_mention=source_mention,
            target_mention=target_mention,
        )
        properties = {
            "exact_quote": quote,
            "source_char_start": quote_start,
            "source_char_end": quote_end,
        }
    except GraphBuilderConfigurationError:
        # 派生状态的规范化 mention 可能不在原文连续出现；硬校验会复用其双锚点。
        if relation_type != "HAS_STATE":
            return None
    return Neo4jRelationship(
        start_node_id=f"{chunk.chunk_id}:{source_key}",
        end_node_id=f"{chunk.chunk_id}:{target_key}",
        type=relation_type,
        properties=properties,
    )


def _pair_payload(pair: RelationPair, *, include_options: bool) -> dict[str, Any]:
    value: dict[str, Any] = {
        "pair_id": pair.pair_id,
        "left": {
            "candidate_key": pair.left_key,
            "entity_type": pair.left_type,
            "mention": pair.left_mention,
            "canonical_name": pair.left_canonical_name,
        },
        "right": {
            "candidate_key": pair.right_key,
            "entity_type": pair.right_type,
            "mention": pair.right_mention,
            "canonical_name": pair.right_canonical_name,
        },
        "evidence": {"start": pair.evidence_start, "end": pair.evidence_end, "text": pair.evidence_text},
    }
    if include_options:
        value["allowed_options"] = [
            {"relation_type": relation_type, "direction": direction}
            for relation_type, direction in pair.options
        ]
    return value


async def _invoke_json(client: Any, prompt: str) -> Mapping[str, Any]:
    """调用模型并要求返回 JSON 对象；格式错误时重试一次。"""
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            response = await client.llm.ainvoke(prompt)
            payload = json.loads(response.content)
            if isinstance(payload, Mapping):
                return payload
            raise GraphBuilderConfigurationError("relation_classifier_response_not_object")
        except (AttributeError, TypeError, json.JSONDecodeError, GraphBuilderConfigurationError) as error:
            last_error = error
    raise GraphBuilderConfigurationError("relation_classifier_response_invalid") from last_error


def _validated_results(
    payload: Mapping[str, Any], pairs: Sequence[RelationPair]
) -> dict[str, Mapping[str, Any]]:
    pair_by_id = {pair.pair_id: pair for pair in pairs}
    pair_ids = set(pair_by_id)
    results = payload.get("results")
    if not isinstance(results, list):
        raise GraphBuilderConfigurationError("relation_classifier_results_missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for result in results:
        if not isinstance(result, Mapping) or result.get("pair_id") not in pair_ids:
            raise GraphBuilderConfigurationError("relation_classifier_pair_unknown")
        pair_id = str(result["pair_id"])
        if pair_id in by_id:
            raise GraphBuilderConfigurationError("relation_classifier_pair_duplicate")
        pair = pair_by_id[pair_id]
        if (
            ("left_mention" in result and result.get("left_mention") != pair.left_mention)
            or ("right_mention" in result and result.get("right_mention") != pair.right_mention)
        ):
            raise GraphBuilderConfigurationError("relation_classifier_pair_echo_mismatch")
        by_id[pair_id] = result
    if set(by_id) != pair_ids:
        raise GraphBuilderConfigurationError("relation_classifier_pair_incomplete")
    return by_id


async def _invoke_validated_results(
    client: Any, prompt: str, pairs: Sequence[RelationPair]
) -> dict[str, Mapping[str, Any]]:
    """结构或端点回显错位时重试整批，最多两次。"""
    last_error: GraphBuilderConfigurationError | None = None
    for _attempt in range(2):
        try:
            return _validated_results(await _invoke_json(client, prompt), pairs)
        except GraphBuilderConfigurationError as error:
            last_error = error
    raise GraphBuilderConfigurationError("relation_classifier_batch_invalid") from last_error


async def _invoke_validated_results_resilient(
    client: Any,
    prompt_builder: Any,
    pairs: Sequence[RelationPair],
) -> tuple[dict[str, Mapping[str, Any]], int]:
    """批量回显持续失败时二分重试，不放宽单项身份和完整性校验。"""
    try:
        return await _invoke_validated_results(client, prompt_builder(pairs), pairs), 1
    except (GraphBuilderConfigurationError, LLMGenerationError):
        if len(pairs) <= 1:
            raise
        middle = len(pairs) // 2
        left, left_calls = await _invoke_validated_results_resilient(
            client, prompt_builder, pairs[:middle]
        )
        right, right_calls = await _invoke_validated_results_resilient(
            client, prompt_builder, pairs[middle:]
        )
        return {**left, **right}, 1 + left_calls + right_calls


async def classify_relationships_two_stage(
    client: Any,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    allowed_relation_types: Collection[str],
    trace: TraceRecorder = NULL_TRACE,
    batch_size: int = 12,
) -> tuple[Neo4jGraph, dict[str, Any]]:
    """先判定关系存在性，再对正例选择关系类型和方向。"""
    pairs = build_relation_pairs(
        chunk=chunk, schema=schema, nodes=nodes,
        allowed_relation_types=allowed_relation_types,
    )
    related: list[RelationPair] = []
    stage1_records: list[dict[str, Any]] = []
    model_call_count = 0
    relation_definitions = {
        relation_type: RELATION_SEMANTICS.get(relation_type, relation_type)
        for relation_type in sorted(allowed_relation_types)
    }
    for offset in range(0, len(pairs), batch_size):
        batch = pairs[offset : offset + batch_size]
        def build_existence_prompt(prompt_pairs: Sequence[RelationPair]) -> str:
            return (
                "你是医学文本关系存在性分类器。只使用每项给出的 evidence，不使用外部知识。"
                "分别判断两个实体之间是否存在下列任一种图谱关系；本阶段不要输出具体关系类型。"
                "只有原文直接陈述且能够落入 allowed_options 的关系才选择 RELATED。"
                "列表中相邻项目、并列、举例、同表出现、全称与缩写、公式变量、参考范围、"
                "检测时使用某材料、共享同一上位项均选择 NO_RELATION。"
                "存在中间实体时不得跨过中间实体；联合条件不得拆成多个单条件直连边。"
                "证据确实表达了直接关系但无法确定是否属于 allowed_options 时才选择 UNCERTAIN。"
                "返回且只返回一个 JSON 对象：{\"results\":[{\"pair_id\":字符串,"
                "\"left_mention\":原样回显,\"right_mention\":原样回显,"
                "\"verdict\":\"RELATED\"或\"NO_RELATION\"或\"UNCERTAIN\",\"reason\":字符串}]}。"
                "必须为每个 pair_id 按输入顺序返回一项。\nRELATION_DEFINITIONS_JSON:\n"
                + json.dumps(relation_definitions, ensure_ascii=False, separators=(",", ":"))
                + "\nPAIR_INPUT_JSON:\n"
                + json.dumps(
                    [_pair_payload(pair, include_options=True) for pair in prompt_pairs],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        result_map, batch_calls = await _invoke_validated_results_resilient(
            client, build_existence_prompt, batch
        )
        model_call_count += batch_calls
        for pair in batch:
            result = result_map[pair.pair_id]
            verdict = result.get("verdict")
            if verdict not in {"RELATED", "NO_RELATION", "UNCERTAIN"}:
                raise GraphBuilderConfigurationError("relation_classifier_verdict_invalid")
            stage1_records.append({"pair_id": pair.pair_id, "verdict": verdict, "reason": result.get("reason", "")})
            # UNCERTAIN 不能在第一阶段被当作负类，否则级联会形成不可恢复的召回损失。
            if verdict in {"RELATED", "UNCERTAIN"}:
                related.append(pair)

    relationships: list[Neo4jRelationship] = []
    stage2_records: list[dict[str, Any]] = []
    for offset in range(0, len(related), batch_size):
        batch = related[offset : offset + batch_size]
        def build_label_prompt(prompt_pairs: Sequence[RelationPair]) -> str:
            return (
                "你是医学文本关系标签分类器。第一阶段只保留了可能存在直接关系的实体对，"
                "但它不是关系成立的保证，你必须再次核验证据。"
                "只使用 evidence，并且只能从每项 allowed_options 中选择一个类型和方向；"
                "若没有任何选项得到直接证据支持则 ABSTAIN。ASSOCIATED_WITH 不是兜底标签，"
                "只有原文明示医学关联且不存在更具体的因果、提示、包含、状态或分类关系时才能使用。"
                "方向 LEFT_TO_RIGHT 表示 left 指向 right，RIGHT_TO_LEFT 表示 right 指向 left。"
                "返回且只返回一个 JSON 对象：{\"results\":[{\"pair_id\":字符串,"
                "\"left_mention\":原样回显,\"right_mention\":原样回显,"
                "\"verdict\":\"CLASSIFIED\"或\"ABSTAIN\",\"relation_type\":字符串或null,"
                "\"direction\":字符串或null,\"reason\":字符串}]}。必须逐项返回。"
                "\nRELATION_DEFINITIONS_JSON:\n"
                + json.dumps(relation_definitions, ensure_ascii=False, separators=(",", ":"))
                + "\nPAIR_INPUT_JSON:\n"
                + json.dumps(
                    [_pair_payload(pair, include_options=True) for pair in prompt_pairs],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        result_map, batch_calls = await _invoke_validated_results_resilient(
            client, build_label_prompt, batch
        )
        model_call_count += batch_calls
        for pair in batch:
            result = result_map[pair.pair_id]
            verdict = result.get("verdict")
            relation_type, direction = result.get("relation_type"), result.get("direction")
            stage2_records.append({
                "pair_id": pair.pair_id, "verdict": verdict,
                "relation_type": relation_type, "direction": direction,
                "reason": result.get("reason", ""),
            })
            if verdict == "ABSTAIN":
                continue
            if verdict != "CLASSIFIED" or (relation_type, direction) not in pair.options:
                # 单项越过 Schema 选项属于分类错误，只丢弃该边并保留审计记录；
                # 不能因为一个坏标签让同批其他合法关系和整个 chunk 一起失败。
                stage2_records[-1]["verdict"] = "INVALID_LABEL"
                stage2_records[-1]["reason_code"] = "relation_classifier_label_invalid"
                continue
            if direction == "LEFT_TO_RIGHT":
                source_key, target_key = pair.left_key, pair.right_key
            else:
                source_key, target_key = pair.right_key, pair.left_key
            relationships.append(Neo4jRelationship(
                start_node_id=f"{chunk.chunk_id}:{source_key}",
                end_node_id=f"{chunk.chunk_id}:{target_key}",
                type=str(relation_type),
                properties={"exact_quote": pair.evidence_text},
            ))
    trace.record(
        "extraction/relation-two-stage",
        chunk_id=chunk.chunk_id,
        pair_count=len(pairs), related_count=len(related),
        classified_count=len(relationships),
    )
    audit = {
        "schema_version": "two-stage-relation-classification/v0.1",
        "chunk_id": chunk.chunk_id,
        "candidate_pair_count": len(pairs),
        "stage1_related_count": len(related),
        "stage2_classified_count": len(relationships),
        "classified_count": len(relationships),
        "model_call_count": model_call_count,
        "pairs": [_pair_payload(pair, include_options=True) for pair in pairs],
        "stage1": stage1_records,
        "stage2": stage2_records,
    }
    return Neo4jGraph(relationships=relationships), audit


async def classify_relationships_one_stage(
    client: Any,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    allowed_relation_types: Collection[str],
    trace: TraceRecorder = NULL_TRACE,
    batch_size: int = 12,
    group_by_evidence: bool = False,
) -> tuple[Neo4jGraph, dict[str, Any]]:
    """对结构候选对一次判断无关系、关系类型和方向。"""
    pairs = build_relation_pairs(
        chunk=chunk,
        schema=schema,
        nodes=nodes,
        allowed_relation_types=allowed_relation_types,
    )
    definitions = {
        relation_type: RELATION_SEMANTICS.get(relation_type, relation_type)
        for relation_type in sorted(allowed_relation_types)
    }
    decision_rules = {
        relation_type: RELATION_DECISION_RULES.get(relation_type, definitions[relation_type])
        for relation_type in definitions
    }
    relationships: list[Neo4jRelationship] = []
    records: list[dict[str, Any]] = []
    model_call_count = 0
    if group_by_evidence:
        evidence_groups: dict[tuple[int, int], list[RelationPair]] = {}
        for pair in pairs:
            evidence_groups.setdefault((pair.evidence_start, pair.evidence_end), []).append(pair)
        batches = [
            group[offset : offset + batch_size]
            for group in evidence_groups.values()
            for offset in range(0, len(group), batch_size)
        ]
    else:
        batches = [pairs[offset : offset + batch_size] for offset in range(0, len(pairs), batch_size)]
    for batch in batches:
        def build_prompt(prompt_pairs: Sequence[RelationPair]) -> str:
            return (
                "你是医学教材知识图谱的闭集关系分类器。实体已经冻结，不得创建、删除、改名或改类型。"
            "每个候选对只能使用自己的 evidence，并且必须从 allowed_options 中选择一个关系类型和方向，"
            "或者选择 NO_RELATION。NO_RELATION 是默认答案：证据不足、无法确定类型或方向时都必须选择它。"
            "不得使用外部医学知识，也不得因为医学常识上可能有关而建立关系。"
            "只有原文直接表达的关系才可分类：共同出现、并列、全称与缩写、公式变量、参考区间、"
            "检测时使用某材料、同表出现都不是关系。标题与其直接列表子项、检验组合与直接指标可以有关系；"
            "存在中间机制时不得跨过中间节点；多个条件共同支持结论时不得简化为任一条件单独指向结论。"
            "CAUSES 必须从原因指向结果；INDICATES 必须从证据或状态指向发现；IS_A 必须从下位项指向上位项；"
            "HAS_STATE 必须从指标指向状态；HAS_METRIC 必须从检验组合指向子组合或指标，"
            "或从复合指标指向其子指标。"
            "ASSOCIATED_WITH 不是兜底关系，只能用于原文以‘相关、有关、关联、联系’等明确谓词"
            "直接连接两个实体，且不存在更具体关系的情况；此时 trigger_quote 必须逐字摘录包含该谓词的证据。"
            "如果理由中出现‘原文未明说但可推断’、‘同表’、‘并列’或‘共同出现’，必须选择 NO_RELATION。"
            "本批次只按 RELATION_DECISION_RULES_JSON 中列出的正类规则判断；其他关系即使成立，也选择 NO_RELATION。"
            "方向 LEFT_TO_RIGHT 表示 left 指向 right，RIGHT_TO_LEFT 表示 right 指向 left。"
            "只返回一个 JSON 对象，顶层只能有 results。results 必须按输入顺序为每个 pair_id 返回一项；"
            "pair_id 是响应身份，不要重复或改写实体 mention："
            "{\"pair_id\":字符串,"
            "\"verdict\":\"CLASSIFIED\"或\"NO_RELATION\","
            "\"relation_type\":字符串或null,\"direction\":字符串或null,"
            "\"trigger_quote\":字符串或null,\"reason\":简短中文理由}。"
            "CLASSIFIED 时类型和方向必须来自 allowed_options，trigger_quote 必须是 evidence 中逐字出现的"
            "最小关系证据；NO_RELATION 时三字段必须为 null。\n"
            "RELATION_DEFINITIONS_JSON:\n"
            + json.dumps(definitions, ensure_ascii=False, separators=(",", ":"))
            + "\nRELATION_DECISION_RULES_JSON:\n"
            + json.dumps(decision_rules, ensure_ascii=False, separators=(",", ":"))
            + "\nPAIR_INPUT_JSON:\n"
                + json.dumps(
                    [_pair_payload(pair, include_options=True) for pair in prompt_pairs],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        result_map, batch_calls = await _invoke_validated_results_resilient(
            client, build_prompt, batch
        )
        model_call_count += batch_calls
        for pair in batch:
            result = result_map[pair.pair_id]
            verdict = result.get("verdict")
            relation_type = result.get("relation_type")
            direction = result.get("direction")
            record = {
                "pair_id": pair.pair_id,
                "verdict": verdict,
                "relation_type": relation_type,
                "direction": direction,
                "trigger_quote": result.get("trigger_quote"),
                "reason": result.get("reason", ""),
            }
            if verdict == "NO_RELATION":
                if (
                    relation_type is not None
                    or direction is not None
                    or result.get("trigger_quote") is not None
                ):
                    record["verdict"] = "INVALID_LABEL"
                    record["reason_code"] = "relation_classifier_negative_has_label"
                records.append(record)
                continue
            if verdict != "CLASSIFIED" or (relation_type, direction) not in pair.options:
                record["verdict"] = "INVALID_LABEL"
                record["reason_code"] = "relation_classifier_label_invalid"
                records.append(record)
                continue
            if not _has_verbatim_trigger(pair.evidence_text, result.get("trigger_quote")):
                record["verdict"] = "INVALID_EVIDENCE"
                record["reason_code"] = "relation_trigger_missing"
                records.append(record)
                continue
            if relation_type == "ASSOCIATED_WITH" and not _has_explicit_association_trigger(
                pair.evidence_text, result.get("trigger_quote")
            ):
                record["verdict"] = "INVALID_EVIDENCE"
                record["reason_code"] = "association_trigger_missing"
                records.append(record)
                continue
            relationship = _relationship_from_classification(
                chunk=chunk,
                pair=pair,
                relation_type=str(relation_type),
                direction=str(direction),
            )
            if relationship is None:
                record["verdict"] = "INVALID_EVIDENCE"
                record["reason_code"] = "relation_classifier_evidence_lacks_endpoints"
            else:
                relationships.append(relationship)
            records.append(record)

    trace.record(
        "extraction/relation-one-stage",
        chunk_id=chunk.chunk_id,
        pair_count=len(pairs),
        classified_count=len(relationships),
    )
    return Neo4jGraph(relationships=relationships), {
        "schema_version": "one-stage-structure-relation-classification/v0.1",
        "chunk_id": chunk.chunk_id,
        "candidate_pair_count": len(pairs),
        "classified_count": len(relationships),
        "model_call_count": model_call_count,
        "batch_policy": "same_evidence_unit" if group_by_evidence else "mixed_evidence_units",
        "pairs": [_pair_payload(pair, include_options=True) for pair in pairs],
        "results": records,
    }


async def classify_relationships_evidence_grouped(
    client: Any,
    **kwargs: Any,
) -> tuple[Neo4jGraph, dict[str, Any]]:
    """一次只分类同一句、同一列表窗口或同一表格行中的候选对。"""
    return await classify_relationships_one_stage(
        client, group_by_evidence=True, **kwargs
    )


async def classify_relationships_verified(
    client: Any,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    allowed_relation_types: Collection[str],
    trace: TraceRecorder = NULL_TRACE,
    batch_size: int = 12,
) -> tuple[Neo4jGraph, dict[str, Any]]:
    """先完成闭集分类，再只复核已判正候选的直接证据充分性。"""
    graph, classification_audit = await classify_relationships_one_stage(
        client,
        chunk=chunk,
        schema=schema,
        nodes=nodes,
        allowed_relation_types=allowed_relation_types,
        trace=trace,
        batch_size=batch_size,
    )
    pair_by_id = {
        pair.pair_id: pair
        for pair in build_relation_pairs(
            chunk=chunk,
            schema=schema,
            nodes=nodes,
            allowed_relation_types=allowed_relation_types,
        )
    }
    positive_records = [
        record
        for record in classification_audit.get("results", [])
        if isinstance(record, Mapping) and record.get("verdict") == "CLASSIFIED"
    ]
    proposed = [
        (pair_by_id[str(record["pair_id"])], record)
        for record in positive_records
        if str(record.get("pair_id")) in pair_by_id
    ]
    verifier_records: list[dict[str, Any]] = []
    kept_pair_ids: set[str] = set()
    verifier_calls = 0
    for offset in range(0, len(proposed), batch_size):
        batch = proposed[offset : offset + batch_size]
        record_by_id = {
            pair.pair_id: record
            for pair, record in batch
        }

        def build_verifier_prompt(prompt_pairs: Sequence[RelationPair]) -> str:
            return (
                "你是医学教材关系候选的证据复核器。候选由同一抽取管线生成，但可能过度推断。"
                "你只能使用每项 evidence，不能使用外部医学知识，也不能改变实体、关系类型或方向。"
                "仅当原文直接支持 proposed_relation_type 和 proposed_direction 时选择 KEEP。"
                "如果只是共同出现、并列、举例、同表出现、共享上位项、全称缩写、公式变量、"
                "参考范围或检测材料，选择 DROP。若存在更具体关系，不能用 ASSOCIATED_WITH 兜底。"
                "存在中间实体或联合条件被拆开时选择 DROP。"
                "返回且只返回一个 JSON 对象，必须逐项按输入顺序返回："
                "{\"results\":[{\"pair_id\":字符串,\"left_mention\":原样回显,"
                "\"right_mention\":原样回显,\"verdict\":\"KEEP\"或\"DROP\",\"reason\":字符串}]}。"
                "\nCANDIDATE_INPUT_JSON:\n"
                + json.dumps([
                    {
                        **_pair_payload(pair, include_options=False),
                        "proposed_relation_type": record_by_id[pair.pair_id].get("relation_type"),
                        "proposed_direction": record_by_id[pair.pair_id].get("direction"),
                    }
                    for pair in prompt_pairs
                ], ensure_ascii=False, separators=(",", ":"))
            )

        prompt_pairs = [pair for pair, _record in batch]
        result_map, batch_calls = await _invoke_validated_results_resilient(
            client, build_verifier_prompt, prompt_pairs
        )
        verifier_calls += batch_calls
        for pair in prompt_pairs:
            result = result_map[pair.pair_id]
            verdict = result.get("verdict")
            reason_code = None
            if verdict not in {"KEEP", "DROP"}:
                verdict = "DROP"
                reason_code = "relation_verifier_verdict_invalid"
            verifier_record = {
                "pair_id": pair.pair_id,
                "verdict": verdict,
                "reason": result.get("reason", ""),
            }
            if reason_code is not None:
                verifier_record["reason_code"] = reason_code
            verifier_records.append(verifier_record)
            if verdict == "KEEP":
                kept_pair_ids.add(pair.pair_id)

    kept_identities: set[tuple[str, str, str]] = set()
    for pair, record in proposed:
        if pair.pair_id not in kept_pair_ids:
            continue
        if record.get("direction") == "LEFT_TO_RIGHT":
            source_key, target_key = pair.left_key, pair.right_key
        else:
            source_key, target_key = pair.right_key, pair.left_key
        kept_identities.add((
            f"{chunk.chunk_id}:{source_key}",
            str(record.get("relation_type")),
            f"{chunk.chunk_id}:{target_key}",
        ))
    kept_relationships = [
        relationship
        for relationship in graph.relationships
        if (
            relationship.start_node_id,
            relationship.type,
            relationship.end_node_id,
        ) in kept_identities
    ]
    audit = {
        "schema_version": "verified-relation-classification/v0.1",
        "chunk_id": chunk.chunk_id,
        "candidate_pair_count": classification_audit.get("candidate_pair_count", 0),
        "classified_before_verification": len(graph.relationships),
        "classified_count": len(kept_relationships),
        "model_call_count": int(classification_audit.get("model_call_count", 0)) + verifier_calls,
        "classification": classification_audit,
        "verification": verifier_records,
    }
    trace.record(
        "extraction/relation-verified",
        chunk_id=chunk.chunk_id,
        classified_before_verification=len(graph.relationships),
        classified_count=len(kept_relationships),
    )
    return Neo4jGraph(relationships=kept_relationships), audit


async def classify_relationships_routed(
    client: Any,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    allowed_relation_types: Collection[str],
    trace: TraceRecorder = NULL_TRACE,
    batch_size: int = 12,
) -> tuple[Neo4jGraph, dict[str, Any]]:
    """按状态、结构和医学语义分路分类，显式关联留给独立抽取任务。"""
    allowed = set(allowed_relation_types)
    relationships: list[Neo4jRelationship] = []
    route_audits: dict[str, Any] = {}
    model_call_count = 0
    candidate_pair_count = 0
    for route_name, route_types in RELATION_ROUTE_DEFINITIONS:
        selected_types = allowed & route_types
        if not selected_types:
            continue
        route_graph, route_audit = await classify_relationships_one_stage(
            client,
            chunk=chunk,
            schema=schema,
            nodes=nodes,
            allowed_relation_types=selected_types,
            trace=trace,
            batch_size=batch_size,
        )
        relationships.extend(route_graph.relationships)
        route_audits[route_name] = route_audit
        candidate_pair_count += int(route_audit.get("candidate_pair_count", 0))
        model_call_count += int(route_audit.get("model_call_count", 0))
    trace.record(
        "extraction/relation-routed",
        chunk_id=chunk.chunk_id,
        classified_count=len(relationships),
    )
    return Neo4jGraph(relationships=relationships), {
        "schema_version": "routed-relation-classification/v0.1",
        "chunk_id": chunk.chunk_id,
        "candidate_pair_count": candidate_pair_count,
        "classified_count": len(relationships),
        "model_call_count": model_call_count,
        "excluded_relation_types": sorted(allowed & {"ASSOCIATED_WITH"}),
        "routes": route_audits,
    }
