"""按典型案例的目标项与禁止项合同评估候选图。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import re
from typing import Any


def _prf1_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """根据标准监督计数计算 P/R/F1，并保留百分比展示值。"""
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6),
        "precision_percent": round(100 * precision, 2),
        "recall_percent": round(100 * recall, 2),
        "f1_percent": round(100 * f1, 2),
    }


def _maximum_matches(
    predicted: Sequence[Any],
    expected: Sequence[Any],
    is_match: Callable[[Any, Any], bool],
) -> tuple[set[int], set[int]]:
    """求确定性的最大一对一匹配，防止一个预测重复贡献多个 TP。"""
    expected_to_prediction: dict[int, int] = {}

    def augment(predicted_index: int, visited: set[int]) -> bool:
        for expected_index, expected_item in enumerate(expected):
            if expected_index in visited or not is_match(
                predicted[predicted_index], expected_item
            ):
                continue
            visited.add(expected_index)
            previous = expected_to_prediction.get(expected_index)
            if previous is None or augment(previous, visited):
                expected_to_prediction[expected_index] = predicted_index
                return True
        return False

    for predicted_index in range(len(predicted)):
        augment(predicted_index, set())
    return set(expected_to_prediction.values()), set(expected_to_prediction)


def _supervised_category_metrics(
    predicted: Sequence[Any],
    expected: Sequence[Any],
    matched_predictions: set[int],
    matched_expected: set[int],
) -> dict[str, Any]:
    """生成一类预测相对完整金标的标准监督指标和错误明细。"""
    tp = len(matched_predictions)
    result = {
        "predicted_total": len(predicted),
        "target_total": len(expected),
        "matched": tp,
        "missed": len(expected) - len(matched_expected),
        "coverage": round(tp / len(expected), 6) if expected else 1.0,
        "matched_targets": [
            item for index, item in enumerate(expected) if index in matched_expected
        ],
        "missed_targets": [
            item for index, item in enumerate(expected) if index not in matched_expected
        ],
        "false_positive_predictions": [
            item for index, item in enumerate(predicted) if index not in matched_predictions
        ],
    }
    result.update(_prf1_counts(tp, len(predicted) - tp, len(expected) - tp))
    return result


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
        rules.add((
            str(node.get("rule_stage_candidate", "UNKNOWN")),
            str(node.get("rule_logic_candidate", "UNKNOWN")),
            inputs,
            outputs,
        ))
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


def _reference_is_inside_scope(
    reference: Mapping[str, Any], scopes: Sequence[Mapping[str, Any]]
) -> bool:
    """判断一条证据是否完整落在同一 chunk 的冻结评测范围内。"""
    chunk_id = reference.get("chunk_id")
    # 节点优先使用 mention 的精确位置；其余工件使用整条证据的位置。
    start = reference.get("mention_char_start", reference.get("char_start"))
    end = reference.get("mention_char_end", reference.get("char_end"))
    if (
        not isinstance(chunk_id, str)
        or not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
    ):
        return False
    return any(
        scope.get("chunk_id") == chunk_id
        and isinstance(scope.get("start"), int)
        and not isinstance(scope.get("start"), bool)
        and isinstance(scope.get("end"), int)
        and not isinstance(scope.get("end"), bool)
        and int(scope["start"]) <= start < end <= int(scope["end"])
        for scope in scopes
    )


def _mapping_references(value: Mapping[str, Any], plural_key: str) -> list[Mapping[str, Any]]:
    """读取候选上的复数证据；没有时回退到单数 source_ref。"""
    plural = value.get(plural_key)
    if isinstance(plural, list):
        references = [item for item in plural if isinstance(item, Mapping)]
        if references:
            return references
    source_ref = value.get("source_ref")
    return [source_ref] if isinstance(source_ref, Mapping) else []


def filter_candidate_graph_by_scopes(
    graph: Mapping[str, Any], scopes: Sequence[Mapping[str, Any]]
) -> dict[str, list[Any]]:
    """按证据位置裁出案例闭集范围内的候选子图。"""
    if not scopes:
        return {
            "nodes": list(graph.get("nodes", [])),
            "relationships": list(graph.get("relationships", [])),
        }

    nodes: list[Any] = []
    included_keys: set[str] = set()
    for node in graph.get("nodes", []):
        if not isinstance(node, Mapping):
            continue
        if node.get("entity_type") == "RuleDefinition":
            references = _mapping_references(node, "rule_evidence_refs")
            included = bool(references) and all(
                _reference_is_inside_scope(reference, scopes) for reference in references
            )
        else:
            references = _mapping_references(node, "source_refs")
            included = any(
                _reference_is_inside_scope(reference, scopes) for reference in references
            )
        if not included:
            continue
        nodes.append(node)
        key = node.get("candidate_key")
        if isinstance(key, str):
            included_keys.add(key)

    relationships: list[Any] = []
    for relationship in graph.get("relationships", []):
        if not isinstance(relationship, Mapping):
            continue
        if (
            relationship.get("source_candidate_key") not in included_keys
            or relationship.get("target_candidate_key") not in included_keys
        ):
            continue
        references = _mapping_references(relationship, "relation_evidence_refs")
        if references and all(
            _reference_is_inside_scope(reference, scopes) for reference in references
        ):
            relationships.append(relationship)
    return {"nodes": nodes, "relationships": relationships}


def _normalized_surface(value: str) -> str:
    """只消除空白和状态 copula，不把全称与缩写机械合并。"""
    compact = re.sub(r"\s+", "", value)
    return re.sub(r"为(?=(?:阳性|阴性|正常|异常|增高|降低|不升高))", "", compact)


def _deduplicate_projected_items(
    items: Sequence[tuple[Any, ...]], key: Callable[[tuple[Any, ...]], tuple[Any, ...]]
) -> list[tuple[Any, ...]]:
    """按评分身份去重，同时保留最先出现的原始 mention 供错误明细展示。"""
    unique: list[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in items:
        identity = key(item)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
    return unique


def _equivalent_mention(left: str, right: str, source_text: str) -> bool:
    """只接受完全相同或原文明示的“全称（缩写）”等价，不使用外部词典。"""
    if left == right:
        return True
    if _normalized_surface(left) == _normalized_surface(right):
        return True

    def source_declares_alias(left_base: str, right_base: str) -> bool:
        for full_name, abbreviation in ((left_base, right_base), (right_base, left_base)):
            pattern = rf"{re.escape(full_name)}\s*[（(]\s*{re.escape(abbreviation)}\s*[）)]"
            if re.search(pattern, source_text):
                return True
        return False

    if source_declares_alias(left, right):
        return True
    # 状态节点保留“指标 + 状态”身份。若状态后缀相同，且原文明示两个指标基名
    # 为全称/缩写，则全称状态与缩写状态也是同一来源概念。
    for suffix in ("不升高", "持续下降", "持续升高", "阳性", "阴性", "正常", "异常", "增高", "降低"):
        if left.endswith(suffix) and right.endswith(suffix):
            left_base = left[: -len(suffix)].strip()
            right_base = right[: -len(suffix)].strip()
            if left_base and right_base and source_declares_alias(left_base, right_base):
                return True
    return False


def _relationship_matches(
    predicted: tuple[Any, ...], expected: tuple[Any, ...], source_text: str
) -> bool:
    """按端点、类型和方向判断一条普通关系是否命中目标。"""
    return (
        predicted[1] == expected[1]
        and _equivalent_mention(str(predicted[0]), str(expected[0]), source_text)
        and _equivalent_mention(str(predicted[2]), str(expected[2]), source_text)
    )


def score_relationship_tier(
    graph: Mapping[str, Any],
    gold_case: Mapping[str, Any],
    *,
    targets: Sequence[Sequence[Any]],
    ignored_targets: Sequence[Sequence[Any]] = (),
    source_text: str = "",
) -> dict[str, Any]:
    """评价一个关系层级，并忽略已归入其他层级的正确预测。"""
    raw_scopes = gold_case.get("evaluation_scopes", [])
    scopes = [item for item in raw_scopes if isinstance(item, Mapping)] \
        if isinstance(raw_scopes, list) else []
    if scopes:
        graph = filter_candidate_graph_by_scopes(graph, scopes)
    projected = project_candidate_graph(graph)
    predicted = _deduplicate_projected_items(
        sorted(projected["relationships"]),
        lambda item: (
            _normalized_surface(str(item[0])),
            item[1],
            _normalized_surface(str(item[2])),
        ),
    )
    ignored = [tuple(item) for item in ignored_targets]
    scored_predictions = [
        item for item in predicted
        if not any(_relationship_matches(item, target, source_text) for target in ignored)
    ]
    expected = [tuple(item) for item in targets]
    matched_predictions, matched_expected = _maximum_matches(
        scored_predictions,
        expected,
        lambda predicted_item, expected_item: _relationship_matches(
            predicted_item, expected_item, source_text
        ),
    )
    result = _supervised_category_metrics(
        scored_predictions,
        expected,
        matched_predictions,
        matched_expected,
    )
    result["ignored_cross_tier_predictions"] = len(predicted) - len(scored_predictions)
    return result


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
        raw_logic = node.get("rule_logic_candidate")
        logic = str(raw_logic) if isinstance(raw_logic, str) else (
            "FORMULA" if "formula" in roles else None
        )
        records.append({
            "candidate_key": rule_key,
            "stage": str(node.get("rule_stage_candidate", "UNKNOWN")),
            "inputs": inputs,
            "outputs": outputs,
            "evidence_text": "\n".join(quotes),
            "logic": logic,
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
    if not isinstance(expected_logic, str) or candidate.get("logic") != expected_logic:
        return False
    candidate_inputs = tuple(str(item) for item in candidate.get("inputs", ()))
    candidate_outputs = tuple(str(item) for item in candidate.get("outputs", ()))
    evidence_text = str(candidate.get("evidence_text", ""))

    def endpoint_is_supported(target: str, endpoints: tuple[str, ...]) -> bool:
        return any(_equivalent_mention(target, endpoint, source_text) for endpoint in endpoints)

    expected_outputs = tuple(str(item) for item in expected.get("outputs", []))
    expected_inputs = tuple(str(item) for item in expected.get("inputs", []))
    # 输出集合必须精确，不允许一个带额外错误输出的候选靠包含金标子集得分。
    if len(candidate_outputs) != len(expected_outputs) or not all(
        endpoint_is_supported(item, candidate_outputs) for item in expected_outputs
    ):
        return False
    # 图上的候选输入也不能超出人工规则定义；未建图的公式参数仍可由证据覆盖。
    if any(
        not any(_equivalent_mention(candidate_input, target, source_text) for target in expected_inputs)
        for candidate_input in candidate_inputs
    ):
        return False
    # 图语义联合规则必须形成完整 RULE_INPUT 子图。原文证据只能证明语义来源，不能替代
    # 知识图谱端点；否则一个没有输入边的 PARTIAL 规则也会被错误计为命中。
    if expected.get("rule_stage") == "GRAPH_COMPOSITE":
        return len(candidate_inputs) == len(expected_inputs) and all(
            endpoint_is_supported(target, candidate_inputs) for target in expected_inputs
        )
    # 保留对历史公式金标的兼容匹配；当前第一阶段图规则金标不会再包含执行器参数。
    def input_is_supported(target: str) -> bool:
        if endpoint_is_supported(target, candidate_inputs) or target in evidence_text:
            return True
        # 人工金标可以显式裁定规范源 OCR；抽取候选仍必须忠实保留原始坏字符。
        return target in source_text and any(
            "OCR" in note and target in note for note in manual_review_notes
        )

    return all(input_is_supported(item) for item in expected_inputs)


def score_candidate_graph(
    graph: Mapping[str, Any], gold_case: Mapping[str, Any], *, source_text: str = ""
) -> dict[str, Any]:
    """在案例冻结范围内将候选图与人工金标一对一匹配。"""
    raw_scopes = gold_case.get("evaluation_scopes", [])
    scopes = [item for item in raw_scopes if isinstance(item, Mapping)] \
        if isinstance(raw_scopes, list) else []
    if scopes:
        graph = filter_candidate_graph_by_scopes(graph, scopes)
    projected = project_candidate_graph(graph)
    expected_entities = [tuple(item) for item in gold_case.get("entities", [])]
    predicted_entities = _deduplicate_projected_items(
        sorted(projected["entities"]),
        lambda item: (item[0], _normalized_surface(str(item[1]))),
    )
    matched_predicted_entities, matched_entities = _maximum_matches(
        predicted_entities,
        expected_entities,
        lambda predicted, expected: (
            predicted[0] == expected[0]
            and _equivalent_mention(str(predicted[1]), str(expected[1]), source_text)
        ),
    )

    expected_relationships = [tuple(item) for item in gold_case.get("relationships", [])]
    predicted_relationships = _deduplicate_projected_items(
        sorted(projected["relationships"]),
        lambda item: (
            _normalized_surface(str(item[0])),
            item[1],
            _normalized_surface(str(item[2])),
        ),
    )
    matched_predicted_relationships, matched_relationships = _maximum_matches(
        predicted_relationships,
        expected_relationships,
        lambda predicted, expected: _relationship_matches(
            predicted, expected, source_text
        ),
    )

    expected_rules = [item for item in gold_case.get("rules", []) if isinstance(item, Mapping)]
    manual_review_notes = tuple(
        str(item) for item in gold_case.get("review_notes", []) if isinstance(item, str)
    )
    candidate_rules: list[dict[str, Any]] = []
    seen_rule_signatures: set[tuple[Any, ...]] = set()
    # 规则金标身份不包含名称或证据位置；同一语义规则跨证据重复时只算一个预测。
    for candidate_rule in _rule_records(graph):
        signature = (
            candidate_rule.get("stage"),
            candidate_rule.get("logic"),
            tuple(candidate_rule.get("inputs", ())),
            tuple(candidate_rule.get("outputs", ())),
        )
        if signature in seen_rule_signatures:
            continue
        seen_rule_signatures.add(signature)
        candidate_rules.append(candidate_rule)
    matched_predicted_rules, matched_rules = _maximum_matches(
        candidate_rules,
        expected_rules,
        lambda predicted, expected: _rule_matches(
            predicted, expected, source_text, manual_review_notes
        ),
    )

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

    public_candidate_rules = [{
        "rule_stage": item.get("stage"),
        "logic": item.get("logic"),
        "inputs": list(item.get("inputs", ())),
        "outputs": list(item.get("outputs", ())),
    } for item in candidate_rules]
    scores: dict[str, Any] = {
        "entities": _supervised_category_metrics(
            predicted_entities,
            expected_entities,
            matched_predicted_entities,
            matched_entities,
        ),
        "relationships": _supervised_category_metrics(
            predicted_relationships,
            expected_relationships,
            matched_predicted_relationships,
            matched_relationships,
        ),
        "rules": _supervised_category_metrics(
            public_candidate_rules,
            expected_rules,
            matched_predicted_rules,
            matched_rules,
        ),
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
    scores["scoring_contract"] = (
        "standard supervised TP/FP/FN against scoped closed-world human gold; "
        "candidate evidence must be fully inside an evaluation scope; one-to-one matching; "
        "source-explicit full-name/abbreviation aliases; "
        "first-pass rules are graph-semantic joint conditions only; executor formulas, ranges, "
        "thresholds and single-indicator temporal calculations are out of scope"
    )
    scores["manual_adjudications"] = list(manual_review_notes)
    return scores
