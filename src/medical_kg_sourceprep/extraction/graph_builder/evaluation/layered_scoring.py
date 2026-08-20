"""对 mention、canonical、映射和关系四层分别计算标准 P/R/F1。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _metrics(predicted: set[tuple[Any, ...]], expected: set[tuple[Any, ...]]) -> dict[str, Any]:
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not expected else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_total": len(predicted),
        "gold_total": len(expected),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "precision_percent": round(precision * 100, 2),
        "recall_percent": round(recall * 100, 2),
        "f1_percent": round(f1 * 100, 2),
    }


def _canonical_identity(entity_type: object, label: object) -> tuple[str, str]:
    return str(entity_type), "".join(str(label).split())


def project_layered_predictions(graph: Mapping[str, Any]) -> dict[str, set[tuple[Any, ...]]]:
    """从候选图投影四层预测身份；所有证据坐标保持 canonical chunk 坐标。"""
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, Mapping)]
    ordinary_nodes = [node for node in nodes if node.get("entity_type") != "RuleDefinition"]
    mention_by_key: dict[str, tuple[Any, ...]] = {}
    canonical_by_key: dict[str, tuple[str, str]] = {}
    mentions: set[tuple[Any, ...]] = set()
    canonical_entities: set[tuple[Any, ...]] = set()
    links: set[tuple[Any, ...]] = set()

    for node in ordinary_nodes:
        key = node.get("candidate_key")
        source_ref = node.get("source_ref")
        entity_type = node.get("entity_type")
        mention = node.get("mention")
        if not isinstance(key, str) or not isinstance(source_ref, Mapping):
            continue
        start = source_ref.get("mention_char_start", source_ref.get("char_start"))
        end = source_ref.get("mention_char_end", source_ref.get("char_end"))
        chunk_id = source_ref.get("chunk_id")
        if not isinstance(chunk_id, str) or not isinstance(start, int) or not isinstance(end, int):
            continue
        mention_identity = (str(entity_type), chunk_id, start, end)
        canonical_label = node.get("canonical_name_candidate") or mention
        canonical_identity = _canonical_identity(entity_type, canonical_label)
        mention_by_key[key] = mention_identity
        canonical_by_key[key] = canonical_identity
        mentions.add(mention_identity)
        canonical_entities.add(canonical_identity)
        links.add((*mention_identity, *canonical_identity))

    relationships = {
        (
            *canonical_by_key[str(item["source_candidate_key"])],
            str(item["relation_type"]),
            *canonical_by_key[str(item["target_candidate_key"])],
        )
        for item in graph.get("relationships", [])
        if isinstance(item, Mapping)
        and item.get("relation_type") not in {"RULE_INPUT", "RULE_OUTPUT"}
        and str(item.get("source_candidate_key")) in canonical_by_key
        and str(item.get("target_candidate_key")) in canonical_by_key
    }
    return {
        "mentions": mentions,
        "canonical_entities": canonical_entities,
        "links": links,
        "relationships": relationships,
    }


def project_layered_gold(case: Mapping[str, Any]) -> dict[str, set[tuple[Any, ...]]]:
    """从 v0.4 主标注投影四套互不混淆的监督金标。"""
    canonical_by_id = {
        str(item["canonical_id"]): _canonical_identity(
            item["entity_type"], item["canonical_label"]
        )
        for item in case.get("canonical_entities", [])
    }
    evidence_by_id = {
        str(item["evidence_unit_id"]): item for item in case.get("evidence_units", [])
    }
    mentions: set[tuple[Any, ...]] = set()
    links: set[tuple[Any, ...]] = set()
    for link in case.get("mention_to_canonical_links", []):
        evidence = evidence_by_id[str(link["evidence_unit_id"])]
        if not evidence.get("mention_eligible"):
            continue
        canonical = canonical_by_id[str(link["canonical_id"])]
        mention = (
            canonical[0],
            str(evidence["chunk_id"]),
            int(evidence["start"]),
            int(evidence["end"]),
        )
        mentions.add(mention)
        links.add((*mention, *canonical))

    relationships = {
        (
            *canonical_by_id[str(item["source_canonical_id"])],
            str(item["relation_type"]),
            *canonical_by_id[str(item["target_canonical_id"])],
        )
        for item in case.get("relationships", [])
    }
    return {
        "mentions": mentions,
        "canonical_entities": set(canonical_by_id.values()),
        "links": links,
        "relationships": relationships,
    }


def _project_scoped(
    graph: Mapping[str, Any], case: Mapping[str, Any]
) -> tuple[dict[str, set[tuple[Any, ...]]], dict[str, set[tuple[Any, ...]]]]:
    """按 v0.5 声明的证据域投影预测与金标，范围外候选不计作 FP。"""
    evidence = {
        str(item["evidence_unit_id"]): item for item in case.get("evidence_units", [])
    }
    canonical = {
        str(item["canonical_id"]): item for item in case.get("canonical_entities", [])
    }
    alias_to_id: dict[tuple[str, str], str] = {}
    for canonical_id, item in canonical.items():
        labels = item.get("accepted_surface_forms", [item["canonical_label"]])
        for label in labels:
            alias_to_id[_canonical_identity(item["entity_type"], label)] = canonical_id

    strict_spans = {
        (str(item["chunk_id"]), int(item["start"]), int(item["end"]))
        for item in evidence.values() if item.get("mention_eligible")
    }
    context_spans = [
        (str(item["chunk_id"]), int(item["start"]), int(item["end"]))
        for item in evidence.values()
    ]
    nodes = [
        item for item in graph.get("nodes", [])
        if isinstance(item, Mapping) and item.get("entity_type") != "RuleDefinition"
    ]
    key_to_canonical: dict[str, str] = {}
    predicted_mentions: set[tuple[Any, ...]] = set()
    predicted_canonical: set[tuple[Any, ...]] = set()
    predicted_links: set[tuple[Any, ...]] = set()
    for node in nodes:
        source_ref = node.get("source_ref")
        key = node.get("candidate_key")
        if not isinstance(source_ref, Mapping) or not isinstance(key, str):
            continue
        chunk_id = source_ref.get("chunk_id")
        start = source_ref.get("mention_char_start", source_ref.get("char_start"))
        end = source_ref.get("mention_char_end", source_ref.get("char_end"))
        if not isinstance(chunk_id, str) or not isinstance(start, int) or not isinstance(end, int):
            continue
        entity_type = str(node.get("entity_type"))
        mention_identity = (entity_type, chunk_id, start, end)
        label = node.get("canonical_name_candidate") or node.get("mention")
        resolved = alias_to_id.get(_canonical_identity(entity_type, label))
        if resolved is not None:
            key_to_canonical[key] = resolved
        if (chunk_id, start, end) in strict_spans:
            predicted_mentions.add(mention_identity)
            link_target: tuple[Any, ...] = (
                (resolved,) if resolved is not None
                else ("UNRESOLVED", entity_type, "".join(str(label).split()))
            )
            predicted_links.add((*mention_identity, *link_target))
        in_context = any(
            chunk_id == scope_chunk and max(start, scope_start) < min(end, scope_end)
            for scope_chunk, scope_start, scope_end in context_spans
        )
        if in_context:
            predicted_canonical.add(
                (resolved,) if resolved is not None
                else ("UNRESOLVED", entity_type, "".join(str(label).split()))
            )

    predicted_relationships = {
        (
            key_to_canonical[str(item["source_candidate_key"])],
            str(item["relation_type"]),
            key_to_canonical[str(item["target_candidate_key"])],
        )
        for item in graph.get("relationships", [])
        if isinstance(item, Mapping)
        and item.get("relation_type") not in {"RULE_INPUT", "RULE_OUTPUT"}
        and str(item.get("source_candidate_key")) in key_to_canonical
        and str(item.get("target_candidate_key")) in key_to_canonical
    }

    gold_mentions: set[tuple[Any, ...]] = set()
    gold_links: set[tuple[Any, ...]] = set()
    for link in case.get("mention_to_canonical_links", []):
        item = evidence[str(link["evidence_unit_id"])]
        if not item.get("mention_eligible"):
            continue
        canonical_item = canonical[str(link["canonical_id"])]
        mention = (
            str(canonical_item["entity_type"]), str(item["chunk_id"]),
            int(item["start"]), int(item["end"]),
        )
        gold_mentions.add(mention)
        gold_links.add((*mention, str(link["canonical_id"])))
    gold = {
        "mentions": gold_mentions,
        "canonical_entities": {(canonical_id,) for canonical_id in canonical},
        "links": gold_links,
        "relationships": {
            (
                str(item["source_canonical_id"]), str(item["relation_type"]),
                str(item["target_canonical_id"]),
            )
            for item in case.get("relationships", [])
        },
    }
    predicted = {
        "mentions": predicted_mentions,
        "canonical_entities": predicted_canonical,
        "links": predicted_links,
        "relationships": predicted_relationships,
    }
    return predicted, gold


def score_layered_case(graph: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    if any("accepted_surface_forms" in item for item in case.get("canonical_entities", [])):
        predicted, expected = _project_scoped(graph, case)
    else:
        predicted = project_layered_predictions(graph)
        expected = project_layered_gold(case)
    return {name: _metrics(predicted[name], expected[name]) for name in expected}


def aggregate_layered_scores(scores: Iterable[Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """按 TP/FP/FN micro 聚合多个案例，避免对案例百分比取平均。"""
    score_list = list(scores)
    result: dict[str, Any] = {}
    for category in ("mentions", "canonical_entities", "links", "relationships"):
        tp = sum(int(score[category]["tp"]) for score in score_list)
        fp = sum(int(score[category]["fp"]) for score in score_list)
        fn = sum(int(score[category]["fn"]) for score in score_list)
        # 聚合只依赖各案例的混淆矩阵计数，不对案例百分比取平均。
        precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[category] = {
            "tp": tp, "fp": fp, "fn": fn,
            "predicted_total": tp + fp, "gold_total": tp + fn,
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(f1, 6),
            "precision_percent": round(precision * 100, 2),
            "recall_percent": round(recall * 100, 2),
            "f1_percent": round(f1 * 100, 2),
        }
    return result
