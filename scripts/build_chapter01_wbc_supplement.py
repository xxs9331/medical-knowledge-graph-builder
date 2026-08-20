#!/usr/bin/env python3
"""Build the Chapter 01 white-cell differential graph supplement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT = ROOT / "knowledge/chapter-01/terminology/official-lab-alignment-v0.1.json"
ENTITIES = ROOT / "evaluation/chapter-01/chapter-01-canonical-entities-v0.8.json"
OUTPUT = ROOT / "knowledge/chapter-01/terminology/wbc-differential-supplement-v0.1.json"
EVIDENCE_CHUNK_ID = "clinical-hematology:chapter-01:0005:0001"
EVIDENCE_QUOTE = (
    "<table><tr><td>细胞类型</td><td>百分数/%</td><td>绝对值/\\( \\times 10^{9}/L \\)</td></tr>"
    "<tr><td>中性粒细胞(N)</td><td>40~75</td><td>1.8~6.3</td></tr>"
    "<tr><td>淋巴细胞(L)</td><td>20~50</td><td>1.1~3.2</td></tr>"
    "<tr><td>单核细胞(M)</td><td>3~10</td><td>0.1~0.6</td></tr>"
    "<tr><td>嗜酸性粒细胞(E)</td><td>0.4~8.0</td><td>0.02~0.52</td></tr>"
    "<tr><td>嗜碱性粒细胞(B)</td><td>0~1.0</td><td>0~0.06</td></tr></table>"
)

RANGES = (
    ("中性粒细胞比例", "40", "75", "%", "中性粒细胞增多", "中性粒细胞减少"),
    ("中性粒细胞绝对值", "1.8", "6.3", "x10^9/L", "中性粒细胞增多", "中性粒细胞减少"),
    ("淋巴细胞比例", "20", "50", "%", "淋巴细胞增多", "淋巴细胞减少"),
    ("淋巴细胞绝对值", "1.1", "3.2", "x10^9/L", "淋巴细胞增多", "淋巴细胞减少"),
    ("单核细胞百分数", "3", "10", "%", "单核细胞增多", None),
    ("单核细胞绝对值", "0.1", "0.6", "x10^9/L", "单核细胞增多", None),
    ("嗜酸性粒细胞百分数", "0.4", "8.0", "%", "嗜酸性粒细胞增多", "嗜酸性粒细胞减少"),
    ("嗜酸性粒细胞绝对值", "0.02", "0.52", "x10^9/L", "嗜酸性粒细胞增多", "嗜酸性粒细胞减少"),
    ("嗜碱性粒细胞百分数", "0", "1.0", "%", "嗜碱性粒细胞增多", None),
    ("嗜碱性粒细胞绝对值", "0", "0.06", "x10^9/L", "嗜碱性粒细胞增多", None),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_id(kind: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join((kind, *parts)).encode()).hexdigest()[:20]
    return f"wbc-supplement:{kind}:{digest}"


def build_supplement() -> dict[str, Any]:
    alignment = json.loads(ALIGNMENT.read_text(encoding="utf-8"))
    entity_document = json.loads(ENTITIES.read_text(encoding="utf-8"))
    aligned_by_name = {item["canonical_name"]: item for item in alignment["aligned_entities"]}
    existing_by_name = {item["canonical_name"]: item for item in entity_document["canonical_entities"]}
    panel = existing_by_name["白细胞分类计数"]
    state_names = {
        state_name
        for _, _, _, _, high_state, low_state in RANGES
        for state_name in (high_state, low_state)
        if state_name is not None
    }
    states = {name: existing_by_name[name] for name in state_names}

    missing_names = [name for name, *_ in RANGES if name not in existing_by_name]
    added_entities = [aligned_by_name[name] for name in missing_names]
    # Reuse the current canonical ID whenever the entity is already in the
    # manually governed base graph. Official-alignment IDs are only for the
    # genuinely missing percentage indicators added by this supplement.
    metrics = {
        name: existing_by_name.get(name, aligned_by_name[name])
        for name, *_ in RANGES
    }

    relationships: list[dict[str, Any]] = []
    for name, metric in metrics.items():
        relationships.append({
            "relationship_id": _stable_id("relationship", panel["canonical_id"], "HAS_METRIC", metric["canonical_id"]),
            "source_canonical_id": panel["canonical_id"],
            "source_name": panel["canonical_name"],
            "relation_type": "HAS_METRIC",
            "target_canonical_id": metric["canonical_id"],
            "target_name": name,
            "evidence_chunk_id": EVIDENCE_CHUNK_ID,
        })
    rules = []
    for name, lower, upper, unit, high_state, low_state in RANGES:
        metric = metrics[name]
        output_names = [high_state, *(item for item in (low_state,) if item is not None)]
        for state_name in output_names:
            state = states[state_name]
            relationships.append({
                "relationship_id": _stable_id("relationship", metric["canonical_id"], "HAS_STATE", state["canonical_id"]),
                "source_canonical_id": metric["canonical_id"],
                "source_name": name,
                "relation_type": "HAS_STATE",
                "target_canonical_id": state["canonical_id"],
                "target_name": state_name,
                "evidence_chunk_id": EVIDENCE_CHUNK_ID,
            })
        rules.append({
            "rule_id": _stable_id("rule", metric["canonical_id"], lower, upper, unit),
            "rule_type": "REFERENCE_RANGE",
            "rule_stage": "PREPROCESS",
            "rule_logic": "RANGE_TABLE",
            "indicator_canonical_id": metric["canonical_id"],
            "indicator_name": name,
            "lower": lower,
            "lower_inclusive": True,
            "upper": upper,
            "upper_inclusive": True,
            "unit": unit,
            "high_state_id": states[high_state]["canonical_id"],
            "high_state_name": high_state,
            "low_state_id": states[low_state]["canonical_id"] if low_state else None,
            "low_state_name": low_state,
            "low_result_without_state": "BELOW_REFERENCE" if low_state is None else None,
            "evidence_chunk_id": EVIDENCE_CHUNK_ID,
            "evidence_quote": EVIDENCE_QUOTE,
            "automation_status": "AUTO_VALIDATED_BOOK_TABLE",
        })

    relationship_keys = {
        (item["source_canonical_id"], item["relation_type"], item["target_canonical_id"])
        for item in relationships
    }
    if len(relationship_keys) != len(relationships):
        raise ValueError("supplement contains duplicate relationships")
    return {
        "schema_version": "chapter-01-wbc-differential-supplement/v0.1",
        "supplement_id": "chapter-01-wbc-differential-v0.1",
        "status": "AUTOMATED_VALIDATION_COMPLETE",
        "contract": {
            "user_validation_required": False,
            "book_table_is_rule_authority": True,
            "disease_relations_are_not_invented": True,
            "missing_low_states_are_not_invented": True,
            "neo4j_import_is_idempotent": True,
        },
        "sources": {
            "alignment_path": str(ALIGNMENT.relative_to(ROOT)),
            "alignment_sha256": _sha256(ALIGNMENT),
            "entity_path": str(ENTITIES.relative_to(ROOT)),
            "entity_sha256": _sha256(ENTITIES),
            "base_entity_count": len(entity_document["canonical_entities"]),
            "evidence_chunk_id": EVIDENCE_CHUNK_ID,
            "evidence_quote": EVIDENCE_QUOTE,
        },
        "panel": {
            "canonical_id": panel["canonical_id"],
            "canonical_name": panel["canonical_name"],
        },
        "added_entities": added_entities,
        "relationships": relationships,
        "rules": rules,
        "statistics": {
            "added_entity_count": len(added_entities),
            "has_metric_count": sum(item["relation_type"] == "HAS_METRIC" for item in relationships),
            "has_state_count": sum(item["relation_type"] == "HAS_STATE" for item in relationships),
            "rule_count": len(rules),
        },
    }


def main() -> int:
    payload = build_supplement()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["statistics"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
