"""从 v0.3 canonical 金标和证据映射生成分层评测集 v0.4。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = ROOT / "evaluation/chapter-01/chapter-01-graph-test-set-v0.3.json"
MAPPING_PATH = ROOT / "runtime/services/inception/gold-evidence-mapping.json"
OFFSET_PATH = ROOT / "runtime/services/inception/documents/offset-manifest.json"
OUTPUT_PATH = ROOT / "evaluation/chapter-01/chapter-01-layered-test-set-v0.4.json"


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _canonical_id(entity_type: str, label: str) -> str:
    return _stable_id("entity", entity_type, label)


if __name__ == "__main__":
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    offsets = json.loads(OFFSET_PATH.read_text(encoding="utf-8"))
    document_chunks = {
        document["case_id"]: document["chunks"] for document in offsets["documents"]
    }
    mappings_by_gold_case: dict[str, list[dict[str, Any]]] = {}
    for item in mapping["mappings"]:
        mappings_by_gold_case.setdefault(item["gold_case_id"], []).append(item)

    cases: list[dict[str, Any]] = []
    for gold_case in graph["cases"]:
        case_id = gold_case["case_id"]
        evidence_units: list[dict[str, Any]] = []
        canonical_entities = [
            {
                "canonical_id": _canonical_id(entity_type, label),
                "entity_type": entity_type,
                "canonical_label": label,
            }
            for entity_type, label in gold_case["entities"]
        ]
        links: list[dict[str, Any]] = []
        unit_by_identity: dict[tuple[Any, ...], str] = {}

        same_span_counts: dict[tuple[str, int, int], int] = {}
        for item in mappings_by_gold_case[case_id]:
            identity = (item["document_id"], item["begin"], item["end"])
            same_span_counts[identity] = same_span_counts.get(identity, 0) + 1

        for item in mappings_by_gold_case[case_id]:
            document_id = item["document_id"]
            begin = int(item["begin"])
            end = int(item["end"])
            quote = str(item["exact_quote"])
            chunk = next(
                (
                    value for value in document_chunks[document_id]
                    if int(value["document_start"]) <= begin < end <= int(value["document_end"])
                ),
                None,
            )
            if chunk is None:
                raise ValueError(f"证据范围没有落入单个 chunk: {case_id}/{item['canonical_label']}")
            chunk_start = begin - int(chunk["document_start"])
            chunk_end = end - int(chunk["document_start"])

            repeated_span = same_span_counts[(document_id, begin, end)] > 1
            method = str(item["mapping_method"])
            if repeated_span:
                derivation = "COORDINATION_DERIVED"
            elif method == "STRUCTURED_CONTEXT":
                derivation = "TABLE_DERIVED" if "<tr>" in quote else "STRUCTURE_DERIVED"
            elif method == "HEURISTIC_CONTEXT":
                derivation = "CONTEXT_DERIVED"
            elif method == "NORMALIZED_SOURCE_SPAN":
                derivation = "NORMALIZED_MENTION"
            else:
                derivation = "DIRECT_MENTION"

            # canonical 名称在证据中逐字出现时，收缩为严格的最小 mention。
            label = str(item["canonical_label"])
            label_offset = quote.find(label)
            mention_eligible = False
            unit_kind = "EVIDENCE_CONTEXT"
            if label_offset >= 0:
                chunk_start += label_offset
                chunk_end = chunk_start + len(label)
                quote = label
                mention_eligible = True
                unit_kind = "MENTION"
            elif method == "NORMALIZED_SOURCE_SPAN" and len(quote) <= 32 and "<" not in quote:
                mention_eligible = not repeated_span
                unit_kind = "MENTION" if mention_eligible else unit_kind

            unit_identity = (chunk["chunk_id"], chunk_start, chunk_end, unit_kind)
            unit_id = unit_by_identity.get(unit_identity)
            if unit_id is None:
                unit_id = _stable_id(
                    "evidence", str(chunk["chunk_id"]), str(chunk_start), str(chunk_end), unit_kind
                )
                unit_by_identity[unit_identity] = unit_id
                evidence_units.append(
                    {
                        "evidence_unit_id": unit_id,
                        "kind": unit_kind,
                        "chunk_id": chunk["chunk_id"],
                        "start": chunk_start,
                        "end": chunk_end,
                        "exact_quote": quote,
                        "mention_eligible": mention_eligible,
                        "gold_status": "GENERATED_GOLD",
                    }
                )
            links.append(
                {
                    "evidence_unit_id": unit_id,
                    "canonical_id": _canonical_id(str(item["entity_type"]), label),
                    "derivation": derivation,
                    "mapping_method": method,
                    "gold_status": "GENERATED_GOLD",
                }
            )

        label_to_id = {
            item["canonical_label"]: item["canonical_id"] for item in canonical_entities
        }
        relationships = [
            {
                "source_canonical_id": label_to_id[source],
                "relation_type": relation_type,
                "target_canonical_id": label_to_id[target],
            }
            for source, relation_type, target in gold_case["relationships"]
        ]
        cases.append(
            {
                "case_id": case_id,
                "title": gold_case["title"],
                "chunk_ids": gold_case["chunk_ids"],
                "evidence_units": evidence_units,
                "canonical_entities": canonical_entities,
                "mention_to_canonical_links": links,
                "relationships": relationships,
            }
        )

    payload = {
        "schema_version": "medical-kg-layered-gold/v0.4",
        "status": "GENERATED_GOLD",
        "gold_provenance": {
            "method": "V03_CANONICAL_PLUS_SOURCE_SPAN_MAPPING",
            "human_approved": False,
            "scoring_eligible": True,
        },
        "source_graph_gold": GRAPH_PATH.name,
        "cases": cases,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "cases": len(cases),
        "mentions": sum(
            unit["mention_eligible"] for case in cases for unit in case["evidence_units"]
        ),
        "evidence_units": sum(len(case["evidence_units"]) for case in cases),
        "canonical_entities": sum(len(case["canonical_entities"]) for case in cases),
        "links": sum(len(case["mention_to_canonical_links"]) for case in cases),
        "relationships": sum(len(case["relationships"]) for case in cases),
    }, ensure_ascii=False, sort_keys=True))
