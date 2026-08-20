"""从 v1.0 关系标注稿生成可逐字回放证据的第一章关系参考集。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from medical_kg_sourceprep.extraction.graph_builder.contract import DEFAULT_CHUNK_MANIFEST
from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk, load_chunk_manifest


ROOT = Path(__file__).resolve().parents[2]
SOURCE_GOLD_PATH = ROOT / "evaluation/chapter-01/chapter-01-relationship-gold-v1.0.json"
ENTITY_PATH = ROOT / "evaluation/chapter-01/chapter-01-canonical-entities-v0.8.json"
OUTPUT_PATH = ROOT / "evaluation/chapter-01/chapter-01-relationship-gold-v1.1.json"
_ASCII_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matches_at(text: str, start: int, candidate: str) -> bool:
    """避免把 MCH 误定位到 MCHC 这类英文缩写的内部。"""
    if not _ASCII_TOKEN.fullmatch(candidate):
        return True
    before = text[start - 1] if start else ""
    end = start + len(candidate)
    after = text[end] if end < len(text) else ""
    return not (before.isalnum() or after.isalnum())


def _find_span(chunk: EvidenceChunk, names: Iterable[str]) -> dict[str, Any] | None:
    """在指定证据块中定位实体的最长逐字名称，返回可回放坐标。"""
    choices = sorted({name for name in names if name}, key=len, reverse=True)
    for name in choices:
        start = chunk.text.find(name)
        while start >= 0:
            if _matches_at(chunk.text, start, name):
                end = start + len(name)
                return {
                    "chunk_id": chunk.chunk_id,
                    "start": start,
                    "end": end,
                    "exact_quote": chunk.text[start:end],
                }
            start = chunk.text.find(name, start + 1)
    return None


def build_gold() -> dict[str, Any]:
    """只保留端点可在声明证据块内逐字定位的关系参考项。"""
    source = json.loads(SOURCE_GOLD_PATH.read_text(encoding="utf-8"))
    catalog = json.loads(ENTITY_PATH.read_text(encoding="utf-8"))
    _manifest, chunks = load_chunk_manifest(DEFAULT_CHUNK_MANIFEST)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    entity_by_id = {
        entity["canonical_id"]: entity for entity in catalog["canonical_entities"]
    }

    cases: list[dict[str, Any]] = []
    excluded = Counter()
    retained = Counter()
    for source_case in source["cases"]:
        relationships: list[dict[str, Any]] = []
        for relationship in source_case["relationships"]:
            evidence_ids = relationship.get("evidence_chunk_ids", [])
            if not evidence_ids:
                excluded["NO_DECLARED_EVIDENCE"] += 1
                continue
            source_entity = entity_by_id[relationship["source_canonical_id"]]
            target_entity = entity_by_id[relationship["target_canonical_id"]]
            source_names = (source_entity["canonical_name"], *source_entity.get("aliases", []))
            target_names = (target_entity["canonical_name"], *target_entity.get("aliases", []))
            spans: list[dict[str, Any]] | None = None
            for evidence_id in evidence_ids:
                chunk = chunk_by_id.get(evidence_id)
                if chunk is None:
                    continue
                source_span = _find_span(chunk, source_names)
                target_span = _find_span(chunk, target_names)
                if source_span is not None and target_span is not None:
                    spans = [source_span, target_span]
                    break
            if spans is None:
                excluded["ENDPOINT_NOT_BOTH_ANCHORED"] += 1
                continue
            relationships.append({
                **relationship,
                "evidence_spans": spans,
                "review_status": "EVIDENCE_ANCHORED_REFERENCE_CANDIDATE",
                "provenance": "chapter-01-relationship-gold-v1.0.json",
            })
            retained[relationship["relation_type"]] += 1
        cases.append({
            **{key: value for key, value in source_case.items() if key != "relationships"},
            "relationships": relationships,
        })

    relationship_count = sum(len(case["relationships"]) for case in cases)
    return {
        "schema_version": "medical-kg-chapter-relationship-gold/v1.1",
        "status": "EVIDENCE_ANCHORED_REFERENCE_CANDIDATE",
        "gold_provenance": {
            "method": "V10_FILTERED_BY_DECLARED_EVIDENCE_AND_VERBATIM_ENDPOINT_ANCHORS",
            "source_gold": str(SOURCE_GOLD_PATH.relative_to(ROOT)),
            "source_gold_sha256": _sha256(SOURCE_GOLD_PATH),
            "canonical_entity_catalog": str(ENTITY_PATH.relative_to(ROOT)),
            "chunk_manifest": str(DEFAULT_CHUNK_MANIFEST.relative_to(ROOT)),
            "human_approved": False,
            "approval_note": (
                "每条保留关系均有声明证据块和源、目标端点的逐字坐标；"
                "关系语义仍需人工批准，不能作为正式发布金标。"
            ),
        },
        "scope_contract": {
            **source["scope_contract"],
            "evidence_requirement": "每条关系必须含同一 EvidenceChunk 内的源、目标逐字锚点",
            "rules_excluded": True,
        },
        "source_canonical_entities": source["source_canonical_entities"],
        "source_manual_graph": source["source_manual_graph"],
        "cases": cases,
        "statistics": {
            "case_count": len(cases),
            "chunk_count": len({cid for case in cases for cid in case["chunk_ids"]}),
            "relationship_count": relationship_count,
            "retained_by_relation_type": dict(sorted(retained.items())),
            "excluded_count": sum(excluded.values()),
            "excluded_by_reason": dict(sorted(excluded.items())),
        },
    }


if __name__ == "__main__":
    payload = build_gold()
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["statistics"], ensure_ascii=False, sort_keys=True))
