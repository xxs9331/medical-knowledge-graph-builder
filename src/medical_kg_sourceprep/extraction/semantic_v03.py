"""Closed-world relation/rule supplementation for chapter semantic extraction v0.3.

This module never creates model entities.  The catalog is the only endpoint
authority and every accepted item is replayed against the frozen chunks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .llm_extraction import EvidenceChunk, atomic_write_json
from ..graph.semantic_graph import ENTITY_TYPES, SEMANTIC_RELATIONS, SEMANTIC_TYPES, SUBJECT_LOGICS

CONTRACT_VERSION = "semantic-candidates/v0.3"
RELATION_PROMPT_VERSION = "semantic-candidates-relation-prompt/v0.3.1"
RULE_PROMPT_VERSION = "semantic-candidates-rule-prompt/v0.3.2"
VALIDATOR_VERSION = "semantic-candidates-validator/v0.3.1"
MAX_RELATIONS_PER_PAGE = 48
MODEL_RELATIONS = {key: value for key, value in SEMANTIC_RELATIONS.items() if not key.endswith("SUPPORTED_BY")}
TRIGGERS = {
    "DEFINES_AS": ("是", "指", "定义", "表示", "即"),
    "POSSIBLY_CAUSED_BY": ("可能由", "原因", "导致", "见于"),
    "SEEN_IN": ("见于", "常见于", "发生于"),
    "LEADS_TO": ("导致", "引起", "可致"),
    "RECOVERY_FACTOR": ("恢复", "改善", "纠正"),
    "CLASSIFIES_AS": ("属于", "分为", "分类为"),
}


class V03ContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    page_id: str
    candidate_key: str
    candidate_id: str
    entity_type: str
    text: str
    source: Mapping[str, Any]
    origin: str = "model"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_id(kind: str, *values: object) -> str:
    raw = json.dumps([CONTRACT_VERSION, kind, *values], ensure_ascii=False, sort_keys=True)
    return f"{kind}:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def _reject(kind: str, index: int, item: Any, reason: str, chunks: Mapping[str, EvidenceChunk]) -> dict[str, Any]:
    ref = item.get("source_ref", {}) if isinstance(item, Mapping) else {}
    chunk = chunks.get(ref.get("chunk_id")) if isinstance(ref, Mapping) else None
    return {"candidate_id": _stable_id("rejected", kind, index, item), "candidate_type": kind,
            "page_id": chunk.page_id if chunk else "", "chunk_id": chunk.chunk_id if chunk else "",
            "reason_code": reason, "candidate_summary": {key: str(item[key])[:120] for key in ("candidate_key", "relation", "rule_key") if isinstance(item, Mapping) and key in item},
            "raw_candidate": item}


def replay_ref(value: Any, chunks: Mapping[str, EvidenceChunk]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise V03ContractError("source_ref is required")
    chunk_id, digest, quote = value.get("chunk_id"), value.get("chunk_sha256"), value.get("exact_quote")
    chunk = chunks.get(chunk_id)
    if not chunk or digest != chunk.chunk_sha256 or not isinstance(quote, str) or not quote:
        raise V03ContractError("chunk hash drift")
    start = chunk.text.find(quote)
    if start < 0 or chunk.text.count(quote) != 1:
        raise V03ContractError("source quote is absent or ambiguous")
    return {"chunk_id": chunk_id, "chunk_sha256": digest, "exact_quote": quote,
            "char_start": start, "char_end": start + len(quote)}


def _component(value: Any, chunks: Mapping[str, EvidenceChunk]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get("text"), str) or not value["text"]:
        raise V03ContractError("rule component is missing")
    source = replay_ref(value.get("source_ref"), chunks)
    chunk = chunks[source["chunk_id"]]
    start = chunk.text.find(value["text"], source["char_start"], source["char_end"])
    if start < 0 or chunk.text.count(value["text"], source["char_start"], source["char_end"]) != 1:
        raise V03ContractError("rule component is not verbatim")
    source["char_start"], source["char_end"] = start, start + len(value["text"])
    return {"text": value["text"], "source": source}


def build_entity_catalog(v02_extraction: Mapping[str, Any], chunks: Sequence[EvidenceChunk],
                        deterministic: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Merge v0.2 entities and local deterministic anchors, never model-new entities."""
    by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    entries: dict[tuple[str, str], CatalogEntry] = {}
    for item in [*v02_extraction.get("candidates", []), *deterministic]:
        if not isinstance(item, Mapping) or item.get("candidate_type") == "relation":
            continue
        entity_type = item.get("entity_type")
        source = item.get("text_span") or item.get("source")
        if entity_type not in ENTITY_TYPES - {"SourceLocator", "InterpretationRule"} or not isinstance(source, Mapping):
            continue
        chunk = by_chunk.get(source.get("chunk_id"))
        if not chunk or not isinstance(item.get("text"), str):
            continue
        page_id = chunk.page_id
        key = str(item.get("candidate_key") or _stable_id("deterministic-key", entity_type, source.get("chunk_id"), source.get("char_start"), source.get("char_end")))
        entry = CatalogEntry(page_id, key, str(item.get("candidate_id") or _stable_id("entity", page_id, key)), entity_type,
                             item["text"], {"chunk_id": chunk.chunk_id, "chunk_sha256": chunk.chunk_sha256,
                             "exact_quote": source.get("exact_quote", item["text"]), "char_start": source.get("char_start"),
                             "char_end": source.get("char_end")}, item.get("origin", "model"))
        entries.setdefault((page_id, key), entry)
    values = [entry.to_dict() for entry in sorted(entries.values(), key=lambda item: (item.page_id, item.candidate_key, item.candidate_id))]
    return {"schema_version": "entity-catalog/v0.3", "status": "frozen", "approved": 0,
            "entries": values, "catalog_sha256": hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}


def validate_relations(payload: Mapping[str, Any], chunks: Sequence[EvidenceChunk], catalog: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != {"relations"} or not isinstance(payload["relations"], list):
        raise V03ContractError("relation-only output must contain relations only")
    by_chunk = {item.chunk_id: item for item in chunks}
    entries = {(item["page_id"], item["candidate_key"]): item for item in catalog.get("entries", [])}
    accepted, rejected = [], []
    for index, item in enumerate(payload["relations"]):
        try:
            if not isinstance(item, Mapping) or item.get("relation") not in MODEL_RELATIONS:
                raise V03ContractError("relation is not model-eligible")
            chunk = by_chunk.get(item.get("source_chunk_id"))
            if not chunk:
                raise V03ContractError("source chunk is unavailable")
            source = entries.get((chunk.page_id, item.get("source_candidate_key")))
            target = entries.get((chunk.page_id, item.get("target_candidate_key")))
            if not source or not target:
                raise V03ContractError("relation endpoint candidate_key is unknown")
            allowed = MODEL_RELATIONS[item["relation"]]
            targets = allowed[1] if isinstance(allowed[1], tuple) else (allowed[1],)
            if source["entity_type"] != allowed[0] or target["entity_type"] not in targets:
                raise V03ContractError("relation direction or endpoint type is invalid")
            source_ref = {"chunk_id": item.get("source_chunk_id"), "chunk_sha256": item.get("source_chunk_sha256"), "exact_quote": item.get("source_quote")}
            evidence = replay_ref(source_ref, by_chunk)
            cue = item.get("relation_cue")
            if not isinstance(cue, str) or not cue or cue not in evidence["exact_quote"]:
                raise V03ContractError("relation cue is missing from source quote")
            if source["text"] not in evidence["exact_quote"] or target["text"] not in evidence["exact_quote"]:
                raise V03ContractError("relation quote lacks both endpoint texts")
            accepted.append({"candidate_id": _stable_id("relation", chunk.page_id, item["source_candidate_key"], item["relation"], item["target_candidate_key"]),
                "candidate_type": "relation", "origin": "model", "page_id": chunk.page_id,
                "source_candidate_key": source["candidate_key"], "target_candidate_key": target["candidate_key"],
                "relation": item["relation"], "evidence_role": "relation", "source": evidence, "relation_cue": cue,
                "status": "candidate"})
        except V03ContractError as exc:
            rejected.append(_reject("relation", index, item, str(exc), by_chunk))
    if len(accepted) > MAX_RELATIONS_PER_PAGE:
        raise V03ContractError("relation limit exceeded")
    return {"schema_version": "relation-only/v0.3", "status": "candidate-only", "approved": 0,
            "candidates": accepted, "rejections": rejected,
            "counts": {"accepted": len(accepted), "rejected": len(rejected)}}


def validate_rules(payload: Mapping[str, Any], chunks: Sequence[EvidenceChunk], catalog: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != {"rules"} or not isinstance(payload["rules"], list):
        raise V03ContractError("rule-only output must contain rules only")
    by_chunk = {item.chunk_id: item for item in chunks}
    entries = {(item["page_id"], item["candidate_key"]): item for item in catalog.get("entries", [])}
    accepted, rejected = [], []
    for index, item in enumerate(payload["rules"]):
        try:
            if not isinstance(item, Mapping) or item.get("entity_type") != "InterpretationRule":
                raise V03ContractError("legacy rule shape is rejected")
            source = replay_ref(item.get("source_ref"), by_chunk)
            page_id = by_chunk[source["chunk_id"]].page_id
            subject_logic = item.get("subject_logic")
            if item.get("semantic_type") not in SEMANTIC_TYPES or subject_logic not in SUBJECT_LOGICS:
                raise V03ContractError("invalid rule semantic contract")
            subjects = item.get("subject_candidate_keys")
            conclusion = item.get("conclusion_candidate_key")
            if not isinstance(subjects, list) or not subjects or len(set(subjects)) != len(subjects):
                raise V03ContractError("rule subjects are missing or duplicated")
            conclusion_entry = entries.get((page_id, conclusion)) if isinstance(conclusion, str) else None
            if not conclusion_entry or conclusion_entry["entity_type"] != "MedicalConcept":
                raise V03ContractError("rule conclusion endpoint is missing")
            for key in subjects:
                subject_entry = entries.get((page_id, key)) if isinstance(key, str) else None
                if not subject_entry or subject_entry["entity_type"] not in {"TestItem", "MedicalConcept"}:
                    raise V03ContractError("rule subject endpoint is unknown")
            components = item.get("components")
            if not isinstance(components, Mapping) or not isinstance(components.get("conditions"), list) or not components["conditions"]:
                raise V03ContractError("rule conditions are required")
            result_components = {"conditions": [_component(value, by_chunk) for value in components["conditions"]],
                                 "conclusion": _component(components.get("conclusion"), by_chunk)}
            if subject_logic in {"ALL", "ANY"}:
                result_components["connector"] = _component(components.get("connector"), by_chunk)
            elif components.get("connector") is not None:
                raise V03ContractError("SINGLE rule cannot have connector")
            evidence_text = " ".join([v["text"] for v in result_components["conditions"]] + [result_components["conclusion"]["text"]])
            if not any(trigger in evidence_text for trigger in TRIGGERS[item["semantic_type"]]):
                raise V03ContractError("semantic_type lacks its verbatim trigger")
            accepted.append({"candidate_id": _stable_id("rule", page_id, item.get("rule_key"), source["chunk_id"], source["char_start"]),
                "candidate_type": "rule", "entity_type": "InterpretationRule", "origin": "model", "page_id": page_id,
                "rule_key": item.get("rule_key") or _stable_id("rule-key", page_id, source["char_start"]),
                "semantic_type": item["semantic_type"], "subject_logic": subject_logic,
                "subject_candidate_keys": subjects, "conclusion_candidate_key": conclusion,
                "components": result_components, "source": source, "status": "candidate"})
        except (V03ContractError, KeyError) as exc:
            rejected.append(_reject("rule", index, item, str(exc), by_chunk))
    return {"schema_version": "rule-only/v0.3", "status": "candidate-only", "approved": 0,
            "candidates": accepted, "rejections": rejected,
            "counts": {"accepted": len(accepted), "rejected": len(rejected)}}


def stable_relations(relation_packages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for package in relation_packages:
        for item in package.get("candidates", []):
            key = (item["page_id"], item["source_candidate_key"], item["relation"], item["target_candidate_key"])
            existing = merged.setdefault(key, dict(item, evidence=[]))
            evidence = dict(item["source"], evidence_role=item.get("evidence_role", "relation"), relation_cue=item["relation_cue"], origin=item.get("origin", "model"))
            if evidence not in existing["evidence"]:
                existing["evidence"].append(evidence)
            if item.get("origin") == "model":
                existing["origin"] = "model"
    return [merged[key] for key in sorted(merged)]


def recover_derived_relations(catalog: Mapping[str, Any], chunks: Sequence[EvidenceChunk]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover only explicit same-page heading/range structure.

    The routine intentionally does not infer medical meaning.  A range is
    linked to an item only when both anchors occur in the same page and the
    range is under an explicit reference-range label.  Population recovery is
    limited to a literal table row containing both anchors.
    """
    by_page: dict[str, list[dict[str, Any]]] = {}
    for entry in catalog.get("entries", []):
        by_page.setdefault(entry["page_id"], []).append(entry)
    chunks_by_page: dict[str, list[EvidenceChunk]] = {}
    for chunk in chunks:
        chunks_by_page.setdefault(chunk.page_id, []).append(chunk)
    accepted, review = [], []
    for page_id, entries in by_page.items():
        items = [entry for entry in entries if entry["entity_type"] == "TestItem"]
        ranges = [entry for entry in entries if entry["entity_type"] == "ReferenceRange"]
        populations = [entry for entry in entries if entry["entity_type"] == "Population"]
        text = "\n".join(chunk.text for chunk in chunks_by_page.get(page_id, []))
        for item in items:
            for value in ranges:
                if item["text"] not in text or value["text"] not in text:
                    continue
                range_pos = text.find(value["text"])
                label_pos = text.rfind("参考区间", 0, range_pos)
                if label_pos < 0:
                    continue
                value_chunk = next((chunk for chunk in chunks_by_page[page_id] if value["text"] in chunk.text), None)
                if value_chunk is None:
                    review.append({"candidate_id": _stable_id("derived-review", page_id, item["candidate_key"], value["candidate_key"]),
                                   "candidate_type": "relation", "page_id": page_id,
                                   "reason_code": "structure-anchor-chunk-unavailable"})
                    continue
                local_label = value_chunk.text.rfind("参考区间", 0, value_chunk.text.find(value["text"]))
                if local_label < 0:
                    review.append({"candidate_id": _stable_id("derived-review", page_id, item["candidate_key"], value["candidate_key"]),
                                   "candidate_type": "relation", "page_id": page_id,
                                   "reason_code": "structure-label-crosses-chunk"})
                    continue
                quote = value_chunk.text[local_label:value_chunk.text.find(value["text"]) + len(value["text"])]
                source = dict(value["source"], chunk_id=value_chunk.chunk_id, chunk_sha256=value_chunk.chunk_sha256,
                              exact_quote=quote, char_start=local_label, char_end=local_label + len(quote))
                evidence = {"evidence_role": "structure", "source_chunk_id": value_chunk.chunk_id,
                            "source_chunk_sha256": value_chunk.chunk_sha256, "source_quote": quote,
                            "relation_cue": "参考区间"}
                accepted.append({"candidate_id": _stable_id("derived-relation", page_id, item["candidate_key"], value["candidate_key"]),
                    "candidate_type": "relation", "origin": "derived", "page_id": page_id,
                    "source_candidate_key": item["candidate_key"], "target_candidate_key": value["candidate_key"],
                    "relation": "ITEM_HAS_REFERENCE_RANGE", "source": source, "relation_cue": "参考区间",
                    "evidence_role": "structure", "structure_basis": "same-page-reference-range-heading", "status": "candidate"})
                for population in populations:
                    if population["text"] in quote:
                        accepted.append({"candidate_id": _stable_id("derived-relation", page_id, value["candidate_key"], population["candidate_key"]),
                            "candidate_type": "relation", "origin": "derived", "page_id": page_id,
                            "source_candidate_key": value["candidate_key"], "target_candidate_key": population["candidate_key"],
                            "relation": "RANGE_APPLIES_TO_POPULATION", "source": source, "relation_cue": population["text"],
                            "evidence_role": "table-row", "structure_basis": "same-range-anchor-row", "status": "candidate"})
    return accepted, review


def build_relation_prompt(page_id: str, chunks: Sequence[EvidenceChunk], catalog_entries: Sequence[Mapping[str, Any]]) -> str:
    return (f"PROMPT_VERSION={RELATION_PROMPT_VERSION}\nReturn JSON only: {{relations:[...]}}. Page={page_id}. "
            "Copy endpoints only from ENTITY_CATALOG; do not create entities or use cross-page keys. "
            "Each relation must use one chunk, and source_quote must contain both endpoint texts and relation_cue verbatim. "
            "Allowed relations=" + json.dumps(sorted(MODEL_RELATIONS), ensure_ascii=False) + "\nENTITY_CATALOG=" +
            json.dumps(list(catalog_entries), ensure_ascii=False, sort_keys=True) + "\nCHUNKS=" +
            json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, sort_keys=True))


def build_rule_prompt(page_id: str, chunks: Sequence[EvidenceChunk], catalog_entries: Sequence[Mapping[str, Any]]) -> str:
    return (f"PROMPT_VERSION={RULE_PROMPT_VERSION}\nReturn JSON only: {{\"rules\":[...]}}. Page={page_id}. "
            "Each rule must use exactly these fields: rule_key,entity_type,semantic_type,subject_logic,"
            "subject_candidate_keys,conclusion_candidate_key,source_ref,components. "
            "entity_type must be InterpretationRule; semantic_type must be one of " + json.dumps(sorted(SEMANTIC_TYPES), ensure_ascii=False) +
            "; subject_logic must be exactly SINGLE, ALL, or ANY. "
            "Every source_ref is an object with exactly {chunk_id,chunk_sha256,exact_quote}; never a string or chunk id. "
            "Every component is {text,source_ref} and verbatim. components has conditions as a non-empty array and exactly one conclusion object. "
            "ALL/ANY also require connector={text,source_ref}; SINGLE must omit connector. "
            "subject_candidate_keys must copy one or more same-page TestItem/MedicalConcept keys from ENTITY_CATALOG; "
            "conclusion_candidate_key must copy exactly one same-page MedicalConcept key. "
            "Reject legacy rule_id/condition/conclusion output. AT_LEAST is not allowed. ENTITY_CATALOG=" +
            json.dumps(list(catalog_entries), ensure_ascii=False, sort_keys=True) + "\nCHUNKS=" +
            json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, sort_keys=True))


def write_catalog(path: Path, catalog: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(catalog))
