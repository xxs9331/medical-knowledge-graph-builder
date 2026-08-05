"""Evidence-bound chapter relation/rule recovery contract v0.4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping, Sequence

from .llm_extraction import EvidenceChunk
from ..graph.semantic_graph import ENTITY_TYPES, SEMANTIC_RELATIONS, SEMANTIC_TYPES, SUBJECT_LOGICS

CONTRACT_VERSION = "semantic-candidates/v0.4"
CATALOG_VERSION = "entity-catalog/v0.4"
ENDPOINT_PROMPT_VERSION = "semantic-endpoint-prompt/v0.4.1"
RELATION_PROMPT_VERSION = "semantic-relation-prompt/v0.4.1"
RULE_PROMPT_VERSION = "semantic-rule-prompt/v0.4.1"
ENDPOINT_VALIDATOR_VERSION = "semantic-validator/v0.4.1"
RELATION_VALIDATOR_VERSION = "semantic-validator/v0.4.1"
RULE_VALIDATOR_VERSION = "semantic-rule-validator/v0.4.2"
ENDPOINT_TYPES = frozenset({"TestItem", "MedicalConcept", "Population", "TestMethod"})
MODEL_RELATIONS = {key: value for key, value in SEMANTIC_RELATIONS.items()
                   if not key.endswith("SUPPORTED_BY") and value[0] != "InterpretationRule"}
_NUMBERED = re.compile(r"^\s*(?:[0-9]+[.、]|[一二三四五六七八九十百千]+、)\s*(\S.*?)\s*$")
_ITEM_LABEL = "【检验项目】"
_REFERENCE_LABEL = "【参考区间】"
_ABNORMAL_LABEL = "【异常结果解读】"
_TRIGGERS = {
    "DEFINES_AS": ("是", "指", "定义", "表示", "即"),
    "POSSIBLY_CAUSED_BY": ("可能由", "原因", "见于", "相关"),
    "SEEN_IN": ("见于", "常见于", "发生于"),
    "LEADS_TO": ("导致", "引起", "可致"),
    "RECOVERY_FACTOR": ("恢复", "改善", "纠正"),
    "CLASSIFIES_AS": ("属于", "分为", "分类", "型"),
    "DIAGNOSTIC_CRITERION": ("诊断", "符合", "为", "<", ">"),
    "REFERENCE_INTERPRETATION": ("参考区间", "参考范围", "正常"),
    "ABNORMAL_RESULT_INTERPRETATION": ("升高", "增高", "降低", "减低", "阳性", "阴性", "异常"),
    "MONITORING_GUIDANCE": ("监测", "观察", "随访"),
    "DIFFERENTIAL_DIAGNOSIS": ("鉴别", "排除"),
    "RISK_ASSOCIATION": ("风险", "相关", "可能"),
    "PROGNOSTIC_INDICATOR": ("预后", "预测", "指标"),
    "INTERPRETATION_CAVEAT": ("注意", "影响", "除外", "不能"),
}


class V04ContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    page_id: str
    candidate_key: str
    candidate_id: str
    entity_type: str
    text: str
    source: Mapping[str, Any]
    origin: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(kind: str, *values: object) -> str:
    return f"{kind}:{hashlib.sha256(_canonical([CONTRACT_VERSION, kind, *values]).encode()).hexdigest()[:24]}"


def _catalog_hash(entries: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical(list(entries)).encode()).hexdigest()


def replay_ref(value: Any, chunks: Mapping[str, EvidenceChunk]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise V04ContractError("source_ref is required")
    chunk_id, digest, quote = value.get("chunk_id"), value.get("chunk_sha256"), value.get("exact_quote")
    chunk = chunks.get(chunk_id)
    if chunk is None or digest != chunk.chunk_sha256:
        raise V04ContractError("chunk hash drift")
    if not isinstance(quote, str) or not quote:
        raise V04ContractError("source quote is missing")
    start = chunk.text.find(quote)
    if start < 0 or chunk.text.count(quote) != 1:
        raise V04ContractError("source quote is absent or ambiguous")
    return {"chunk_id": chunk_id, "chunk_sha256": digest, "exact_quote": quote,
            "char_start": start, "char_end": start + len(quote)}


def _text_span(text: Any, source: Mapping[str, Any], chunks: Mapping[str, EvidenceChunk]) -> dict[str, Any]:
    if not isinstance(text, str) or not text:
        raise V04ContractError("candidate text is missing")
    chunk = chunks[source["chunk_id"]]
    start = chunk.text.find(text, source["char_start"], source["char_end"])
    if start < 0 or chunk.text.count(text, source["char_start"], source["char_end"]) != 1:
        raise V04ContractError("candidate text is not a unique verbatim substring")
    return {"chunk_id": source["chunk_id"], "chunk_sha256": source["chunk_sha256"],
            "exact_quote": text, "char_start": start, "char_end": start + len(text)}


def _reject(kind: str, index: int, item: Any, reason: str, chunks: Mapping[str, EvidenceChunk]) -> dict[str, Any]:
    ref = item.get("source_ref") if isinstance(item, Mapping) else None
    chunk = chunks.get(ref.get("chunk_id")) if isinstance(ref, Mapping) else None
    raw = dict(item) if isinstance(item, Mapping) else item
    return {"candidate_id": _stable_id("rejected", kind, index, raw), "candidate_type": kind,
            "page_id": chunk.page_id if chunk else "", "chunk_id": chunk.chunk_id if chunk else "",
            "reason_code": reason, "summary": f"{kind} candidate rejected: {reason}",
            "raw_candidate": raw}


def build_base_catalog(v02_extraction: Mapping[str, Any], v02_database: Path,
                       chunks: Sequence[EvidenceChunk]) -> dict[str, Any]:
    """Merge accepted model entities with deterministic semantic graph records."""
    by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    model_ids = {item.get("candidate_id") for item in v02_extraction.get("candidates", [])
                 if isinstance(item, Mapping) and item.get("candidate_type") != "relation"}
    entries: dict[tuple[str, str], CatalogEntry] = {}
    with sqlite3.connect(v02_database) as db:
        rows = db.execute("""
            SELECT r.record_id, r.candidate_key, r.entity_type, r.text,
                   l.chunk_id, l.char_start, l.char_end, l.exact_quote
            FROM semantic_records r
            JOIN semantic_source_locators l ON l.record_id = r.record_id
            WHERE r.entity_type NOT IN ('SourceLocator', 'InterpretationRule')
            ORDER BY r.record_id
        """).fetchall()
    for record_id, candidate_key, entity_type, text, chunk_id, start, end, quote in rows:
        chunk = by_chunk.get(chunk_id)
        if chunk is None or chunk.text[start:end] != text:
            raise V04ContractError("catalog source locator drift")
        origin = "model" if record_id in model_ids else "derived"
        entry = CatalogEntry(chunk.page_id, candidate_key, record_id, entity_type, text,
                             {"chunk_id": chunk_id, "chunk_sha256": chunk.chunk_sha256,
                              "exact_quote": quote, "char_start": start, "char_end": end}, origin)
        key = (entry.page_id, entry.candidate_key)
        if key in entries and entries[key] != entry:
            raise V04ContractError("catalog candidate_key collision")
        entries[key] = entry
    values = [entry.to_dict() for entry in sorted(entries.values(), key=lambda value: (value.page_id, value.candidate_key))]
    return {"schema_version": CATALOG_VERSION, "status": "frozen", "approved": 0,
            "entries": values, "catalog_sha256": _catalog_hash(values)}


def validate_endpoints(payload: Mapping[str, Any], chunks: Sequence[EvidenceChunk],
                       catalog: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != {"endpoints"} or not isinstance(payload["endpoints"], list):
        raise V04ContractError("endpoint-only output must contain endpoints only")
    by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    existing = {(entry["page_id"], entry["candidate_key"]): entry for entry in catalog.get("entries", [])}
    accepted, rejected = [], []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(payload["endpoints"]):
        try:
            if not isinstance(item, Mapping) or set(item) != {"candidate_key", "entity_type", "text", "source_ref"}:
                raise V04ContractError("endpoint shape is invalid")
            if item.get("entity_type") not in ENDPOINT_TYPES:
                raise V04ContractError("endpoint type is invalid")
            source = replay_ref(item.get("source_ref"), by_chunk)
            text_span = _text_span(item.get("text"), source, by_chunk)
            page_id = by_chunk[source["chunk_id"]].page_id
            candidate_key = item.get("candidate_key")
            if not isinstance(candidate_key, str) or not candidate_key:
                raise V04ContractError("candidate_key is required")
            scoped = (page_id, candidate_key)
            if scoped in seen:
                raise V04ContractError("duplicate endpoint candidate_key")
            if scoped in existing:
                if existing[scoped]["entity_type"] == item["entity_type"] and existing[scoped]["text"] == item["text"]:
                    continue
                raise V04ContractError("endpoint candidate_key collides with frozen catalog")
            seen.add(scoped)
            accepted.append({"candidate_id": _stable_id("endpoint", page_id, candidate_key, text_span),
                "candidate_key": candidate_key, "candidate_type": "entity", "entity_type": item["entity_type"],
                "text": item["text"], "page_id": page_id, "source": source, "text_span": text_span,
                "origin": "endpoint-gap", "status": "candidate"})
        except V04ContractError as exc:
            rejected.append(_reject("endpoint", index, item, str(exc), by_chunk))
    return {"schema_version": "endpoint-only/v0.4", "status": "candidate-only", "approved": 0,
            "candidates": accepted, "rejections": rejected,
            "counts": {"accepted": len(accepted), "rejected": len(rejected)}}


def augment_catalog(catalog: Mapping[str, Any], endpoint_packages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entries = {(item["page_id"], item["candidate_key"]): dict(item) for item in catalog.get("entries", [])}
    for package in endpoint_packages:
        for item in package.get("candidates", []):
            entry = {key: item[key] for key in ("page_id", "candidate_key", "candidate_id", "entity_type", "text", "source", "origin")}
            scoped = (entry["page_id"], entry["candidate_key"])
            if scoped in entries and entries[scoped] != entry:
                raise V04ContractError("augmented catalog collision")
            entries[scoped] = entry
    values = [entries[key] for key in sorted(entries)]
    return {"schema_version": CATALOG_VERSION, "status": "frozen", "approved": 0,
            "entries": values, "catalog_sha256": _catalog_hash(values)}


def _page_layout(chunks: Sequence[EvidenceChunk]) -> tuple[str, dict[str, int]]:
    ordered = sorted(chunks, key=lambda value: (value.start_offset or 0, value.chunk_id))
    offsets, parts, cursor = {}, [], 0
    for chunk in ordered:
        offsets[chunk.chunk_id] = cursor
        parts.append(chunk.text)
        cursor += len(chunk.text) + 1
    return "\n".join(parts), offsets


def recover_structural_relations(catalog: Mapping[str, Any], chunks: Sequence[EvidenceChunk]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover section-bound range edges without same-page endpoint pairing."""
    entries_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chunks_by_page: dict[str, list[EvidenceChunk]] = defaultdict(list)
    for entry in catalog.get("entries", []):
        entries_by_page[entry["page_id"]].append(entry)
    for chunk in chunks:
        chunks_by_page[chunk.page_id].append(chunk)
    accepted, review = [], []
    for page_id, entries in entries_by_page.items():
        page_text, offsets = _page_layout(chunks_by_page[page_id])
        derived_items = [entry for entry in entries if entry["entity_type"] == "TestItem" and entry["origin"] == "derived"]
        derived_ranges = [entry for entry in entries if entry["entity_type"] == "ReferenceRange" and entry["origin"] == "derived"]
        item_positions = sorted((offsets[item["source"]["chunk_id"]] + item["source"]["char_start"], item) for item in derived_items)
        for number, (item_pos, item) in enumerate(item_positions):
            section_end = item_positions[number + 1][0] if number + 1 < len(item_positions) else len(page_text)
            section = page_text[item_pos:section_end]
            label_local = section.find(_REFERENCE_LABEL)
            if label_local < 0:
                continue
            label_pos = item_pos + label_local
            abnormal_local = section.find(_ABNORMAL_LABEL, label_local + len(_REFERENCE_LABEL))
            reference_end = item_pos + abnormal_local if abnormal_local >= 0 else section_end
            for value in derived_ranges:
                value_pos = offsets[value["source"]["chunk_id"]] + value["source"]["char_start"]
                if not label_pos < value_pos < reference_end:
                    continue
                structure = _structure_evidence(chunks_by_page[page_id], value, _REFERENCE_LABEL)
                if structure is None:
                    review.append({"candidate_id": _stable_id("derived-review", page_id, item["candidate_key"], value["candidate_key"]),
                                   "candidate_type": "relation", "page_id": page_id,
                                   "chunk_id": value["source"]["chunk_id"],
                                   "reason_code": "structure anchor is absent or ambiguous",
                                   "summary": "derived range relation lacks a unique structural anchor",
                                   "raw_candidate": {"source_candidate_key": item["candidate_key"],
                                                     "target_candidate_key": value["candidate_key"],
                                                     "relation": "ITEM_HAS_REFERENCE_RANGE"}})
                    continue
                evidence = [_anchor_evidence(item, "source-anchor", chunks_by_page[page_id]),
                            _anchor_evidence(value, "target-anchor", chunks_by_page[page_id]), structure]
                accepted.append(_relation_record(page_id, item, "ITEM_HAS_REFERENCE_RANGE", value,
                                                  "derived", evidence, structure))
        populations = [entry for entry in entries if entry["entity_type"] == "Population"]
        ranges = [entry for entry in entries if entry["entity_type"] == "ReferenceRange"]
        for value in ranges:
            value_chunk = next((chunk for chunk in chunks_by_page[page_id] if chunk.chunk_id == value["source"]["chunk_id"]), None)
            if value_chunk is None:
                continue
            line_start = value_chunk.text.rfind("\n", 0, value["source"]["char_start"]) + 1
            line_end = value_chunk.text.find("\n", value["source"]["char_end"])
            line_end = len(value_chunk.text) if line_end < 0 else line_end
            same_line = [population for population in populations
                         if population["source"]["chunk_id"] == value_chunk.chunk_id
                         and line_start <= population["source"]["char_start"] < value["source"]["char_start"]]
            if not same_line:
                continue
            population = max(same_line, key=lambda item: item["source"]["char_start"])
            line = value_chunk.text[line_start:line_end]
            if value_chunk.text.count(line) != 1:
                continue
            structure = {"evidence_role": "table-row", "chunk_id": value_chunk.chunk_id,
                         "chunk_sha256": value_chunk.chunk_sha256, "exact_quote": line,
                         "char_start": line_start, "char_end": line_end, "relation_cue": population["text"],
                         "origin": "derived"}
            evidence = [_anchor_evidence(value, "source-anchor", chunks_by_page[page_id]),
                        _anchor_evidence(population, "target-anchor", chunks_by_page[page_id]), structure]
            accepted.append(_relation_record(page_id, value, "RANGE_APPLIES_TO_POPULATION", population,
                                              "derived", evidence, structure))
    return accepted, review


def _structure_evidence(chunks: Sequence[EvidenceChunk], target: Mapping[str, Any], cue: str) -> dict[str, Any] | None:
    chunk = next((value for value in chunks if value.chunk_id == target["source"]["chunk_id"]), None)
    if chunk is None:
        return None
    start = chunk.text.rfind(cue, 0, target["source"]["char_start"])
    if start < 0:
        return None
    quote = chunk.text[start:target["source"]["char_end"]]
    if not quote or chunk.text.count(quote) != 1:
        return None
    return {"evidence_role": "structure", "chunk_id": chunk.chunk_id,
            "chunk_sha256": chunk.chunk_sha256, "exact_quote": quote,
            "char_start": start, "char_end": start + len(quote), "relation_cue": cue,
            "origin": "derived"}


def _anchor_evidence(entry: Mapping[str, Any], role: str,
                     chunks: Sequence[EvidenceChunk]) -> dict[str, Any]:
    source = entry["source"]
    chunk = next(value for value in chunks if value.chunk_id == source["chunk_id"])
    start, end = source["char_start"], source["char_end"]
    quote = source["exact_quote"]
    if chunk.text.count(quote) != 1:
        line_start = chunk.text.rfind("\n", 0, start) + 1
        line_end = chunk.text.find("\n", end)
        line_end = len(chunk.text) if line_end < 0 else line_end
        quote = chunk.text[line_start:line_end]
        start, end = line_start, line_end
        if not quote or chunk.text.count(quote) != 1:
            for radius in (24, 48, 96, 192, len(chunk.text)):
                start, end = max(0, source["char_start"] - radius), min(len(chunk.text), source["char_end"] + radius)
                quote = chunk.text[start:end]
                if quote and chunk.text.count(quote) == 1:
                    break
    return {"evidence_role": role, "chunk_id": source["chunk_id"],
            "chunk_sha256": source["chunk_sha256"], "exact_quote": quote,
            "char_start": start, "char_end": end,
            "relation_cue": entry["text"], "origin": "derived"}


def _relation_record(page_id: str, source: Mapping[str, Any], relation: str, target: Mapping[str, Any],
                     origin: str, evidence: Sequence[Mapping[str, Any]], primary: Mapping[str, Any]) -> dict[str, Any]:
    return {"candidate_id": _stable_id("relation", page_id, source["candidate_key"], relation, target["candidate_key"]),
            "candidate_type": "relation", "page_id": page_id,
            "source_candidate_key": source["candidate_key"], "target_candidate_key": target["candidate_key"],
            "relation": relation, "origin": origin, "source": {"chunk_id": primary["chunk_id"],
            "chunk_sha256": primary["chunk_sha256"], "exact_quote": primary["exact_quote"],
            "char_start": primary["char_start"], "char_end": primary["char_end"]},
            "relation_cue": primary["relation_cue"], "evidence": list(evidence), "status": "candidate"}


def baseline_model_relations(v02_extraction: Mapping[str, Any], catalog: Mapping[str, Any],
                             chunks: Sequence[EvidenceChunk]) -> list[dict[str, Any]]:
    by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    entries = {(entry["page_id"], entry["candidate_key"]): entry for entry in catalog.get("entries", [])}
    values = []
    for item in v02_extraction.get("candidates", []):
        if not isinstance(item, Mapping) or item.get("candidate_type") != "relation":
            continue
        anchor = item["source"]
        chunk = by_chunk[anchor["chunk_id"]]
        source = entries[(chunk.page_id, item["source_candidate_key"])]
        target = entries[(chunk.page_id, item["target_candidate_key"])]
        evidence = [{"evidence_role": "relation", "chunk_id": anchor["chunk_id"],
                     "chunk_sha256": anchor["chunk_sha256"], "exact_quote": anchor["exact_quote"],
                     "char_start": anchor["char_start"], "char_end": anchor["char_end"],
                     "relation_cue": item["relation_cue"], "origin": "model"}]
        values.append(_relation_record(chunk.page_id, source, item["relation"], target, "model", evidence, evidence[0]))
    return values


def audit_superseded_v02_relations(v02_database: Path, model_relations: Sequence[Mapping[str, Any]],
                                   recovered_relations: Sequence[Mapping[str, Any]],
                                   catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Explain why old deterministic triples were not silently carried forward."""
    entries_by_id = {entry["candidate_id"]: entry for entry in catalog.get("entries", [])}
    model_triples = {(item["page_id"], item["source_candidate_key"], item["relation"],
                      item["target_candidate_key"]) for item in model_relations}
    recovered_triples = {(item["page_id"], item["source_candidate_key"], item["relation"],
                          item["target_candidate_key"]) for item in recovered_relations}
    with sqlite3.connect(v02_database) as db:
        rows = db.execute("""
            SELECT source_id, relation, target_id FROM semantic_edges
            WHERE relation IN ('ITEM_MEASURED_BY_METHOD', 'ITEM_HAS_REFERENCE_RANGE',
                               'RANGE_APPLIES_TO_POPULATION')
            ORDER BY source_id, relation, target_id
        """).fetchall()
    review = []
    for source_id, relation, target_id in rows:
        source, target = entries_by_id.get(source_id), entries_by_id.get(target_id)
        if source is None or target is None:
            raise V04ContractError("v0.2 projected relation has a dangling endpoint")
        page_id = source["page_id"]
        triple = (page_id, source["candidate_key"], relation, target["candidate_key"])
        if triple in model_triples or triple in recovered_triples:
            continue
        cross_page = source["page_id"] != target["page_id"]
        reason = "superseded_v02_cross_page_projection" if cross_page else "superseded_v02_structure_not_recovered"
        review.append({"candidate_id": _stable_id("v02-projection-review", source_id, relation, target_id),
                       "candidate_type": "relation", "page_id": page_id,
                       "chunk_id": source["source"]["chunk_id"], "reason_code": reason,
                       "summary": ("v0.2 deterministic projection crossed page boundaries"
                                   if cross_page else "v0.2 deterministic projection lacks v0.4 structural confirmation"),
                       "raw_candidate": {"source_candidate_key": source["candidate_key"],
                                         "source_page_id": source["page_id"], "relation": relation,
                                         "target_candidate_key": target["candidate_key"],
                                         "target_page_id": target["page_id"]}})
    return review


def validate_relations(payload: Mapping[str, Any], chunks: Sequence[EvidenceChunk], catalog: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != {"relations"} or not isinstance(payload["relations"], list):
        raise V04ContractError("relation-only output must contain relations only")
    by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    entries = {(entry["page_id"], entry["candidate_key"]): entry for entry in catalog.get("entries", [])}
    accepted, rejected = [], []
    for index, item in enumerate(payload["relations"]):
        try:
            if not isinstance(item, Mapping) or item.get("relation") not in MODEL_RELATIONS:
                raise V04ContractError("relation is not model-eligible")
            source_ref = {"chunk_id": item.get("source_chunk_id"), "chunk_sha256": item.get("source_chunk_sha256"),
                          "exact_quote": item.get("source_quote")}
            evidence = replay_ref(source_ref, by_chunk)
            chunk = by_chunk[evidence["chunk_id"]]
            source = entries.get((chunk.page_id, item.get("source_candidate_key")))
            target = entries.get((chunk.page_id, item.get("target_candidate_key")))
            if source is None or target is None:
                raise V04ContractError("relation endpoint candidate_key is unknown")
            allowed = MODEL_RELATIONS[item["relation"]]
            targets = allowed[1] if isinstance(allowed[1], tuple) else (allowed[1],)
            if source["entity_type"] != allowed[0] or target["entity_type"] not in targets:
                raise V04ContractError("relation direction or endpoint type is invalid")
            cue = item.get("relation_cue")
            if not isinstance(cue, str) or not cue or cue not in evidence["exact_quote"]:
                raise V04ContractError("relation cue is missing from source quote")
            if source["text"] not in evidence["exact_quote"] or target["text"] not in evidence["exact_quote"]:
                raise V04ContractError("relation quote lacks both endpoint texts")
            relation_evidence = [{"evidence_role": "relation", **evidence, "relation_cue": cue, "origin": "model"}]
            accepted.append(_relation_record(chunk.page_id, source, item["relation"], target, "model", relation_evidence, relation_evidence[0]))
        except (V04ContractError, KeyError) as exc:
            rejected.append(_reject("relation", index, item, str(exc), by_chunk))
    return {"schema_version": "relation-only/v0.4", "status": "candidate-only", "approved": 0,
            "candidates": accepted, "rejections": rejected,
            "counts": {"accepted": len(accepted), "rejected": len(rejected)}}


def _component(value: Any, chunks: Mapping[str, EvidenceChunk]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"text", "source_ref"}:
        raise V04ContractError("rule component shape is invalid")
    source = replay_ref(value["source_ref"], chunks)
    text_span = _text_span(value["text"], source, chunks)
    return {"text": value["text"], "source": text_span}


def validate_rules(payload: Mapping[str, Any], chunks: Sequence[EvidenceChunk], catalog: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != {"rules"} or not isinstance(payload["rules"], list):
        raise V04ContractError("rule-only output must contain rules only")
    by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    entries = {(entry["page_id"], entry["candidate_key"]): entry for entry in catalog.get("entries", [])}
    accepted, rejected = [], []
    for index, item in enumerate(payload["rules"]):
        try:
            if not isinstance(item, Mapping) or item.get("entity_type") != "InterpretationRule":
                raise V04ContractError("rule shape is invalid")
            source = replay_ref(item.get("source_ref"), by_chunk)
            page_id = by_chunk[source["chunk_id"]].page_id
            semantic_type, subject_logic = item.get("semantic_type"), item.get("subject_logic")
            if semantic_type not in SEMANTIC_TYPES or subject_logic not in SUBJECT_LOGICS:
                raise V04ContractError("invalid rule semantic contract")
            subjects = _endpoint_keys(item.get("subject_candidate_keys"), page_id, entries, {"TestItem", "MedicalConcept"}, "subject")
            conclusions = _endpoint_keys([item.get("conclusion_candidate_key")], page_id, entries, {"MedicalConcept"}, "conclusion")
            populations = _endpoint_keys(item.get("population_candidate_keys", []), page_id, entries, {"Population"}, "population", optional=True)
            methods = _endpoint_keys(item.get("method_candidate_keys", []), page_id, entries, {"TestMethod"}, "method", optional=True)
            components = item.get("components")
            if not isinstance(components, Mapping) or not isinstance(components.get("conditions"), list) or not components["conditions"]:
                raise V04ContractError("rule conditions are required")
            result = {"conditions": [_component(value, by_chunk) for value in components["conditions"]],
                      "conclusion": _component(components.get("conclusion"), by_chunk)}
            if subject_logic in {"ALL", "ANY"}:
                result["connector"] = _component(components.get("connector"), by_chunk)
            elif components.get("connector") is not None:
                raise V04ContractError("SINGLE rule cannot have connector")
            condition_text = " ".join(value["text"] for value in result["conditions"])
            for key in subjects:
                if entries[(page_id, key)]["text"] not in condition_text:
                    raise V04ContractError("rule subject endpoint is not grounded in conditions")
            conclusion_key = conclusions[0]
            if entries[(page_id, conclusion_key)]["text"] not in result["conclusion"]["text"]:
                raise V04ContractError("rule conclusion endpoint is not grounded in conclusion")
            if conclusion_key in subjects:
                raise V04ContractError("rule subject and conclusion endpoints must differ")
            evidence_text = " ".join([source["exact_quote"]]
                                     + [value["text"] for value in result["conditions"]]
                                     + [result["conclusion"]["text"]])
            if not any(trigger in evidence_text for trigger in _TRIGGERS[semantic_type]):
                raise V04ContractError("semantic_type lacks its verbatim trigger")
            rule_key = item.get("rule_key")
            if not isinstance(rule_key, str) or not rule_key:
                raise V04ContractError("rule_key is required")
            accepted.append({"candidate_id": _stable_id("rule", page_id, rule_key, source), "candidate_type": "rule",
                "entity_type": "InterpretationRule", "rule_key": rule_key, "page_id": page_id,
                "semantic_type": semantic_type, "subject_logic": subject_logic,
                "subject_candidate_keys": subjects, "conclusion_candidate_key": conclusion_key,
                "population_candidate_keys": populations, "method_candidate_keys": methods,
                "components": result, "source": source, "origin": "model", "status": "candidate"})
        except (V04ContractError, KeyError) as exc:
            rejected.append(_reject("rule", index, item, str(exc), by_chunk))
    return {"schema_version": "rule-only/v0.4", "status": "candidate-only", "approved": 0,
            "candidates": accepted, "rejections": rejected,
            "counts": {"accepted": len(accepted), "rejected": len(rejected)}}


def _endpoint_keys(values: Any, page_id: str, entries: Mapping[tuple[str, str], Mapping[str, Any]],
                   allowed: set[str], role: str, optional: bool = False) -> list[str]:
    if not isinstance(values, list) or (not values and not optional) or len(values) != len(set(values)):
        raise V04ContractError(f"rule {role} keys are missing or duplicated")
    for key in values:
        entry = entries.get((page_id, key)) if isinstance(key, str) else None
        if entry is None or entry["entity_type"] not in allowed:
            raise V04ContractError(f"rule {role} endpoint is unknown")
    return values


def stable_relations(packages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for package in packages:
        for item in package.get("candidates", []):
            key = (item["page_id"], item["source_candidate_key"], item["relation"], item["target_candidate_key"])
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(item, evidence=list(item.get("evidence", [])))
                continue
            for evidence in item.get("evidence", []):
                if evidence not in existing["evidence"]:
                    existing["evidence"].append(evidence)
            if item.get("origin") == "model":
                existing["origin"] = "model"
                existing["source"] = item["source"]
                existing["relation_cue"] = item["relation_cue"]
    return [merged[key] for key in sorted(merged)]


def build_endpoint_prompt(page_id: str, chunks: Sequence[EvidenceChunk], entries: Sequence[Mapping[str, Any]]) -> str:
    example = {"endpoints": [{"candidate_key": "总铁结合力降低", "entity_type": "MedicalConcept",
        "text": "总铁结合力降低", "source_ref": {"chunk_id": "CHUNK_ID", "chunk_sha256": "HASH",
        "exact_quote": "总铁结合力降低: 转铁蛋白合成不足"}}]}
    return (f"PROMPT_VERSION={ENDPOINT_PROMPT_VERSION}\n只返回一个 JSON 对象，顶层字段只能是 endpoints。Page={page_id}。"
            "任务是补齐后续规则需要、但 ENTITY_CATALOG 尚不存在的页内端点，不是重新抽取全部实体。"
            "优先检查【异常结果解读】及含升高、降低、见于、导致、排除、诊断、监测的逐字句子。"
            "主体检验项目用 TestItem；疾病、临床状态、升高/降低状态和规则唯一结论用 MedicalConcept；另可用 Population、TestMethod。"
            "不要批量抄录与规则无关的疾病列表，不要输出 ReferenceRange，不要输出目录中已有端点。"
            "每项字段必须且只能是 candidate_key,entity_type,text,source_ref。candidate_key 建议等于逐字 text。"
            "source_ref 必须完整包含 chunk_id,chunk_sha256,exact_quote；exact_quote 在该 chunk 中只能出现一次，text 必须是其中连续逐字且唯一的子串。"
            "禁止改写、合并列表、补同义词、跨 chunk 或外部知识。示例只展示形状，不得复制示例标识：" +
            _canonical(example) + "\nENTITY_CATALOG=" + _canonical(list(entries)) +
            "\nCHUNKS=" + _canonical([asdict(chunk) for chunk in chunks]))


def build_relation_prompt(page_id: str, chunks: Sequence[EvidenceChunk], entries: Sequence[Mapping[str, Any]]) -> str:
    shape = {"source_candidate_key": "目录中的key", "target_candidate_key": "目录中的key",
             "relation": "ITEM_MEASURED_BY_METHOD", "source_chunk_id": "CHUNK_ID",
             "source_chunk_sha256": "HASH", "source_quote": "包含两个端点和采用的唯一逐字引文",
             "relation_cue": "采用"}
    return (f"PROMPT_VERSION={RELATION_PROMPT_VERSION}\n只返回一个 JSON 对象，顶层字段只能是 relations。Page={page_id}。"
            "每条关系必须且只能使用此字段形状：" + _canonical(shape) + "。"
            "source_candidate_key 和 target_candidate_key 必须逐字复制同页 ENTITY_CATALOG 的 key，不得使用 id、endpoint 数组、relation_type 或自造 key。"
            "source_quote 必须是单个 chunk 中唯一出现的连续逐字引文，并同时包含两个端点的 text 与 relation_cue。"
            "禁止跨 chunk，禁止从标题或表格结构推断，结构关系由本地程序处理。允许关系=" +
            _canonical(sorted(MODEL_RELATIONS)) + "\nENTITY_CATALOG=" + _canonical(list(entries)) +
            "\nCHUNKS=" + _canonical([asdict(chunk) for chunk in chunks]))


def build_rule_prompt(page_id: str, chunks: Sequence[EvidenceChunk], entries: Sequence[Mapping[str, Any]]) -> str:
    example = {"rule_key": "D-二聚体正常排除DVT", "entity_type": "InterpretationRule",
               "semantic_type": "DIFFERENTIAL_DIAGNOSIS", "subject_logic": "SINGLE",
               "subject_candidate_keys": ["D-二聚体"], "conclusion_candidate_key": "深静脉血栓",
               "population_candidate_keys": [], "method_candidate_keys": [],
               "source_ref": {"chunk_id": "CHUNK_ID", "chunk_sha256": "HASH", "exact_quote": "D-二聚体正常，对排除深静脉血栓(DVT)有重要价值。"},
               "components": {"conditions": [{"text": "D-二聚体正常", "source_ref": {"chunk_id": "CHUNK_ID", "chunk_sha256": "HASH", "exact_quote": "D-二聚体正常，对排除深静脉血栓(DVT)有重要价值。"}}],
                              "conclusion": {"text": "排除深静脉血栓(DVT)", "source_ref": {"chunk_id": "CHUNK_ID", "chunk_sha256": "HASH", "exact_quote": "D-二聚体正常，对排除深静脉血栓(DVT)有重要价值。"}}}}
    return (f"PROMPT_VERSION={RULE_PROMPT_VERSION}\n只返回一个 JSON 对象，顶层字段只能是 rules。Page={page_id}。"
            "每条规则字段必须且只能是 rule_key,entity_type,semantic_type,subject_logic,subject_candidate_keys,conclusion_candidate_key,population_candidate_keys,method_candidate_keys,source_ref,components。"
            "entity_type 固定为 InterpretationRule；semantic_type 只能是 " + _canonical(sorted(SEMANTIC_TYPES)) +
            "；subject_logic 只能是 SINGLE、ALL、ANY。所有端点 key 必须逐字复制同页 ENTITY_CATALOG。"
            "source_ref 以及每个组件的 source_ref 都必须是完整的 {chunk_id,chunk_sha256,exact_quote} 对象；exact_quote 必须在 chunk 中唯一出现。"
            "每个组件 text 必须是它自己的 exact_quote 中连续逐字子串，禁止把箭头改写成升高/降低、把标点改写成同时、合并列表或补词。"
            "conditions 至少一个且 conclusion 恰好一个。SINGLE 不得有 connector；ALL/ANY 必须有原文真实存在的 connector，表格相邻列或空格不算 connector。"
            "语义触发词必须逐字出现在整条 source_ref 或组件中。没有完整主体、唯一结论和逐字组件就不要输出。"
            "以下示例只展示形状，CHUNK_ID/HASH 必须替换为输入值，不得复制示例内容。EXAMPLE=" +
            _canonical(example) + "\nENTITY_CATALOG=" + _canonical(list(entries)) + "\nCHUNKS=" + _canonical([asdict(chunk) for chunk in chunks]))
