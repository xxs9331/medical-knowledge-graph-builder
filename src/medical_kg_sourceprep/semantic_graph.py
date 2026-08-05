"""Conservative, provenance-bound semantic projection over a v0.2 graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Mapping, Sequence


SEMANTIC_SOURCE_PACKAGE_VERSION = "semantic-source-package/v0.3"
ENTITY_TYPES = frozenset({
    "TestItem", "TestMethod", "ReferenceRange", "InterpretationRule", "MedicalConcept",
    "Population", "SourceLocator",
})
SEMANTIC_TYPES = frozenset({
    "DEFINES_AS", "POSSIBLY_CAUSED_BY", "SEEN_IN", "LEADS_TO", "RECOVERY_FACTOR",
    "CLASSIFIES_AS", "DIAGNOSTIC_CRITERION", "REFERENCE_INTERPRETATION",
    "ABNORMAL_RESULT_INTERPRETATION", "MONITORING_GUIDANCE", "DIFFERENTIAL_DIAGNOSIS",
    "RISK_ASSOCIATION", "PROGNOSTIC_INDICATOR", "INTERPRETATION_CAVEAT",
})
SUBJECT_LOGICS = frozenset({"SINGLE", "ALL", "ANY"})
SEMANTIC_RELATIONS = {
    "ITEM_MEASURED_BY_METHOD": ("TestItem", "TestMethod"),
    "ITEM_HAS_REFERENCE_RANGE": ("TestItem", "ReferenceRange"),
    "RANGE_APPLIES_TO_POPULATION": ("ReferenceRange", "Population"),
    "ITEM_SUPPORTED_BY": ("TestItem", "SourceLocator"),
    "RANGE_SUPPORTED_BY": ("ReferenceRange", "SourceLocator"),
    "RULE_HAS_SUBJECT": ("InterpretationRule", ("TestItem", "MedicalConcept")),
    "RULE_HAS_CONCLUSION": ("InterpretationRule", ("MedicalConcept", "TestItem")),
    "RULE_APPLIES_TO_POPULATION": ("InterpretationRule", "Population"),
    "RULE_REQUIRES_METHOD": ("InterpretationRule", "TestMethod"),
    "RULE_SUPPORTED_BY": ("InterpretationRule", "SourceLocator"),
}
STATUSES = frozenset({"candidate", "reviewed", "approved", "rejected"})
_NUMBERED_SECTION = re.compile(r"^\s*(?:[0-9]+[.、]|[一二三四五六七八九十百千]+、)\s*(\S.*?)\s*$")
_REFERENCE_LABEL = "【参考区间】"
_ITEM_LABEL = "【检验项目】"
_ABNORMAL_LABEL = "【异常结果解读】"


class SemanticGraphError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    reviewer: str
    reviewed_at: str
    rationale: str
    source_version: str


@dataclass(frozen=True, slots=True)
class SemanticRecord:
    record_id: str
    entity_type: str
    status: str
    text: str
    chunk_id: str
    char_start: int
    char_end: int
    semantic_type: str | None = None
    subject_logic: str | None = None
    review: ReviewRecord | None = None
    rule_payload: Mapping[str, object] | None = None
    candidate_key: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticRelation:
    """A semantic edge with independent, replayable relation evidence."""

    source_id: str
    relation: str
    target_id: str
    source_chunk_id: str
    source_quote: str
    relation_cue: str
    source_chunk_sha256: str | None = None
    origin: str = "model"
    evidence: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticBuildResult:
    database_path: Path
    node_count: int
    edge_count: int
    status_counts: Mapping[str, int]
    unstructured_reasons: Mapping[str, int]
    package_hash: str


def _hash(value: object) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _id(prefix: str, chunk_id: str, offset: int, end: int | None = None) -> str:
    span = f"{chunk_id}:{offset}" if end is None else f"{chunk_id}:{offset}:{end}"
    return f"semantic:{prefix}:{hashlib.sha256(span.encode()).hexdigest()[:20]}"


def _record_value(record: SemanticRecord) -> dict[str, object]:
    value = asdict(record)
    return value


class SemanticGraphBuilder:
    """Copy a validated v0.2 graph and add auditable, non-executable semantics."""

    def build(self, database_path: Path, base_graph: Path, source_package: Mapping[str, object],
              records: Sequence[SemanticRecord] = (),
              relations: Sequence[tuple[str, str, str] | SemanticRelation] = ()) -> SemanticBuildResult:
        database_path, base_graph = Path(database_path), Path(base_graph)
        if database_path.exists() or not base_graph.is_file():
            raise SemanticGraphError("output exists or base knowledge graph is unavailable")
        if not isinstance(source_package, Mapping) or not isinstance(
            source_package.get("content_sha256"), str
        ):
            raise SemanticGraphError("semantic source package must be a validated book manifest")
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{database_path.name}.", suffix=".sqlite", dir=database_path.parent,
            delete=False,
        ) as handle:
            staging = Path(handle.name)
        try:
            shutil.copyfile(base_graph, staging)
            with sqlite3.connect(staging) as db:
                db.execute("PRAGMA foreign_keys = ON")
                self._schema(db)
                schema = db.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()
                if schema is None or schema[0] != "knowledge-graph/v0.2":
                    raise SemanticGraphError("base graph is not knowledge-graph/v0.2")
                book = db.execute("SELECT payload FROM nodes WHERE node_type = 'Book'").fetchone()
                if book is None or json.loads(book[0]).get("manifest_hash") != source_package[
                    "content_sha256"
                ]:
                    raise SemanticGraphError("base graph manifest binding drift")
                chunks = self._chunks(db, source_package)
                projected_records, projected_relations = self._project(chunks)
                all_records = [*records, *projected_records]
                identifiers: dict[str, SemanticRecord] = {}
                for record in all_records:
                    self._insert(db, record, chunks, identifiers)
                support_relations = self._source_locator_records(db, chunks, identifiers)
                all_relations = self._relation_values([*relations, *projected_relations, *support_relations])
                self._relations(db, all_relations, identifiers)
                package_hash = _hash({
                    "version": SEMANTIC_SOURCE_PACKAGE_VERSION,
                    "manifest": source_package["content_sha256"],
                    "records": sorted(
                        (_record_value(record) for record in identifiers.values()),
                        key=lambda value: str(value["record_id"]),
                    ),
                    "relations": [
                        {"triple": (value.source_id, value.relation, value.target_id),
                         "origin": value.origin, "evidence": list(value.evidence),
                         "source_quote": value.source_quote, "relation_cue": value.relation_cue}
                        if isinstance(value, SemanticRelation) else value
                        for value in all_relations
                    ],
                })
                db.executemany("INSERT INTO semantic_metadata VALUES (?, ?)", (
                    ("schema_version", SEMANTIC_SOURCE_PACKAGE_VERSION),
                    ("base_manifest_hash", source_package["content_sha256"]),
                    ("source_package_hash", package_hash),
                    ("record_count", str(len(identifiers))),
                    ("relation_count", str(len(all_relations))),
                ))
                self._mirror_main_graph(db, identifiers, all_relations)
                if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise SemanticGraphError("SQLite integrity check failed")
                if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    raise SemanticGraphError("SQLite foreign key check failed")
                count = db.execute("SELECT COUNT(*) FROM semantic_records").fetchone()[0]
                edge_count = db.execute("SELECT COUNT(*) FROM semantic_edges").fetchone()[0]
                statuses = dict(
                    db.execute("SELECT status, COUNT(*) FROM semantic_records GROUP BY status")
                )
                reasons = dict(
                    db.execute("SELECT reason, COUNT(*) FROM semantic_unstructured GROUP BY reason")
                )
            staging.replace(database_path)
        except Exception:
            staging.unlink(missing_ok=True)
            raise
        return SemanticBuildResult(database_path, count, edge_count, statuses, reasons, package_hash)

    @staticmethod
    def _schema(db: sqlite3.Connection) -> None:
        db.executescript("""
        CREATE TABLE semantic_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE semantic_records (record_id TEXT PRIMARY KEY, candidate_key TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            status TEXT NOT NULL, text TEXT NOT NULL, semantic_type TEXT, subject_logic TEXT,
            review_json TEXT, rule_payload_json TEXT);
        CREATE TABLE semantic_source_locators (locator_id TEXT PRIMARY KEY,
            record_id TEXT NOT NULL REFERENCES semantic_records(record_id), chunk_id TEXT NOT NULL,
            chunk_sha256 TEXT NOT NULL, printed_page INTEGER, source_pdf_page INTEGER,
            char_start INTEGER NOT NULL, char_end INTEGER NOT NULL, exact_quote TEXT NOT NULL,
            anchor_json TEXT NOT NULL);
        CREATE TABLE semantic_edges (edge_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES semantic_records(record_id), relation TEXT NOT NULL,
            target_id TEXT NOT NULL REFERENCES semantic_records(record_id),
            origin TEXT NOT NULL CHECK(origin IN ('model', 'derived')),
            UNIQUE(source_id, relation, target_id));
        CREATE TABLE semantic_relation_evidence (evidence_id TEXT PRIMARY KEY,
            edge_id TEXT NOT NULL, evidence_role TEXT NOT NULL,
            source_chunk_id TEXT NOT NULL, source_chunk_sha256 TEXT NOT NULL,
            source_quote TEXT NOT NULL, relation_cue TEXT NOT NULL,
            char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
            FOREIGN KEY(edge_id) REFERENCES semantic_edges(edge_id));
        CREATE TABLE semantic_unstructured (chunk_id TEXT NOT NULL, reason TEXT NOT NULL, PRIMARY KEY(chunk_id, reason));
        """)

    @staticmethod
    def _chunks(db: sqlite3.Connection, manifest: Mapping[str, object]) -> dict[
        str, tuple[str, int, int, str, int | None, int | None, int]
    ]:
        pages = {page["page_id"]: page for page in manifest.get("pages", []) if isinstance(page, Mapping)}
        expected = {chunk["chunk_id"]: chunk for chunk in manifest.get("chunks", []) if isinstance(chunk, Mapping)}
        rows = db.execute(
            "SELECT chunk_id, page_id, start_offset, end_offset, content FROM chunk_text ORDER BY chunk_id"
        ).fetchall()
        result = {}
        for chunk_id, page_id, start, end, text in rows:
            chunk, page = expected.get(chunk_id), pages.get(page_id)
            if (chunk is None or page is None
                    or chunk.get("chunk_sha256") != hashlib.sha256(text.encode()).hexdigest()
                    or (start, end) != (chunk.get("cleaned_char_start"), chunk.get("cleaned_char_end"))):
                raise SemanticGraphError("base graph and source package chunk binding drift")
            result[chunk_id] = (
                text, start, end, hashlib.sha256(text.encode()).hexdigest(),
                page.get("printed_page_number"), page.get("source_pdf_page_number"),
                page["chapter_page_index"],
            )
        if set(result) != set(expected):
            raise SemanticGraphError("base graph lacks a source package chunk")
        return result

    def _insert(self, db: sqlite3.Connection, record: SemanticRecord,
                chunks: Mapping[str, tuple[object, ...]],
                identifiers: dict[str, SemanticRecord]) -> None:
        if (not isinstance(record, SemanticRecord) or record.record_id in identifiers
                or record.entity_type not in ENTITY_TYPES or record.status not in STATUSES):
            raise SemanticGraphError("invalid or duplicate semantic record")
        if record.entity_type == "InterpretationRule" and (
            record.semantic_type not in SEMANTIC_TYPES or record.subject_logic not in SUBJECT_LOGICS
        ):
            raise SemanticGraphError("rule requires fixed semantic type and subject logic")
        if record.status == "approved":
            review = record.review
            fields = (review.reviewer, review.reviewed_at, review.rationale, review.source_version) if review else ()
            if not review or not all(isinstance(value, str) and value.strip() for value in fields):
                raise SemanticGraphError("approved record requires complete review")
        if record.chunk_id not in chunks or not isinstance(record.text, str) or not record.text:
            raise SemanticGraphError("semantic record lacks source binding")
        text, _, _, digest, printed, pdf, _ = chunks[record.chunk_id]
        if (not isinstance(record.char_start, int) or not isinstance(record.char_end, int)
                or record.char_start < 0 or record.char_end <= record.char_start
                or text[record.char_start:record.char_end] != record.text):
            raise SemanticGraphError("source locator anchor drift")
        locator = f"semantic:anchor:{_hash([record.record_id, record.chunk_id, record.char_start])[:20]}"
        anchor = {"chunk_id": record.chunk_id, "chunk_sha256": digest,
                  "char_start": record.char_start, "char_end": record.char_end,
                  "exact_quote": record.text}
        db.execute("INSERT INTO semantic_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            record.record_id, record.candidate_key or record.record_id,
            record.entity_type, record.status, record.text,
            record.semantic_type, record.subject_logic,
            json.dumps(asdict(record.review), sort_keys=True) if record.review else None,
            json.dumps(record.rule_payload, ensure_ascii=False, sort_keys=True) if record.rule_payload else None,
        ))
        db.execute("INSERT INTO semantic_source_locators VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            locator, record.record_id, record.chunk_id, digest, printed, pdf,
            record.char_start, record.char_end, record.text, json.dumps(anchor, sort_keys=True),
        ))
        identifiers[record.record_id] = record

    def _source_locator_records(self, db: sqlite3.Connection, chunks: Mapping[str, tuple[object, ...]], identifiers: dict[str, SemanticRecord]) -> list[tuple[str, str, str]]:
        relations = []
        for record in list(identifiers.values()):
            if record.entity_type not in {"TestItem", "ReferenceRange", "InterpretationRule"}:
                continue
            locator_id = _id(
                "locator", record.chunk_id, record.char_start, record.char_end
            )
            locator = SemanticRecord(
                locator_id, "SourceLocator", "candidate", record.text,
                record.chunk_id, record.char_start, record.char_end,
            )
            existing = identifiers.get(locator_id)
            if existing is None:
                self._insert(db, locator, chunks, identifiers)
            elif existing != locator:
                raise SemanticGraphError("source locator identity collision")
            relation = {"TestItem": "ITEM_SUPPORTED_BY", "ReferenceRange": "RANGE_SUPPORTED_BY", "InterpretationRule": "RULE_SUPPORTED_BY"}[record.entity_type]
            relations.append((record.record_id, relation, locator_id))
        return relations

    @staticmethod
    def _relation_values(relations: Sequence[tuple[str, str, str] | SemanticRelation]) -> list[tuple[str, str, str] | SemanticRelation]:
        values: dict[tuple[str, str, str], tuple[str, str, str] | SemanticRelation] = {}
        for value in relations:
            key = (value.source_id, value.relation, value.target_id) if isinstance(value, SemanticRelation) else tuple(value)
            existing = values.get(key)
            if isinstance(existing, SemanticRelation) and isinstance(value, SemanticRelation):
                evidence = list(existing.evidence or ())
                if not evidence:
                    evidence.append({"evidence_role": "relation", "source_chunk_id": existing.source_chunk_id,
                        "source_chunk_sha256": existing.source_chunk_sha256, "source_quote": existing.source_quote,
                        "relation_cue": existing.relation_cue})
                for item in value.evidence or ({"evidence_role": "relation", "source_chunk_id": value.source_chunk_id,
                    "source_chunk_sha256": value.source_chunk_sha256, "source_quote": value.source_quote,
                    "relation_cue": value.relation_cue},):
                    if item not in evidence:
                        evidence.append(item)
                preferred = value if value.origin == "model" and existing.origin != "model" else existing
                values[key] = SemanticRelation(preferred.source_id, preferred.relation, preferred.target_id,
                    preferred.source_chunk_id, preferred.source_quote, preferred.relation_cue,
                    preferred.source_chunk_sha256, "model" if {existing.origin, value.origin} == {"model", "derived"} else preferred.origin,
                    tuple(evidence))
            elif existing is None or isinstance(value, SemanticRelation):
                values[key] = value
        return [values[key] for key in sorted(values)]

    @staticmethod
    def _relations(db: sqlite3.Connection, relations: Sequence[tuple[str, str, str] | SemanticRelation], records: Mapping[str, SemanticRecord]) -> None:
        conclusion_counts: dict[str, int] = {}
        simple: dict[tuple[str, str, str], SemanticRelation | None] = {}
        for value in relations:
            if isinstance(value, SemanticRelation):
                key = (value.source_id, value.relation, value.target_id)
                if key in simple and simple[key] != value:
                    raise SemanticGraphError("duplicate relation has conflicting evidence")
                simple[key] = value
            else:
                simple.setdefault(tuple(value), None)
        for number, (source, relation, target) in enumerate(sorted(simple), 1):
            allowed = SEMANTIC_RELATIONS.get(relation)
            targets = allowed[1] if allowed and isinstance(allowed[1], tuple) else (allowed[1],) if allowed else ()
            if not allowed or source not in records or target not in records or records[source].entity_type != allowed[0] or records[target].entity_type not in targets:
                raise SemanticGraphError("semantic relation is not allowed for these entity types")
            if relation == "RULE_HAS_CONCLUSION":
                conclusion_counts[source] = conclusion_counts.get(source, 0) + 1
                if conclusion_counts[source] > 1:
                    raise SemanticGraphError("each rule has one conclusion")
            edge_id = f"semantic-edge-{number:06d}"
            evidence = simple[(source, relation, target)]
            origin = evidence.origin if evidence is not None else "model"
            db.execute("INSERT INTO semantic_edges VALUES (?, ?, ?, ?, ?)", (edge_id, source, relation, target, origin))
            if evidence is not None:
                if evidence.origin not in {"model", "derived"}:
                    raise SemanticGraphError("relation origin is invalid")
                if evidence.source_chunk_id not in {row[0] for row in db.execute("SELECT chunk_id FROM chunk_text") }:
                    raise SemanticGraphError("relation evidence chunk is unavailable")
                row = db.execute("SELECT content, start_offset FROM chunk_text WHERE chunk_id = ?", (evidence.source_chunk_id,)).fetchone()
                digest = hashlib.sha256(row[0].encode()).hexdigest()
                if evidence.source_chunk_sha256 not in (None, digest):
                    raise SemanticGraphError("relation evidence hash drift")
                start = row[0].find(evidence.source_quote)
                if start < 0 or row[0].count(evidence.source_quote) != 1 or evidence.relation_cue not in evidence.source_quote:
                    raise SemanticGraphError("relation evidence quote is absent, ambiguous, or lacks cue")
                values = evidence.evidence or ({"evidence_role": "relation", "source_chunk_id": evidence.source_chunk_id,
                    "source_chunk_sha256": evidence.source_chunk_sha256, "source_quote": evidence.source_quote,
                    "relation_cue": evidence.relation_cue},)
                for evidence_number, item in enumerate(values, 1):
                    chunk_id = item.get("source_chunk_id")
                    quote = item.get("source_quote")
                    cue = item.get("relation_cue")
                    role = item.get("evidence_role", "relation")
                    if not all(isinstance(value, str) and value for value in (chunk_id, quote, cue, role)):
                        raise SemanticGraphError("relation evidence is incomplete")
                    row = db.execute("SELECT content FROM chunk_text WHERE chunk_id = ?", (chunk_id,)).fetchone()
                    if row is None:
                        raise SemanticGraphError("relation evidence chunk is unavailable")
                    chunk_text = row[0]
                    evidence_start = chunk_text.find(quote)
                    evidence_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                    supplied_hash = item.get("source_chunk_sha256")
                    if supplied_hash not in (None, evidence_hash) or evidence_start < 0 or chunk_text.count(quote) != 1 or cue not in quote:
                        raise SemanticGraphError("relation evidence quote is absent, ambiguous, or lacks cue")
                    evidence_id = f"{edge_id}:evidence-{evidence_number:03d}"
                    db.execute("INSERT INTO semantic_relation_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (evidence_id, edge_id, role, chunk_id, evidence_hash, quote, cue,
                         evidence_start, evidence_start + len(quote)))

    @staticmethod
    def _mirror_main_graph(db: sqlite3.Connection, records: Mapping[str, SemanticRecord], relations: Sequence[tuple[str, str, str]]) -> None:
        locator_rows = {record_id: (chunk_id, anchor_json) for record_id, chunk_id, anchor_json in db.execute("SELECT record_id, chunk_id, anchor_json FROM semantic_source_locators")}
        for record_id in sorted(records):
            record = records[record_id]
            payload = json.dumps({"text": record.text, "status": record.status, "semantic_type": record.semantic_type, "subject_logic": record.subject_logic}, ensure_ascii=False, sort_keys=True)
            db.execute("INSERT INTO nodes VALUES (?, ?, ?)", (record_id, record.entity_type, payload))
            chunk_id, anchor_json = locator_rows[record_id]
            db.execute("INSERT INTO anchors VALUES (?, ?, ?, ?)", (f"{record_id}:anchor", record_id, "semantic_source", anchor_json))
            if record.entity_type == "SourceLocator":
                edge_id = f"semantic-structural-{_hash([record_id, chunk_id])[:20]}"
                db.execute("INSERT INTO edges VALUES (?, ?, ?, ?, ?)", (edge_id, record_id, chunk_id, "SOURCE_LOCATOR_TARGETS_CHUNK", "structural"))
        for value in relations:
            source, relation, target = (value.source_id, value.relation, value.target_id) if isinstance(value, SemanticRelation) else value
            edge_id = f"semantic-edge-{_hash([source, relation, target])[:20]}"
            db.execute("INSERT INTO edges VALUES (?, ?, ?, ?, ?)", (edge_id, source, target, relation, "semantic"))

    def _project(self, chunks: Mapping[str, tuple[object, ...]]) -> tuple[list[SemanticRecord], list[tuple[str, str, str]]]:
        records: list[SemanticRecord] = []
        relations: list[tuple[str, str, str]] = []
        pending_item: SemanticRecord | None = None
        pending_reference: tuple[SemanticRecord, int] | None = None
        promoted: set[str] = set()
        ordered = sorted(chunks.items(), key=lambda item: (item[1][6], item[1][1], item[0]))
        for chunk_id, (text, *_ignored) in ordered:
            for line_start, line in self._lines(str(text)):
                if re.match(r"^\s*#\s+", line):
                    pending_item = None
                    pending_reference = None
                    continue
                section = _NUMBERED_SECTION.match(line)
                if section:
                    title = section.group(1).strip()
                    offset = line_start + line.index(title)
                    pending_item = SemanticRecord(_id("item", chunk_id, offset), "TestItem", "candidate", title, chunk_id, offset, offset + len(title))
                    pending_reference = None
                    continue
                if line.startswith(_ITEM_LABEL) and line[len(_ITEM_LABEL):].strip():
                    title = line[len(_ITEM_LABEL):].strip()
                    offset = line_start + line.index(title)
                    pending_item = SemanticRecord(_id("item", chunk_id, offset), "TestItem", "candidate", title, chunk_id, offset, offset + len(title))
                    pending_reference = None
                    continue
                if line.startswith(_REFERENCE_LABEL):
                    value = line[len(_REFERENCE_LABEL):].strip()
                    if pending_item is None:
                        continue
                    if value:
                        self._promote_range(records, relations, promoted, pending_item, chunk_id, line_start + line.index(value), value)
                    else:
                        pending_reference = (pending_item, line_start)
                    continue
                if pending_reference and line.strip() and not line.startswith(_ABNORMAL_LABEL):
                    item, _ = pending_reference
                    value = line.strip()
                    offset = line_start + line.index(value)
                    self._promote_range(records, relations, promoted, item, chunk_id, offset, value)
                    pending_reference = None
        return records, relations

    @staticmethod
    def _promote_range(records: list[SemanticRecord], relations: list[tuple[str, str, str]], promoted: set[str], item: SemanticRecord, chunk_id: str, offset: int, value: str) -> None:
        if item.record_id not in promoted:
            records.append(item)
            promoted.add(item.record_id)
        range_id = _id("range", chunk_id, offset)
        records.append(SemanticRecord(range_id, "ReferenceRange", "candidate", value, chunk_id, offset, offset + len(value)))
        relations.append((item.record_id, "ITEM_HAS_REFERENCE_RANGE", range_id))

    @staticmethod
    def _lines(text: str):
        offset = 0
        for line in text.splitlines(keepends=True):
            yield offset, line.rstrip("\r\n")
            offset += len(line)
