"""Deterministic, provenance-bound SQLite knowledge graphs for prepared books.

The parser intentionally recognizes only explicit, generic Markdown labels. It
never infers a subject or manufactures a rule from a table, formula, or prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Mapping, Sequence
import unicodedata

from ..provenance.book_sources import (
    SourceProvenanceError,
    create_text_anchor,
    validate_book_manifest,
)
from ..provenance.package_validation import ChunkPackageError, sha256_bytes, validate_chunk_layout


GRAPH_SCHEMA_VERSION = "knowledge-graph/v0.2"
STRUCTURAL = "structural"
SEMANTIC = "semantic"


class GraphBuildError(ValueError):
    """Raised when graph input cannot be represented without losing provenance."""


@dataclass(frozen=True, slots=True)
class PageText:
    page_id: str
    raw_text: str
    cleaned_text: str


@dataclass(frozen=True, slots=True)
class RuleVersion:
    rule_version_id: str
    rule_id: str
    version: str
    status: str


@dataclass(frozen=True, slots=True)
class AtomicPredicate:
    predicate_id: str
    expression_id: str
    text: str
    anchor_id: str


@dataclass(frozen=True, slots=True)
class RuleExpression:
    expression_id: str
    rule_version_id: str
    operator: str
    conclusion_anchor_id: str


@dataclass(frozen=True, slots=True)
class GraphBuildResult:
    database_path: Path
    node_count: int
    edge_count: int
    rule_count: int
    candidate_rule_count: int


def _line_at(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _unique_raw_anchor(
    raw_text: str, cleaned_text: str, quote: str, cleaned_offset: int, context_chars: int = 32
) -> tuple[int, int]:
    """Map a cleaned quote to its sole raw occurrence using retained context."""
    for width in range(context_chars, -1, -1):
        prefix = cleaned_text[max(0, cleaned_offset - width) : cleaned_offset]
        suffix = cleaned_text[
            cleaned_offset + len(quote) : cleaned_offset + len(quote) + width
        ]
        matches: list[int] = []
        search_from = 0
        while True:
            raw_offset = raw_text.find(quote, search_from)
            if raw_offset < 0:
                break
            if raw_text[:raw_offset].endswith(prefix) and raw_text[
                raw_offset + len(quote) :
            ].startswith(suffix):
                matches.append(raw_offset)
            search_from = raw_offset + 1
        if len(matches) == 1:
            return matches[0], width
    raise GraphBuildError("raw text cannot uniquely map anchor quote with cleaned context")


def _unique_raw_offset(
    raw_text: str, cleaned_text: str, quote: str, cleaned_offset: int, context_chars: int = 32
) -> int:
    return _unique_raw_anchor(
        raw_text, cleaned_text, quote, cleaned_offset, context_chars
    )[0]


def _safe_id(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    readable = unicodedata.normalize("NFKD", normalized).encode("ascii", "ignore").decode("ascii")
    readable = re.sub(r"[^A-Za-z0-9:._-]+", "-", readable).strip("-").lower() or "unicode"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{readable}-{digest}"


@dataclass(frozen=True, slots=True)
class _TableCell:
    display_text: str
    source_quote: str
    source_start: int
    source_end: int
    is_header: bool


class _DisplayTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _display_text(source_quote: str) -> str:
    parser = _DisplayTextParser()
    parser.feed(source_quote)
    parser.close()
    return "".join(parser.parts).strip()


class _TableParser:
    """Parse complete table cells while retaining their original lexical spans."""

    def __init__(self, source: str) -> None:
        self.caption = ""
        self.rows: list[list[_TableCell]] = []
        caption = re.search(r"(?is)<caption\b[^>]*>(.*?)</caption\s*>", source)
        if caption is not None:
            self.caption = _display_text(caption.group(1))
        for row_match in re.finditer(r"(?is)<tr\b[^>]*>(.*?)</tr\s*>", source):
            row: list[_TableCell] = []
            for cell_match in re.finditer(r"(?is)<(td|th)\b[^>]*>(.*?)</\1\s*>", row_match.group(1)):
                source_quote = cell_match.group(2)
                display_text = _display_text(source_quote)
                if not display_text:
                    continue
                relative_start = row_match.start(1) + cell_match.start(2)
                row.append(
                    _TableCell(
                        display_text=display_text,
                        source_quote=source_quote,
                        source_start=relative_start,
                        source_end=relative_start + len(source_quote),
                        is_header=cell_match.group(1).lower() == "th",
                    )
                )
            if row:
                self.rows.append(row)


class KnowledgeGraphBuilder:
    """Build a graph only from an already validated manifest and supplied page text."""

    def build(
        self,
        database_path: Path,
        book_manifest: Mapping[str, Any],
        pages: Sequence[PageText],
    ) -> GraphBuildResult:
        database_path = Path(database_path)
        if database_path.exists():
            raise GraphBuildError("database path already exists; refusing to overwrite")
        try:
            validate_book_manifest(book_manifest)
        except SourceProvenanceError as error:
            raise GraphBuildError(f"invalid book manifest: {error}") from error
        page_texts = self._validate_pages(book_manifest, pages)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{database_path.name}.", suffix=".sqlite", dir=database_path.parent, delete=False
        ) as temporary:
            staging = Path(temporary.name)
        try:
            with sqlite3.connect(staging) as connection:
                connection.execute("PRAGMA foreign_keys = ON")
                self._create_schema(connection)
                self._populate(connection, book_manifest, page_texts)
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise GraphBuildError("SQLite integrity check failed")
                node_count = connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
                edge_count = connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
                rule_count = connection.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
                candidate_count = connection.execute(
                    "SELECT COUNT(*) FROM rules WHERE status = 'candidate'"
                ).fetchone()[0]
            staging.replace(database_path)
        except Exception:
            staging.unlink(missing_ok=True)
            raise
        return GraphBuildResult(database_path, node_count, edge_count, rule_count, candidate_count)

    def _validate_pages(
        self, manifest: Mapping[str, Any], pages: Sequence[PageText]
    ) -> dict[str, PageText]:
        expected = {page["page_id"]: page for page in manifest["pages"]}
        supplied: dict[str, PageText] = {}
        for page in pages:
            if not isinstance(page, PageText) or page.page_id in supplied:
                raise GraphBuildError("page text records must have unique PageText page_id values")
            if page.page_id not in expected:
                raise GraphBuildError("page text references an unknown manifest page")
            record = expected[page.page_id]
            if sha256_bytes(page.raw_text.encode("utf-8")) != record["raw_sha256"] or sha256_bytes(page.cleaned_text.encode("utf-8")) != record["cleaned_sha256"]:
                raise GraphBuildError("page text hash drift")
            supplied[page.page_id] = page
        if set(supplied) != set(expected):
            raise GraphBuildError("page text must cover every manifest page")
        self._validate_chunks(manifest, supplied)
        return supplied

    def _validate_chunks(self, manifest: Mapping[str, Any], pages: Mapping[str, PageText]) -> None:
        try:
            validate_chunk_layout(
                manifest,
                {page_id: page.cleaned_text for page_id, page in pages.items()},
            )
        except ChunkPackageError as error:
            raise GraphBuildError(str(error)) from error

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE nodes (node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE edges (
                edge_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES nodes(node_id),
                target_id TEXT NOT NULL REFERENCES nodes(node_id), relation TEXT NOT NULL,
                edge_kind TEXT NOT NULL CHECK(edge_kind IN ('structural', 'semantic')),
                UNIQUE(source_id, target_id, relation)
            );
            CREATE TABLE anchors (
                anchor_id TEXT PRIMARY KEY, node_id TEXT NOT NULL REFERENCES nodes(node_id),
                role TEXT NOT NULL, anchor_json TEXT NOT NULL
            );
            CREATE TABLE chunk_text (
                chunk_id TEXT PRIMARY KEY REFERENCES nodes(node_id),
                page_id TEXT NOT NULL REFERENCES nodes(node_id), start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL, content TEXT NOT NULL
            );
            CREATE TABLE rules (
                rule_id TEXT PRIMARY KEY REFERENCES nodes(node_id), status TEXT NOT NULL CHECK(status IN ('candidate', 'approved')),
                version_id TEXT NOT NULL REFERENCES nodes(node_id), expression_id TEXT NOT NULL REFERENCES nodes(node_id),
                review_record TEXT
            );
            CREATE TABLE table_cells (
                table_id TEXT NOT NULL REFERENCES nodes(node_id), row_index INTEGER NOT NULL,
                column_index INTEGER NOT NULL, is_header INTEGER NOT NULL, text TEXT NOT NULL,
                anchor_id TEXT NOT NULL REFERENCES anchors(anchor_id), table_title TEXT NOT NULL,
                PRIMARY KEY(table_id, row_index, column_index)
            );
            """
        )

    def _populate(
        self, connection: sqlite3.Connection, manifest: Mapping[str, Any], pages: Mapping[str, PageText]) -> None:
        def node(node_id: str, node_type: str, **payload: Any) -> None:
            connection.execute("INSERT INTO nodes VALUES (?, ?, ?)", (node_id, node_type, json.dumps(payload, sort_keys=True)))

        edge_number = 0
        def edge(source: str, target: str, relation: str, kind: str) -> None:
            nonlocal edge_number
            edge_number += 1
            connection.execute("INSERT INTO edges VALUES (?, ?, ?, ?, ?)", (f"edge-{edge_number:06d}", source, target, relation, kind))

        anchor_number = 0
        def anchor(node_id: str, role: str, page: PageText, page_meta: Mapping[str, Any], text: str, offset: int) -> str:
            nonlocal anchor_number
            if not text:
                raise GraphBuildError(f"empty {role}")
            if page.cleaned_text[offset : offset + len(text)] != text:
                raise GraphBuildError(f"{role} cleaned offset does not match anchor quote")
            raw_offset, context_chars = _unique_raw_anchor(
                page.raw_text, page.cleaned_text, text, offset
            )
            anchor_number += 1
            anchor_id = f"{page.page_id}:anchor-{anchor_number:06d}"
            try:
                record = create_text_anchor(
                    anchor_id=anchor_id, page_id=page.page_id, raw_text=page.raw_text, cleaned_text=page.cleaned_text,
                    raw_char_start=raw_offset, raw_char_end=raw_offset + len(text), cleaned_char_start=offset,
                    cleaned_char_end=offset + len(text), source_line_start=_line_at(page.cleaned_text, offset),
                    source_line_end=_line_at(page.cleaned_text, offset + len(text) - 1),
                    printed_page_number=page_meta["printed_page_number"], source_pdf_page_number=page_meta["source_pdf_page_number"],
                    review_status=page_meta["review_status"],
                    context_chars=context_chars,
                )
            except SourceProvenanceError as error:
                raise GraphBuildError(f"{role} anchor cannot be replayed in raw and cleaned text: {error}") from error
            connection.execute("INSERT INTO anchors VALUES (?, ?, ?, ?)", (anchor_id, node_id, role, json.dumps(record, sort_keys=True)))
            return anchor_id

        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("schema_version", GRAPH_SCHEMA_VERSION))
        book = manifest["book"]
        book_id = book["book_id"]
        node(book_id, "Book", title=book["title"], edition=book["edition"], manifest_hash=manifest["content_sha256"])
        page_meta = {record["page_id"]: record for record in manifest["pages"]}
        chapter_id: str | None = None
        section_id: str | None = None
        item_id: str | None = None
        item_count = rule_count = table_count = formula_count = 0
        heading_ordinal = 0
        previous_chunk: str | None = None
        for page_record in manifest["pages"]:
            page_id = page_record["page_id"]
            page = pages[page_id]
            node(page_id, "Page", chapter_page_index=page_record["chapter_page_index"])
            edge(book_id, page_id, "BOOK_HAS_PAGE", STRUCTURAL)
            for chunk in sorted((c for c in manifest["chunks"] if c["page_id"] == page_id), key=lambda c: c["cleaned_char_start"]):
                chunk_id = chunk["chunk_id"]
                node(chunk_id, "EvidenceChunk", page_id=page_id, start=chunk["cleaned_char_start"], end=chunk["cleaned_char_end"])
                connection.execute(
                    "INSERT INTO chunk_text VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, page_id, chunk["cleaned_char_start"], chunk["cleaned_char_end"], page.cleaned_text[chunk["cleaned_char_start"]:chunk["cleaned_char_end"]]),
                )
                edge(chunk_id, page_id, "CHUNK_ON_PAGE", STRUCTURAL)
                if previous_chunk is not None:
                    edge(previous_chunk, chunk_id, "CHUNK_NEXT", STRUCTURAL)
                previous_chunk = chunk_id
            for match in re.finditer(r"(?m)^(#{1,6})\s+(.+?)\s*$", page.cleaned_text):
                level, title = len(match.group(1)), match.group(2).strip()
                heading_ordinal += 1
                location = f"p{page_record['chapter_page_index']:04d}-h{heading_ordinal:04d}"
                if level == 1:
                    chapter_id = f"{book_id}:chapter-{_safe_id(title)}:{location}"
                    node(chapter_id, "Chapter", title=title)
                    edge(book_id, chapter_id, "BOOK_HAS_CHAPTER", STRUCTURAL)
                    anchor(chapter_id, "chapter_title", page, page_meta[page_id], title, match.start(2))
                    section_id = None
                elif level >= 2:
                    if chapter_id is None:
                        continue
                    section_id = f"{chapter_id}:section-{_safe_id(title)}:{location}"
                    node(section_id, "Section", title=title)
                    edge(chapter_id, section_id, "CHAPTER_HAS_SECTION", STRUCTURAL)
                    anchor(section_id, "section_title", page, page_meta[page_id], title, match.start(2))
            for match in re.finditer(r"(?m)^Test Item:\s*(.+?)\s*$", page.cleaned_text):
                item_count += 1
                item_id = f"{book_id}:test-item-{item_count:04d}"
                label = match.group(1)
                node(item_id, "TestItem", label=label)
                parent = section_id or chapter_id or book_id
                edge(parent, item_id, "SECTION_HAS_TEST_ITEM" if section_id else "BOOK_HAS_TEST_ITEM", STRUCTURAL)
                anchor(item_id, "test_item", page, page_meta[page_id], label, match.start(1))
            self._add_labeled_evidence(connection, node, anchor, page, page_meta[page_id], item_id)
            rule_count = self._add_rules(connection, node, edge, anchor, page, page_meta[page_id], item_id, rule_count)
            table_count = self._add_tables(connection, node, anchor, page, page_meta[page_id], table_count)
            formula_count = self._add_formulas(connection, node, anchor, page, page_meta[page_id], formula_count)

    @staticmethod
    def _add_labeled_evidence(connection: sqlite3.Connection, node: Any, anchor: Any, page: PageText, page_meta: Mapping[str, Any], item_id: str | None) -> None:
        for label, node_type in (("Reference Range", "ReferenceRange"), ("Explanation", "Explanation")):
            for index, match in enumerate(re.finditer(rf"(?m)^{re.escape(label)}:\s*(.+?)\s*$", page.cleaned_text), 1):
                evidence_id = f"{page.page_id}:{_safe_id(label.lower())}-{index:04d}"
                node(evidence_id, node_type, text=match.group(1), test_item_id=item_id)
                anchor(evidence_id, label.lower().replace(" ", "_"), page, page_meta, match.group(1), match.start(1))

    @staticmethod
    def _add_rules(connection: sqlite3.Connection, node: Any, edge: Any, anchor: Any, page: PageText, page_meta: Mapping[str, Any], item_id: str | None, rule_count: int) -> int:
        for match in re.finditer(r"(?m)^Rule:\s*IF\s*(.*?)\s*THEN\s*(.*?)\s*$", page.cleaned_text):
            if item_id is None:
                continue
            condition, conclusion = match.group(1), match.group(2)
            if not condition:
                raise GraphBuildError("empty rule condition")
            if not conclusion:
                raise GraphBuildError("empty rule conclusion")
            parts = [part.strip() for part in re.split(r"\s+AND\s+", condition) if part.strip()]
            if not parts:
                raise GraphBuildError("empty rule condition")
            rule_count += 1
            rule_id = f"{item_id}:rule-{rule_count:04d}"
            version_id, expression_id = f"{rule_id}:version-1", f"{rule_id}:expression-1"
            node(rule_id, "KnowledgeRule", status="candidate")
            node(version_id, "RuleVersion", version="1", status="candidate")
            node(expression_id, "RuleExpression", operator="AND")
            edge(item_id, rule_id, "TEST_ITEM_HAS_RULE", SEMANTIC)
            edge(rule_id, version_id, "RULE_HAS_VERSION", SEMANTIC)
            edge(rule_id, expression_id, "RULE_HAS_EXPRESSION", SEMANTIC)
            cursor = match.start(1)
            for predicate_index, part in enumerate(parts, 1):
                offset = page.cleaned_text.find(part, cursor, match.end(1))
                if offset < 0:
                    raise GraphBuildError("rule condition cannot be anchored")
                predicate_id = f"{expression_id}:predicate-{predicate_index:04d}"
                node(predicate_id, "AtomicPredicate", text=part)
                predicate_anchor = anchor(predicate_id, "predicate", page, page_meta, part, offset)
                edge(expression_id, predicate_id, "EXPRESSION_HAS_PREDICATE", SEMANTIC)
                edge(predicate_id, page.page_id, "PREDICATE_SUPPORTED_BY", SEMANTIC)
                cursor = offset + len(part)
            conclusion_offset = match.start(2)
            conclusion_id = f"{rule_id}:conclusion"
            node(conclusion_id, "Conclusion", text=conclusion)
            anchor(conclusion_id, "conclusion", page, page_meta, conclusion, conclusion_offset)
            edge(conclusion_id, page.page_id, "CONCLUSION_SUPPORTED_BY", SEMANTIC)
            connection.execute("INSERT INTO rules VALUES (?, ?, ?, ?, NULL)", (rule_id, "candidate", version_id, expression_id))
        return rule_count

    @staticmethod
    def _add_tables(connection: sqlite3.Connection, node: Any, anchor: Any, page: PageText, page_meta: Mapping[str, Any], table_count: int) -> int:
        for match in re.finditer(r"(?is)<table\b[^>]*>.*?</table\s*>", page.cleaned_text):
            parser = _TableParser(match.group(0))
            if not parser.rows:
                continue
            table_count += 1
            table_id = f"{page.page_id}:table-{table_count:04d}"
            title = parser.caption
            node(table_id, "Table", title=title)
            for row_index, row in enumerate(parser.rows):
                for column_index, cell in enumerate(row):
                    offset = match.start() + cell.source_start
                    cell_id = f"{table_id}:cell-{row_index:04d}-{column_index:04d}"
                    node(cell_id, "TableCell", text=cell.display_text, row=row_index, column=column_index)
                    cell_anchor = anchor(cell_id, "table_cell", page, page_meta, cell.source_quote, offset)
                    connection.execute("INSERT INTO table_cells VALUES (?, ?, ?, ?, ?, ?, ?)", (table_id, row_index, column_index, int(cell.is_header), cell.display_text, cell_anchor, title))
        return table_count

    @staticmethod
    def _add_formulas(connection: sqlite3.Connection, node: Any, anchor: Any, page: PageText, page_meta: Mapping[str, Any], formula_count: int) -> int:
        matches = list(re.finditer(r"(?m)^Formula:\s*(.+?)\s*$", page.cleaned_text))
        matches.extend(re.finditer(r"(?s)\\\\\[\s*(.*?)\s*\\\\\]", page.cleaned_text))
        matches.extend(re.finditer(r"(?s)\$\$\s*(.*?)\s*\$\$", page.cleaned_text))
        for match in sorted(matches, key=lambda value: value.start()):
            formula_count += 1
            formula_id = f"{page.page_id}:formula-{formula_count:04d}"
            text = match.group(1)
            node(formula_id, "Formula", text=text)
            anchor(formula_id, "formula", page, page_meta, text, match.start(1))
        return formula_count


class KnowledgeGraph:
    """Read-only query helper, except explicit reviewed candidate approval."""

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def integrity_check(self) -> str:
        with self._connect() as connection:
            return str(connection.execute("PRAGMA integrity_check").fetchone()[0])

    def count_nodes(self, node_type: str) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM nodes WHERE node_type = ?", (node_type,)).fetchone()[0])

    def count_edges(self, edge_kind: str) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM edges WHERE edge_kind = ?", (edge_kind,)).fetchone()[0])

    def rules(self) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute("SELECT rule_id, status, version_id, expression_id, review_record FROM rules ORDER BY rule_id").fetchall()
            result = []
            for rule_id, status, version_id, expression_id, review_record in rows:
                anchors = connection.execute("SELECT role, anchor_id FROM anchors WHERE node_id IN (SELECT node_id FROM nodes WHERE node_id LIKE ?) ORDER BY anchor_id", (f"{expression_id}:predicate-%",)).fetchall()
                conclusion = connection.execute("SELECT anchor_id FROM anchors WHERE node_id = ?", (f"{rule_id}:conclusion",)).fetchone()
                result.append({"rule_id": rule_id, "status": status, "version_id": version_id, "expression_id": expression_id, "review_record": review_record, "condition_anchors": tuple(anchor_id for role, anchor_id in anchors if role == "predicate"), "conclusion_anchor": conclusion[0] if conclusion else None})
            return tuple(result)

    def table_cells(self) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute("SELECT table_id, row_index, column_index, is_header, text, anchor_id, table_title FROM table_cells ORDER BY table_id, row_index, column_index").fetchall()
        return tuple({"table_id": row[0], "row": row[1], "column": row[2], "is_header": bool(row[3]), "text": row[4], "anchor_id": row[5], "table_title": row[6]} for row in rows)

    def formulas(self) -> tuple[dict[str, Any], ...]:
        with self._connect() as connection:
            rows = connection.execute("SELECT payload FROM nodes WHERE node_type = 'Formula' ORDER BY node_id").fetchall()
        return tuple(json.loads(row[0]) for row in rows)

    def reconstruct_page(self, page_id: str) -> str:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT content FROM chunk_text WHERE page_id = ? ORDER BY start_offset", (page_id,)
            ).fetchall()
        if not rows:
            raise GraphBuildError("unknown page")
        return "".join(row[0] for row in rows)

    def set_rule_status(self, rule_id: str, status: str, review_record: str | None = None) -> None:
        if status != "approved" or not isinstance(review_record, str) or not review_record.strip():
            raise GraphBuildError("candidate approval requires an explicit review record")
        with self._connect() as connection:
            updated = connection.execute("UPDATE rules SET status = ?, review_record = ? WHERE rule_id = ? AND status = 'candidate'", (status, review_record, rule_id)).rowcount
            if updated != 1:
                raise GraphBuildError("rule is missing or is not a candidate")


def read_graph(database_path: Path) -> KnowledgeGraph:
    """Open an existing graph for read-only consumption and explicit review updates."""
    if not Path(database_path).is_file():
        raise GraphBuildError("knowledge graph database does not exist")
    return KnowledgeGraph(database_path)


__all__ = [
    "AtomicPredicate", "GRAPH_SCHEMA_VERSION", "GraphBuildError", "GraphBuildResult",
    "KnowledgeGraph", "KnowledgeGraphBuilder", "PageText", "RuleExpression", "RuleVersion",
    "read_graph",
]
