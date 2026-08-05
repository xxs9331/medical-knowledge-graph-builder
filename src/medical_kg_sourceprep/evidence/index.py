"""Validated evidence index construction and retrieval."""

from __future__ import annotations

import math
import os
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..provenance.package_validation import (
    ChunkPackageError,
    sha256_bytes,
    validate_chunk_package,
)
from ..graph.graph_retrieval import GraphRetrievalError, graph_query_diagnostic, graph_retrieve

INDEX_SCHEMA_VERSION = "evidence-index/v0.1"
MAX_QUERY_CHARS = 400


class QaError(ValueError):
    """Raised when a QA input cannot be safely validated or served."""


@dataclass(frozen=True, slots=True)
class ProvenanceContext:
    """Validated chunk-package facts used to verify index search locations."""

    chunks: dict[str, dict[str, Any]]


def _validate_package(package: Path) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    try:
        return validate_chunk_package(package)
    except ChunkPackageError as error:
        raise QaError(str(error)) from error



def build_evidence_index(package: Path, output: Path, generation_timestamp: str | None = None) -> dict[str, Any]:
    """Build one validated SQLite evidence graph without overwriting output."""
    if output.exists():
        raise QaError("output already exists; refusing to overwrite an index")
    if output.resolve() in package.resolve().parents or package.resolve() in output.resolve().parents:
        raise QaError("output must not overlap chunk package")
    manifest_bytes, manifest, chunks = _validate_package(package)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".sqlite", dir=output.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        connection = sqlite3.connect(temporary)
        with connection:
            connection.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE documents (document_id TEXT PRIMARY KEY, chapter_id TEXT NOT NULL);
                CREATE TABLE pages (page_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, chapter_page_index INTEGER NOT NULL, printed_page_number INTEGER NOT NULL, source_pdf_page_number INTEGER NOT NULL, review_status TEXT NOT NULL);
                CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, page_id TEXT NOT NULL, text TEXT NOT NULL, chunk_sha256 TEXT NOT NULL, cleaned_char_start INTEGER NOT NULL, cleaned_char_end INTEGER NOT NULL, review_status TEXT NOT NULL);
                CREATE TABLE edges (edge_id TEXT PRIMARY KEY, edge_type TEXT NOT NULL, from_id TEXT NOT NULL, to_id TEXT NOT NULL);
            """)
            metadata = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "chunk_manifest_sha256": sha256_bytes(manifest_bytes),
                "source_manifest_sha256": manifest["source_manifest_sha256"],
                "generation_timestamp": generation_timestamp or "unspecified",
                "answer_mode": "extractive",
            }
            connection.executemany("INSERT INTO metadata VALUES (?, ?)", sorted(metadata.items()))
            connection.execute("INSERT INTO documents VALUES (?, ?)", (manifest["document_id"], manifest["chapter_id"]))
            for page in manifest["pages"]:
                connection.execute("INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?)", (page["page_id"], manifest["document_id"], page["chapter_page_index"], page["printed_page_number"], page["source_pdf_page_number"], page.get("review_status", "unknown")))
                connection.execute("INSERT INTO edges VALUES (?, ?, ?, ?)", (f"document:{page['page_id']}", "DOCUMENT_HAS_PAGE", manifest["document_id"], page["page_id"]))
            previous: str | None = None
            for chunk in chunks:
                connection.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)", (chunk["chunk_id"], chunk["page_id"], chunk["text"], chunk["chunk_sha256"], chunk["cleaned_char_start"], chunk["cleaned_char_end"], chunk.get("review_status", "unknown")))
                connection.execute("INSERT INTO edges VALUES (?, ?, ?, ?)", (f"page:{chunk['chunk_id']}", "PAGE_HAS_CHUNK", chunk["page_id"], chunk["chunk_id"]))
                if previous is not None:
                    connection.execute("INSERT INTO edges VALUES (?, ?, ?, ?)", (f"next:{previous}:{chunk['chunk_id']}", "CHUNK_NEXT", previous, chunk["chunk_id"]))
                previous = chunk["chunk_id"]
        connection.close()
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"schema_version": INDEX_SCHEMA_VERSION, "document_count": 1, "page_count": len(manifest["pages"]), "chunk_count": len(chunks), "edge_count": len(manifest["pages"]) + len(chunks) + max(0, len(chunks) - 1), "chunk_manifest_sha256": sha256_bytes(manifest_bytes)}


def _terms(query: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", query).lower().strip()
    latin = re.findall(r"[a-z0-9]+", normalized)
    han = "".join(re.findall(r"[\u3400-\u9fff]", normalized))
    grams = [han[index:index + 2] for index in range(max(0, len(han) - 1))] or list(han)
    return sorted(set(latin + grams))


def query_index(
    index: Path, query: str, top_k: int = 5, provenance: ProvenanceContext | None = None
) -> list[dict[str, Any]]:
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
        raise QaError("query must be a non-empty bounded string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise QaError("top_k must be between 1 and 20")
    terms = _terms(query)
    if not terms:
        return []
    connection = sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT c.chunk_id, c.page_id, c.text, c.chunk_sha256, c.cleaned_char_start, "
            "c.cleaned_char_end, p.printed_page_number, p.source_pdf_page_number, "
            "p.chapter_page_index, c.review_status FROM chunks c "
            "JOIN pages p ON p.page_id=c.page_id ORDER BY c.chunk_id"
        ).fetchall()
    except sqlite3.Error as error:
        raise QaError("index is not a readable evidence index") from error
    finally:
        connection.close()
    document_frequency = Counter(term for term in terms for row in rows if term in unicodedata.normalize("NFKC", row["text"]).lower())
    hits = []
    normalized_query = unicodedata.normalize("NFKC", query).lower()
    for row in rows:
        text = unicodedata.normalize("NFKC", row["text"]).lower()
        matched = [term for term in terms if term in text]
        if not matched:
            continue
        term_score = sum(
            math.log((len(rows) + 1) / (document_frequency[term] + 1)) + 1
            for term in matched
        )
        exact_bonus = 1.0 if normalized_query in text else 0.0
        hits.append(
            {
                **dict(row),
                **_location(dict(row), provenance),
                "score": round(term_score + exact_bonus, 6),
                "matched_terms": matched,
                "score_components": {
                    "term_idf": round(term_score, 6),
                    "exact_substring_bonus": exact_bonus,
                },
                "retrieval_reason": "term_match" + ("+exact_query" if exact_bonus else ""),
            }
        )
    ranked = sorted(hits, key=lambda item: (-item["score"], item["chunk_id"]))[:top_k]
    if len(ranked) == top_k:
        return ranked

    selected_ids = {item["chunk_id"] for item in ranked}
    row_positions = {row["chunk_id"]: position for position, row in enumerate(rows)}
    for hit in ranked:
        position = row_positions[hit["chunk_id"]]
        for neighbor_position in (position - 1, position + 1):
            if not 0 <= neighbor_position < len(rows):
                continue
            neighbor = dict(rows[neighbor_position])
            if neighbor["chunk_id"] in selected_ids:
                continue
            neighbor.update(
                {
                    **_location(neighbor, provenance),
                    "score": 0.0,
                    "matched_terms": [],
                    "score_components": {"term_idf": 0.0, "exact_substring_bonus": 0.0},
                    "retrieval_reason": "adjacent_chunk_expansion",
                }
            )
            ranked.append(neighbor)
            selected_ids.add(neighbor["chunk_id"])
            if len(ranked) == top_k:
                return ranked
    return ranked


def query_index_with_graph(
    index: Path,
    knowledge_graph: Path | None,
    query: str,
    top_k: int = 5,
    provenance: ProvenanceContext | None = None,
    *,
    graph_query: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lexical = query_index(index, query, top_k, provenance)
    if knowledge_graph is None:
        return lexical, {"lexical": {"count": len(lexical)}, "graph": {"enabled": False, "count": 0}}
    try:
        diagnostic = graph_query_diagnostic(knowledge_graph, graph_query or query)
        graph_hits = graph_retrieve(
            knowledge_graph, index, graph_query or query, top_k=min(20, top_k * 3)
        )
    except GraphRetrievalError as error:
        raise QaError(str(error)) from error
    connection = sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = {
            row["chunk_id"]: dict(row)
            for row in connection.execute(
                "SELECT c.chunk_id, c.page_id, c.text, c.chunk_sha256, c.cleaned_char_start, "
                "c.cleaned_char_end, p.printed_page_number, p.source_pdf_page_number, "
                "p.chapter_page_index, c.review_status FROM chunks c "
                "JOIN pages p ON p.page_id=c.page_id ORDER BY c.chunk_id"
            )
        }
    except sqlite3.Error as error:
        raise QaError("index is not a readable evidence index") from error
    finally:
        connection.close()
    combined = {item["chunk_id"]: dict(item) for item in lexical}
    for hit in graph_hits:
        item = combined.get(hit.chunk_id)
        if item is None:
            row = rows.get(hit.chunk_id)
            if row is None:
                raise QaError("graph projected an unavailable evidence chunk")
            item = {
                **row,
                **_location(row, provenance),
                "score": 0.0,
                "matched_terms": [],
                "score_components": {"term_idf": 0.0, "exact_substring_bonus": 0.0},
                "retrieval_reason": "graph_path",
            }
            combined[hit.chunk_id] = item
        item["score_components"] = {**item["score_components"], "graph_path": hit.graph_score}
        item["score"] = round(float(item["score"]) + hit.graph_score * 2.0, 6)
        if "graph_path" not in item["retrieval_reason"]:
            item["retrieval_reason"] += "+graph_path"
        item["graph"] = {
            "status": hit.graph_status,
            "score": hit.graph_score,
            "path_relations": list(hit.path_relations),
            "matched_node_ids": list(hit.matched_node_ids),
            "matched_node_names": list(hit.matched_node_names),
            "path_node_ids": list(hit.path_node_ids),
            "path_node_names": list(hit.path_node_names),
            "path_node_types": list(hit.path_node_types),
            "path_triples": list(hit.path_triples),
            "match_mode": hit.match_mode,
        }
    ranked = sorted(combined.values(), key=lambda item: (-item["score"], item["chunk_id"]))[:top_k]
    channels = {
        "lexical": {"count": len(lexical)},
        "graph": {
            "enabled": True,
            "coverage": "chapter-01-only",
            "status": (
                graph_hits[0].graph_status if graph_hits
                else diagnostic.get("graph_status")
            ),
            "count": len(graph_hits),
            "returned_count": sum("graph" in item for item in ranked),
            "query_diagnostic": diagnostic,
        },
    }
    return ranked, channels


def _load_provenance(index: Path, package: Path) -> ProvenanceContext:
    manifest_bytes, manifest, chunks = _validate_package(package)
    if _index_meta(index).get("chunk_manifest_sha256") != sha256_bytes(manifest_bytes):
        raise QaError("chunk package manifest hash does not bind to index")
    pages = {page["page_id"]: page for page in manifest["pages"]}
    for page in pages.values():
        for field in ("source_line_start", "source_line_end"):
            if isinstance(page.get(field), bool) or not isinstance(page.get(field), int) or page[field] < 1:
                raise QaError("chunk package lacks valid source page line ranges")
        if page["source_line_end"] < page["source_line_start"]:
            raise QaError("chunk package source page line range is invalid")
    mapped: dict[str, dict[str, Any]] = {}
    by_page: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        by_page.setdefault(chunk["page_id"], []).append(chunk)
    for page_id, page_chunks in by_page.items():
        offset = 0
        next_line = 1
        for chunk in sorted(page_chunks, key=lambda item: item["cleaned_char_start"]):
            if chunk["cleaned_char_start"] != offset:
                raise QaError("chunk package offsets are not contiguous within a page")
            text = chunk["text"]
            newline_count = text.count("\n")
            mapped[chunk["chunk_id"]] = {
                "page_id": page_id,
                "chunk_sha256": chunk["chunk_sha256"],
                "cleaned_char_start": chunk["cleaned_char_start"],
                "cleaned_char_end": chunk["cleaned_char_end"],
                "chapter_page_index": chunk["chapter_page_index"],
                "printed_page_number": chunk["printed_page_number"],
                "source_pdf_page_number": chunk["source_pdf_page_number"],
                "review_status": chunk["review_status"],
                "exact_quote": text,
                "markdown_line_start": next_line,
                "markdown_line_end": next_line
                + newline_count
                - (1 if text.endswith("\n") else 0),
                "source_page_line_start": pages[page_id]["source_line_start"],
                "source_page_line_end": pages[page_id]["source_line_end"],
            }
            next_line += newline_count
            offset = chunk["cleaned_char_end"]
    if len(mapped) != len(chunks):
        raise QaError("chunk package provenance map is incomplete")
    return ProvenanceContext(mapped)


def _location(row: dict[str, Any], provenance: ProvenanceContext | None) -> dict[str, Any]:
    base = {
        "exact_quote": row["text"],
        "location_status": "unavailable",
        "markdown_line_start": None,
        "markdown_line_end": None,
        "source_page_line_start": None,
        "source_page_line_end": None,
    }
    if provenance is None:
        return base
    expected = provenance.chunks.get(row["chunk_id"])
    if expected is None:
        raise QaError("index chunk is absent from validated chunk package")
    for field in (
        "page_id",
        "chunk_sha256",
        "cleaned_char_start",
        "cleaned_char_end",
        "chapter_page_index",
        "printed_page_number",
        "source_pdf_page_number",
        "review_status",
    ):
        if row[field] != expected[field]:
            raise QaError("index location does not match validated chunk package")
    if row["text"] != expected["exact_quote"] or sha256_bytes(row["text"].encode("utf-8")) != row["chunk_sha256"]:
        raise QaError("index text does not match validated chunk package")
    return {**expected, "location_status": "verified"}


def _validate_index_bindings(index: Path, provenance: ProvenanceContext) -> None:
    connection = sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT c.chunk_id, c.page_id, c.text, c.chunk_sha256, "
            "c.cleaned_char_start, c.cleaned_char_end, c.review_status, "
            "p.chapter_page_index, p.printed_page_number, "
            "p.source_pdf_page_number FROM chunks c "
            "JOIN pages p ON p.page_id=c.page_id ORDER BY c.chunk_id"
        ).fetchall()
    except sqlite3.Error as error:
        raise QaError("index is not a readable evidence index") from error
    finally:
        connection.close()
    if len(rows) != len(provenance.chunks):
        raise QaError("index chunks do not match validated chunk package")
    for row in rows:
        _location(dict(row), provenance)


def _index_meta(index: Path) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True)
    try:
        return dict(connection.execute("SELECT key, value FROM metadata ORDER BY key"))
    finally:
        connection.close()
