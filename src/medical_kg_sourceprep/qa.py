"""Deterministic, provenance-bound local evidence retrieval and QA server."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, ProxyHandler, build_opener

from .analysis import AnalysisRule, analyze_report, result_to_dict
from .desktop_app import DesktopAppError, css as desktop_css, html as desktop_html
from .desktop_app import javascript as desktop_javascript
from .desktop_app import parse_report_payload
from .graph_retrieval import GraphRetrievalError, graph_query_diagnostic, graph_retrieve
from .paddleocr_report import (
    PaddleOcrJobsClient,
    PaddleOcrReportError,
    image_report_job,
)

CHUNK_SCHEMA_VERSION = "evidence-chunk-package/v0.1"
INDEX_SCHEMA_VERSION = "evidence-index/v0.1"
MAX_BODY_BYTES = 256 * 1024
MAX_OCR_BODY_BYTES = 14 * 1024 * 1024
MAX_QUERY_CHARS = 400


class QaError(ValueError):
    """Raised when a QA input cannot be safely validated or served."""


@dataclass(frozen=True, slots=True)
class ProvenanceContext:
    """Validated chunk-package facts used to verify index search locations."""

    chunks: dict[str, dict[str, Any]]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        value = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QaError(f"{label} must be readable UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise QaError(f"{label} must be an object")
    return raw, value


def _safe_child(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise QaError("chunk_path must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise QaError("chunk_path must be a safe relative path")
    result = (root / path).resolve()
    if root not in result.parents:
        raise QaError("chunk_path escapes chunk package")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QaError(f"{label} must be an integer")
    return value


def _validate_package(package: Path) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    root = package.resolve()
    if not root.is_dir():
        raise QaError("chunk package must be an existing directory")
    manifest_bytes, manifest = _read_json(root / "manifest.json", "chunk manifest")
    required = {"schema_version", "source_manifest_sha256", "document_id", "chapter_id", "page_count", "chunk_count", "pages", "chunks"}
    if not required <= set(manifest) or manifest["schema_version"] != CHUNK_SCHEMA_VERSION:
        raise QaError("unsupported or incomplete chunk manifest")
    if not all(isinstance(manifest[key], str) and manifest[key] for key in ("document_id", "chapter_id", "source_manifest_sha256")):
        raise QaError("chunk manifest identifiers are invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest["source_manifest_sha256"]):
        raise QaError("source manifest hash must be SHA-256")
    pages = manifest["pages"]
    chunks = manifest["chunks"]
    if not isinstance(pages, list) or not isinstance(chunks, list):
        raise QaError("chunk manifest pages and chunks must be lists")
    if _integer(manifest["page_count"], "page_count") != len(pages) or _integer(manifest["chunk_count"], "chunk_count") != len(chunks):
        raise QaError("chunk manifest counts do not match records")
    page_records: dict[str, dict[str, Any]] = {}
    for expected_index, page in enumerate(pages):
        if (
            not isinstance(page, dict)
            or not isinstance(page.get("page_id"), str)
            or not page["page_id"]
        ):
            raise QaError("page record is invalid")
        if page["page_id"] in page_records or _integer(page.get("chapter_page_index"), "chapter_page_index") != expected_index:
            raise QaError("page IDs must be unique and page indexes contiguous")
        page_records[page["page_id"]] = page
        for name in ("printed_page_number", "source_pdf_page_number"):
            _integer(page.get(name), name)
        if not isinstance(page.get("review_status"), str) or not page["review_status"]:
            raise QaError("page review status is invalid")
        if not isinstance(page.get("cleaned_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", page["cleaned_sha256"]
        ):
            raise QaError("page cleaned hash must be SHA-256")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise QaError("chunk record is invalid")
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
            raise QaError("chunk IDs must be non-empty and unique")
        seen.add(chunk_id)
        page = page_records.get(chunk.get("page_id"))
        if page is None or chunk.get("document_id") != manifest["document_id"]:
            raise QaError("chunk provenance does not bind to manifest")
        if chunk.get("chapter_id") != manifest["chapter_id"]:
            raise QaError("chunk chapter does not bind to manifest")
        if any(
            chunk.get(field) != page.get(field)
            for field in (
                "chapter_page_index",
                "printed_page_number",
                "source_pdf_page_number",
                "review_status",
            )
        ):
            raise QaError("chunk page mapping does not bind to manifest")
        if chunk.get("source_cleaned_sha256") != page.get("cleaned_sha256"):
            raise QaError("chunk source hash does not bind to manifest")
        if chunk.get("source_cleaned_path") != page.get("cleaned_path"):
            raise QaError("chunk source path does not bind to manifest")
        if chunk.get("source_page") != page:
            raise QaError("chunk source page does not bind to manifest")
        path = _safe_child(root, chunk.get("chunk_path"))
        try:
            content = path.read_bytes()
            text = content.decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise QaError("chunk file must be readable UTF-8") from error
        expected_hash = chunk.get("chunk_sha256")
        if (
            not isinstance(expected_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
            or _sha256(content) != expected_hash
        ):
            raise QaError("chunk file hash mismatch")
        if "\r" in text:
            raise QaError("chunk files must use LF line endings")
        start = _integer(chunk.get("cleaned_char_start"), "cleaned_char_start")
        end = _integer(chunk.get("cleaned_char_end"), "cleaned_char_end")
        if start < 0 or end <= start or end - start != len(text):
            raise QaError("chunk character offsets are invalid")
        if "char_count" in chunk and _integer(chunk["char_count"], "char_count") != len(text):
            raise QaError("chunk character count is invalid")
        record = dict(chunk)
        record["text"] = text
        validated.append(record)

    chunks_by_page: dict[str, list[dict[str, Any]]] = {
        page_id: [] for page_id in page_records
    }
    for chunk in validated:
        chunks_by_page[chunk["page_id"]].append(chunk)
    for page_id, page in page_records.items():
        offset = 0
        page_parts: list[str] = []
        for chunk in sorted(
            chunks_by_page[page_id], key=lambda item: item["cleaned_char_start"]
        ):
            if chunk["cleaned_char_start"] != offset:
                raise QaError("chunk offsets are not contiguous within a page")
            page_parts.append(chunk["text"])
            offset = chunk["cleaned_char_end"]
        if _sha256("".join(page_parts).encode("utf-8")) != page["cleaned_sha256"]:
            raise QaError("chunk reconstruction does not match cleaned page hash")
    return manifest_bytes, manifest, validated


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
                "chunk_manifest_sha256": _sha256(manifest_bytes),
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
    return {"schema_version": INDEX_SCHEMA_VERSION, "document_count": 1, "page_count": len(manifest["pages"]), "chunk_count": len(chunks), "edge_count": len(manifest["pages"]) + len(chunks) + max(0, len(chunks) - 1), "chunk_manifest_sha256": _sha256(manifest_bytes)}


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
    if _index_meta(index).get("chunk_manifest_sha256") != _sha256(manifest_bytes):
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
    if row["text"] != expected["exact_quote"] or _sha256(row["text"].encode("utf-8")) != row["chunk_sha256"]:
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


def _provider_config() -> tuple[str, str, str, float]:
    base_url = os.environ.get("MEDICAL_KG_QA_BASE_URL")
    api_key = os.environ.get("MEDICAL_KG_QA_API_KEY")
    model = os.environ.get("MEDICAL_KG_QA_MODEL")
    if not all((base_url, api_key, model)):
        raise QaError("openai-compatible mode requires explicit MEDICAL_KG_QA provider settings")
    try:
        timeout = float(os.environ.get("MEDICAL_KG_QA_TIMEOUT", "15"))
    except ValueError as error:
        raise QaError("MEDICAL_KG_QA_TIMEOUT must be numeric") from error
    if not 0 < timeout <= 60 or not base_url.startswith(("http://", "https://")):
        raise QaError("openai-compatible provider configuration is invalid")
    return base_url.rstrip("/"), api_key, model, timeout


def _model_answer(query: str, evidence: list[dict[str, Any]]) -> str:
    base_url, api_key, model, timeout = _provider_config()
    context = "\n".join(f"[{number}] {item['text']}" for number, item in enumerate(evidence, 1))
    payload = {"model": model, "messages": [{"role": "user", "content": f"Answer only from this evidence. Cite at least one bracketed evidence number.\nQuestion: {query}\nEvidence:\n{context}"}], "temperature": 0}
    request = Request(f"{base_url}/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        response = build_opener(ProxyHandler({})).open(request, timeout=timeout)
        with response:
            value = json.loads(response.read().decode("utf-8"))
        answer = value["choices"][0]["message"]["content"]
    except (URLError, OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise QaError("openai-compatible provider returned no usable answer") from error
    if not isinstance(answer, str) or not answer.strip() or not re.search(r"\[(?:[1-9][0-9]*)\]", answer):
        raise QaError("openai-compatible answer must cite supplied evidence")
    allowed = {str(number) for number in range(1, len(evidence) + 1)}
    if any(number not in allowed for number in re.findall(r"\[([1-9][0-9]*)\]", answer)):
        raise QaError("openai-compatible answer cited unavailable evidence")
    return answer.strip()


def _answer(
    index: Path, query: str, top_k: int, mode: str = "extractive",
    provenance: ProvenanceContext | None = None,
    knowledge_graph: Path | None = None,
) -> dict[str, Any]:
    evidence, channels = query_index_with_graph(index, knowledge_graph, query, top_k, provenance)
    if not evidence:
        return {"mode": mode, "answer": "未检索到足够证据。", "citations": [], "evidence": [], "channels": channels}
    sentences = []
    citations = []
    for number, item in enumerate(evidence, 1):
        sentence = re.split(r"(?<=[。！？.!?])\s*", item["text"].strip())[0]
        sentences.append(f"{sentence} [{number}]")
        citations.append({key: item[key] for key in ("chunk_id", "printed_page_number", "source_pdf_page_number", "chapter_page_index")})
    answer = " ".join(sentences) if mode == "extractive" else _model_answer(query, evidence)
    return {"mode": mode, "answer": answer, "citations": citations, "evidence": evidence, "channels": channels}


class _QaHandler(BaseHTTPRequestHandler):
    server_version = "LocalEvidenceQA/0.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    @property
    def qa_index(self) -> Path:
        return self.server.qa_index  # type: ignore[attr-defined]

    def _send(self, status: int, body: Any, content_type: str = "application/json; charset=utf-8") -> None:
        payload = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Content-Security-Policy", "default-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self._send(200, {"status": "ready", "mode": "extractive", "graph_enabled": self.server.qa_knowledge_graph is not None})  # type: ignore[attr-defined]
        elif self.path == "/api/meta":
            self._send(200, _index_meta(self.qa_index))
        elif self.path == "/":
            self._send(200, desktop_html(), "text/html; charset=utf-8")
        elif self.path == "/assets/app.js":
            self._send(200, desktop_javascript().encode(), "application/javascript; charset=utf-8")
        elif self.path == "/assets/app.css":
            self._send(200, desktop_css().encode(), "text/css; charset=utf-8")
        elif self.path == "/source.pdf" and self.server.qa_source_pdf is not None:  # type: ignore[attr-defined]
            try:
                self._send(200, self.server.qa_source_pdf.read_bytes(), "application/pdf")  # type: ignore[attr-defined]
            except OSError:
                self._send(404, {"error": "not_found"})
        else:
            self._send(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {
            "/api/search", "/api/answer", "/api/report-analysis",
            "/api/report-generation", "/api/report-ocr",
        }:
            self._send(404, {"error": "not_found"})
            return
        try:
            try:
                size = int(self.headers.get("Content-Length", "-1"))
            except ValueError as error:
                raise QaError("request body is invalid") from error
            body_limit = MAX_OCR_BODY_BYTES if self.path == "/api/report-ocr" else MAX_BODY_BYTES
            if size < 0 or size > body_limit:
                raise QaError("request body is invalid")
            value = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(value, dict):
                raise QaError("JSON body must be an object")
            if self.path == "/api/report-ocr":
                if set(value) != {"filename", "content_base64"}:
                    raise QaError("OCR request must contain filename and content_base64")
                filename = value.get("filename")
                encoded = value.get("content_base64")
                if not isinstance(filename, str) or not filename or not isinstance(encoded, str):
                    raise QaError("OCR request fields are invalid")
                try:
                    image = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as error:
                    raise QaError("content_base64 is invalid") from error
                try:
                    ocr_client = self.server.qa_ocr_client  # type: ignore[attr-defined]
                    if ocr_client is None:
                        ocr_client = PaddleOcrJobsClient.from_environment()
                    image_result, job = image_report_job(image, filename, client=ocr_client)
                except PaddleOcrReportError as error:
                    raise QaError(str(error)) from error
                result = {
                    "report": dict(image_result.report),
                    "job": {**job.summary(), "validation_model": "PP-OCRv6"},
                }
            elif self.path == "/api/report-generation":
                from .report_pipeline import ReportPipelineError, analyze_report_document

                try:
                    result = analyze_report_document(
                        value,
                        self.qa_index,
                        knowledge_graph=self.server.qa_knowledge_graph,  # type: ignore[attr-defined]
                        provenance=self.server.qa_provenance,  # type: ignore[attr-defined]
                        transport=self.server.qa_report_transport,  # type: ignore[attr-defined]
                    ).to_dict()
                except ReportPipelineError as error:
                    raise QaError(str(error)) from error
            elif self.path == "/api/report-analysis":
                report = parse_report_payload(value)
                analysis = analyze_report(
                    report,
                    self.server.qa_analysis_rules,  # type: ignore[attr-defined]
                    approved_book_registry=self.server.qa_book_registry,  # type: ignore[attr-defined]
                )
                result = {
                    "analysis": result_to_dict(analysis),
                    "rule_status": {
                        "approved_rule_count": sum(
                            rule.status == "approved" for rule in self.server.qa_analysis_rules  # type: ignore[attr-defined]
                        ),
                        "message": "暂无 approved 规则，未生成医学解释。"
                        if not self.server.qa_analysis_rules  # type: ignore[attr-defined]
                        else "规则已按确定性条件计算。",
                    },
                }
            else:
                query = value.get("query")
                top_k = value.get("top_k", 5)
                if self.path.endswith("search"):
                    evidence, channels = query_index_with_graph(
                        self.qa_index, self.server.qa_knowledge_graph, query, top_k,
                        self.server.qa_provenance,  # type: ignore[attr-defined]
                    )
                    result = {"evidence": evidence, "channels": channels}
                else:
                    result = _answer(
                        self.qa_index, query, top_k, self.server.qa_answer_mode,
                        self.server.qa_provenance, self.server.qa_knowledge_graph,  # type: ignore[attr-defined]
                    )
            self._send(200, result)
        except (UnicodeDecodeError, json.JSONDecodeError, QaError, DesktopAppError) as error:
            self._send(400, {"error": "invalid_request", "detail": str(error)})


def _index_meta(index: Path) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{index.resolve()}?mode=ro", uri=True)
    try:
        return dict(connection.execute("SELECT key, value FROM metadata ORDER BY key"))
    finally:
        connection.close()


def make_server(index: Path, host: str = "127.0.0.1", port: int = 18852, answer_mode: str = "extractive", *, analysis_rules: tuple[AnalysisRule, ...] = (), approved_book_registry: dict[str, dict[str, Any]] | None = None, source_pdf_path: Path | None = None, chunk_package: Path | None = None, knowledge_graph: Path | None = None, report_transport: Any | None = None, ocr_client: PaddleOcrJobsClient | None = None, allow_lan: bool = False) -> ThreadingHTTPServer:
    if answer_mode not in {"extractive", "openai-compatible"}:
        raise QaError("unsupported answer mode")
    if answer_mode == "openai-compatible":
        _provider_config()
    if host not in {"127.0.0.1", "::1", "localhost"} and not allow_lan:
        raise QaError("non-loopback binding requires explicit allow_lan")
    if allow_lan and host not in {"0.0.0.0", "::"}:
        raise QaError("LAN mode must bind an unspecified interface address")
    _index_meta(index)
    if source_pdf_path is not None and (not source_pdf_path.is_file() or source_pdf_path.suffix.lower() != ".pdf"):
        raise QaError("source PDF must be an existing local PDF file")
    provenance = _load_provenance(index, chunk_package) if chunk_package is not None else None
    if provenance is not None:
        _validate_index_bindings(index, provenance)
    if knowledge_graph is not None:
        if not knowledge_graph.is_file():
            raise QaError("knowledge graph must be an existing SQLite file")
        try:
            graph_retrieve(knowledge_graph, index, "__startup_validation__", top_k=1)
        except GraphRetrievalError as error:
            raise QaError(str(error)) from error
    server = ThreadingHTTPServer((host, port), _QaHandler)
    server.qa_index = index.resolve()  # type: ignore[attr-defined]
    server.qa_answer_mode = answer_mode  # type: ignore[attr-defined]
    server.qa_analysis_rules = tuple(analysis_rules)  # type: ignore[attr-defined]
    server.qa_book_registry = approved_book_registry  # type: ignore[attr-defined]
    server.qa_report_transport = report_transport  # type: ignore[attr-defined]
    server.qa_ocr_client = ocr_client  # type: ignore[attr-defined]
    server.qa_source_pdf = source_pdf_path.resolve() if source_pdf_path else None  # type: ignore[attr-defined]
    server.qa_provenance = provenance  # type: ignore[attr-defined]
    server.qa_knowledge_graph = knowledge_graph.resolve() if knowledge_graph else None  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-evidence-index")
    build.add_argument("--chunk-package", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--generation-timestamp")
    serve = commands.add_parser("serve-qa")
    serve.add_argument("--index", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=18852)
    serve.add_argument("--answer-mode", default="extractive", choices=("extractive", "openai-compatible"))
    serve.add_argument("--source-pdf", type=Path)
    serve.add_argument("--chunk-package", type=Path)
    serve.add_argument("--knowledge-graph", type=Path)
    serve.add_argument("--allow-lan", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "build-evidence-index":
            print(json.dumps(build_evidence_index(args.chunk_package, args.output, args.generation_timestamp), ensure_ascii=False))
        else:
            make_server(args.index, args.host, args.port, args.answer_mode, source_pdf_path=args.source_pdf, chunk_package=args.chunk_package, knowledge_graph=args.knowledge_graph, allow_lan=args.allow_lan).serve_forever()
    except QaError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
