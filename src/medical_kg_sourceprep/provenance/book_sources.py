"""Versioned provenance contracts for book-derived source packages.

The module consumes metadata and text supplied by callers.  It never opens a
PDF, performs OCR, or stores source text in a manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

BOOK_MANIFEST_SCHEMA_VERSION = "book-source-manifest/v0.2"
ANCHOR_SCHEMA_VERSION = "text-anchor/v0.2"
UNAVAILABLE = "unavailable"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVIEW_STATUSES = frozenset(
    {"verified against source PDF", "accepted from upstream page markers", "unreviewed"}
)


class SourceProvenanceError(ValueError):
    """Raised when source provenance cannot be established conservatively."""


@dataclass(frozen=True)
class SourceBook:
    book_id: str
    title: str
    edition: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SourcePdf:
    pdf_id: str
    locator: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SourceMarkdown:
    markdown_id: str
    locator: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CleanedPage:
    page_id: str
    chapter_page_index: int
    raw_path: str
    cleaned_path: str
    raw_sha256: str
    cleaned_sha256: str
    source_line_start: int | None
    source_line_end: int | None
    printed_page_number: int | None
    source_pdf_page_number: int | None
    review_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TextAnchor:
    anchor_id: str
    page_id: str
    raw_char_start: int
    raw_char_end: int
    cleaned_char_start: int
    cleaned_char_end: int
    exact_quote: str
    prefix: str
    suffix: str
    raw_sha256: str
    cleaned_sha256: str
    source_line_start: int | None
    source_line_end: int | None
    printed_page_number: int | None
    source_pdf_page_number: int | None
    pdf_bbox: str
    review_status: str
    schema_version: str = ANCHOR_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_id(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._-]*", value)
    ):
        raise SourceProvenanceError(f"{name} must be a stable non-empty identifier")
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise SourceProvenanceError(f"{name} must be a lowercase SHA-256")
    return value


def _require_span(name: str, start: object, end: object, length: int) -> tuple[int, int]:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)):
        raise SourceProvenanceError(f"{name} character offsets must be integers")
    if start < 0 or end <= start or end > length:
        raise SourceProvenanceError(f"{name} character offsets are outside the source text")
    return start, end


def _require_page_semantics(record: Mapping[str, Any]) -> None:
    line_start = record.get("source_line_start")
    line_end = record.get("source_line_end")
    if (line_start is None) != (line_end is None):
        raise SourceProvenanceError("source line range must be complete or unavailable")
    if line_start is not None and (
        isinstance(line_start, bool)
        or isinstance(line_end, bool)
        or not isinstance(line_start, int)
        or not isinstance(line_end, int)
        or line_start < 1
        or line_end < line_start
    ):
        raise SourceProvenanceError("source line range must be positive integers or unavailable")
    if line_start is not None and line_end < line_start:
        raise SourceProvenanceError("source line range is invalid")
    for field in ("printed_page_number", "source_pdf_page_number"):
        value = record.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise SourceProvenanceError(f"{field} must be a positive integer or null")
    if record.get("review_status") not in _REVIEW_STATUSES:
        raise SourceProvenanceError("review_status is not an allowed conservative status")


def _candidates(text: str, quote: str, prefix: str, suffix: str) -> list[int]:
    starts: list[int] = []
    start = 0
    while True:
        index = text.find(quote, start)
        if index < 0:
            return starts
        if text[:index].endswith(prefix) and text[index + len(quote) :].startswith(suffix):
            starts.append(index)
        start = index + 1


def create_text_anchor(
    *,
    anchor_id: str,
    page_id: str,
    raw_text: str,
    cleaned_text: str,
    raw_char_start: int,
    raw_char_end: int,
    cleaned_char_start: int,
    cleaned_char_end: int,
    source_line_start: int | None,
    source_line_end: int | None,
    printed_page_number: int | None,
    source_pdf_page_number: int | None,
    review_status: str,
    context_chars: int = 32,
) -> dict[str, Any]:
    """Create a dual-offset anchor; both source versions must agree on the quote."""
    _require_id("anchor_id", anchor_id)
    _require_id("page_id", page_id)
    if not isinstance(raw_text, str) or not isinstance(cleaned_text, str):
        raise SourceProvenanceError("raw_text and cleaned_text must be strings")
    if isinstance(context_chars, bool) or not isinstance(context_chars, int) or context_chars < 0:
        raise SourceProvenanceError("context_chars must be a non-negative integer")
    raw_start, raw_end = _require_span("raw", raw_char_start, raw_char_end, len(raw_text))
    clean_start, clean_end = _require_span(
        "cleaned", cleaned_char_start, cleaned_char_end, len(cleaned_text)
    )
    quote = cleaned_text[clean_start:clean_end]
    if raw_text[raw_start:raw_end] != quote:
        raise SourceProvenanceError("raw and cleaned offsets must identify the same exact quote")
    record = TextAnchor(
        anchor_id=anchor_id,
        page_id=page_id,
        raw_char_start=raw_start,
        raw_char_end=raw_end,
        cleaned_char_start=clean_start,
        cleaned_char_end=clean_end,
        exact_quote=quote,
        prefix=cleaned_text[max(0, clean_start - context_chars) : clean_start],
        suffix=cleaned_text[clean_end : clean_end + context_chars],
        raw_sha256=_sha256(raw_text),
        cleaned_sha256=_sha256(cleaned_text),
        source_line_start=source_line_start,
        source_line_end=source_line_end,
        printed_page_number=printed_page_number,
        source_pdf_page_number=source_pdf_page_number,
        pdf_bbox=UNAVAILABLE,
        review_status=review_status,
    ).to_dict()
    validate_text_anchor(record, raw_text, cleaned_text)
    return record


def validate_text_anchor(anchor: Mapping[str, Any], raw_text: str, cleaned_text: str) -> None:
    """Validate hashes, offsets, context and page semantics, failing closed on drift."""
    required = set(TextAnchor.__dataclass_fields__)
    if not isinstance(anchor, Mapping) or set(anchor) != required:
        raise SourceProvenanceError("text anchor has missing or unknown fields")
    if anchor["schema_version"] != ANCHOR_SCHEMA_VERSION:
        raise SourceProvenanceError("unsupported text anchor schema version")
    _require_id("anchor_id", anchor["anchor_id"])
    _require_id("page_id", anchor["page_id"])
    _require_page_semantics(anchor)
    if anchor["pdf_bbox"] != UNAVAILABLE:
        raise SourceProvenanceError(
            "PDF bounding boxes must be unavailable unless a future schema defines them"
        )
    if (
        _sha256(raw_text) != anchor["raw_sha256"]
        or _sha256(cleaned_text) != anchor["cleaned_sha256"]
    ):
        raise SourceProvenanceError("text anchor hash mismatch")
    raw_start, raw_end = _require_span(
        "raw", anchor["raw_char_start"], anchor["raw_char_end"], len(raw_text)
    )
    clean_start, clean_end = _require_span(
        "cleaned", anchor["cleaned_char_start"], anchor["cleaned_char_end"], len(cleaned_text)
    )
    quote = anchor["exact_quote"]
    if not isinstance(quote, str) or not quote:
        raise SourceProvenanceError("exact_quote must be a non-empty string")
    if raw_text[raw_start:raw_end] != quote or cleaned_text[clean_start:clean_end] != quote:
        raise SourceProvenanceError("text anchor offsets no longer match exact quote")
    if not isinstance(anchor["prefix"], str) or not isinstance(anchor["suffix"], str):
        raise SourceProvenanceError("text anchor context must be strings")
    for text in (raw_text, cleaned_text):
        candidates = _candidates(text, quote, anchor["prefix"], anchor["suffix"])
        if len(candidates) != 1:
            raise SourceProvenanceError(
                "text anchor quote is ambiguous or context no longer matches"
            )


def replay_text_anchor(anchor: Mapping[str, Any], raw_text: str, cleaned_text: str) -> str:
    """Return the verified quote after replaying both text locations."""
    validate_text_anchor(anchor, raw_text, cleaned_text)
    return str(anchor["exact_quote"])


def _source_record(
    record_type: type[SourceBook] | type[SourcePdf] | type[SourceMarkdown],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        record = record_type(**dict(value))
    except TypeError as error:
        raise SourceProvenanceError(f"invalid {record_type.__name__} fields") from error
    data = record.to_dict()
    for field, item in data.items():
        if field.endswith("_id"):
            _require_id(field, item)
        elif field == "sha256":
            _require_sha256(field, item)
        elif not isinstance(item, str) or not item:
            raise SourceProvenanceError(f"{field} must be a non-empty string")
    return data


def _page_record(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        record = CleanedPage(**dict(value))
    except TypeError as error:
        raise SourceProvenanceError("invalid CleanedPage fields") from error
    data = record.to_dict()
    _require_id("page_id", data["page_id"])
    if isinstance(data["chapter_page_index"], bool) or not isinstance(
        data["chapter_page_index"], int
    ):
        raise SourceProvenanceError("chapter_page_index must be an integer")
    for field in ("raw_path", "cleaned_path"):
        if not isinstance(data[field], str) or not data[field]:
            raise SourceProvenanceError(f"{field} must be a non-empty string")
    _require_sha256("raw_sha256", data["raw_sha256"])
    _require_sha256("cleaned_sha256", data["cleaned_sha256"])
    _require_page_semantics(data)
    return data


def build_book_manifest(
    *,
    book: Mapping[str, Any],
    pdf: Mapping[str, Any],
    markdown: Mapping[str, Any],
    pages: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic manifest from existing page and chunk metadata."""
    page_records = [_page_record(page) for page in pages]
    if [page["chapter_page_index"] for page in page_records] != list(range(len(page_records))):
        raise SourceProvenanceError("chapter_page_index values must be contiguous from zero")
    if len({page["page_id"] for page in page_records}) != len(page_records):
        raise SourceProvenanceError("page_id values must be unique")
    page_ids = {page["page_id"] for page in page_records}
    chunk_records: list[dict[str, Any]] = []
    for chunk in chunks:
        record = dict(chunk)
        required = {"chunk_id", "page_id", "cleaned_char_start", "cleaned_char_end", "chunk_sha256"}
        if set(record) != required:
            raise SourceProvenanceError(
                "chunk records must contain only their stable provenance fields"
            )
        _require_id("chunk_id", record["chunk_id"])
        _require_id("page_id", record["page_id"])
        if record["page_id"] not in page_ids:
            raise SourceProvenanceError("chunk references an unknown page_id")
        offsets = ("cleaned_char_start", "cleaned_char_end")
        if (
            any(
                isinstance(record[field], bool) or not isinstance(record[field], int)
                for field in offsets
            )
            or record["cleaned_char_start"] < 0
            or record["cleaned_char_end"] <= record["cleaned_char_start"]
        ):
            raise SourceProvenanceError("chunk character range is invalid")
        _require_sha256("chunk_sha256", record["chunk_sha256"])
        chunk_records.append(record)
    if len({chunk["chunk_id"] for chunk in chunk_records}) != len(chunk_records):
        raise SourceProvenanceError("chunk_id values must be unique")
    manifest = {
        "schema_version": BOOK_MANIFEST_SCHEMA_VERSION,
        "book": _source_record(SourceBook, book),
        "pdf": _source_record(SourcePdf, pdf),
        "markdown": _source_record(SourceMarkdown, markdown),
        "pages": page_records,
        "chunks": chunk_records,
    }
    manifest["content_sha256"] = _canonical_sha256(manifest)
    return manifest


def build_book_manifest_from_packages(
    *,
    book: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    chunk_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt existing v0.1 source/chunk manifest records without opening their files."""
    required_source = {
        "document_id", "source_pdf_locator", "source_pdf_sha256", "input_path_locator",
        "input_sha256", "pages",
    }
    if not isinstance(source_manifest, Mapping) or not required_source <= set(source_manifest):
        raise SourceProvenanceError("source manifest lacks fields required for v0.2 adaptation")
    if not isinstance(chunk_manifest, Mapping) or not isinstance(
        chunk_manifest.get("chunks"), list
    ):
        raise SourceProvenanceError("chunk manifest must provide a chunk list")
    document_id = source_manifest["document_id"]
    _require_id("document_id", document_id)
    pages: list[dict[str, Any]] = []
    for page in source_manifest["pages"]:
        if not isinstance(page, Mapping):
            raise SourceProvenanceError("source manifest page must be an object")
        pages.append({
            "page_id": page.get("page_id"),
            "chapter_page_index": page.get("chapter_page_index"),
            "raw_path": page.get("raw_path"),
            "cleaned_path": page.get("cleaned_path"),
            "raw_sha256": page.get("raw_sha256"),
            "cleaned_sha256": page.get("cleaned_sha256"),
            "source_line_start": page.get("source_line_start"),
            "source_line_end": page.get("source_line_end"),
            "printed_page_number": page.get("printed_page_number"),
            "source_pdf_page_number": page.get("source_pdf_page_number"),
            "review_status": page.get("review_status"),
        })
    chunks = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "page_id": chunk.get("page_id"),
            "cleaned_char_start": chunk.get("cleaned_char_start"),
            "cleaned_char_end": chunk.get("cleaned_char_end"),
            "chunk_sha256": chunk.get("chunk_sha256"),
        }
        for chunk in chunk_manifest["chunks"]
        if isinstance(chunk, Mapping)
    ]
    if len(chunks) != len(chunk_manifest["chunks"]):
        raise SourceProvenanceError("chunk manifest chunk must be an object")
    return build_book_manifest(
        book=book,
        pdf={
            "pdf_id": f"{document_id}:pdf",
            "locator": source_manifest["source_pdf_locator"],
            "sha256": source_manifest["source_pdf_sha256"],
        },
        markdown={
            "markdown_id": f"{document_id}:markdown",
            "locator": source_manifest["input_path_locator"],
            "sha256": source_manifest["input_sha256"],
        },
        pages=pages,
        chunks=chunks,
    )


def validate_book_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the manifest's version, record semantics and deterministic hash chain."""
    expected = {"schema_version", "book", "pdf", "markdown", "pages", "chunks", "content_sha256"}
    if not isinstance(manifest, Mapping) or set(manifest) != expected:
        raise SourceProvenanceError("book manifest has missing or unknown fields")
    if manifest["schema_version"] != BOOK_MANIFEST_SCHEMA_VERSION:
        raise SourceProvenanceError("unsupported book manifest schema version")
    rebuilt = build_book_manifest(
        book=manifest["book"], pdf=manifest["pdf"], markdown=manifest["markdown"],
        pages=manifest["pages"], chunks=manifest["chunks"],
    )
    if manifest["content_sha256"] != rebuilt["content_sha256"]:
        raise SourceProvenanceError("book manifest content hash mismatch")
