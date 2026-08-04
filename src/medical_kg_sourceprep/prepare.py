"""Build a deterministic, page-aware source package from existing Markdown."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "source-package/v0.1"
TOOL_VERSION = "0.1.0"
PAGE_MAP_SCHEMA_VERSION = "source-page-map/v0.1"
PAGE_MAP_TOP_LEVEL_FIELDS = {"schema_version", "input_sha256", "page_count", "pages"}
PAGE_MAP_PAGE_FIELDS = {
    "chapter_page_index",
    "printed_page_number",
    "source_pdf_page_number",
    "start_line",
    "end_line",
    "review_status",
}
VERIFIED_REVIEW_STATUS = "verified against source PDF"
UPSTREAM_MARKERS_REVIEW_STATUS = "accepted from upstream page markers"
ACCEPTED_REVIEW_STATUSES = frozenset({VERIFIED_REVIEW_STATUS, UPSTREAM_MARKERS_REVIEW_STATUS})


class PreparationError(ValueError):
    """Raised when provenance cannot be established conservatively."""


def _natural_key(path: Path) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name))


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _input_sha256(path: Path) -> str:
    """Hash the caller's exact input bytes, including directory layout boundaries."""
    if path.is_file():
        return _sha256(path.read_bytes())
    digest = hashlib.sha256()
    for page in sorted(path.glob("*.md"), key=_natural_key):
        relative_name = page.relative_to(path).as_posix().encode("utf-8")
        content = page.read_bytes()
        for value in (relative_name, content):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def _normalise(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"


def _validate_identifier(name: str, value: str) -> None:
    if not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise PreparationError(f"invalid {name}; use letters, numbers, dot, underscore, or hyphen")


def _edge_number_indexes(lines: list[str]) -> set[int]:
    nonblank = [index for index, line in enumerate(lines) if line.strip()]
    candidates = nonblank[:2] + nonblank[-2:]
    return {index for index in candidates if re.fullmatch(r"\d{1,5}", lines[index].strip())}


def _clean_pages(raw_pages: list[str]) -> list[str]:
    split_pages = [page.rstrip("\n").split("\n") for page in raw_pages]
    first_last: Counter[str] = Counter()
    for lines in split_pages:
        nonblank = [line.strip() for line in lines if line.strip()]
        if nonblank:
            first_last.update({nonblank[0], nonblank[-1]})
    repeated_furniture = {line for line, count in first_last.items() if count >= 2 and not re.fullmatch(r"\d+", line)}
    cleaned: list[str] = []
    for lines in split_pages:
        remove = _edge_number_indexes(lines)
        nonblank = [index for index, line in enumerate(lines) if line.strip()]
        for index in (nonblank[:1] + nonblank[-1:]):
            if lines[index].strip() in repeated_furniture:
                remove.add(index)
        kept = [line for index, line in enumerate(lines) if index not in remove]
        cleaned.append(_normalise("\n".join(kept)))
    return cleaned


def _read_directory(path: Path, page_count: int) -> list[str]:
    pages = sorted(path.glob("*.md"), key=_natural_key)
    if len(pages) != page_count:
        raise PreparationError(f"expected {page_count} Markdown pages, found {len(pages)}")
    if not pages:
        raise PreparationError("input directory contains no Markdown pages")
    return [_normalise(page.read_text(encoding="utf-8")) for page in pages]


def _read_combined(path: Path, printed_page_start: int, page_count: int) -> list[str]:
    lines = _normalise(path.read_text(encoding="utf-8")).splitlines()
    expected = list(range(printed_page_start, printed_page_start + page_count))
    marker_indexes: dict[int, list[int]] = {number: [] for number in expected}
    for index, line in enumerate(lines):
        if re.fullmatch(r"\d{1,5}", line.strip()):
            number = int(line.strip())
            if number in marker_indexes:
                marker_indexes[number].append(index)
    if any(len(marker_indexes[number]) != 1 for number in expected):
        raise PreparationError("combined input needs exactly one standalone marker for every expected printed page")
    markers = [marker_indexes[number][0] for number in expected]
    if markers != sorted(markers):
        raise PreparationError("combined page markers are not monotonic")
    header_starts = _repeated_header_starts(lines, page_count)
    if header_starts is not None:
        segments = [lines[start:end] for start, end in zip(header_starts, header_starts[1:] + [len(lines)])]
        pages = _pages_from_segments(segments, expected)
        if pages is not None:
            return pages
    if any(line.strip() for line in lines[: markers[0]]):
        raise PreparationError("content before the first page marker lacks an unambiguous page boundary")
    if any(right - left < 2 for left, right in pairwise(markers)):
        raise PreparationError("combined page markers do not provide unambiguous boundaries")
    pages: list[str] = []
    for index, marker in enumerate(markers):
        start = marker + 1
        end = markers[index + 1] if index + 1 < len(markers) else len(lines)
        content = lines[start:end]
        if not any(line.strip() for line in content):
            raise PreparationError("combined page boundary has no page content")
        pages.append(_normalise("\n".join(content)))
    return pages


def _parse_page_map(page_map_path: Path) -> tuple[dict[str, Any], bytes]:
    if not page_map_path.is_file():
        raise PreparationError("page map must be a JSON file")
    page_map_bytes = page_map_path.read_bytes()
    try:
        page_map = json.loads(page_map_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparationError("page map must be valid UTF-8 JSON") from error
    if not isinstance(page_map, dict) or set(page_map) != PAGE_MAP_TOP_LEVEL_FIELDS:
        raise PreparationError("page map has missing or unknown top-level fields")
    return page_map, page_map_bytes


def _page_map_pages(
    page_map_path: Path,
    input_bytes: bytes,
    printed_page_start: int,
    source_pdf_page_start: int,
    page_count: int,
) -> tuple[list[str], list[dict[str, Any]], bytes]:
    page_map, page_map_bytes = _parse_page_map(page_map_path)
    input_hash = _sha256(input_bytes)
    if page_map["schema_version"] != PAGE_MAP_SCHEMA_VERSION:
        raise PreparationError("unsupported page map schema version")
    if page_map["input_sha256"] != input_hash:
        raise PreparationError("page map input_sha256 does not match exact input bytes")
    if page_map["page_count"] != page_count or not isinstance(page_map["page_count"], int):
        raise PreparationError("page map page_count does not match CLI page count")
    pages = page_map["pages"]
    if not isinstance(pages, list) or len(pages) != page_count:
        raise PreparationError("page map pages must match CLI page count")
    try:
        lines = _normalise(input_bytes.decode("utf-8")).splitlines()
    except UnicodeDecodeError as error:
        raise PreparationError("combined Markdown must be UTF-8") from error
    if not lines:
        raise PreparationError("combined Markdown has no logical lines")

    raw_pages: list[str] = []
    expected_start = 1
    for index, page in enumerate(pages):
        if not isinstance(page, dict) or set(page) != PAGE_MAP_PAGE_FIELDS:
            raise PreparationError("page map has missing or unknown page fields")
        if any(isinstance(page[field], bool) or not isinstance(page[field], int) for field in PAGE_MAP_PAGE_FIELDS - {"review_status"}):
            raise PreparationError("page map numeric fields must be integers")
        if page["review_status"] not in ACCEPTED_REVIEW_STATUSES:
            raise PreparationError(
                "page map review_status must be verified against source PDF "
                "or accepted from upstream page markers"
            )
        if page["chapter_page_index"] != index:
            raise PreparationError("page map chapter page indexes must match the CLI sequence")
        if page["printed_page_number"] != printed_page_start + index:
            raise PreparationError("page map printed page numbers must match the CLI sequence")
        if page["source_pdf_page_number"] != source_pdf_page_start + index:
            raise PreparationError("page map source PDF page numbers must match the CLI sequence")
        start_line = page["start_line"]
        end_line = page["end_line"]
        if start_line != expected_start or end_line < start_line or end_line > len(lines):
            raise PreparationError("page map line ranges must form an exact contiguous partition")
        page_lines = lines[start_line - 1 : end_line]
        if not any(line.strip() for line in page_lines):
            raise PreparationError("page map cannot define an empty page")
        marker_indexes = [line_index for line_index, line in enumerate(page_lines) if line.strip() == str(page["printed_page_number"])]
        edge_indexes = _edge_number_indexes(page_lines)
        if len(marker_indexes) != 1 or marker_indexes[0] not in edge_indexes:
            raise PreparationError("page map page range needs one edge printed-page marker")
        raw_pages.append(_normalise("\n".join(page_lines)))
        expected_start = end_line + 1
    if expected_start != len(lines) + 1:
        raise PreparationError("page map leaves input lines unassigned")
    return raw_pages, pages, page_map_bytes


def _repeated_header_starts(lines: list[str], page_count: int) -> list[int] | None:
    """Find a single repeated non-numeric line that proves each page start."""
    occurrences: dict[str, list[int]] = {}
    for index, line in enumerate(lines):
        text = line.strip()
        if text and not re.fullmatch(r"\d+", text):
            occurrences.setdefault(text, []).append(index)
    candidates = [indexes for indexes in occurrences.values() if len(indexes) == page_count and indexes[0] == 0]
    return candidates[0] if len(candidates) == 1 else None


def _pages_from_segments(segments: list[list[str]], expected: list[int]) -> list[str] | None:
    pages: list[str] = []
    for segment, number in zip(segments, expected, strict=True):
        markers = [index for index, line in enumerate(segment) if line.strip() == str(number)]
        if len(markers) != 1:
            return None
        marker = markers[0]
        nonblank = [index for index, line in enumerate(segment) if line.strip()]
        if marker not in nonblank[:2] + nonblank[-2:]:
            return None
        page = segment[:marker] + segment[marker + 1 :]
        if not any(line.strip() for line in page):
            return None
        pages.append(_normalise("\n".join(page)))
    return pages


def _validate_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise PreparationError(f"input does not exist: {input_path}")
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_output == resolved_input or resolved_input in resolved_output.parents:
        raise PreparationError("output path must not overlap the input path")
    if output_path.exists():
        raise PreparationError("output path already exists; refusing to overwrite a source package")


def prepare_source(
    input_path: Path,
    output_path: Path,
    document_id: str,
    chapter_id: str,
    ocr_engine: str,
    source_pdf_locator: str,
    source_pdf_sha256: str,
    printed_page_start: int,
    source_pdf_page_start: int,
    page_count: int,
    page_map_path: Path | None = None,
    generation_timestamp: str | None = None,
) -> dict[str, Any]:
    """Prepare raw and cleaned page artifacts, refusing uncertain provenance."""
    _validate_identifier("document_id", document_id)
    _validate_identifier("chapter_id", chapter_id)
    if not ocr_engine or not source_pdf_locator:
        raise PreparationError("OCR engine and source PDF locator are required")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", source_pdf_sha256):
        raise PreparationError("source_pdf_sha256 must be a SHA-256 hex digest")
    if min(printed_page_start, source_pdf_page_start, page_count) < 1:
        raise PreparationError("page starts and page count must be positive")
    _validate_paths(input_path, output_path)
    page_map_bytes: bytes | None = None
    page_map_pages: list[dict[str, Any]] | None = None
    input_bytes: bytes | None = None
    if input_path.is_dir():
        if page_map_path is not None:
            raise PreparationError("page map is valid only with one combined Markdown input file")
        raw_pages = _read_directory(input_path, page_count)
    elif input_path.is_file():
        if page_map_path is None:
            raw_pages = _read_combined(input_path, printed_page_start, page_count)
        else:
            input_bytes = input_path.read_bytes()
            raw_pages, page_map_pages, page_map_bytes = _page_map_pages(
                page_map_path,
                input_bytes,
                printed_page_start,
                source_pdf_page_start,
                page_count,
            )
    else:
        raise PreparationError("input must be a Markdown file or directory")
    cleaned_pages = _clean_pages(raw_pages)
    input_hash = _sha256(input_bytes) if input_bytes is not None else _input_sha256(input_path)
    if page_map_path is not None and (
        input_path.read_bytes() != input_bytes or page_map_path.read_bytes() != page_map_bytes
    ):
        raise PreparationError("input or page map changed during preparation")
    timestamp = generation_timestamp or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    staging_parent = output_path.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=staging_parent))
    try:
        records: list[dict[str, Any]] = []
        for index, (raw, cleaned) in enumerate(zip(raw_pages, cleaned_pages, strict=True)):
            review_status = (
                page_map_pages[index]["review_status"]
                if page_map_pages is not None
                else "verified-boundary"
            )
            page_id = f"{document_id}:{chapter_id}:{index:04d}"
            raw_relative = Path("pages/raw") / f"{index:04d}.md"
            cleaned_relative = Path("pages/cleaned") / f"{index:04d}.md"
            for relative, content in ((raw_relative, raw), (cleaned_relative, cleaned)):
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding="utf-8", newline="\n")
            records.append({
                "page_id": page_id,
                "printed_page_number": printed_page_start + index,
                "source_pdf_page_number": source_pdf_page_start + index,
                "chapter_page_index": index,
                "raw_path": raw_relative.as_posix(),
                "cleaned_path": cleaned_relative.as_posix(),
                "raw_sha256": _sha256(raw.encode("utf-8")),
                "cleaned_sha256": _sha256(cleaned.encode("utf-8")),
                "warnings": [],
                "review_status": review_status,
            })
            if page_map_pages is not None:
                page_map_page = page_map_pages[index]
                records[-1].update({
                    "source_line_start": page_map_page["start_line"],
                    "source_line_end": page_map_page["end_line"],
                    "boundary_review_status": page_map_page["review_status"],
                })
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "document_id": document_id,
            "chapter_id": chapter_id,
            "ocr_engine": ocr_engine,
            "source_pdf_locator": source_pdf_locator,
            "source_pdf_sha256": source_pdf_sha256.lower(),
            "input_path_locator": str(input_path),
            "input_sha256": input_hash,
            "cleaning": {"tool": "medical_kg_sourceprep", "version": TOOL_VERSION, "config": "edge-furniture-v1"},
            "generation_timestamp": timestamp,
            "page_count": page_count,
            "pages": records,
        }
        if page_map_path is not None and page_map_bytes is not None:
            manifest.update({
                "page_map_locator": str(page_map_path),
                "page_map_sha256": _sha256(page_map_bytes),
            })
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        staging.replace(output_path)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
