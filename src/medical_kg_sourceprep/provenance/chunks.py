"""Create deterministic, page-local EvidenceChunk packages from source packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SOURCE_SCHEMA_VERSION = "source-package/v0.1"
CHUNK_SCHEMA_VERSION = "evidence-chunk-package/v0.1"
TOOL_VERSION = "0.1.0"
_REQUIRED_MANIFEST_FIELDS = {"schema_version", "document_id", "chapter_id", "page_count", "pages"}
_REQUIRED_PAGE_FIELDS = {
    "page_id",
    "chapter_page_index",
    "printed_page_number",
    "source_pdf_page_number",
    "cleaned_path",
    "cleaned_sha256",
    "review_status",
    "warnings",
}


class ChunkingError(ValueError):
    """Raised when a source package cannot be chunked without losing provenance."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_utf8_lf(path: Path, description: str) -> tuple[bytes, str]:
    try:
        content = path.read_bytes()
        text = content.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ChunkingError(f"{description} must be readable UTF-8") from error
    if "\r" in text:
        raise ChunkingError(f"{description} must use LF line endings")
    return content, text


def _safe_child(root: Path, relative: str, description: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not relative or ".." in path.parts:
        raise ChunkingError(f"{description} must be a safe relative path")
    resolved = (root / path).resolve()
    if root not in resolved.parents:
        raise ChunkingError(f"{description} escapes the source package")
    return resolved


def _validate_source(
    source_package: Path,
) -> tuple[Path, bytes, dict[str, Any], list[tuple[dict[str, Any], bytes, str]]]:
    if not source_package.is_dir():
        raise ChunkingError("source package must be an existing directory")
    root = source_package.resolve()
    manifest_path = root / "manifest.json"
    manifest_bytes, manifest_text = _read_utf8_lf(manifest_path, "source manifest")
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as error:
        raise ChunkingError("source manifest must be valid JSON") from error
    if not isinstance(manifest, dict) or not _REQUIRED_MANIFEST_FIELDS <= set(manifest):
        raise ChunkingError("source manifest is missing required fields")
    if manifest["schema_version"] != SOURCE_SCHEMA_VERSION:
        raise ChunkingError("unsupported source manifest schema version")
    if not all(
        isinstance(manifest[field], str) and manifest[field]
        for field in ("document_id", "chapter_id")
    ):
        raise ChunkingError("source manifest document_id and chapter_id must be non-empty strings")
    pages = manifest["pages"]
    if not isinstance(manifest["page_count"], int) or isinstance(manifest["page_count"], bool):
        raise ChunkingError("source manifest page_count must be an integer")
    if not isinstance(pages, list) or len(pages) != manifest["page_count"]:
        raise ChunkingError("source manifest page_count must match pages")

    page_ids: set[str] = set()
    page_indexes: set[int] = set()
    captured: list[tuple[dict[str, Any], bytes, str]] = []
    for page in pages:
        if not isinstance(page, dict) or not _REQUIRED_PAGE_FIELDS <= set(page):
            raise ChunkingError("source manifest page is missing required fields")
        page_id = page["page_id"]
        if not isinstance(page_id, str) or not page_id:
            raise ChunkingError("source page_id must be a non-empty string")
        if page_id in page_ids:
            raise ChunkingError("source manifest has duplicate page_id")
        page_ids.add(page_id)
        index = page["chapter_page_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index in page_indexes:
            raise ChunkingError("source manifest has duplicate or invalid chapter_page_index")
        page_indexes.add(index)
        if any(
            isinstance(page[field], bool) or not isinstance(page[field], int)
            for field in ("printed_page_number", "source_pdf_page_number")
        ):
            raise ChunkingError("source page numbers must be integers")
        if not isinstance(page["review_status"], str) or not isinstance(page["warnings"], list):
            raise ChunkingError("source review fields are invalid")
        expected_hash = page["cleaned_sha256"]
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ChunkingError("source cleaned_sha256 must be lowercase SHA-256")
        cleaned_path = _safe_child(root, page["cleaned_path"], "source cleaned_path")
        if not cleaned_path.is_file():
            raise ChunkingError("source cleaned file is missing")
        content, text = _read_utf8_lf(cleaned_path, "source cleaned file")
        if _sha256(content) != expected_hash:
            raise ChunkingError("source cleaned file hash mismatch")
        captured.append((page, content, text))
    if page_indexes != set(range(len(pages))):
        raise ChunkingError("source chapter_page_index values must be contiguous from zero")
    return manifest_path, manifest_bytes, manifest, captured


def _protected_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(
        r"<table\b[^>]*>.*?</table\s*>", text, re.IGNORECASE | re.DOTALL
    ):
        spans.append(match.span())
    for match in re.finditer(r"<table\b[^>]*>", text, re.IGNORECASE):
        if not any(start <= match.start() < end for start, end in spans):
            spans.append((match.start(), len(text)))
    for match in re.finditer(r"\\\\\[.*?\\\\\]", text, re.DOTALL):
        spans.append(match.span())
    for match in re.finditer(r"\$\$.*?\$\$", text, re.DOTALL):
        spans.append(match.span())

    fence_start = re.compile(r"(?m)^\s*(`{3,}|~{3,})[^\n]*(?:\n|$)")
    for match in fence_start.finditer(text):
        marker = match.group(1)
        close_pattern = rf"(?m)^\s*{re.escape(marker[0])}{{{len(marker)},}}\s*(?:\n|$)"
        close = re.compile(close_pattern).search(text, match.end())
        spans.append((match.start(), close.end() if close else len(text)))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _preferred_cut(
    text: str, start: int, limit: int, spans: list[tuple[int, int]]
) -> int | None:
    def allowed(position: int) -> bool:
        return position > start and not any(left < position < right for left, right in spans)

    blank = [
        index + 2
        for index in range(start, limit - 1)
        if text[index : index + 2] == "\n\n" and allowed(index + 2)
    ]
    if blank:
        return blank[-1]
    newlines = [
        index + 1 for index in range(start, limit) if text[index] == "\n" and allowed(index + 1)
    ]
    return newlines[-1] if newlines else None


def _slice_page(text: str, max_chars: int) -> list[tuple[int, int, list[str]]]:
    spans = _protected_spans(text)
    chunks: list[tuple[int, int, list[str]]] = []
    start = 0
    while start < len(text):
        target = min(len(text), start + max_chars)
        crossing = next((span for span in spans if span[0] < target < span[1]), None)
        active = next(
            (span for span in spans if span[0] <= start < span[1] and target < span[1]), None
        )
        if active is not None:
            end = active[1]
        elif crossing is not None:
            end = _preferred_cut(text, start, crossing[0], spans) or (
                crossing[0] if crossing[0] > start else crossing[1]
            )
        else:
            end = _preferred_cut(text, start, target, spans) or target
        warnings = ["oversize_atomic_block"] if any(
            left >= start and right <= end and right - left > max_chars for left, right in spans
        ) else []
        chunks.append((start, end, warnings))
        start = end
    return chunks


def _validate_output(source_package: Path, output_path: Path) -> None:
    source_root = source_package.resolve()
    output_root = output_path.resolve()
    if (
        output_root == source_root
        or source_root in output_root.parents
        or output_root in source_root.parents
    ):
        raise ChunkingError("output path must not overlap the source package")
    if output_path.exists():
        raise ChunkingError("output path already exists; refusing to overwrite a chunk package")


def prepare_chunks(
    source_package: Path,
    output_path: Path,
    max_chars: int,
    generation_timestamp: str | None = None,
) -> dict[str, Any]:
    """Slice a verified source package without crossing pages or rewriting text."""
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ChunkingError("max_chars must be a positive integer")
    _validate_output(source_package, output_path)
    manifest_path, manifest_bytes, source_manifest, pages = _validate_source(source_package)
    timestamp = generation_timestamp or (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    staging_parent = output_path.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.staging-", dir=staging_parent))
    try:
        chunks: list[dict[str, Any]] = []
        for page, _, text in pages:
            for index, (start, end, warnings) in enumerate(_slice_page(text, max_chars)):
                content = text[start:end]
                relative = (
                    Path("chunks") / f"{page['chapter_page_index']:04d}" / f"{index:04d}.md"
                )
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding="utf-8", newline="\n")
                chunks.append(
                    {
                        "chunk_id": f"{page['page_id']}:{index:04d}",
                        "page_id": page["page_id"],
                        "document_id": source_manifest["document_id"],
                        "chapter_id": source_manifest["chapter_id"],
                        "chapter_page_index": page["chapter_page_index"],
                        "printed_page_number": page["printed_page_number"],
                        "source_pdf_page_number": page["source_pdf_page_number"],
                        "source_cleaned_path": page["cleaned_path"],
                        "source_cleaned_sha256": page["cleaned_sha256"],
                        "cleaned_char_start": start,
                        "cleaned_char_end": end,
                        "chunk_path": relative.as_posix(),
                        "chunk_sha256": _sha256(content.encode("utf-8")),
                        "char_count": end - start,
                        "warnings": warnings,
                        "review_status": page["review_status"],
                        "source_page": page,
                    }
                )
        if manifest_path.read_bytes() != manifest_bytes or any(
            _safe_child(
                source_package.resolve(), page["cleaned_path"], "source cleaned_path"
            ).read_bytes()
            != content
            for page, content, _ in pages
        ):
            raise ChunkingError("source package changed during chunking")
        manifest = {
            "schema_version": CHUNK_SCHEMA_VERSION,
            "source_manifest_locator": str(manifest_path),
            "source_manifest_sha256": _sha256(manifest_bytes),
            "document_id": source_manifest["document_id"],
            "chapter_id": source_manifest["chapter_id"],
            "chunker": {
                "tool": "medical_kg_sourceprep",
                "version": TOOL_VERSION,
                "config": {"max_chars": max_chars},
            },
            "generation_timestamp": timestamp,
            "page_count": source_manifest["page_count"],
            "chunk_count": len(chunks),
            "pages": [page for page, _, _ in pages],
            "chunks": chunks,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        staging.replace(output_path)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-chars", required=True, type=int)
    parser.add_argument("--generation-timestamp")
    args = parser.parse_args(argv)
    try:
        prepare_chunks(args.source_package, args.output, args.max_chars, args.generation_timestamp)
    except ChunkingError as error:
        parser.error(str(error))
    print((args.output / "manifest.json").as_posix())


if __name__ == "__main__":
    main(sys.argv[1:])
