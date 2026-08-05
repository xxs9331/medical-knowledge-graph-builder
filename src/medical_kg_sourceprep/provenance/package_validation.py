"""Shared validation for provenance-bound EvidenceChunk packages."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

CHUNK_SCHEMA_VERSION = "evidence-chunk-package/v0.1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ChunkPackageError(ValueError):
    """Raised when a chunk package cannot be replayed safely."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ChunkPackageError(f"{label} must be an integer")
    return value


def _safe_child(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ChunkPackageError("chunk_path must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ChunkPackageError("chunk_path must be a safe relative path")
    resolved = (root / relative).resolve()
    if root not in resolved.parents:
        raise ChunkPackageError("chunk_path escapes chunk package")
    return resolved


def _read_manifest(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ChunkPackageError("chunk manifest must be readable UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ChunkPackageError("chunk manifest must be an object")
    return raw, value


def validate_chunk_package(
    package_or_manifest: Path,
    *,
    strict: bool = True,
) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    """Validate a chunk package and return manifest bytes, manifest, and text chunks.

    The input may be either the package directory or its ``manifest.json`` path.
    All returned chunk records contain the UTF-8 text replayed from disk.
    """
    input_path = Path(package_or_manifest).resolve()
    root = input_path if input_path.is_dir() else input_path.parent
    manifest_path = root / "manifest.json"
    if input_path.is_file() and input_path.name != "manifest.json":
        raise ChunkPackageError("chunk package must be a directory or manifest.json")
    if not root.is_dir():
        raise ChunkPackageError("chunk package must be an existing directory")
    manifest_bytes, manifest = _read_manifest(manifest_path)
    required = {
        "schema_version", "source_manifest_sha256", "document_id", "chapter_id",
        "page_count", "chunk_count", "pages", "chunks",
    }
    if not required <= set(manifest) or manifest["schema_version"] != CHUNK_SCHEMA_VERSION:
        raise ChunkPackageError("unsupported or incomplete chunk manifest")
    if not all(
        isinstance(manifest[key], str) and manifest[key]
        for key in ("document_id", "chapter_id", "source_manifest_sha256")
    ):
        raise ChunkPackageError("chunk manifest identifiers are invalid")
    if not _SHA256.fullmatch(manifest["source_manifest_sha256"]):
        raise ChunkPackageError("source manifest hash must be SHA-256")
    pages = manifest["pages"]
    chunks = manifest["chunks"]
    if not isinstance(pages, list) or not isinstance(chunks, list):
        raise ChunkPackageError("chunk manifest pages and chunks must be lists")
    if (
        _integer(manifest["page_count"], "page_count") != len(pages)
        or _integer(manifest["chunk_count"], "chunk_count") != len(chunks)
    ):
        raise ChunkPackageError("chunk manifest counts do not match records")

    page_records: dict[str, dict[str, Any]] = {}
    for expected_index, page in enumerate(pages):
        if not isinstance(page, dict) or not isinstance(page.get("page_id"), str) or not page["page_id"]:
            raise ChunkPackageError("page record is invalid")
        if page["page_id"] in page_records or _integer(
            page.get("chapter_page_index"), "chapter_page_index"
        ) != expected_index:
            raise ChunkPackageError("page IDs must be unique and page indexes contiguous")
        page_records[page["page_id"]] = page
        for name in ("printed_page_number", "source_pdf_page_number"):
            _integer(page.get(name), name)
        if not isinstance(page.get("review_status"), str) or not page["review_status"]:
            raise ChunkPackageError("page review status is invalid")
        if not isinstance(page.get("cleaned_sha256"), str) or not _SHA256.fullmatch(
            page["cleaned_sha256"]
        ):
            raise ChunkPackageError("page cleaned hash must be SHA-256")

    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ChunkPackageError("chunk record is invalid")
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
            raise ChunkPackageError("chunk IDs must be non-empty and unique")
        seen.add(chunk_id)
        page = page_records.get(chunk.get("page_id"))
        if page is None or chunk.get("document_id") != manifest["document_id"]:
            raise ChunkPackageError("chunk provenance does not bind to manifest")
        if chunk.get("chapter_id") != manifest["chapter_id"]:
            raise ChunkPackageError("chunk chapter does not bind to manifest")
        if strict:
            if any(
                chunk.get(field) != page.get(field)
                for field in (
                    "chapter_page_index", "printed_page_number", "source_pdf_page_number",
                    "review_status",
                )
            ):
                raise ChunkPackageError("chunk page mapping does not bind to manifest")
            if chunk.get("source_cleaned_sha256") != page.get("cleaned_sha256"):
                raise ChunkPackageError("chunk source hash does not bind to manifest")
            if chunk.get("source_cleaned_path") != page.get("cleaned_path"):
                raise ChunkPackageError("chunk source path does not bind to manifest")
            if chunk.get("source_page") != page:
                raise ChunkPackageError("chunk source page does not bind to manifest")
        try:
            path = _safe_child(root, chunk.get("chunk_path"))
        except ChunkPackageError as error:
            if not strict and "safe relative path" in str(error):
                raise ChunkPackageError("chunk path escapes manifest root") from error
            raise
        try:
            content = path.read_bytes()
            text = content.decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise ChunkPackageError("chunk file must be readable UTF-8") from error
        expected_hash = chunk.get("chunk_sha256")
        if (
            not isinstance(expected_hash, str)
            or not _SHA256.fullmatch(expected_hash)
            or sha256_bytes(content) != expected_hash
        ):
            raise ChunkPackageError("chunk file hash mismatch")
        if strict and "\r" in text:
            raise ChunkPackageError("chunk files must use LF line endings")
        start = _integer(chunk.get("cleaned_char_start"), "cleaned_char_start")
        end = _integer(chunk.get("cleaned_char_end"), "cleaned_char_end")
        if strict and (start < 0 or end <= start or end - start != len(text)):
            raise ChunkPackageError("chunk character offsets are invalid")
        if strict and "char_count" in chunk and _integer(chunk["char_count"], "char_count") != len(text):
            raise ChunkPackageError("chunk character count is invalid")
        record = dict(chunk)
        record["text"] = text
        validated.append(record)

    if not strict:
        return manifest_bytes, manifest, validated

    chunks_by_page: dict[str, list[dict[str, Any]]] = {page_id: [] for page_id in page_records}
    for chunk in validated:
        chunks_by_page[chunk["page_id"]].append(chunk)
    for page_id, page in page_records.items():
        offset = 0
        page_parts: list[str] = []
        for chunk in sorted(chunks_by_page[page_id], key=lambda item: item["cleaned_char_start"]):
            if chunk["cleaned_char_start"] != offset:
                raise ChunkPackageError("chunk offsets are not contiguous within a page")
            page_parts.append(chunk["text"])
            offset = chunk["cleaned_char_end"]
        if sha256_bytes("".join(page_parts).encode("utf-8")) != page["cleaned_sha256"]:
            raise ChunkPackageError("chunk reconstruction does not match cleaned page hash")
    return manifest_bytes, manifest, validated


def validate_chunk_layout(manifest: Mapping[str, Any], page_texts: Mapping[str, str]) -> None:
    """Verify that manifest chunks replay every supplied cleaned page exactly once."""
    identifiers: set[str] = set()
    by_page: dict[str, list[Mapping[str, Any]]] = {page_id: [] for page_id in page_texts}
    for chunk in manifest.get("chunks", []):
        if not isinstance(chunk, Mapping):
            raise ChunkPackageError("chunk record is invalid")
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in identifiers:
            raise ChunkPackageError("duplicate or invalid chunk_id")
        identifiers.add(chunk_id)
        page_id = chunk.get("page_id")
        if page_id not in page_texts:
            raise ChunkPackageError("chunk references an unknown page")
        by_page[page_id].append(chunk)
    for page_id, chunks in by_page.items():
        chunks.sort(key=lambda item: item["cleaned_char_start"])
        cursor = 0
        text = page_texts[page_id]
        for chunk in chunks:
            start, end = chunk.get("cleaned_char_start"), chunk.get("cleaned_char_end")
            if not isinstance(start, int) or not isinstance(end, int) or start != cursor or end <= start:
                raise ChunkPackageError("page chunks cannot reassemble text")
            if end > len(text) or sha256_bytes(text[start:end].encode("utf-8")) != chunk.get("chunk_sha256"):
                raise ChunkPackageError("chunk hash drift")
            cursor = end
        if cursor != len(text):
            raise ChunkPackageError("page chunks cannot reassemble text")


__all__ = [
    "CHUNK_SCHEMA_VERSION",
    "ChunkPackageError",
    "sha256_bytes",
    "validate_chunk_layout",
    "validate_chunk_package",
]
