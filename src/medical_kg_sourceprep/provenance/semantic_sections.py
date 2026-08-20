"""Build deterministic, heading-aware semantic sections over EvidenceChunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .package_validation import ChunkPackageError, sha256_bytes, validate_chunk_package

SECTION_SCHEMA_VERSION = "semantic-section-package/v0.2"
TOOL_VERSION = "0.2.0"

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_CHINESE_SECTION = re.compile(r"^([一二三四五六七八九十百]+)、\s*(\S.*)$")
_CHINESE_SUBSECTION = re.compile(
    r"^[（(]([一二三四五六七八九十百]+)[）)]\s*(\S.*)$"
)
_LABEL_HEADING = re.compile(r"^【([^】]+)】\s*$")
_CROSS_REFERENCE = re.compile(r"^见(?!于|表|图).{1,80}[。.．]?$")


class SemanticSectionError(ValueError):
    """Raised when semantic sections cannot be built or replayed safely."""


@dataclass(frozen=True, slots=True)
class _Piece:
    page_id: str
    page_index: int
    page_start: int
    page_end: int
    text: str


@dataclass(frozen=True, slots=True)
class _Heading:
    level: int
    title: str
    kind: str


@dataclass(slots=True)
class _Span:
    chunk_id: str
    chunk_sha256: str
    page_id: str
    page_start: int
    page_end: int
    chunk_start: int
    chunk_end: int
    quote: str


def _heading(line: str) -> _Heading | None:
    stripped = line.strip()
    if not stripped:
        return None
    match = _MARKDOWN_HEADING.fullmatch(stripped)
    if match:
        return _Heading(min(len(match.group(1)), 3), match.group(2).strip(), "markdown")
    match = _CHINESE_SECTION.fullmatch(stripped)
    if match:
        return _Heading(2, match.group(2).strip(), "chinese_section")
    match = _CHINESE_SUBSECTION.fullmatch(stripped)
    if match:
        return _Heading(3, match.group(2).strip(), "chinese_subsection")
    return None


def _label(line: str) -> str | None:
    match = _LABEL_HEADING.fullmatch(line.strip())
    return match.group(1).strip() if match else None


def _page_pieces(manifest: dict[str, Any], chunks: Sequence[dict[str, Any]]) -> list[_Piece]:
    by_page: dict[str, list[dict[str, Any]]] = {
        page["page_id"]: [] for page in manifest["pages"]
    }
    for chunk in chunks:
        by_page[chunk["page_id"]].append(chunk)

    pieces: list[_Piece] = []
    for page in manifest["pages"]:
        page_id = page["page_id"]
        page_index = page["chapter_page_index"]
        text = "".join(
            chunk["text"]
            for chunk in sorted(by_page[page_id], key=lambda value: value["cleaned_char_start"])
        )
        cursor = 0
        for line in text.splitlines(keepends=True):
            end = cursor + len(line)
            pieces.append(_Piece(page_id, page_index, cursor, end, line))
            cursor = end
        if cursor < len(text):
            pieces.append(_Piece(page_id, page_index, cursor, len(text), text[cursor:]))
        if not text:
            pieces.append(_Piece(page_id, page_index, 0, 0, ""))
    return pieces


def _structural_groups(
    pieces: Sequence[_Piece],
) -> list[tuple[tuple[str, ...], list[_Piece]]]:
    path: list[str] = []
    current_path: tuple[str, ...] = ()
    current: list[_Piece] = []
    groups: list[tuple[tuple[str, ...], list[_Piece]]] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append((current_path, current))
            current = []

    for piece in pieces:
        heading = _heading(piece.text)
        if heading is not None:
            flush()
            del path[heading.level - 1 :]
            while len(path) < heading.level - 1:
                path.append("未命名层级")
            path.append(heading.title)
            current_path = tuple(path)
        current.append(piece)
    flush()
    return groups


def _split_oversize(pieces: Sequence[_Piece], target_chars: int, max_chars: int) -> list[list[_Piece]]:
    if sum(len(piece.text) for piece in pieces) <= max_chars:
        return [list(pieces)]

    result: list[list[_Piece]] = []
    current: list[_Piece] = []
    size = 0
    last_blank_cut: int | None = None

    def flush(cut: int | None = None) -> None:
        nonlocal current, size, last_blank_cut
        if cut is None:
            cut = len(current)
        if cut <= 0:
            return
        result.append(current[:cut])
        current = current[cut:]
        size = sum(len(piece.text) for piece in current)
        last_blank_cut = next(
            (index for index in range(len(current), 0, -1) if not current[index - 1].text.strip()),
            None,
        )

    for piece in pieces:
        if len(piece.text) > max_chars:
            flush()
            start = 0
            while start < len(piece.text):
                end = min(len(piece.text), start + max_chars)
                result.append(
                    [
                        _Piece(
                            piece.page_id,
                            piece.page_index,
                            piece.page_start + start,
                            piece.page_start + end,
                            piece.text[start:end],
                        )
                    ]
                )
                start = end
            continue
        if current and size + len(piece.text) > max_chars:
            flush(last_blank_cut if last_blank_cut and size >= target_chars else None)
            if current and size + len(piece.text) > max_chars:
                flush()
        current.append(piece)
        size += len(piece.text)
        if not piece.text.strip():
            last_blank_cut = len(current)
        if size >= target_chars and last_blank_cut is not None:
            flush(last_blank_cut)
    flush()
    return result


def _source_spans(
    pieces: Sequence[_Piece], chunks_by_page: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    spans: list[_Span] = []
    for piece in pieces:
        if not piece.text:
            continue
        for chunk in chunks_by_page[piece.page_id]:
            start = max(piece.page_start, chunk["cleaned_char_start"])
            end = min(piece.page_end, chunk["cleaned_char_end"])
            if start >= end:
                continue
            text_start = start - piece.page_start
            text_end = end - piece.page_start
            quote = piece.text[text_start:text_end]
            chunk_start = start - chunk["cleaned_char_start"]
            chunk_end = end - chunk["cleaned_char_start"]
            if (
                spans
                and spans[-1].chunk_id == chunk["chunk_id"]
                and spans[-1].page_end == start
                and spans[-1].chunk_end == chunk_start
            ):
                spans[-1].page_end = end
                spans[-1].chunk_end = chunk_end
                spans[-1].quote += quote
            else:
                spans.append(
                    _Span(
                        chunk["chunk_id"],
                        chunk["chunk_sha256"],
                        piece.page_id,
                        start,
                        end,
                        chunk_start,
                        chunk_end,
                        quote,
                    )
                )
    return [
        {
            "chunk_id": span.chunk_id,
            "chunk_sha256": span.chunk_sha256,
            "page_id": span.page_id,
            "page_char_start": span.page_start,
            "page_char_end": span.page_end,
            "chunk_char_start": span.chunk_start,
            "chunk_char_end": span.chunk_end,
            "exact_quote_sha256": sha256_bytes(span.quote.encode("utf-8")),
        }
        for span in spans
    ]


def _section_kind(pieces: Sequence[_Piece]) -> str:
    meaningful = [
        piece.text
        for piece in pieces
        if piece.text.strip() and _heading(piece.text) is None and _label(piece.text) is None
    ]
    return "semantic" if meaningful else "structural"


def _normalized_line(value: str) -> str:
    return re.sub(r"\s+", "", value).strip()


def _running_header_spans(text: str, path: Sequence[str]) -> list[dict[str, Any]]:
    """Locate repeated bare chapter headers without changing the evidence text."""
    if not path:
        return []
    root_heading = _normalized_line(path[0])
    spans: list[dict[str, Any]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and _normalized_line(stripped) == root_heading:
            leading = len(line) - len(line.lstrip())
            start = cursor + leading
            end = start + len(stripped)
            spans.append(
                {
                    "kind": "running_header",
                    "section_char_start": start,
                    "section_char_end": end,
                    "text_sha256": sha256_bytes(stripped.encode("utf-8")),
                }
            )
        cursor += len(line)
    return spans


def _mask_noise(text: str, noise_spans: Sequence[dict[str, Any]]) -> str:
    """Blank noise while preserving every character offset and line break."""
    characters = list(text)
    for span in noise_spans:
        start = int(span["section_char_start"])
        end = int(span["section_char_end"])
        for index in range(start, end):
            if characters[index] not in "\r\n":
                characters[index] = " "
    return "".join(characters)


def _path_is_parent(path: Sequence[str], following: Sequence[str]) -> bool:
    return len(following) > len(path) and tuple(following[: len(path)]) == tuple(path)


def _content_role(
    record: dict[str, Any], following: dict[str, Any] | None, input_text: str
) -> tuple[str, bool, str]:
    if record["kind"] == "structural":
        return "structural", False, "metadata_only"
    path = [str(value) for value in record["section_path"]]
    if any("索引" in _normalized_line(value) for value in path):
        return "index", False, "alias_dictionary"

    meaningful_lines = [
        line.strip()
        for line in input_text.splitlines()
        if line.strip() and _heading(line) is None and _label(line) is None
    ]
    body = "".join(meaningful_lines)
    if body and len(body) <= 100 and _CROSS_REFERENCE.fullmatch(body):
        return "cross_reference", False, "cross_reference_resolution"

    following_path = following["section_path"] if following is not None else []
    if _path_is_parent(path, following_path):
        return "introduction", True, "entity_and_conservative_relation"
    return "clinical_content", True, "entity_relation_rule"


def build_semantic_sections(
    evidence_package: Path,
    output_path: Path,
    *,
    target_chars: int = 2400,
    max_chars: int = 4000,
    generation_timestamp: str | None = None,
) -> dict[str, Any]:
    """Create a replayable semantic-section package without changing canonical chunks."""
    if (
        isinstance(target_chars, bool)
        or isinstance(max_chars, bool)
        or not isinstance(target_chars, int)
        or not isinstance(max_chars, int)
        or target_chars < 1
        or max_chars < target_chars
    ):
        raise SemanticSectionError("target_chars and max_chars must be positive and ordered")
    source_root = Path(evidence_package).resolve()
    destination = Path(output_path)
    output_root = destination.resolve()
    if output_root == source_root or source_root in output_root.parents or output_root in source_root.parents:
        raise SemanticSectionError("output path must not overlap the evidence package")
    if destination.exists():
        raise SemanticSectionError("output path already exists; refusing to overwrite")
    try:
        manifest_bytes, evidence_manifest, chunks = validate_chunk_package(source_root)
    except ChunkPackageError as error:
        raise SemanticSectionError(str(error)) from error

    chunks_by_page: dict[str, list[dict[str, Any]]] = {
        page["page_id"]: [] for page in evidence_manifest["pages"]
    }
    for chunk in chunks:
        chunks_by_page[chunk["page_id"]].append(chunk)
    for values in chunks_by_page.values():
        values.sort(key=lambda value: value["cleaned_char_start"])

    groups = _structural_groups(_page_pieces(evidence_manifest, chunks))
    planned: list[tuple[tuple[str, ...], list[_Piece], int, int]] = []
    for path, pieces in groups:
        parts = _split_oversize(pieces, target_chars, max_chars)
        for part_index, part in enumerate(parts, start=1):
            planned.append((path, part, part_index, len(parts)))

    timestamp = generation_timestamp or (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        records: list[dict[str, Any]] = []
        section_texts: list[str] = []
        for ordinal, (path, pieces, part_index, part_count) in enumerate(planned, start=1):
            text = "".join(piece.text for piece in pieces)
            section_texts.append(text)
            relative = Path("sections") / f"{ordinal:06d}.md"
            section_file = staging / relative
            section_file.parent.mkdir(parents=True, exist_ok=True)
            section_file.write_text(text, encoding="utf-8", newline="\n")
            spans = _source_spans(pieces, chunks_by_page)
            page_ids = list(dict.fromkeys(piece.page_id for piece in pieces if piece.text))
            facets = list(
                dict.fromkeys(label for piece in pieces if (label := _label(piece.text)) is not None)
            )
            noise_spans = _running_header_spans(text, path)
            input_text = _mask_noise(text, noise_spans)
            input_relative = Path("input-views") / f"{ordinal:06d}.md"
            input_file = staging / input_relative
            input_file.parent.mkdir(parents=True, exist_ok=True)
            input_file.write_text(input_text, encoding="utf-8", newline="\n")
            records.append(
                {
                    "section_id": f"{evidence_manifest['document_id']}:section:{ordinal:06d}",
                    "section_path": list(path),
                    "title": path[-1] if path else "前置内容",
                    "facets": facets,
                    "kind": _section_kind(pieces),
                    "part_index": part_index,
                    "part_count": part_count,
                    "section_path_text": " / ".join(path) if path else "前置内容",
                    "section_path_source": "deterministic_heading_rules",
                    "section_file": relative.as_posix(),
                    "section_sha256": sha256_bytes(text.encode("utf-8")),
                    "char_count": len(text),
                    "input_view_file": input_relative.as_posix(),
                    "input_view_sha256": sha256_bytes(input_text.encode("utf-8")),
                    "input_view_char_count": len(input_text),
                    "noise_spans": noise_spans,
                    "page_ids": page_ids,
                    "source_spans": spans,
                }
            )

        for index, record in enumerate(records):
            input_text = _mask_noise(section_texts[index], record["noise_spans"])
            following = records[index + 1] if index + 1 < len(records) else None
            role, eligible, route = _content_role(record, following, input_text)
            record["content_role"] = role
            record["extraction_eligible"] = eligible
            record["extraction_route"] = route

        if (source_root / "manifest.json").read_bytes() != manifest_bytes:
            raise SemanticSectionError("evidence package changed during sectioning")
        manifest = {
            "schema_version": SECTION_SCHEMA_VERSION,
            "document_id": evidence_manifest["document_id"],
            "chapter_id": evidence_manifest["chapter_id"],
            "source_evidence_manifest": str(source_root / "manifest.json"),
            "source_evidence_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "generation_timestamp": timestamp,
            "sectioner": {
                "tool": "medical_kg_sourceprep",
                "version": TOOL_VERSION,
                "config": {"target_chars": target_chars, "max_chars": max_chars},
                "heading_rules": [
                    "markdown_h1_to_h3",
                    "chinese_section_numeral",
                    "chinese_parenthesized_subsection",
                ],
                "role_rules": [
                    "structural_heading_only",
                    "appendix_index",
                    "short_cross_reference",
                    "parent_section_introduction",
                    "clinical_content_default",
                ],
                "input_view_policy": "mask_running_headers_preserve_offsets",
            },
            "source_counts": {
                "pages": evidence_manifest["page_count"],
                "chunks": evidence_manifest["chunk_count"],
                "characters": sum(len(chunk["text"]) for chunk in chunks),
            },
            "section_count": len(records),
            "sections": records,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        staging.replace(destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-chars", type=int, default=2400)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--generation-timestamp")
    args = parser.parse_args(argv)
    try:
        build_semantic_sections(
            args.evidence_package,
            args.output,
            target_chars=args.target_chars,
            max_chars=args.max_chars,
            generation_timestamp=args.generation_timestamp,
        )
    except SemanticSectionError as error:
        parser.error(str(error))
    print((args.output / "manifest.json").as_posix())


if __name__ == "__main__":
    main(sys.argv[1:])
