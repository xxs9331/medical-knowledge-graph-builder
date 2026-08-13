#!/usr/bin/env python3
"""Run resumable candidate-only extraction for selected full-book chapters."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medical_kg_sourceprep.extraction.artifacts import sha256_path
from medical_kg_sourceprep.extraction.graph_builder.client import create_deepseek_graph_builder
from medical_kg_sourceprep.extraction.graph_builder.contract import DEFAULT_SCHEMA_PATH
from medical_kg_sourceprep.extraction.graph_builder.runner import run_candidate_graph
from medical_kg_sourceprep.extraction.graph_builder.schema import load_candidate_graph_schema
from medical_kg_sourceprep.extraction.llm_extraction import atomic_write_json, load_chunk_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "source-packages/canonical/evidence/full-book-v0.2/manifest.json"
DEFAULT_OUTPUT = ROOT / "runtime/candidates/selected-chapters/v0.1"
CHAPTER_RANGES = {"02": (27, 40), "06": (64, 73), "13": (118, 141), "18": (171, 184)}


def select_chunks(manifest_path: Path, chapters: list[str], min_chars: int) -> list[Any]:
    _manifest, chunks = load_chunk_manifest(manifest_path)
    ranges = [CHAPTER_RANGES[chapter] for chapter in chapters]
    return [
        chunk for chunk in chunks
        if len(chunk.text.strip()) >= min_chars
        and any(start <= chunk.page_index <= end for start, end in ranges)
    ]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    chunks = select_chunks(args.manifest, args.chapters, args.min_chars)
    if args.limit is not None:
        chunks = chunks[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / "progress.json"
    progress: dict[str, Any] = {
        "schema_version": "selected-chapter-candidate-progress/v0.1",
        "publication_status": "HOLD",
        "approved": 0,
        "chapters": args.chapters,
        "manifest_sha256": sha256_path(args.manifest),
        "selected_chunk_ids": [chunk.chunk_id for chunk in chunks],
        "completed": [],
        "failed": [],
    }
    if progress_path.exists():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved.get("manifest_sha256") != progress["manifest_sha256"]:
            raise RuntimeError("existing progress manifest hash does not match input")
        progress["completed"] = saved.get("completed", [])
        progress["failed"] = saved.get("failed", [])
    completed = {item["chunk_id"] for item in progress["completed"]}
    schema = load_candidate_graph_schema(args.schema)
    client = create_deepseek_graph_builder()
    try:
        for position, chunk in enumerate(chunks, 1):
            if chunk.chunk_id in completed:
                continue
            page = f"{chunk.page_index:04d}"
            part = chunk.chunk_id.rsplit(":", 1)[-1]
            output_dir = args.output / "chunks" / page / part
            try:
                summary = await run_candidate_graph(
                    client,
                    chunk=chunk,
                    schema=schema,
                    schema_path=args.schema,
                    output_dir=output_dir,
                    source_manifest_sha256=progress["manifest_sha256"],
                    run_id=f"selected-{page}-{part}",
                )
                progress["completed"].append({"chunk_id": chunk.chunk_id, "summary": summary})
                completed.add(chunk.chunk_id)
                progress["failed"] = [
                    item for item in progress["failed"] if item["chunk_id"] != chunk.chunk_id
                ]
            except Exception as error:  # preserve later chunks and an auditable failure record
                progress["failed"] = [
                    item for item in progress["failed"] if item["chunk_id"] != chunk.chunk_id
                ]
                progress["failed"].append({
                    "chunk_id": chunk.chunk_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                })
            progress["updated_at"] = datetime.now(UTC).isoformat()
            progress["position"] = position
            atomic_write_json(progress_path, progress)
    finally:
        await client.aclose()
    return progress


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chapters", nargs="+", choices=sorted(CHAPTER_RANGES),
                        default=sorted(CHAPTER_RANGES))
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps({
        "selected": len(result["selected_chunk_ids"]),
        "completed": len(result["completed"]),
        "failed": len(result["failed"]),
        "output": str(args.output),
    }, ensure_ascii=False, sort_keys=True))
    return 0 if not result["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
