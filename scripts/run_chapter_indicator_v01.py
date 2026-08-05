"""Extract the Chapter 01 indicator library and Label Studio review package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

from medical_kg_sourceprep.rules.indicator_catalog import (
    CATALOG_VERSION,
    LABEL_STUDIO_CONFIG,
    LABEL_STUDIO_VERSION,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    VALIDATOR_VERSION,
    aggregate_indicators,
    build_indicator_library,
    build_indicator_prompt,
    derive_table_column_indicators,
    hydrate_legacy_sources,
    label_studio_tasks,
    legacy_testitem_proposals,
    load_index_entries,
    validate_indicator_response,
)
from medical_kg_sourceprep.extraction.artifacts import (
    atomic_write_text as _atomic_write_text,
    load_json as _load_json,
    sha256_path as _sha,
)
from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk, atomic_write_json, load_chunk_manifest
from run_chapter_semantic_v02 import MODEL, _post

ROOT = Path(__file__).resolve().parents[1]


def _page_chunks(chunks: tuple[EvidenceChunk, ...]) -> dict[str, list[EvidenceChunk]]:
    pages: dict[str, list[EvidenceChunk]] = {}
    for chunk in chunks:
        pages.setdefault(chunk.page_id, []).append(chunk)
    return pages


def probe(chunks_manifest: Path, key: str, post: Callable[[str, str], dict] = _post) -> dict[str, Any]:
    manifest, chunks = load_chunk_manifest(chunks_manifest)
    first_page = _page_chunks(chunks)[manifest["pages"][0]["page_id"]]
    raw = post(key, build_indicator_prompt(first_page[0].page_id, first_page))
    validated = validate_indicator_response(raw, first_page)
    by_chunk = {chunk.chunk_id: chunk for chunk in first_page}
    diagnostics = []

    def inspect(value: Any, path: str = "root") -> None:
        if isinstance(value, dict):
            if set(value) == {"chunk_id", "chunk_sha256", "exact_quote"}:
                source = by_chunk.get(value.get("chunk_id"))
                quote = value.get("exact_quote")
                diagnostics.append({
                    "path": path, "chunk_known": source is not None,
                    "hash_match": source is not None and value.get("chunk_sha256") == source.chunk_sha256,
                    "quote_length": len(quote) if isinstance(quote, str) else None,
                    "quote_occurrences": source.text.count(quote) if source is not None and isinstance(quote, str) else None,
                })
            else:
                for name, child in value.items():
                    inspect(child, f"{path}.{name}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{path}[{index}]")

    inspect(raw)
    return {
        "status": "pass", "endpoint": "https://api.deepseek.com/chat/completions",
        "model": MODEL, "trust_env": False, "thinking": "disabled",
        "page_id": first_page[0].page_id, "accepted": len(validated["candidates"]),
        "rejected": len(validated["rejections"]),
        "rejection_reasons": sorted({item["reason_code"] for item in validated["rejections"]}),
        "source_ref_diagnostics": diagnostics,
    }


def run(
    chunks_manifest: Path,
    index_manifest: Path,
    legacy_extraction: Path,
    output: Path,
    key: str,
    post: Callable[[str, str], dict] = _post,
) -> dict[str, Any]:
    manifest, chunks_tuple = load_chunk_manifest(chunks_manifest)
    if manifest.get("chapter_id") != "chapter-01" or manifest.get("page_count") != 24:
        raise RuntimeError("indicator run is limited to the 24-page Chapter 01 package")
    chunks = {chunk.chunk_id: chunk for chunk in chunks_tuple}
    pages = _page_chunks(chunks_tuple)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": "chapter-indicator-checkpoint/v0.1",
        "input_manifest_sha256": _sha(chunks_manifest),
        "index_manifest_sha256": _sha(index_manifest),
        "legacy_extraction_sha256": _sha(legacy_extraction),
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "trust_env": False,
        "thinking": "disabled",
    }
    checkpoint_path = output / "checkpoint.json"
    checkpoint = _load_json(checkpoint_path) if checkpoint_path.exists() else {}
    drift = {name for name, value in identity.items()
             if name in checkpoint and checkpoint.get(name) != value}
    if drift:
        raise RuntimeError("checkpoint input, prompt, validator, or provider identity drift")
    checkpoint.update(identity)
    checkpoint.setdefault("pages", {})
    for page in manifest["pages"]:
        page_id = page["page_id"]
        if checkpoint["pages"].get(page_id, {}).get("status") == "success":
            continue
        page_chunks = pages[page_id]
        try:
            raw = post(key, build_indicator_prompt(page_id, page_chunks))
            validated = validate_indicator_response(raw, page_chunks)
            checkpoint["pages"][page_id] = {
                "status": "success", "chunk_ids": [chunk.chunk_id for chunk in page_chunks],
                "output": validated,
            }
            atomic_write_json(checkpoint_path, checkpoint)
        except Exception as exc:
            checkpoint["pages"][page_id] = {
                "status": "failed", "chunk_ids": [chunk.chunk_id for chunk in page_chunks],
                "error": f"{type(exc).__name__}: {str(exc)[:180]}",
            }
            atomic_write_json(checkpoint_path, checkpoint)
            raise

    packages = [checkpoint["pages"][page["page_id"]]["output"] for page in manifest["pages"]]
    model_proposals = [item for package in packages for item in package["candidates"]]
    rejections = [item for package in packages for item in package["rejections"]]
    legacy = hydrate_legacy_sources(
        legacy_testitem_proposals(_load_json(legacy_extraction)), chunks
    )
    proposals = [*model_proposals, *legacy]
    index_entries = load_index_entries(index_manifest)
    derived_indicators = derive_table_column_indicators(chunks_tuple)
    input_hashes = {
        "chapter_chunks_manifest_sha256": identity["input_manifest_sha256"],
        "full_book_manifest_sha256": identity["index_manifest_sha256"],
        "legacy_v02_extraction_sha256": identity["legacy_extraction_sha256"],
    }
    library, unmatched_index = build_indicator_library(
        proposals, index_entries, input_hashes=input_hashes,
        derived_indicators=derived_indicators,
    )
    tasks = label_studio_tasks(library, chunks_tuple)
    preannotation_count = sum(
        len(task["predictions"][0]["result"]) for task in tasks
    )
    blank_task_count = sum(
        not task["predictions"][0]["result"] for task in tasks
    )
    index_aliases = [alias for item in library["indicators"] for alias in item["index_aliases"]]
    extraction = {
        "schema_version": SCHEMA_VERSION, "status": "candidate-only", "hold": True,
        "approved": 0, "pages": len(packages), "packages": packages,
        "legacy_v02_candidates": legacy, "candidate_count": len(proposals),
        "rejections": rejections,
    }
    atomic_write_json(output / "indicator-extraction.json", extraction)
    atomic_write_json(output / "indicator-library.json", library)
    atomic_write_json(output / "label-studio-tasks.json", tasks)
    _atomic_write_text(output / "label-studio-config.xml", LABEL_STUDIO_CONFIG)
    atomic_write_json(output / "label-studio-tasks-v0.5-indicator-ner-only.json", tasks)
    _atomic_write_text(
        output / "label-studio-config-v0.5-indicator-ner-only.xml",
        LABEL_STUDIO_CONFIG,
    )
    atomic_write_json(output / "review-queue.json", {
        "schema_version": "indicator-review-queue/v0.1", "status": "HOLD",
        "approved": 0, "items": rejections,
        "counts": {
            "validator_rejections": len(rejections),
            "label_studio_tasks": len(tasks),
            "label_studio_preannotations": preannotation_count,
        },
    })
    atomic_write_json(output / "index-alias-audit.json", {
        "schema_version": "indicator-index-alias-audit/v0.1",
        "policy": "index-may-attach-aliases-to-body-anchored-indicators-but-cannot-create-indicators",
        "index_entries": len(index_entries), "attached_aliases": len(index_aliases),
        "matched_rows": sorted({alias["source"]["exact_quote"] for alias in index_aliases}),
        "unmatched_entries": len(unmatched_index),
    })
    manifest_out = {
        "schema_version": "chapter-indicator-run/v0.1", "status": "candidate-only",
        "hold": True, "approved": 0, "provider": "deepseek-direct", "model": MODEL,
        "prompt_version": PROMPT_VERSION, "validator_version": VALIDATOR_VERSION,
        "catalog_version": CATALOG_VERSION, "label_studio_version": LABEL_STUDIO_VERSION,
        "trust_env": False, "thinking": "disabled", "input": {
            "chapter_id": manifest["chapter_id"], "pages": manifest["page_count"],
            "chunks": manifest["chunk_count"], **input_hashes,
        },
        "counts": {
            "pages_completed": len(packages), "model_proposals": len(model_proposals),
            "legacy_v02_proposals": len(legacy),
            "derived_table_indicators": len(derived_indicators),
            "premerge_groups": len(aggregate_indicators(proposals)),
            "indicators": library["indicator_count"], "index_aliases": len(index_aliases),
            "validator_rejections": len(rejections), "label_studio_tasks": len(tasks),
            "label_studio_preannotations": preannotation_count,
            "label_studio_blank_tasks": blank_task_count,
            "label_studio_excluded_noncontiguous_derived": len(derived_indicators),
        },
        "artifacts": {
            "indicator_library_sha256": _sha(output / "indicator-library.json"),
            "label_studio_tasks_sha256": _sha(output / "label-studio-tasks.json"),
            "label_studio_config_sha256": _sha(output / "label-studio-config.xml"),
            "label_studio_indicator_ner_tasks_sha256": _sha(
                output / "label-studio-tasks-v0.5-indicator-ner-only.json"
            ),
            "label_studio_indicator_ner_config_sha256": _sha(
                output / "label-studio-config-v0.5-indicator-ner-only.xml"
            ),
        },
    }
    atomic_write_json(output / "run-manifest.json", manifest_out)
    return manifest_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path,
                        default=ROOT / "source-packages/chunks/chapter-01/manifest.json")
    parser.add_argument("--index-manifest", type=Path,
                        default=ROOT / "source-packages/full-book-v0.2/manifest.json")
    parser.add_argument("--legacy-extraction", type=Path,
                        default=ROOT / "runtime/chapter-01-semantic-kg-deepseek-direct-v0.2/extraction.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runtime/chapter-01-indicator-library-deepseek-direct-v0.1")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    key = sys.stdin.readline().strip()
    if not key:
        print("DEEPSEEK_API_KEY must be supplied through stdin", file=sys.stderr)
        return 2
    result = probe(args.chunks, key) if args.probe_only else run(
        args.chunks, args.index_manifest, args.legacy_extraction, args.output, key
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
