"""Run the five-category entity-only DeepSeek extraction for Chapter 01."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from medical_kg_sourceprep.extraction.entity_extraction import PROMPT_VERSION, build_entity_prompt, merge_entities, validate_page_result
from medical_kg_sourceprep.extraction.llm_extraction import atomic_write_json
from run_chapter_semantic_v02 import MODEL, _post


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_pages(source_manifest: Path) -> list[dict]:
    source = _load(source_manifest)
    return sorted(source["pages"], key=lambda page: page["chapter_page_index"])


def run(source_manifest: Path, output: Path, key: str, limit: int | None = None) -> dict:
    pages = _source_pages(source_manifest)
    if len(pages) != 24:
        raise RuntimeError(f"chapter input must contain 24 pages, got {len(pages)}")
    selected = pages if limit is None else pages[:limit]
    chapter_text = "\n".join(
        (source_manifest.parent / page["cleaned_path"]).read_text(encoding="utf-8") for page in pages
    )
    input_hash = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    checkpoint = _load(checkpoint_path) if checkpoint_path.exists() else {}
    identity = {
        "schema_version": "chapter-entity-checkpoint/v0.1",
        "source_manifest_sha256": input_hash,
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "provider": "deepseek-direct",
        "thinking": "disabled",
        "pages": len(pages),
    }
    if checkpoint and any(checkpoint.get(name) != value for name, value in identity.items()):
        raise RuntimeError("checkpoint source or provider identity drift")
    checkpoint.update(identity)
    checkpoint.setdefault("page_results", {})

    for page in selected:
        page_index = page["chapter_page_index"]
        page_key = f"page:{page_index:04d}"
        if checkpoint["page_results"].get(page_key, {}).get("status") == "success":
            continue
        page_path = source_manifest.parent / page["cleaned_path"]
        text = page_path.read_text(encoding="utf-8")
        try:
            raw = _post(key, build_entity_prompt(page_key, text))
            validated = validate_page_result(raw, text, page_index, grounding_text=chapter_text)
        except Exception as exc:
            checkpoint["page_results"][page_key] = {
                "status": "failed",
                "page_index": page_index,
                "error": str(exc)[:240],
            }
            atomic_write_json(checkpoint_path, checkpoint)
            raise
        checkpoint["page_results"][page_key] = {
            "status": "success",
            "page_index": page_index,
            "source_path": page["cleaned_path"],
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "entity_count": len(validated["entities"]),
            "rejection_count": len(validated["rejections"]),
            "output": validated,
        }
        atomic_write_json(checkpoint_path, checkpoint)
        print(f"page {page_index + 1}/24: {len(validated['entities'])} accepted, {len(validated['rejections'])} rejected", file=sys.stderr)

    if limit is not None and len(selected) < len(pages):
        return {"status": "probe", "pages_completed": len(selected), "checkpoint": str(checkpoint_path)}

    page_results = [checkpoint["page_results"][f"page:{index:04d}"]["output"] for index in range(len(pages))]
    merged = merge_entities(page_results)
    all_rejections = [item for result in page_results for item in result["rejections"]]
    extraction = {
        "schema_version": "chapter-entity-candidates/v0.1",
        "status": "candidate-only",
        "approved": 0,
        "hold": True,
        "provider": "deepseek-direct",
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "source_manifest_sha256": input_hash,
        "entities": merged["entities"],
        "audit": {
            "page_count": len(pages),
            "raw_page_entities": sum(len(result["entities"]) for result in page_results),
            "merged_entities": len(merged["entities"]),
            "rejections": len(all_rejections),
            "conflicts": merged["conflicts"],
            "category_resolutions": merged["category_resolutions"],
            "category_counts": merged["counts"],
        },
    }
    atomic_write_json(output / "entities.json", merged["entities"])
    atomic_write_json(output / "extraction.json", extraction)
    atomic_write_json(output / "review-queue.json", {
        "status": "HOLD",
        "items": all_rejections,
        "conflicts": merged["conflicts"],
        "counts": {"review_required": len(all_rejections) + len(merged["conflicts"])},
    })
    manifest = {
        "schema_version": "chapter-entity-run/v0.1",
        "status": "candidate-only",
        "hold": True,
        "approved": 0,
        "provider": "deepseek-direct",
        "model": MODEL,
        "source": {"manifest_sha256": input_hash, "pages": len(pages)},
        "counts": extraction["audit"],
    }
    atomic_write_json(output / "run-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "source-packages/chapter-01/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/chapter-01-entity-extraction-deepseek-direct-v0.1")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N pages as a probe")
    args = parser.parse_args()
    key = sys.stdin.readline().strip()
    if not key:
        print("DEEPSEEK_API_KEY must be supplied through hidden stdin", file=sys.stderr)
        return 2
    run(args.source, args.output, key, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
