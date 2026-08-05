"""Build a candidate ontology projection from the Chapter 01 entity output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from medical_kg_sourceprep.extraction.entity_ontology import build_ontology_candidate
from medical_kg_sourceprep.extraction.artifacts import load_json as _load, sha256_path as _sha
from medical_kg_sourceprep.extraction.llm_extraction import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
RAW_DEFAULT = ROOT / "runtime/chapter-01-entity-extraction-deepseek-direct-v0.3/entities.json"
RULES_DEFAULT = ROOT / "runtime/chapter-01-indicator-rule-functions-deepseek-direct-v0.1/rules.json"
OUTPUT_DEFAULT = ROOT / "runtime/chapter-01-entity-extraction-deepseek-direct-v0.4"
SOURCE_DEFAULT = ROOT / "source-packages/chapter-01/manifest.json"


def _source_pages(manifest_path: Path) -> list[dict]:
    manifest = _load(manifest_path)
    pages: list[dict] = []
    for page in sorted(manifest["pages"], key=lambda value: value["chapter_page_index"]):
        path = manifest_path.parent / page["cleaned_path"]
        pages.append({
            **page,
            "text": path.read_text(encoding="utf-8"),
            "cleaned_sha256": page.get("cleaned_sha256") or _sha(path),
        })
    return pages


def run(source_manifest: Path, raw_entities_path: Path, rules_path: Path, output: Path) -> dict:
    raw_entities = _load(raw_entities_path)
    rules = _load(rules_path)
    pages = _source_pages(source_manifest)
    candidate = build_ontology_candidate(raw_entities, pages, rules)
    output.mkdir(parents=True, exist_ok=True)

    atomic_write_json(output / "entities.json", candidate["entities"])
    atomic_write_json(output / "ontology-relations.json", {
        "schema_version": "chapter-entity-ontology-relations/v0.1",
        "status": "candidate-only",
        "hold": True,
        "approved": 0,
        "relations": candidate["relations"],
    })
    atomic_write_json(output / "rule-alignment.json", candidate["rule_alignment"])
    atomic_write_json(output / "review-queue.json", {
        "schema_version": "chapter-entity-ontology-review/v0.1",
        "status": "HOLD",
        "items": candidate["review_items"],
        "counts": {"review_required": len(candidate["review_items"])},
    })
    extraction = {
        "schema_version": "chapter-entity-ontology-candidate/v0.1",
        "status": "candidate-only",
        "hold": True,
        "approved": 0,
        "derived_from": {
            "raw_entities": str(raw_entities_path),
            "raw_entities_sha256": _sha(raw_entities_path),
            "rules": str(rules_path),
            "rules_sha256": _sha(rules_path),
        },
        "source_manifest_sha256": _sha(source_manifest),
        "audit": candidate["audit"],
    }
    atomic_write_json(output / "extraction.json", extraction)
    manifest = {
        "schema_version": "chapter-entity-ontology-run/v0.1",
        "status": "candidate-only",
        "hold": True,
        "approved": 0,
        "derived_from": extraction["derived_from"],
        "source": {"manifest_sha256": _sha(source_manifest), "pages": len(pages)},
        "counts": candidate["audit"],
        "artifacts": {
            "entities": "entities.json",
            "relations": "ontology-relations.json",
            "rule_alignment": "rule-alignment.json",
            "review_queue": "review-queue.json",
        },
    }
    atomic_write_json(output / "run-manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), file=sys.stderr)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--raw-entities", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--rules", type=Path, default=RULES_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    run(args.source, args.raw_entities, args.rules, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
