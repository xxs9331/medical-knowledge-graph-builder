"""Run the independent relation-only/rule-only chapter v0.3 supplement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from medical_kg_sourceprep.provenance.book_sources import build_book_manifest_from_packages
from medical_kg_sourceprep.extraction.artifacts import directory_sha256, load_json as _load_json, sha256_path as _sha
from medical_kg_sourceprep.graph.knowledge_graph import PageText
from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk, atomic_write_json, load_chunk_manifest
from medical_kg_sourceprep.graph.semantic_graph import SemanticGraphBuilder, SemanticRecord, SemanticRelation
from medical_kg_sourceprep.extraction.semantic_v03 import (
    RELATION_PROMPT_VERSION, RULE_PROMPT_VERSION, VALIDATOR_VERSION,
    build_entity_catalog, build_relation_prompt, build_rule_prompt, recover_derived_relations,
    stable_relations, validate_relations, validate_rules,
)
from run_chapter_semantic_v02 import MODEL, _post

ROOT = Path(__file__).resolve().parents[1]


def _page_chunks(chunks: tuple[EvidenceChunk, ...]) -> dict[str, list[EvidenceChunk]]:
    return {f"page:{index:04d}": [chunk for chunk in chunks if chunk.page_index == index] for index in range(24)}


def _graph_records(v02: dict, chunks: dict[str, EvidenceChunk]) -> tuple[list[SemanticRecord], dict[tuple[str, str], SemanticRecord]]:
    records, _ = __import__("run_chapter_semantic_v02", fromlist=["_records"])._records(v02, chunks)
    by_key = {}
    for record in records:
        if record.candidate_key:
            by_key[(chunks[record.chunk_id].page_id, record.candidate_key)] = record
    return records, by_key


def run(chunks_manifest: Path, source_manifest: Path, v02_dir: Path, output: Path, key: str | None,
        rerun_rule_contract: bool = False) -> dict:
    manifest, chunks_tuple = load_chunk_manifest(chunks_manifest)
    if manifest.get("page_count") != 24 or manifest.get("chunk_count") != 44:
        raise RuntimeError("chapter input must contain exactly 24 pages and 44 chunks")
    chunks = {chunk.chunk_id: chunk for chunk in chunks_tuple}
    v02 = _load_json(v02_dir / "extraction.json")
    catalog = build_entity_catalog(v02, chunks_tuple)
    output.mkdir(parents=True, exist_ok=True)
    catalog_path = output / "entity-catalog.json"
    atomic_write_json(catalog_path, catalog)
    if key is None:
        return {"schema_version": "chapter-semantic-kg-run/v0.3", "status": "catalog-only",
                "entity_catalog_sha256": catalog["catalog_sha256"], "pages": 24}
    input_hash, catalog_hash = _sha(chunks_manifest), catalog["catalog_sha256"]
    checkpoint_path = output / "checkpoint.json"
    checkpoint = _load_json(checkpoint_path) if checkpoint_path.exists() else {}
    identity = {"schema_version": "chapter-semantic-checkpoint/v0.3", "input_manifest_sha256": input_hash,
                "entity_catalog_sha256": catalog_hash, "model": MODEL, "validator_version": VALIDATOR_VERSION,
                "relation_prompt_version": RELATION_PROMPT_VERSION, "rule_prompt_version": RULE_PROMPT_VERSION,
                "trust_env": False, "thinking": "disabled", "api_ip": os.environ.get("DEEPSEEK_API_IP")}
    mismatches = {key_name for key_name, value in identity.items()
                  if key_name in checkpoint and checkpoint.get(key_name) != value}
    if mismatches:
        if rerun_rule_contract and mismatches == {"rule_prompt_version"}:
            checkpoint["superseded_rule_stage"] = checkpoint.get("stages", {}).get("rule", {})
            checkpoint.setdefault("stages", {})["rule"] = {}
        else:
            raise RuntimeError("checkpoint input, catalog, prompt, or provider identity drift")
    checkpoint.update(identity)
    checkpoint.setdefault("stages", {}).setdefault("relation", {})
    checkpoint.setdefault("stages", {}).setdefault("rule", {})
    pages = _page_chunks(chunks_tuple)
    by_page_catalog = {}
    for entry in catalog["entries"]:
        by_page_catalog.setdefault(entry["page_id"], []).append(entry)
    for page_id, page_chunks in pages.items():
        entries = by_page_catalog.get(page_id, [])
        for stage, prompt_builder, validator in (("relation", build_relation_prompt, validate_relations), ("rule", build_rule_prompt, validate_rules)):
            if checkpoint["stages"].get(stage, {}).get(page_id, {}).get("status") == "success":
                continue
            try:
                raw = _post(key, prompt_builder(page_id, page_chunks, entries))
                validated = validator(raw, page_chunks, catalog)
                checkpoint["stages"][stage][page_id] = {"status": "success", "chunk_ids": [chunk.chunk_id for chunk in page_chunks], "output": validated}
                atomic_write_json(checkpoint_path, checkpoint)
            except Exception as exc:
                checkpoint["stages"][stage][page_id] = {"status": "failed", "chunk_ids": [chunk.chunk_id for chunk in page_chunks], "error": str(exc)[:240]}
                atomic_write_json(checkpoint_path, checkpoint)
                raise
    relation_packages = list(checkpoint["stages"]["relation"][page_id]["output"] for page_id in pages)
    rule_packages = list(checkpoint["stages"]["rule"][page_id]["output"] for page_id in pages)
    derived, derived_review = recover_derived_relations(catalog, chunks_tuple)
    relations = stable_relations([*relation_packages, {"candidates": derived}])
    extraction = {"schema_version": "semantic-candidates/v0.3", "status": "candidate-only", "approved": 0,
                  "entity_catalog_sha256": catalog_hash, "relations": relations,
                  "rules": [item for package in rule_packages for item in package["candidates"]],
                  "rejections": [item for package in [*relation_packages, *rule_packages] for item in package["rejections"]] + derived_review}
    atomic_write_json(output / "relation-extraction.json", {"status": "candidate-only", "pages": 24, "packages": relation_packages, "candidates": relations})
    atomic_write_json(output / "rule-extraction.json", {"status": "candidate-only", "pages": 24, "packages": rule_packages, "candidates": extraction["rules"]})
    atomic_write_json(output / "extraction.json", extraction)
    atomic_write_json(output / "review-queue.json", {"status": "HOLD", "items": extraction["rejections"], "counts": {"review_required": len(extraction["rejections"])}})
    atomic_write_json(output / "gold-template.json", {
        "schema_version": "semantic-gold-template/v0.3", "status": "HOLD",
        "generated_from_model": False, "source_manifest_sha256": _sha(source_manifest),
        "entity_catalog_sha256": catalog_hash,
        "pages": [{"page_id": page_id, "gold_status": "unreviewed", "relations": [], "rules": [],
                   "reviewer": None, "reviewed_at": None} for page_id in pages],
    })
    from collections import Counter
    atomic_write_json(output / "quality-comparison.json", {
        "schema_version": "semantic-quality-comparison/v0.3", "gold_status": "not_generated_from_model",
        "precision_recall_f1": "HOLD", "v02": {"accepted": len(v02.get("candidates", [])), "rejected": len(v02.get("rejections", [])), "approved": 0},
        "v03": {"relations": len(relations), "rules": len(extraction["rules"]), "rejected": len(extraction["rejections"]), "approved": 0,
                "relation_origins": dict(Counter(item.get("origin") for item in relations)),
                "rejection_reasons": dict(Counter(item.get("reason_code", "unknown") for item in extraction["rejections"]))},
        "v02_directory_sha256": directory_sha256(v02_dir, exclude_names={"quality-comparison.json"}),
    })
    source = _load_json(source_manifest)
    book_manifest = build_book_manifest_from_packages(book={"book_id": "clinical-hematology", "title": "Clinical Hematology", "edition": "source-package"}, source_manifest=source, chunk_manifest=manifest)
    pages_text = tuple(PageText(page["page_id"], (source_manifest.parent / page["raw_path"]).read_text(encoding="utf-8"), (source_manifest.parent / page["cleaned_path"]).read_text(encoding="utf-8")) for page in source["pages"])
    records, by_key = _graph_records(v02, chunks)
    graph_relations = []
    for item in relations:
        source_record, target_record = by_key.get((item["page_id"], item["source_candidate_key"])), by_key.get((item["page_id"], item["target_candidate_key"]))
        if source_record and target_record:
            graph_relations.append(SemanticRelation(source_record.record_id, item["relation"], target_record.record_id, item["source"]["chunk_id"], item["source"]["exact_quote"], item["relation_cue"], item["source"]["chunk_sha256"], item.get("origin", "model")))
    for item in extraction["rules"]:
        source = item["source"]
        rule = SemanticRecord(item["candidate_id"], "InterpretationRule", "candidate", source["exact_quote"], source["chunk_id"], source["char_start"], source["char_end"], item["semantic_type"], item["subject_logic"], rule_payload=item["components"], candidate_key=item["rule_key"])
        records.append(rule)
        for key in item["subject_candidate_keys"]:
            target = by_key.get((item["page_id"], key))
            if target: graph_relations.append((rule.record_id, "RULE_HAS_SUBJECT", target.record_id))
        target = by_key.get((item["page_id"], item["conclusion_candidate_key"]))
        if target: graph_relations.append((rule.record_id, "RULE_HAS_CONCLUSION", target.record_id))
    graph_path = output / "knowledge.sqlite"
    staging_graph = output / ".knowledge.sqlite.v03.build"
    staging_graph.unlink(missing_ok=True)
    graph = SemanticGraphBuilder().build(staging_graph, v02_dir / "base-knowledge.sqlite", book_manifest, records, graph_relations)
    staging_graph.replace(graph_path)
    run_manifest = {"schema_version": "chapter-semantic-kg-run/v0.3", "provider": "deepseek-direct", "model": MODEL, "status": "candidate-only", "hold": True,
        "input": {"chunk_manifest_sha256": input_hash, "entity_catalog_sha256": catalog_hash, "pages": 24, "chunks": 44},
        "stages": {"relation": {"completed": 24}, "rule": {"completed": 24}},
        "counts": {"relations": len(relations), "rules": len(extraction["rules"]), "rejections": len(extraction["rejections"]), "approved": 0},
        "graph": {"node_count": graph.node_count, "edge_count": graph.edge_count, "package_hash": graph.package_hash}}
    atomic_write_json(output / "run-manifest.json", run_manifest)
    return run_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=ROOT / "source-packages/chunks/chapter-01/manifest.json")
    parser.add_argument("--source", type=Path, default=ROOT / "source-packages/chapter-01/manifest.json")
    parser.add_argument("--v02", type=Path, default=ROOT / "runtime/chapter-01-semantic-kg-deepseek-direct-v0.2")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/chapter-01-semantic-kg-deepseek-direct-v0.3")
    parser.add_argument("--catalog-only", action="store_true", help="write the frozen v0.2-derived catalog without contacting the provider")
    parser.add_argument("--rerun-rule-contract", action="store_true", help="reset only the rule stage after a rule-prompt contract revision")
    args = parser.parse_args()
    key = None if args.catalog_only else sys.stdin.readline().strip()
    if key is None and not args.catalog_only:
        print("DEEPSEEK_API_KEY must be supplied through hidden stdin", file=sys.stderr)
        return 2
    run(args.chunks, args.source, args.v02, args.output, key, args.rerun_rule_contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
