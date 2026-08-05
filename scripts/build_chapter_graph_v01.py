#!/usr/bin/env python3
"""Merge Chapter 01 extraction artifacts into triples and a queryable graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_kg_sourceprep.chapter_graph_build import (
    ChapterGraphBuilder,
    build_entity_evidence,
    validate_rule_evidence,
    write_graph_package,
)


DEFAULTS = {
    "entities": ROOT / "runtime/chapter-01-entity-extraction-deepseek-direct-v0.4/entities.json",
    "ontology": ROOT / "runtime/chapter-01-entity-extraction-deepseek-direct-v0.4/ontology-relations.json",
    "catalog": ROOT / "runtime/chapter-01-semantic-kg-deepseek-direct-v0.4/entity-catalog.json",
    "relations": ROOT / "runtime/chapter-01-semantic-kg-deepseek-direct-v0.4/relation-extraction.json",
    "rules": ROOT / "runtime/chapter-01-semantic-kg-deepseek-direct-v0.4/rule-extraction.json",
    "references": ROOT / "runtime/chapter-01-book-rule-library-v0.1/reference-rules.json",
    "core_rules": ROOT / "runtime/chapter-01-book-rule-library-v0.1/core-rules.json",
    "temporal_rules": ROOT / "runtime/chapter-01-book-rule-library-v0.1/temporal-rules.json",
    "rule_quality": ROOT / "runtime/chapter-01-book-rule-library-v0.1/quality-report.json",
    "manual": ROOT / "runtime/chapter-01-indicator-rule-functions-reviewed-v0.2/manual-review.json",
    "entity_checkpoint": ROOT / "runtime/chapter-01-entity-extraction-deepseek-direct-v0.3/checkpoint.json",
    "rule_alignment": ROOT / "runtime/chapter-01-entity-extraction-deepseek-direct-v0.4/rule-alignment.json",
    "source_manifest": ROOT / "source-packages/chapter-01/manifest.json",
    "chunk_manifest": ROOT / "source-packages/chunks/chapter-01/manifest.json",
    "output": ROOT / "runtime/chapter-01-knowledge-graph-v0.2",
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name}", type=Path, default=default)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in DEFAULTS if name != "output"}
    checkpoint = _load(paths["entity_checkpoint"])
    source_manifest = _load(paths["source_manifest"])
    entity_evidence = build_entity_evidence(checkpoint, source_manifest, paths["source_manifest"].parent)
    reference_rules = _load(paths["references"])
    core_rules = _load(paths["core_rules"])
    temporal_rules = _load(paths["temporal_rules"])
    evidence_anchors_replayed = validate_rule_evidence(
        [reference_rules, core_rules, temporal_rules],
        _load(paths["chunk_manifest"]),
        paths["chunk_manifest"].parent,
    )
    for slot in _load(paths["rule_alignment"]).get("slots", []):
        slot_name = slot.get("rule_slot")
        if not slot_name:
            continue
        values = entity_evidence.setdefault(slot_name, [])
        for source in slot.get("source_evidence", []):
            evidence = {
                **source,
                "exact_quote": source.get("quote"),
                "evidence_kind": (
                    "verbatim" if slot_name in source.get("quote", "") else "context-derived"
                ),
                "derivation_reason": slot.get("reason"),
            }
            if evidence not in values:
                values.append(evidence)
    graph = ChapterGraphBuilder().build(
        _load(paths["entities"]), _load(paths["ontology"]), _load(paths["catalog"]),
        _load(paths["relations"]), _load(paths["rules"]), reference_rules,
        core_rules, temporal_rules, _load(paths["rule_quality"]),
        _load(paths["manual"]), entity_evidence,
    )
    write_graph_package(
        args.output,
        graph,
        {name: _sha(path) for name, path in paths.items()},
        {"book_rule_evidence_anchors_replayed": evidence_anchors_replayed},
    )
    print(json.dumps({"output": str(args.output), **graph["counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
