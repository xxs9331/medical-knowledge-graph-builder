#!/usr/bin/env python3
"""Replace Neo4j staging with the reviewed Chapter 01 entity/relation snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENTITIES = ROOT / "evaluation/chapter-01/chapter-01-canonical-entities-v0.8.json"
DEFAULT_RELATIONSHIPS = ROOT / "evaluation/chapter-01/chapter-01-relationship-gold-v1.0.json"
ALLOWED_ENTITY_TYPES = {"ClinicalContext", "Disease", "IndicatorState", "LabIndicator", "LabPanel"}
ALLOWED_RELATION_TYPES = {
    "ASSOCIATED_WITH", "CAUSES", "HAS_METRIC", "HAS_STATE", "INDICATES", "IS_A",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batches(rows: list[dict[str, Any]], size: int = 250) -> Iterable[list[dict[str, Any]]]:
    for offset in range(0, len(rows), size):
        yield rows[offset:offset + size]


def _read_auth_from_stdin() -> tuple[str, str]:
    for line in sys.stdin:
        if not line.startswith("NEO4J_AUTH="):
            continue
        value = line.removeprefix("NEO4J_AUTH=").strip().strip("'\"")
        username, separator, password = value.partition("/")
        if separator and username and password:
            return username, password
    raise ValueError("stdin does not contain a valid NEO4J_AUTH=username/password entry")


def _load_snapshot(entity_path: Path, relationship_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    entity_doc = json.loads(entity_path.read_text(encoding="utf-8"))
    relationship_doc = json.loads(relationship_path.read_text(encoding="utf-8"))
    entities = entity_doc["canonical_entities"]
    relationships = [
        relationship
        for case in relationship_doc["cases"]
        for relationship in case["relationships"]
    ]
    entity_ids = {entity["canonical_id"] for entity in entities}
    if len(entity_ids) != len(entities):
        raise ValueError("canonical entity IDs are not unique")
    if any(entity["entity_type"] not in ALLOWED_ENTITY_TYPES for entity in entities):
        raise ValueError("snapshot contains an unsupported entity type")
    identities = {
        (item["source_canonical_id"], item["relation_type"], item["target_canonical_id"])
        for item in relationships
    }
    if len(identities) != len(relationships):
        raise ValueError("relationship identities are not unique")
    for relationship in relationships:
        if relationship["relation_type"] not in ALLOWED_RELATION_TYPES:
            raise ValueError(f"unsupported relationship type: {relationship['relation_type']}")
        if not {
            relationship["source_canonical_id"], relationship["target_canonical_id"],
        } <= entity_ids:
            raise ValueError(f"dangling relationship: {relationship['gold_relation_id']}")
    metadata = {
        "snapshot_id": "chapter-01-reviewed-entity-v0.8-relationship-v1.0",
        "entity_schema_version": entity_doc["schema_version"],
        "relationship_schema_version": relationship_doc["schema_version"],
        "entity_source_sha256": _sha256(entity_path),
        "relationship_source_sha256": _sha256(relationship_path),
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "publication_scope": "REVIEWED_ENTITY_AND_ORDINARY_RELATION_STAGING",
    }
    return entities, relationships, metadata


def _import_snapshot(driver: Any, database: str, entities: list[dict[str, Any]], relationships: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    entities_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        entities_by_type[entity["entity_type"]].append({
            "canonical_id": entity["canonical_id"],
            "canonical_name": entity["canonical_name"],
            "entity_type": entity["entity_type"],
            "aliases_json": json.dumps(entity.get("aliases", []), ensure_ascii=False),
            "mention_ids_json": json.dumps(entity.get("mention_ids", []), ensure_ascii=False),
            "derivations_json": json.dumps(entity.get("derivations", []), ensure_ascii=False),
            "review_status": entity.get("review_status"),
            "snapshot_id": metadata["snapshot_id"],
        })
    relationships_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relationship in relationships:
        relationships_by_type[relationship["relation_type"]].append({
            "source_id": relationship["source_canonical_id"],
            "target_id": relationship["target_canonical_id"],
            "gold_relation_id": relationship["gold_relation_id"],
            "evidence_chunk_ids_json": json.dumps(relationship["evidence_chunk_ids"], ensure_ascii=False),
            "annotation_rationale": relationship["annotation_rationale"],
            "review_status": relationship["review_status"],
            "snapshot_id": metadata["snapshot_id"],
        })

    with driver.session(database=database) as session:
        before = session.run(
            "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS relationships"
        ).single().data()
        session.run("MATCH (n) DETACH DELETE n").consume()
        session.run(
            "CREATE CONSTRAINT canonical_entity_id IF NOT EXISTS "
            "FOR (n:CanonicalEntity) REQUIRE n.canonical_id IS UNIQUE"
        ).consume()
        for entity_type, rows in sorted(entities_by_type.items()):
            query = (
                f"UNWIND $rows AS row CREATE (n:CanonicalEntity:{entity_type}) "
                "SET n = row"
            )
            for batch in _batches(rows):
                session.run(query, rows=batch).consume()
        for relation_type, rows in sorted(relationships_by_type.items()):
            query = (
                "UNWIND $rows AS row "
                "MATCH (source:CanonicalEntity {canonical_id: row.source_id}) "
                "MATCH (target:CanonicalEntity {canonical_id: row.target_id}) "
                f"CREATE (source)-[r:{relation_type}]->(target) "
                "SET r.gold_relation_id = row.gold_relation_id, "
                "r.evidence_chunk_ids_json = row.evidence_chunk_ids_json, "
                "r.annotation_rationale = row.annotation_rationale, "
                "r.review_status = row.review_status, r.snapshot_id = row.snapshot_id"
            )
            for batch in _batches(rows):
                session.run(query, rows=batch).consume()
        session.run("CREATE (metadata:GraphSnapshot) SET metadata = $metadata", metadata=metadata).consume()
        after = session.run(
            "MATCH (n:CanonicalEntity) WITH count(n) AS entities "
            "MATCH (:CanonicalEntity)-[r]->(:CanonicalEntity) "
            "RETURN entities, count(r) AS relationships"
        ).single().data()
        snapshot = session.run(
            "MATCH (m:GraphSnapshot {snapshot_id: $snapshot_id}) RETURN m.entity_count AS entities, "
            "m.relationship_count AS relationships",
            snapshot_id=metadata["snapshot_id"],
        ).single().data()
    return {"before": before, "after": after, "metadata": snapshot}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    parser.add_argument("--relationships", type=Path, default=DEFAULT_RELATIONSHIPS)
    parser.add_argument("--uri", default="bolt://127.0.0.1:7687")
    parser.add_argument("--database", default="neo4j")
    auth_group = parser.add_mutually_exclusive_group(required=True)
    auth_group.add_argument("--auth-stdin", action="store_true")
    auth_group.add_argument("--no-auth", action="store_true")
    args = parser.parse_args()
    auth = None
    if args.auth_stdin:
        auth = _read_auth_from_stdin()
    entities, relationships, metadata = _load_snapshot(args.entities, args.relationships)
    with GraphDatabase.driver(args.uri, auth=auth) as driver:
        driver.verify_connectivity()
        result = _import_snapshot(driver, args.database, entities, relationships, metadata)
    if result["after"] != {
        "entities": metadata["entity_count"],
        "relationships": metadata["relationship_count"],
    } or result["metadata"] != result["after"]:
        raise RuntimeError(f"Neo4j post-import verification failed: {result}")
    print(json.dumps({"snapshot": metadata, **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
