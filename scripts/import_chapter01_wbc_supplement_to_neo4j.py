#!/usr/bin/env python3
"""Idempotently import the Chapter 01 white-cell differential supplement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUPPLEMENT = ROOT / "knowledge/chapter-01/terminology/wbc-differential-supplement-v0.1.json"


def _read_auth_from_stdin() -> tuple[str, str]:
    for line in sys.stdin:
        if line.startswith("NEO4J_AUTH="):
            username, separator, password = line.removeprefix("NEO4J_AUTH=").strip().strip("'\"").partition("/")
            if separator and username and password:
                return username, password
    raise ValueError("stdin does not contain NEO4J_AUTH=username/password")


def load_supplement(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    statistics = payload["statistics"]
    expected_statistics = {
        "has_metric_count": 10,
        "has_state_count": 16,
        "rule_count": 10,
    }
    if any(statistics.get(key) != value for key, value in expected_statistics.items()):
        raise ValueError(f"unexpected supplement statistics: {statistics}")
    if statistics.get("added_entity_count") != len(payload["added_entities"]):
        raise ValueError("supplement entity count does not match payload")
    if payload["status"] != "AUTOMATED_VALIDATION_COMPLETE":
        raise ValueError("supplement has not passed automated validation")
    return payload


def apply_supplement(driver: Any, database: str, payload: dict[str, Any]) -> dict[str, Any]:
    supplement_id = payload["supplement_id"]
    with driver.session(database=database) as session:
        snapshot = session.run(
            "MATCH (m:GraphSnapshot {snapshot_id: 'chapter-01-reviewed-entity-v0.8-relationship-v1.0'}) "
            "RETURN m.entity_count AS entity_count"
        ).single()
        expected_base_entity_count = payload["sources"]["base_entity_count"]
        if snapshot is None or snapshot["entity_count"] != expected_base_entity_count:
            raise RuntimeError("Neo4j base snapshot is not the expected Chapter 01 v0.8/v1.0 graph")

        session.run(
            "CREATE CONSTRAINT supplement_rule_id IF NOT EXISTS "
            "FOR (n:ReferenceRangeRule) REQUIRE n.rule_id IS UNIQUE"
        ).consume()
        session.run(
            "CREATE CONSTRAINT graph_supplement_id IF NOT EXISTS "
            "FOR (n:GraphSupplement) REQUIRE n.supplement_id IS UNIQUE"
        ).consume()
        session.run(
            "UNWIND $rows AS row "
            "MERGE (n:CanonicalEntity:LabIndicator {canonical_id: row.canonical_id}) "
            "SET n.canonical_name = row.canonical_name, n.entity_type = 'LabIndicator', "
            "n.aliases_json = row.aliases_json, n.mention_ids_json = row.mention_ids_json, "
            "n.derivations_json = row.derivations_json, n.official_term_ids_json = row.official_term_ids_json, "
            "n.automation_status = row.automation_status, n.supplement_id = $supplement_id",
            rows=[{
                "canonical_id": item["canonical_id"],
                "canonical_name": item["canonical_name"],
                "aliases_json": json.dumps(item["aliases"], ensure_ascii=False),
                "mention_ids_json": json.dumps(item["mention_ids"], ensure_ascii=False),
                "derivations_json": json.dumps(item["derivations"], ensure_ascii=False),
                "official_term_ids_json": json.dumps(item["official_term_ids"], ensure_ascii=False),
                "automation_status": item["automation_status"],
            } for item in payload["added_entities"]],
            supplement_id=supplement_id,
        ).consume()
        for relation_type in ("HAS_METRIC", "HAS_STATE"):
            rows = [item for item in payload["relationships"] if item["relation_type"] == relation_type]
            session.run(
                "UNWIND $rows AS row "
                "MATCH (source:CanonicalEntity {canonical_id: row.source_canonical_id}) "
                "MATCH (target:CanonicalEntity {canonical_id: row.target_canonical_id}) "
                f"MERGE (source)-[r:{relation_type}]->(target) "
                "SET r.supplement_relationship_id = row.relationship_id, "
                "r.evidence_chunk_id = row.evidence_chunk_id, r.automation_status = 'AUTO_VALIDATED_BOOK_TABLE', "
                "r.supplement_id = $supplement_id",
                rows=rows,
                supplement_id=supplement_id,
            ).consume()
        session.run(
            "UNWIND $rows AS row "
            "MATCH (indicator:CanonicalEntity {canonical_id: row.indicator_canonical_id}) "
            "MERGE (rule:ReferenceRangeRule:AutomatedRule {rule_id: row.rule_id}) "
            "SET rule += row.properties, rule.supplement_id = $supplement_id "
            "MERGE (indicator)-[:RULE_INPUT {supplement_id: $supplement_id}]->(rule)",
            rows=[{
                "rule_id": item["rule_id"],
                "indicator_canonical_id": item["indicator_canonical_id"],
                "properties": {key: value for key, value in item.items() if key not in {"rule_id", "indicator_canonical_id"} and value is not None},
            } for item in payload["rules"]],
            supplement_id=supplement_id,
        ).consume()
        output_rows = [
            {"rule_id": item["rule_id"], "state_id": state_id, "result": result}
            for item in payload["rules"]
            for state_id, result in ((item["high_state_id"], "ABOVE_REFERENCE"), (item["low_state_id"], "BELOW_REFERENCE"))
            if state_id is not None
        ]
        session.run(
            "UNWIND $rows AS row MATCH (rule:ReferenceRangeRule {rule_id: row.rule_id}) "
            "MATCH (state:CanonicalEntity {canonical_id: row.state_id}) "
            "MERGE (rule)-[r:RULE_OUTPUT {result: row.result}]->(state) "
            "SET r.supplement_id = $supplement_id",
            rows=output_rows,
            supplement_id=supplement_id,
        ).consume()
        session.run(
            "MERGE (m:GraphSupplement {supplement_id: $supplement_id}) "
            "SET m.status = 'IMPORTED', m.schema_version = $schema_version, "
            "m.added_entity_count = $added_entity_count, m.relationship_count = $relationship_count, "
            "m.rule_count = $rule_count",
            supplement_id=supplement_id,
            schema_version=payload["schema_version"],
            added_entity_count=len(payload["added_entities"]),
            relationship_count=len(payload["relationships"]),
            rule_count=len(payload["rules"]),
        ).consume()
        result = session.run(
            "MATCH (m:GraphSupplement {supplement_id: $supplement_id}) "
            "OPTIONAL MATCH (n:CanonicalEntity {supplement_id: $supplement_id}) "
            "WITH m, count(n) AS entities "
            "OPTIONAL MATCH ()-[r {supplement_id: $supplement_id}]->() "
            "WITH m, entities, count(r) AS relationships "
            "OPTIONAL MATCH (rule:ReferenceRangeRule {supplement_id: $supplement_id}) "
            "RETURN m.status AS status, entities, relationships, count(rule) AS rules",
            supplement_id=supplement_id,
        ).single()
        return dict(result) if result is not None else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement", type=Path, default=DEFAULT_SUPPLEMENT)
    parser.add_argument("--uri", default="bolt://127.0.0.1:7687")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--validate-only", action="store_true")
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument("--auth-stdin", action="store_true")
    auth.add_argument("--no-auth", action="store_true")
    args = parser.parse_args()
    payload = load_supplement(args.supplement)
    if args.validate_only:
        print(json.dumps(payload["statistics"], ensure_ascii=False, sort_keys=True))
        return 0
    if not (args.auth_stdin or args.no_auth):
        parser.error("one of --auth-stdin or --no-auth is required unless --validate-only is used")
    neo4j_auth = _read_auth_from_stdin() if args.auth_stdin else None
    with GraphDatabase.driver(args.uri, auth=neo4j_auth) as driver:
        driver.verify_connectivity()
        result = apply_supplement(driver, args.database, payload)
    expected = {
        "status": "IMPORTED",
        "entities": len(payload["added_entities"]),
        "relationships": len(payload["relationships"]) + len(payload["rules"]) + sum(
            item["low_state_id"] is not None for item in payload["rules"]
        ) + len(payload["rules"]),
        "rules": len(payload["rules"]),
    }
    if result != expected:
        raise RuntimeError(f"Neo4j supplement verification failed: expected {expected}, got {result}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
