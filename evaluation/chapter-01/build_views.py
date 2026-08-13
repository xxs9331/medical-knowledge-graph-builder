from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
GRAPH_PATH = ROOT / "chapter-01-graph-test-set-v0.1.json"


def _write_view(
    graph: dict[str, Any],
    *,
    filename: str,
    schema_version: str,
    field: str,
    identity: list[str],
) -> None:
    view = {
        "schema_version": schema_version,
        "status": graph["status"],
        "source_graph_dataset": GRAPH_PATH.name,
        "source_document": graph["source_document"],
        "identity": identity,
        "cases": [
            {
                "case_id": case["case_id"],
                "chunk_ids": case["chunk_ids"],
                "expected": case[field],
            }
            for case in graph["cases"]
        ],
    }
    (ROOT / filename).write_text(
        json.dumps(view, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    _write_view(
        graph,
        filename="chapter-01-entity-test-set-v0.1.json",
        schema_version="medical-kg-entity-test-set/v0.1",
        field="entities",
        identity=["entity_type", "mention"],
    )
    _write_view(
        graph,
        filename="chapter-01-relationship-test-set-v0.1.json",
        schema_version="medical-kg-relationship-test-set/v0.1",
        field="relationships",
        identity=["start_mention", "relation_type", "end_mention"],
    )
    _write_view(
        graph,
        filename="chapter-01-rule-test-set-v0.1.json",
        schema_version="medical-kg-rule-test-set/v0.1",
        field="rules",
        identity=["rule_stage", "ordered_inputs", "ordered_outputs", "logic"],
    )


if __name__ == "__main__":
    main()
