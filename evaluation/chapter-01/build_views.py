from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
VERSION = "v0.3"
GRAPH_PATH = ROOT / f"chapter-01-graph-test-set-{VERSION}.json"


def _write_view(
    graph: dict[str, Any],
    *,
    filename: str,
    schema_version: str,
    field: str,
    identity: list[str],
    case_extras: tuple[str, ...] = (),
) -> None:
    view = {
        "schema_version": schema_version,
        "status": graph["status"],
        "scoring_status": graph["scoring_status"],
        "source_graph_dataset": GRAPH_PATH.name,
        "source_document": graph["source_document"],
        "source_chunk_manifest": graph["source_chunk_manifest"],
        "source_chunk_manifest_sha256": graph["source_chunk_manifest_sha256"],
        "annotation_method": graph["annotation_method"],
        "scope_contract": graph["scope_contract"],
        "identity": identity,
        "cases": [
            {
                "case_id": case["case_id"],
                "chunk_ids": case["chunk_ids"],
                "evaluation_scopes": case["evaluation_scopes"],
                "expected": case[field],
                **(
                    {"forbidden": case["must_not_extract"]}
                    if field == "relationships" else {}
                ),
                **{key: case[key] for key in case_extras if key in case},
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
        filename=f"chapter-01-entity-test-set-{VERSION}.json",
        schema_version="medical-kg-entity-test-set/v0.3",
        field="entities",
        identity=["entity_type", "mention"],
    )
    _write_view(
        graph,
        filename=f"chapter-01-relationship-test-set-{VERSION}.json",
        schema_version="medical-kg-relationship-test-set/v0.3",
        field="relationships",
        identity=["start_mention", "relation_type", "end_mention"],
        case_extras=("review_notes", "held_semantics"),
    )
    _write_view(
        graph,
        filename=f"chapter-01-rule-test-set-{VERSION}.json",
        schema_version="medical-kg-rule-test-set/v0.3",
        field="rules",
        identity=["rule_stage", "ordered_inputs", "ordered_outputs", "logic"],
        case_extras=("executor_rules", "held_rules", "review_notes"),
    )


if __name__ == "__main__":
    main()
