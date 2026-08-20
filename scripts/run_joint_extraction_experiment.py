#!/usr/bin/env python3
"""在固定实体工件上运行联合抽取 C/D 组开发实验。"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from medical_kg_sourceprep.extraction.graph_builder.client import create_deepseek_graph_builder
from medical_kg_sourceprep.extraction.graph_builder.contract import (
    DEFAULT_CHUNK_MANIFEST,
    DEFAULT_SCHEMA_PATH,
    PROJECT_ROOT,
    GraphBuilderConfigurationError,
)
from medical_kg_sourceprep.extraction.graph_builder.evaluation.aggregation import aggregate_supervised_prf1
from medical_kg_sourceprep.extraction.graph_builder.evaluation.scoring import merge_candidate_graphs, score_candidate_graph
from medical_kg_sourceprep.extraction.graph_builder.joint_extraction import extract_joint_candidates
from medical_kg_sourceprep.extraction.graph_builder.schema import load_candidate_graph_schema
from medical_kg_sourceprep.extraction.llm_extraction import atomic_write_json, load_chunk_manifest


GOLD_PATH = PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json"
BASELINE_ROOT = PROJECT_ROOT / "runtime/evaluations/typical-cases/structured-rules-v0.10-full"
OUTPUT_ROOT = PROJECT_ROOT / "runtime/evaluations/typical-cases/joint-extraction-v0.1-pilot"
DEFAULT_CASE_IDS = ("TC-01", "TC-03", "TC-08")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GraphBuilderConfigurationError(f"joint_experiment_json_not_object:{path}")
    return value


def _baseline_graph_path(root: Path, chunk_id: str) -> Path:
    slug = "-".join(chunk_id.rsplit(":", 2)[-2:])
    return root / "chunks" / slug / "candidate-graph" / "graph.json"


def _rule_projection(graph: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rule_keys = {
        str(node["candidate_key"])
        for node in graph.get("nodes", [])
        if isinstance(node, Mapping) and node.get("entity_type") == "RuleDefinition"
    }
    nodes = [
        dict(node) for node in graph.get("nodes", [])
        if isinstance(node, Mapping) and str(node.get("candidate_key")) in rule_keys
    ]
    relationships = [
        dict(relation) for relation in graph.get("relationships", [])
        if isinstance(relation, Mapping)
        and relation.get("relation_type") in {"RULE_INPUT", "RULE_OUTPUT"}
    ]
    return nodes, relationships


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    gold = _load_object(cast(Path, args.gold))
    if gold.get("status") not in {"HUMAN_VALIDATED", "HUMAN_REVIEW_REQUIRED"}:
        raise GraphBuilderConfigurationError("joint_experiment_gold_status_invalid")
    requested = set(cast(list[str], args.case_ids))
    cases = [
        case for case in gold.get("cases", [])
        if isinstance(case, dict) and case.get("case_id") in requested
    ]
    if {str(case["case_id"]) for case in cases} != requested:
        raise GraphBuilderConfigurationError("joint_experiment_case_missing")

    _manifest, chunks = load_chunk_manifest(cast(Path, args.manifest))
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    schema = load_candidate_graph_schema(cast(Path, args.schema))
    baseline_root = cast(Path, args.baseline_root)
    output_root = cast(Path, args.output_root)

    cases_by_chunk: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        for chunk_id in case.get("chunk_ids", []):
            cases_by_chunk[str(chunk_id)].append(str(case["case_id"]))

    rescore_only = bool(args.rescore_only)
    client = None if rescore_only else create_deepseek_graph_builder()
    baseline_graphs: dict[str, dict[str, Any]] = {}
    joint_graphs: dict[str, dict[str, Any]] = {}
    union_graphs: dict[str, dict[str, Any]] = {}
    full_joint_only = bool(args.full_joint_only)
    include_baseline_union = bool(args.include_baseline_union)
    try:
        for index, chunk_id in enumerate(cases_by_chunk, start=1):
            print(f"[{index}/{len(cases_by_chunk)}] {chunk_id}", flush=True)
            if chunk_id not in chunk_by_id:
                raise GraphBuilderConfigurationError(f"joint_experiment_chunk_missing:{chunk_id}")
            baseline = _load_object(_baseline_graph_path(baseline_root, chunk_id))
            baseline_graphs[chunk_id] = baseline
            slug = "-".join(chunk_id.rsplit(":", 2)[-2:])
            if rescore_only:
                graph_name = "group-b-graph.json" if full_joint_only else "group-c-graph.json"
                joint_graph = _load_object(output_root / "chunks" / slug / graph_name)
            else:
                if client is None:
                    raise GraphBuilderConfigurationError("joint_experiment_client_missing")
                result = await extract_joint_candidates(
                    client,
                    chunk=chunk_by_id[chunk_id],
                    schema=schema,
                    frozen_nodes=([] if full_joint_only else [
                        node for node in baseline.get("nodes", []) if isinstance(node, Mapping)
                    ]),
                )
                rule_nodes, rule_relationships = _rule_projection(baseline)
                joint_graph = {
                    "nodes": [*result["nodes"], *rule_nodes],
                    "relationships": [*result["relationships"], *rule_relationships],
                }
                atomic_write_json(output_root / "chunks" / slug / "joint-extraction.json", result)
            joint_graphs[chunk_id] = joint_graph
            if full_joint_only:
                if not rescore_only:
                    atomic_write_json(output_root / "chunks" / slug / "group-b-graph.json", joint_graph)
                if include_baseline_union:
                    union_graphs[chunk_id] = merge_candidate_graphs((baseline, joint_graph))
                    atomic_write_json(
                        output_root / "chunks" / slug / "group-d-union-graph.json",
                        union_graphs[chunk_id],
                    )
            else:
                union_graphs[chunk_id] = merge_candidate_graphs((baseline, joint_graph))
                if not rescore_only:
                    atomic_write_json(output_root / "chunks" / slug / "group-c-graph.json", joint_graph)
                    atomic_write_json(
                        output_root / "chunks" / slug / "group-d-union-graph.json",
                        union_graphs[chunk_id],
                    )
    finally:
        if client is not None:
            await client.aclose()

    if full_joint_only:
        comparison_groups = (
            (("A", baseline_graphs), ("B", joint_graphs), ("D", union_graphs))
            if include_baseline_union
            else (("A", baseline_graphs), ("B", joint_graphs))
        )
    else:
        comparison_groups = (("A", baseline_graphs), ("C", joint_graphs), ("D", union_graphs))
    group_scores: dict[str, list[dict[str, Any]]] = {
        group: [] for group, _graphs in comparison_groups
    }
    for case in cases:
        chunk_ids = [str(value) for value in case["chunk_ids"]]
        source_text = "\n\n".join(chunk_by_id[value].text for value in chunk_ids)
        for group, graphs in comparison_groups:
            score = score_candidate_graph(
                merge_candidate_graphs(graphs[value] for value in chunk_ids),
                case,
                source_text=source_text,
            )
            group_scores[group].append({
                "case_id": case["case_id"],
                "chunk_ids": chunk_ids,
                "score": score,
            })

    report = {
        "schema_version": "joint-extraction-experiment/v0.1",
        "joint_protocol_version": "joint-entity-relation-candidates/v0.2",
        "status": "development-evaluation-only",
        "publication_status": "HOLD",
        "gold_status": gold.get("status"),
        "gold_exposed_to_model": False,
        "experiment_mode": (
            "full_joint_with_baseline_union"
            if full_joint_only and include_baseline_union
            else "full_joint" if full_joint_only else "frozen_entity_joint"
        ),
        "groups": {
            group: {
                "prf1": aggregate_supervised_prf1(scores, "score"),
                "cases": scores,
            }
            for group, scores in group_scores.items()
        },
    }
    atomic_write_json(output_root / "experiment-comparison.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行固定实体上的联合抽取 C/D 组实验")
    _ = parser.add_argument("--case-id", action="append", dest="case_ids", default=[])
    _ = parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    _ = parser.add_argument("--manifest", type=Path, default=DEFAULT_CHUNK_MANIFEST)
    _ = parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    _ = parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    _ = parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    _ = parser.add_argument(
        "--full-joint-only",
        action="store_true",
        help="运行 B 组：不向联合模型提供任何冻结实体，只比较 A/B。",
    )
    _ = parser.add_argument(
        "--rescore-only",
        action="store_true",
        help="不调用模型，直接读取输出目录中的组图并按当前评分器重新计分。",
    )
    _ = parser.add_argument(
        "--include-baseline-union",
        action="store_true",
        help="端到端 B 组之外，同时评估分阶段基线 A 与 B 的候选并集 D。",
    )
    arguments = parser.parse_args()
    if not arguments.case_ids:
        arguments.case_ids = list(DEFAULT_CASE_IDS)
    result = asyncio.run(_run(arguments))
    print(json.dumps({
        group: value["prf1"] for group, value in result["groups"].items()
    }, ensure_ascii=False, indent=2, sort_keys=True))
