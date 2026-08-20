"""合成并评测当前最佳实体、关系和规则候选图。"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[4]))
    __package__ = "medical_kg_sourceprep.extraction.graph_builder.runner"

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...artifacts import sha256_path
from ...llm_extraction import atomic_write_json, load_chunk_manifest
from ..contract import DEFAULT_CHUNK_MANIFEST, DEFAULT_SCHEMA_PATH, PROJECT_ROOT
from ..evaluation.aggregation import aggregate_supervised_prf1
from ..evaluation.artifacts import load_json_object
from ..evaluation.scoring import merge_candidate_graphs, score_candidate_graph
from ..rule_gate import partition_invalid_rules
from ..schema import load_candidate_graph_schema
from ..validation import build_rule_relationships_from_definitions
from .rule_semantic_gate_evaluation import _string_list


GOLD_PATH = PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json"
T1_ROOT = PROJECT_ROOT / "runtime/evaluations/table-context-prompt-t1-full"
RULE_ROOT = PROJECT_ROOT / "runtime/evaluations/rule-coordination-r2"
OUTPUT_ROOT = PROJECT_ROOT / "runtime/evaluations/best-pipeline-v0.1"
RUN_PAIRS = tuple(
    (
        f"20260817-table-context-full-r0{index}",
        f"20260817-rule-coordination-r0{index}",
    )
    for index in range(1, 4)
)


def _chunk_slug(chunk_id: str) -> str:
    return f"{chunk_id.split(':')[-2]}-{chunk_id.split(':')[-1]}"


def _t1_graph_path(run_id: str, chunk_id: str) -> Path:
    return T1_ROOT / run_id / "chunks" / _chunk_slug(chunk_id) / "hybrid-replace-graph.json"


def _rule_graph_path(run_id: str, chunk_id: str) -> Path:
    return RULE_ROOT / run_id / "chunks" / _chunk_slug(chunk_id) / "r1-rule-graph.json"


def _build_rule_edges(
    *,
    schema: Mapping[str, Any],
    business_nodes: Sequence[Mapping[str, Any]],
    rule_nodes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """在最终业务节点上重建规则边，并剔除端点或结构无效的规则。"""
    retained_rules = [dict(item) for item in rule_nodes]
    first = build_rule_relationships_from_definitions(
        schema=schema,
        nodes=[*business_nodes, *retained_rules],
    )
    invalid_keys = sorted(first.invalid_rule_keys)
    if invalid_keys:
        retained_rules = [
            item for item in retained_rules if item.get("candidate_key") not in invalid_keys
        ]
        first = build_rule_relationships_from_definitions(
            schema=schema,
            nodes=[*business_nodes, *retained_rules],
        )
    return retained_rules, list(first.accepted), invalid_keys


def _compose_chunk_graph(
    *,
    schema: Mapping[str, Any],
    t1_graph: Mapping[str, Any],
    rule_graph: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """用 T1-H 业务图替换规则图的业务端点，得到完整候选图。"""
    business_nodes = [
        item for item in t1_graph.get("nodes", [])
        if isinstance(item, Mapping) and item.get("entity_type") != "RuleDefinition"
    ]
    proposed_rules = [
        item for item in rule_graph.get("nodes", [])
        if isinstance(item, Mapping) and item.get("entity_type") == "RuleDefinition"
    ]
    gated_rules, gate_rejections = partition_invalid_rules(proposed_rules)
    retained_rules, rule_edges, invalid_keys = _build_rule_edges(
        schema=schema,
        business_nodes=business_nodes,
        rule_nodes=gated_rules,
    )
    ordinary_relationships = [
        item for item in t1_graph.get("relationships", [])
        if isinstance(item, Mapping)
        and item.get("relation_type") not in {"RULE_INPUT", "RULE_OUTPUT"}
    ]
    graph = {
        "schema_version": "best-pipeline-candidate-graph/v0.1",
        "status": "candidate-only",
        "publication_status": "HOLD",
        "nodes": [*map(dict, business_nodes), *retained_rules],
        "relationships": [*map(dict, ordinary_relationships), *rule_edges],
    }
    return graph, {
        "business_node_count": len(business_nodes),
        "ordinary_relationship_count": len(ordinary_relationships),
        "proposed_rule_count": len(proposed_rules),
        "retained_rule_count": len(retained_rules),
        "rule_edge_count": len(rule_edges),
        "gate_rejections": gate_rejections,
        "invalid_rule_keys": invalid_keys,
    }


def _mean_metrics(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """计算三次 P/R/F1 百分比的算术均值。"""
    categories = ("entities", "relationships", "rules", "graph")
    return {
        category: {
            metric: round(
                sum(float(run["prf1"]["categories"].get(category, run["prf1"])[metric])
                    for run in runs) / len(runs),
                2,
            )
            for metric in ("precision_percent", "recall_percent", "f1_percent")
        }
        for category in categories
    }


def run_best_pipeline_evaluation(*, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """对三组对应冻结重复合成完整候选图并重新评分。"""
    gold = load_json_object(GOLD_PATH)
    cases = [item for item in gold.get("cases", []) if isinstance(item, Mapping)]
    selected_chunk_ids = list(dict.fromkeys(
        chunk_id for case in cases for chunk_id in _string_list(case.get("chunk_ids", []))
    ))
    _manifest, chunks = load_chunk_manifest(DEFAULT_CHUNK_MANIFEST)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    schema = load_candidate_graph_schema(DEFAULT_SCHEMA_PATH)
    runs: list[dict[str, Any]] = []

    for repeat_index, (t1_run_id, rule_run_id) in enumerate(RUN_PAIRS, start=1):
        run_output = output_root / f"repeat-{repeat_index:02d}"
        graphs: dict[str, dict[str, Any]] = {}
        chunk_records: dict[str, dict[str, Any]] = {}
        for chunk_id in selected_chunk_ids:
            t1_path = _t1_graph_path(t1_run_id, chunk_id)
            rule_path = _rule_graph_path(rule_run_id, chunk_id)
            graph, composition = _compose_chunk_graph(
                schema=schema,
                t1_graph=load_json_object(t1_path),
                rule_graph=load_json_object(rule_path),
            )
            graphs[chunk_id] = graph
            graph_path = run_output / "chunks" / _chunk_slug(chunk_id) / "graph.json"
            atomic_write_json(graph_path, graph)
            chunk_records[chunk_id] = {
                "t1_input": {"path": str(t1_path), "sha256": sha256_path(t1_path)},
                "rule_input": {"path": str(rule_path), "sha256": sha256_path(rule_path)},
                "output": {"path": str(graph_path), "sha256": sha256_path(graph_path)},
                "composition": composition,
            }

        case_results: list[dict[str, Any]] = []
        for case in cases:
            chunk_ids = _string_list(case.get("chunk_ids", []))
            source_text = "\n\n".join(chunks_by_id[chunk_id].text for chunk_id in chunk_ids)
            score = score_candidate_graph(
                merge_candidate_graphs(graphs[chunk_id] for chunk_id in chunk_ids),
                case,
                source_text=source_text,
            )
            case_results.append({
                "case_id": case["case_id"],
                "chunk_ids": chunk_ids,
                "score": score,
            })
        prf1 = aggregate_supervised_prf1(case_results, "score")
        run = {
            "repeat_index": repeat_index,
            "t1_run_id": t1_run_id,
            "rule_run_id": rule_run_id,
            "prf1": prf1,
            "cases": case_results,
            "chunks": chunk_records,
        }
        atomic_write_json(run_output / "evaluation-result.json", run)
        runs.append(run)

    result = {
        "schema_version": "best-pipeline-evaluation/v0.1",
        "status": "development-evaluation-only",
        "publication_status": "HOLD",
        "gold_exposed_to_model": False,
        "model_calls": 0,
        "case_count": len(cases),
        "unique_chunk_count": len(selected_chunk_ids),
        "composition": (
            "T1-H business nodes and ordinary relationships + rule-semantic-prompt/v0.3 "
            "+ deterministic self-reference/example-output gate"
        ),
        "runs": runs,
        "three_run_mean_percent": _mean_metrics(runs),
        "boundary": (
            "Frozen component composition on eight development cases; not an independent test, "
            "full-book result, or publication approval."
        ),
    }
    atomic_write_json(output_root / "evaluation-result.json", result)
    return result


if __name__ == "__main__":
    report = run_best_pipeline_evaluation()
    print(json.dumps({
        "runs": [
            {
                "repeat_index": item["repeat_index"],
                "categories": item["prf1"]["categories"],
                "graph": item["prf1"]["graph"],
            }
            for item in report["runs"]
        ],
        "three_run_mean_percent": report["three_run_mean_percent"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
