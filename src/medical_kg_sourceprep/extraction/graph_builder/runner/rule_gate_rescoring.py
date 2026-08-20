"""在冻结规则候选上离线重评分当前确定性门控。"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[4]))
    __package__ = "medical_kg_sourceprep.extraction.graph_builder.runner"

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...llm_extraction import atomic_write_json, load_chunk_manifest
from ..contract import DEFAULT_CHUNK_MANIFEST, DEFAULT_SCHEMA_PATH, PROJECT_ROOT
from ..evaluation.artifacts import load_json_object
from ..evaluation.scoring import merge_candidate_graphs, score_candidate_graph
from ..rule_gate import partition_invalid_rules
from ..schema import load_candidate_graph_schema
from .rule_semantic_gate_evaluation import _rule_graph, _rule_metrics, _string_list


GOLD_PATH = PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json"
INPUT_ROOT = PROJECT_ROOT / "runtime/evaluations/rule-coordination-r2"
OUTPUT_PATH = (
    PROJECT_ROOT / "runtime/evaluations/rule-example-output-gate-r3/evaluation-result.json"
)
RUN_IDS = tuple(f"20260817-rule-coordination-r0{index}" for index in range(1, 4))


def _chunk_graph_path(run_root: Path, chunk_id: str) -> Path:
    suffix = f"{chunk_id.split(':')[-2]}-{chunk_id.split(':')[-1]}"
    return run_root / "chunks" / suffix / "r1-rule-graph.json"


def run_rule_gate_rescoring(*, output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    """对三次 R3 冻结候选应用当前门控并重新计算规则指标。"""
    gold = load_json_object(GOLD_PATH)
    cases = [item for item in gold.get("cases", []) if isinstance(item, Mapping)]
    _manifest, chunks = load_chunk_manifest(DEFAULT_CHUNK_MANIFEST)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    schema = load_candidate_graph_schema(DEFAULT_SCHEMA_PATH)
    runs: list[dict[str, Any]] = []
    for run_id in RUN_IDS:
        input_run = INPUT_ROOT / run_id
        chunk_ids = list(dict.fromkeys(
            str(chunk_id)
            for case in cases
            for chunk_id in _string_list(case.get("chunk_ids", []))
        ))
        gated_graphs: dict[str, dict[str, list[Any]]] = {}
        rejections: list[dict[str, Any]] = []
        for chunk_id in chunk_ids:
            graph = load_json_object(_chunk_graph_path(input_run, chunk_id))
            business_nodes = [
                item for item in graph.get("nodes", [])
                if isinstance(item, Mapping) and item.get("entity_type") != "RuleDefinition"
            ]
            rule_nodes = [
                item for item in graph.get("nodes", [])
                if isinstance(item, Mapping) and item.get("entity_type") == "RuleDefinition"
            ]
            accepted_rules, rejected_rules = partition_invalid_rules(rule_nodes)
            gated_graph, structure = _rule_graph(
                schema=schema,
                business_nodes=business_nodes,
                rule_nodes=accepted_rules,
            )
            gated_graphs[chunk_id] = gated_graph
            rejections.extend({"chunk_id": chunk_id, **item} for item in rejected_rules)
            if structure["invalid_rule_keys"]:
                raise RuntimeError(f"unexpected_invalid_rule_structure:{chunk_id}")

        case_results: list[dict[str, Any]] = []
        for case in cases:
            case_chunk_ids = _string_list(case.get("chunk_ids", []))
            source_text = "\n\n".join(
                chunks_by_id[chunk_id].text for chunk_id in case_chunk_ids
            )
            score = score_candidate_graph(
                merge_candidate_graphs(gated_graphs[chunk_id] for chunk_id in case_chunk_ids),
                case,
                source_text=source_text,
            )["rules"]
            case_results.append({"case_id": case["case_id"], "r4": score})
        runs.append({
            "run_id": run_id,
            "metrics": _rule_metrics(case_results, "r4"),
            "gate_rejections": rejections,
            "cases": case_results,
        })
    result = {
        "schema_version": "rule-gate-rescoring/v0.1",
        "status": "evaluation-only",
        "publication_status": "HOLD",
        "model_calls": 0,
        "gold_exposed_to_model": False,
        "gate_contract": [
            "rule_inputs_intersect_rule_outputs_is_rejected",
            "explicit_example_output_is_rejected",
        ],
        "runs": runs,
    }
    atomic_write_json(output_path, result)
    return result


if __name__ == "__main__":
    report = run_rule_gate_rescoring()
    print(json.dumps(
        {item["run_id"]: item["metrics"] for item in report["runs"]},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))
