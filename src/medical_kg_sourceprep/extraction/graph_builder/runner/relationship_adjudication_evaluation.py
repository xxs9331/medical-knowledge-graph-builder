"""对结构关系候选并集运行通用 Judge 裁决实验。"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[4]))
    __package__ = "medical_kg_sourceprep.extraction.graph_builder.runner"

import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...llm_extraction import EvidenceChunk, atomic_write_json, load_chunk_manifest
from ..client import create_deepseek_graph_builder
from ..contract import DEFAULT_CHUNK_MANIFEST, DEFAULT_SCHEMA_PATH, PROJECT_ROOT
from ..evaluation.aggregation import aggregate_supervised_prf1
from ..evaluation.artifacts import load_json_object
from ..evaluation.scoring import merge_candidate_graphs, score_candidate_graph
from ..judge import JUDGE_BATCH_SIZE, judge_candidate_graph
from ..relation_adjudication import (
    apply_relationship_judgments,
    build_relationship_candidate_union,
    mask_text_outside_ranges,
)
from ..schema import load_candidate_graph_schema


GOLD_PATH = PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json"
BASELINE_ROOT = PROJECT_ROOT / "runtime/evaluations/best-pipeline-v0.1"
PROPOSAL_ROOT = (
    PROJECT_ROOT / "runtime/evaluations/relationship-structure-refinement-v0.3-authorized"
)
OUTPUT_ROOT = PROJECT_ROOT / "runtime/evaluations/relationship-candidate-adjudication-v0.1"


def _chunk_slug(chunk_id: str) -> str:
    return "-".join(chunk_id.rsplit(":", 2)[-2:])


def _baseline_graph_path(repeat_index: int, chunk_id: str) -> Path:
    return (
        BASELINE_ROOT
        / f"repeat-{repeat_index:02d}"
        / "chunks"
        / _chunk_slug(chunk_id)
        / "graph.json"
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("relationship_adjudication_string_list_invalid")
    return list(value)


def _proposal_records(repeat_index: int) -> list[Mapping[str, Any]]:
    report = load_json_object(PROPOSAL_ROOT / "evaluation-result.json")
    raw_runs = report.get("runs")
    runs = raw_runs if isinstance(raw_runs, list) else []
    run = next(
        item
        for item in runs
        if isinstance(item, Mapping) and item.get("repeat_index") == repeat_index
    )
    raw_routes = run.get("routes")
    routes = raw_routes if isinstance(raw_routes, list) else []
    return [item for item in routes if isinstance(item, Mapping)]


def _masked_chunk(chunk: EvidenceChunk, ranges: Sequence[tuple[int, int]]) -> EvidenceChunk:
    text = mask_text_outside_ranges(chunk.text, ranges)
    return EvidenceChunk(
        chunk.chunk_id,
        text,
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def prepare_adjudication_inputs(
    *, repeat_index: int, output_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """从冻结基线和已保存提案生成 Judge 输入，不调用模型。"""
    _manifest, chunks = load_chunk_manifest(DEFAULT_CHUNK_MANIFEST)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    records_by_chunk: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in _proposal_records(repeat_index):
        chunk_id = record.get("chunk_id")
        if isinstance(chunk_id, str):
            records_by_chunk[chunk_id].append(record)

    prepared: dict[str, dict[str, Any]] = {}
    total_candidates = 0
    total_batches = 0
    for chunk_id, records in records_by_chunk.items():
        baseline = load_json_object(_baseline_graph_path(repeat_index, chunk_id))
        proposal_graphs: list[dict[str, Any]] = []
        evidence_ranges: list[tuple[int, int]] = []
        for record in records:
            artifact = record.get("artifact")
            start, end = record.get("start"), record.get("end")
            if not isinstance(artifact, str) or not isinstance(start, int) or not isinstance(end, int):
                raise ValueError("relationship_adjudication_route_record_invalid")
            proposal_graphs.append(load_json_object(PROPOSAL_ROOT / artifact))
            evidence_ranges.append((start, end))

        judge_graph, preserved_graph = build_relationship_candidate_union(
            baseline=baseline,
            proposal_graphs=proposal_graphs,
            evidence_ranges=evidence_ranges,
        )
        chunk_output = output_root / f"repeat-{repeat_index:02d}" / "chunks" / _chunk_slug(chunk_id)
        graph_path = chunk_output / "judge-input-graph.json"
        preserved_path = chunk_output / "preserved-graph.json"
        atomic_write_json(graph_path, judge_graph)
        atomic_write_json(preserved_path, preserved_graph)
        candidate_count = len(judge_graph["relationships"])
        batch_count = (candidate_count + JUDGE_BATCH_SIZE - 1) // JUDGE_BATCH_SIZE
        total_candidates += candidate_count
        total_batches += batch_count
        prepared[chunk_id] = {
            "chunk": _masked_chunk(chunks_by_id[chunk_id], evidence_ranges),
            "graph_path": graph_path,
            "preserved_path": preserved_path,
            "judge_output_path": chunk_output / "judge-result.json",
            "adjudicated_graph_path": chunk_output / "graph.json",
            "evidence_ranges": evidence_ranges,
            "candidate_count": candidate_count,
            "batch_count": batch_count,
        }

    manifest = {
        "schema_version": "relationship-adjudication-preparation/v0.1",
        "repeat_index": repeat_index,
        "status": "prepared-no-model-calls",
        "gold_exposed_to_model": False,
        "source_view": "equal-length mask outside selected evidence ranges",
        "candidate_count": total_candidates,
        "judge_batch_size": JUDGE_BATCH_SIZE,
        "expected_model_calls": total_batches,
        "chunks": {
            chunk_id: {
                key: str(value) if isinstance(value, Path) else value
                for key, value in record.items()
                if key != "chunk"
            }
            for chunk_id, record in prepared.items()
        },
    }
    atomic_write_json(output_root / f"repeat-{repeat_index:02d}" / "prepared-manifest.json", manifest)
    return prepared, manifest


async def run_adjudication_pilot(
    *, repeat_index: int = 2, output_root: Path = OUTPUT_ROOT
) -> dict[str, Any]:
    """调用 Judge 过滤候选并集，再对同一组 8 个开发案例复评。"""
    prepared, preparation = prepare_adjudication_inputs(
        repeat_index=repeat_index,
        output_root=output_root,
    )
    schema = load_candidate_graph_schema(DEFAULT_SCHEMA_PATH)
    client = create_deepseek_graph_builder()
    adjudicated_graphs: dict[str, dict[str, Any]] = {}
    adjudication_counts: dict[str, Any] = {}
    try:
        for chunk_id, record in prepared.items():
            judge = await judge_candidate_graph(
                client,
                graph_path=record["graph_path"],
                chunks=[record["chunk"]],
                schema=schema,
                output_path=record["judge_output_path"],
                case_id=f"RELATION-ADJUDICATION:{chunk_id}:r{repeat_index:02d}",
                item_kinds={"relationship"},
            )
            graph, counts = apply_relationship_judgments(
                judge_graph=load_json_object(record["graph_path"]),
                preserved_graph=load_json_object(record["preserved_path"]),
                judge_document=judge,
            )
            atomic_write_json(record["adjudicated_graph_path"], graph)
            adjudicated_graphs[chunk_id] = graph
            adjudication_counts[chunk_id] = counts
    finally:
        await client.aclose()

    gold = load_json_object(GOLD_PATH)
    cases = [item for item in gold.get("cases", []) if isinstance(item, Mapping)]
    _manifest, chunks = load_chunk_manifest(DEFAULT_CHUNK_MANIFEST)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    all_chunk_ids = list(dict.fromkeys(
        chunk_id for case in cases for chunk_id in _string_list(case.get("chunk_ids"))
    ))
    baseline_graphs = {
        chunk_id: load_json_object(_baseline_graph_path(repeat_index, chunk_id))
        for chunk_id in all_chunk_ids
    }
    final_graphs = {**baseline_graphs, **adjudicated_graphs}
    case_results: list[dict[str, Any]] = []
    for case in cases:
        chunk_ids = _string_list(case.get("chunk_ids"))
        source_text = "\n\n".join(chunks_by_id[chunk_id].text for chunk_id in chunk_ids)
        score = score_candidate_graph(
            merge_candidate_graphs(final_graphs[chunk_id] for chunk_id in chunk_ids),
            case,
            source_text=source_text,
        )
        case_results.append({"case_id": case["case_id"], "score": score})
    prf1 = aggregate_supervised_prf1(case_results, "score")
    result = {
        "schema_version": "relationship-adjudication-evaluation/v0.1",
        "status": "development-evaluation-only",
        "publication_status": "HOLD",
        "repeat_index": repeat_index,
        "gold_exposed_to_model": False,
        "preparation": preparation,
        "adjudication_counts": adjudication_counts,
        "prf1": prf1,
        "cases": case_results,
        "boundary": (
            "Candidate union and source-grounded Judge on development scopes; "
            "not an independent test or publication approval."
        ),
    }
    atomic_write_json(output_root / f"repeat-{repeat_index:02d}" / "evaluation-result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行通用关系候选 Judge 裁决实验")
    parser.add_argument("--repeat-index", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.prepare_only:
        _prepared, summary = prepare_adjudication_inputs(
            repeat_index=arguments.repeat_index,
            output_root=OUTPUT_ROOT,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        report = asyncio.run(run_adjudication_pilot(repeat_index=arguments.repeat_index))
        print(json.dumps({
            "relationships": report["prf1"]["categories"]["relationships"],
            "graph": report["prf1"]["graph"],
            "adjudication_counts": report["adjudication_counts"],
        }, ensure_ascii=False, indent=2, sort_keys=True))
