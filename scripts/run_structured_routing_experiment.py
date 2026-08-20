#!/usr/bin/env python3
"""运行表格、列表和参考区间的结构路由联合抽取实验。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from medical_kg_sourceprep.extraction.graph_builder.client import create_deepseek_graph_builder
from medical_kg_sourceprep.extraction.graph_builder.contract import (
    DEFAULT_CHUNK_MANIFEST,
    DEFAULT_SCHEMA_PATH,
    DEEPSEEK_MODEL,
    PROJECT_ROOT,
    GraphBuilderConfigurationError,
)
from medical_kg_sourceprep.extraction.graph_builder.evaluation.aggregation import (
    aggregate_supervised_prf1,
)
from medical_kg_sourceprep.extraction.graph_builder.evaluation.scoring import (
    merge_candidate_graphs,
    score_candidate_graph,
)
from medical_kg_sourceprep.extraction.graph_builder.joint_extraction import (
    RoutedEvidenceGroup,
    TABLE_PROMPT_VERSION_CONTEXT,
    TABLE_PROMPT_VERSION_ROWS,
    TABLE_PROMPT_VERSIONS,
    build_routed_evidence_groups,
    extract_joint_candidates,
)
from medical_kg_sourceprep.extraction.graph_builder.schema import load_candidate_graph_schema
from medical_kg_sourceprep.extraction.llm_extraction import atomic_write_json, load_chunk_manifest


PROJECT_ID = "structured-routing-h1"
GOLD_PATH = PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json"
BASELINE_ROOT = PROJECT_ROOT / "runtime/evaluations/typical-cases/structured-rules-v0.10-full"
DEFAULT_CASE_IDS = ("TC-01", "TC-03", "TC-08")
RULE_RELATION_TYPES = {"RULE_INPUT", "RULE_OUTPUT"}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GraphBuilderConfigurationError(f"structured_routing_json_not_object:{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _baseline_graph_path(root: Path, chunk_id: str) -> Path:
    slug = "-".join(chunk_id.rsplit(":", 2)[-2:])
    return root / "chunks" / slug / "candidate-graph" / "graph.json"


def _case_graph_documents(
    *,
    baseline_root: Path,
    graphs: Mapping[str, Mapping[str, Any]],
    case_id: str,
    chunk_ids: Iterable[str],
) -> tuple[list[Mapping[str, Any]], Path | None]:
    """合并 chunk 图，并为所有对照组复用同一份跨 chunk 关系图。"""
    documents = [graphs[chunk_id] for chunk_id in chunk_ids]
    case_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", case_id).strip("-")
    cross_chunk_path = baseline_root / "cases" / case_slug / "cross-chunk-graph.json"
    if cross_chunk_path.is_file():
        documents.append(_load_object(cross_chunk_path))
        return documents, cross_chunk_path
    return documents, None


def _relation_spans(relationship: Mapping[str, Any]) -> list[tuple[int, int]]:
    """读取已经回放的关系证据位置，用于替换相同结构范围内的旧关系。"""
    references = relationship.get("relation_evidence_refs")
    if not isinstance(references, list):
        reference = relationship.get("source_ref")
        references = [reference] if isinstance(reference, Mapping) else []
    spans: list[tuple[int, int]] = []
    for reference in references:
        if not isinstance(reference, Mapping):
            continue
        start, end = reference.get("char_start"), reference.get("char_end")
        if (
            isinstance(start, int) and not isinstance(start, bool)
            and isinstance(end, int) and not isinstance(end, bool)
            and start < end
        ):
            spans.append((start, end))
    return spans


def _overlaps_route(
    relationship: Mapping[str, Any], groups: Iterable[RoutedEvidenceGroup]
) -> bool:
    spans = _relation_spans(relationship)
    return any(
        max(start, group.start) < min(end, group.end)
        for start, end in spans
        for group in groups
    )


def _build_comparison_graphs(
    baseline: Mapping[str, Any],
    route_results: Iterable[Mapping[str, Any]],
    groups: Iterable[RoutedEvidenceGroup],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """构造主实验替换图 H 和次要并集图 U。"""
    group_list = list(groups)
    route_graph = merge_candidate_graphs(
        {
            "nodes": result.get("nodes", []),
            "relationships": result.get("relationships", []),
        }
        for result in route_results
    )
    preserved_relationships = [
        relationship
        for relationship in baseline.get("relationships", [])
        if isinstance(relationship, Mapping)
        and (
            relationship.get("relation_type") in RULE_RELATION_TYPES
            or not _overlaps_route(relationship, group_list)
        )
    ]
    replacement = merge_candidate_graphs((
        {"nodes": baseline.get("nodes", []), "relationships": preserved_relationships},
        route_graph,
    ))
    union = merge_candidate_graphs((baseline, route_graph))
    return replacement, union


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _metric_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        group: {
            "graph": value["prf1"]["graph"],
            **value["prf1"]["categories"],
        }
        for group, value in report["groups"].items()
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now().astimezone()
    started_clock = perf_counter()
    output_root = cast(Path, args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise GraphBuilderConfigurationError(f"structured_routing_output_not_empty:{output_root}")

    gold_path = cast(Path, args.gold)
    manifest_path = cast(Path, args.manifest)
    schema_path = cast(Path, args.schema)
    baseline_root = cast(Path, args.baseline_root)
    project_id = cast(str, args.project_id)
    table_prompt_version = cast(str, args.table_prompt_version)
    gold = _load_object(gold_path)
    requested = set(cast(list[str], args.case_ids))
    cases = [
        case for case in gold.get("cases", [])
        if isinstance(case, dict) and case.get("case_id") in requested
    ]
    if {str(case["case_id"]) for case in cases} != requested:
        raise GraphBuilderConfigurationError("structured_routing_case_missing")

    _manifest, chunks = load_chunk_manifest(manifest_path)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    schema = load_candidate_graph_schema(schema_path)
    cases_by_chunk: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        for chunk_id in case.get("chunk_ids", []):
            cases_by_chunk[str(chunk_id)].append(str(case["case_id"]))

    baseline_graphs: dict[str, dict[str, Any]] = {}
    replacement_graphs: dict[str, dict[str, Any]] = {}
    union_graphs: dict[str, dict[str, Any]] = {}
    route_manifest: list[dict[str, Any]] = []
    model_calls = 0
    input_tokens = output_tokens = total_tokens = 0
    client = create_deepseek_graph_builder()
    try:
        for chunk_index, chunk_id in enumerate(cases_by_chunk, start=1):
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                raise GraphBuilderConfigurationError(f"structured_routing_chunk_missing:{chunk_id}")
            baseline = _load_object(_baseline_graph_path(baseline_root, chunk_id))
            baseline_graphs[chunk_id] = baseline
            groups = build_routed_evidence_groups(
                chunk.text,
                table_prompt_version=table_prompt_version,
            )
            slug = "-".join(chunk_id.rsplit(":", 2)[-2:])
            print(
                f"[{chunk_index}/{len(cases_by_chunk)}] {chunk_id}: {len(groups)} 个结构组",
                flush=True,
            )
            route_results: list[dict[str, Any]] = []
            for group_index, group in enumerate(groups, start=1):
                print(
                    f"  [{group_index}/{len(groups)}] {group.group_id} ({group.route})",
                    flush=True,
                )
                result = await extract_joint_candidates(
                    client,
                    chunk=chunk,
                    schema=schema,
                    frozen_nodes=[
                        node for node in baseline.get("nodes", [])
                        if isinstance(node, Mapping)
                    ],
                    evidence_units=group.units,
                    route_instructions=group.instructions,
                )
                route_results.append(result)
                model_calls += 1
                usage = result.get("response_diagnostics", [{}])[0].get("usage", {})
                input_tokens += int(usage.get("input_tokens") or 0)
                output_tokens += int(usage.get("output_tokens") or 0)
                total_tokens += int(usage.get("total_tokens") or 0)
                route_path = output_root / "chunks" / slug / "routes" / f"{group.group_id}.json"
                atomic_write_json(route_path, result)
                route_manifest.append({
                    "chunk_id": chunk_id,
                    "group_id": group.group_id,
                    "route": group.route,
                    "start": group.start,
                    "end": group.end,
                    "evidence_unit_ids": [unit.unit_id for unit in group.units],
                    "artifact": str(route_path.relative_to(output_root)),
                    "accepted_nodes": len(result.get("nodes", [])),
                    "proposed_nodes": len(result.get("proposed_node_keys", [])),
                    "accepted_relationships": len(result.get("relationships", [])),
                    "review_items": len(result.get("review_items", [])),
                })

            replacement, union = _build_comparison_graphs(baseline, route_results, groups)
            replacement_graphs[chunk_id] = replacement
            union_graphs[chunk_id] = union
            atomic_write_json(output_root / "chunks" / slug / "hybrid-replace-graph.json", replacement)
            atomic_write_json(output_root / "chunks" / slug / "hybrid-union-graph.json", union)
    finally:
        await client.aclose()

    context_prompt = table_prompt_version == TABLE_PROMPT_VERSION_CONTEXT
    comparison_groups = (
        ("A", baseline_graphs, "staged-baseline-v0.10"),
        (
            "H",
            replacement_graphs,
            "table-context-replace-v0.2" if context_prompt else "structured-route-replace-v0.1",
        ),
        (
            "U",
            union_graphs,
            "table-context-union-v0.2" if context_prompt else "structured-route-union-v0.1",
        ),
    )
    group_scores: dict[str, list[dict[str, Any]]] = {group: [] for group, _, _ in comparison_groups}
    for case in cases:
        chunk_ids = [str(value) for value in case["chunk_ids"]]
        source_text = "\n\n".join(chunk_by_id[value].text for value in chunk_ids)
        for group, graphs, _treatment_id in comparison_groups:
            graph_documents, cross_chunk_path = _case_graph_documents(
                baseline_root=baseline_root,
                graphs=graphs,
                case_id=str(case["case_id"]),
                chunk_ids=chunk_ids,
            )
            score = score_candidate_graph(
                merge_candidate_graphs(graph_documents),
                case,
                source_text=source_text,
            )
            group_scores[group].append({
                "case_id": case["case_id"],
                "chunk_ids": chunk_ids,
                "cross_chunk_graph": str(cross_chunk_path) if cross_chunk_path else None,
                "score": score,
            })

    report = {
        "schema_version": "structured-routing-experiment/v0.1",
        "project_id": project_id,
        "run_id": output_root.name,
        "repeat_index": int(args.repeat_index),
        "status": "development-evaluation-only",
        "publication_status": "HOLD",
        "gold_exposed_to_model": False,
        "cases": sorted(requested),
        "groups": {
            group: {
                "treatment_id": treatment_id,
                "prf1": aggregate_supervised_prf1(scores, "score"),
                "cases": scores,
            }
            for group, _graphs, treatment_id in comparison_groups
            for scores in (group_scores[group],)
        },
        "route_manifest": route_manifest,
    }
    atomic_write_json(output_root / "evaluation-result.json", report)

    ended_at = datetime.now().astimezone()
    manifest = {
        "schema_version": "structured-routing-run-manifest/v0.1",
        "project_id": project_id,
        "run_id": output_root.name,
        "run_outcome": "completed",
        "publication_status": "HOLD",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": round(perf_counter() - started_clock, 3),
        "source_commit": _git_commit(),
        "dirty_tree": True,
        "inputs": {
            "chunk_manifest": str(manifest_path),
            "chunk_manifest_sha256": _sha256(manifest_path),
            "schema": str(schema_path),
            "schema_sha256": _sha256(schema_path),
            "gold": str(gold_path),
            "gold_sha256": _sha256(gold_path),
            "gold_exposed_to_model": False,
            "baseline_root": str(baseline_root),
        },
        "configuration": {
            "model": DEEPSEEK_MODEL,
            "provider": "DeepSeek",
            "temperature": 0,
            "seed": "not_available",
            "retry_policy": "joint payload format retry at most once",
            "table_prompt_version": table_prompt_version,
            "joint_extraction_source_sha256": _sha256(
                PROJECT_ROOT / "src/medical_kg_sourceprep/extraction/graph_builder/joint_extraction.py"
            ),
            "scorer_sha256": _sha256(
                PROJECT_ROOT / "src/medical_kg_sourceprep/extraction/graph_builder/evaluation/scoring.py"
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "execution": {
            "processed_chunks": len(cases_by_chunk),
            "failed_chunks": 0,
            "skipped_chunks": 0,
            "model_calls": model_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
        "artifacts": {
            "evaluation_result": "evaluation-result.json",
            "route_artifact_count": len(route_manifest),
        },
    }
    atomic_write_json(output_root / "run-manifest.json", manifest)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行结构路由联合抽取实验")
    _ = parser.add_argument("--project-id", default=PROJECT_ID)
    _ = parser.add_argument(
        "--table-prompt-version",
        choices=sorted(TABLE_PROMPT_VERSIONS),
        default=TABLE_PROMPT_VERSION_ROWS,
    )
    _ = parser.add_argument("--case-id", action="append", dest="case_ids", default=[])
    _ = parser.add_argument("--repeat-index", type=int, required=True)
    _ = parser.add_argument("--output-root", type=Path, required=True)
    _ = parser.add_argument("--gold", type=Path, default=GOLD_PATH)
    _ = parser.add_argument("--manifest", type=Path, default=DEFAULT_CHUNK_MANIFEST)
    _ = parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    _ = parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    arguments = parser.parse_args()
    if not arguments.case_ids:
        arguments.case_ids = list(DEFAULT_CASE_IDS)
    experiment = asyncio.run(_run(arguments))
    print(json.dumps(_metric_summary(experiment), ensure_ascii=False, indent=2, sort_keys=True))
