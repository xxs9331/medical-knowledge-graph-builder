"""评估分类列表、共享谓词和表格父项三类关系结构精修。"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[4]))
    __package__ = "medical_kg_sourceprep.extraction.graph_builder.runner"

import asyncio
import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from ...llm_extraction import atomic_write_json, load_chunk_manifest
from ..client import create_deepseek_graph_builder
from ..contract import (
    DEFAULT_CHUNK_MANIFEST,
    DEFAULT_SCHEMA_PATH,
    DEEPSEEK_MODEL,
    PROJECT_ROOT,
    GraphBuilderConfigurationError,
)
from ..evaluation.aggregation import aggregate_supervised_prf1
from ..evaluation.artifacts import load_json_object
from ..evaluation.scoring import merge_candidate_graphs, score_candidate_graph
from ..joint_extraction import (
    TABLE_PROMPT_VERSION_REFINED,
    RoutedEvidenceGroup,
    build_routed_evidence_groups,
    extract_joint_candidates,
)
from ..schema import load_candidate_graph_schema


GOLD_PATH = PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json"
BASELINE_ROOT = PROJECT_ROOT / "runtime/evaluations/best-pipeline-v0.1"
OUTPUT_ROOT = PROJECT_ROOT / "runtime/evaluations/relationship-structure-refinement-v0.3-authorized"
TARGET_ROUTES = frozenset({"classification_list", "shared_predicate", "table"})
TARGET_CASE_IDS = frozenset({"TC-02", "TC-05", "TC-06"})
TARGET_CHUNK_IDS = (
    "clinical-hematology:chapter-01:0004:0000",
    "clinical-hematology:chapter-01:0001:0000",
)
RULE_RELATION_TYPES = frozenset({"RULE_INPUT", "RULE_OUTPUT"})


def _chunk_slug(chunk_id: str) -> str:
    """把规范 chunk_id 转成现有实验目录使用的短名称。"""
    return "-".join(chunk_id.rsplit(":", 2)[-2:])


def _baseline_graph_path(repeat_index: int, chunk_id: str) -> Path:
    return (
        BASELINE_ROOT
        / f"repeat-{repeat_index:02d}"
        / "chunks"
        / _chunk_slug(chunk_id)
        / "graph.json"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise GraphBuilderConfigurationError("relationship_refinement_string_list_invalid")
    return list(value)


def _relation_spans(relationship: Mapping[str, Any]) -> list[tuple[int, int]]:
    """读取已校验关系的证据范围，用于只替换命中的结构区域。"""
    references = relationship.get("relation_evidence_refs")
    if not isinstance(references, list):
        source_ref = relationship.get("source_ref")
        references = [source_ref] if isinstance(source_ref, Mapping) else []
    spans: list[tuple[int, int]] = []
    for reference in references:
        if not isinstance(reference, Mapping):
            continue
        start, end = reference.get("char_start"), reference.get("char_end")
        if (
            isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and start < end
        ):
            spans.append((start, end))
    return spans


def _overlaps_group(
    relationship: Mapping[str, Any], groups: Iterable[RoutedEvidenceGroup]
) -> bool:
    return any(
        max(start, group.start) < min(end, group.end)
        for start, end in _relation_spans(relationship)
        for group in groups
    )


def _group_inside_target_scopes(
    group: RoutedEvidenceGroup,
    *,
    chunk_id: str,
    target_scopes: Sequence[Mapping[str, Any]],
) -> bool:
    """只调用与三个预注册开发案例范围相交的结构组。"""
    return any(
        scope.get("chunk_id") == chunk_id
        and isinstance(scope.get("start"), int)
        and not isinstance(scope.get("start"), bool)
        and isinstance(scope.get("end"), int)
        and not isinstance(scope.get("end"), bool)
        and max(group.start, int(scope["start"])) < min(group.end, int(scope["end"]))
        for scope in target_scopes
    )


def _replace_structured_ranges(
    baseline: Mapping[str, Any],
    route_results: Sequence[Mapping[str, Any]],
    groups: Sequence[RoutedEvidenceGroup],
) -> dict[str, list[Any]]:
    """保留范围外候选，用精修结果替换范围内的普通关系。"""
    preserved_relationships = [
        relationship
        for relationship in baseline.get("relationships", [])
        if isinstance(relationship, Mapping)
        and (
            relationship.get("relation_type") in RULE_RELATION_TYPES
            or not _overlaps_group(relationship, groups)
        )
    ]
    route_graph = merge_candidate_graphs(
        {
            "nodes": result.get("nodes", []),
            "relationships": result.get("relationships", []),
        }
        for result in route_results
    )
    return merge_candidate_graphs((
        {
            "nodes": baseline.get("nodes", []),
            "relationships": preserved_relationships,
        },
        route_graph,
    ))


def _score_cases(
    *,
    cases: Sequence[Mapping[str, Any]],
    graphs: Mapping[str, Mapping[str, Any]],
    chunks_by_id: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_results: list[dict[str, Any]] = []
    for case in cases:
        chunk_ids = _string_list(case.get("chunk_ids"))
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
    return case_results, aggregate_supervised_prf1(case_results, "score")


def _three_run_mean(runs: Sequence[Mapping[str, Any]], group: str) -> dict[str, Any]:
    categories = ("entities", "relationships", "rules", "graph")
    return {
        category: {
            metric: round(
                sum(
                    float(run[group]["prf1"]["categories"].get(category, run[group]["prf1"])[metric])
                    for run in runs
                )
                / len(runs),
                2,
            )
            for metric in ("precision_percent", "recall_percent", "f1_percent")
        }
        for category in categories
    }


async def run_relationship_structure_refinement(
    *, output_root: Path = OUTPUT_ROOT
) -> dict[str, Any]:
    """运行三次真实模型精修，并与每次对应的冻结最佳图比较。"""
    if output_root.exists() and any(output_root.iterdir()):
        raise GraphBuilderConfigurationError(
            f"relationship_refinement_output_not_empty:{output_root}"
        )
    started_at = datetime.now().astimezone()
    started_clock = perf_counter()
    gold = load_json_object(GOLD_PATH)
    cases = [item for item in gold.get("cases", []) if isinstance(item, Mapping)]
    target_scopes: list[Mapping[str, Any]] = []
    for case in cases:
        if case.get("case_id") not in TARGET_CASE_IDS:
            continue
        raw_scopes = case.get("evaluation_scopes")
        if isinstance(raw_scopes, list):
            target_scopes.extend(
                scope for scope in raw_scopes if isinstance(scope, Mapping)
            )
    all_chunk_ids = list(dict.fromkeys(
        chunk_id for case in cases for chunk_id in _string_list(case.get("chunk_ids"))
    ))
    _manifest, chunks = load_chunk_manifest(DEFAULT_CHUNK_MANIFEST)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    schema = load_candidate_graph_schema(DEFAULT_SCHEMA_PATH)
    client = create_deepseek_graph_builder()
    runs: list[dict[str, Any]] = []
    total_model_calls = 0
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        for repeat_index in range(1, 4):
            print(f"[repeat {repeat_index}/3] 读取对应冻结最佳图", flush=True)
            baseline_graphs = {
                chunk_id: load_json_object(_baseline_graph_path(repeat_index, chunk_id))
                for chunk_id in all_chunk_ids
            }
            refined_graphs = dict(baseline_graphs)
            route_records: list[dict[str, Any]] = []

            for chunk_id in TARGET_CHUNK_IDS:
                chunk = chunks_by_id[chunk_id]
                groups = [
                    group
                    for group in build_routed_evidence_groups(
                        chunk.text,
                        table_prompt_version=TABLE_PROMPT_VERSION_REFINED,
                    )
                    if group.route in TARGET_ROUTES
                    and _group_inside_target_scopes(
                        group,
                        chunk_id=chunk_id,
                        target_scopes=target_scopes,
                    )
                ]
                if not groups:
                    raise GraphBuilderConfigurationError(
                        f"relationship_refinement_route_missing:{chunk_id}"
                    )
                route_results: list[dict[str, Any]] = []
                for group_index, group in enumerate(groups, start=1):
                    print(
                        "".join((
                            f"  {chunk_id} [{group_index}/{len(groups)}] ",
                            f"{group.group_id} ({group.route})",
                        )),
                        flush=True,
                    )
                    result = await extract_joint_candidates(
                        client,
                        chunk=chunk,
                        schema=schema,
                        frozen_nodes=[
                            item
                            for item in baseline_graphs[chunk_id].get("nodes", [])
                            if isinstance(item, Mapping)
                        ],
                        evidence_units=group.units,
                        route_instructions=group.instructions,
                    )
                    route_results.append(result)
                    total_model_calls += 1
                    diagnostic = result.get("response_diagnostics", [{}])[0]
                    usage_value = diagnostic.get("usage") if isinstance(diagnostic, Mapping) else None
                    usage = usage_value if isinstance(usage_value, Mapping) else {}
                    for key in total_usage:
                        token_count = usage.get(key)
                        if isinstance(token_count, int) and not isinstance(token_count, bool):
                            total_usage[key] += token_count
                    route_path = (
                        output_root
                        / f"repeat-{repeat_index:02d}"
                        / "chunks"
                        / _chunk_slug(chunk_id)
                        / "routes"
                        / f"{group.group_id}.json"
                    )
                    atomic_write_json(route_path, result)
                    route_records.append({
                        "chunk_id": chunk_id,
                        "group_id": group.group_id,
                        "route": group.route,
                        "start": group.start,
                        "end": group.end,
                        "artifact": str(route_path.relative_to(output_root)),
                        "accepted_nodes": len(result.get("nodes", [])),
                        "proposed_nodes": len(result.get("proposed_node_keys", [])),
                        "accepted_relationships": len(result.get("relationships", [])),
                        "review_items": len(result.get("review_items", [])),
                    })

                refined_graphs[chunk_id] = _replace_structured_ranges(
                    baseline_graphs[chunk_id],
                    route_results,
                    groups,
                )
                atomic_write_json(
                    output_root
                    / f"repeat-{repeat_index:02d}"
                    / "chunks"
                    / _chunk_slug(chunk_id)
                    / "graph.json",
                    refined_graphs[chunk_id],
                )

            if len(route_records) != 3:
                raise GraphBuilderConfigurationError(
                    f"relationship_refinement_route_count_invalid:{len(route_records)}"
                )

            baseline_cases, baseline_prf1 = _score_cases(
                cases=cases,
                graphs=baseline_graphs,
                chunks_by_id=chunks_by_id,
            )
            refined_cases, refined_prf1 = _score_cases(
                cases=cases,
                graphs=refined_graphs,
                chunks_by_id=chunks_by_id,
            )
            run = {
                "repeat_index": repeat_index,
                "baseline": {"prf1": baseline_prf1, "cases": baseline_cases},
                "refined": {"prf1": refined_prf1, "cases": refined_cases},
                "routes": route_records,
            }
            atomic_write_json(
                output_root / f"repeat-{repeat_index:02d}" / "evaluation-result.json",
                run,
            )
            runs.append(run)
    finally:
        await client.aclose()

    result = {
        "schema_version": "relationship-structure-refinement/v0.3",
        "status": "development-evaluation-only",
        "publication_status": "HOLD",
        "gold_exposed_to_model": False,
        "model": DEEPSEEK_MODEL,
        "prompt_version": TABLE_PROMPT_VERSION_REFINED,
        "case_count": len(cases),
        "target_routes": sorted(TARGET_ROUTES),
        "target_case_ids": sorted(TARGET_CASE_IDS),
        "runs": runs,
        "three_run_mean_percent": {
            "baseline": _three_run_mean(runs, "baseline"),
            "refined": _three_run_mean(runs, "refined"),
        },
        "execution": {
            "model_calls": total_model_calls,
            **total_usage,
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now().astimezone().isoformat(),
            "duration_seconds": round(perf_counter() - started_clock, 3),
        },
        "reproducibility": {
            "source_commit": _git_commit(),
            "dirty_tree": True,
            "development_scopes_used_for_route_selection": True,
            "gold_path": str(GOLD_PATH),
            "gold_sha256": _sha256(GOLD_PATH),
            "chunk_manifest": str(DEFAULT_CHUNK_MANIFEST),
            "chunk_manifest_sha256": _sha256(DEFAULT_CHUNK_MANIFEST),
            "schema_path": str(DEFAULT_SCHEMA_PATH),
            "schema_sha256": _sha256(DEFAULT_SCHEMA_PATH),
            "joint_extraction_sha256": _sha256(
                PROJECT_ROOT
                / "src/medical_kg_sourceprep/extraction/graph_builder/joint_extraction.py"
            ),
        },
        "boundary": (
            "Eight development cases only; gold was used after extraction for scoring, "
            "not exposed to the model. Candidate graphs remain HOLD."
        ),
    }
    atomic_write_json(output_root / "evaluation-result.json", result)
    return result


if __name__ == "__main__":
    report = asyncio.run(run_relationship_structure_refinement())
    print(json.dumps({
        "runs": [
            {
                "repeat_index": run["repeat_index"],
                "baseline_relationships": run["baseline"]["prf1"]["categories"]["relationships"],
                "refined_relationships": run["refined"]["prf1"]["categories"]["relationships"],
                "refined_graph": run["refined"]["prf1"]["graph"],
            }
            for run in report["runs"]
        ],
        "three_run_mean_percent": report["three_run_mean_percent"],
        "execution": report["execution"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
