"""评估结构局部候选与单阶段闭集关系分类，不调用 Judge。"""

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
import math
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from ...llm_extraction import EvidenceChunk, atomic_write_json, load_chunk_manifest
from ..client import create_deepseek_graph_builder
from ..contract import (
    DEFAULT_CHUNK_MANIFEST,
    DEFAULT_SCHEMA_PATH,
    DEEPSEEK_MODEL,
    ORDINARY_RELATION_TYPES,
    PROJECT_ROOT,
    STATE_RELATION_TYPES,
    GraphBuilderConfigurationError,
)
from ..evaluation.aggregation import aggregate_supervised_prf1
from ..evaluation.artifacts import load_json_object
from ..evaluation.scoring import merge_candidate_graphs, score_candidate_graph
from ..joint_extraction import (
    TABLE_PROMPT_VERSION_REFINED,
    RoutedEvidenceGroup,
    build_routed_evidence_groups,
)
from ..relation_classifier import RelationPair, build_relation_pairs, classify_relationships_one_stage
from ..schema import load_candidate_graph_schema
from ..validation import normalize_candidate_relationships


GOLD_PATH = PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json"
BEST_ROOT = PROJECT_ROOT / "runtime/evaluations/best-pipeline-v0.1"
REFINED_ROOT = (
    PROJECT_ROOT / "runtime/evaluations/relationship-structure-refinement-v0.3-authorized"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "runtime/evaluations/structure-closed-relation-c1/20260817-144253-closed-r01"
)
RULE_RELATION_TYPES = frozenset({"RULE_INPUT", "RULE_OUTPUT"})
CLASSIFIED_RELATION_TYPES = frozenset(ORDINARY_RELATION_TYPES) | frozenset(STATE_RELATION_TYPES)


def _chunk_slug(chunk_id: str) -> str:
    return "-".join(chunk_id.rsplit(":", 2)[-2:])


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
        raise GraphBuilderConfigurationError("structure_closed_string_list_invalid")
    return list(value)


def _input_graph_path(repeat_index: int, chunk_id: str) -> Path:
    """优先读取结构精修图；未精修的 chunk 回退到同轮最佳流水线图。"""
    slug = _chunk_slug(chunk_id)
    refined = REFINED_ROOT / f"repeat-{repeat_index:02d}" / "chunks" / slug / "graph.json"
    if refined.is_file():
        return refined
    return BEST_ROOT / f"repeat-{repeat_index:02d}" / "chunks" / slug / "graph.json"


def _relation_spans(relationship: Mapping[str, Any]) -> list[tuple[int, int]]:
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


def _overlaps_groups(
    relationship: Mapping[str, Any], groups: Iterable[RoutedEvidenceGroup]
) -> bool:
    return any(
        max(start, group.start) < min(end, group.end)
        for start, end in _relation_spans(relationship)
        for group in groups
    )


def _compose_groups(
    *,
    baseline: Mapping[str, Any],
    classified_relationships: Sequence[Mapping[str, Any]],
    structured_groups: Sequence[RoutedEvidenceGroup],
) -> dict[str, dict[str, Any]]:
    """构造全替换、并集和保留结构范围三种无 Judge 对照图。"""
    nodes = [dict(item) for item in baseline.get("nodes", []) if isinstance(item, Mapping)]
    baseline_relationships = [
        dict(item) for item in baseline.get("relationships", []) if isinstance(item, Mapping)
    ]
    rules = [
        item for item in baseline_relationships
        if item.get("relation_type") in RULE_RELATION_TYPES
    ]
    classified = [dict(item) for item in classified_relationships]
    structured_baseline = [
        item for item in baseline_relationships
        if item.get("relation_type") not in RULE_RELATION_TYPES
        and _overlaps_groups(item, structured_groups)
    ]
    residual_classified = [
        item for item in classified if not _overlaps_groups(item, structured_groups)
    ]

    def graph(relationships: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        merged = merge_candidate_graphs((
            {"nodes": nodes, "relationships": [dict(item) for item in relationships]},
        ))
        return {
            "schema_version": "structure-closed-relation-candidate-graph/v0.1",
            "status": "candidate-only",
            "publication_status": "HOLD",
            **merged,
        }

    return {
        "A": graph(baseline_relationships),
        "B": graph([*rules, *classified]),
        "C": graph([*baseline_relationships, *classified]),
        "D": graph([*rules, *structured_baseline, *residual_classified]),
    }


def _gold_candidate_coverage(
    *,
    cases: Sequence[Mapping[str, Any]],
    nodes_by_chunk: Mapping[str, Sequence[Mapping[str, Any]]],
    pairs_by_chunk: Mapping[str, Sequence[RelationPair]],
) -> dict[str, Any]:
    """仅做开发集候选门控；统计端点同 chunk 可达的金标关系是否进入候选。"""
    accessible = covered = total = 0
    case_records: list[dict[str, Any]] = []
    for case in cases:
        case_accessible = case_covered = 0
        chunk_ids = _string_list(case.get("chunk_ids"))
        relationships = case.get("relationships")
        if not isinstance(relationships, list):
            raise GraphBuilderConfigurationError("structure_closed_gold_relationships_invalid")
        for relationship in relationships:
            if not (
                isinstance(relationship, list)
                and len(relationship) == 3
                and all(isinstance(item, str) for item in relationship)
            ):
                raise GraphBuilderConfigurationError("structure_closed_gold_relationship_invalid")
            source, relation_type, target = relationship
            total += 1
            shared_chunks = [
                chunk_id
                for chunk_id in chunk_ids
                if any(node.get("mention") == source for node in nodes_by_chunk[chunk_id])
                and any(node.get("mention") == target for node in nodes_by_chunk[chunk_id])
            ]
            if not shared_chunks:
                continue
            accessible += 1
            case_accessible += 1
            present = any(
                (
                    pair.left_mention == source
                    and pair.right_mention == target
                    and (relation_type, "LEFT_TO_RIGHT") in pair.options
                )
                or (
                    pair.left_mention == target
                    and pair.right_mention == source
                    and (relation_type, "RIGHT_TO_LEFT") in pair.options
                )
                for chunk_id in shared_chunks
                for pair in pairs_by_chunk[chunk_id]
            )
            if present:
                covered += 1
                case_covered += 1
        case_records.append({
            "case_id": case.get("case_id"),
            "accessible": case_accessible,
            "covered": case_covered,
        })
    coverage = covered / accessible if accessible else 1.0
    return {
        "gold_total": total,
        "endpoint_accessible": accessible,
        "candidate_covered": covered,
        "coverage": coverage,
        "coverage_percent": round(coverage * 100, 2),
        "cases": case_records,
    }


def _score_groups(
    *,
    cases: Sequence[Mapping[str, Any]],
    graphs_by_group: Mapping[str, Mapping[str, Mapping[str, Any]]],
    chunks_by_id: Mapping[str, EvidenceChunk],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_name, graphs in graphs_by_group.items():
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
                "case_id": case.get("case_id"),
                "chunk_ids": chunk_ids,
                "score": score,
            })
        result[group_name] = {
            "prf1": aggregate_supervised_prf1(case_results, "score"),
            "cases": case_results,
        }
    return result


async def run_structure_closed_relation_evaluation(
    *,
    repeat_index: int,
    output_root: Path,
    batch_size: int = 16,
) -> dict[str, Any]:
    """运行一轮冻结实体闭集分类，并生成 A/B/C/D 关系评分。"""
    if repeat_index not in {1, 2, 3}:
        raise GraphBuilderConfigurationError("structure_closed_repeat_invalid")
    if batch_size < 1:
        raise GraphBuilderConfigurationError("structure_closed_batch_size_invalid")
    if output_root.exists() and any(output_root.iterdir()):
        raise GraphBuilderConfigurationError(f"structure_closed_output_not_empty:{output_root}")

    started_at = datetime.now().astimezone()
    started_clock = perf_counter()
    gold = load_json_object(GOLD_PATH)
    cases = [item for item in gold.get("cases", []) if isinstance(item, Mapping)]
    chunk_ids = list(dict.fromkeys(
        chunk_id for case in cases for chunk_id in _string_list(case.get("chunk_ids"))
    ))
    _manifest, chunks = load_chunk_manifest(DEFAULT_CHUNK_MANIFEST)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    schema = load_candidate_graph_schema(DEFAULT_SCHEMA_PATH)
    baseline_by_chunk: dict[str, dict[str, Any]] = {}
    nodes_by_chunk: dict[str, list[Mapping[str, Any]]] = {}
    pairs_by_chunk: dict[str, list[RelationPair]] = {}
    groups_by_chunk: dict[str, list[RoutedEvidenceGroup]] = {}

    for chunk_id in chunk_ids:
        baseline = load_json_object(_input_graph_path(repeat_index, chunk_id))
        nodes = [
            item for item in baseline.get("nodes", [])
            if isinstance(item, Mapping)
            and item.get("entity_type") != "RuleDefinition"
            and item.get("extraction_status") == "VALID"
        ]
        baseline_by_chunk[chunk_id] = baseline
        nodes_by_chunk[chunk_id] = nodes
        pairs_by_chunk[chunk_id] = build_relation_pairs(
            chunk=chunks_by_id[chunk_id],
            schema=schema,
            nodes=nodes,
            allowed_relation_types=CLASSIFIED_RELATION_TYPES,
        )
        groups_by_chunk[chunk_id] = build_routed_evidence_groups(
            chunks_by_id[chunk_id].text,
            table_prompt_version=TABLE_PROMPT_VERSION_REFINED,
        )

    coverage = _gold_candidate_coverage(
        cases=cases,
        nodes_by_chunk=nodes_by_chunk,
        pairs_by_chunk=pairs_by_chunk,
    )
    planned_batches = sum(
        math.ceil(len(pairs) / batch_size) for pairs in pairs_by_chunk.values()
    )
    gate = {
        "minimum_coverage_percent": 95.0,
        "maximum_batches": 80,
        "coverage_passed": coverage["coverage_percent"] >= 95.0,
        "budget_passed": planned_batches <= 80,
        "planned_batches": planned_batches,
    }
    atomic_write_json(output_root / "candidate-gate.json", {"coverage": coverage, "gate": gate})
    if not gate["coverage_passed"] or not gate["budget_passed"]:
        raise GraphBuilderConfigurationError("structure_closed_candidate_gate_failed")

    client = create_deepseek_graph_builder()
    graphs_by_group: dict[str, dict[str, Mapping[str, Any]]] = {
        group: {} for group in ("A", "B", "C", "D")
    }
    chunk_records: dict[str, Any] = {}
    try:
        for index, chunk_id in enumerate(chunk_ids, start=1):
            chunk = chunks_by_id[chunk_id]
            pairs = pairs_by_chunk[chunk_id]
            print(
                f"[{index}/{len(chunk_ids)}] {chunk_id}: {len(pairs)} 个候选，"
                f"{math.ceil(len(pairs) / batch_size)} 批",
                flush=True,
            )
            raw_graph, audit = await classify_relationships_one_stage(
                client,
                chunk=chunk,
                schema=schema,
                nodes=nodes_by_chunk[chunk_id],
                allowed_relation_types=CLASSIFIED_RELATION_TYPES,
                batch_size=batch_size,
            )
            normalized = normalize_candidate_relationships(
                raw_graph,
                chunk=chunk,
                schema=schema,
                nodes=nodes_by_chunk[chunk_id],
                allowed_relation_types=CLASSIFIED_RELATION_TYPES,
                validate_rule_structures=False,
            )
            classified_relationships = list(normalized.accepted)
            composed = _compose_groups(
                baseline=baseline_by_chunk[chunk_id],
                classified_relationships=classified_relationships,
                structured_groups=groups_by_chunk[chunk_id],
            )
            slug = _chunk_slug(chunk_id)
            chunk_root = output_root / "chunks" / slug
            atomic_write_json(chunk_root / "classification-audit.json", audit)
            atomic_write_json(chunk_root / "validation.json", {
                "accepted": len(classified_relationships),
                "review_items": normalized.review_items,
                "judge": "not_run",
            })
            for group_name, graph in composed.items():
                graphs_by_group[group_name][chunk_id] = graph
                atomic_write_json(chunk_root / f"group-{group_name.lower()}-graph.json", graph)
            chunk_records[chunk_id] = {
                "input_graph": str(_input_graph_path(repeat_index, chunk_id)),
                "candidate_pair_count": len(pairs),
                "batch_count": math.ceil(len(pairs) / batch_size),
                "classified_count": audit["classified_count"],
                "accepted_count": len(classified_relationships),
                "review_count": len(normalized.review_items),
                "structured_group_count": len(groups_by_chunk[chunk_id]),
            }
    finally:
        await client.aclose()

    groups = _score_groups(
        cases=cases,
        graphs_by_group=graphs_by_group,
        chunks_by_id=chunks_by_id,
    )
    result = {
        "schema_version": "structure-closed-relation-evaluation/v0.1",
        "status": "development-evaluation-only",
        "publication_status": "HOLD",
        "judge": "not_run",
        "gold_exposed_to_model": False,
        "development_gold_used_for_candidate_gate": True,
        "model": DEEPSEEK_MODEL,
        "prompt_version": "one-stage-structure-relation-classification/v0.1",
        "repeat_index": repeat_index,
        "case_count": len(cases),
        "chunk_count": len(chunk_ids),
        "candidate_gate": {"coverage": coverage, "gate": gate},
        "groups": groups,
        "chunks": chunk_records,
        "execution": {
            "model_calls": planned_batches,
            "batch_size": batch_size,
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now().astimezone().isoformat(),
            "duration_seconds": round(perf_counter() - started_clock, 3),
        },
        "reproducibility": {
            "source_commit": _git_commit(),
            "dirty_tree": True,
            "gold_path": str(GOLD_PATH),
            "gold_sha256": _sha256(GOLD_PATH),
            "chunk_manifest": str(DEFAULT_CHUNK_MANIFEST),
            "chunk_manifest_sha256": _sha256(DEFAULT_CHUNK_MANIFEST),
            "schema_path": str(DEFAULT_SCHEMA_PATH),
            "schema_sha256": _sha256(DEFAULT_SCHEMA_PATH),
        },
        "boundary": (
            "Eight development cases only. Gold was used by code for candidate coverage and final scoring, "
            "never included in model prompts. All candidate graphs remain HOLD."
        ),
    }
    atomic_write_json(output_root / "evaluation-result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行无 Judge 的结构闭集关系分类实验")
    parser.add_argument("--repeat-index", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    report = asyncio.run(run_structure_closed_relation_evaluation(
        repeat_index=args.repeat_index,
        output_root=args.output_root,
        batch_size=args.batch_size,
    ))
    print(json.dumps({
        "candidate_gate": report["candidate_gate"],
        "relationship_prf1": {
            group: value["prf1"]["categories"]["relationships"]
            for group, value in report["groups"].items()
        },
    }, ensure_ascii=False, indent=2, sort_keys=True))
