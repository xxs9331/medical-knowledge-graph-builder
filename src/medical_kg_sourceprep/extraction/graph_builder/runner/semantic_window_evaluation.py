"""在语义 section 内构建功能窗口，并运行候选图监督评测。"""

from __future__ import annotations

if __package__ in {None, ""}:
    # 允许直接执行本文件；正常包导入不会进入该分支。
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[4]))
    __package__ = "medical_kg_sourceprep.extraction.graph_builder.runner"

import argparse
import asyncio
import hashlib
import json
import re
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ...artifacts import sha256_path
from ...llm_extraction import EvidenceChunk, atomic_write_json
from ..candidate_graph import run_candidate_graph
from ..client import DeepSeekGraphBuilderClient, create_deepseek_graph_builder
from ..contract import (
    DEFAULT_CHUNK_MANIFEST,
    DEFAULT_SCHEMA_PATH,
    PROJECT_ROOT,
    GraphBuilderConfigurationError,
)
from ..evaluation.aggregation import aggregate_case_scores, aggregate_supervised_prf1
from ..evaluation.artifacts import first_extraction_is_usable, load_json_object
from ..evaluation.scoring import merge_candidate_graphs, score_candidate_graph
from ..schema import load_candidate_graph_schema
from ..trace import JsonlTrace
from .semantic_section_evaluation import (
    DEFAULT_GOLD_PATH,
    DEFAULT_SEMANTIC_ROOT,
    JsonObject,
    SemanticSection,
    map_cases_to_semantic_sections,
)


DEFAULT_WINDOW_OUTPUT = (
    PROJECT_ROOT
    / "runtime/evaluations/semantic-window-s2-pilot/20260817-semantic-window-s2-r01"
)
MAX_FUNCTION_BLOCK_CHARS = 520
_ABNORMAL_HEADER = re.compile(r"^【异常结果解读】\s*$", re.MULTILINE)
_TOP_LEVEL_ITEM = re.compile(r"^[ \t]*[（(][0-9]+[)）]", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class SemanticWindow:
    """可精确回放到上游 semantic section 的一个连续抽取窗口。"""

    window_id: str
    source_section_id: str
    start: int
    end: int
    text: str
    sha256: str
    route: str


def _split_function_block(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """只在超长功能块的一级编号处切分，禁止按固定字符截断结构。"""
    if end - start <= MAX_FUNCTION_BLOCK_CHARS:
        return [(start, end)]
    headings = [start + match.start() for match in _TOP_LEVEL_ITEM.finditer(text[start:end])]
    # 功能标题后的首个编号属于该窗口开头；只从第二个一级编号开始建立新边界。
    split_points = [position for index, position in enumerate(headings) if index > 0]
    if not split_points:
        return [(start, end)]
    boundaries = [start, *split_points, end]
    return [
        (left, right)
        for left, right in zip(boundaries, boundaries[1:])
        if left < right
    ]


def build_semantic_windows(section: SemanticSection) -> list[SemanticWindow]:
    """按冻结功能边界把一个 section 切成连续、无缺口的抽取窗口。"""
    abnormal = _ABNORMAL_HEADER.search(section.text)
    primary_boundaries = [0]
    if abnormal is not None and abnormal.start() > 0:
        primary_boundaries.append(abnormal.start())
    primary_boundaries.append(len(section.text))

    spans: list[tuple[int, int]] = []
    for start, end in zip(primary_boundaries, primary_boundaries[1:]):
        spans.extend(_split_function_block(section.text, start, end))
    if not spans or spans[0][0] != 0 or spans[-1][1] != len(section.text):
        raise GraphBuilderConfigurationError(
            f"semantic_window_coverage_invalid:{section.section_id}"
        )
    if any(left[1] != right[0] for left, right in zip(spans, spans[1:])):
        raise GraphBuilderConfigurationError(
            f"semantic_window_gap_or_overlap:{section.section_id}"
        )

    windows: list[SemanticWindow] = []
    for index, (start, end) in enumerate(spans, start=1):
        window_text = section.text[start:end]
        route = "definition-reference" if start == 0 else "abnormal-interpretation"
        windows.append(SemanticWindow(
            window_id=f"{section.section_id}:window:{index:04d}",
            source_section_id=section.section_id,
            start=start,
            end=end,
            text=window_text,
            sha256=hashlib.sha256(window_text.encode("utf-8")).hexdigest(),
            route=route,
        ))
    return windows


def map_cases_to_semantic_windows(
    *,
    gold_path: Path,
    canonical_manifest_path: Path,
    semantic_root: Path,
    case_ids: set[str] | None = None,
) -> tuple[JsonObject, list[JsonObject], dict[str, SemanticWindow], list[JsonObject]]:
    """依据案例原文范围选择相交窗口，不读取任何抽取目标字段。"""
    dataset, section_cases, sections, section_mappings = map_cases_to_semantic_sections(
        gold_path=gold_path,
        canonical_manifest_path=canonical_manifest_path,
        semantic_root=semantic_root,
        case_ids=case_ids,
    )
    windows_by_section = {
        section_id: build_semantic_windows(section)
        for section_id, section in sections.items()
        if any(str(item["semantic_section_id"]) == section_id for item in section_mappings)
    }

    mapped_cases: list[JsonObject] = []
    mapping_records: list[JsonObject] = []
    selected_windows: dict[str, SemanticWindow] = {}
    for case, section_mapping in zip(section_cases, section_mappings, strict=True):
        section_id = str(section_mapping["semantic_section_id"])
        semantic_scope = cast(Mapping[str, object], section_mapping["semantic_scope"])
        scope_start = semantic_scope.get("start")
        scope_end = semantic_scope.get("end")
        if not isinstance(scope_start, int) or not isinstance(scope_end, int):
            raise GraphBuilderConfigurationError(
                f"semantic_window_scope_invalid:{case.get('case_id')}"
            )

        local_scopes: list[JsonObject] = []
        case_window_ids: list[str] = []
        for window in windows_by_section[section_id]:
            overlap_start = max(scope_start, window.start)
            overlap_end = min(scope_end, window.end)
            if overlap_start >= overlap_end:
                continue
            selected_windows[window.window_id] = window
            case_window_ids.append(window.window_id)
            local_scopes.append({
                "chunk_id": window.window_id,
                "start": overlap_start - window.start,
                "end": overlap_end - window.start,
            })
        if not case_window_ids:
            raise GraphBuilderConfigurationError(
                f"semantic_window_case_unmapped:{case.get('case_id')}"
            )

        mapped_case = deepcopy(case)
        mapped_case["chunk_ids"] = case_window_ids
        mapped_case["evaluation_scopes"] = local_scopes
        mapped_cases.append(mapped_case)
        mapping_records.append({
            "case_id": case.get("case_id"),
            "semantic_section_id": section_id,
            "semantic_scope": dict(semantic_scope),
            "window_ids": case_window_ids,
            "window_scopes": local_scopes,
        })
    return dataset, mapped_cases, selected_windows, mapping_records


def _window_manifest(windows: Mapping[str, SemanticWindow]) -> JsonObject:
    """生成候选图绑定的窗口来源清单。"""
    return {
        "schema_version": "semantic-extraction-window-package/v0.1",
        "status": "derived-input",
        "window_count": len(windows),
        "windowing": {
            "abnormal_header_boundary": True,
            "max_function_block_chars": MAX_FUNCTION_BLOCK_CHARS,
            "oversized_split_boundary": "top_level_parenthesized_item",
            "hard_character_split": False,
        },
        "windows": [{
            "window_id": window.window_id,
            "source_section_id": window.source_section_id,
            "section_char_start": window.start,
            "section_char_end": window.end,
            "char_count": len(window.text),
            "sha256": window.sha256,
            "route": window.route,
        } for window in windows.values()],
    }


async def run_semantic_window_evaluation(
    client: DeepSeekGraphBuilderClient,
    *,
    gold_path: Path = DEFAULT_GOLD_PATH,
    canonical_manifest_path: Path = DEFAULT_CHUNK_MANIFEST,
    semantic_root: Path = DEFAULT_SEMANTIC_ROOT,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    output_root: Path = DEFAULT_WINDOW_OUTPUT,
    case_ids: set[str] | None = None,
    relation_extraction_mode: str = "generative",
    progress: Callable[[str], None] | None = None,
) -> JsonObject:
    """抽取目标盲选择的语义窗口，并在全部模型调用后执行监督评分。"""
    dataset, mapped_cases, windows, mappings = map_cases_to_semantic_windows(
        gold_path=gold_path,
        canonical_manifest_path=canonical_manifest_path,
        semantic_root=semantic_root,
        case_ids=case_ids,
    )
    schema = load_candidate_graph_schema(schema_path)
    window_manifest_path = output_root / "window-manifest.json"
    atomic_write_json(window_manifest_path, _window_manifest(windows))
    window_manifest_sha256 = sha256_path(window_manifest_path)

    trace_run_id = str(uuid.uuid4())
    trace = JsonlTrace(
        output_root / "trace" / f"{trace_run_id}.jsonl",
        run_id=trace_run_id,
    )
    trace.record(
        "run/start",
        workflow="semantic_window_evaluation",
        gold_exposed_to_models=False,
        window_count=len(windows),
        relation_extraction_mode=relation_extraction_mode,
    )

    graphs: dict[str, JsonObject] = {}
    graph_paths: dict[str, str] = {}
    for index, window in enumerate(windows.values(), start=1):
        if progress is not None:
            progress(f"[{index}/{len(windows)}] {window.window_id} section_span={window.start}:{window.end}")
        section_slug = window.source_section_id.rsplit(":", 1)[-1]
        window_slug = window.window_id.rsplit(":", 1)[-1]
        graph_dir = output_root / "windows" / f"{section_slug}-{window_slug}" / "candidate-graph"
        chunk = EvidenceChunk(window.window_id, window.text, window.sha256)
        if not first_extraction_is_usable(graph_dir):
            _ = await run_candidate_graph(
                client,
                chunk=chunk,
                schema=schema,
                schema_path=schema_path,
                output_dir=graph_dir,
                source_manifest_sha256=window_manifest_sha256,
                run_id=f"semantic-s2-{section_slug}-{window_slug}",
                relation_extraction_mode=relation_extraction_mode,
                trace=trace,
            )
        if not first_extraction_is_usable(graph_dir):
            raise GraphBuilderConfigurationError(
                f"semantic_window_extraction_unusable:{window.window_id}"
            )
        graph_path = graph_dir / "graph.json"
        graphs[window.window_id] = cast(JsonObject, load_json_object(graph_path))
        graph_paths[window.window_id] = str(graph_path)

    case_results: list[JsonObject] = []
    for case, mapping in zip(mapped_cases, mappings, strict=True):
        window_ids = cast(list[object], mapping["window_ids"])
        selected_ids = [str(item) for item in window_ids]
        merged = merge_candidate_graphs(graphs[window_id] for window_id in selected_ids)
        source_text = "\n\n".join(windows[window_id].text for window_id in selected_ids)
        score = cast(JsonObject, score_candidate_graph(merged, case, source_text=source_text))
        entities = cast(Mapping[str, object], score["entities"])
        relationships = cast(Mapping[str, object], score["relationships"])
        rules = cast(Mapping[str, object], score["rules"])
        case_results.append({
            "case_id": case["case_id"],
            "window_ids": selected_ids,
            "score": score,
        })
        trace.record(
            "scoring/case",
            case_id=case["case_id"],
            window_ids=selected_ids,
            entity_f1=entities["f1"],
            relationship_f1=relationships["f1"],
            rule_f1=rules["f1"],
        )

    supervised_gold = cast(JsonObject, aggregate_case_scores(case_results, "score"))
    prf1 = cast(JsonObject, aggregate_supervised_prf1(case_results, "score"))
    report: JsonObject = {
        "schema_version": "semantic-window-evaluation/v0.1",
        "status": "evaluation-only",
        "publication_status": "HOLD",
        "treatment_id": "semantic-window-staged-v0.1",
        "gold_status": dataset.get("status"),
        "gold_annotation_method": dataset.get("annotation_method"),
        "gold_scope_contract": dataset.get("scope_contract"),
        "gold_exposed_to_models": False,
        "case_count": len(case_results),
        "unique_window_count": len(windows),
        "configuration": {
            "input_unit": "semantic_function_window",
            "mapping_strategy": "target_blind_scope_intersection/v0.1",
            "relation_extraction_mode": relation_extraction_mode,
            "cross_window_relation_enabled": False,
            "judge_enabled": False,
        },
        "input_hashes": {
            "semantic_manifest": sha256_path(semantic_root / "manifest.json"),
            "canonical_manifest": sha256_path(canonical_manifest_path),
            "window_manifest": window_manifest_sha256,
            "schema": sha256_path(schema_path),
            "gold": sha256_path(gold_path),
        },
        "mapping": mappings,
        "supervised_gold": supervised_gold,
        "prf1": prf1,
        "cases": case_results,
        "window_graphs": graph_paths,
        "trace": {"run_id": trace.run_id, "events": str(trace.path)},
    }
    atomic_write_json(output_root / "evaluation-result.json", report)
    trace.record(
        "run/end",
        workflow="semantic_window_evaluation",
        status="success",
        case_count=len(case_results),
        window_count=len(windows),
        report_path=output_root / "evaluation-result.json",
        graph_f1=prf1["f1"],
    )
    return report


def semantic_window_summary(report: Mapping[str, object]) -> JsonObject:
    """生成命令行需要的四类标准监督 P/R/F1。"""
    prf1 = cast(Mapping[str, object], report["prf1"])
    categories = cast(Mapping[str, object], prf1["categories"])
    return {
        "case_count": report["case_count"],
        "unique_window_count": report["unique_window_count"],
        "entity": categories["entities"],
        "relationship": categories["relationships"],
        "rule": categories["rules"],
        "graph": prf1["graph"],
        "publication_status": report["publication_status"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 semantic section 功能切窗评测")
    _ = parser.add_argument("--semantic-root", type=Path, default=DEFAULT_SEMANTIC_ROOT)
    _ = parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CHUNK_MANIFEST)
    _ = parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    _ = parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    _ = parser.add_argument("--output-root", type=Path, default=DEFAULT_WINDOW_OUTPUT)
    _ = parser.add_argument("--case-id", action="append", dest="case_ids")
    args = parser.parse_args()

    semantic_root = cast(Path, getattr(args, "semantic_root"))
    canonical_manifest = cast(Path, getattr(args, "canonical_manifest"))
    gold = cast(Path, getattr(args, "gold"))
    schema_path = cast(Path, getattr(args, "schema"))
    output_root = cast(Path, getattr(args, "output_root"))
    selected_case_ids = cast(list[str] | None, getattr(args, "case_ids"))

    async def _run() -> JsonObject:
        llm_client = create_deepseek_graph_builder()
        try:
            return await run_semantic_window_evaluation(
                llm_client,
                gold_path=gold,
                canonical_manifest_path=canonical_manifest,
                semantic_root=semantic_root,
                schema_path=schema_path,
                output_root=output_root,
                case_ids=set(selected_case_ids) if selected_case_ids else None,
                progress=print,
            )
        finally:
            await llm_client.aclose()

    print(json.dumps(
        semantic_window_summary(asyncio.run(_run())),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))
