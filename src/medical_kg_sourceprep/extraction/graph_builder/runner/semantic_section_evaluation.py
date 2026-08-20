"""使用真实语义 section 运行候选图抽取和监督评分实验。

本模块只改变模型的输入单位：先把典型案例冻结的原文范围映射到语义 section，
再复用生产候选图工作流抽取。案例答案不会进入映射或模型提示词；所有模型调用结束
后，才用映射后的范围执行确定性 P/R/F1 评分。
"""

from __future__ import annotations

if __package__ in {None, ""}:
    # 允许直接执行本文件；正常作为包导入时不会进入该分支。
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
from ...llm_extraction import EvidenceChunk, atomic_write_json, load_chunk_manifest
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
from ..evaluation.scoring import score_candidate_graph
from ..schema import load_candidate_graph_schema
from ..trace import JsonlTrace


JsonObject = dict[str, object]


DEFAULT_GOLD_PATH = PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json"
DEFAULT_SEMANTIC_ROOT = (
    PROJECT_ROOT / "source-packages/derived/semantic-sections/full-book-v0.2"
)
DEFAULT_SEMANTIC_OUTPUT = (
    PROJECT_ROOT
    / "runtime/evaluations/semantic-section-s1-pilot/20260817-semantic-s1-r01"
)

# 这些行只承担版面或层级导航功能，不是案例抽取目标。忽略它们时仍保留正文字符到
# section 原坐标的映射，最终证据范围继续使用未压缩输入视图中的真实位置。
_STRUCTURAL_LINE_PATTERNS = (
    re.compile(r"^【[^】]+】$"),
    re.compile(r"^第[一二三四五六七八九十百零〇两0-9]+章(?:\s|$)"),
    re.compile(r"^[一二三四五六七八九十百零〇两0-9]+、"),
)


@dataclass(frozen=True, slots=True)
class SemanticSection:
    """已经校验输入视图哈希的一个可抽取语义 section。"""

    section_id: str
    text: str
    input_view_sha256: str
    content_role: str
    extraction_route: str
    section_path: tuple[str, ...]


def _compacted_content(text: str) -> tuple[str, tuple[int, ...]]:
    """删除结构行和空白，同时记录每个保留字符在原文中的位置。"""
    compacted: list[str] = []
    positions: list[int] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        stripped = content.strip()
        ignored = any(pattern.search(stripped) for pattern in _STRUCTURAL_LINE_PATTERNS)
        if not ignored:
            for local_index, character in enumerate(content):
                if not character.isspace():
                    compacted.append(character)
                    positions.append(offset + local_index)
        offset += len(line)
    return "".join(compacted), tuple(positions)


def load_semantic_sections(semantic_root: Path) -> tuple[JsonObject, dict[str, SemanticSection]]:
    """读取语义分块 manifest，并逐项验证真实输入视图的长度和哈希。"""
    manifest_path = semantic_root / "manifest.json"
    manifest = cast(JsonObject, load_json_object(manifest_path))
    raw_sections = manifest.get("sections")
    if not isinstance(raw_sections, list):
        raise GraphBuilderConfigurationError("semantic_section_manifest_invalid")

    sections: dict[str, SemanticSection] = {}
    for raw_value in cast(list[object], raw_sections):
        if not isinstance(raw_value, Mapping):
            continue
        raw = cast(Mapping[str, object], raw_value)
        if raw.get("extraction_eligible") is not True:
            continue
        section_id = raw.get("section_id")
        relative_path = raw.get("input_view_file")
        expected_sha256 = raw.get("input_view_sha256")
        expected_char_count = raw.get("input_view_char_count")
        if not all(isinstance(value, str) and value for value in (
            section_id, relative_path, expected_sha256
        )):
            raise GraphBuilderConfigurationError("semantic_section_entry_invalid")
        input_path = semantic_root / str(relative_path)
        try:
            text = input_path.read_text(encoding="utf-8")
        except OSError as error:
            raise GraphBuilderConfigurationError(
                f"semantic_section_input_missing:{section_id}"
            ) from error
        actual_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual_sha256 != expected_sha256 or len(text) != expected_char_count:
            raise GraphBuilderConfigurationError(
                f"semantic_section_input_mismatch:{section_id}"
            )
        if str(section_id) in sections:
            raise GraphBuilderConfigurationError(
                f"semantic_section_duplicate:{section_id}"
            )
        raw_path = raw.get("section_path", [])
        section_path = (
            tuple(str(item) for item in cast(list[object], raw_path))
            if isinstance(raw_path, list) else ()
        )
        sections[str(section_id)] = SemanticSection(
            section_id=str(section_id),
            text=text,
            input_view_sha256=str(expected_sha256),
            content_role=str(raw.get("content_role", "")),
            extraction_route=str(raw.get("extraction_route", "")),
            section_path=section_path,
        )
    if not sections:
        raise GraphBuilderConfigurationError("semantic_sections_missing")
    return manifest, sections


def map_cases_to_semantic_sections(
    *,
    gold_path: Path,
    canonical_manifest_path: Path,
    semantic_root: Path,
    case_ids: set[str] | None = None,
) -> tuple[JsonObject, list[JsonObject], dict[str, SemanticSection], list[JsonObject]]:
    """仅按冻结原文范围把案例映射到唯一语义 section，不读取目标答案。"""
    dataset = cast(JsonObject, load_json_object(gold_path))
    raw_cases = dataset.get("cases")
    if not isinstance(raw_cases, list):
        raise GraphBuilderConfigurationError("evaluation_cases_missing")
    selected_cases: list[Mapping[str, object]] = []
    for case_value in cast(list[object], raw_cases):
        if not isinstance(case_value, Mapping):
            continue
        case = cast(Mapping[str, object], case_value)
        if case_ids is None or case.get("case_id") in case_ids:
            selected_cases.append(case)
    if not selected_cases:
        raise GraphBuilderConfigurationError("evaluation_cases_missing")

    _canonical_manifest, canonical_chunks = load_chunk_manifest(canonical_manifest_path)
    chunk_lookup = {chunk.chunk_id: chunk for chunk in canonical_chunks}
    _semantic_manifest, sections = load_semantic_sections(semantic_root)
    compacted_sections = {
        section_id: _compacted_content(section.text)
        for section_id, section in sections.items()
    }

    mapped_cases: list[JsonObject] = []
    mapping_records: list[JsonObject] = []
    for source_case in selected_cases:
        case_id = str(source_case.get("case_id", ""))
        raw_scopes = source_case.get("evaluation_scopes")
        if not case_id or not isinstance(raw_scopes, list) or not raw_scopes:
            raise GraphBuilderConfigurationError(f"evaluation_scope_missing:{case_id}")

        mapped_fragments: list[tuple[str, int, int]] = []
        original_scopes: list[JsonObject] = []
        for scope_value in cast(list[object], raw_scopes):
            if not isinstance(scope_value, Mapping):
                raise GraphBuilderConfigurationError(f"evaluation_scope_invalid:{case_id}")
            scope = cast(Mapping[str, object], scope_value)
            chunk_id = scope.get("chunk_id")
            start = scope.get("start")
            end = scope.get("end")
            chunk = chunk_lookup.get(str(chunk_id))
            if (
                chunk is None
                or not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or not 0 <= start < end <= len(chunk.text)
            ):
                raise GraphBuilderConfigurationError(f"evaluation_scope_invalid:{case_id}")
            scope_text = chunk.text[start:end]
            compacted_scope, _positions = _compacted_content(scope_text)
            if not compacted_scope:
                raise GraphBuilderConfigurationError(f"evaluation_scope_empty:{case_id}")

            matches: list[tuple[str, int, int]] = []
            for section_id, (compacted_section, positions) in compacted_sections.items():
                match_start = compacted_section.find(compacted_scope)
                if match_start < 0:
                    continue
                if compacted_section.find(compacted_scope, match_start + 1) >= 0:
                    raise GraphBuilderConfigurationError(
                        f"semantic_scope_repeated_in_section:{case_id}:{section_id}"
                    )
                match_end = match_start + len(compacted_scope)
                matches.append((
                    section_id,
                    positions[match_start],
                    positions[match_end - 1] + 1,
                ))
            if len(matches) != 1:
                raise GraphBuilderConfigurationError(
                    f"semantic_scope_match_count:{case_id}:{len(matches)}"
                )
            mapped_fragments.append(matches[0])
            original_scopes.append({"chunk_id": str(chunk_id), "start": start, "end": end})

        section_ids = {item[0] for item in mapped_fragments}
        if len(section_ids) != 1:
            raise GraphBuilderConfigurationError(
                f"semantic_case_crosses_sections:{case_id}:{len(section_ids)}"
            )
        section_id = next(iter(section_ids))
        mapped_start = min(item[1] for item in mapped_fragments)
        mapped_end = max(item[2] for item in mapped_fragments)

        # 深拷贝后只替换定位字段；实体、关系、规则和禁止项保持原始金标内容与分母。
        mapped_case = deepcopy(dict(source_case))
        mapped_case["chunk_ids"] = [section_id]
        mapped_case["evaluation_scopes"] = [{
            "chunk_id": section_id,
            "start": mapped_start,
            "end": mapped_end,
        }]
        mapped_cases.append(mapped_case)
        mapping_records.append({
            "case_id": case_id,
            "original_scopes": original_scopes,
            "semantic_section_id": section_id,
            "semantic_scope": {"start": mapped_start, "end": mapped_end},
            "content_role": sections[section_id].content_role,
            "extraction_route": sections[section_id].extraction_route,
            "section_path": list(sections[section_id].section_path),
        })

    return dataset, mapped_cases, sections, mapping_records


async def run_semantic_section_evaluation(
    client: DeepSeekGraphBuilderClient,
    *,
    gold_path: Path = DEFAULT_GOLD_PATH,
    canonical_manifest_path: Path = DEFAULT_CHUNK_MANIFEST,
    semantic_root: Path = DEFAULT_SEMANTIC_ROOT,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    output_root: Path = DEFAULT_SEMANTIC_OUTPUT,
    case_ids: set[str] | None = None,
    relation_extraction_mode: str = "generative",
    progress: Callable[[str], None] | None = None,
) -> JsonObject:
    """抽取所有命中的真实语义 section，并在模型完成后执行金标评分。"""
    dataset, mapped_cases, sections, mappings = map_cases_to_semantic_sections(
        gold_path=gold_path,
        canonical_manifest_path=canonical_manifest_path,
        semantic_root=semantic_root,
        case_ids=case_ids,
    )
    schema = load_candidate_graph_schema(schema_path)
    semantic_manifest_path = semantic_root / "manifest.json"
    semantic_manifest_sha256 = sha256_path(semantic_manifest_path)
    section_ids = [str(item["semantic_section_id"]) for item in mappings]
    section_ids = list(dict.fromkeys(section_ids))
    trace_run_id = str(uuid.uuid4())
    trace = JsonlTrace(
        output_root / "trace" / f"{trace_run_id}.jsonl",
        run_id=trace_run_id,
    )
    trace.record(
        "run/start",
        workflow="semantic_section_evaluation",
        gold_exposed_to_models=False,
        section_count=len(section_ids),
        relation_extraction_mode=relation_extraction_mode,
    )

    graphs: dict[str, JsonObject] = {}
    graph_paths: dict[str, str] = {}
    # 第一阶段只使用 section 原文、Schema 和运行配置。此循环内禁止访问案例目标字段。
    for index, section_id in enumerate(section_ids, start=1):
        section = sections[section_id]
        if progress is not None:
            progress(f"[{index}/{len(section_ids)}] {section_id} {section.section_path[-1] if section.section_path else ''}")
        slug = section_id.rsplit(":", 1)[-1]
        graph_dir = output_root / "sections" / slug / "candidate-graph"
        chunk = EvidenceChunk(section.section_id, section.text, section.input_view_sha256)
        if not first_extraction_is_usable(graph_dir):
            _ = await run_candidate_graph(
                client,
                chunk=chunk,
                schema=schema,
                schema_path=schema_path,
                output_dir=graph_dir,
                source_manifest_sha256=semantic_manifest_sha256,
                run_id=f"semantic-s1-{slug}",
                relation_extraction_mode=relation_extraction_mode,
                trace=trace,
            )
        if not first_extraction_is_usable(graph_dir):
            raise GraphBuilderConfigurationError(
                f"semantic_section_extraction_unusable:{section_id}"
            )
        graph_path = graph_dir / "graph.json"
        graphs[section_id] = cast(JsonObject, load_json_object(graph_path))
        graph_paths[section_id] = str(graph_path)

    # 第二阶段才读取 copied case 中的答案字段。模型已经全部完成，金标不会进入提示词。
    case_results: list[JsonObject] = []
    for case, mapping in zip(mapped_cases, mappings, strict=True):
        section_id = str(mapping["semantic_section_id"])
        score = cast(JsonObject, score_candidate_graph(
            graphs[section_id],
            case,
            source_text=sections[section_id].text,
        ))
        entities = cast(Mapping[str, object], score["entities"])
        relationships = cast(Mapping[str, object], score["relationships"])
        rules = cast(Mapping[str, object], score["rules"])
        case_results.append({
            "case_id": case["case_id"],
            "semantic_section_id": section_id,
            "score": score,
        })
        trace.record(
            "scoring/case",
            case_id=case["case_id"],
            section_id=section_id,
            entity_f1=entities["f1"],
            relationship_f1=relationships["f1"],
            rule_f1=rules["f1"],
        )

    supervised_gold = cast(JsonObject, aggregate_case_scores(case_results, "score"))
    prf1 = cast(JsonObject, aggregate_supervised_prf1(case_results, "score"))
    report: JsonObject = {
        "schema_version": "semantic-section-evaluation/v0.1",
        "status": "evaluation-only",
        "publication_status": "HOLD",
        "treatment_id": "semantic-section-staged-v0.1",
        "gold_status": dataset.get("status"),
        "gold_annotation_method": dataset.get("annotation_method"),
        "gold_scope_contract": dataset.get("scope_contract"),
        "gold_exposed_to_models": False,
        "case_count": len(case_results),
        "unique_section_count": len(section_ids),
        "configuration": {
            "input_unit": "semantic_section_input_view",
            "mapping_strategy": "target_blind_compacted_exact_match/v0.1",
            "relation_extraction_mode": relation_extraction_mode,
            "judge_enabled": False,
        },
        "input_hashes": {
            "semantic_manifest": semantic_manifest_sha256,
            "canonical_manifest": sha256_path(canonical_manifest_path),
            "schema": sha256_path(schema_path),
            "gold": sha256_path(gold_path),
        },
        "mapping": mappings,
        "supervised_gold": supervised_gold,
        "prf1": prf1,
        "cases": case_results,
        "section_graphs": graph_paths,
        "trace": {"run_id": trace.run_id, "events": str(trace.path)},
    }
    atomic_write_json(output_root / "evaluation-result.json", report)
    trace.record(
        "run/end",
        workflow="semantic_section_evaluation",
        status="success",
        case_count=len(case_results),
        section_count=len(section_ids),
        report_path=output_root / "evaluation-result.json",
        graph_f1=prf1["f1"],
    )
    return report


def semantic_evaluation_summary(report: Mapping[str, object]) -> JsonObject:
    """生成便于命令行查看的实体、关系、规则和全图 P/R/F1 摘要。"""
    prf1 = cast(Mapping[str, object], report["prf1"])
    categories = cast(Mapping[str, object], prf1["categories"])
    return {
        "case_count": report["case_count"],
        "unique_section_count": report["unique_section_count"],
        "entity": categories["entities"],
        "relationship": categories["relationships"],
        "rule": categories["rules"],
        "graph": prf1["graph"],
        "publication_status": report["publication_status"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行真实语义 section 候选图评测实验")
    _ = parser.add_argument("--semantic-root", type=Path, default=DEFAULT_SEMANTIC_ROOT)
    _ = parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CHUNK_MANIFEST)
    _ = parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    _ = parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    _ = parser.add_argument("--output-root", type=Path, default=DEFAULT_SEMANTIC_OUTPUT)
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
            return await run_semantic_section_evaluation(
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
        semantic_evaluation_summary(asyncio.run(_run())),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ))
