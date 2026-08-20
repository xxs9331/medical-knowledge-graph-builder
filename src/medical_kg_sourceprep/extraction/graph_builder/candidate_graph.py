"""候选图谱的分阶段抽取编排，以及命令行入口所需的辅助函数。

本模块以一个 ``EvidenceChunk`` 为处理单位，依次完成业务实体、RuleDefinition、
普通关系和规则输入输出边的抽取。模型只提出候选；每个阶段的结果都必须经过本地
确定性校验，最终只写候选工件，不直接写入 Neo4j，也不自动批准发布。
"""

from __future__ import annotations

if __package__ in {None, ""}:
    # 允许直接执行本文件查看底部 demo；正常作为包导入时不会进入该分支。
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[3]))
    __package__ = "medical_kg_sourceprep.extraction.graph_builder"

import argparse
import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor, OnError
from neo4j_graphrag.experimental.components.types import (
    Neo4jGraph,
    Neo4jNode,
    TextChunk,
    TextChunks,
)

from ..artifacts import sha256_path
from .artifacts import (
    _candidate_display,
    _public_candidate_nodes,
    candidate_summary,
    write_candidate_artifacts,
    _model_phase_failure_hold,
)
from .client import DeepSeekGraphBuilderClient, create_deepseek_graph_builder
from .contract import (
    BUSINESS_NODE_TYPES,
    DEFAULT_CHUNK_ID,
    DEFAULT_CHUNK_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCHEMA_PATH,
    DEEPSEEK_MODEL,
    ENTITY_DISCOVERY_EXAMPLES,
    ENTITY_DISCOVERY_PROMPT_TEMPLATE,
    CROSS_CHUNK_RELATION_PROMPT_TEMPLATE,
    NODE_PROMPT_TEMPLATE,
    STATE_RELATION_PROMPT_TEMPLATE,
    ORDINARY_RELATION_PROMPT_TEMPLATE,
    RULE_NODE_PROMPT_TEMPLATE,
    ORDINARY_RELATION_TYPES,
    STATE_RELATION_TYPES,
    SMOKE_TEXT,
    GraphBuilderConfigurationError,
)
from .schema import _extract_graph, build_graphrag_schema, load_candidate_graph_schema
from .relation_classifier import classify_relationships_two_stage
from .rule_gate import partition_invalid_rules
from .trace import NULL_TRACE, TraceRecorder
from .validation import (
    CandidateNormalization,
    _catalog_for_prompt,
    _hold,
    build_rule_relationships_from_definitions,
    normalize_candidate_nodes,
    normalize_candidate_relationships,
    normalize_cross_chunk_relationships,
)
from ..llm_extraction import EvidenceChunk, atomic_write_json, load_chunk_manifest


async def run_candidate_graph(
    client: DeepSeekGraphBuilderClient,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    output_dir: Path,
    source_manifest_sha256: str,
    run_id: str | None = None,
    revision_context: str = "",
    relation_extraction_mode: str = "generative",
    trace: TraceRecorder = NULL_TRACE,
) -> dict[str, Any]:
    """对一个 chunk 依次执行实体、规则、普通关系和规则边四个候选抽取阶段。

    四个阶段共享同一段原文，但后续阶段只能引用前序阶段产生的 ``VALID`` 冻结
    端点。返回值是本轮候选摘要；完整节点、关系、审查项和 Judge 队列写入
    ``output_dir``，整个流程不会连接或修改 Neo4j 数据库。
    """
    # 四个模型阶段都读取完全相同的 chunk 原文，避免预处理版本不一致造成证据漂移。
    input_text = chunk.text
    # 显式 run_id 优先；未提供时使用本轮输出目录名，保证摘要和工件目录身份一致。
    effective_run_id = run_id or output_dir.name
    if relation_extraction_mode not in {"generative", "two-stage-classification"}:
        raise GraphBuilderConfigurationError("relation_extraction_mode_invalid")

    def revised_prompt(template: str) -> str:
        """仅在二次抽取实验中附加上一轮审查反馈；默认生产行为完全不变。"""
        if not revision_context:
            return template
        # GraphRAG 会对模板执行 str.format；转义反馈中的 JSON 花括号，避免被当作占位符。
        escaped_context = revision_context.replace("{", "{{").replace("}", "}}")
        return template + """

Revision context from an independent audit of the previous extraction:
{revision_context}
Use it only to avoid previously identified extraction errors. Re-extract the complete phase from
the original Input text; do not merely patch the previous output, and do not assume the audit is
correct when it conflicts with the source or Schema.
""".replace("{revision_context}", escaped_context)

    async def extract_with_retry(
        *, phase: str, graph_schema: Any, prompt_template: str, examples: str,
        diagnostics: list[dict[str, Any]],
    ) -> tuple[Neo4jGraph | None, LLMGenerationError | None, int]:
        """每阶段总共尝试两次；失败响应只保存安全形状诊断。"""
        with trace.stage(f"extraction/{phase}", chunk_id=chunk.chunk_id) as stage:
            last_error: LLMGenerationError | None = None
            for attempt in range(1, 3):
                try:
                    graph = await _extract_graph(
                        client, chunk=chunk, graph_schema=graph_schema,
                        prompt_template=prompt_template, examples=examples, input_text=input_text,
                        response_diagnostics=diagnostics,
                    )
                    stage.update(
                        attempts=attempt,
                        proposed_node_count=len(graph.nodes),
                        proposed_relationship_count=len(graph.relationships),
                    )
                    return graph, None, attempt
                except LLMGenerationError as error:
                    last_error = error
            stage.update(status="model_error", attempts=2, error_type="LLMGenerationError")
            return None, last_error, 2

    def record_validation(phase: str, result: CandidateNormalization) -> None:
        """记录硬校验的三路分流数量，不复制候选正文或审查原因。"""
        trace.record(
            f"validation/{phase}",
            chunk_id=chunk.chunk_id,
            accepted_count=len(result.accepted),
            review_count=len(result.review_items),
            judge_draft_count=len(result.judge_drafts),
        )

    # 四个阶段产生的 Judge 草稿统一汇总，最后写入同一个 judge-queue.json。
    judge_drafts: list[dict[str, Any]] = []

    # 第一阶段：只抽取业务实体。普通实体的来源由代码按 mention 逐字定位；表格箭头等
    # 无连续状态词的场景允许模型额外给出表头和数据行双锚点，供既有校验器回放。
    entity_response_diagnostics: list[dict[str, Any]] = []
    entity_graph, entity_error, entity_attempts = await extract_with_retry(
        phase="entity",
        graph_schema=build_graphrag_schema(
            schema,
            relation_types=(),
            node_types=sorted(BUSINESS_NODE_TYPES),
            node_property_names=(
                "mention", "extraction_reason", "table_state_evidence_json",
                "derived_entity_evidence_json",
            ),
        ),
        prompt_template=revised_prompt(ENTITY_DISCOVERY_PROMPT_TEMPLATE),
        examples=ENTITY_DISCOVERY_EXAMPLES,
        diagnostics=entity_response_diagnostics,
    )
    if entity_error is not None:
        # 实体是所有后续端点的基础。实体阶段连续两次失败后，无法安全继续规则和
        # 关系抽取，因此立即写出仅包含失败审查项的工件并结束本 chunk。
        entity_nodes = []
        entity_holds = [_model_phase_failure_hold(
            stage="entity", phase="entity_phase", error=entity_error,
            response_diagnostics=entity_response_diagnostics, attempts=entity_attempts,
        )]
        write_candidate_artifacts(
            output_dir,
            schema=schema,
            schema_path=schema_path,
            chunk=chunk,
            run_id=effective_run_id,
            source_manifest_sha256=source_manifest_sha256,
            nodes=[],
            relationships=[],
            holds=entity_holds, judge_drafts=judge_drafts,
        )
        trace.record(
            "candidate-graph/artifact",
            chunk_id=chunk.chunk_id,
            graph_path=output_dir / "graph.json",
            node_count=0,
            relationship_count=0,
            review_count=len(entity_holds),
            judge_draft_count=len(judge_drafts),
            status="incomplete",
        )
        return candidate_summary(
            chunk=chunk, nodes=[], relationships=[], holds=entity_holds, output_dir=output_dir, judge_drafts=judge_drafts,
        )
    # 第一步模型只负责发现业务实体。这里进行本地确定性处理：补齐原文位置、
    # 生成 candidate_key、去重，并将结果分流到 accepted/review/Judge。
    entity_result = normalize_candidate_nodes(
        entity_graph or Neo4jGraph(),
        chunk=chunk,
        schema=schema,
        allowed_node_types=BUSINESS_NODE_TYPES,
        derive_entity_provenance=True,
    )
    record_validation("entity", entity_result)
    # CandidateNormalization 为兼容旧调用支持解包：第一个值是 accepted，第二个值
    # 是 review_items。这里的 entity_nodes 尚未发布，只是本轮保留的候选节点。
    entity_nodes, entity_holds = entity_result
    judge_drafts.extend(entity_result.judge_drafts)

    # 第二阶段抽取 RuleDefinition。_catalog_for_prompt() 不调用模型，它只把
    # entity_nodes 中的 VALID 实体裁剪成冻结目录 JSON，作为 examples 参数放入提示词。
    # 规则模型只能引用该目录中的实体，不能在规则阶段重新创建业务实体。
    rule_response_diagnostics: list[dict[str, Any]] = []
    rule_graph, rule_error, rule_attempts = await extract_with_retry(
        phase="rule-node",
        graph_schema=build_graphrag_schema(schema, relation_types=(), node_types=("RuleDefinition",)),
        prompt_template=revised_prompt(RULE_NODE_PROMPT_TEMPLATE),
        examples=_catalog_for_prompt(entity_nodes), diagnostics=rule_response_diagnostics,
    )
    if rule_error is not None:
        # 连续两次模型调用失败时，本阶段不产生规则节点，只留下阶段失败审查项。
        rule_nodes = []
        rule_holds = [_model_phase_failure_hold(
            stage="rule", phase="rule_phase", error=rule_error, response_diagnostics=rule_response_diagnostics,
            attempts=rule_attempts,
        )]
    else:
        # 模型成功返回后，RuleDefinition 仍需经过节点校验；不能直接并入候选图。
        rule_result = normalize_candidate_nodes(
            rule_graph or Neo4jGraph(),
            chunk=chunk,
            schema=schema,
            allowed_node_types=("RuleDefinition",),
            allowed_rule_stages=("GRAPH_COMPOSITE",),
            allowed_rule_logics=("ALL", "ALL_SAME_WINDOW"),
        )
        record_validation("rule-node", rule_result)
        rule_nodes, rule_holds = rule_result
        judge_drafts.extend(rule_result.judge_drafts)
        # 规则不能把自身条件重新声明为结论，也不能把“如/例如”引出的例子当结论。
        # 两类显式结构错误都在模型提示词之后确定性拒绝，不依赖疾病名称或案例金标。
        rule_nodes, rejected_rules = partition_invalid_rules(rule_nodes)
        rule_holds.extend(
            _hold("rule", index, str(item["reason_code"]), item)
            for index, item in enumerate(rejected_rules)
        )
    # 此时只有节点，没有关系。VALID 和 PARTIAL 都可保留在候选集合中，
    # 但只有 VALID 可以成为随后关系模型及关系校验器可引用的端点。
    nodes = [*entity_nodes, *rule_nodes]
    frozen_nodes = [item for item in nodes if item.get("extraction_status") == "VALID"]

    # 第三阶段只绑定 LabIndicator -> IndicatorState。封闭任务单独穷举，可以避免
    # HAS_STATE 被更显眼的疾病关联或因果关系挤出模型输出。
    state_response_diagnostics: list[dict[str, Any]] = []
    state_audit: dict[str, Any] | None = None
    if relation_extraction_mode == "two-stage-classification":
        state_relation_graph, state_audit = await classify_relationships_two_stage(
            client, chunk=chunk, schema=schema, nodes=frozen_nodes,
            allowed_relation_types=STATE_RELATION_TYPES, trace=trace,
        )
        state_error, state_attempts = None, 1
    else:
        state_relation_graph, state_error, state_attempts = await extract_with_retry(
            phase="state-relation",
            graph_schema=build_graphrag_schema(
                schema, relation_types=sorted(STATE_RELATION_TYPES),
                node_types=sorted(BUSINESS_NODE_TYPES), node_property_names=("mention",),
                relationship_property_names=("exact_quote", "exact_quote_occurrence_index"),
            ),
            prompt_template=revised_prompt(STATE_RELATION_PROMPT_TEMPLATE),
            examples=_catalog_for_prompt(entity_nodes), diagnostics=state_response_diagnostics,
        )
    if state_error is not None:
        state_result = normalize_candidate_relationships(
            Neo4jGraph(), chunk=chunk, schema=schema, nodes=frozen_nodes,
            allowed_relation_types=sorted(STATE_RELATION_TYPES), validate_rule_structures=False,
        )
        state_result.review_items.append(_model_phase_failure_hold(
            stage="relation", phase="state_relation_phase", error=state_error,
            response_diagnostics=state_response_diagnostics, attempts=state_attempts,
        ))
    else:
        state_result = normalize_candidate_relationships(
            state_relation_graph or Neo4jGraph(), chunk=chunk, schema=schema,
            nodes=frozen_nodes, allowed_relation_types=sorted(STATE_RELATION_TYPES),
            validate_rule_structures=False,
        )
    record_validation("state-relation", state_result)
    state_relationships, state_holds = state_result
    judge_drafts.extend(state_result.judge_drafts)

    # 第四阶段抽取普通业务关系。这里仍然传 entity_nodes 的冻结目录，因为普通关系
    # 只能连接业务实体，不应把 RuleDefinition 当作普通关系端点。
    ordinary_response_diagnostics: list[dict[str, Any]] = []
    ordinary_audit: dict[str, Any] | None = None
    if relation_extraction_mode == "two-stage-classification":
        ordinary_relation_graph, ordinary_audit = await classify_relationships_two_stage(
            client, chunk=chunk, schema=schema, nodes=frozen_nodes,
            allowed_relation_types=ORDINARY_RELATION_TYPES, trace=trace,
        )
        ordinary_error, ordinary_attempts = None, 1
    else:
        ordinary_relation_graph, ordinary_error, ordinary_attempts = await extract_with_retry(
            phase="ordinary-relation",
            graph_schema=build_graphrag_schema(
                schema, relation_types=sorted(ORDINARY_RELATION_TYPES), node_types=sorted(BUSINESS_NODE_TYPES),
                # GraphRAG 要求每种节点至少声明一个属性；关系阶段仅保留 mention，
                # 不再重复实体来源、规则和生命周期等无关字段。
                node_property_names=("mention",),
                relationship_property_names=("exact_quote", "exact_quote_occurrence_index"),
            ),
            prompt_template=revised_prompt(ORDINARY_RELATION_PROMPT_TEMPLATE),
            examples=_catalog_for_prompt(entity_nodes),
            diagnostics=ordinary_response_diagnostics,
        )
    if ordinary_error is not None:
        # 即使模型阶段失败，也调用一次空图校验以获得统一的 CandidateNormalization，
        # 再追加阶段失败项，避免成功与失败分支返回不同的数据结构。
        ordinary_result = normalize_candidate_relationships(
            Neo4jGraph(),
            chunk=chunk,
            schema=schema,
            nodes=frozen_nodes,
            allowed_relation_types=sorted(ORDINARY_RELATION_TYPES),
            validate_rule_structures=False,
        )
        ordinary_result.review_items.append(_model_phase_failure_hold(
            stage="relation", phase="ordinary_relation_phase", error=ordinary_error,
            response_diagnostics=ordinary_response_diagnostics, attempts=ordinary_attempts,
        ))
    else:
        # 关系模型只提出候选；本地仍会复验关系类型、冻结端点、原文证据和重复项。
        ordinary_result = normalize_candidate_relationships(
            ordinary_relation_graph or Neo4jGraph(), chunk=chunk, schema=schema, nodes=frozen_nodes,
            allowed_relation_types=sorted(ORDINARY_RELATION_TYPES),
            validate_rule_structures=False,
        )
    record_validation("ordinary-relation", ordinary_result)
    ordinary_relationships, ordinary_holds = ordinary_result
    judge_drafts.extend(ordinary_result.judge_drafts)

    # 第五阶段把 RuleDefinition 的结构化表达式确定性投影为 RULE_INPUT / RULE_OUTPUT。
    # 规则语义已经由上一阶段模型提出；这里不再让另一次模型调用重复解释输入输出方向。
    if any(item.get("entity_type") == "RuleDefinition" for item in frozen_nodes):
        rule_result = build_rule_relationships_from_definitions(
            schema=schema, nodes=frozen_nodes
        )
        record_validation("rule-edge", rule_result)
        rule_relationships, rule_edge_holds = rule_result
        invalid_rule_keys = rule_result.invalid_rule_keys
        judge_drafts.extend(rule_result.judge_drafts)
        trace.record(
            "extraction/rule-edge/deterministic",
            chunk_id=chunk.chunk_id,
            relationship_count=len(rule_relationships),
        )
    else:
        # 没有 VALID 规则节点时，规则边阶段自然为空，不属于模型调用失败。
        rule_relationships, rule_edge_holds, invalid_rule_keys = [], [], set()
        trace.record(
            "extraction/rule-edge/skipped",
            chunk_id=chunk.chunk_id,
            reason="no_valid_rule_definition",
        )

    # 合并两类关系。若规则结构复验认定某条规则无效，则从公开候选节点中移除该规则，
    # 但相关原因仍保留在审查队列，不能无记录地静默丢弃。
    relationships = [*state_relationships, *ordinary_relationships, *rule_relationships]
    nodes = _public_candidate_nodes(
        [node for node in nodes if node["candidate_key"] not in invalid_rule_keys]
    )
    # 五阶段审查项按执行顺序合并，便于回放一个 chunk 的完整处理过程。
    holds = [*entity_holds, *rule_holds, *state_holds, *ordinary_holds, *rule_edge_holds]

    # 最终统一写出 graph.json、review-queue.json、judge-queue.json 和运行清单。
    # 所有候选仍为 HOLD；此处没有 Neo4j 数据库写入动作。
    write_candidate_artifacts(
        output_dir,
        schema=schema,
        schema_path=schema_path,
        chunk=chunk,
        run_id=effective_run_id,
        source_manifest_sha256=source_manifest_sha256,
        nodes=nodes,
        relationships=relationships,
        holds=holds, judge_drafts=judge_drafts,
    )
    if relation_extraction_mode == "two-stage-classification":
        atomic_write_json(output_dir / "relation-classification.json", {
            "schema_version": "two-stage-relation-classification-run/v0.1",
            "status": "experiment-only",
            "chunk_id": chunk.chunk_id,
            "state_relations": state_audit,
            "ordinary_relations": ordinary_audit,
        })
    trace.record(
        "candidate-graph/artifact",
        chunk_id=chunk.chunk_id,
        graph_path=output_dir / "graph.json",
        node_count=len(nodes),
        relationship_count=len(relationships),
        review_count=len(holds),
        judge_draft_count=len(judge_drafts),
    )
    return candidate_summary(
        chunk=chunk, nodes=nodes, relationships=relationships, holds=holds, output_dir=output_dir,
        judge_drafts=judge_drafts,
    )


async def extract_cross_chunk_relationships(
    client: DeepSeekGraphBuilderClient,
    *,
    chunks: Sequence[EvidenceChunk],
    schema: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    trace: TraceRecorder = NULL_TRACE,
) -> CandidateNormalization:
    """基于多个真实 EvidenceChunk 提出并校验跨 chunk 普通关系。"""
    if len(chunks) < 2:
        raise GraphBuilderConfigurationError("cross_chunk_extraction_requires_multiple_chunks")
    frozen_nodes = [
        item for item in nodes
        if item.get("extraction_status") == "VALID" and item.get("entity_type") in BUSINESS_NODE_TYPES
    ]
    source_chunks_json = json.dumps(
        {
            "source_chunks": [
                {"chunk_id": chunk.chunk_id, "chunk_sha256": chunk.chunk_sha256, "text": chunk.text}
                for chunk in chunks
            ]
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    graph_schema = build_graphrag_schema(
        schema,
        relation_types=sorted(ORDINARY_RELATION_TYPES),
        node_types=sorted(BUSINESS_NODE_TYPES),
        node_property_names=("mention",),
        relationship_property_names=("relation_evidence_json",),
    )
    diagnostics: list[dict[str, Any]] = []
    with trace.stage(
        "extraction/cross-chunk-relation",
        chunk_ids=[chunk.chunk_id for chunk in chunks],
    ) as stage:
        last_error: LLMGenerationError | None = None
        for attempt in range(1, 3):
            try:
                graph = await _extract_graph(
                    client,
                    # GraphRAG 需要一个技术 uid；真正来源身份由 relation_evidence_json
                    # 中的规范 chunk_id 表达，并由多 chunk 校验器逐项回放。
                    chunk=chunks[0],
                    graph_schema=graph_schema,
                    prompt_template=CROSS_CHUNK_RELATION_PROMPT_TEMPLATE,
                    examples=_catalog_for_prompt(frozen_nodes),
                    input_text=source_chunks_json,
                    response_diagnostics=diagnostics,
                )
                result = normalize_cross_chunk_relationships(
                    graph,
                    chunks=chunks,
                    schema=schema,
                    nodes=frozen_nodes,
                    allowed_relation_types=ORDINARY_RELATION_TYPES,
                )
                stage.update(
                    attempts=attempt,
                    proposed_relationship_count=len(graph.relationships),
                    accepted_count=len(result.accepted),
                    review_count=len(result.review_items),
                )
                return result
            except LLMGenerationError as error:
                last_error = error
        stage.update(status="model_error", attempts=2, error_type="LLMGenerationError")
    assert last_error is not None
    raise last_error


async def run_candidate_block(
    client: DeepSeekGraphBuilderClient,
    *,
    chunk_id: str = DEFAULT_CHUNK_ID,
    manifest_path: Path = DEFAULT_CHUNK_MANIFEST,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
) -> dict[str, Any]:
    """从规范清单加载一个 chunk，并在全新的候选运行目录中执行完整流程。"""
    # Schema 和 chunk 都从规范文件读取，调用方不能传入未登记的临时 chunk_id。
    schema = load_candidate_graph_schema(schema_path)
    _manifest, chunks = load_chunk_manifest(manifest_path)
    selected = next((item for item in chunks if item.chunk_id == chunk_id), None)
    if selected is None:
        raise GraphBuilderConfigurationError(f"chunk_id is not in the canonical manifest: {chunk_id}")
    # 每次运行使用独立目录；run_id 只允许文件系统安全字符，且禁止覆盖已有运行清单。
    effective_run_id = run_id or _default_run_id(chunk_id)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", effective_run_id):
        raise GraphBuilderConfigurationError("run_id contains unsupported characters")
    run_output_dir = output_dir / effective_run_id
    if (run_output_dir / "run-manifest.json").exists():
        raise GraphBuilderConfigurationError(f"run_id already exists: {effective_run_id}")
    return await run_candidate_graph(
        client,
        chunk=selected,
        schema=schema,
        schema_path=schema_path,
        output_dir=run_output_dir,
        source_manifest_sha256=sha256_path(manifest_path),
        run_id=effective_run_id,
    )


def _default_run_id(chunk_id: str) -> str:
    """根据 chunk ID 和 UTC 时间生成不会互相覆盖的默认运行 ID。"""
    chunk_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", chunk_id).strip("-")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{chunk_slug}-{timestamp}"


async def run_smoke(client: DeepSeekGraphBuilderClient) -> dict[str, int | str]:
    """用固定的非患者文本做内存冒烟测试，只返回节点和关系数量。"""
    # DeepSeek 仅支持 JSON Object 模式，不支持 GraphRAG 的 JSON Schema 结构化模式。
    extractor = LLMEntityRelationExtractor(
        llm=client.llm,
        create_lexical_graph=False,
        on_error=OnError.RAISE,
        max_concurrency=1,
        use_structured_output=False,
    )
    graph = await extractor.run(
        chunks=TextChunks(chunks=[TextChunk(text=SMOKE_TEXT, index=0)])
    )
    return {
        "model": DEEPSEEK_MODEL,
        "node_count": len(graph.nodes),
        "relationship_count": len(graph.relationships),
    }


async def _run_smoke_main() -> dict[str, int | str]:
    """创建并可靠关闭客户端，供命令行冒烟测试入口调用。"""
    client = create_deepseek_graph_builder()
    try:
        return await run_smoke(client)
    finally:
        await client.aclose()


async def _run_candidate_main(args: argparse.Namespace) -> dict[str, Any]:
    """把命令行参数转交给单 chunk 候选运行，并在结束时关闭客户端。"""
    client = create_deepseek_graph_builder()
    try:
        return await run_candidate_block(
            client,
            chunk_id=args.chunk_id,
            manifest_path=args.manifest,
            schema_path=args.schema,
            output_dir=args.output,
            run_id=args.run_id,
        )
    finally:
        await client.aclose()


if __name__ == "__main__":
    # 真实普通关系抽取 demo：读取一个规范 chunk，但跳过实体模型调用，直接使用此前
    # 已观察到的实体模型结果。随后执行实体本地校验、生成关系提示词、调用一次
    # 关系模型并执行关系本地校验。不会抽取规则，也不会写候选工件或 Neo4j。
    #
    # 本 demo 不主动 print。需要观察中间结果时，可以自行打印：
    # chunk.text、entity_result、entity_nodes、frozen_entity_catalog_json、
    # rendered_prompt、relationship_graph、relationship_result 或 diagnostics。
    async def _ordinary_relation_demo() -> None:
        # 选择已经实际运行过实体抽取的 chunk，保证下方记录的实体结果有真实来源。
        chunk_id = "clinical-hematology:chapter-01:0012:0000"

        # 加载候选图 Schema，以及规范 manifest 中登记的所有 chunk。
        schema = load_candidate_graph_schema(DEFAULT_SCHEMA_PATH)
        _manifest, chunks = load_chunk_manifest(DEFAULT_CHUNK_MANIFEST)
        # 根据 chunk_id 取得原文；关系模型和本地证据校验都使用同一份 chunk.text。
        chunk = next(item for item in chunks if item.chunk_id == chunk_id)

        # 这些语义字段来自该 chunk 此前的真实实体模型输出。重复的“转铁蛋白”和
        # “总铁结合力”也原样保留，以便先观察本地实体去重，再抽取关系。
        raw_entity_nodes = [
            ("ClinicalContext", "严重的肝病", "原文明示的疾病背景，影响检验结果解释。"),
            ("ClinicalContext", "营养不良", "原文明示的临床背景，影响检验结果解释。"),
            ("LabIndicator", "转铁蛋白", "原文明示的检验指标。"),
            ("IndicatorState", "转铁蛋白合成减少", "原文明示该指标的减少状态。"),
            ("ClinicalContext", "肾病综合征", "原文明示的疾病背景，影响检验结果解释。"),
            ("ClinicalContext", "大量蛋白质从尿液丢失", "原文明示的病理机制背景。"),
            ("LabIndicator", "转铁蛋白", "原文明示的检验指标。"),
            ("IndicatorState", "转铁蛋白减少", "原文明示该指标的减少状态。"),
            ("LabIndicator", "总铁结合力", "原文明示的检验指标。"),
            ("LabIndicator", "血清铁", "原文明示的检验指标。"),
            ("LabIndicator", "总铁结合力", "原文明示的检验指标。"),
            ("LabIndicator", "亚铁嗪显色法", "原文明示的检验方法，作为检验指标背景。"),
        ]

        # 将记录下来的轻量实体结果包装成 GraphRAG 的 Neo4jGraph。这里的 id 只是
        # 本次内存响应中的临时 ID，不会成为最终 candidate_key，也不会写入数据库。
        entity_graph = Neo4jGraph(nodes=[
            Neo4jNode(
                id=f"recorded-entity-{index}",
                label=label,
                properties={"mention": mention, "extraction_reason": reason},
            )
            for index, (label, mention, reason) in enumerate(raw_entity_nodes)
        ])

        # 执行实体本地硬校验：检查允许类型和 mention，使用代码定位原文位置，
        # 生成稳定 candidate_key，并去除同一响应中的重复实体。
        entity_result = normalize_candidate_nodes(
            entity_graph,
            chunk=chunk,
            schema=schema,
            allowed_node_types=BUSINESS_NODE_TYPES,
            derive_entity_provenance=True,
        )
        # accepted 是本地允许保留的候选节点；实体校验失败项分别位于
        # entity_result.review_items 和 entity_result.judge_drafts。
        entity_nodes = entity_result.accepted

        # 将 accepted 中的 VALID 实体裁剪成关系模型唯一允许引用的冻结目录。
        # PARTIAL、审查状态、完整来源坐标等不会被塞入关系提示词。
        frozen_entity_catalog_json = _catalog_for_prompt(entity_nodes)

        # 构造普通关系阶段的 GraphRAG Schema：允许业务实体作为端点，只允许
        # ORDINARY_RELATION_TYPES 中声明的普通关系，不允许规则输入输出边。
        relationship_schema = build_graphrag_schema(
            schema,
            relation_types=sorted(ORDINARY_RELATION_TYPES),
            node_types=sorted(BUSINESS_NODE_TYPES),
            node_property_names=("mention",),
            relationship_property_names=("exact_quote", "exact_quote_occurrence_index"),
        )

        # 提前渲染一份最终提示词，便于自行 print 检查。真正调用模型时
        # _extract_graph 会使用相同模板、Schema、冻结目录和 chunk 原文再次渲染。
        rendered_prompt = ORDINARY_RELATION_PROMPT_TEMPLATE.format(
            schema=relationship_schema.model_dump(exclude_none=True),
            examples=frozen_entity_catalog_json,
            text=chunk.text,
        )
        print(rendered_prompt)
        # 创建真实 DeepSeek 客户端。本 demo 从这里开始会产生一次模型调用费用。
        client = create_deepseek_graph_builder()
        # diagnostics 只记录响应结构和 token 用量等安全诊断，不保存完整模型响应。
        diagnostics: list[dict[str, Any]] = []
        try:
            # 只执行普通关系抽取。模型返回 Neo4jGraph，其中 nodes 应为空，
            # relationships 的两个端点必须引用 frozen_entity_catalog_json 中的 candidate_key。
            relationship_graph = await _extract_graph(
                client,
                chunk=chunk,
                graph_schema=relationship_schema,
                prompt_template=ORDINARY_RELATION_PROMPT_TEMPLATE,
                examples=frozen_entity_catalog_json,
                input_text=chunk.text,
                response_diagnostics=diagnostics,
            )
        finally:
            # 客户端必须在创建它的同一个事件循环中关闭，避免 Event loop is closed。
            await client.aclose()

        # 模型输出仍只是候选。这里复验关系类型、冻结端点、原文引语、cue、方向和重复项，
        # 最终分流到 accepted、review_items 和 judge_drafts；不会写 Neo4j。
        relationship_result = normalize_candidate_relationships(
            relationship_graph,
            chunk=chunk,
            schema=schema,
            nodes=[node for node in entity_nodes if node.get("extraction_status") == "VALID"],
            allowed_relation_types=sorted(ORDINARY_RELATION_TYPES),
            validate_rule_structures=False,
        )
        print(relationship_result)
    # asyncio.run 创建并管理本次 demo 的事件循环，使模型调用与客户端关闭处于同一循环。
    asyncio.run(_ordinary_relation_demo())
