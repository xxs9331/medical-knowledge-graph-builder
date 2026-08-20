"""业务实体和 RuleDefinition 候选节点的本地接纳逻辑。

模型的职责是提出候选类型、名称和抽取理由；本模块不判断医学含义是否正确，而是
用确定性规则补齐或复验来源位置、生成稳定候选键、去除重复，并把每条记录分流为：

- ``accepted``：可回放的候选记录，可能是完整的 ``VALID``，也可能是待复核的 ``PARTIAL``；
- ``review_items``：说明被拒绝或需要进一步复核的机器可读审查记录；
- ``judge_drafts``：无法本地接纳、但保留了足够身份信息以供后续 Judge 处理的最小草稿。
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
import re
from typing import Any

from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jNode

if __package__ in {None, ""}:
    # 允许直接执行本文件观察底部演示；正常作为包导入时仍使用相对导入。
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from medical_kg_sourceprep.extraction.graph_builder.contract import (
        RULE_LOGIC_TYPES,
        TRIAL_NODE_TYPES,
        GraphBuilderConfigurationError,
    )
    from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk
    from medical_kg_sourceprep.extraction.graph_builder.validation.provenance import (
        _candidate_key,
        _derived_entity_candidate_key,
        _normalize_rule_expression,
        _parse_derived_entity_evidence,
        _parse_rule_endpoint_mentions,
        _parse_rule_evidence,
        _parse_table_state_evidence,
        _rule_candidate_key,
        _rule_expression_endpoints,
        _source_ref,
        _source_refs_for_mention,
        _table_state_candidate_key,
    )
    from medical_kg_sourceprep.extraction.graph_builder.validation.result import CandidateNormalization
    from medical_kg_sourceprep.extraction.graph_builder.validation.review import (
        _hold,
        _judge_draft,
        _mark_partial,
        _node_judge_draft,
        _node_summary,
        _relationship_judge_draft,
        _relationship_summary,
        _review_item,
    )
else:
    from ..contract import RULE_LOGIC_TYPES, TRIAL_NODE_TYPES, GraphBuilderConfigurationError
    from ...llm_extraction import EvidenceChunk
    from .provenance import (
        _candidate_key,
        _derived_entity_candidate_key,
        _normalize_rule_expression,
        _parse_derived_entity_evidence,
        _parse_rule_endpoint_mentions,
        _parse_rule_evidence,
        _parse_table_state_evidence,
        _rule_candidate_key,
        _rule_expression_endpoints,
        _source_ref,
        _source_refs_for_mention,
        _table_state_candidate_key,
    )
    from .result import CandidateNormalization
    from .review import (
        _hold,
        _judge_draft,
        _mark_partial,
        _node_judge_draft,
        _node_summary,
        _relationship_judge_draft,
        _relationship_summary,
        _review_item,
    )


def _derive_unique_exclusion_evidence(
    chunk: EvidenceChunk, endpoints: Sequence[str],
) -> list[dict[str, Any]] | None:
    """仅在唯一原句含全部端点和排除词时确定性补齐证据。"""
    candidates: list[str] = []
    for match in re.finditer(r"[^。\n]*(?:排除|不能|不可能|无|不支持|不科学)[^。\n]*。?", chunk.text):
        quote = match.group(0).strip()
        if "<table" in quote.lower():
            continue
        normalized = re.sub(r"\s+", "", quote)
        if quote and all(re.sub(r"\s+", "", endpoint) in normalized for endpoint in endpoints):
            candidates.append(quote)
    if len(candidates) != 1:
        return None
    return _parse_rule_evidence(
        chunk,
        [{"role": "diagnostic_exclusion", "exact_quote": candidates[0]}],
    )


def normalize_candidate_nodes(
    graph: Neo4jGraph,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    allowed_node_types: Collection[str] = TRIAL_NODE_TYPES,
    allowed_rule_stages: Collection[str] = ("PREPROCESS", "GRAPH_COMPOSITE", "UNKNOWN"),
    allowed_rule_logics: Collection[str] = RULE_LOGIC_TYPES,
    derive_entity_provenance: bool = False,
) -> CandidateNormalization:
    """校验一个模型节点列表，并返回候选、审查项和 Judge 草稿。

    参数 ``graph`` 是模型返回的 ``Neo4jGraph``，本函数只读取其中的节点；若实体
    阶段错误返回了关系，也会作为阶段错位记录写入审查项。``allowed_node_types``
    让同一个函数可复用在实体阶段和仅允许 ``RuleDefinition`` 的规则阶段；
    ``allowed_rule_stages`` 和 ``allowed_rule_logics`` 用于让具体工作流进一步收窄规则边界。

    ``derive_entity_provenance`` 仅用于轻量实体发现：模型只输出类型、mention 和
    extraction_reason，代码据逐字出现回填普通实体来源。它不能为表格箭头等非连续
    语义状态编造锚点，因此这类记录仍进入 Judge 队列。
    """
    # Schema 的整体结构已在进入本函数前加载并验证；这里仅需要调用阶段给出的类型白名单。
    del schema
    accepted: list[dict[str, Any]] = []
    # RuleDefinition 的身份依赖结构化输入、输出和证据位置，故也在第一轮后统一生成。
    pending_rules: list[
        tuple[
            int, str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...], str, str,
            list[dict[str, Any]], tuple[str, ...], Any, Any,
        ]
    ] = []
    # holds 是历史名称；其中放的是 review queue 项，既包含 REJECTED 也包含 REVIEW_REQUIRED。
    holds: list[dict[str, Any]] = []
    judge_drafts: list[dict[str, Any]] = []
    # 同一次模型响应中，相同候选键只保留首条，后续重复项只记审查记录。
    seen_keys: set[str] = set()

    # 第一轮只验证每个模型节点自身；跨节点绑定在后续统一处理。
    for index, node in enumerate(graph.nodes):
        summary = _node_summary(node)
        try:
            entity_type = node.label
            properties = node.properties
            # label 不在当前阶段允许范围内，说明模型把其他阶段的内容混进来了。
            if entity_type not in allowed_node_types:
                raise GraphBuilderConfigurationError("entity_type_not_enabled_for_trial")
            if entity_type == "RuleDefinition":
                # RuleDefinition 不是业务实体，不能带 mention / exact_quote 这一套实体字段。
                if any(properties.get(field) not in (None, "") for field in (
                    "mention", "canonical_name_candidate", "exact_quote"
                )):
                    raise GraphBuilderConfigurationError("rule_definition_uses_business_fields")
                raw_rule_stage = properties.get("rule_stage_candidate")
                rule_warnings: list[str] = []
                # 阶段值未知时仍保留规则候选，但标记为 UNKNOWN，后续会降为 PARTIAL。
                if isinstance(raw_rule_stage, str) and raw_rule_stage in {
                    "PREPROCESS", "GRAPH_COMPOSITE", "UNKNOWN",
                }:
                    rule_stage = raw_rule_stage
                else:
                    rule_stage = "UNKNOWN"
                    rule_warnings.append("RULE_STAGE_UNKNOWN")
                if rule_stage not in allowed_rule_stages:
                    raise GraphBuilderConfigurationError("rule_stage_not_enabled_for_trial")
                raw_inputs = properties.get("rule_inputs_json")
                raw_outputs = properties.get("rule_outputs_json")
                raw_excluded_outputs = properties.get("rule_excluded_outputs_json")
                if (
                    raw_inputs not in (None, "")
                    or raw_outputs not in (None, "")
                    or raw_excluded_outputs not in (None, "")
                ):
                    # 新合同直接使用 JSON 端点数组；名称和表达式不参与结构判断。
                    rule_inputs = _parse_rule_endpoint_mentions(
                        raw_inputs, field="rule_inputs", minimum=1,
                    )
                    rule_outputs = _parse_rule_endpoint_mentions(
                        raw_outputs if raw_outputs not in (None, "") else [],
                        field="rule_outputs", minimum=0,
                    )
                    rule_excluded_outputs = _parse_rule_endpoint_mentions(
                        raw_excluded_outputs if raw_excluded_outputs not in (None, "") else [],
                        field="rule_excluded_outputs", minimum=0,
                    )
                    if not rule_outputs and not rule_excluded_outputs:
                        raise GraphBuilderConfigurationError("rule_outputs_insufficient")
                    raw_expression = properties.get("rule_expression")
                    expression = raw_expression.strip() if isinstance(raw_expression, str) else ""
                else:
                    # 兼容历史候选：旧工件仍可从 r=A(a,b) 表达式恢复端点。
                    expression, _output, _expression_name = _normalize_rule_expression(
                        properties.get("rule_expression")
                    )
                    rule_outputs, rule_inputs = _rule_expression_endpoints(expression)
                    rule_excluded_outputs = ()
                raw_rule_name = properties.get("rule_name")
                rule_name = raw_rule_name.strip() if isinstance(raw_rule_name, str) else ""
                # 模型已给出完整排除端点但漏证据时，只允许从唯一原句确定性回填坐标。
                # 多句或端点不齐仍走严格拒绝，不能借此补医学语义。
                raw_evidence = properties.get("rule_evidence_json")
                evidence_refs = (
                    _derive_unique_exclusion_evidence(
                        chunk, (*rule_inputs, *rule_excluded_outputs)
                    )
                    if rule_excluded_outputs
                    else None
                )
                if evidence_refs is None:
                    evidence_refs = _parse_rule_evidence(chunk, raw_evidence)
                raw_rule_logic = properties.get("rule_logic_candidate")
                if isinstance(raw_rule_logic, str) and raw_rule_logic in RULE_LOGIC_TYPES:
                    rule_logic = raw_rule_logic
                elif raw_rule_logic in (None, ""):
                    # 兼容旧候选：只从证据角色和阶段恢复不含医学判断的明显逻辑形状。
                    evidence_roles = {str(item["role"]).lower() for item in evidence_refs}
                    if "formula" in evidence_roles:
                        rule_logic = "FORMULA"
                    elif rule_stage == "GRAPH_COMPOSITE":
                        rule_logic = "ALL"
                    elif any("trend" in role or "time" in role for role in evidence_roles):
                        rule_logic = "TREND"
                    elif any("range" in role or "table" in role or "severity" in role for role in evidence_roles):
                        rule_logic = "RANGE_TABLE"
                    else:
                        rule_logic = "UNKNOWN"
                        rule_warnings.append("RULE_LOGIC_UNKNOWN")
                else:
                    rule_logic = "UNKNOWN"
                    rule_warnings.append("RULE_LOGIC_UNKNOWN")
                if rule_logic not in allowed_rule_logics:
                    raise GraphBuilderConfigurationError("rule_logic_not_enabled_for_trial")
                pending_rules.append((
                    index,
                    rule_stage,
                    rule_logic,
                    rule_inputs,
                    rule_outputs,
                    rule_excluded_outputs,
                    expression,
                    rule_name,
                    evidence_refs,
                    tuple(rule_warnings),
                    raw_rule_stage,
                    raw_rule_logic,
                ))
                continue

            mention = properties.get("mention")
            # 业务实体至少必须有名称；没有名称既不能检索原文，也不能形成 Judge 的最小身份。
            if not isinstance(mention, str) or not mention:
                raise GraphBuilderConfigurationError("mention_missing")
            canonical = properties.get("canonical_name_candidate")
            if derive_entity_provenance:
                # 轻量阶段不要求模型做规范化，先以原文 mention 作为候选规范名。
                canonical = mention
            elif not isinstance(canonical, str) or not canonical:
                raise GraphBuilderConfigurationError("canonical_name_missing")
            table_state_evidence = properties.get("table_state_evidence_json")
            derived_entity_evidence = properties.get("derived_entity_evidence_json")
            if table_state_evidence is not None and derived_entity_evidence is not None:
                raise GraphBuilderConfigurationError("entity_uses_multiple_derived_evidence_contracts")
            if table_state_evidence is not None:
                # 这是表格箭头等“原文没有连续状态词”的特殊通道：必须依赖模型提供的
                # 表头和行双锚点回放，不允许同时伪造普通实体 exact_quote。
                if entity_type != "IndicatorState":
                    raise GraphBuilderConfigurationError("table_state_evidence_requires_indicator_state")
                if properties.get("exact_quote") not in (None, ""):
                    raise GraphBuilderConfigurationError("table_state_evidence_uses_exact_quote")
                source_ref, table_state_evidence_refs = _parse_table_state_evidence(
                    chunk, value=table_state_evidence
                )
                candidate_key = _table_state_candidate_key(
                    chunk=chunk, mention=mention, evidence_refs=table_state_evidence_refs
                )
                source_refs = [source_ref]
                derived_entity_evidence_refs: list[dict[str, Any]] = []
                derivation_type: str | None = None
            elif derived_entity_evidence is not None:
                if properties.get("exact_quote") not in (None, ""):
                    raise GraphBuilderConfigurationError("derived_entity_evidence_uses_exact_quote")
                derivation_type, source_ref, derived_entity_evidence_refs = (
                    _parse_derived_entity_evidence(chunk, value=derived_entity_evidence)
                )
                candidate_key = _derived_entity_candidate_key(
                    entity_type=entity_type,
                    mention=mention,
                    derivation_type=derivation_type,
                    evidence_refs=derived_entity_evidence_refs,
                )
                source_refs = [source_ref]
                table_state_evidence_refs = []
            elif derive_entity_provenance:
                # 代码查找 mention 的每一次逐字出现：第一处为主锚点，所有位置都保留在
                # source_refs，防止后续关系或规则误把“首次出现”当作唯一证据。
                source_refs = _source_refs_for_mention(chunk, mention)
                source_ref = source_refs[0]
                table_state_evidence_refs = []
                derived_entity_evidence_refs = []
                derivation_type = None
                candidate_key = _candidate_key(entity_type, mention, source_ref)
            else:
                # 旧的完整提示词模式由模型提供 exact_quote 和位置，代码只接受可逐字回放的值。
                source_ref = _source_ref(
                    chunk,
                    mention,
                    properties.get("exact_quote"),
                    exact_quote_occurrence_index=properties.get("exact_quote_occurrence_index"),
                    mention_occurrence_index=properties.get("mention_occurrence_index"),
                    source_char_start=properties.get("source_char_start"),
                    source_char_end=properties.get("source_char_end"),
                )
                if canonical not in source_ref["exact_quote"]:
                    raise GraphBuilderConfigurationError("canonical_name_not_in_exact_quote")
                table_state_evidence_refs = []
                derived_entity_evidence_refs = []
                derivation_type = None
                candidate_key = _candidate_key(entity_type, mention, source_ref)
                source_refs = [source_ref]
            if candidate_key in seen_keys:
                raise GraphBuilderConfigurationError("duplicate_candidate")
            seen_keys.add(candidate_key)
            # 到此为止已具备类型、名称、来源和稳定身份；所有候选仍是 HOLD，尚未发布。
            record: dict[str, Any] = {
                "candidate_key": candidate_key,
                "entity_type": entity_type,
                "mention": mention,
                "canonical_name_candidate": canonical,
                "source_ref": source_ref,
                "extraction_status": "VALID",
                "review_status": "PENDING",
                "publication_status": "HOLD",
            }
            extraction_reason = properties.get("extraction_reason")
            if isinstance(extraction_reason, str) and extraction_reason.strip():
                # 理由来自模型，保留给后续语义评测或人工追溯，不作为硬校验判断依据。
                record["extraction_reason"] = extraction_reason.strip()
            if len(source_refs) > 1:
                # source_ref 是稳定的主锚点；source_refs 保存同名词其余逐字出现位置。
                record["source_refs"] = source_refs
            if table_state_evidence_refs:
                record["table_state_evidence_refs"] = table_state_evidence_refs
            if derived_entity_evidence_refs:
                record["derivation_type"] = derivation_type
                record["derived_entity_evidence_refs"] = derived_entity_evidence_refs
            # IndicatorState 在此阶段和其他业务实体一样只校验自身。它属于哪个
            # LabIndicator 是 HAS_STATE 图边的语义，应由后续关系阶段统一抽取和校验。
            accepted.append(record)
        except GraphBuilderConfigurationError as error:
            # 失败时先判断这条模型输出是否仍含最小身份信息；有则可交给 Judge，
            # 没有则只能记录 REJECTED，不能保留不完整的自由文本。
            draft = _node_judge_draft(node)
            if draft is None or str(error) in {"duplicate_candidate"}:
                holds.append(_hold("entity", index, str(error), summary))
            else:
                judge_drafts.append(_judge_draft("entity", index, str(error), draft))
                holds.append(_review_item(
                    "entity", index, "REVIEW_REQUIRED", str(error), summary,
                ))

    # 第二轮：规则节点与业务实体不同，规则身份由结构化端点与证据位置共同决定。
    for (
        index,
        rule_stage,
        rule_logic,
        rule_inputs,
        rule_outputs,
        rule_excluded_outputs,
        expression,
        rule_name,
        evidence_refs,
        pending_rule_warnings,
        raw_rule_stage,
        raw_rule_logic,
    ) in pending_rules:
        candidate_key = _rule_candidate_key(
            chunk=chunk,
            rule_stage=rule_stage,
            rule_logic=rule_logic,
            rule_expression=expression,
            rule_inputs=rule_inputs,
            rule_outputs=rule_outputs,
            rule_excluded_outputs=rule_excluded_outputs,
            rule_evidence_refs=evidence_refs,
        )
        if candidate_key in seen_keys:
            # 同一规则表达式和同一组证据位置重复出现，只保留首次候选。
            holds.append(_hold(
                "entity", index, "duplicate_rule_identity",
                {
                    "rule_candidate_key": candidate_key,
                    "rule_inputs": list(rule_inputs),
                    "rule_outputs": list(rule_outputs),
                    "rule_excluded_outputs": list(rule_excluded_outputs),
                },
            ))
            continue
        seen_keys.add(candidate_key)
        record: dict[str, Any] = {
            "candidate_key": candidate_key,
            "rule_candidate_key": candidate_key,
            "entity_type": "RuleDefinition",
            "rule_inputs": list(rule_inputs),
            "rule_outputs": list(rule_outputs),
            "rule_excluded_outputs": list(rule_excluded_outputs),
            "rule_name": rule_name,
            "rule_stage_candidate": rule_stage,
            "rule_logic_candidate": rule_logic,
            "rule_evidence_refs": evidence_refs,
            "extraction_status": "VALID",
            "review_status": "PENDING",
            "publication_status": "HOLD",
            "_model_node_index": index,
        }
        if expression:
            record["rule_expression"] = expression
        if rule_stage == "UNKNOWN" or rule_logic == "UNKNOWN" or pending_rule_warnings:
            # 规则文本和证据仍可回放，所以不拒绝；只是禁止它传播到后续冻结目录。
            status_warnings = tuple(sorted(set(
                (
                    *pending_rule_warnings,
                    "RULE_STAGE_UNKNOWN" if rule_stage == "UNKNOWN" else "",
                    "RULE_LOGIC_UNKNOWN" if rule_logic == "UNKNOWN" else "",
                )
            )))
            status_warnings = tuple(warning for warning in status_warnings if warning)
            _mark_partial(record, *status_warnings)
            holds.append(_review_item(
                "rule",
                index,
                "REVIEW_REQUIRED",
                (
                    "rule_stage_unknown" if raw_rule_stage == "UNKNOWN"
                    else "rule_stage_candidate_invalid" if rule_stage == "UNKNOWN"
                    else "rule_logic_unknown" if raw_rule_logic == "UNKNOWN"
                    else "rule_logic_candidate_invalid"
                ),
                {
                    "rule_candidate_key": candidate_key,
                    "rule_inputs": list(rule_inputs),
                    "rule_outputs": list(rule_outputs),
                    "rule_stage_candidate": raw_rule_stage,
                    "rule_logic_candidate": raw_rule_logic,
                },
                warnings=status_warnings,
            ))
        accepted.append(record)

    # 最后处理阶段错位：实体阶段原则上只允许节点。关系不能偷偷带入关系阶段，
    # 但具有类型和两个端点的最小草稿可由未来 Judge 决定是否应重新路由。
    for index, relationship in enumerate(graph.relationships):
        draft = _relationship_judge_draft(relationship)
        if draft is None:
            holds.append(_hold(
                "entity", index, "entity_phase_relationship_not_allowed", _relationship_summary(relationship)
            ))
        else:
            judge_drafts.append(_judge_draft("entity", index, "entity_phase_relationship_not_allowed", draft))
            holds.append(_review_item(
                "entity", index, "REVIEW_REQUIRED", "entity_phase_relationship_not_allowed",
                _relationship_summary(relationship),
            ))
    return CandidateNormalization(accepted=accepted, review_items=holds, judge_drafts=judge_drafts)


def _catalog_for_prompt(nodes: Sequence[Mapping[str, Any]]) -> str:
    """把 ``VALID`` 节点裁剪为后续阶段唯一可引用的冻结目录 JSON。

    后续规则和关系模型只能引用这里的 ``candidate_key``，不能重新造实体或引用
    ``PARTIAL`` 节点。目录刻意不带 ``source_ref``：后续阶段的关系、规则必须提供
    自己的原文证据，而不是借用实体首次出现的位置。表格派生状态仅公开一个布尔
    标记，提示 HAS_STATE 可复用该状态已验证的表格双锚点，不泄露完整坐标。
    """
    import json

    catalog = []
    for item in nodes:
        # PARTIAL 仍供审计与 Judge 使用，但不能成为后续模型阶段的冻结端点。
        if item.get("extraction_status") != "VALID":
            continue
        # 目录只给模型解析端点所需的最小字段，避免把本地审查状态或全部原文重复塞回提示词。
        entry = {"candidate_key": item["candidate_key"], "entity_type": item["entity_type"]}
        if item["entity_type"] == "RuleDefinition":
            # 规则结构以输入输出数组为准；名称和旧表达式仅用于展示。
            entry.update({
                "rule_inputs": item["rule_inputs"],
                "rule_outputs": item["rule_outputs"],
                "rule_excluded_outputs": item.get("rule_excluded_outputs", []),
                "rule_logic_candidate": item["rule_logic_candidate"],
                "rule_evidence_roles": [ref["role"] for ref in item["rule_evidence_refs"]],
            })
            if item.get("rule_name"):
                entry["rule_name"] = item["rule_name"]
        else:
            # 关系阶段需用业务实体名称与 candidate_key 对齐端点。
            entry.update({
                "mention": item["mention"],
                "canonical_name_candidate": item["canonical_name_candidate"],
                **({"has_table_state_evidence": True} if item.get("table_state_evidence_refs") else {}),
                **({"has_derived_entity_evidence": True} if item.get("derived_entity_evidence_refs") else {}),
            })
            source_ref = item.get("source_ref")
            if isinstance(source_ref, Mapping) and isinstance(source_ref.get("chunk_id"), str):
                entry["source_chunk_id"] = source_ref["chunk_id"]
            canonical_id = item.get("canonical_id")
            if isinstance(canonical_id, str) and canonical_id:
                entry["canonical_id"] = canonical_id
            source_refs = item.get("source_refs")
            if isinstance(source_refs, list):
                surface_mentions = sorted({
                    str(ref["exact_quote"])
                    for ref in source_refs
                    if isinstance(ref, Mapping)
                    and isinstance(ref.get("exact_quote"), str)
                    and ref.get("exact_quote")
                })
                if surface_mentions:
                    entry["surface_mentions"] = surface_mentions
        catalog.append(entry)
    return json.dumps({"frozen_candidate_catalog": catalog}, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    # 此处演示“模型已提取实体后”的二次处理：不调用模型，也不写入运行工件。
    # 可自行在末尾添加 print(graph.nodes)、print(result) 或 print(frozen_catalog_json) 查看中间结果。
    import hashlib

    from medical_kg_sourceprep.extraction.graph_builder.contract import (
        BUSINESS_NODE_TYPES,
        DEFAULT_SCHEMA_PATH,
    )
    from medical_kg_sourceprep.extraction.graph_builder.schema import load_candidate_graph_schema
    from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk

    text = "肝硬化使转铁蛋白减少。"
    chunk = EvidenceChunk("demo:nodes", text, hashlib.sha256(text.encode()).hexdigest())
    schema = load_candidate_graph_schema(DEFAULT_SCHEMA_PATH)

    graph = Neo4jGraph(nodes=[
        Neo4jNode(
            id="indicator",
            label="LabIndicator",
            properties={"mention": "转铁蛋白", "extraction_reason": "原文明示的检验指标。"},
        ),
        Neo4jNode(
            id="state",
            label="IndicatorState",
            properties={"mention": "转铁蛋白减少", "extraction_reason": "原文明示的指标状态。"},
        ),
        Neo4jNode(
            id="duplicate",
            label="LabIndicator",
            properties={"mention": "转铁蛋白", "extraction_reason": "重复模型输出。"},
        ),
    ])

    # 本地校验只负责分流节点，还没有调用关系抽取模型。
    # normalization.accepted 是本地接纳的候选节点，其中可能同时包含 VALID 和 PARTIAL。
    normalization = normalize_candidate_nodes(
        graph,
        chunk=chunk,
        schema=schema,
        allowed_node_types=BUSINESS_NODE_TYPES,
        derive_entity_provenance=True,
    )
    accepted_nodes = normalization.accepted
    # print(accepted_nodes)

    # 这一步才把已接纳节点转换为后续模型的输入。函数会自动排除 PARTIAL，
    # 最终 JSON 同时供“规则抽取”和“普通关系抽取”引用，并不执行任何抽取。
    downstream_prompt_catalog_json = _catalog_for_prompt(accepted_nodes)
    print(downstream_prompt_catalog_json)
