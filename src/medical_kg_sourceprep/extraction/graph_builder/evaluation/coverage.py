"""使用独立大模型审查候选图的抽取遗漏，并回放建议证据。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...llm_extraction import EvidenceChunk, atomic_write_json
from ..contract import GraphBuilderConfigurationError


async def audit_extraction_coverage(
    client: Any,
    *,
    chunk: EvidenceChunk,
    schema: Mapping[str, Any],
    graph_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """独立查找候选图遗漏，不读取人工金标，并复验所有建议证据。"""
    graph_bytes = graph_path.read_bytes()
    # 英文提示词的中文语义：
    # 1. 只比较原文、当前 Schema 和已抽取候选图，查找明确出现但遗漏的内容；
    # 2. 不校验已有候选，不使用外部医学知识，也不依据 OCR 猜测原文；
    # 3. 全称和缩写若都逐字出现，应作为不同 mention 检查；
    # 4. 公式必须保留原文明示的输入和输出；
    # 5. 只返回 missing_items，最多 20 条高置信建议，并提供可逐字回放的证据；
    # 6. 不返回字符位置，位置统一由下面的确定性代码计算。
    prompt = (
        "You are auditing extraction coverage, not validating existing candidates. Use only SOURCE_DATA. "
        "Compare the original text with EXTRACTED_GRAPH and report only schema-compatible entities, "
        "relationships, or rules that are explicitly stated in the source but absent from the graph. "
        "Treat an abbreviation and its expanded name as distinct mentions when both occur verbatim. "
        "For formulas, preserve every explicit input term and output mention; do not use outside medical "
        "knowledge to repair OCR. Do not repeat existing candidates. Return exactly one JSON object with "
        "only missing_items. Each item must have kind (node|relationship|rule), reason, candidate, and "
        "evidence_spans [{chunk_id,exact_quote}]. exact_quote must be a non-empty verbatim substring of "
        "the source and, for a node, must contain the candidate mention. Do not calculate character offsets. "
        "Return at most 20 high-confidence missing items. Keep reason under 120 characters and candidate "
        "limited to fields required by the current Schema; do not copy Schema definitions into the output. "
        "Return an empty array when nothing is missing.\n"
        "SOURCE_DATA:\n" + json.dumps({
            "chunk": {"chunk_id": chunk.chunk_id, "text": chunk.text},
            "schema": schema,
            "extracted_graph": json.loads(graph_bytes),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    items: list[Any] | None = None
    last_error: Exception | None = None
    # 模型偶尔会返回截断 JSON。这里最多调用两次，但不会尝试修补不完整 JSON，
    # 防止本地代码擅自补全模型原本没有给出的医学语义。
    for _attempt in range(2):
        response = await client.llm.ainvoke(prompt)
        try:
            value = json.loads(response.content)
            candidate_items = value.get("missing_items") if isinstance(value, dict) else None
            if not isinstance(candidate_items, list):
                raise GraphBuilderConfigurationError("coverage_response_invalid")
            items = candidate_items
            break
        except (AttributeError, TypeError, json.JSONDecodeError, GraphBuilderConfigurationError) as error:
            last_error = error
    if items is None:
        raise GraphBuilderConfigurationError("coverage_response_json_invalid") from last_error

    validated: list[dict[str, Any]] = []
    rejected = 0
    for item in items:
        if not isinstance(item, dict) or item.get("kind") not in {"node", "relationship", "rule"}:
            rejected += 1
            continue
        candidate = item.get("candidate")
        if item["kind"] == "node":
            # 节点证据以 mention 本身为最小边界；找不到逐字 mention 就拒绝该建议。
            mention = candidate.get("mention") if isinstance(candidate, dict) else None
            if not isinstance(mention, str) or not mention:
                rejected += 1
                continue
            start = chunk.text.find(mention)
            if start < 0:
                rejected += 1
                continue
            validated.append({
                **item,
                "evidence_spans": [{
                    "chunk_id": chunk.chunk_id,
                    "start": start,
                    "end": start + len(mention),
                    "exact_quote": mention,
                }],
            })
            continue

        # 关系和规则需要模型给出联合语义证据；每段证据都必须来自当前 chunk。
        spans = item.get("evidence_spans")
        if not isinstance(spans, list) or not spans:
            rejected += 1
            continue
        replayed: list[dict[str, Any]] = []
        for span in spans:
            if not isinstance(span, dict) or span.get("chunk_id") != chunk.chunk_id:
                replayed = []
                break
            exact_quote = span.get("exact_quote")
            if not isinstance(exact_quote, str) or not exact_quote:
                replayed = []
                break
            start = chunk.text.find(exact_quote)
            if start < 0:
                replayed = []
                break
            replayed.append({
                "chunk_id": chunk.chunk_id,
                "start": start,
                "end": start + len(exact_quote),
                "exact_quote": exact_quote,
            })
        if not replayed:
            rejected += 1
            continue
        validated.append({**item, "evidence_spans": replayed})

    # 遗漏审查只是下一轮抽取的建议，不会直接写入候选图，更不能批准发布。
    document = {
        "schema_version": "candidate-graph-coverage-audit/v0.1",
        "status": "candidate-only",
        "publication_status": "HOLD",
        "approved": 0,
        "gold_answers_exposed": False,
        "input_graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
        "rejected_invalid_items": rejected,
        "missing_items": validated,
    }
    atomic_write_json(output_path, document)
    return document
