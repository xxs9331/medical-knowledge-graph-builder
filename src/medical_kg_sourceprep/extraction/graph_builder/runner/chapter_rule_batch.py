"""使用百炼 Batch File 离线抽取并评测第一章图规则。"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[4]))
    __package__ = "medical_kg_sourceprep.extraction.graph_builder.runner"

import argparse
import asyncio
import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from neo4j_graphrag.generation.prompts import PromptTemplate

from ...llm_extraction import atomic_write_json, load_chunk_manifest
from ..client import DeepSeekGraphBuilderClient
from ..contract import (
    DEFAULT_CHUNK_MANIFEST,
    DEFAULT_SCHEMA_PATH,
    PROJECT_ROOT,
    RULE_NODE_PROMPT_TEMPLATE,
    RULE_NODE_PROMPT_VERSION,
    GraphBuilderConfigurationError,
)
from ..evaluation.scoring import merge_candidate_graphs, score_candidate_graph
from ..rule_gate import partition_invalid_rules
from ..schema import build_graphrag_schema, load_candidate_graph_schema
from ..validation import _catalog_for_prompt
from .chapter_candidate_graph import DEFAULT_CANONICAL_PATH, load_frozen_catalog
from .chapter_rule_nodes import build_canonical_endpoints_by_chunk, _canonical_rule_graph
from .rule_semantic_gate_evaluation import _extract_rule_nodes, _rule_metrics, _string_list


DEFAULT_ROOT = PROJECT_ROOT / "runtime/experiments/chapter01-qwen-rule-batch-v0.1"
DEFAULT_GOLD = PROJECT_ROOT / "evaluation/chapter-01/chapter-01-rule-test-set-v0.3.json"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-flash"


def _client():
    """创建无代理百炼客户端；Key 只从当前进程环境读取。"""
    from openai import OpenAI

    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise GraphBuilderConfigurationError("DASHSCOPE_API_KEY is required")
    return OpenAI(api_key=key, base_url=BASE_URL, timeout=60, max_retries=0)


def prepare(output_root: Path) -> dict[str, Any]:
    """生成不含金标的逐 chunk Batch JSONL。"""
    manifest, chunks = load_chunk_manifest(DEFAULT_CHUNK_MANIFEST)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    canonical, mentions, entities = load_frozen_catalog(DEFAULT_CANONICAL_PATH)
    endpoints_by_chunk = build_canonical_endpoints_by_chunk(
        canonical=canonical,
        mentions=mentions,
        entities=entities,
        chunks_by_id=chunks_by_id,
    )
    schema = load_candidate_graph_schema(DEFAULT_SCHEMA_PATH)
    graph_schema = build_graphrag_schema(
        schema, relation_types=(), node_types=("RuleDefinition",)
    )
    template = PromptTemplate(RULE_NODE_PROMPT_TEMPLATE, expected_inputs=[])
    output_root.mkdir(parents=True, exist_ok=True)
    input_path = output_root / "batch-input.jsonl"
    records: list[dict[str, Any]] = []
    with input_path.open("w", encoding="utf-8") as stream:
        for index, chunk in enumerate(chunks):
            custom_id = f"chapter01-rule-{index:04d}"
            prompt = template.format(
                text=chunk.text,
                schema=graph_schema.model_dump(exclude_none=True),
                examples=_catalog_for_prompt(endpoints_by_chunk.get(chunk.chunk_id, [])),
            )
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "enable_thinking": False,
                    "temperature": 0,
                },
            }
            stream.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            records.append({"custom_id": custom_id, "chunk_id": chunk.chunk_id})
    report = {
        "schema_version": "chapter-rule-qwen-batch-plan/v0.1",
        "status": "prepared",
        "publication_status": "HOLD",
        "gold_exposed_to_model": False,
        "model": MODEL,
        "prompt_version": RULE_NODE_PROMPT_VERSION,
        "chapter_id": manifest.get("chapter_id"),
        "request_count": len(records),
        "input_file": str(input_path),
        "records": records,
    }
    atomic_write_json(output_root / "batch-plan.json", report)
    return report


def submit(output_root: Path) -> dict[str, Any]:
    """上传 JSONL 并创建 24 小时离线批任务。"""
    plan = json.loads((output_root / "batch-plan.json").read_text(encoding="utf-8"))
    client = _client()
    with Path(plan["input_file"]).open("rb") as stream:
        uploaded = client.files.create(file=stream, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"experiment": "chapter01-qwen-rule-batch-v0.1"},
    )
    report = {
        "schema_version": "chapter-rule-qwen-batch-job/v0.1",
        "status": batch.status,
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
    }
    atomic_write_json(output_root / "batch-job.json", report)
    return report


def status(output_root: Path) -> dict[str, Any]:
    """查询任务状态并更新不含凭据的本地记录。"""
    job = json.loads((output_root / "batch-job.json").read_text(encoding="utf-8"))
    batch = _client().batches.retrieve(job["batch_id"])
    report = {
        **job,
        "status": batch.status,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "request_counts": (
            batch.request_counts.model_dump() if batch.request_counts is not None else None
        ),
    }
    atomic_write_json(output_root / "batch-job.json", report)
    return report


class _ReplayLLM:
    def __init__(self, content: str, usage: Mapping[str, Any]) -> None:
        self.content = content
        self.usage = usage

    async def ainvoke(self, _prompt: str, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            content=self.content,
            usage=SimpleNamespace(
                request_tokens=int(self.usage.get("prompt_tokens", 0)),
                response_tokens=int(self.usage.get("completion_tokens", 0)),
                total_tokens=int(self.usage.get("total_tokens", 0)),
            ),
        )

    async def aclose(self) -> None:
        return None


async def collect(output_root: Path) -> dict[str, Any]:
    """下载模型输出，经既有门禁生成候选规则并本地计算 PRF1。"""
    job = status(output_root)
    if job["status"] != "completed" or not job.get("output_file_id"):
        raise GraphBuilderConfigurationError(f"batch_not_completed:{job['status']}")
    content = _client().files.content(job["output_file_id"]).text
    output_path = output_root / "batch-output.jsonl"
    output_path.write_text(content, encoding="utf-8")
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    row_by_id = {str(row["custom_id"]): row for row in rows}
    plan = json.loads((output_root / "batch-plan.json").read_text(encoding="utf-8"))
    _manifest, chunks = load_chunk_manifest(DEFAULT_CHUNK_MANIFEST)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    canonical, mentions, entities = load_frozen_catalog(DEFAULT_CANONICAL_PATH)
    endpoints_by_chunk = build_canonical_endpoints_by_chunk(
        canonical=canonical, mentions=mentions, entities=entities, chunks_by_id=chunks_by_id
    )
    schema = load_candidate_graph_schema(DEFAULT_SCHEMA_PATH)
    graphs: dict[str, dict[str, list[Any]]] = {}
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for record in plan["records"]:
        row = row_by_id[record["custom_id"]]
        body = row["response"]["body"]
        raw = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        replay = DeepSeekGraphBuilderClient(llm=_ReplayLLM(raw, usage), http_client=SimpleNamespace())
        chunk = chunks_by_id[record["chunk_id"]]
        endpoints = endpoints_by_chunk.get(chunk.chunk_id, [])
        extracted, _model_record = await _extract_rule_nodes(
            replay, chunk=chunk, schema=schema, business_nodes=endpoints
        )
        gated, _rejections = partition_invalid_rules(extracted, strict_graph_shapes=True)
        rules, edges, _reviews = _canonical_rule_graph(
            schema=schema, business_nodes=endpoints, rule_nodes=gated
        )
        graphs[chunk.chunk_id] = {"nodes": [*endpoints, *rules], "relationships": edges}
        for target, source in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            total_usage[target] += int(usage.get(source, 0))
    gold = json.loads(DEFAULT_GOLD.read_text(encoding="utf-8"))
    case_results: list[dict[str, Any]] = []
    for case in gold["cases"]:
        chunk_ids = _string_list(case.get("chunk_ids", []))
        source_text = "\n\n".join(chunks_by_id[chunk_id].text for chunk_id in chunk_ids)
        score = score_candidate_graph(
            merge_candidate_graphs(graphs[chunk_id] for chunk_id in chunk_ids),
            case,
            source_text=source_text,
        )["rules"]
        case_results.append({"case_id": case["case_id"], "rules": score})
    report = {
        "schema_version": "chapter-rule-qwen-batch-evaluation/v0.1",
        "status": "evaluation-only",
        "publication_status": "HOLD",
        "gold_exposed_to_model": False,
        "model": MODEL,
        "usage": total_usage,
        "metrics": _rule_metrics(case_results, "rules"),
        "cases": case_results,
    }
    atomic_write_json(output_root / "evaluation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第一章 qwen-flash 离线规则抽取")
    parser.add_argument("action", choices=("prepare", "submit", "status", "collect"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.output_root)
    elif args.action == "submit":
        result = submit(args.output_root)
    elif args.action == "status":
        result = status(args.output_root)
    else:
        result = asyncio.run(collect(args.output_root))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
