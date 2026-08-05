"""Structured report -> book evidence -> DeepSeek -> Markdown pipeline.

This is an expression layer over book evidence.  It does not create approved
rules or turn model output into a diagnosis, treatment, or medication plan.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence
from urllib import error as urlerror
from urllib import request

from .desktop_app import DesktopAppError, parse_report_payload
from .graph_retrieval import (
    GraphRetrievalError,
    graph_query_diagnostic,
    graph_reasoning_paths,
)
from .qa import ProvenanceContext, QaError, query_index, query_index_with_graph
from .report_model import AbnormalFlag, EvaluationResult, Observation, evaluate_observation


ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
TIMEOUT_SECONDS = 60
MAX_RETRIES = 2
MAX_EVIDENCE_PER_METRIC = 3

VALIDATION_ISSUE_LABELS = {
    "missing_unit": "缺少单位",
    "invalid_interval": "缺少有效参考区间",
    "reversed_interval": "参考区间上下限异常",
    "invalid_value": "检验结果或参考值无法解析",
    "non_finite_value": "检验结果或参考值不是有限数值",
    "incompatible_unit": "单位无法安全换算",
    "report_flag_conflict": "报告升降标记与参考区间计算不一致",
}


class ReportPipelineError(ValueError):
    """Raised when the pipeline cannot produce a trustworthy report."""


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    chunk_id: str
    text: str
    printed_page_number: int
    source_pdf_page_number: int
    chunk_sha256: str
    score: float
    retrieval_reason: str
    graph: Mapping[str, Any] | None = None
    location: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "exact_quote": self.text,
            "printed_page_number": self.printed_page_number,
            "source_pdf_page_number": self.source_pdf_page_number,
            "chunk_sha256": self.chunk_sha256,
            "score": self.score,
            "retrieval_reason": self.retrieval_reason,
            "location_status": "indexed",
        }
        if self.location:
            value.update(self.location)
        if self.graph:
            value["graph"] = dict(self.graph)
        return value


@dataclass(frozen=True, slots=True)
class MetricInput:
    metric_id: str
    observation: Observation
    evaluation: EvaluationResult
    evidence: tuple[Evidence, ...]
    graph_diagnostics: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ReportDocument:
    markdown: str
    report: Mapping[str, Any]
    metrics: tuple[MetricInput, ...]
    channels: Mapping[str, Any]
    reasoning_paths: tuple[Mapping[str, Any], ...] = ()
    reasoning_rejections: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        evidence_by_id = {
            evidence.evidence_id: evidence
            for metric in self.metrics
            for evidence in metric.evidence
        }
        return {
            "markdown": self.markdown,
            "report": dict(self.report),
            "metrics": [
                {
                    "metric_id": item.metric_id,
                    "raw_name": item.observation.raw_name,
                    "value": str(item.evaluation.normalized.value)
                    if item.evaluation.normalized.value is not None
                    else None,
                    "unit": item.evaluation.normalized.unit,
                    "computed_flag": item.evaluation.evidence.computed_flag.value
                    if item.evaluation.evidence.computed_flag
                    else None,
                    "validation_issues": [
                        {
                            **error.to_dict(),
                            "label": VALIDATION_ISSUE_LABELS.get(error.code, "数据格式需要核对"),
                        }
                        for error in item.evaluation.evidence.errors
                    ],
                    "evidence_ids": [evidence.evidence_id for evidence in item.evidence],
                }
                for item in self.metrics
            ],
            "evidence": [evidence.to_dict() for evidence in evidence_by_id.values()],
            "channels": dict(self.channels),
            "reasoning_paths": [dict(item) for item in self.reasoning_paths],
            "reasoning_rejections": [dict(item) for item in self.reasoning_rejections],
        }


class JsonTransport(Protocol):
    def post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class DeepSeekTransport:
    """Official endpoint transport with an explicit no-proxy opener."""

    def __init__(self, api_key: str, *, opener: Any | None = None) -> None:
        if not api_key.strip():
            raise ReportPipelineError("DEEPSEEK_API_KEY is required before model call")
        self._api_key = api_key
        self._opener = opener or _direct_opener(os.environ.get("DEEPSEEK_API_IP"))

    def post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        req = request.Request(
            ENDPOINT,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "medical-kg-sourceprep/report-pipeline-v0.1",
            },
        )
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                with self._opener.open(req, timeout=TIMEOUT_SECONDS) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                if not isinstance(decoded, Mapping):
                    raise ReportPipelineError("DeepSeek response must be a JSON object")
                return decoded
            except urlerror.HTTPError as error:
                last_error = error
                if error.code != 429 and not 500 <= error.code <= 599:
                    raise ReportPipelineError(f"DeepSeek request failed with HTTP {error.code}") from error
            except (urlerror.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
                raise ReportPipelineError("DeepSeek request failed") from error
            if attempt < MAX_RETRIES:
                time.sleep(1 + attempt)
        raise ReportPipelineError("DeepSeek request retries exhausted") from last_error


def _direct_opener(ip: str | None) -> Any:
    """Disable environment proxies and optionally pin the API address."""
    no_proxy = request.ProxyHandler({})
    if not ip:
        return request.build_opener(no_proxy)

    class PinnedConnection(http.client.HTTPSConnection):
        def connect(self) -> None:
            self.sock = socket.create_connection((ip, self.port), self.timeout)
            self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)

    class PinnedHandler(request.HTTPSHandler):
        def https_open(self, req: request.Request) -> Any:
            return self.do_open(PinnedConnection, req, context=self._context)

    return request.build_opener(no_proxy, PinnedHandler(context=ssl.create_default_context()))


def _metric_id(index: int, observation: Observation) -> str:
    return observation.standard_name or observation.abbreviation or observation.raw_name or f"metric-{index}"


def _retrieval_queries(
    observation: Observation, flag: AbnormalFlag
) -> tuple[tuple[str, str], ...]:
    state = "升高" if flag is AbnormalFlag.HIGH else "降低"
    candidates = (
        ("abbreviation", observation.abbreviation),
        ("standard_name", observation.standard_name),
        ("raw_name", observation.raw_name),
    )
    queries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for channel, name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        queries.append((channel, f"{name} {state}"))
    return tuple(queries)


def _evidence(row: Mapping[str, Any]) -> Evidence:
    chunk_id = row.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ReportPipelineError("retrieval result has no chunk_id")
    graph = row.get("graph")
    if graph is not None and not isinstance(graph, Mapping):
        raise ReportPipelineError("retrieval graph metadata is invalid")
    location = {
        key: row[key]
        for key in (
            "page_id", "chapter_page_index", "cleaned_char_start", "cleaned_char_end",
            "markdown_line_start", "markdown_line_end", "source_page_line_start",
            "source_page_line_end", "location_status",
        )
        if key in row
    }
    return Evidence(
        evidence_id=f"E{chunk_id}",
        chunk_id=chunk_id,
        text=str(row.get("text", "")),
        printed_page_number=int(row["printed_page_number"]),
        source_pdf_page_number=int(row["source_pdf_page_number"]),
        chunk_sha256=str(row["chunk_sha256"]),
        score=float(row.get("score", 0.0)),
        retrieval_reason=str(row.get("retrieval_reason", "")),
        graph=dict(graph) if graph else None,
        location=location,
    )


def collect_metrics(
    report: Mapping[str, Any],
    index: Path,
    *,
    knowledge_graph: Path | None = None,
    provenance: ProvenanceContext | None = None,
) -> tuple[MetricInput, ...]:
    try:
        parsed = parse_report_payload(report)
    except DesktopAppError as error:
        raise ReportPipelineError(str(error)) from error
    metrics: list[MetricInput] = []
    for index_number, (metric_id, observation) in enumerate(parsed.items()):
        evaluation = evaluate_observation(observation)
        flag = evaluation.evidence.computed_flag
        rows: list[Mapping[str, Any]] = []
        graph_diagnostics: list[Mapping[str, Any]] = []
        if flag in (AbnormalFlag.HIGH, AbnormalFlag.LOW):
            try:
                channels = []
                for channel, query in _retrieval_queries(observation, flag):
                    if knowledge_graph is None:
                        items = query_index(index, query, top_k=10, provenance=provenance)
                    else:
                        graph_term = query.rsplit(" ", 1)[0]
                        graph_diagnostics.append(graph_query_diagnostic(knowledge_graph, graph_term))
                        items, _ = query_index_with_graph(
                            index,
                            knowledge_graph,
                            query,
                            top_k=10,
                            provenance=provenance,
                            graph_query=graph_term,
                        )
                    channels.append((channel, items))
            except (QaError, GraphRetrievalError) as error:
                raise ReportPipelineError(str(error)) from error
            for rank in range(max((len(items) for _, items in channels), default=0)):
                for channel, items in channels:
                    if rank >= len(items):
                        continue
                    row = dict(items[rank])
                    row["retrieval_reason"] = (
                        f"{channel}:" + str(row.get("retrieval_reason", "term_match"))
                    )
                    rows.append(row)
        deduplicated: dict[str, Evidence] = {}
        for row in rows:
            item = _evidence(row)
            existing = deduplicated.get(item.chunk_id)
            if existing is None or (existing.graph is None and item.graph is not None):
                deduplicated[item.chunk_id] = item
            if len(deduplicated) == MAX_EVIDENCE_PER_METRIC:
                break
        diagnostics_by_query = {str(item.get("query")): item for item in graph_diagnostics}
        metrics.append(MetricInput(
            metric_id or _metric_id(index_number, observation), observation, evaluation,
            tuple(deduplicated.values()), tuple(diagnostics_by_query.values()),
        ))
    return tuple(metrics)


def _safe_metric(metric: MetricInput) -> dict[str, Any]:
    normalized = metric.evaluation.normalized
    computation = metric.evaluation.evidence
    return {
        "metric_id": metric.metric_id,
        "raw_name": normalized.raw_name,
        "standard_name": normalized.standard_name,
        "abbreviation": normalized.abbreviation,
        "value": str(normalized.value) if normalized.value is not None else None,
        "unit": normalized.unit,
        "reference_interval": {"lower": str(normalized.lower) if normalized.lower is not None else None, "upper": str(normalized.upper) if normalized.upper is not None else None},
        "computed_flag": computation.computed_flag.value if computation.computed_flag else None,
        "errors": [error.to_dict() for error in computation.errors],
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "chunk_id": item.chunk_id,
                "text": item.text,
                "retrieval_reason": item.retrieval_reason,
                **({"graph": dict(item.graph)} if item.graph else {}),
            }
            for item in metric.evidence
        ],
    }


def _retrieval_channels(
    metrics: Sequence[MetricInput], knowledge_graph: Path | None
) -> dict[str, Any]:
    evidence = {
        item.evidence_id: item
        for metric in metrics
        for item in metric.evidence
    }
    graph_evidence = [item for item in evidence.values() if item.graph]
    lexical_evidence = [
        item
        for item in evidence.values()
        if item.retrieval_reason.rsplit(":", 1)[-1] != "graph_path"
    ]
    graph_enabled = knowledge_graph is not None
    diagnostics = [item for metric in metrics for item in metric.graph_diagnostics]
    diagnostic_counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        status = str(diagnostic.get("status", "unknown"))
        diagnostic_counts[status] = diagnostic_counts.get(status, 0) + 1
    return {
        "mode": "lexical+knowledge_graph" if graph_enabled else "lexical",
        "lexical": {
            "enabled": True,
            "coverage": "full-book",
            "evidence_count": len(lexical_evidence),
        },
        "graph": {
            "enabled": graph_enabled,
            "coverage": "chapter-01-only" if graph_enabled else None,
            "status": "candidate-only" if graph_enabled else None,
            "evidence_count": len(graph_evidence),
            "query_diagnostics": diagnostics,
            "query_diagnostic_counts": dict(sorted(diagnostic_counts.items())),
        },
    }


def _reasoning_context(
    metrics: Sequence[MetricInput], knowledge_graph: Path | None, evidence_index: Path
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    if knowledge_graph is None:
        return (), ()
    observations = []
    for metric in metrics:
        flag = metric.evaluation.evidence.computed_flag
        if flag not in (AbnormalFlag.HIGH, AbnormalFlag.LOW):
            continue
        normalized = metric.evaluation.normalized
        terms = [
            value for value in (
                normalized.raw_name, normalized.standard_name, normalized.abbreviation,
            ) if value
        ]
        observations.append({
            "metric_id": metric.metric_id,
            "terms": terms,
            "computed_flag": flag.value,
        })
    try:
        result = graph_reasoning_paths(knowledge_graph, evidence_index, observations)
    except GraphRetrievalError as error:
        raise ReportPipelineError(str(error)) from error
    evidence_ids_by_chunk = {
        item.chunk_id: item.evidence_id
        for metric in metrics
        for item in metric.evidence
    }
    accepted: list[Mapping[str, Any]] = []
    rejections = list(result.rejections)
    for path in result.paths:
        value = dict(path)
        value["evidence_ids"] = [
            evidence_ids_by_chunk[chunk_id]
            for chunk_id in value.get("chunk_ids", [])
            if chunk_id in evidence_ids_by_chunk
        ]
        if not value["evidence_ids"]:
            rejections.append({
                "rule_id": value.get("rule_id"),
                "rule_name": value.get("rule_name"),
                "reason": "evidence_not_in_report",
            })
            continue
        accepted.append(value)
    return tuple(accepted), tuple(rejections)


def _prompt(
    metrics: Sequence[MetricInput], reasoning_paths: Sequence[Mapping[str, Any]] = ()
) -> str:
    abnormal_metrics = [
        item
        for item in metrics
        if item.evaluation.evidence.computed_flag in (AbnormalFlag.HIGH, AbnormalFlag.LOW)
    ]
    abnormal_metric_ids = [item.metric_id for item in abnormal_metrics]
    flags = [item.evaluation.evidence.computed_flag for item in metrics]
    payload = {
        "abnormal_metric_ids": abnormal_metric_ids,
        "report_overview": {
            "total": len(metrics),
            "normal": sum(flag is AbnormalFlag.NORMAL for flag in flags),
            "high": sum(flag is AbnormalFlag.HIGH for flag in flags),
            "low": sum(flag is AbnormalFlag.LOW for flag in flags),
            "indeterminate": sum(flag is None for flag in flags),
        },
        "metrics": [_safe_metric(item) for item in abnormal_metrics],
        "candidate_reasoning_paths": [dict(item) for item in reasoning_paths],
    }
    return (
        "你是证据约束的医学报告表达器。只能使用 INPUT 中的程序判定和书内证据，禁止使用常识补充。"
        "evidence 中的 candidate-only graph 仅用于召回、排序和发现关联，不能单独支持医学结论；"
        "candidate_reasoning_paths 是未经批准且未执行的候选关联路径，只能帮助合并分析；"
        "不得把路径状态当成诊断结果，也不得引用路径中没有对应 evidence_ids 的内容。"
        "所有医学表述仍必须由同一 evidence 项的 text 原文支持并引用其 evidence_id。"
        "不得诊断、给出治疗/用药结论。每个医学解释的 evidence_ids 必须引用本次输入中的 evidence_id；"
        "相关异常指标可以交叉引用 INPUT 中其他异常指标下的证据；"
        "abnormal_analyses 必须且只能包含 abnormal_metric_ids 中的指标，每个指标恰好一次；"
        "computed_flag 为空或 normal 的指标不得进入 abnormal_analyses。"
        "summary 只能概括 report_overview 计数和 metrics 中的程序异常，不得推测未提供的指标。"
        "没有证据时只能写‘知识库证据不足，无法作出书内解释’，并使用空数组。返回 JSON，不要 Markdown。"
        "结构必须是 {summary:string, abnormal_analyses:[{metric_id:string, analysis:string, evidence_ids:string[]}], "
        "association_analysis:{analysis:string,evidence_ids:string[]}, attention_suggestions:[{text:string,evidence_ids:string[]}], "
        "insufficient_evidence:[string]}。建议只能是关注/复查提示，不得是治疗或用药；"
        "attention_suggestions 中每项必须有非空 evidence_ids，没有可引用证据就省略该建议。INPUT:\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _content(response: Mapping[str, Any]) -> str:
    try:
        value = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ReportPipelineError("DeepSeek response has no message content") from error
    if not isinstance(value, str) or not value.strip():
        raise ReportPipelineError("DeepSeek response content is empty")
    return value


def validate_model_result(value: object, evidence_ids: set[str], metric_ids: set[str], abnormal_metric_ids: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"summary", "abnormal_analyses", "association_analysis", "attention_suggestions", "insufficient_evidence"}:
        raise ReportPipelineError("DeepSeek JSON does not match the report schema")
    if not isinstance(value["summary"], str) or not value["summary"].strip() or not isinstance(value["abnormal_analyses"], list):
        raise ReportPipelineError("DeepSeek report summary or abnormal_analyses is invalid")
    used: set[str] = set()
    seen_metrics: set[str] = set()
    for item in value["abnormal_analyses"]:
        if not isinstance(item, dict) or set(item) != {"metric_id", "analysis", "evidence_ids"} or item["metric_id"] not in metric_ids or not isinstance(item["analysis"], str) or not isinstance(item["evidence_ids"], list):
            raise ReportPipelineError("DeepSeek abnormal analysis is invalid")
        if abnormal_metric_ids is not None and item["metric_id"] not in abnormal_metric_ids:
            raise ReportPipelineError("DeepSeek analyzed a non-abnormal metric")
        if item["metric_id"] in seen_metrics:
            raise ReportPipelineError("DeepSeek repeated an abnormal metric analysis")
        ids = item["evidence_ids"]
        if any(not isinstance(item_id, str) or item_id not in evidence_ids for item_id in ids):
            raise ReportPipelineError("DeepSeek returned an unknown evidence ID")
        if item["metric_id"] in (abnormal_metric_ids or set()) and not ids and "知识库证据不足" not in item["analysis"]:
            raise ReportPipelineError("DeepSeek medical analysis has no evidence citations")
        seen_metrics.add(item["metric_id"])
        used.update(ids)
    association = value["association_analysis"]
    if not isinstance(association, dict) or set(association) != {"analysis", "evidence_ids"} or not isinstance(association["analysis"], str) or not isinstance(association["evidence_ids"], list):
        raise ReportPipelineError("DeepSeek association analysis is invalid")
    if any(item_id not in evidence_ids for item_id in association["evidence_ids"]):
        raise ReportPipelineError("DeepSeek returned an unknown association evidence ID")
    if association["analysis"].strip() and not association["evidence_ids"] and "知识库证据不足" not in association["analysis"]:
        raise ReportPipelineError("DeepSeek association analysis has no evidence citations")
    for suggestion in value["attention_suggestions"]:
        if not isinstance(suggestion, dict) or set(suggestion) != {"text", "evidence_ids"} or not isinstance(suggestion["text"], str) or not isinstance(suggestion["evidence_ids"], list):
            raise ReportPipelineError("DeepSeek suggestion is invalid")
        if any(item_id not in evidence_ids for item_id in suggestion["evidence_ids"]):
            raise ReportPipelineError("DeepSeek returned an unknown suggestion evidence ID")
        if suggestion["text"].strip() and not suggestion["evidence_ids"]:
            raise ReportPipelineError("DeepSeek suggestion has no evidence citations")
    if not isinstance(value["insufficient_evidence"], list) or any(not isinstance(item, str) for item in value["insufficient_evidence"]):
        raise ReportPipelineError("DeepSeek insufficient_evidence is invalid")
    if abnormal_metric_ids is not None and abnormal_metric_ids != seen_metrics:
        raise ReportPipelineError("DeepSeek omitted an abnormal metric analysis")
    return value


def render_markdown(
    metrics: Sequence[MetricInput],
    result: Mapping[str, Any],
    channels: Mapping[str, Any] | None = None,
    reasoning_paths: Sequence[Mapping[str, Any]] = (),
) -> str:
    evidence = {item.evidence_id: item for metric in metrics for item in metric.evidence}
    evidence_numbers = {item_id: index for index, item_id in enumerate(evidence, 1)}

    def references(ids: Sequence[str]) -> str:
        links = []
        for item_id in ids:
            item = evidence[item_id]
            number = evidence_numbers[item_id]
            links.append(
                f"[证据 {number}：书内第 {item.printed_page_number} 页 / "
                f"PDF 第 {item.source_pdf_page_number} 页]"
                f"(/source.pdf#page={item.source_pdf_page_number})"
            )
        return "、".join(links)

    graph = channels.get("graph", {}) if channels else {}
    graph_enabled = bool(graph.get("enabled"))
    notice = "程序异常判定、整书检索证据"
    if graph_enabled:
        notice += "和第一章候选知识图谱辅助召回"
    lines = [
        "# 体检报告分析", "",
        f"> 这是基于{notice}生成的辅助性摘要，不构成诊断、治疗或用药建议。", "",
        "## 分析依据", "",
        "- 异常判定：程序按报告参考区间重算",
        "- 书内检索：全书证据索引",
        (
            f"- 知识图谱：第一章候选图谱（candidate-only），辅助召回证据 "
            f"{int(graph.get('evidence_count', 0))} 条"
            if graph_enabled else "- 知识图谱：未启用"
        ),
        "", "## 摘要", "", str(result["summary"]), "", "## 异常指标", "",
    ]
    analyses = {item["metric_id"]: item for item in result["abnormal_analyses"]}
    for metric in metrics:
        computation = metric.evaluation.evidence
        if computation.computed_flag not in (AbnormalFlag.HIGH, AbnormalFlag.LOW):
            continue
        item = analyses.get(metric.metric_id, {"analysis": "知识库证据不足，无法作出书内解释。", "evidence_ids": []})
        lines.append(f"### {metric.observation.raw_name}（{computation.computed_flag.value}）")
        lines.append("")
        lines.append(str(item["analysis"]))
        lines.append("")
        if item["evidence_ids"]:
            lines.append("证据：" + references(item["evidence_ids"]))
            lines.append("")
    lines += ["## 关联分析", "", str(result["association_analysis"]["analysis"]), ""]
    if reasoning_paths:
        lines += ["## 候选推理路径", ""]
        lines += [
            "> 以下路径只用于合并书内关联，未执行图谱规则，不构成诊断。", "",
        ]
        for path in reasoning_paths:
            metric_names = "、".join(str(item) for item in path.get("matched_metric_ids", []))
            lines.append(f"### {path.get('rule_name', '候选规则')}")
            lines.append("")
            lines.append(f"状态：`{path.get('status', 'candidate-only')}`；共同命中指标：{metric_names}。")
            lines.append("")
            for triple in path.get("triples", []):
                lines.append(
                    f"- {triple.get('subject_name')} -[{triple.get('predicate')}]-> {triple.get('object_name')}"
                )
            evidence_ids = path.get("evidence_ids", [])
            if evidence_ids:
                lines += ["", "路径证据：" + references(evidence_ids), ""]
    if result["attention_suggestions"]:
        lines += ["## 关注建议", ""]
        lines += [f"- {item['text']}（{references(item['evidence_ids'])}）" for item in result["attention_suggestions"]]
        lines.append("")
    if result["insufficient_evidence"]:
        lines += ["## 证据不足", ""] + [f"- {item}" for item in result["insufficient_evidence"]] + [""]
    metrics_to_check = [metric for metric in metrics if metric.evaluation.evidence.errors]
    if metrics_to_check:
        lines += ["## 数据待核对", ""]
        for metric in metrics_to_check:
            reasons = "；".join(
                VALIDATION_ISSUE_LABELS.get(error.code, "数据格式需要核对")
                for error in metric.evaluation.evidence.errors
            )
            lines.append(f"- {metric.observation.raw_name}：{reasons}")
        lines.append("")
    lines += ["## 书内证据", ""]
    for item_id, item in evidence.items():
        number = evidence_numbers[item_id]
        lines += [f"### 证据 {number}", "", f"[打开原 PDF 第 {item.source_pdf_page_number} 页](/source.pdf#page={item.source_pdf_page_number})", "", f"书内第 {item.printed_page_number} 页；PDF 第 {item.source_pdf_page_number} 页。", "", "> " + item.text.replace("\n", "\n> "), "", f"evidence_id: `{item_id}`；chunk_id: `{item.chunk_id}`；chunk_sha256: `{item.chunk_sha256}`。", ""]
        if item.graph:
            nodes = "、".join(item.graph.get("matched_node_names", [])) or "直接命中"
            path = " -> ".join(item.graph.get("path_relations", [])) or "直接命中"
            lines += [
                f"图谱辅助召回：`{item.graph.get('status', 'candidate-only')}`；"
                f"命中节点：{nodes}；图路径：{path}。",
                "",
            ]
    return "\n".join(lines).rstrip() + "\n"


def analyze_report_document(
    report: Mapping[str, Any],
    index: Path,
    *,
    knowledge_graph: Path | None = None,
    provenance: ProvenanceContext | None = None,
    transport: JsonTransport | None = None,
) -> ReportDocument:
    metrics = collect_metrics(
        report, index, knowledge_graph=knowledge_graph, provenance=provenance
    )
    channels = _retrieval_channels(metrics, knowledge_graph)
    reasoning_paths, reasoning_rejections = _reasoning_context(
        metrics, knowledge_graph, index
    )
    channels = {
        **channels,
        "graph": {
            **channels["graph"],
            "reasoning_path_count": len(reasoning_paths),
            "reasoning_rejection_count": len(reasoning_rejections),
        },
    }
    if transport is None:
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        transport = DeepSeekTransport(key)
    payload = {"model": MODEL, "temperature": 0, "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}, "messages": [{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": _prompt(metrics, reasoning_paths)}]}
    try:
        model_result = validate_model_result(
            json.loads(_content(transport.post(payload))),
            {item.evidence_id for metric in metrics for item in metric.evidence},
            {item.metric_id for item in metrics},
            {item.metric_id for item in metrics if item.evaluation.evidence.computed_flag in (AbnormalFlag.HIGH, AbnormalFlag.LOW)},
        )
    except (json.JSONDecodeError, TypeError) as error:
        raise ReportPipelineError("DeepSeek returned invalid JSON") from error
    return ReportDocument(
        render_markdown(metrics, model_result, channels, reasoning_paths),
        model_result, metrics, channels, reasoning_paths, reasoning_rejections,
    )


def analyze_report(
    report: Mapping[str, Any],
    index: Path,
    *,
    knowledge_graph: Path | None = None,
    provenance: ProvenanceContext | None = None,
    transport: JsonTransport | None = None,
) -> str:
    return analyze_report_document(
        report,
        index,
        knowledge_graph=knowledge_graph,
        provenance=provenance,
        transport=transport,
    ).markdown


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--report", type=Path, required=True)
    analyze.add_argument("--index", type=Path, required=True)
    analyze.add_argument("--knowledge-graph", type=Path)
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping):
            raise ReportPipelineError("report JSON must be an object")
        rendered = analyze_report(
            report, args.index, knowledge_graph=args.knowledge_graph
        )
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    except (OSError, json.JSONDecodeError, ReportPipelineError) as error:
        parser.error(str(error))
    print(args.output)


if __name__ == "__main__":
    main()
