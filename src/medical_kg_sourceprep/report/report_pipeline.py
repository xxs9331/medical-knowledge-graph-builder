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
import sqlite3
import ssl
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence
from urllib import error as urlerror
from urllib import request

from .desktop_app import DesktopAppError, parse_report_payload
from ..graph.graph_retrieval import (
    GraphRetrievalError,
    graph_query_diagnostic,
    graph_reasoning_paths,
)
from .lab_terminology import canonicalize_laboratory_term
from ..api.qa import ProvenanceContext, QaError, query_index, query_index_with_graph
from .report_model import AbnormalFlag, EvaluationResult, Observation, evaluate_observation


ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
TIMEOUT_SECONDS = 60
MAX_RETRIES = 2
MAX_EVIDENCE_PER_METRIC = 3

VALIDATION_ISSUE_LABELS = {
    "missing_unit": "缺少单位",
    "default_unit_applied": "使用受控默认单位",
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
                    "unit_source": item.evaluation.normalized.unit_source,
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
                        # Bare abbreviations such as NEUT are ambiguous between an
                        # absolute count and a percentage.  The parsed standard name
                        # retains the report-column meaning, so use it for graph
                        # lookup while preserving the abbreviation in diagnostics.
                        if channel == "abbreviation" and observation.standard_name:
                            graph_query_term, _ = canonicalize_laboratory_term(
                                observation.standard_name, observation.abbreviation
                            )
                        else:
                            graph_query_term, _ = canonicalize_laboratory_term(
                                graph_term, graph_term
                            )
                        diagnostic = graph_query_diagnostic(knowledge_graph, graph_query_term)
                        graph_diagnostics.append({
                            **diagnostic,
                            "query": graph_term,
                            "resolved_query": graph_query_term,
                        })
                        items, _ = query_index_with_graph(
                            index,
                            knowledge_graph,
                            query,
                            top_k=10,
                            provenance=provenance,
                            graph_query=graph_query_term,
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
            graph_bound_rows = [
                row for row in rows if isinstance(row.get("graph"), Mapping)
            ]
            if graph_bound_rows:
                rows = graph_bound_rows
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
    payload = normalized.to_dict(include_inclusive=False)
    payload.update({
        "metric_id": metric.metric_id,
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
    })
    return payload


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
    graph_status = _graph_runtime_status(metrics)
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
            "status": graph_status if graph_enabled else None,
            "evidence_count": len(graph_evidence),
            "query_diagnostics": diagnostics,
            "query_diagnostic_counts": dict(sorted(diagnostic_counts.items())),
        },
    }


def _graph_runtime_status(metrics: Sequence[MetricInput]) -> str | None:
    statuses = {
        str(item.graph.get("status"))
        for metric in metrics
        for item in metric.evidence
        if item.graph and item.graph.get("status")
    }
    for metric in metrics:
        statuses.update(
            str(item.get("graph_status"))
            for item in metric.graph_diagnostics
            if item.get("graph_status")
        )
    return "final" if "final" in statuses else "candidate-only" if statuses else None


_FINAL_ACTIONABLE_RULE_STATUSES = frozenset({
    "final-case-match",
    "final-case-match-precondition-derived",
})


def _reportable_reasoning_paths(
    paths: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Project final graph output to rules with applicable conclusions only."""
    return tuple(
        path for path in paths
        if path.get("graph_status") != "final"
        or path.get("status") in _FINAL_ACTIONABLE_RULE_STATUSES
    )


def _reasoning_context(
    metrics: Sequence[MetricInput], knowledge_graph: Path | None, evidence_index: Path,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[tuple[MetricInput, ...], tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    if knowledge_graph is None:
        return tuple(metrics), (), ()
    observations = []
    for metric in metrics:
        flag = metric.evaluation.evidence.computed_flag
        normalized = metric.evaluation.normalized
        terms = [
            value for value in (
                normalized.raw_name, normalized.standard_name, normalized.abbreviation,
            ) if value
        ]
        observations.append({
            "metric_id": metric.metric_id,
            "terms": terms,
            "value": str(normalized.value) if normalized.value is not None else None,
            "unit": normalized.unit,
            "unit_source": normalized.unit_source,
            "computed_flag": flag.value if flag is not None else None,
            "reference_interval": {
                "lower": str(normalized.lower) if normalized.lower is not None else None,
                "upper": str(normalized.upper) if normalized.upper is not None else None,
            },
            "sample_type": normalized.sample_type,
            "method": normalized.method,
        })
    sex = str((metadata or {}).get("patient_sex") or "").strip().casefold()
    sex_value = {
        "男": "男", "男性": "男", "male": "男", "m": "男",
        "女": "女", "女性": "女", "female": "女", "f": "女",
    }.get(sex)
    context_facts = ({
        "性别": {"metric_id": "metadata:patient_sex", "value": sex_value, "unit": None}
    } if sex_value else {})
    try:
        result = graph_reasoning_paths(
            knowledge_graph, evidence_index, observations, context_facts=context_facts
        )
    except GraphRetrievalError as error:
        raise ReportPipelineError(str(error)) from error
    reportable_paths = _reportable_reasoning_paths(result.paths)
    metric_positions = {metric.metric_id: index for index, metric in enumerate(metrics)}
    expanded_metrics = list(metrics)
    existing_chunks = {
        item.chunk_id for metric in expanded_metrics for item in metric.evidence
    }
    needed_chunks = {
        str(chunk_id)
        for path in reportable_paths
        for chunk_id in path.get("chunk_ids", [])
        if str(chunk_id) not in existing_chunks
    }
    rows: dict[str, Mapping[str, Any]] = {}
    if needed_chunks:
        placeholders = ",".join("?" for _ in needed_chunks)
        try:
            connection = sqlite3.connect(f"file:{Path(evidence_index).resolve()}?mode=ro", uri=True)
            connection.row_factory = sqlite3.Row
            rows = {
                str(row["chunk_id"]): dict(row)
                for row in connection.execute(
                    "SELECT c.chunk_id,c.text,c.chunk_sha256,c.cleaned_char_start,c.cleaned_char_end,"
                    "p.printed_page_number,p.source_pdf_page_number "
                    "FROM chunks c JOIN pages p ON p.page_id=c.page_id "
                    f"WHERE c.chunk_id IN ({placeholders})",
                    tuple(sorted(needed_chunks)),
                )
            }
        except sqlite3.Error as error:
            raise ReportPipelineError("reasoning evidence index is unreadable") from error
        finally:
            if "connection" in locals():
                connection.close()
    for path in reportable_paths:
        metric_id = next(
            (str(value) for value in path.get("matched_metric_ids", []) if str(value) in metric_positions),
            None,
        )
        if metric_id is None:
            continue
        position = metric_positions[metric_id]
        additions: list[Evidence] = []
        for chunk_id in path.get("chunk_ids", []):
            chunk_id = str(chunk_id)
            if chunk_id in existing_chunks:
                continue
            row = rows.get(chunk_id)
            if row is None:
                continue
            additions.append(Evidence(
                evidence_id="E" + chunk_id,
                chunk_id=chunk_id,
                text=str(row["text"]),
                printed_page_number=int(row["printed_page_number"]),
                source_pdf_page_number=int(row["source_pdf_page_number"]),
                chunk_sha256=str(row["chunk_sha256"]),
                score=0.0,
                retrieval_reason="graph_reasoning_path",
                graph={
                    "status": path.get("graph_status"),
                    "reasoning_path_id": path.get("path_id"),
                    "rule_id": path.get("rule_id"),
                },
            ))
            existing_chunks.add(chunk_id)
        if additions:
            metric = expanded_metrics[position]
            expanded_metrics[position] = MetricInput(
                metric.metric_id, metric.observation, metric.evaluation,
                (*metric.evidence, *additions), metric.graph_diagnostics,
            )
    metrics = tuple(expanded_metrics)
    evidence_ids_by_chunk = {
        item.chunk_id: item.evidence_id
        for metric in metrics
        for item in metric.evidence
    }
    accepted: list[Mapping[str, Any]] = []
    rejections = list(result.rejections)
    for path in reportable_paths:
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
    return metrics, tuple(accepted), tuple(rejections)


def _used_evidence_ids(
    result: Mapping[str, Any], reasoning_paths: Sequence[Mapping[str, Any]]
) -> set[str]:
    used: set[str] = set()
    for item in result.get("abnormal_analyses", []):
        used.update(str(value) for value in item.get("evidence_ids", []))
    association = result.get("association_analysis", {})
    if isinstance(association, Mapping):
        used.update(str(value) for value in association.get("evidence_ids", []))
    for item in result.get("attention_suggestions", []):
        used.update(str(value) for value in item.get("evidence_ids", []))
    for path in reasoning_paths:
        used.update(str(value) for value in path.get("evidence_ids", []))
    return used


def _prune_metric_evidence(
    metrics: Sequence[MetricInput], used_evidence_ids: set[str]
) -> tuple[MetricInput, ...]:
    return tuple(
        MetricInput(
            metric.metric_id,
            metric.observation,
            metric.evaluation,
            tuple(item for item in metric.evidence if item.evidence_id in used_evidence_ids),
            metric.graph_diagnostics,
        )
        for metric in metrics
    )


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
    graph_statuses = {
        str(path.get("graph_status")) for path in reasoning_paths if path.get("graph_status")
    }
    graph_statuses.update(
        str(item.graph.get("status"))
        for metric in metrics for item in metric.evidence
        if item.graph and item.graph.get("status")
    )
    final_graph = "final" in graph_statuses
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
        "final_rule_paths" if final_graph else "candidate_reasoning_paths": [
            dict(item) for item in reasoning_paths
        ],
    }
    graph_instruction = (
        "evidence 中的 final graph 是最终图谱投影；仅当路径的 diagnostic_use=allowed 且 approved_execution=true 时，"
        "可将该规则匹配作为书内规则结论的结构化依据；"
        if final_graph else
        "evidence 中的 candidate-only graph 仅用于召回、排序和发现关联，不能单独支持医学结论；"
    )
    path_instruction = (
        "final_rule_paths 是多指标规则结论的唯一授权来源：仅可使用其中状态为 final-case-match 或 "
        "final-case-match-precondition-derived 的结论。未出现在该数组的规则代表未命中、前置条件失败、歧义或不支持；"
        "不得在 association_analysis 中据此拼接疾病、贫血形态或其他多指标模式。final rule paths 已按最终图规则计算，"
        "但仍须引用同次输入中的书内 evidence_id，且不构成诊断；"
        if final_graph else
        "candidate_reasoning_paths 是未经批准的候选条件求值，只能帮助合并分析；"
    )
    instructions = (
        "你是证据约束的医学报告表达器。只能使用 INPUT 中的程序判定和书内证据，禁止使用常识补充。"
        + graph_instruction + path_instruction +
        (
            "rule_evaluation 可以说明报告实际值是否满足书中规则case，但不得替代未验证前置条件；"
            if final_graph else
            "candidate_evaluation 可以说明报告实际值是否满足书中候选case，但不得替代未验证前置条件；"
        ) +
        "路径状态包含 precondition-unverified 时，只能描述数值组合满足候选条件，禁止直接写成患者已患该疾病。"
        "路径状态包含 precondition-derived 时，表示显式上游候选规则满足了前置条件，但仍只能称候选判定；"
        "路径状态包含 precondition-failed 时，不得输出该路径的疾病分类结果。"
        "不得把百分数降低等同于绝对值降低，不得用降低方向证据解释升高结果，反之亦然。"
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
    )
    return instructions + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


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
    final_graph = graph.get("status") == "final"
    notice = "程序异常判定、整书检索证据"
    if graph_enabled:
        notice += "和第一章最终知识图谱" if final_graph else "和第一章候选知识图谱辅助召回"
    lines = [
        "# 体检报告分析", "",
        f"> 这是基于{notice}生成的辅助性摘要，不构成诊断、治疗或用药建议。", "",
        "## 分析依据", "",
        "- 异常判定：程序按报告参考区间重算",
        "- 书内检索：全书证据索引",
        (
            f"- 知识图谱：第一章{'最终' if final_graph else '候选'}图谱（{graph.get('status') or 'unknown'}），辅助召回证据 "
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
    association_evidence = result["association_analysis"].get("evidence_ids", [])
    if association_evidence:
        lines += ["证据：" + references(association_evidence), ""]
    if reasoning_paths:
        lines += ["## 规则推理路径" if final_graph else "## 候选推理路径", ""]
        lines += [
            (
                "> 以下路径来自最终图谱规则执行，并附带书内证据；不构成诊断。"
                if final_graph else
                "> 以下路径已作为大模型关联分析的受限上下文；只做候选条件匹配，未进入approved判定，不构成诊断。"
            ),
            "",
        ]
        for path in reasoning_paths:
            metric_names = "、".join(str(item) for item in path.get("matched_metric_ids", []))
            lines.append(f"### {path.get('rule_name', '候选规则')}")
            lines.append("")
            lines.append(f"状态：`{path.get('status', 'candidate-only')}`；共同命中指标：{metric_names}。")
            lines.append("")
            evaluation = path.get("candidate_evaluation")
            if isinstance(evaluation, Mapping):
                if evaluation.get("candidate_result") is not None:
                    lines.append(
                        f"{'规则' if final_graph else '候选'}结果：{evaluation.get('candidate_result')}。"
                    )
                    lines.append("")
                for condition in evaluation.get("condition_trace", []):
                    expected = condition.get("expected_value")
                    if expected is None and condition.get("expected_lower") is not None:
                        expected = f"{condition.get('expected_lower')}~{condition.get('expected_upper')}"
                    actual = str(condition.get("actual_value"))
                    if condition.get("actual_unit"):
                        actual += f" {condition.get('actual_unit')}"
                    lines.append(
                        f"- 条件 `{condition.get('input')} {condition.get('op')} {expected}`："
                        f"实际值 `{actual}`，"
                        f"结果 `{condition.get('status')}`"
                    )
                if evaluation.get("condition_trace"):
                    lines.append("")
            preconditions = path.get("preconditions", [])
            precondition_evaluations = path.get("precondition_evaluations", [])
            if path.get("preconditions_verified") and precondition_evaluations:
                lines.append("已由图谱规则链确认前置条件：" + "；".join(
                    f"{item.get('context')} {item.get('op')} {item.get('expected_value')}"
                    f"（来源：{item.get('source_rule_name')} → {item.get('source_candidate_result')}）"
                    for item in precondition_evaluations
                ))
                lines.append("")
            elif precondition_evaluations:
                lines.append("前置条件未满足：" + "；".join(
                    f"{item.get('context')}：{item.get('actual_value')}"
                    f"（来源：{item.get('source_rule_name')}）"
                    for item in precondition_evaluations
                ))
                lines.append("")
            elif preconditions:
                lines.append("未验证前置条件：" + "；".join(
                    f"{item.get('context') or item.get('input')} {item.get('op')} {item.get('value')}"
                    for item in preconditions
                ))
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
    metrics, reasoning_paths, reasoning_rejections = _reasoning_context(
        metrics, knowledge_graph, index,
        report.get("metadata") if isinstance(report.get("metadata"), Mapping) else None,
    )
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
    metrics = _prune_metric_evidence(
        metrics, _used_evidence_ids(model_result, reasoning_paths)
    )
    channels = _retrieval_channels(metrics, knowledge_graph)
    channels = {
        **channels,
        "graph": {
            **channels["graph"],
            "reasoning_path_count": len(reasoning_paths),
            "reasoning_rejection_count": len(reasoning_rejections),
            "reasoning_context_sent_to_model": bool(reasoning_paths),
        },
    }
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
