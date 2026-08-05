"""Deterministic, provenance-bound local QA HTTP server."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, ProxyHandler, build_opener

from ..evidence.index import (
    ProvenanceContext,
    QaError,
    _index_meta,
    _load_provenance,
    _validate_index_bindings,
    build_evidence_index,
    query_index_with_graph,
)
from ..graph.graph_retrieval import GraphRetrievalError, graph_retrieve
from ..report.desktop_app import DesktopAppError, css as desktop_css, html as desktop_html
from ..report.desktop_app import javascript as desktop_javascript
from ..report.desktop_app import parse_report_payload
from ..report.paddleocr_report import (
    PaddleOcrJobsClient,
    PaddleOcrReportError,
    image_report_job,
)
from ..rules.analysis import AnalysisRule, analyze_report, result_to_dict

MAX_BODY_BYTES = 256 * 1024
MAX_OCR_BODY_BYTES = 14 * 1024 * 1024

def _provider_config() -> tuple[str, str, str, float]:
    base_url = os.environ.get("MEDICAL_KG_QA_BASE_URL")
    api_key = os.environ.get("MEDICAL_KG_QA_API_KEY")
    model = os.environ.get("MEDICAL_KG_QA_MODEL")
    if not all((base_url, api_key, model)):
        raise QaError("openai-compatible mode requires explicit MEDICAL_KG_QA provider settings")
    try:
        timeout = float(os.environ.get("MEDICAL_KG_QA_TIMEOUT", "15"))
    except ValueError as error:
        raise QaError("MEDICAL_KG_QA_TIMEOUT must be numeric") from error
    if not 0 < timeout <= 60 or not base_url.startswith(("http://", "https://")):
        raise QaError("openai-compatible provider configuration is invalid")
    return base_url.rstrip("/"), api_key, model, timeout


def _model_answer(query: str, evidence: list[dict[str, Any]]) -> str:
    base_url, api_key, model, timeout = _provider_config()
    context = "\n".join(f"[{number}] {item['text']}" for number, item in enumerate(evidence, 1))
    payload = {"model": model, "messages": [{"role": "user", "content": f"Answer only from this evidence. Cite at least one bracketed evidence number.\nQuestion: {query}\nEvidence:\n{context}"}], "temperature": 0}
    request = Request(f"{base_url}/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        response = build_opener(ProxyHandler({})).open(request, timeout=timeout)
        with response:
            value = json.loads(response.read().decode("utf-8"))
        answer = value["choices"][0]["message"]["content"]
    except (URLError, OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise QaError("openai-compatible provider returned no usable answer") from error
    if not isinstance(answer, str) or not answer.strip() or not re.search(r"\[(?:[1-9][0-9]*)\]", answer):
        raise QaError("openai-compatible answer must cite supplied evidence")
    allowed = {str(number) for number in range(1, len(evidence) + 1)}
    if any(number not in allowed for number in re.findall(r"\[([1-9][0-9]*)\]", answer)):
        raise QaError("openai-compatible answer cited unavailable evidence")
    return answer.strip()


def _answer(
    index: Path, query: str, top_k: int, mode: str = "extractive",
    provenance: ProvenanceContext | None = None,
    knowledge_graph: Path | None = None,
) -> dict[str, Any]:
    evidence, channels = query_index_with_graph(index, knowledge_graph, query, top_k, provenance)
    if not evidence:
        return {"mode": mode, "answer": "未检索到足够证据。", "citations": [], "evidence": [], "channels": channels}
    sentences = []
    citations = []
    for number, item in enumerate(evidence, 1):
        sentence = re.split(r"(?<=[。！？.!?])\s*", item["text"].strip())[0]
        sentences.append(f"{sentence} [{number}]")
        citations.append({key: item[key] for key in ("chunk_id", "printed_page_number", "source_pdf_page_number", "chapter_page_index")})
    answer = " ".join(sentences) if mode == "extractive" else _model_answer(query, evidence)
    return {"mode": mode, "answer": answer, "citations": citations, "evidence": evidence, "channels": channels}


class _QaHandler(BaseHTTPRequestHandler):
    server_version = "LocalEvidenceQA/0.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    @property
    def qa_index(self) -> Path:
        return self.server.qa_index  # type: ignore[attr-defined]

    def _send(self, status: int, body: Any, content_type: str = "application/json; charset=utf-8") -> None:
        payload = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Content-Security-Policy", "default-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self._send(200, {"status": "ready", "mode": "extractive", "graph_enabled": self.server.qa_knowledge_graph is not None})  # type: ignore[attr-defined]
        elif self.path == "/api/meta":
            self._send(200, _index_meta(self.qa_index))
        elif self.path == "/":
            self._send(200, desktop_html(), "text/html; charset=utf-8")
        elif self.path == "/assets/app.js":
            self._send(200, desktop_javascript().encode(), "application/javascript; charset=utf-8")
        elif self.path == "/assets/app.css":
            self._send(200, desktop_css().encode(), "text/css; charset=utf-8")
        elif self.path == "/source.pdf" and self.server.qa_source_pdf is not None:  # type: ignore[attr-defined]
            try:
                self._send(200, self.server.qa_source_pdf.read_bytes(), "application/pdf")  # type: ignore[attr-defined]
            except OSError:
                self._send(404, {"error": "not_found"})
        else:
            self._send(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {
            "/api/search", "/api/answer", "/api/report-analysis",
            "/api/report-generation", "/api/report-ocr",
        }:
            self._send(404, {"error": "not_found"})
            return
        try:
            try:
                size = int(self.headers.get("Content-Length", "-1"))
            except ValueError as error:
                raise QaError("request body is invalid") from error
            body_limit = MAX_OCR_BODY_BYTES if self.path == "/api/report-ocr" else MAX_BODY_BYTES
            if size < 0 or size > body_limit:
                raise QaError("request body is invalid")
            value = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(value, dict):
                raise QaError("JSON body must be an object")
            if self.path == "/api/report-ocr":
                if set(value) != {"filename", "content_base64"}:
                    raise QaError("OCR request must contain filename and content_base64")
                filename = value.get("filename")
                encoded = value.get("content_base64")
                if not isinstance(filename, str) or not filename or not isinstance(encoded, str):
                    raise QaError("OCR request fields are invalid")
                try:
                    image = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as error:
                    raise QaError("content_base64 is invalid") from error
                try:
                    ocr_client = self.server.qa_ocr_client  # type: ignore[attr-defined]
                    if ocr_client is None:
                        ocr_client = PaddleOcrJobsClient.from_environment()
                    image_result, job = image_report_job(image, filename, client=ocr_client)
                except PaddleOcrReportError as error:
                    raise QaError(str(error)) from error
                result = {
                    "report": dict(image_result.report),
                    "job": {**job.summary(), "validation_model": "PP-OCRv6"},
                }
            elif self.path == "/api/report-generation":
                from ..report.report_pipeline import ReportPipelineError, analyze_report_document

                try:
                    result = analyze_report_document(
                        value,
                        self.qa_index,
                        knowledge_graph=self.server.qa_knowledge_graph,  # type: ignore[attr-defined]
                        provenance=self.server.qa_provenance,  # type: ignore[attr-defined]
                        transport=self.server.qa_report_transport,  # type: ignore[attr-defined]
                    ).to_dict()
                except ReportPipelineError as error:
                    raise QaError(str(error)) from error
            elif self.path == "/api/report-analysis":
                report = parse_report_payload(value)
                analysis = analyze_report(
                    report,
                    self.server.qa_analysis_rules,  # type: ignore[attr-defined]
                    approved_book_registry=self.server.qa_book_registry,  # type: ignore[attr-defined]
                )
                result = {
                    "analysis": result_to_dict(analysis),
                    "rule_status": {
                        "approved_rule_count": sum(
                            rule.status == "approved" for rule in self.server.qa_analysis_rules  # type: ignore[attr-defined]
                        ),
                        "message": "暂无 approved 规则，未生成医学解释。"
                        if not self.server.qa_analysis_rules  # type: ignore[attr-defined]
                        else "规则已按确定性条件计算。",
                    },
                }
            else:
                query = value.get("query")
                top_k = value.get("top_k", 5)
                if self.path.endswith("search"):
                    evidence, channels = query_index_with_graph(
                        self.qa_index, self.server.qa_knowledge_graph, query, top_k,
                        self.server.qa_provenance,  # type: ignore[attr-defined]
                    )
                    result = {"evidence": evidence, "channels": channels}
                else:
                    result = _answer(
                        self.qa_index, query, top_k, self.server.qa_answer_mode,
                        self.server.qa_provenance, self.server.qa_knowledge_graph,  # type: ignore[attr-defined]
                    )
            self._send(200, result)
        except (UnicodeDecodeError, json.JSONDecodeError, QaError, DesktopAppError) as error:
            self._send(400, {"error": "invalid_request", "detail": str(error)})


def make_server(index: Path, host: str = "127.0.0.1", port: int = 18852, answer_mode: str = "extractive", *, analysis_rules: tuple[AnalysisRule, ...] = (), approved_book_registry: dict[str, dict[str, Any]] | None = None, source_pdf_path: Path | None = None, chunk_package: Path | None = None, knowledge_graph: Path | None = None, report_transport: Any | None = None, ocr_client: PaddleOcrJobsClient | None = None, allow_lan: bool = False) -> ThreadingHTTPServer:
    if answer_mode not in {"extractive", "openai-compatible"}:
        raise QaError("unsupported answer mode")
    if answer_mode == "openai-compatible":
        _provider_config()
    if host not in {"127.0.0.1", "::1", "localhost"} and not allow_lan:
        raise QaError("non-loopback binding requires explicit allow_lan")
    if allow_lan and host not in {"0.0.0.0", "::"}:
        raise QaError("LAN mode must bind an unspecified interface address")
    _index_meta(index)
    if source_pdf_path is not None and (not source_pdf_path.is_file() or source_pdf_path.suffix.lower() != ".pdf"):
        raise QaError("source PDF must be an existing local PDF file")
    provenance = _load_provenance(index, chunk_package) if chunk_package is not None else None
    if provenance is not None:
        _validate_index_bindings(index, provenance)
    if knowledge_graph is not None:
        if not knowledge_graph.is_file():
            raise QaError("knowledge graph must be an existing SQLite file")
        try:
            graph_retrieve(knowledge_graph, index, "__startup_validation__", top_k=1)
        except GraphRetrievalError as error:
            raise QaError(str(error)) from error
    server = ThreadingHTTPServer((host, port), _QaHandler)
    server.qa_index = index.resolve()  # type: ignore[attr-defined]
    server.qa_answer_mode = answer_mode  # type: ignore[attr-defined]
    server.qa_analysis_rules = tuple(analysis_rules)  # type: ignore[attr-defined]
    server.qa_book_registry = approved_book_registry  # type: ignore[attr-defined]
    server.qa_report_transport = report_transport  # type: ignore[attr-defined]
    server.qa_ocr_client = ocr_client  # type: ignore[attr-defined]
    server.qa_source_pdf = source_pdf_path.resolve() if source_pdf_path else None  # type: ignore[attr-defined]
    server.qa_provenance = provenance  # type: ignore[attr-defined]
    server.qa_knowledge_graph = knowledge_graph.resolve() if knowledge_graph else None  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-evidence-index")
    build.add_argument("--chunk-package", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--generation-timestamp")
    serve = commands.add_parser("serve-qa")
    serve.add_argument("--index", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=18852)
    serve.add_argument("--answer-mode", default="extractive", choices=("extractive", "openai-compatible"))
    serve.add_argument("--source-pdf", type=Path)
    serve.add_argument("--chunk-package", type=Path)
    serve.add_argument("--knowledge-graph", type=Path)
    serve.add_argument("--allow-lan", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "build-evidence-index":
            print(json.dumps(build_evidence_index(args.chunk_package, args.output, args.generation_timestamp), ensure_ascii=False))
        else:
            make_server(args.index, args.host, args.port, args.answer_mode, source_pdf_path=args.source_pdf, chunk_package=args.chunk_package, knowledge_graph=args.knowledge_graph, allow_lan=args.allow_lan).serve_forever()
    except QaError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
