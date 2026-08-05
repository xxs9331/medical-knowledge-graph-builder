import base64
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from medical_kg_sourceprep.analysis import AnalysisRule
from medical_kg_sourceprep.composite_rules import AtomicPredicate, CandidateStatus, ReviewRecord, TextAnchor
from medical_kg_sourceprep.qa import MAX_BODY_BYTES, build_evidence_index, make_server
from tests.test_qa import _candidate_graph, _chunk_package
from tests.test_report_pipeline import FakeTransport, _model, _report
from tests.test_paddleocr_report import _vl_record
from medical_kg_sourceprep.paddleocr_report import PaddleOcrJobResult


class _FakeOcrClient:
    def process_image(self, image, filename, model):
        if not image.startswith(b"\x89PNG") or filename != "report.png":
            raise AssertionError("unexpected OCR request")
        if model == "PP-OCRv6":
            from tests.test_paddleocr_report import _response

            return PaddleOcrJobResult(
                "job-2", model, "done", "https://result.example/text.jsonl",
                ({"result": {"ocrResults": _response()["result"]["ocrResults"]}},),
            )
        if model != "PaddleOCR-VL-1.6":
            raise AssertionError("unexpected OCR model")
        return PaddleOcrJobResult(
            "job-1", model, "done", "https://result.example/job.jsonl", (_vl_record(),)
        )


class DesktopApplicationTests(unittest.TestCase):
    def test_report_analysis_emits_claim_only_with_three_chain_rule(self) -> None:
        from tests.test_evidence_policy import _provenance

        book_source, registry = _provenance()
        rule = AnalysisRule(
            "synthetic-rule", "1.0.0", CandidateStatus.APPROVED.value,
            AtomicPredicate(
                "synthetic_metric", "synthetic condition", "ge", 10, "U",
                TextAnchor("synthetic-anchor", "book-fixed-1", 1, "synthetic condition"),
            ),
            ReviewRecord("reviewer", "approved", "1.0.0", "synthetic review"),
            "synthetic conclusion", {"synthetic_metric": book_source},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "evidence.sqlite"
            build_evidence_index(_chunk_package(root), index, "2026-01-01T00:00:00Z")
            server = make_server(index, "127.0.0.1", 0, analysis_rules=(rule,), approved_book_registry=registry)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(
                    f"http://127.0.0.1:{server.server_port}/api/report-analysis",
                    data=json.dumps({
                        "schema_version": "structured-report/v0.2",
                        "observations": [{
                            "raw_name": "synthetic_metric", "value": "12", "unit": "U",
                            "reference_interval": {"lower": "1", "upper": "10"},
                        }],
                    }).encode(), headers={"Content-Type": "application/json"}, method="POST",
                )
                with build_opener(ProxyHandler({})).open(request, timeout=3) as response:
                    result = json.loads(response.read())
                self.assertEqual(len(result["analysis"]["claims"]), 1)
                bundle = result["analysis"]["citation_bundles"][0]
                self.assertEqual({item["source_type"] for item in bundle["sources"]}, {"report", "book"})
                self.assertIn(bundle["computation"]["report_citation_id"], bundle["computation"]["citation_ids"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)

    def test_report_analysis_is_private_bounded_and_has_no_claim_without_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "evidence.sqlite"
            package = _chunk_package(root)
            build_evidence_index(package, index, "2026-01-01T00:00:00Z")
            manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
            first = manifest["chunks"][0]
            graph = _candidate_graph(
                root / "candidate.sqlite",
                first["chunk_id"],
                (package / first["chunk_path"]).read_text(encoding="utf-8"),
                node_name="alpha",
            )
            server = make_server(
                index,
                "127.0.0.1",
                0,
                chunk_package=package,
                knowledge_graph=graph,
                report_transport=FakeTransport(_model()),
                ocr_client=_FakeOcrClient(),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            opener = build_opener(ProxyHandler({}))
            base = f"http://127.0.0.1:{server.server_port}"
            report = {
                "schema_version": "structured-report/v0.2",
                "metadata": {"patient_name": "private-name", "patient_identifier": "private-id"},
                "observations": [{
                    "raw_name": "synthetic_metric", "standard_name": "synthetic_metric",
                    "value": "12", "unit": "U", "reference_interval": {"lower": "1", "upper": "10"},
                    "report_flag": "high",
                }],
            }
            try:
                request = Request(
                    base + "/api/report-analysis", data=json.dumps(report).encode(),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with opener.open(request, timeout=3) as response:
                    result = json.loads(response.read())
                serialized = json.dumps(result, ensure_ascii=False)
                self.assertEqual(result["analysis"]["claims"], [])
                self.assertEqual(result["rule_status"]["approved_rule_count"], 0)
                self.assertEqual(result["analysis"]["abnormalities"][0]["computed_flag"], "high")
                self.assertNotIn("private-name", serialized)
                self.assertNotIn("private-id", serialized)
                self.assertNotIn("patient_name", serialized)
                ocr_request = Request(
                    base + "/api/report-ocr",
                    data=json.dumps({
                        "filename": "report.png",
                        "content_base64": base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode(),
                    }).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with opener.open(ocr_request, timeout=3) as response:
                    ocr_result = json.loads(response.read())
                self.assertEqual(ocr_result["job"]["model"], "PaddleOCR-VL-1.6")
                self.assertEqual(ocr_result["job"]["validation_model"], "PP-OCRv6")
                self.assertEqual(len(ocr_result["report"]["observations"]), 3)
                parse_report = ocr_result["report"]
                self.assertEqual(parse_report["schema_version"], "structured-report/v0.2")
                generated_request = Request(
                    base + "/api/report-generation",
                    data=json.dumps(_report()).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with opener.open(generated_request, timeout=3) as response:
                    generated = json.loads(response.read())
                self.assertIn("# 体检报告分析", generated["markdown"])
                self.assertEqual(generated["report"]["summary"], _model()["summary"])
                self.assertEqual(generated["metrics"][0]["computed_flag"], "high")
                self.assertEqual(generated["evidence"][0]["source_pdf_page_number"], 21)
                self.assertTrue(generated["channels"]["graph"]["enabled"])
                self.assertGreaterEqual(generated["channels"]["graph"]["evidence_count"], 1)
                self.assertIn("graph", generated["evidence"][0])
                self.assertNotIn("private", generated["markdown"])
                with self.assertRaises(HTTPError) as unknown:
                    opener.open(Request(base + "/api/report-analysis", data=b'{"unknown":true}', method="POST"), timeout=3)
                self.assertEqual(unknown.exception.code, 400)
                with self.assertRaises(HTTPError) as oversized:
                    opener.open(
                        Request(base + "/api/report-analysis", data=b"x" * (MAX_BODY_BYTES + 1), method="POST"),
                        timeout=3,
                    )
                self.assertEqual(oversized.exception.code, 400)
                with opener.open(base + "/", timeout=3) as response:
                    html = response.read().decode()
                with opener.open(base + "/assets/app.js", timeout=3) as response:
                    javascript = response.read().decode()
                with opener.open(base + "/assets/app.css", timeout=3) as response:
                    css = response.read().decode()
                self.assertIn('id="report-form"', html)
                self.assertIn('id="ocr-form"', html)
                self.assertIn('accept="image/png,image/jpeg"', html)
                self.assertIn('maxlength="262144"', html)
                self.assertIn('id="search-tab"', html)
                self.assertIn("textContent", javascript)
                self.assertIn("无法判定", javascript)
                self.assertIn("validation_model", javascript)
                self.assertIn("数据待核对", javascript)
                self.assertIn("validation_issues", javascript)
                self.assertNotIn("innerHTML", javascript)
                self.assertNotIn("localStorage", javascript)
                self.assertIn("/api/report-generation", javascript)
                self.assertIn("/api/report-ocr", javascript)
                self.assertGreaterEqual(javascript.count("$('#report-json').value=''"), 2)
                self.assertNotIn("FileReader", javascript)
                self.assertNotIn("$('#claims')", javascript)
                self.assertIn("reportState.evidence", javascript)
                self.assertIn("evidenceDrawer(evidence)", javascript)
                self.assertIn("id=\"markdown-view\"", html)
                self.assertIn("Cleaned 字符区间（左闭右开）", javascript)
                self.assertIn("Cleaned Markdown 页内行", javascript)
                self.assertIn("上游来源 Markdown 行范围", javascript)
                self.assertIn("命中图节点", javascript)
                self.assertIn("图路径", javascript)
                self.assertIn("分析依据", javascript)
                self.assertIn("正在合并书内检索与图谱证据", javascript)
                self.assertIn("图谱辅助召回", javascript)
                self.assertIn("候选推理路径", javascript)
                self.assertIn("缺词诊断", javascript)
                self.assertIn("evidenceDrawer(item)", javascript)
                self.assertIn("/source.pdf#page=", javascript)
                self.assertNotIn("drawer('书内证据',JSON.stringify", javascript)
                self.assertIn("grid-template-columns:380px minmax(0,1fr)", css)
                self.assertIn(".evidence-quote", css)
                self.assertIn(".analysis-basis", css)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
