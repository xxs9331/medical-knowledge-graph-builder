import json
from io import BytesIO
import unittest
from urllib.error import HTTPError

from medical_kg_sourceprep.report.desktop_app import parse_report_payload
from medical_kg_sourceprep.report.paddleocr_report import (
    PaddleOcrClient,
    OcrDocument,
    OcrLine,
    OcrPage,
    PaddleOcrJobResult,
    PaddleOcrJobsClient,
    PaddleOcrReportError,
    convert_layout_job_to_report,
    convert_ocr_to_report,
    image_report_job,
    ocr_document_from_job,
    parse_ocr_response,
    _reconcile_missing_units,
)


def _box(x: int, y: int, width: int = 70) -> list[int]:
    return [x, y, x + width, y + 12]


def _response() -> dict:
    texts = [
        "中日友好医院", "报告日期：2026-04-15", "性别：女", "年龄：30", "样本类型：血清",
        "检验项目", "缩写", "结果", "单位", "参考范围",
        "天冬氨酸氨基转移酶", "AST", "37", "U/L", "0-31", "↑",
        "白蛋白", "ALB", "39.7", "g/L", "40-55", "↓",
        "C反应蛋白", "CRP", "4.24", "mg/L", "≤3.00",
    ]
    boxes = [
        _box(0, 0), _box(0, 20), _box(180, 20), _box(260, 20), _box(340, 20),
        _box(0, 50), _box(190, 50), _box(260, 50), _box(330, 50), _box(410, 50),
        _box(0, 80, 170), _box(190, 80), _box(260, 80), _box(330, 80), _box(410, 80), _box(490, 80),
        _box(0, 110), _box(190, 110), _box(260, 110), _box(330, 110), _box(410, 110), _box(490, 110),
        _box(0, 140), _box(190, 140), _box(260, 140), _box(330, 140), _box(410, 140),
    ]
    return {
        "logId": "request-1",
        "errorCode": 0,
        "errorMsg": "Success",
        "result": {
            "ocrResults": [{
                "prunedResult": {
                    "rec_texts": texts,
                    "rec_scores": [0.99] * len(texts),
                    "rec_boxes": boxes,
                }
            }]
        },
    }


def _vl_record() -> dict:
    markdown = """报告日期：2026-07-16
<table><tr><th>序号</th><th>项目代号</th><th>项目名称</th><th>结果</th><th>标志</th><th>单位</th><th>参考值</th></tr>
<tr><td>1</td><td>ALT</td><td>谷丙转氨酶</td><td>8</td><td></td><td>U/L</td><td>7--50</td></tr>
<tr><td>2</td><td>CH</td><td>总胆固醇</td><td>6.05</td><td>\\uparrow</td><td>mmol/L</td><td>0--5.2</td></tr>
<tr><td>3</td><td>A/G</td><td>白球比</td><td>1.60</td><td></td><td></td><td>1.5--2.5</td></tr></table>"""
    return _layout_record(markdown)


def _vl_variant_record() -> dict:
    markdown = """<table><tr><th>序号</th><th>英文缩写</th><th>项目</th><th>结果</th><th>提示</th><th>单位</th><th>参考范围</th><th>检验方法</th></tr>
<tr><td>1</td><td>ALT</td><td>血清丙氨酸氨基转移酶测定</td><td>13.7</td><td></td><td>U/L</td><td>[0.0-40.0]</td><td>速率法</td></tr>
<tr><td>2</td><td>CK-MB</td><td>肌酸激酶同工酶</td><td>1.50</td><td></td><td>ng/ml</td><td>[&lt;=5.00]</td><td>免疫法</td></tr>
<tr><td>3</td><td>HDL-C</td><td>高密度脂蛋白胆固醇</td><td>1.11</td><td>\\downarrow</td><td>mmol/L</td><td>[&gt;1.15]</td><td>直接法</td></tr>
<tr><td>4</td><td>LDL-C</td><td>低密度脂蛋白胆固醇</td><td>2.38</td><td>*</td><td>mmol/L</td><td>ASCVD风险低危目标值&lt;3.4mmol/L</td><td>直接法</td></tr></table>"""
    return _layout_record(markdown)


def _vl_embedded_code_record() -> dict:
    markdown = """<table><tr><th>检验项目</th><th>英文</th><th>结果</th><th>参考值</th><th>单位</th></tr>
<tr><td>1 血清丙氨酸氨基转移酶 ALT</td><td>AST</td><td>15.39</td><td>0-40</td><td>U/L</td></tr>
<tr><td>2 血清高密度脂蛋白胆固醇 HDL-C</td><td></td><td>2.38</td><td>$ \\geq $1.04</td><td>mmol/L</td></tr>
<tr><td>3 尿酸</td><td>UA</td><td>303.03</td><td>142-416</td><td>$ \\mu $mol/L</td></tr></table>"""
    return _layout_record(markdown)


def _layout_record(markdown: str) -> dict:
    table_start = markdown.find("<table")
    blocks = []
    if table_start > 0:
        blocks.append({
            "block_id": 0,
            "block_order": 0,
            "block_label": "text",
            "block_content": markdown[:table_start].strip(),
            "block_bbox": [0, 0, 100, 20],
        })
    blocks.append({
        "block_id": 1,
        "block_order": 1,
        "block_label": "table",
        "block_content": markdown[table_start:] if table_start >= 0 else markdown,
        "block_bbox": [0, 20, 100, 100],
    })
    return {"result": {"layoutParsingResults": [{
        "prunedResult": {"parsing_res_list": blocks},
        "markdown": {"text": markdown},
    }]}}


class _Response:
    status = 200

    def __init__(self, value: object) -> None:
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.value, ensure_ascii=False).encode()


class _Opener:
    def __init__(self, value: object) -> None:
        self.value = value
        self.request = None
        self.timeout = None

    def open(self, api_request, timeout):
        self.request = api_request
        self.timeout = timeout
        return _Response(self.value)


class _JobsOpener:
    def __init__(self) -> None:
        self.requests = []
        self.polls = 0

    def open(self, api_request, timeout):
        self.requests.append(api_request)
        if api_request.full_url.endswith("/api/v2/ocr/jobs"):
            return _Response({"data": {"jobId": "job-1"}})
        if api_request.full_url.endswith("/api/v2/ocr/jobs/job-1"):
            self.polls += 1
            if self.polls == 1:
                return _Response({"data": {"state": "running"}})
            return _Response({
                "data": {
                    "state": "done",
                    "resultUrl": {"jsonUrl": "https://result.example/job-1.jsonl"},
                }
            })
        if api_request.full_url == "https://result.example/job-1.jsonl":
            value = {"result": {"ocrResults": _response()["result"]["ocrResults"]}}
            response = _Response(value)
            response.read = lambda: (json.dumps(value, ensure_ascii=False) + "\n").encode()
            return response
        raise AssertionError(api_request.full_url)


class PaddleOcrReportTests(unittest.TestCase):
    def test_official_request_contract_and_structured_report(self) -> None:
        opener = _Opener(_response())
        client = PaddleOcrClient(
            "https://example.aistudio-hub.baidu.com/ocr", "private-token", opener=opener
        )
        document = client.recognize_image(b"\x89PNG\r\n\x1a\nimage", "report.png")
        self.assertEqual(opener.request.get_header("Authorization"), "token private-token")
        payload = json.loads(opener.request.data)
        self.assertEqual(payload["fileType"], 1)
        self.assertFalse(payload["visualize"])
        self.assertTrue(payload["useDocOrientationClassify"])
        self.assertNotIn("private-token", opener.request.data.decode())

        report = convert_ocr_to_report(document)
        parsed = parse_report_payload(report)
        self.assertEqual(report["metadata"]["hospital"], "中日友好医院")
        self.assertEqual(report["metadata"]["report_date"], "2026-04-15")
        self.assertEqual(report["metadata"]["sample_type"], "血清")
        self.assertEqual(parsed["天冬氨酸氨基转移酶"].report_flag.value, "high")
        self.assertEqual(parsed["白蛋白"].report_flag.value, "low")
        self.assertEqual(parsed["C反应蛋白"].reference_interval.lower, None)
        self.assertEqual(parsed["C反应蛋白"].reference_interval.upper, "3.00")

    def test_invalid_response_fails_closed(self) -> None:
        with self.assertRaisesRegex(PaddleOcrReportError, "rec_scores"):
            value = _response()
            value["result"]["ocrResults"][0]["prunedResult"]["rec_scores"] = []
            parse_ocr_response(value)

    def test_credentials_and_http_errors_do_not_leak_token(self) -> None:
        class FailingOpener:
            def open(self, api_request, timeout):
                raise HTTPError(api_request.full_url, 403, "private-token", {}, None)

        client = PaddleOcrClient(
            "https://example.aistudio-hub.baidu.com/ocr",
            "private-token",
            opener=FailingOpener(),
        )
        with self.assertRaises(PaddleOcrReportError) as raised:
            client.recognize_image(b"\xff\xd8\xffimage", "report.jpg")
        self.assertNotIn("private-token", str(raised.exception))
        self.assertIn("HTTP 403", str(raised.exception))

    def test_input_is_bounded_and_https_only(self) -> None:
        with self.assertRaisesRegex(PaddleOcrReportError, "HTTPS"):
            PaddleOcrClient("http://example.com/ocr", "token")
        client = PaddleOcrClient(
            "https://example.aistudio-hub.baidu.com/ocr", "token", opener=_Opener(_response())
        )
        with self.assertRaisesRegex(PaddleOcrReportError, "PNG or JPEG"):
            client.recognize_image(b"data", "report.gif")
        with self.assertRaisesRegex(PaddleOcrReportError, "does not match"):
            client.recognize_image(b"not-png", "report.png")

    def test_async_jobs_contract_polls_and_downloads_jsonl(self) -> None:
        opener = _JobsOpener()
        client = PaddleOcrJobsClient(
            "private-token",
            opener=opener,
            sleep=lambda _seconds: None,
            poll_interval_seconds=0.01,
        )
        result = client.process_url(
            "https://example.com/report.png",
            "PP-OCRv6",
        )
        submitted = opener.requests[0]
        self.assertEqual(submitted.get_header("Authorization"), "bearer private-token")
        payload = json.loads(submitted.data)
        self.assertEqual(payload["model"], "PP-OCRv6")
        self.assertEqual(payload["optionalPayload"]["useTextlineOrientation"], False)
        self.assertEqual(result.state, "done")
        self.assertEqual(result.summary()["ocr_pages"], 1)
        self.assertIsNone(opener.requests[-1].get_header("Authorization"))

    def test_async_job_submission_retries_only_queue_busy_code(self) -> None:
        class BusyOpener:
            def __init__(self):
                self.calls = 0

            def open(self, api_request, timeout):
                self.calls += 1
                if self.calls < 3:
                    body = BytesIO(b'{"code":10010,"message":"busy"}')
                    raise HTTPError(api_request.full_url, 400, "busy", {}, body)
                return _Response({"data": {"jobId": "job-after-retry"}})

        delays = []
        opener = BusyOpener()
        client = PaddleOcrJobsClient("private-token", opener=opener, sleep=delays.append)
        job_id = client._submit(b"{}", "application/json")
        self.assertEqual(job_id, "job-after-retry")
        self.assertEqual(opener.calls, 3)
        self.assertEqual(delays, [1, 2])

    def test_async_job_submission_does_not_retry_invalid_request(self) -> None:
        class InvalidOpener:
            calls = 0

            def open(self, api_request, timeout):
                self.calls += 1
                body = BytesIO(b'{"code":10001,"message":"invalid"}')
                raise HTTPError(api_request.full_url, 400, "invalid", {}, body)

        opener = InvalidOpener()
        client = PaddleOcrJobsClient("private-token", opener=opener, sleep=lambda _seconds: None)
        with self.assertRaisesRegex(PaddleOcrReportError, "code 10001"):
            client._submit(b"{}", "application/json")
        self.assertEqual(opener.calls, 1)

    def test_async_local_image_uses_multipart_without_token_in_body(self) -> None:
        opener = _JobsOpener()
        client = PaddleOcrJobsClient(
            "private-token",
            opener=opener,
            sleep=lambda _seconds: None,
            poll_interval_seconds=0.01,
        )
        client.process_image(b"\x89PNG\r\n\x1a\nimage", "report.png", "PaddleOCR-VL-1.6")
        submitted = opener.requests[0]
        self.assertIn("multipart/form-data", submitted.get_header("Content-type"))
        self.assertIn(b'PaddleOCR-VL-1.6', submitted.data)
        self.assertNotIn(b"private-token", submitted.data)

    def test_completed_pp_ocr_job_converts_to_structured_report(self) -> None:
        record = {"result": {"ocrResults": _response()["result"]["ocrResults"]}}
        job = PaddleOcrJobResult(
            "job-1", "PP-OCRv6", "done", "https://result.example/job.jsonl", (record,)
        )
        document = ocr_document_from_job(job)
        report = convert_ocr_to_report(document)
        self.assertEqual(document.request_id, "job-1")
        self.assertEqual(len(report["observations"]), 3)
        self.assertEqual(report["observations"][0]["report_flag"], "high")

    def test_vl_layout_table_converts_columns_and_double_hyphen_ranges(self) -> None:
        job = PaddleOcrJobResult(
            "job-1", "PaddleOCR-VL-1.6", "done", "https://result.example/job.jsonl",
            (_vl_record(),),
        )
        report = convert_layout_job_to_report(job)
        parsed = parse_report_payload(report)
        self.assertEqual(report["metadata"]["report_date"], "2026-07-16")
        self.assertEqual(parsed["丙氨酸氨基转移酶"].abbreviation, "ALT")
        self.assertEqual(parsed["丙氨酸氨基转移酶"].reference_interval.upper, "50")
        self.assertEqual(parsed["丙氨酸氨基转移酶"].report_flag.value, "normal")
        self.assertEqual(parsed["总胆固醇"].report_flag.value, "high")
        self.assertEqual(parsed["白蛋白/球蛋白比值"].unit, "-")

    def test_vl_layout_rejects_reversed_interval(self) -> None:
        record = _vl_record()
        text = record["result"]["layoutParsingResults"][0]["markdown"]["text"]
        page = record["result"]["layoutParsingResults"][0]
        page["markdown"]["text"] = text.replace("7--50", "50-7")
        table = page["prunedResult"]["parsing_res_list"][-1]
        table["block_content"] = table["block_content"].replace("7--50", "50-7")
        job = PaddleOcrJobResult(
            "job-1", "PaddleOCR-VL-1.6", "done", "https://result.example/job.jsonl", (record,)
        )
        with self.assertRaisesRegex(PaddleOcrReportError, "invalid laboratory table row"):
            convert_layout_job_to_report(job)

    def test_vl_layout_accepts_variant_headers_brackets_and_one_sided_ranges(self) -> None:
        job = PaddleOcrJobResult(
            "job-1", "PaddleOCR-VL-1.6", "done", "https://result.example/job.jsonl",
            (_vl_variant_record(),),
        )
        report = convert_layout_job_to_report(job)
        parsed = parse_report_payload(report)
        self.assertEqual(len(parsed), 4)
        self.assertEqual(parsed["血清丙氨酸氨基转移酶测定"].reference_interval.upper, "40.0")
        self.assertEqual(parsed["肌酸激酶-MB同工酶"].reference_interval.lower, None)
        self.assertEqual(parsed["肌酸激酶-MB同工酶"].reference_interval.upper, "5.00")
        self.assertEqual(parsed["高密度脂蛋白胆固醇"].report_flag.value, "low")
        self.assertEqual(parsed["低密度脂蛋白胆固醇"].report_flag, None)

    def test_vl_layout_prefers_code_embedded_with_name_and_normalizes_latex(self) -> None:
        job = PaddleOcrJobResult(
            "job-1", "PaddleOCR-VL-1.6", "done", "https://result.example/job.jsonl",
            (_vl_embedded_code_record(),),
        )
        parsed = parse_report_payload(convert_layout_job_to_report(job))
        self.assertEqual(parsed["丙氨酸氨基转移酶"].abbreviation, "ALT")
        self.assertEqual(parsed["高密度脂蛋白胆固醇"].abbreviation, "HDL-C")
        self.assertEqual(parsed["高密度脂蛋白胆固醇"].reference_interval.lower, "1.04")
        self.assertEqual(parsed["尿酸"].unit, "μmol/L")

    def test_vl_layout_discovers_non_first_header_and_repeated_column_groups(self) -> None:
        markdown = """<table>
<tr><td colspan="10">bounded report metadata</td></tr>
<tr><th>项目</th><th>中文名称</th><th>结果</th><th>单位</th><th>参考范围</th><th>项目</th><th>中文名称</th><th>结果</th><th>单位</th><th>参考范围</th></tr>
<tr><td>★WBC</td><td>白细胞计数</td><td>9.7\\uparrow</td><td>*10^9/L</td><td>3.5-9.5</td><td>MCV</td><td>平均红细胞体积</td><td>72.8\\downarrow</td><td>fL</td><td>82-100</td></tr>
<tr><td>LYM</td><td>淋巴细胞计数</td><td>1.48</td><td>*10^9/L</td><td>1.1-3.2</td><td>PLT</td><td>血小板计数</td><td>304</td><td>*10^9/L</td><td>125-350</td></tr>
</table>"""
        job = PaddleOcrJobResult(
            "job-1", "PaddleOCR-VL-1.6", "done", "https://result.example/job.jsonl",
            (_layout_record(markdown),),
        )
        parsed = parse_report_payload(convert_layout_job_to_report(job))
        self.assertEqual(len(parsed), 4)
        self.assertEqual(parsed["白细胞计数"].abbreviation, "WBC")
        self.assertEqual(parsed["白细胞计数"].report_flag.value, "high")
        self.assertEqual(parsed["平均红细胞体积"].report_flag.value, "low")
        self.assertEqual(parsed["血小板计数"].report_flag.value, "normal")
        self.assertEqual(
            [item["raw_name"] for item in convert_layout_job_to_report(job)["observations"]],
            ["白细胞计数", "淋巴细胞计数", "平均红细胞体积", "血小板计数"],
        )

    def test_vl_layout_deduplicates_cells_repeated_by_rowspan_provenance(self) -> None:
        markdown = """<table>
<tr><th>项目代号</th><th>项目名称</th><th>结果</th><th>单位</th><th>参考范围</th><th>项目代号</th><th>项目名称</th><th>结果</th><th>单位</th><th>参考范围</th></tr>
<tr><td rowspan="2">AST</td><td rowspan="2">天冬氨酸氨基转移酶</td><td rowspan="2">37</td><td rowspan="2">U/L</td><td rowspan="2">0-31</td><td>CR</td><td>肌酐</td><td>50.3</td><td>μmol/L</td><td>44-97</td></tr>
<tr><td>eGFR</td><td>估算肾小球滤过率</td><td>109.45</td><td>mL/min</td><td>&gt;90</td></tr>
</table>"""
        job = PaddleOcrJobResult(
            "job-1", "PaddleOCR-VL-1.6", "done", "https://result.example/job.jsonl",
            (_layout_record(markdown),),
        )
        parsed = parse_report_payload(convert_layout_job_to_report(job))
        self.assertEqual(len(parsed), 3)
        self.assertEqual(list(parsed).count("天冬氨酸氨基转移酶"), 1)

    def test_vl_layout_splits_direction_from_unit_and_normalizes_latex_power(self) -> None:
        markdown = """<table>
<tr><th>项目代号</th><th>项目名称</th><th>结果</th><th>单位</th><th>参考范围</th></tr>
<tr><td>AST</td><td>天冬氨酸氨基转移酶</td><td>37</td><td>\\uparrow U/L</td><td>0-31</td></tr>
<tr><td>ALB</td><td>白蛋白</td><td>39.7</td><td>\\downarrow g/L</td><td>40-55</td></tr>
<tr><td>eGFR</td><td>估算肾小球滤过率</td><td>109.45</td><td>$ ml/min/1.73m^{2} $</td><td>计算公式</td></tr>
<tr><td>HDL-</td><td>高密度脂蛋白胆固醇</td><td>0.79</td><td>mmol/L</td><td>1.00-2.20</td></tr>
</table>"""
        job = PaddleOcrJobResult(
            "job-1", "PaddleOCR-VL-1.6", "done", "https://result.example/job.jsonl",
            (_layout_record(markdown),),
        )
        parsed = parse_report_payload(convert_layout_job_to_report(job))
        self.assertEqual(parsed["天冬氨酸氨基转移酶"].unit, "U/L")
        self.assertEqual(parsed["天冬氨酸氨基转移酶"].report_flag.value, "high")
        self.assertEqual(parsed["白蛋白"].unit, "g/L")
        self.assertEqual(parsed["白蛋白"].report_flag.value, "low")
        self.assertEqual(parsed["估算肾小球滤过率"].unit, "ml/min/1.73m^2")
        self.assertEqual(parsed["高密度脂蛋白胆固醇"].abbreviation, "HDL-C")

    def test_vl_layout_normalizes_terms_and_extracts_method_without_changing_value(self) -> None:
        markdown = """中日友好医院 临床检验结果报告单
<table><tr><th>项目</th><th>中文名称</th><th>结果</th><th>单位</th><th>参考范围</th></tr>
<tr><td>★ALB</td><td>★白蛋白定量(BCG法)</td><td>39.7</td><td>g/L</td><td>40-55</td></tr>
<tr><td>★HDL-</td><td>★高密度脂蛋白胆固醇</td><td>0.79</td><td>mmol/L</td><td>1-2.2</td></tr>
<tr><td>*HBDH</td><td>*α-羟丁酸脱氢酶</td><td>195</td><td>U/L</td><td>76-218</td></tr>
<tr><td>ST</td><td>血清天门冬氨酸氨基转移酶</td><td>21.29</td><td>U/L</td><td>0-40</td></tr>
</table>"""
        job = PaddleOcrJobResult(
            "job-1", "PaddleOCR-VL-1.6", "done", "https://result.example/job.jsonl",
            (_layout_record(markdown),),
        )
        report = convert_layout_job_to_report(job)
        parsed = parse_report_payload(report)
        self.assertEqual(report["metadata"]["hospital"], "中日友好医院")
        self.assertEqual(parsed["白蛋白"].raw_name, "白蛋白定量")
        self.assertEqual(parsed["白蛋白"].method, "BCG法")
        self.assertEqual(parsed["高密度脂蛋白胆固醇"].abbreviation, "HDL-C")
        self.assertEqual(parsed["α-羟丁酸脱氢酶"].value, "195")
        self.assertEqual(parsed["天冬氨酸氨基转移酶"].abbreviation, "AST")

    def test_raw_ocr_may_fill_only_an_independently_matched_missing_unit(self) -> None:
        report = {
            "observations": [
                {
                    "raw_name": "尿酸", "standard_name": "尿酸", "abbreviation": "UA",
                    "value": "303.03", "unit": None,
                },
                {
                    "raw_name": "肌酐", "standard_name": "肌酐", "abbreviation": "CR",
                    "value": "50", "unit": None,
                },
            ]
        }
        lines = tuple(
            OcrLine(text, 0.99, (float(x), float(y), float(x + 60), float(y + 12)))
            for y, values in (
                (10, ("尿酸", "UA", "303.03", "μmol/L", "142-416")),
                (30, ("肌酐", "CR", "51", "μmol/L", "44-115")),
            )
            for x, text in enumerate(values, 1)
        )
        _reconcile_missing_units(report, OcrDocument("job", (OcrPage(0, lines),)))
        self.assertEqual(report["observations"][0]["unit"], "μmol/L")
        self.assertIsNone(report["observations"][1]["unit"])

    def test_layout_rejects_a_truncated_unit_ending_in_an_operator(self) -> None:
        markdown = """<table>
<tr><th>项目</th><th>中文名称</th><th>结果</th><th>单位</th><th>参考范围</th></tr>
<tr><td>RBC</td><td>红细胞计数</td><td>5.15</td><td>*10^12/</td><td>3.8-5.1</td></tr>
</table>"""
        job = PaddleOcrJobResult(
            "job-1", "PaddleOCR-VL-1.6", "done", "https://result.example/job.jsonl",
            (_layout_record(markdown),),
        )
        report = convert_layout_job_to_report(job)
        self.assertIsNone(report["observations"][0]["unit"])

    def test_vl_layout_normalizes_generic_latex_name_and_unit_variants(self) -> None:
        markdown = r"""<table>
<tr><th>项目</th><th>中文名称</th><th>结果</th><th>单位</th><th>参考范围</th></tr>
<tr><td>^{{*}}TP</td><td>^{{*}}总蛋白量</td><td>69.9</td><td>g/L</td><td>60-80</td></tr>
<tr><td>^{{*}}HBDH</td><td>^{{*}}\alpha-羟丁酸脱氢酶</td><td>195</td><td>U/L</td><td>76-218</td></tr>
<tr><td>^{{*}}\beta2-</td><td>\beta2\text{-}微球蛋白</td><td>2.75</td><td>mg/L</td><td>1-3</td></tr>
<tr><td>C_{1q}</td><td>C_{1q}循环复合物</td><td>212</td><td>mg/L</td><td>159-233</td></tr>
<tr><td>eGFR</td><td>估算肾小球滤过率</td><td>109.45</td><td>\mathrm{mL}/\mathrm{min}/1.73\mathrm{m}^{2}</td><td>公式计算</td></tr>
</table>"""
        job = PaddleOcrJobResult(
            "job-1", "PaddleOCR-VL-1.6", "done", "https://result.example/job.jsonl",
            (_layout_record(markdown),),
        )
        parsed = parse_report_payload(convert_layout_job_to_report(job))
        self.assertEqual(parsed["总蛋白"].abbreviation, "TP")
        self.assertEqual(parsed["α-羟丁酸脱氢酶"].abbreviation, "HBDH")
        self.assertEqual(parsed["β2微球蛋白"].abbreviation, "β2-MG")
        self.assertEqual(parsed["C1q循环复合物"].abbreviation, "C1q")
        self.assertEqual(parsed["估算肾小球滤过率"].unit, "mL/min/1.73m^2")

    def test_vl_layout_canonicalizes_neut_and_lym_percentage_codes(self) -> None:
        markdown = """<table>
<tr><th>项目</th><th>结果</th><th>单位</th><th>参考范围</th></tr>
<tr><td>NEUT%</td><td>76</td><td>%</td><td>40-75</td></tr>
<tr><td>LYM%</td><td>19</td><td>%</td><td>20-50</td></tr>
</table>"""
        job = PaddleOcrJobResult(
            "job-1", "PaddleOCR-VL-1.6", "done", "https://result.example/job.jsonl",
            (_layout_record(markdown),),
        )

        parsed = parse_report_payload(convert_layout_job_to_report(job))

        self.assertEqual(set(parsed), {"中性粒细胞百分数", "淋巴细胞百分数"})
        self.assertEqual(parsed["中性粒细胞百分数"].abbreviation, "NEUT")
        self.assertEqual(parsed["淋巴细胞百分数"].abbreviation, "LYM")

    def test_image_report_job_uses_layout_and_text_ocr(self) -> None:
        class FakeJobsClient:
            models = []

            def process_image(self, image, filename, model):
                self.models.append(model)
                if model == "PP-OCRv6":
                    return PaddleOcrJobResult(
                        "job-2", model, "done", "https://result.example/text.jsonl",
                        ({"result": {"ocrResults": _response()["result"]["ocrResults"]}},),
                    )
                return PaddleOcrJobResult(
                    "job-1", model, "done", "https://result.example/job.jsonl", (_vl_record(),)
                )

        client = FakeJobsClient()
        result, job = image_report_job(
            b"\x89PNG\r\n\x1a\nimage", "report.png", client=client
        )
        self.assertEqual(client.models, ["PaddleOCR-VL-1.6", "PP-OCRv6"])
        self.assertEqual(job.summary()["layout_pages"], 1)
        self.assertEqual(len(result.report["observations"]), 3)
        self.assertEqual(result.report["metadata"]["patient_sex"], "女")
        self.assertEqual(result.report["metadata"]["patient_age_years"], 30)
        self.assertEqual(result.report["metadata"]["sample_type"], "血清")
        self.assertIsNotNone(result.ocr)


if __name__ == "__main__":
    unittest.main()
