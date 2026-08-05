"""PaddleOCR AI Studio Jobs API adapter for laboratory report images.

The adapter submits one bounded image to the official asynchronous OCR jobs
endpoint, validates its JSONL result, and conservatively converts table rows to
the existing ``structured-report/v0.2`` contract. It never persists images.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
from statistics import median
import time
from typing import Any, Mapping, Sequence
from urllib import error as urlerror
from urllib import parse, request
import uuid

from .layout_grid import GridCell, block_text, layout_blocks, table_grids
from .lab_terminology import canonicalize_laboratory_term
from .report_model import resolve_report_flag


API_URL_ENV = "PADDLEOCR_OCR_API_URL"
ACCESS_TOKEN_ENV = "PADDLEOCR_ACCESS_TOKEN"
TIMEOUT_ENV = "PADDLEOCR_OCR_TIMEOUT"
JOB_URL_ENV = "PADDLEOCR_JOB_URL"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_POLL_TIMEOUT_SECONDS = 600.0
DEFAULT_JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
JOB_SUBMIT_RETRIES = 2
MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
REPORT_SCHEMA_VERSION = "structured-report/v0.2"
SUPPORTED_JOB_MODELS = {"PaddleOCR-VL-1.6", "PP-OCRv6", "PP-StructureV3"}


class PaddleOcrReportError(ValueError):
    """Raised when OCR cannot produce a trustworthy structured report."""


class PaddleOcrApiError(PaddleOcrReportError):
    """A bounded hosted-API failure that retains only status and business code."""

    def __init__(self, status: int, api_code: int | str | None = None) -> None:
        self.status = status
        self.api_code = api_code
        suffix = f" (code {api_code})" if api_code is not None else ""
        super().__init__(f"PaddleOCR jobs request failed with HTTP {status}{suffix}")


@dataclass(frozen=True, slots=True)
class OcrLine:
    text: str
    score: float | None
    box: tuple[float, float, float, float] | None

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "score": self.score, "box": list(self.box) if self.box else None}


@dataclass(frozen=True, slots=True)
class OcrPage:
    page_index: int
    lines: tuple[OcrLine, ...]

    def to_dict(self) -> dict[str, object]:
        return {"page_index": self.page_index, "lines": [line.to_dict() for line in self.lines]}


@dataclass(frozen=True, slots=True)
class OcrDocument:
    request_id: str | None
    pages: tuple[OcrPage, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "paddleocr-document/v0.1",
            "request_id": self.request_id,
            "pages": [page.to_dict() for page in self.pages],
        }


@dataclass(frozen=True, slots=True)
class ImageReportResult:
    report: Mapping[str, Any]
    ocr: OcrDocument | None

    def to_dict(self) -> dict[str, object]:
        return {"report": dict(self.report), "ocr": self.ocr.to_dict() if self.ocr else None}


class PaddleOcrClient:
    """Minimal no-proxy client for the PaddleOCR official synchronous API."""

    def __init__(
        self,
        api_url: str,
        access_token: str,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        opener: Any | None = None,
    ) -> None:
        self.api_url = _validate_api_url(api_url)
        if not access_token.strip():
            raise PaddleOcrReportError(f"{ACCESS_TOKEN_ENV} is required")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise PaddleOcrReportError("PaddleOCR timeout must be a positive finite number")
        self._access_token = access_token.strip()
        self._timeout_seconds = float(timeout_seconds)
        self._opener = opener or request.build_opener(request.ProxyHandler({}))

    @classmethod
    def from_environment(cls, *, opener: Any | None = None) -> "PaddleOcrClient":
        raw_timeout = os.environ.get(TIMEOUT_ENV, "").strip()
        try:
            timeout = float(raw_timeout) if raw_timeout else DEFAULT_TIMEOUT_SECONDS
        except ValueError as error:
            raise PaddleOcrReportError(f"{TIMEOUT_ENV} must be numeric") from error
        return cls(
            os.environ.get(API_URL_ENV, ""),
            os.environ.get(ACCESS_TOKEN_ENV, ""),
            timeout_seconds=timeout,
            opener=opener,
        )

    def recognize_image(self, image: bytes, filename: str) -> OcrDocument:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            raise PaddleOcrReportError("image must be PNG or JPEG")
        if not image or len(image) > MAX_IMAGE_BYTES:
            raise PaddleOcrReportError("image must be non-empty and at most 10 MiB")
        if not _matches_image_signature(image, suffix):
            raise PaddleOcrReportError("image content does not match its filename extension")

        payload = {
            "file": base64.b64encode(image).decode("ascii"),
            "fileType": 1,
            "useDocOrientationClassify": True,
            "useDocUnwarping": True,
            "useTextlineOrientation": True,
            "visualize": False,
        }
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        api_request = request.Request(
            self.api_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"token {self._access_token}",
                "Content-Type": "application/json",
                "Client-Platform": "medical-report-demo",
            },
        )
        try:
            with self._opener.open(api_request, timeout=self._timeout_seconds) as response:
                status = getattr(response, "status", 200)
                raw = response.read()
        except urlerror.HTTPError as error:
            raise PaddleOcrReportError(f"PaddleOCR request failed with HTTP {error.code}") from error
        except (urlerror.URLError, TimeoutError, OSError) as error:
            raise PaddleOcrReportError("PaddleOCR request failed") from error
        if status != 200:
            raise PaddleOcrReportError(f"PaddleOCR request failed with HTTP {status}")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PaddleOcrReportError("PaddleOCR returned invalid JSON") from error
        return parse_ocr_response(decoded)


@dataclass(frozen=True, slots=True)
class PaddleOcrJobResult:
    job_id: str
    model: str
    state: str
    jsonl_url: str
    records: tuple[Mapping[str, Any], ...]

    def summary(self) -> dict[str, object]:
        ocr_pages = 0
        layout_pages = 0
        text_chars = 0
        for record in self.records:
            result = record.get("result")
            if not isinstance(result, Mapping):
                continue
            raw_ocr = result.get("ocrResults")
            if isinstance(raw_ocr, list):
                ocr_pages += len(raw_ocr)
                for page in raw_ocr:
                    if isinstance(page, Mapping) and isinstance(page.get("prunedResult"), Mapping):
                        texts = page["prunedResult"].get("rec_texts", [])
                        if isinstance(texts, list):
                            text_chars += sum(len(text) for text in texts if isinstance(text, str))
            raw_layout = result.get("layoutParsingResults")
            if isinstance(raw_layout, list):
                layout_pages += len(raw_layout)
                for page in raw_layout:
                    if not isinstance(page, Mapping) or not isinstance(page.get("markdown"), Mapping):
                        continue
                    markdown = page["markdown"].get("text")
                    if isinstance(markdown, str):
                        text_chars += len(markdown)
        return {
            "job_id": self.job_id,
            "model": self.model,
            "state": self.state,
            "record_count": len(self.records),
            "ocr_pages": ocr_pages,
            "layout_pages": layout_pages,
            "text_chars": text_chars,
        }


class PaddleOcrJobsClient:
    """Client for the AI Studio asynchronous ``/api/v2/ocr/jobs`` API."""

    def __init__(
        self,
        access_token: str,
        *,
        job_url: str = DEFAULT_JOB_URL,
        request_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        poll_timeout_seconds: float = DEFAULT_POLL_TIMEOUT_SECONDS,
        poll_interval_seconds: float = 5.0,
        opener: Any | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        if not access_token.strip():
            raise PaddleOcrReportError(f"{ACCESS_TOKEN_ENV} is required")
        self.job_url = _validate_job_url(job_url)
        for name, value in (
            ("request timeout", request_timeout_seconds),
            ("poll timeout", poll_timeout_seconds),
            ("poll interval", poll_interval_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise PaddleOcrReportError(f"PaddleOCR {name} must be a positive finite number")
        self._access_token = access_token.strip()
        self._request_timeout = float(request_timeout_seconds)
        self._poll_timeout = float(poll_timeout_seconds)
        self._poll_interval = float(poll_interval_seconds)
        self._opener = opener or request.build_opener(request.ProxyHandler({}))
        self._sleep = sleep

    @classmethod
    def from_environment(cls, **kwargs: Any) -> "PaddleOcrJobsClient":
        return cls(
            os.environ.get(ACCESS_TOKEN_ENV, ""),
            job_url=os.environ.get(JOB_URL_ENV, DEFAULT_JOB_URL),
            **kwargs,
        )

    def process_url(self, file_url: str, model: str) -> PaddleOcrJobResult:
        parsed = parse.urlsplit(file_url.strip())
        if parsed.scheme != "https" or not parsed.hostname:
            raise PaddleOcrReportError("PaddleOCR input URL must use HTTPS")
        payload = {
            "fileUrl": file_url.strip(),
            "model": _validate_job_model(model),
            "optionalPayload": _job_options(model),
        }
        job_id = self._submit(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
            "application/json",
        )
        return self._poll_and_download(job_id, model)

    def process_image(self, image: bytes, filename: str, model: str) -> PaddleOcrJobResult:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_IMAGE_SUFFIXES:
            raise PaddleOcrReportError("image must be PNG or JPEG")
        if not image or len(image) > MAX_IMAGE_BYTES:
            raise PaddleOcrReportError("image must be non-empty and at most 10 MiB")
        if not _matches_image_signature(image, suffix):
            raise PaddleOcrReportError("image content does not match its filename extension")
        resolved_model = _validate_job_model(model)
        boundary = "----medical-report-" + uuid.uuid4().hex
        body = _multipart_body(
            boundary,
            {
                "model": resolved_model,
                "optionalPayload": json.dumps(_job_options(resolved_model), separators=(",", ":")),
            },
            filename,
            image,
        )
        job_id = self._submit(body, f"multipart/form-data; boundary={boundary}")
        return self._poll_and_download(job_id, resolved_model)

    def _submit(self, body: bytes, content_type: str) -> str:
        value: Mapping[str, Any] | None = None
        for attempt in range(JOB_SUBMIT_RETRIES + 1):
            try:
                value = self._json_request(
                    self.job_url, method="POST", body=body, content_type=content_type
                )
                break
            except PaddleOcrApiError as error:
                recoverable = error.status in {429, 503} or error.api_code == 10010
                if not recoverable or attempt == JOB_SUBMIT_RETRIES:
                    raise
                self._sleep(1 + attempt)
        assert value is not None
        data = value.get("data")
        job_id = data.get("jobId") if isinstance(data, Mapping) else None
        if not isinstance(job_id, str) or not job_id:
            raise PaddleOcrReportError("PaddleOCR job submission returned no jobId")
        return job_id

    def _poll_and_download(self, job_id: str, model: str) -> PaddleOcrJobResult:
        deadline = time.monotonic() + self._poll_timeout
        while time.monotonic() < deadline:
            value = self._json_request(f"{self.job_url}/{parse.quote(job_id, safe='')}", method="GET")
            data = value.get("data")
            state = data.get("state") if isinstance(data, Mapping) else None
            if state == "done":
                result_url = data.get("resultUrl")
                jsonl_url = result_url.get("jsonUrl") if isinstance(result_url, Mapping) else None
                if not isinstance(jsonl_url, str):
                    raise PaddleOcrReportError("PaddleOCR completed job has no JSONL URL")
                records = self._download_jsonl(jsonl_url)
                return PaddleOcrJobResult(job_id, model, state, jsonl_url, records)
            if state == "failed":
                raise PaddleOcrReportError("PaddleOCR job failed")
            if state not in {"pending", "running"}:
                raise PaddleOcrReportError("PaddleOCR job returned an unknown state")
            self._sleep(self._poll_interval)
        raise PaddleOcrReportError("PaddleOCR job polling timed out")

    def _download_jsonl(self, jsonl_url: str) -> tuple[Mapping[str, Any], ...]:
        parsed = parse.urlsplit(jsonl_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise PaddleOcrReportError("PaddleOCR JSONL URL must use HTTPS")
        raw = self._raw_request(jsonl_url, method="GET", include_auth=False)
        records: list[Mapping[str, Any]] = []
        try:
            for line in raw.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping) or not isinstance(value.get("result"), Mapping):
                    raise PaddleOcrReportError("PaddleOCR JSONL record is invalid")
                records.append(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PaddleOcrReportError("PaddleOCR returned invalid JSONL") from error
        if not records:
            raise PaddleOcrReportError("PaddleOCR returned empty JSONL")
        return tuple(records)

    def _json_request(
        self,
        url: str,
        *,
        method: str,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> Mapping[str, Any]:
        raw = self._raw_request(url, method=method, body=body, content_type=content_type)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PaddleOcrReportError("PaddleOCR jobs API returned invalid JSON") from error
        if not isinstance(value, Mapping):
            raise PaddleOcrReportError("PaddleOCR jobs API response must be an object")
        return value

    def _raw_request(
        self,
        url: str,
        *,
        method: str,
        body: bytes | None = None,
        content_type: str | None = None,
        include_auth: bool = True,
    ) -> bytes:
        headers = {"Authorization": f"bearer {self._access_token}"} if include_auth else {}
        if content_type:
            headers["Content-Type"] = content_type
        api_request = request.Request(url, data=body, method=method, headers=headers)
        try:
            with self._opener.open(api_request, timeout=self._request_timeout) as response:
                status = getattr(response, "status", 200)
                raw = response.read()
        except urlerror.HTTPError as error:
            try:
                api_code = _api_error_code(error.read())
            except (OSError, AttributeError):
                api_code = None
            raise PaddleOcrApiError(error.code, api_code) from error
        except (urlerror.URLError, TimeoutError, OSError) as error:
            raise PaddleOcrReportError("PaddleOCR jobs request failed") from error
        if status != 200:
            raise PaddleOcrApiError(status, _api_error_code(raw))
        return raw


def _api_error_code(raw: bytes) -> int | str | None:
    if not raw or len(raw) > 64 * 1024:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    for key in ("code", "errorCode", "error_code"):
        code = value.get(key)
        if isinstance(code, int) and not isinstance(code, bool):
            return code
        if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", code):
            return code
    return None


def parse_ocr_response(value: object) -> OcrDocument:
    if not isinstance(value, Mapping):
        raise PaddleOcrReportError("PaddleOCR response must be an object")
    error_code = value.get("errorCode", 0)
    if error_code != 0:
        raise PaddleOcrReportError(f"PaddleOCR API error {error_code}")
    result = value.get("result")
    if not isinstance(result, Mapping):
        raise PaddleOcrReportError("PaddleOCR response has no result object")
    raw_pages = result.get("ocrResults")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise PaddleOcrReportError("PaddleOCR response has no OCR pages")

    pages: list[OcrPage] = []
    for page_index, raw_page in enumerate(raw_pages):
        if not isinstance(raw_page, Mapping) or not isinstance(raw_page.get("prunedResult"), Mapping):
            raise PaddleOcrReportError("PaddleOCR page result is invalid")
        pruned = raw_page["prunedResult"]
        texts = pruned.get("rec_texts")
        scores = pruned.get("rec_scores")
        boxes = pruned.get("rec_boxes")
        if not isinstance(texts, list) or not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise PaddleOcrReportError("PaddleOCR rec_texts is invalid")
        if scores is not None and (
            not isinstance(scores, list)
            or len(scores) != len(texts)
            or any(isinstance(score, bool) or not isinstance(score, (int, float)) for score in scores)
        ):
            raise PaddleOcrReportError("PaddleOCR rec_scores is invalid")
        if boxes is not None and (
            not isinstance(boxes, list)
            or len(boxes) != len(texts)
            or any(not _valid_box(box) for box in boxes)
        ):
            raise PaddleOcrReportError("PaddleOCR rec_boxes is invalid")
        lines = tuple(
            OcrLine(
                text=text.strip(),
                score=float(scores[index]) if scores is not None else None,
                box=tuple(float(number) for number in boxes[index]) if boxes is not None else None,
            )
            for index, text in enumerate(texts)
        )
        pages.append(OcrPage(page_index, lines))
    request_id = value.get("logId") or value.get("jobId")
    return OcrDocument(str(request_id) if request_id is not None else None, tuple(pages))


def convert_ocr_to_report(document: OcrDocument) -> dict[str, Any]:
    all_lines = [line.text for page in document.pages for line in page.lines]
    metadata = _metadata(all_lines)
    observations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in document.pages:
        for row in _rows(page.lines):
            observation = _observation(row)
            if observation is None:
                continue
            identifier = str(observation["standard_name"])
            if identifier in seen:
                continue
            seen.add(identifier)
            observations.append(observation)
    if not observations:
        raise PaddleOcrReportError("no laboratory observation rows could be identified")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metadata": metadata,
        "observations": observations,
    }


def image_report(image: bytes, filename: str, *, client: PaddleOcrClient | None = None) -> ImageReportResult:
    resolved = client or PaddleOcrClient.from_environment()
    document = resolved.recognize_image(image, filename)
    return ImageReportResult(convert_ocr_to_report(document), document)


def ocr_document_from_job(result: PaddleOcrJobResult) -> OcrDocument:
    """Convert a completed PP-OCRv6 job into the bounded OCR document contract."""
    if result.state != "done" or result.model != "PP-OCRv6":
        raise PaddleOcrReportError("structured report conversion requires a completed PP-OCRv6 job")
    pages: list[object] = []
    for record in result.records:
        raw_result = record.get("result")
        raw_pages = raw_result.get("ocrResults") if isinstance(raw_result, Mapping) else None
        if isinstance(raw_pages, list):
            pages.extend(raw_pages)
    return parse_ocr_response({
        "jobId": result.job_id,
        "errorCode": 0,
        "result": {"ocrResults": pages},
    })


def image_report_job(
    image: bytes,
    filename: str,
    *,
    client: PaddleOcrJobsClient | None = None,
) -> tuple[ImageReportResult, PaddleOcrJobResult]:
    """Use layout OCR for the table and raw OCR for independent metadata text."""
    resolved = client or PaddleOcrJobsClient.from_environment()
    layout_job = resolved.process_image(image, filename, "PaddleOCR-VL-1.6")
    text_job = resolved.process_image(image, filename, "PP-OCRv6")
    document = ocr_document_from_job(text_job)
    report = convert_layout_job_to_report(layout_job)
    _reconcile_missing_units(report, document)
    text_metadata = _metadata(
        [line.text for page in document.pages for line in page.lines]
    )
    report["metadata"] = {**text_metadata, **report["metadata"]}
    return ImageReportResult(report, document), layout_job


def _reconcile_missing_units(report: dict[str, Any], document: OcrDocument) -> None:
    """Fill a missing layout unit only when raw OCR independently agrees on metric and value."""
    try:
        text_report = convert_ocr_to_report(document)
    except PaddleOcrReportError:
        return
    candidates = text_report.get("observations")
    if not isinstance(candidates, list):
        return
    for observation in report.get("observations", []):
        if not isinstance(observation, dict) or observation.get("unit"):
            continue
        matches: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or candidate.get("value") != observation.get("value"):
                continue
            same_code = (
                isinstance(observation.get("abbreviation"), str)
                and observation.get("abbreviation") == candidate.get("abbreviation")
            )
            same_name = observation.get("standard_name") == candidate.get("standard_name")
            if not (same_code or same_name):
                continue
            unit = _validated_layout_unit(str(candidate.get("unit") or ""))
            if unit:
                matches.append(unit)
        distinct = set(matches)
        if len(distinct) == 1:
            observation["unit"] = distinct.pop()


_TABLE_HEADERS = {
    "code": {"项目代号", "项目代码", "缩写", "英文缩写", "英文", "代号"},
    "name": {"项目名称", "中文名称", "检验项目", "检测项目"},
    "value": {"结果", "检验结果", "检测结果"},
    "flag": {"标志", "提示", "异常标志"},
    "unit": {"单位"},
    "range": {"参考值", "参考范围", "参考区间"},
    "method": {"方法", "检验方法", "检测方法"},
}


def convert_layout_job_to_report(result: PaddleOcrJobResult) -> dict[str, Any]:
    """Convert normalized PaddleOCR-VL table grids to the report contract."""
    if result.state != "done" or result.model != "PaddleOCR-VL-1.6":
        raise PaddleOcrReportError("layout report conversion requires a completed PaddleOCR-VL-1.6 job")
    blocks = layout_blocks(result.records)
    if not blocks:
        raise PaddleOcrReportError("PaddleOCR-VL returned no structured layout blocks")
    try:
        grids = table_grids(blocks)
    except ValueError as error:
        raise PaddleOcrReportError("PaddleOCR-VL returned an invalid table grid") from error
    observations: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_sources: set[tuple[str, ...]] = set()
    for grid in grids:
        header_index, groups = _table_schema(grid.rows)
        if header_index is None:
            continue
        for columns in groups:
            for row in grid.rows[header_index + 1:]:
                source_signature = tuple(
                    row[position].source_ref
                    for field, position in sorted(columns.items())
                    if field in {"code", "name", "value", "flag", "unit", "range"}
                    and position < len(row)
                )
                if source_signature in seen_sources:
                    continue
                observation = _table_observation(row, columns)
                if observation is None:
                    continue
                seen_sources.add(source_signature)
                identifier = str(observation["standard_name"])
                if identifier in seen:
                    raise PaddleOcrReportError("PaddleOCR-VL table contains duplicate observation names")
                seen.add(identifier)
                observations.append(observation)
    if not observations:
        raise PaddleOcrReportError("no laboratory observation table could be identified")
    metadata_lines = list(block_text(blocks))
    metadata_lines.extend(
        cell.text for grid in grids for row in grid.rows for cell in row if cell.text
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metadata": _metadata(metadata_lines),
        "observations": observations,
    }


def _table_schema(
    rows: Sequence[Sequence[GridCell]],
) -> tuple[int | None, tuple[dict[str, int], ...]]:
    for index, row in enumerate(rows):
        groups = _column_groups(row)
        if groups:
            return index, groups
    return None, ()


def _column_groups(header: Sequence[GridCell]) -> tuple[dict[str, int], ...]:
    normalized = [re.sub(r"[\s:：]", "", cell.text) for cell in header]
    roles: list[str | None] = []
    for index, text in enumerate(normalized):
        if text == "项目":
            role = "code" if index + 1 < len(normalized) and normalized[index + 1] in _TABLE_HEADERS["name"] else "name"
        else:
            role = next((field for field, aliases in _TABLE_HEADERS.items() if text in aliases), None)
        roles.append(role)
    names = [index for index, role in enumerate(roles) if role == "name"]
    starts = [name - 1 if name and roles[name - 1] == "code" else name for name in names]
    groups: list[dict[str, int]] = []
    for group_index, start in enumerate(starts):
        end = starts[group_index + 1] if group_index + 1 < len(starts) else len(header)
        columns: dict[str, int] = {}
        for position in range(start, end):
            role = roles[position]
            if role is not None:
                columns.setdefault(role, position)
        if {"name", "value", "range"} <= set(columns):
            groups.append(columns)
    return tuple(groups)


def _table_observation(
    row: Sequence[GridCell], columns: Mapping[str, int]
) -> dict[str, Any] | None:
    def cell(field: str) -> str:
        index = columns.get(field)
        return row[index].text.strip() if index is not None and index < len(row) else ""

    raw_name = re.sub(r"^(?:[★*]\s*)?(?:\d+\s+)?", "", cell("name")).strip()
    trailing_code = re.search(r"\s+([A-Za-z][A-Za-z0-9%/'._+-]{0,19})$", raw_name)
    if trailing_code:
        code = trailing_code.group(1)
        name = raw_name[: trailing_code.start()].strip()
    else:
        code = _strip_layout_marker(_normalize_layout_text(cell("code")))
        name = raw_name
    name = _strip_layout_marker(_normalize_layout_text(name))
    name, method_from_name = _split_method(name)
    if not _CODE.fullmatch(code) or code.endswith(("-", "+", "/", ".")):
        code = ""
    value_text = cell("value").replace(r"\uparrow", "↑").replace(r"\downarrow", "↓")
    value_match = _VALUE.fullmatch(value_text)
    raw_range = cell("range")
    bounds = _reference_bounds(raw_range)
    if not name or value_match is None:
        return None
    if bounds is None:
        range_shape = _normalize_reference_text(raw_range)
        if range_shape and re.fullmatch(r"[\[\]()<>≤≥=+\-\d.]+", range_shape):
            raise PaddleOcrReportError("PaddleOCR-VL returned an invalid laboratory table row")
        lower, upper = None, None
    else:
        lower, upper = bounds
    raw_unit = cell("unit")
    if r"\uparrow" in raw_unit or "↑" in raw_unit:
        unit_flag = "H"
    elif r"\downarrow" in raw_unit or "↓" in raw_unit:
        unit_flag = "L"
    else:
        unit_flag = None
    raw_unit = (
        raw_unit.replace(r"\uparrow", "")
        .replace(r"\downarrow", "")
        .replace("↑", "")
        .replace("↓", "")
    )
    explicit_flag = re.sub(r"[\\\s]", "", cell("flag")).lower()
    if explicit_flag in {"↑", "uparrow", "high", "h"}:
        source_flag = "H"
    elif explicit_flag in {"↓", "downarrow", "low", "l"}:
        source_flag = "L"
    else:
        source_flag = unit_flag or value_match.group("flag")
    value = value_match.group("value")
    unit = _validated_layout_unit(raw_unit)
    if unit is None and not _normalize_layout_text(raw_unit):
        unit: str | None = "-" if code.upper() in {"A/G", "AST/ALT"} else None
    report_flag = resolve_report_flag(value, lower, upper, source_flag) if source_flag or lower is not None or upper is not None else None
    standard_name, abbreviation = canonicalize_laboratory_term(name, code or None)
    return {
        "raw_name": name,
        "standard_name": standard_name,
        "abbreviation": abbreviation,
        "value": value,
        "unit": unit,
        "reference_interval": {"lower": lower, "upper": upper},
        "report_flag": report_flag,
        "method": _normalize_layout_text(cell("method")) or method_from_name,
    }


def _split_method(name: str) -> tuple[str, str | None]:
    match = re.fullmatch(r"(.+?)[(（]([^()（）]{1,30}法)[)）]", name)
    if match is None:
        return name, None
    return match.group(1), match.group(2)


def _validate_api_url(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise PaddleOcrReportError(f"{API_URL_ENV} is required")
    parsed = parse.urlsplit(raw)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.path.rstrip("/").endswith("/ocr")
    ):
        raise PaddleOcrReportError(f"{API_URL_ENV} must be an HTTPS endpoint ending with /ocr")
    return raw


def _validate_job_url(value: str) -> str:
    raw = value.strip().rstrip("/")
    parsed = parse.urlsplit(raw)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith("/api/v2/ocr/jobs")
    ):
        raise PaddleOcrReportError(f"{JOB_URL_ENV} must be an HTTPS /api/v2/ocr/jobs endpoint")
    return raw


def _validate_job_model(model: str) -> str:
    if model not in SUPPORTED_JOB_MODELS:
        raise PaddleOcrReportError("unsupported PaddleOCR jobs model")
    return model


def _job_options(model: str) -> dict[str, bool]:
    options = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
    }
    if model == "PP-OCRv6":
        options["useTextlineOrientation"] = False
    else:
        options["useChartRecognition"] = False
    return options


def _multipart_body(
    boundary: str,
    fields: Mapping[str, str],
    filename: str,
    image: bytes,
) -> bytes:
    if any(character in filename for character in ('"', "\r", "\n")):
        raise PaddleOcrReportError("image filename is invalid")
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode("utf-8"),
            b"\r\n",
        ])
    content_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
    chunks.extend([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        image,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return b"".join(chunks)


def _matches_image_signature(image: bytes, suffix: str) -> bool:
    if suffix == ".png":
        return image.startswith(b"\x89PNG\r\n\x1a\n")
    return image.startswith(b"\xff\xd8\xff")


def _valid_box(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 4
        and all(not isinstance(number, bool) and isinstance(number, (int, float)) for number in value)
        and value[0] <= value[2]
        and value[1] <= value[3]
    )


def _rows(lines: Sequence[OcrLine]) -> list[list[str]]:
    positioned = [line for line in lines if line.box is not None]
    if len(positioned) != len(lines):
        return [[part for part in re.split(r"\s+", line.text) if part] for line in lines]
    heights = [line.box[3] - line.box[1] for line in positioned if line.box and line.box[3] > line.box[1]]
    tolerance = max(4.0, (median(heights) if heights else 10.0) * 0.65)
    groups: list[list[OcrLine]] = []
    centers: list[float] = []
    for line in sorted(positioned, key=lambda item: ((item.box[1] + item.box[3]) / 2, item.box[0])):  # type: ignore[index]
        center = (line.box[1] + line.box[3]) / 2  # type: ignore[index]
        if groups and abs(center - centers[-1]) <= tolerance:
            groups[-1].append(line)
            centers[-1] = sum((item.box[1] + item.box[3]) / 2 for item in groups[-1] if item.box) / len(groups[-1])
        else:
            groups.append([line])
            centers.append(center)
    return [[item.text for item in sorted(group, key=lambda line: line.box[0])] for group in groups]  # type: ignore[index]


_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_FULL_RANGE = re.compile(rf"^\s*(?P<lower>{_NUMBER})\s*(?:-|~|—|–|至)\s*(?P<upper>{_NUMBER})\s*$")
_UPPER_RANGE = re.compile(rf"^\s*(?:<|<=|≤)\s*(?P<upper>{_NUMBER})\s*$")
_LOWER_RANGE = re.compile(rf"^\s*(?:>|>=|≥)\s*(?P<lower>{_NUMBER})\s*$")
_VALUE = re.compile(rf"^\s*(?P<value>{_NUMBER})\s*(?P<flag>↑|↓|H|L)?\s*$", re.IGNORECASE)
_CODE = re.compile(r"^[A-Za-z0-9βγ][A-Za-z0-9βγ%/'._+-]{0,19}$")


def _normalize_layout_text(value: str) -> str:
    normalized = value.replace("$", "")
    for source, target in ((r"\mu", "μ"), (r"\gamma", "γ"), (r"\alpha", "α"), (r"\beta", "β")):
        normalized = normalized.replace(source, target)
    normalized = re.sub(
        r"\\(?:text|mathrm|mathbf|operatorname)\{([^{}]*)\}", r"\1", normalized
    )
    normalized = re.sub(r"_\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\^\{([+-]?\d+)\}", r"^\1", normalized)
    normalized = re.sub(r"\^\{\{([*★])\}\}", r"\1", normalized)
    normalized = normalized.replace(r"\cdot", "*")
    return "".join(normalized.split())


def _strip_layout_marker(value: str) -> str:
    return re.sub(r"^[★*]+", "", value).strip()


def _validated_layout_unit(value: str) -> str | None:
    unit = _normalize_layout_text(value)
    if not unit or unit.upper() == "NULL":
        return None
    if unit.endswith(("/", "*", "^", "+", "-")):
        return None
    if re.fullmatch(r"[A-Za-zμ%0-9/().*^+\-²³]+", unit) and len(unit) <= 40:
        return unit
    return None


def _normalize_reference_text(value: str) -> str:
    normalized = _normalize_layout_text(value).replace("−", "-")
    normalized = normalized.replace(r"\geq", ">=").replace(r"\leq", "<=")
    pairs = (("[", "]"), ("【", "】"), ("(", ")"))
    for opening, closing in pairs:
        if normalized.startswith(opening) and normalized.endswith(closing):
            normalized = normalized[1:-1]
            break
    return normalized


def _reference_bounds(value: str) -> tuple[str | None, str | None] | None:
    normalized = _normalize_reference_text(value)
    match = _FULL_RANGE.fullmatch(normalized)
    if match:
        lower = match.group("lower")
        upper = match.group("upper")
        try:
            low_number = Decimal(lower)
            high_number = Decimal(upper)
            if low_number > high_number and "--" in normalized and upper.startswith("-"):
                corrected = upper[1:]
                if low_number <= Decimal(corrected):
                    upper = corrected
                    high_number = Decimal(corrected)
            if low_number > high_number:
                return None
        except InvalidOperation:
            return None
        return lower, upper
    match = _UPPER_RANGE.fullmatch(normalized)
    if match:
        return None, match.group("upper")
    match = _LOWER_RANGE.fullmatch(normalized)
    if match:
        return match.group("lower"), None
    return None


def _observation(cells: Sequence[str]) -> dict[str, Any] | None:
    normalized = [cell.strip() for cell in cells if cell.strip()]
    if len(normalized) < 3 or any(keyword in "".join(normalized) for keyword in ("参考范围", "检验项目", "检测项目")):
        return None
    range_index = -1
    lower: str | None = None
    upper: str | None = None
    for index, cell in enumerate(normalized):
        bounds = _reference_bounds(cell)
        if bounds is not None:
            range_index, (lower, upper) = index, bounds
            break
    if range_index < 0:
        return None

    value_index = -1
    value: str | None = None
    source_flag: str | None = None
    for index in range(range_index - 1, -1, -1):
        match = _VALUE.match(normalized[index])
        if match:
            value_index = index
            value = match.group("value")
            source_flag = match.group("flag")
            break
    if value_index <= 0 or value is None:
        return None
    for cell in normalized[value_index + 1 :]:
        if cell in {"↑", "H", "high"}:
            source_flag = "H"
        elif cell in {"↓", "L", "low"}:
            source_flag = "L"

    names = [cell for cell in normalized[:value_index] if not cell.isdigit()]
    if not names:
        return None
    abbreviation = names[-1] if len(names) > 1 and _CODE.fullmatch(names[-1]) else None
    raw_name = "".join(names[:-1] if abbreviation else names).strip()
    if not raw_name or _VALUE.fullmatch(raw_name):
        return None
    unit_cells = [cell for cell in normalized[value_index + 1 : range_index] if cell not in {"↑", "↓", "H", "L", "high", "low"}]
    unit = " ".join(unit_cells) or None
    report_flag = resolve_report_flag(value, lower, upper, source_flag)
    standard_name, normalized_abbreviation = canonicalize_laboratory_term(
        raw_name, abbreviation
    )
    return {
        "raw_name": raw_name,
        "standard_name": standard_name,
        "abbreviation": normalized_abbreviation,
        "value": value,
        "unit": unit,
        "reference_interval": {"lower": lower, "upper": upper},
        "report_flag": report_flag,
    }


def _metadata(lines: Sequence[str]) -> dict[str, object]:
    joined = "\n".join(lines)
    metadata: dict[str, object] = {}
    hospital = next((line.strip() for line in lines if "医院" in line and len(line.strip()) <= 100), None)
    if hospital:
        cleaned_hospital = re.sub(r"\s*(?:临床)?检验(?:结果)?报告单.*$", "", hospital).strip()
        metadata["hospital"] = cleaned_hospital or hospital
    date_match = re.search(r"(20\d{2})[年./-](\d{1,2})[月./-](\d{1,2})日?", joined)
    if date_match:
        try:
            metadata["report_date"] = datetime(
                int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
            ).date().isoformat()
        except ValueError:
            pass
    sex_match = re.search(r"性别\s*[:：]?\s*(男|女)", joined)
    if sex_match:
        metadata["patient_sex"] = sex_match.group(1)
    age_match = re.search(r"年龄\s*[:：]?\s*(\d{1,3})", joined)
    if age_match and 0 < int(age_match.group(1)) < 130:
        metadata["patient_age_years"] = int(age_match.group(1))
    sample_match = re.search(r"(?:样本类型|标本类型|标本种类)\s*[:：]?\s*([^\s,，;；]{1,20})", joined)
    if sample_match:
        metadata["sample_type"] = sample_match.group(1)
    department_match = re.search(r"(?:科别|科室)\s*[:：]?\s*([^\s,，;；<]{1,30})", joined)
    if department_match:
        metadata["department"] = department_match.group(1)
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result, _job = image_report_job(args.image.read_bytes(), args.image.name)
        args.output.write_text(
            json.dumps(result.report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, PaddleOcrReportError) as error:
        parser.error(str(error))
    print(args.output)


if __name__ == "__main__":
    main()
