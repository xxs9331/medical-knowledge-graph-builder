"""Extract Chapter 01 indicator rules with overlapping page-owned windows."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable
from urllib import error as urlerror, request

from medical_kg_sourceprep.rules.indicator_rule_functions import (
    MAX_RULES_PER_PAGE,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    VALIDATOR_VERSION,
    WINDOW_POLICY_VERSION,
    RuleWindow,
    audit_candidates,
    build_rule_prompt,
    build_rule_windows,
    prompt_template_document,
    stable_candidates,
    validate_rule_response,
)
from medical_kg_sourceprep.extraction.artifacts import (
    atomic_write_text as _atomic_write_text,
    load_json as _load_json,
    sha256_path as _sha,
)
from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk, atomic_write_json, load_chunk_manifest


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
RUN_VERSION = "chapter-indicator-rule-functions-run/v0.1"
CHECKPOINT_VERSION = "chapter-indicator-rule-functions-checkpoint/v0.1"
PROBE_PAGE_INDEXES = (2, 4)  # Printed pages 6 and 8.
MAX_TOKENS = 8192
TIMEOUT_SECONDS = 120
RETRY_DELAYS_SECONDS = (0, 2, 8, 20)
_JSON_FENCE = re.compile(r"^```(?:json)?\s*(?P<body>[\s\S]*?)\s*```$", re.IGNORECASE)


def _redact(value: Exception, key: str) -> str:
    message = f"{type(value).__name__}: {value}"
    if key:
        message = message.replace(key, "[REDACTED]")
    return message[:300]


def _provider_post(
    key: str,
    prompt: str,
    *,
    max_tokens: int = MAX_TOKENS,
    timeout_seconds: int = TIMEOUT_SECONDS,
) -> tuple[list[Any], dict[str, Any]]:
    """Call the official endpoint directly and require a top-level JSON array."""
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "messages": [
            {"role": "system", "content": "只返回合法 JSON 数组，不使用外部知识。"},
            {"role": "user", "content": prompt},
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False).encode()
    req = request.Request(
        ENDPOINT,
        data=encoded,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "medical-kg-sourceprep/indicator-rules-v0.1",
        },
    )
    opener = request.build_opener(request.ProxyHandler({}))
    last_error: Exception | None = None
    for attempt, delay in enumerate(RETRY_DELAYS_SECONDS, 1):
        if delay:
            time.sleep(delay)
        try:
            with opener.open(req, timeout=timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            choice = decoded.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("empty provider content")
            reasoning_content = message.get("reasoning_content")
            reasoning_tokens = (
                decoded.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens")
            )
            if reasoning_content not in (None, "") or reasoning_tokens not in (None, 0):
                raise RuntimeError("provider returned thinking output despite thinking.type=disabled")
            format_wrapper = None
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                fenced = _JSON_FENCE.fullmatch(content.strip())
                if fenced is None:
                    raise
                parsed = json.loads(fenced.group("body"))
                format_wrapper = "markdown_fence"
            if not isinstance(parsed, list):
                raise RuntimeError("provider content is not a top-level JSON array")
            return parsed, {
                "finish_reason": choice.get("finish_reason"),
                "reasoning_content": None,
                "reasoning_tokens": reasoning_tokens,
                "attempts": attempt,
                "format_wrapper": format_wrapper,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
                "usage": {
                    name: decoded.get("usage", {}).get(name)
                    for name in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
            }
        except urlerror.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 501, 502, 503, 504}:
                raise
        except (urlerror.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
        if attempt == len(RETRY_DELAYS_SECONDS):
            break
    raise RuntimeError(f"provider retries exhausted: {last_error}")


ProviderPost = Callable[[str, str], tuple[list[Any], dict[str, Any]]]


def _extract_window(
    window: RuleWindow,
    library: dict[str, Any],
    key: str,
    provider_post: ProviderPost,
) -> dict[str, Any]:
    raw, provider = provider_post(key, build_rule_prompt(window, library))
    output = validate_rule_response(raw, window, library)
    return {
        "status": "success",
        "target_page_id": window.target_page_id,
        "target_page_index": window.target_page_index,
        "window_chunk_ids": [segment.chunk.chunk_id for segment in window.segments],
        "provider": provider,
        "raw": raw,
        "output": output,
    }


def _checkpoint(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    value = _load_json(path) if path.exists() else {}
    drift = sorted(
        name for name, expected in identity.items()
        if name in value and value.get(name) != expected
    )
    if drift:
        raise RuntimeError(f"checkpoint identity drift: {','.join(drift)}")
    value.update(identity)
    value.setdefault("pages", {})
    return value


def _revalidate_checkpoint(
    path: Path,
    identity: dict[str, Any],
    windows_by_page: dict[str, RuleWindow],
    library: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError("checkpoint is required for local revalidation")
    value = _load_json(path)
    allowed_drift = {"validator_version"}
    drift = sorted(
        name for name, expected in identity.items()
        if name in value and value.get(name) != expected and name not in allowed_drift
    )
    if drift:
        raise RuntimeError(f"checkpoint identity drift cannot be revalidated: {','.join(drift)}")
    for page_id, page in value.get("pages", {}).items():
        if page.get("status") != "success" or not isinstance(page.get("raw"), list):
            continue
        page["output"] = validate_rule_response(page["raw"], windows_by_page[page_id], library)
        page["revalidated_from"] = value.get("validator_version")
    value.update(identity)
    atomic_write_json(path, value)
    return value


def _run_windows(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    windows: list[RuleWindow],
    library: dict[str, Any],
    key: str,
    workers: int,
    provider_post: ProviderPost,
) -> None:
    pending = [
        window for window in windows
        if checkpoint["pages"].get(window.target_page_id, {}).get("status") != "success"
    ]
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_extract_window, window, library, key, provider_post): window
            for window in pending
        }
        for future in as_completed(futures):
            window = futures[future]
            try:
                checkpoint["pages"][window.target_page_id] = future.result()
            except Exception as exc:
                checkpoint["pages"][window.target_page_id] = {
                    "status": "failed",
                    "target_page_id": window.target_page_id,
                    "target_page_index": window.target_page_index,
                    "window_chunk_ids": [segment.chunk.chunk_id for segment in window.segments],
                    "error": _redact(exc, key),
                }
            atomic_write_json(checkpoint_path, checkpoint)


def _successful_packages(
    checkpoint: dict[str, Any],
    page_ids: list[str],
) -> list[dict[str, Any]]:
    return [
        checkpoint["pages"][page_id]["output"]
        for page_id in page_ids
        if checkpoint["pages"].get(page_id, {}).get("status") == "success"
    ]


def _provider_summary(checkpoint: dict[str, Any], page_ids: list[str]) -> dict[str, Any]:
    metadata = [
        checkpoint["pages"][page_id]["provider"]
        for page_id in page_ids
        if checkpoint["pages"].get(page_id, {}).get("status") == "success"
    ]
    usage = {
        name: sum(item.get("usage", {}).get(name) or 0 for item in metadata)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    return {
        "calls": len(metadata),
        "response_wrappers": dict(Counter(
            item.get("format_wrapper") or "none" for item in metadata
        )),
        "attempts": dict(Counter(str(item.get("attempts")) for item in metadata)),
        "reasoning_outputs": sum(
            item.get("reasoning_content") is not None or item.get("reasoning_tokens") not in (None, 0)
            for item in metadata
        ),
        "usage": usage,
    }


def _write_full_artifacts(
    output: Path,
    manifest: dict[str, Any],
    checkpoint: dict[str, Any],
    chunks: tuple[EvidenceChunk, ...],
    library: dict[str, Any],
    library_path: Path,
    chunks_manifest: Path,
    workers: int,
) -> dict[str, Any]:
    page_ids = [page["page_id"] for page in manifest["pages"]]
    packages = _successful_packages(checkpoint, page_ids)
    candidates = stable_candidates(packages)
    rejections = [item for package in packages for item in package["rejections"]]
    audit = audit_candidates(candidates, chunks)
    failed_pages = [
        page_id for page_id in page_ids
        if checkpoint["pages"].get(page_id, {}).get("status") != "success"
    ]
    status = "complete" if not failed_pages else "partial_failure"
    rules = [candidate["rule"] for candidate in candidates]
    atomic_write_json(output / "rules.json", rules)  # type: ignore[arg-type]
    extraction = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate-only", "hold": True, "approved": 0,
        "pages": len(page_ids) - len(failed_pages),
        "packages": packages,
        "candidates": candidates,
        "rejections": rejections,
        "audit": audit,
    }
    atomic_write_json(output / "rule-extraction.json", extraction)
    by_reason = Counter(item["reason_code"] for item in rejections)
    atomic_write_json(output / "review-queue.json", {
        "schema_version": "indicator-rule-review-queue/v0.1",
        "status": "HOLD", "approved": 0,
        "items": rejections,
        "counts": {"review_required": len(rejections), "by_reason": dict(by_reason)},
    })
    atomic_write_json(output / "gold-template.json", {
        "schema_version": "indicator-rule-gold-template/v0.1",
        "status": "HOLD", "generated_from_model": False,
        "pages": [{
            "page_id": page_id,
            "gold_status": "unreviewed",
            "rules": [], "reviewer": None, "reviewed_at": None,
        } for page_id in page_ids],
    })
    provider = _provider_summary(checkpoint, page_ids)
    quality_audit = {
        "schema_version": "indicator-rule-quality-audit/v0.1",
        "status": "HOLD",
        "candidate_status": "candidate-only",
        "approved": 0,
        "scope": {
            "chapter_id": manifest["chapter_id"],
            "pages": manifest["page_count"],
            "chunks": manifest["chunk_count"],
            "window_policy": WINDOW_POLICY_VERSION,
        },
        "evidence_replay": audit,
        "accepted_candidates": [{
            "candidate_id": candidate["candidate_id"],
            "page_id": candidate["page_id"],
            "rule_expression": candidate["rule"]["rule_expression"],
            "has_formula": candidate["rule"]["formula"] is not None,
            "output_catalog_match": candidate["output_catalog_match"],
            "source_chunk_ids": candidate["source"]["source_chunk_ids"],
        } for candidate in candidates],
        "rejections": {
            "count": len(rejections),
            "by_reason": dict(by_reason),
        },
        "provider_format": {
            "response_wrappers": provider["response_wrappers"],
            "attempts": provider["attempts"],
            "reasoning_outputs": provider["reasoning_outputs"],
        },
        "human_review": {
            "required": True,
            "gold_status": "unreviewed",
            "precision_recall_f1_reported": False,
        },
    }
    atomic_write_json(output / "quality-audit.json", quality_audit)
    run_manifest = {
        "schema_version": RUN_VERSION,
        "status": status,
        "candidate_status": "candidate-only", "hold": True, "approved": 0,
        "provider": "deepseek-direct", "model": MODEL,
        "configuration": {
            "temperature": 0, "thinking": "disabled", "trust_env": False,
            "fixed_ip": False, "max_workers": workers,
            "max_rules_per_page": MAX_RULES_PER_PAGE,
            "window_policy": WINDOW_POLICY_VERSION,
            "prompt_version": PROMPT_VERSION,
            "validator_version": VALIDATOR_VERSION,
        },
        "input": {
            "chapter_id": manifest["chapter_id"],
            "pages": manifest["page_count"], "chunks": manifest["chunk_count"],
            "chunk_manifest_sha256": _sha(chunks_manifest),
            "indicator_library_sha256": _sha(library_path),
            "indicator_catalog_sha256": library["catalog_sha256"],
        },
        "counts": {
            "pages_success": len(page_ids) - len(failed_pages),
            "pages_failed": len(failed_pages),
            "rules": len(candidates), "rejections": len(rejections),
            "formula_rules": sum(candidate["rule"]["formula"] is not None for candidate in candidates),
            "output_catalog_matches": sum(candidate["output_catalog_match"] for candidate in candidates),
            "output_catalog_unmatched": sum(not candidate["output_catalog_match"] for candidate in candidates),
        },
        "failed_pages": failed_pages,
        "provider_summary": provider,
        "audit": audit,
    }
    atomic_write_json(output / "run-manifest.json", run_manifest)
    run_manifest["artifacts"] = {
        "rules_sha256": _sha(output / "rules.json"),
        "extraction_sha256": _sha(output / "rule-extraction.json"),
        "review_queue_sha256": _sha(output / "review-queue.json"),
        "quality_audit_sha256": _sha(output / "quality-audit.json"),
        "prompt_template_sha256": _sha(output / "prompt-template.txt"),
    }
    atomic_write_json(output / "run-manifest.json", run_manifest)
    return run_manifest


def run(
    chunks_manifest: Path,
    library_path: Path,
    output: Path,
    key: str,
    *,
    workers: int = 4,
    probe: bool = False,
    revalidate: bool = False,
    provider_post: ProviderPost = _provider_post,
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be from 1 to 8")
    manifest, chunks = load_chunk_manifest(chunks_manifest)
    if manifest.get("chapter_id") != "chapter-01" or manifest.get("page_count") != 24 or manifest.get("chunk_count") != 44:
        raise RuntimeError("rule extraction is limited to Chapter 01 with 24 pages and 44 chunks")
    library = _load_json(library_path)
    if library.get("schema_version") != "indicator-library/v0.1" or library.get("indicator_count") != 50:
        raise RuntimeError("rule extraction requires the frozen 50-candidate indicator library")
    windows_by_page = build_rule_windows(chunks)
    page_ids = [page["page_id"] for page in manifest["pages"]]
    windows = [windows_by_page[page_id] for page_id in page_ids]
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(output / "prompt-template.txt", prompt_template_document())
    identity = {
        "schema_version": CHECKPOINT_VERSION,
        "chunk_manifest_sha256": _sha(chunks_manifest),
        "indicator_library_sha256": _sha(library_path),
        "indicator_catalog_sha256": library["catalog_sha256"],
        "model": MODEL, "provider": "deepseek-direct",
        "temperature": 0, "thinking": "disabled", "trust_env": False,
        "fixed_ip": False, "configured_workers": workers,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "window_policy": WINDOW_POLICY_VERSION,
        "max_rules_per_page": MAX_RULES_PER_PAGE,
        "max_tokens": MAX_TOKENS,
    }
    checkpoint_path = output / "checkpoint.json"
    checkpoint = (
        _revalidate_checkpoint(checkpoint_path, identity, windows_by_page, library)
        if revalidate else _checkpoint(checkpoint_path, identity)
    )
    if probe:
        selected = [window for window in windows if window.target_page_index in PROBE_PAGE_INDEXES]
        _run_windows(checkpoint_path, checkpoint, selected, library, key, 1, provider_post)
        selected_ids = [window.target_page_id for window in selected]
        packages = _successful_packages(checkpoint, selected_ids)
        candidates = stable_candidates(packages)
        failed = [
            page_id for page_id in selected_ids
            if checkpoint["pages"].get(page_id, {}).get("status") != "success"
        ]
        audit = audit_candidates(candidates, chunks)
        status = "pass" if not failed and candidates else "fail"
        summary = {
            "schema_version": "indicator-rule-probe/v0.1",
            "status": status,
            "page_indexes": list(PROBE_PAGE_INDEXES),
            "pages_success": len(selected_ids) - len(failed),
            "failed_pages": failed,
            "accepted_rules": len(candidates),
            "rejections": sum(len(package["rejections"]) for package in packages),
            "provider_summary": _provider_summary(checkpoint, selected_ids),
            "audit": audit,
        }
        atomic_write_json(output / "probe-summary.json", summary)
        return summary
    _run_windows(checkpoint_path, checkpoint, windows, library, key, workers, provider_post)
    return _write_full_artifacts(
        output, manifest, checkpoint, chunks, library, library_path, chunks_manifest, workers
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks", type=Path,
        default=ROOT / "source-packages/canonical/evidence/chapter-01/manifest.json",
    )
    parser.add_argument(
        "--indicator-library", type=Path,
        default=ROOT / "runtime/candidates/chapter-01/indicator-library-v0.1/indicator-library.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "runtime/candidates/chapter-01/indicator-rules-v0.1",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--revalidate", action="store_true")
    args = parser.parse_args()
    key = "" if args.revalidate else sys.stdin.readline().strip()
    if not key and not args.revalidate:
        print("DEEPSEEK_API_KEY must be supplied through stdin", file=sys.stderr)
        return 2
    result = run(
        args.chunks, args.indicator_library, args.output, key,
        workers=args.workers, probe=args.probe, revalidate=args.revalidate,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") in {"pass", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
