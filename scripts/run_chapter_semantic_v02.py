"""Run the page-local DeepSeek direct v0.2 extraction.

The key is read from stdin and is never part of a checkpoint or artifact.  A
missing key is an intentional, harmless dry-run failure rather than a prompt
to use the old v0.1 output as new evidence.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import ssl
import sys
import time
from types import SimpleNamespace
from urllib import error as urlerror, request

from medical_kg_sourceprep.provenance.book_sources import build_book_manifest_from_packages
from medical_kg_sourceprep.extraction.artifacts import load_json, sha256_path
from medical_kg_sourceprep.graph.knowledge_graph import KnowledgeGraphBuilder, PageText
from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk, atomic_write_json, load_chunk_manifest
from medical_kg_sourceprep.extraction.semantic_contract import (
    PROMPT_VERSION, VALIDATOR_VERSION, build_v02_prompt, validate_v02,
)
from medical_kg_sourceprep.graph.semantic_graph import SemanticGraphBuilder, SemanticRecord, SemanticRelation

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"


def _pinned_opener(ip: str | None) -> object:
    """Use direct HTTPS while preserving api.deepseek.com certificate/SNI."""
    if not ip:
        return request.build_opener(request.ProxyHandler({}))

    class PinnedConnection(http.client.HTTPSConnection):
        def connect(self):
            self.sock = socket.create_connection((ip, self.port), self.timeout, self.source_address)
            self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)

    class PinnedHandler(request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(PinnedConnection, req, context=self._context)

    return request.build_opener(request.ProxyHandler({}), PinnedHandler(context=ssl.create_default_context()))


def _post(key: str, prompt: str, opener: object | None = None) -> dict:
    opener = opener or _pinned_opener(os.environ.get("DEEPSEEK_API_IP"))
    payload = {"model": MODEL, "temperature": 0, "max_tokens": 32768,
               "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": "Return JSON only. Use no outside knowledge."},
                            {"role": "user", "content": prompt}]}
    req = request.Request(ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode(), method="POST",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                   "User-Agent": "medical-kg-sourceprep/0.4"})
    last_error: Exception | None = None
    for attempt, delay in enumerate((0, 1, 2), 1):
        if delay:
            time.sleep(delay)
        try:
            with opener.open(req, timeout=180) as response:  # type: ignore[attr-defined]
                decoded = json.loads(response.read().decode("utf-8"))
            break
        except urlerror.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 501, 502, 503, 504} or attempt == 3:
                raise
        except (urlerror.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == 3:
                raise
    else:
        raise last_error or RuntimeError("transport retries exhausted")
    content = decoded.get("choices", [{}])[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("empty provider content")
    return json.loads(content)


def _records(result: dict, chunks: dict[str, EvidenceChunk]) -> tuple[list[SemanticRecord], list[SemanticRelation]]:
    records: list[SemanticRecord] = []
    by_key: dict[tuple[str, str], SemanticRecord] = {}
    for item in result["candidates"]:
        if item["candidate_type"] == "relation":
            continue
        source = item["text_span"]
        record = SemanticRecord(item["candidate_id"], item["entity_type"], "candidate", item["text"],
                                source["chunk_id"], source["char_start"], source["char_end"],
                                item.get("semantic_type"), item.get("subject_logic"),
                                rule_payload=item.get("components"), candidate_key=item["candidate_key"])
        records.append(record)
        page_id = chunks[source["chunk_id"]].page_id
        scoped_key = (page_id, item["candidate_key"])
        if scoped_key in by_key:
            raise RuntimeError("candidate_key is not unique within its page")
        by_key[scoped_key] = record
    relations: list[SemanticRelation] = []
    for item in result["candidates"]:
        if item["candidate_type"] != "relation":
            continue
        anchor = item["source"]
        page_id = chunks[anchor["chunk_id"]].page_id
        source = by_key[(page_id, item["source_candidate_key"])]
        target = by_key[(page_id, item["target_candidate_key"])]
        relations.append(SemanticRelation(source.record_id, item["relation"], target.record_id,
                                          anchor["chunk_id"], anchor["exact_quote"], item["relation_cue"],
                                          anchor["chunk_sha256"]))
    return records, relations


def _materialize_text_spans(candidates: list[dict], chunks: dict[str, EvidenceChunk]) -> list[dict]:
    """Upgrade already-checkpointed v0.2 candidates without changing model data."""
    result = []
    for original in candidates:
        item = dict(original)
        if item.get("candidate_type") != "relation" and "text_span" not in item:
            source = item["source"]
            chunk = chunks[source["chunk_id"]]
            start = chunk.text.find(item["text"], source["char_start"], source["char_end"])
            if start < 0 or chunk.text.count(item["text"], source["char_start"], source["char_end"]) != 1:
                raise RuntimeError("checkpoint candidate text span cannot be replayed")
            item["text_span"] = {"chunk_id": source["chunk_id"],
                                 "chunk_sha256": source["chunk_sha256"],
                                 "exact_quote": item["text"], "char_start": start,
                                 "char_end": start + len(item["text"])}
        result.append(item)
    return result


def run(chunks_manifest: Path, source_manifest: Path, output: Path, key: str) -> dict:
    manifest, chunks_tuple = load_chunk_manifest(chunks_manifest)
    if manifest.get("page_count") != 24 or manifest.get("chunk_count") != 44:
        raise RuntimeError("chapter input must contain exactly 24 pages and 44 chunks")
    chunks = {chunk.chunk_id: chunk for chunk in chunks_tuple}
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    input_hash = sha256_path(chunks_manifest)
    saved = load_json(checkpoint_path) if checkpoint_path.exists() else {}
    if saved and (saved.get("input_manifest_sha256") != input_hash or saved.get("model") != MODEL
                  or saved.get("prompt_version") != PROMPT_VERSION
                  or saved.get("validator_version") != VALIDATOR_VERSION):
        raise RuntimeError("checkpoint input or provider identity drift")
    saved.update({"schema_version": "chapter-semantic-checkpoint/v0.2", "input_manifest_sha256": input_hash,
                  "model": MODEL, "prompt_version": PROMPT_VERSION,
                  "validator_version": VALIDATOR_VERSION, "trust_env": False,
                  "limits": {"extractions": 128, "relations": 48}})
    pages = {index: [chunk for chunk in chunks_tuple if chunk.page_index == index] for index in range(24)}
    for index, page_chunks in pages.items():
        page_id = f"page:{index:04d}"
        if saved.get("pages", {}).get(page_id, {}).get("status") == "success":
            continue
        window = SimpleNamespace(text=json.dumps([
            {"chunk_id": chunk.chunk_id, "chunk_sha256": chunk.chunk_sha256, "text": chunk.text}
            for chunk in page_chunks
        ], ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        try:
            raw = _post(key, build_v02_prompt(window))
            validated = validate_v02(raw, page_chunks)
        except Exception as exc:
            saved.setdefault("pages", {})[page_id] = {"status": "failed", "error": str(exc)[:240],
                                                       "chunk_ids": [c.chunk_id for c in page_chunks]}
            atomic_write_json(checkpoint_path, saved)
            raise
        saved.setdefault("pages", {})[page_id] = {"status": "success", "chunk_ids": [c.chunk_id for c in page_chunks],
                                                   "output": validated}
        atomic_write_json(checkpoint_path, saved)
    all_results = [saved["pages"][f"page:{index:04d}"]["output"] for index in range(24)]
    candidates = _materialize_text_spans(
        [item for result in all_results for item in result["candidates"]], chunks
    )
    rejections = [item for result in all_results for item in result["rejections"]]
    package = {"schema_version": "semantic-candidates/v0.2", "status": "candidate-only",
               "approved": 0, "candidates": candidates, "rejections": rejections}
    atomic_write_json(output / "extraction.json", package)
    atomic_write_json(output / "review-queue.json", {"status": "HOLD", "items": rejections,
                                                       "counts": {"review_required": len(rejections)}})
    records, relations = _records({"candidates": candidates}, chunks)
    source = load_json(source_manifest)
    book_manifest = build_book_manifest_from_packages(book={"book_id": "clinical-hematology", "title": "Clinical Hematology", "edition": "source-package"}, source_manifest=source, chunk_manifest=manifest)
    pages_text = tuple(PageText(page["page_id"], (source_manifest.parent / page["raw_path"]).read_text(encoding="utf-8"), (source_manifest.parent / page["cleaned_path"]).read_text(encoding="utf-8")) for page in source["pages"])
    base = output / "base-knowledge.sqlite"
    if not base.exists():
        KnowledgeGraphBuilder().build(base, book_manifest, pages_text)
    graph = SemanticGraphBuilder().build(output / "knowledge.sqlite", base, book_manifest, records, relations)
    run_manifest = {"schema_version": "chapter-semantic-kg-run/v0.2", "provider": "deepseek-direct",
                    "model": MODEL, "status": "candidate-only", "hold": True,
                    "input": {"chunk_manifest_sha256": input_hash, "pages": 24, "chunks": 44},
                    "limits": {"max_extractions_per_page": 128, "max_relations_per_page": 48},
                    "graph": {"node_count": graph.node_count, "edge_count": graph.edge_count,
                              "status_counts": dict(graph.status_counts), "package_hash": graph.package_hash},
                    "counts": {"candidates": len(candidates), "rejections": len(rejections), "approved": 0}}
    atomic_write_json(output / "run-manifest.json", run_manifest)
    return run_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=ROOT / "source-packages/chunks/chapter-01/manifest.json")
    parser.add_argument("--source", type=Path, default=ROOT / "source-packages/chapter-01/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/chapter-01-semantic-kg-deepseek-direct-v0.2")
    args = parser.parse_args()
    # Read one hidden stdin line so a PTY can submit the secret with newline;
    # the value is never written to checkpoints or logs.
    key = sys.stdin.readline().strip()
    if not key:
        print("DEEPSEEK_API_KEY must be supplied through hidden stdin", file=sys.stderr)
        return 2
    run(args.chunks, args.source, args.output, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
