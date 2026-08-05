"""Run the staged chapter v0.4 endpoint, relation, and rule recovery."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import sys
import time
from urllib import error as urlerror, request

from medical_kg_sourceprep.provenance.book_sources import build_book_manifest_from_packages
from medical_kg_sourceprep.extraction.artifacts import (
    canonical_json as _canonical,
    directory_sha256,
    load_json as _load_json,
    sha256_path as _sha,
)
from medical_kg_sourceprep.graph.knowledge_graph import PageText
from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk, atomic_write_json, load_chunk_manifest
from medical_kg_sourceprep.graph.semantic_graph import SemanticGraphBuilder, SemanticRecord, SemanticRelation
from medical_kg_sourceprep.extraction.semantic_v04 import (
    CATALOG_VERSION, CONTRACT_VERSION, ENDPOINT_PROMPT_VERSION, ENDPOINT_VALIDATOR_VERSION,
    RELATION_PROMPT_VERSION, RELATION_VALIDATOR_VERSION, RULE_PROMPT_VERSION,
    RULE_VALIDATOR_VERSION, augment_catalog, audit_superseded_v02_relations,
    baseline_model_relations,
    build_base_catalog, build_endpoint_prompt, build_relation_prompt, build_rule_prompt,
    recover_structural_relations, stable_relations,
    validate_endpoints, validate_relations, validate_rules,
)
from run_chapter_semantic_v02 import ENDPOINT, MODEL, _pinned_opener, _records


ROOT = Path(__file__).resolve().parents[1]
RUN_VERSION = "chapter-semantic-kg-run/v0.4"
CHECKPOINT_VERSION = "chapter-semantic-checkpoint/v0.4"


def _pages(chunks: tuple[EvidenceChunk, ...]) -> dict[str, list[EvidenceChunk]]:
    result: dict[str, list[EvidenceChunk]] = {}
    for chunk in sorted(chunks, key=lambda value: (value.page_index, value.start_offset or 0, value.chunk_id)):
        result.setdefault(chunk.page_id, []).append(chunk)
    return result


def _raise_provider_timeout(_signum, _frame) -> None:
    raise TimeoutError("provider hard timeout")


def _provider_post(key: str, prompt: str, max_tokens: int = 8192,
                   hard_timeout_seconds: int = 90) -> tuple[dict, dict]:
    """Call the official host directly and retain non-thinking response metadata."""
    payload = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
               "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": "Return JSON only. Use no outside knowledge."},
                            {"role": "user", "content": prompt}]}
    req = request.Request(ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode(), method="POST",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                   "User-Agent": "medical-kg-sourceprep/0.4"})
    opener = _pinned_opener(os.environ.get("DEEPSEEK_API_IP"))
    last_format_error: json.JSONDecodeError | None = None
    for format_attempt in range(3):
        decoded: dict | None = None
        retry_delays = (0, 5, 20, 60, 120)
        for attempt, delay in enumerate(retry_delays, 1):
            if delay:
                time.sleep(delay)
            try:
                previous_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, _raise_provider_timeout)
                signal.setitimer(signal.ITIMER_REAL, hard_timeout_seconds)
                try:
                    with opener.open(req, timeout=hard_timeout_seconds) as response:  # type: ignore[attr-defined]
                        decoded = json.loads(response.read().decode("utf-8"))
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous_handler)
                break
            except urlerror.HTTPError as exc:
                if exc.code not in {429, 500, 501, 502, 503, 504} or attempt == len(retry_delays):
                    raise
            except (urlerror.URLError, TimeoutError):
                if attempt == len(retry_delays):
                    raise
        if decoded is None:
            raise RuntimeError("transport retries exhausted")
        choice = decoded.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("empty provider content")
        reasoning_tokens = (decoded.get("usage", {}).get("completion_tokens_details", {})
                            .get("reasoning_tokens"))
        reasoning_content = message.get("reasoning_content")
        if reasoning_content not in (None, "") or reasoning_tokens not in (None, 0):
            raise RuntimeError("provider returned thinking output despite thinking.type=disabled")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            last_format_error = exc
            if format_attempt < 2:
                continue
            break
        metadata = {"finish_reason": choice.get("finish_reason"),
                    "reasoning_content": None if reasoning_content in (None, "") else "present",
                    "reasoning_tokens": reasoning_tokens,
                    "format_attempts": format_attempt + 1,
                    "max_tokens": max_tokens, "hard_timeout_seconds": hard_timeout_seconds,
                    "usage": {key_name: decoded.get("usage", {}).get(key_name)
                              for key_name in ("prompt_tokens", "completion_tokens", "total_tokens")}}
        return parsed, metadata
    raise RuntimeError(f"provider returned invalid JSON on three complete attempts: {last_format_error}")


def _checkpoint(path: Path, identity: dict) -> dict:
    value = _load_json(path) if path.exists() else {}
    drift = sorted(key for key, expected in identity.items() if key in value and value[key] != expected)
    if drift:
        raise RuntimeError(f"checkpoint identity drift: {','.join(drift)}")
    value.update(identity)
    value.setdefault("pages", {})
    return value


def _run_stage(path: Path, identity: dict, pages: dict[str, list[EvidenceChunk]],
               entries_by_page: dict[str, list[dict]], prompt_builder, validator,
               catalog: dict, key: str, provider_limits: dict) -> tuple[list[dict], list[dict]]:
    checkpoint = _checkpoint(path, identity)
    for page_id, page_chunks in pages.items():
        if checkpoint["pages"].get(page_id, {}).get("status") == "success":
            continue
        try:
            raw, provider = _provider_post(
                key, prompt_builder(page_id, page_chunks, entries_by_page.get(page_id, [])),
                provider_limits["max_tokens"], provider_limits["hard_timeout_seconds"])
            output = validator(raw, page_chunks, catalog)
            checkpoint["pages"][page_id] = {"status": "success",
                "chunk_ids": [chunk.chunk_id for chunk in page_chunks], "provider": provider,
                "output": output}
        except Exception as exc:
            checkpoint["pages"][page_id] = {"status": "failed",
                "chunk_ids": [chunk.chunk_id for chunk in page_chunks], "error": str(exc)[:240]}
            atomic_write_json(path, checkpoint)
            raise
        atomic_write_json(path, checkpoint)
    packages = [checkpoint["pages"][page_id]["output"] for page_id in pages]
    provider_metadata = [checkpoint["pages"][page_id]["provider"] for page_id in pages]
    return packages, provider_metadata


def _entries_by_page(catalog: dict) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for entry in catalog["entries"]:
        result.setdefault(entry["page_id"], []).append(entry)
    return result


def _stage_api_ip(path: Path, current_ip: str | None) -> str | None:
    if not path.exists():
        return current_ip
    saved = _load_json(path)
    pages = saved.get("pages", {})
    if len(pages) == 24 and all(value.get("status") == "success" for value in pages.values()):
        return saved.get("api_ip")
    return current_ip


def _relation_for_graph(item: dict, catalog_by_key: dict[tuple[str, str], dict]) -> SemanticRelation:
    source = catalog_by_key[(item["page_id"], item["source_candidate_key"])]
    target = catalog_by_key[(item["page_id"], item["target_candidate_key"])]
    anchor = item["source"]
    evidence = tuple({"evidence_role": value["evidence_role"],
                      "source_chunk_id": value["chunk_id"],
                      "source_chunk_sha256": value["chunk_sha256"],
                      "source_quote": value["exact_quote"],
                      "relation_cue": value["relation_cue"]} for value in item["evidence"])
    return SemanticRelation(source["candidate_id"], item["relation"], target["candidate_id"],
                            anchor["chunk_id"], anchor["exact_quote"], item["relation_cue"],
                            anchor["chunk_sha256"], item["origin"], evidence)


def _rule_records(rules: list[dict], catalog_by_key: dict[tuple[str, str], dict]) -> tuple[list[SemanticRecord], list[tuple[str, str, str]]]:
    records, relations = [], []
    for item in rules:
        source = item["source"]
        record = SemanticRecord(item["candidate_id"], "InterpretationRule", "candidate",
                                source["exact_quote"], source["chunk_id"], source["char_start"],
                                source["char_end"], item["semantic_type"], item["subject_logic"],
                                rule_payload=item["components"], candidate_key=item["rule_key"])
        records.append(record)
        for key in item["subject_candidate_keys"]:
            relations.append((record.record_id, "RULE_HAS_SUBJECT",
                              catalog_by_key[(item["page_id"], key)]["candidate_id"]))
        relations.append((record.record_id, "RULE_HAS_CONCLUSION",
                          catalog_by_key[(item["page_id"], item["conclusion_candidate_key"])]["candidate_id"]))
        for key in item["population_candidate_keys"]:
            relations.append((record.record_id, "RULE_APPLIES_TO_POPULATION",
                              catalog_by_key[(item["page_id"], key)]["candidate_id"]))
        for key in item["method_candidate_keys"]:
            relations.append((record.record_id, "RULE_REQUIRES_METHOD",
                              catalog_by_key[(item["page_id"], key)]["candidate_id"]))
    return records, relations


def _audit(extraction: dict, chunks: dict[str, EvidenceChunk], catalog: dict) -> dict:
    replayed = 0
    for relation in extraction["relations"]:
        for evidence in relation["evidence"]:
            chunk = chunks[evidence["chunk_id"]]
            if (evidence["chunk_sha256"] != chunk.chunk_sha256
                    or chunk.text.count(evidence["exact_quote"]) != 1
                    or evidence["relation_cue"] not in evidence["exact_quote"]):
                raise RuntimeError("relation evidence replay audit failed")
            replayed += 1
    components = 0
    for rule in extraction["rules"]:
        values = [*rule["components"]["conditions"], rule["components"]["conclusion"]]
        if rule["components"].get("connector"):
            values.append(rule["components"]["connector"])
        for component in values:
            source = component["source"]
            chunk = chunks[source["chunk_id"]]
            if source["chunk_sha256"] != chunk.chunk_sha256 or chunk.text[source["char_start"]:source["char_end"]] != component["text"]:
                raise RuntimeError("rule component replay audit failed")
            components += 1
    valid_keys = {(entry["page_id"], entry["candidate_key"]) for entry in catalog["entries"]}
    for relation in extraction["relations"]:
        if ((relation["page_id"], relation["source_candidate_key"]) not in valid_keys
                or (relation["page_id"], relation["target_candidate_key"]) not in valid_keys):
            raise RuntimeError("dangling or cross-page relation endpoint")
    return {"relation_evidence": {"replayed": replayed, "rate": 1.0},
            "rule_components": {"replayed": components, "rate": 1.0},
            "dangling_endpoints": 0, "cross_page_edges": 0}


def run(chunks_manifest: Path, source_manifest: Path, v02_dir: Path, v03_dir: Path,
        output: Path, key: str | None) -> dict:
    manifest, chunks_tuple = load_chunk_manifest(chunks_manifest)
    pages = _pages(chunks_tuple)
    if manifest.get("page_count") != 24 or manifest.get("chunk_count") != 44 or len(pages) != 24:
        raise RuntimeError("chapter input must contain exactly 24 pages and 44 chunks")
    chunks = {chunk.chunk_id: chunk for chunk in chunks_tuple}
    v02 = _load_json(v02_dir / "extraction.json")
    output.mkdir(parents=True, exist_ok=True)
    base_catalog = build_base_catalog(v02, v02_dir / "knowledge.sqlite", chunks_tuple)
    input_identity = {"schema_version": CHECKPOINT_VERSION,
        "input_manifest_sha256": _sha(chunks_manifest), "v02_extraction_sha256": _sha(v02_dir / "extraction.json"),
        "base_catalog_sha256": base_catalog["catalog_sha256"], "model": MODEL,
        "provider": "deepseek-direct", "trust_env": False, "thinking": "disabled",
        "api_host": "api.deepseek.com"}
    catalog_checkpoint = {**input_identity, "stage": "catalog", "catalog_version": CATALOG_VERSION,
                          "status": "success", "count": len(base_catalog["entries"])}
    atomic_write_json(output / "catalog-checkpoint.json", catalog_checkpoint)
    if key is None:
        atomic_write_json(output / "entity-catalog.json", base_catalog)
        return {"schema_version": RUN_VERSION, "status": "catalog-only", "hold": True,
                "entity_catalog_sha256": base_catalog["catalog_sha256"],
                "counts": {"entities": len(base_catalog["entries"]), "approved": 0}}

    endpoint_path = output / "endpoint-checkpoint.json"
    standard_limits = {"max_tokens": 8192, "hard_timeout_seconds": 90,
                       "retry_delays_seconds": [0, 5, 20, 60, 120]}
    rule_limits = {"max_tokens": 16384, "hard_timeout_seconds": 180,
                   "retry_delays_seconds": [0, 5, 20, 60, 120]}
    endpoint_identity = {**input_identity, **standard_limits, "stage": "endpoint",
                         "prompt_version": ENDPOINT_PROMPT_VERSION,
                         "validator_version": ENDPOINT_VALIDATOR_VERSION,
                         "api_ip": _stage_api_ip(endpoint_path, os.environ.get("DEEPSEEK_API_IP"))}
    endpoint_packages, endpoint_provider = _run_stage(
        endpoint_path, endpoint_identity, pages, _entries_by_page(base_catalog),
        build_endpoint_prompt, validate_endpoints, base_catalog, key, standard_limits)
    catalog = augment_catalog(base_catalog, endpoint_packages)
    atomic_write_json(output / "entity-catalog.json", catalog)
    endpoint_candidates = [item for package in endpoint_packages for item in package["candidates"]]
    atomic_write_json(output / "endpoint-extraction.json", {"schema_version": "endpoint-extraction/v0.4",
        "status": "candidate-only", "approved": 0, "pages": 24, "packages": endpoint_packages,
        "candidates": endpoint_candidates})

    augmented_identity = {**input_identity, "entity_catalog_sha256": catalog["catalog_sha256"]}
    relation_path, rule_path = output / "relation-checkpoint.json", output / "rule-checkpoint.json"
    relation_identity = {**augmented_identity, **standard_limits, "stage": "relation",
                         "prompt_version": RELATION_PROMPT_VERSION,
                         "validator_version": RELATION_VALIDATOR_VERSION,
                         "api_ip": _stage_api_ip(relation_path, os.environ.get("DEEPSEEK_API_IP"))}
    rule_identity = {**augmented_identity, **rule_limits, "stage": "rule",
                     "prompt_version": RULE_PROMPT_VERSION,
                     "validator_version": RULE_VALIDATOR_VERSION,
                     "api_ip": _stage_api_ip(rule_path, os.environ.get("DEEPSEEK_API_IP"))}
    entries_by_page = _entries_by_page(catalog)
    relation_packages, relation_provider = _run_stage(
        relation_path, relation_identity, pages, entries_by_page,
        build_relation_prompt, validate_relations, catalog, key, standard_limits)
    rule_packages, rule_provider = _run_stage(
        rule_path, rule_identity, pages, entries_by_page,
        build_rule_prompt, validate_rules, catalog, key, rule_limits)

    structural, structural_review = recover_structural_relations(catalog, chunks_tuple)
    baseline = baseline_model_relations(v02, catalog, chunks_tuple)
    superseded_review = audit_superseded_v02_relations(
        v02_dir / "knowledge.sqlite", baseline, structural, catalog)
    model_relations = [item for package in relation_packages for item in package["candidates"]]
    relations = stable_relations([{"candidates": baseline}, {"candidates": structural},
                                  {"candidates": model_relations}])
    rules = [item for package in rule_packages for item in package["candidates"]]
    rejections = ([item for package in endpoint_packages for item in package["rejections"]]
                  + [item for package in relation_packages for item in package["rejections"]]
                  + [item for package in rule_packages for item in package["rejections"]]
                  + structural_review + superseded_review)
    extraction = {"schema_version": CONTRACT_VERSION, "status": "candidate-only", "approved": 0,
                  "entity_catalog_sha256": catalog["catalog_sha256"], "endpoints": endpoint_candidates,
                  "relations": relations, "rules": rules, "rejections": rejections}
    audit = _audit(extraction, chunks, catalog)
    extraction["audit"] = audit
    atomic_write_json(output / "relation-extraction.json", {"schema_version": "relation-extraction/v0.4",
        "status": "candidate-only", "approved": 0, "pages": 24, "packages": relation_packages,
        "baseline_model_candidates": baseline, "derived_candidates": structural,
        "superseded_v02_review": superseded_review, "candidates": relations})
    atomic_write_json(output / "rule-extraction.json", {"schema_version": "rule-extraction/v0.4",
        "status": "candidate-only", "approved": 0, "pages": 24, "packages": rule_packages,
        "candidates": rules})
    atomic_write_json(output / "extraction.json", extraction)
    atomic_write_json(output / "review-queue.json", {"schema_version": "semantic-review-queue/v0.4",
        "status": "HOLD", "approved": 0, "items": rejections,
        "counts": {"review_required": len(rejections),
                   "by_reason": dict(Counter(item["reason_code"] for item in rejections))}})
    atomic_write_json(output / "gold-template.json", {"schema_version": "semantic-gold-template/v0.4",
        "status": "HOLD", "generated_from_model": False, "source_manifest_sha256": _sha(source_manifest),
        "entity_catalog_sha256": catalog["catalog_sha256"],
        "pages": [{"page_id": page_id, "gold_status": "unreviewed", "relations": [], "rules": [],
                   "reviewer": None, "reviewed_at": None} for page_id in pages]})

    source = _load_json(source_manifest)
    book_manifest = build_book_manifest_from_packages(
        book={"book_id": "clinical-hematology", "title": "Clinical Hematology", "edition": "source-package"},
        source_manifest=source, chunk_manifest=manifest)
    page_text = tuple(PageText(page["page_id"],
        (source_manifest.parent / page["raw_path"]).read_text(encoding="utf-8"),
        (source_manifest.parent / page["cleaned_path"]).read_text(encoding="utf-8")) for page in source["pages"])
    del page_text  # source package validation occurs through book_manifest and the frozen base graph.
    records, _old_relations = _records(v02, chunks)
    for item in endpoint_candidates:
        source_span = item["text_span"]
        records.append(SemanticRecord(item["candidate_id"], item["entity_type"], "candidate", item["text"],
            source_span["chunk_id"], source_span["char_start"], source_span["char_end"], candidate_key=item["candidate_key"]))
    catalog_by_key = {(entry["page_id"], entry["candidate_key"]): entry for entry in catalog["entries"]}
    rule_records, rule_edges = _rule_records(rules, catalog_by_key)
    records.extend(rule_records)
    graph_relations = [_relation_for_graph(item, catalog_by_key) for item in relations]
    graph_relations.extend(rule_edges)
    graph_path = output / "knowledge.sqlite"
    staging_graph = output / ".knowledge.sqlite.v04.build"
    staging_graph.unlink(missing_ok=True)
    graph = SemanticGraphBuilder().build(staging_graph, v02_dir / "base-knowledge.sqlite",
                                         book_manifest, records, graph_relations)
    staging_graph.replace(graph_path)

    v03_extraction = _load_json(v03_dir / "extraction.json") if (v03_dir / "extraction.json").exists() else {}
    comparison = {"schema_version": "semantic-quality-comparison/v0.4",
        "gold_status": "not_generated_from_model", "precision_recall_f1": "HOLD",
        "v02": {"entities": len([item for item in v02.get("candidates", []) if item.get("candidate_type") != "relation"]),
                "model_relations": len(baseline), "approved": 0},
        "v03": {"status": "superseded", "relations": len(v03_extraction.get("relations", [])),
                "rules": len(v03_extraction.get("rules", [])), "approved": 0},
        "v04": {"catalog_entities": len(catalog["entries"]), "endpoint_additions": len(endpoint_candidates),
                "relations": len(relations), "rules": len(rules), "rejected": len(rejections),
                "relation_origins": dict(Counter(item["origin"] for item in relations)),
                "relation_types": dict(Counter(item["relation"] for item in relations)),
                "rule_types": dict(Counter(item["semantic_type"] for item in rules)), "approved": 0},
        "audit": audit, "v02_directory_sha256": directory_sha256(v02_dir),
        "v03_directory_sha256": directory_sha256(v03_dir)}
    atomic_write_json(output / "quality-comparison.json", comparison)
    run_manifest = {"schema_version": RUN_VERSION, "provider": "deepseek-direct", "model": MODEL,
        "status": "candidate-only", "hold": True, "supersedes": "chapter-01-semantic-kg-deepseek-direct-v0.3",
        "input": {"chunk_manifest_sha256": input_identity["input_manifest_sha256"],
                  "v02_extraction_sha256": input_identity["v02_extraction_sha256"],
                  "base_catalog_sha256": base_catalog["catalog_sha256"],
                  "entity_catalog_sha256": catalog["catalog_sha256"], "pages": 24, "chunks": 44},
        "stages": {"catalog": {"completed": 1}, "endpoint": {"completed": 24},
                   "relation": {"completed": 24}, "rule": {"completed": 24}, "aggregate": {"completed": 1}},
        "provider_contract": {"endpoint": ENDPOINT, "trust_env": False, "thinking": "disabled",
                              "stage_limits": {"endpoint": standard_limits, "relation": standard_limits,
                                               "rule": rule_limits},
                              "stage_api_ips": {"endpoint": endpoint_identity["api_ip"],
                                                "relation": relation_identity["api_ip"],
                                                "rule": rule_identity["api_ip"]},
                              "reasoning_tokens": [item["reasoning_tokens"] for item in endpoint_provider + relation_provider + rule_provider]},
        "counts": {"catalog_entities": len(catalog["entries"]), "endpoint_additions": len(endpoint_candidates),
                   "relations": len(relations), "rules": len(rules), "rejections": len(rejections), "approved": 0},
        "audit": audit, "graph": {"node_count": graph.node_count, "edge_count": graph.edge_count,
                                  "package_hash": graph.package_hash}}
    atomic_write_json(output / "run-manifest.json", run_manifest)
    aggregate = {**augmented_identity, "schema_version": CHECKPOINT_VERSION, "stage": "aggregate",
                 "status": "success", "extraction_sha256": _sha(output / "extraction.json"),
                 "database_sha256": _sha(graph_path), "run_manifest_sha256": _sha(output / "run-manifest.json")}
    atomic_write_json(output / "aggregate-checkpoint.json", aggregate)
    return run_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=ROOT / "source-packages/chunks/chapter-01/manifest.json")
    parser.add_argument("--source", type=Path, default=ROOT / "source-packages/chapter-01/manifest.json")
    parser.add_argument("--v02", type=Path, default=ROOT / "runtime/chapter-01-semantic-kg-deepseek-direct-v0.2")
    parser.add_argument("--v03", type=Path, default=ROOT / "runtime/chapter-01-semantic-kg-deepseek-direct-v0.3")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/chapter-01-semantic-kg-deepseek-direct-v0.4")
    parser.add_argument("--catalog-only", action="store_true")
    args = parser.parse_args()
    if args.catalog_only:
        key = None
    else:
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key and sys.stdin.isatty():
            import getpass
            key = getpass.getpass("DeepSeek API key: ")
        elif not key:
            key = sys.stdin.readline().strip()
        if not key:
            print("DEEPSEEK_API_KEY must be supplied through hidden stdin", file=sys.stderr)
            return 2
    run(args.chunks, args.source, args.v02, args.v03, args.output, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
