"""Generic LangExtract-to-semantic-graph candidate projection.

This boundary accepts LangExtract's native ``extractions`` envelope and emits
only hash-bound candidate records.  Ambiguous or incomplete model output is
kept in a bounded review queue instead of being repaired or inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .llm_extraction import EvidenceChunk, atomic_write_json
from .book_sources import build_book_manifest_from_packages
from .knowledge_graph import KnowledgeGraphBuilder, PageText
from .semantic_graph import SemanticGraphBuilder
from .semantic_graph import ENTITY_TYPES, SEMANTIC_RELATIONS, SEMANTIC_TYPES, SUBJECT_LOGICS, SemanticRecord


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    records: tuple[SemanticRecord, ...]
    relations: tuple[tuple[str, str, str], ...]
    review_queue: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ChapterExtractionProvider:
    provider: str
    endpoint: str
    model: str
    api_key: str
    user_agent: str
    supports_json_schema: bool

    @property
    def identity(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "endpoint": self.endpoint,
            "model": self.model,
        }


class TransientTransportError(RuntimeError):
    """A bounded LangExtract/OpenAI transport failure safe to persist."""


_TRANSIENT_DELAYS = (1.0, 2.0)
LANGEXTRACT_CHAPTER_MAX_OUTPUT_TOKENS = 32768
LANGEXTRACT_MAX_EXTRACTIONS_PER_PAGE = 128
LANGEXTRACT_MAX_RELATIONS_PER_PAGE = 48


def _resolve_chapter_provider(
    provider: str, *, env: Mapping[str, str] | None = None
) -> ChapterExtractionProvider:
    environment = os.environ if env is None else env
    if provider == "deepseek-direct":
        key = environment.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for deepseek-direct")
        return ChapterExtractionProvider(
            provider="deepseek-direct",
            endpoint="https://api.deepseek.com",
            model="deepseek-v4-flash",
            api_key=key,
            user_agent="medical-kg-sourceprep/0.3",
            supports_json_schema=False,
        )
    if provider == "opencode-go":
        key = __import__(
            "medical_kg_sourceprep.llm_extraction", fromlist=["load_opencode_key"]
        ).load_opencode_key(env=environment)
        return ChapterExtractionProvider(
            provider="opencode-go",
            endpoint="https://opencode.ai/zen/go/v1",
            model="deepseek-v4-flash",
            api_key=key,
            user_agent="opencode-ai/1.0",
            supports_json_schema=True,
        )
    raise ValueError(f"unsupported chapter extraction provider: {provider}")


def _nullable(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {"anyOf": [dict(schema), {"type": "null"}]}


def _langextract_output_schema() -> dict[str, Any]:
    """Return the strict raw envelope consumed by LangExtract's resolver."""
    text_ref = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "source_chunk_id": {"type": "string"},
            "source_quote": {"type": "string"},
        },
        "required": ["text", "source_chunk_id", "source_quote"],
        "additionalProperties": False,
    }
    variants = []
    for extraction_class, attributes in (
        ("entity", {
            "kind": {"type": "string", "enum": ["entity"]},
            "entity_type": {"type": "string", "enum": sorted(ENTITY_TYPES - {"SourceLocator", "InterpretationRule"})},
            "source_chunk_id": {"type": "string"},
            "source_quote": {"type": "string"},
        }),
        ("rule", {
            "kind": {"type": "string", "enum": ["rule"]},
            "entity_type": {"type": "string", "enum": ["InterpretationRule"]},
            "semantic_type": {"type": "string", "enum": sorted(SEMANTIC_TYPES)},
            "subject_logic": {"type": "string", "enum": sorted(SUBJECT_LOGICS)},
            "source_chunk_id": {"type": "string"},
            "source_quote": {"type": "string"},
            "conditions": _nullable({"type": "array", "items": text_ref}),
            "conclusion": _nullable(text_ref),
        }),
        ("relation", {
            "kind": {"type": "string", "enum": ["relation"]},
            "relation_type": {"type": "string", "enum": sorted(SEMANTIC_RELATIONS)},
            "source_text": {"type": "string"},
            "target_text": {"type": "string"},
            "source_chunk_id": {"type": "string"},
            "source_quote": {"type": "string"},
        }),
    ):
        attribute_name = f"{extraction_class}_attributes"
        variants.append({
            "type": "object",
            "properties": {
                extraction_class: {"type": "string"},
                attribute_name: {
                    "type": "object",
                    "properties": attributes,
                    "required": list(attributes),
                    "additionalProperties": False,
                },
            },
            "required": [extraction_class, attribute_name],
            "additionalProperties": False,
        })
    return {
        "type": "object",
        "properties": {
            "extractions": {"type": "array", "items": {"anyOf": variants}},
        },
        "required": ["extractions"],
        "additionalProperties": False,
    }


def _redact_error(error: Exception, secrets: Sequence[str]) -> str:
    message = str(error)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message[:240]


def _is_transient_transport_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return True
    message = str(error).casefold()
    return any(marker in message for marker in (
        "router.unavailable", "router unavailable", "http 500", "http 501",
        "http 502", "http 503", "http 504", "internal server error",
        "connection error", "temporary failure in name resolution",
    ))


def _run_with_transient_retry(operation: Any, *, sleep: Any = time.sleep,
                              delays: Sequence[float] = _TRANSIENT_DELAYS,
                              secrets: Sequence[str] = ()) -> Any:
    """Run one transport operation with a finite, explicit retry budget."""
    for attempt in range(1, len(delays) + 2):
        try:
            return operation()
        except Exception as error:
            if not _is_transient_transport_error(error):
                raise
            if attempt > len(delays):
                raise TransientTransportError(
                    f"transient transport retries exhausted attempts={attempt}: "
                    f"{_redact_error(error, secrets)}"
                ) from None
            sleep(delays[attempt - 1])


def _id(kind: str, entity_type: str, chunk_id: str, start: int) -> str:
    digest = hashlib.sha256(f"{kind}:{entity_type}:{chunk_id}:{start}".encode()).hexdigest()[:20]
    return f"semantic:langextract:{kind}:{digest}"


def _items(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = payload.get("extractions")
    if not isinstance(values, list):
        raise ValueError("LangExtract payload must contain an extractions list")
    return [value for value in values if isinstance(value, Mapping)]


def _ref(attrs: Mapping[str, Any], chunks: Mapping[str, EvidenceChunk]) -> tuple[EvidenceChunk, int, int]:
    source = attrs.get("source_ref") if isinstance(attrs.get("source_ref"), Mapping) else attrs
    chunk_id = source.get("chunk_id", source.get("source_chunk_id"))
    quote = source.get("exact_quote", source.get("source_quote"))
    if not isinstance(chunk_id, str) or not isinstance(quote, str) or not quote:
        raise ValueError("missing source quote or chunk")
    chunk = chunks.get(chunk_id)
    if chunk is None or source.get("chunk_sha256", chunk.chunk_sha256) != chunk.chunk_sha256:
        raise ValueError("source hash or chunk is invalid")
    start = chunk.text.find(quote)
    if start < 0 or chunk.text.count(quote) != 1:
        raise ValueError("source quote is absent or ambiguous")
    return chunk, start, start + len(quote)


def _text_ref(value: Any, chunks: Mapping[str, EvidenceChunk]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("rule component lacks source_ref")
    chunk, start, end = _ref(value, chunks)
    return {"chunk_id": chunk.chunk_id, "chunk_sha256": chunk.chunk_sha256,
            "exact_quote": chunk.text[start:end], "char_start": start, "char_end": end,
            "text": value.get("text", chunk.text[start:end])}


def adapt_langextract(payload: Mapping[str, Any], chunks: Sequence[EvidenceChunk]) -> ProjectionResult:
    """Project native LangExtract output into fixed, provenance-bound records."""
    by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    records: list[SemanticRecord] = []
    review: list[dict[str, Any]] = []
    endpoints: dict[str, list[SemanticRecord]] = {}
    relation_candidates: list[Mapping[str, Any]] = []
    for index, extraction in enumerate(_items(payload)):
        attrs = extraction.get("attributes") if isinstance(extraction.get("attributes"), Mapping) else {}
        kind = str(attrs.get("kind", extraction.get("extraction_class", ""))).lower()
        if kind in {"relation", "relations"}:
            relation_candidates.append(attrs)
            continue
        entity_type = attrs.get("entity_type", attrs.get("type"))
        if kind in {"rule", "rules", "interpretation", "reference_interval"}:
            entity_type = "InterpretationRule"
        if entity_type not in ENTITY_TYPES - {"SourceLocator"}:
            review.append({"index": index, "reason": "unknown entity type"})
            continue
        text = extraction.get("extraction_text", attrs.get("text"))
        if not isinstance(text, str) or not text:
            review.append({"index": index, "reason": "missing candidate text"})
            continue
        try:
            chunk, start, end = _ref(attrs, by_chunk)
            if chunk.text[start:end] != text:
                start = chunk.text.find(text, start, end)
                if start < 0:
                    raise ValueError("candidate text is not inside source quote")
                end = start + len(text)
            semantic_type = attrs.get("semantic_type")
            subject_logic = attrs.get("subject_logic")
            rule_payload = None
            if entity_type == "InterpretationRule":
                if semantic_type not in SEMANTIC_TYPES or subject_logic not in SUBJECT_LOGICS:
                    raise ValueError("rule contract is incomplete")
                rule_payload = {}
                conditions = attrs.get("conditions", attrs.get("condition"))
                conclusion = attrs.get("conclusion")
                if conditions is not None or conclusion is not None:
                    if not isinstance(conditions, list) or not isinstance(conclusion, Mapping):
                        raise ValueError("composite rule lacks condition or conclusion")
                    rule_payload["conditions"] = [_text_ref(item, by_chunk) for item in conditions]
                    rule_payload["conclusion"] = _text_ref(conclusion, by_chunk)
                for key in ("qualifiers", "qualifier", "at_least", "connector"):
                    if key in attrs:
                        rule_payload[key] = attrs[key]
            record = SemanticRecord(_id("rule" if entity_type == "InterpretationRule" else "entity", entity_type, chunk.chunk_id, start), entity_type, "candidate", text, chunk.chunk_id, start, end, semantic_type, subject_logic, rule_payload=rule_payload)
            records.append(record)
            endpoints.setdefault(text, []).append(record)
        except ValueError as error:
            review.append({"index": index, "reason": str(error)[:160]})
    relations: list[tuple[str, str, str]] = []
    for index, attrs in enumerate(relation_candidates):
        name = attrs.get("relation_type", attrs.get("relation"))
        source_id, target_id = attrs.get("source_id"), attrs.get("target_id")
        if not isinstance(source_id, str) or not isinstance(target_id, str):
            source_text, target_text = attrs.get("source_text"), attrs.get("target_text")
            sources, targets = endpoints.get(source_text, []), endpoints.get(target_text, [])
            if len(sources) != 1 or len(targets) != 1:
                review.append({"index": index, "reason": "relation endpoint is missing or ambiguous"})
                continue
            source_id, target_id = sources[0].record_id, targets[0].record_id
        allowed = SEMANTIC_RELATIONS.get(name)
        source = next((r for r in records if r.record_id == source_id), None)
        target = next((r for r in records if r.record_id == target_id), None)
        targets_allowed = allowed[1] if allowed and isinstance(allowed[1], tuple) else (allowed[1],) if allowed else ()
        if not allowed or source is None or target is None or source.entity_type != allowed[0] or target.entity_type not in targets_allowed:
            review.append({"index": index, "reason": "relation is unknown, dangling, or has invalid direction"})
            continue
        relations.append((source_id, name, target_id))
    return ProjectionResult(tuple(records), tuple(sorted(set(relations))), tuple(review))


__all__ = ["ProjectionResult", "adapt_langextract"]


def _langextract_prompt() -> str:
    return """Extract only explicit facts from the supplied source chunks. Return exactly one JSON object with this envelope and no other text: {"extractions":[...]}.
Use extraction classes entity, rule, or relation. Every extraction must include source_chunk_id and source_quote copied verbatim.
Keep source_quote text verbatim after JSON decoding: escape every source backslash as \\ inside JSON strings.
Entity entity_type is one of TestItem, TestMethod, ReferenceRange, MedicalConcept, Population.
Rule attributes must include entity_type=InterpretationRule, semantic_type one of DEFINES_AS, POSSIBLY_CAUSED_BY, SEEN_IN, LEADS_TO, RECOVERY_FACTOR, CLASSIFIES_AS, and subject_logic SINGLE, ALL, or ANY. Include conditions and one conclusion as anchored {text, source_chunk_id, source_quote} objects when present.
Relations must use only the ten fixed relation names and source_text/target_text.
Return at most 128 extractions total, including at most 48 relations. Never repeat the same extraction class, extraction text, source_chunk_id, and source_quote. Attributes may contain only the contract fields described above; omit unsupported or ambiguous output. Do not infer, normalize, repair, or use outside knowledge."""


def _native_extractions(result: Any) -> dict[str, Any]:
    values = []
    for extraction in getattr(result, "extractions", ()):
        attrs = dict(getattr(extraction, "attributes", None) or {})
        values.append({"extraction_class": getattr(extraction, "extraction_class", ""),
                       "extraction_text": getattr(extraction, "extraction_text", ""),
                       "attributes": attrs})
    return {"extractions": values}


def _extraction_identity(value: Mapping[str, Any]) -> tuple[object, ...]:
    attrs = value.get("attributes") if isinstance(value.get("attributes"), Mapping) else {}
    return (
        value.get("extraction_class"),
        value.get("extraction_text"),
        attrs.get("source_chunk_id", attrs.get("chunk_id")),
        attrs.get("source_quote", attrs.get("exact_quote")),
    )


def _normalize_exact_duplicates(payload: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    """Keep the first exact extraction while preserving the raw page limit."""
    values = payload.get("extractions")
    if not isinstance(values, list):
        raise ValueError("page extraction payload must contain an extractions list")
    if len(values) > LANGEXTRACT_MAX_EXTRACTIONS_PER_PAGE:
        raise ValueError("page extraction count exceeds configured limit")
    normalized: list[Any] = []
    seen: set[tuple[object, ...]] = set()
    duplicate_count = 0
    for value in values:
        if not isinstance(value, Mapping):
            normalized.append(value)
            continue
        identity = _extraction_identity(value)
        if identity in seen:
            duplicate_count += 1
            continue
        seen.add(identity)
        normalized.append(value)
    return {"extractions": normalized}, duplicate_count


def _validate_page_extractions(payload: Mapping[str, Any]) -> None:
    values = payload.get("extractions")
    if not isinstance(values, list):
        raise ValueError("page extraction payload must contain an extractions list")
    if len(values) > LANGEXTRACT_MAX_EXTRACTIONS_PER_PAGE:
        raise ValueError("page extraction count exceeds configured limit")
    relation_count = 0
    seen: set[tuple[object, ...]] = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("page extraction entry must be an object")
        attrs = value.get("attributes") if isinstance(value.get("attributes"), Mapping) else {}
        extraction_class = str(value.get("extraction_class", "")).lower()
        kind = str(attrs.get("kind", extraction_class)).lower()
        if kind in {"relation", "relations"}:
            relation_count += 1
        identity = _extraction_identity(value)
        if identity in seen:
            raise ValueError("page extraction payload contains a duplicate")
        seen.add(identity)
    if relation_count > LANGEXTRACT_MAX_RELATIONS_PER_PAGE:
        raise ValueError("page relation count exceeds configured limit")


def _chapter_status(windows: Mapping[str, Any]) -> str:
    expected = {f"page:{page_index:04d}" for page_index in range(24)}
    successful = {
        window_id for window_id, window in windows.items()
        if window_id in expected and isinstance(window, Mapping)
        and window.get("status") == "success"
    }
    return "all-success" if successful == expected else "partial-success"


def _disable_langextract_thinking(model: Any) -> Any:
    """Preserve the OpenCode Go no-thinking transport contract.

    LangExtract's OpenAI provider merges arbitrary model kwargs but only sends a
    fixed allowlist to chat completions.  ``thinking`` is not on that list.
    """
    build_request = model._build_chat_completions_params

    def build_with_disabled_thinking(prompt: str, config: dict[str, Any]) -> dict[str, Any]:
        request = build_request(prompt, config)
        extra_body = request.get("extra_body")
        request["extra_body"] = {
            **(extra_body if isinstance(extra_body, Mapping) else {}),
            "thinking": {"type": "disabled"},
        }
        return request

    model._build_chat_completions_params = build_with_disabled_thinking
    return model


def _json_schema_is_unsupported(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code != 400:
        return False
    message = str(error).casefold()
    return "json_schema" in message or (
        "response_format" in message
        and any(marker in message for marker in ("unsupported", "invalid", "not support"))
    )


def _enable_strict_schema_when_supported(model: Any, schema: Any,
                                         *, sleep: Any = time.sleep,
                                         secrets: Sequence[str] = ()) -> bool:
    """Probe the endpoint once and apply the production schema when honored."""
    request = model._build_chat_completions_params(
        'Return exactly {"extractions":[]}.', {"max_output_tokens": 128}
    )
    request["response_format"] = schema.response_format

    def probe() -> Any:
        return model._client.chat.completions.create(**request)

    try:
        response = _run_with_transient_retry(probe, sleep=sleep, secrets=secrets)
    except Exception as error:
        if _json_schema_is_unsupported(error):
            return False
        raise
    content = response.choices[0].message.content
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("json_schema probe returned malformed JSON") from error
    if payload != {"extractions": []}:
        raise RuntimeError("json_schema probe did not honor the production envelope")
    model.apply_schema(schema)
    return True


def run_chapter(
    chunks_manifest: Path,
    source_manifest: Path,
    output_dir: Path,
    *,
    provider: str = "opencode-go",
) -> dict[str, Any]:
    """Run bounded page-local LangExtract and atomically write chapter artifacts."""
    try:
        import httpx
        import langextract as lx
        from langextract.data import ExampleData, Extraction
        from langextract.providers.openai import OpenAILanguageModel
        from langextract.providers.schemas.openai import OpenAISchema
        from openai import OpenAI
    except Exception as error:
        raise RuntimeError("LangExtract dependency unavailable") from error
    chunk_manifest = json.loads(chunks_manifest.read_text(encoding="utf-8"))
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if chunk_manifest.get("page_count") != 24 or chunk_manifest.get("chunk_count") != 44:
        raise RuntimeError("chapter source coverage is not 24 pages and 44 chunks")
    _, chunks_tuple = __import__("medical_kg_sourceprep.llm_extraction", fromlist=["load_chunk_manifest"]).load_chunk_manifest(chunks_manifest)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks_tuple}
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.json"
    saved = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    provider_config = _resolve_chapter_provider(provider)
    input_manifest_sha256 = hashlib.sha256(chunks_manifest.read_bytes()).hexdigest()
    if saved and saved.get("input_manifest_sha256") != input_manifest_sha256:
        raise RuntimeError("chapter checkpoint input manifest does not match")
    if saved and saved.get("configuration") not in (None, provider_config.identity):
        raise RuntimeError("chapter checkpoint provider configuration does not match")
    if not saved:
        saved = {
            "schema_version": "langextract-chapter-checkpoint/v0.1",
            "input_manifest_sha256": input_manifest_sha256,
            "configuration": provider_config.identity,
            "windows": {},
        }
    elif "configuration" not in saved:
        saved["configuration"] = provider_config.identity
        atomic_write_json(checkpoint, saved)
    saved["limits"] = {
        "max_extractions_per_page": LANGEXTRACT_MAX_EXTRACTIONS_PER_PAGE,
        "max_relations_per_page": LANGEXTRACT_MAX_RELATIONS_PER_PAGE,
    }
    atomic_write_json(checkpoint, saved)
    key = provider_config.api_key
    client = httpx.Client(timeout=180, trust_env=False)
    try:
        model = OpenAILanguageModel(
            model_id=provider_config.model, api_key=key, base_url=provider_config.endpoint,
            temperature=0, max_output_tokens=LANGEXTRACT_CHAPTER_MAX_OUTPUT_TOKENS,
        )
        model._client = OpenAI(
            api_key=key,
            base_url=provider_config.endpoint,
            timeout=180,
            max_retries=0,
            http_client=client,
            default_headers={"User-Agent": provider_config.user_agent},
        )
        _disable_langextract_thinking(model)
        strict_schema_enabled = False
        if provider_config.supports_json_schema:
            strict_schema = OpenAISchema.from_schema_dict(_langextract_output_schema())
            strict_schema_enabled = _enable_strict_schema_when_supported(
                model, strict_schema, secrets=(key,)
            )
        examples = [ExampleData(text="item uses method; condition yields conclusion", extractions=[
            Extraction("entity", "item", attributes={"kind": "entity", "entity_type": "TestItem", "source_chunk_id": "example", "source_quote": "item"}),
            Extraction("entity", "method", attributes={"kind": "entity", "entity_type": "TestMethod", "source_chunk_id": "example", "source_quote": "method"}),
            Extraction("rule", "condition yields conclusion", attributes={"kind": "rule", "entity_type": "InterpretationRule", "semantic_type": "DEFINES_AS", "subject_logic": "SINGLE", "source_chunk_id": "example", "source_quote": "condition yields conclusion", "conditions": [{"text": "condition", "source_chunk_id": "example", "source_quote": "condition"}], "conclusion": {"text": "conclusion", "source_chunk_id": "example", "source_quote": "conclusion"}}),
            Extraction("relation", "item uses method", attributes={"kind": "relation", "relation_type": "ITEM_MEASURED_BY_METHOD", "source_text": "item", "target_text": "method", "source_chunk_id": "example", "source_quote": "item uses method"}),
        ])]
        for page_index in range(24):
            page_chunks = [chunk for chunk in chunks_tuple if chunk.page_index == page_index]
            window_id = f"page:{page_index:04d}"
            if saved.get("windows", {}).get(window_id, {}).get("status") == "success":
                continue
            text = "\n\n".join(f"[source_chunk_id={chunk.chunk_id}]\n{chunk.text}" for chunk in page_chunks)
            try:
                result = _run_with_transient_retry(
                    lambda: lx.extract(text, prompt_description=_langextract_prompt(), examples=examples,
                                       model=model, extraction_passes=1, max_char_buffer=12000,
                                       batch_length=1, max_workers=1, show_progress=False,
                                       use_schema_constraints=False,
                                       resolver_params={"suppress_parse_errors": False}),
                    secrets=(key,),
                )
                native, duplicate_extraction_count = _normalize_exact_duplicates(
                    _native_extractions(result)
                )
                _validate_page_extractions(native)
            except Exception as error:
                saved.setdefault("windows", {})[window_id] = {
                    "status": "failed", "page_index": page_index,
                    "chunk_ids": [chunk.chunk_id for chunk in page_chunks],
                    "error": _redact_error(error, (key,)),
                }
                atomic_write_json(checkpoint, saved)
                raise
            saved.setdefault("windows", {})[window_id] = {
                "status": "success", "page_index": page_index,
                "chunk_ids": [c.chunk_id for c in page_chunks],
                "duplicate_extraction_count": duplicate_extraction_count,
                "output": native,
            }
            atomic_write_json(checkpoint, saved)
    finally:
        client.close()
    expected_windows = {f"page:{page_index:04d}" for page_index in range(24)}
    successful_windows = {
        window_id for window_id, window in saved.get("windows", {}).items()
        if isinstance(window, Mapping) and window.get("status") == "success"
    }
    if successful_windows != expected_windows:
        raise RuntimeError(
            f"chapter extraction incomplete: {len(successful_windows)}/24 successful pages"
        )
    native_payload = {"extractions": [
        item for window_id, window in saved.get("windows", {}).items()
        if window_id in expected_windows and isinstance(window, Mapping)
        and window.get("status") == "success"
        for item in window.get("output", {}).get("extractions", [])
    ]}
    projection = adapt_langextract(native_payload, chunks_tuple)
    extraction = {"provider": "langextract", "model": f"{provider_config.provider}/{provider_config.model}", "status": _chapter_status(saved.get("windows", {})), "page_count": 24, "chunk_count": 44, "extractions": native_payload["extractions"], "review_queue_count": len(projection.review_queue)}
    (output_dir / "extraction.json").write_text(json.dumps(extraction, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    book_manifest = build_book_manifest_from_packages(book={"book_id": "clinical-hematology", "title": "Clinical Hematology", "edition": "source-package"}, source_manifest=source, chunk_manifest=chunk_manifest)
    pages = tuple(PageText(page["page_id"], (source_manifest.parent / page["raw_path"]).read_text(encoding="utf-8"), (source_manifest.parent / page["cleaned_path"]).read_text(encoding="utf-8")) for page in source["pages"])
    base_path = output_dir / "base-knowledge.sqlite"
    if not base_path.exists(): KnowledgeGraphBuilder().build(base_path, book_manifest, pages)
    graph = SemanticGraphBuilder().build(output_dir / "knowledge.sqlite", base_path, book_manifest, projection.records, projection.relations)
    (output_dir / "review-queue.json").write_text(json.dumps({"status": "candidate-only", "items": list(projection.review_queue), "counts": {"review_required": len(projection.review_queue)}}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"schema_version": "chapter-semantic-kg-run/v0.1", "provider": "langextract", "status": extraction["status"], "input": {"source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(), "chunk_manifest_sha256": hashlib.sha256(chunks_manifest.read_bytes()).hexdigest(), "pages": 24, "chunks": 44}, "graph": {"node_count": graph.node_count, "edge_count": graph.edge_count, "status_counts": dict(graph.status_counts), "approved_count": 0, "package_hash": graph.package_hash}, "configuration": {**provider_config.identity, "prompt": "langextract-semantic-graph-v0.1", "structured_output": "json_schema" if strict_schema_enabled else "json_object", "trust_env": False, "max_extractions_per_page": LANGEXTRACT_MAX_EXTRACTIONS_PER_PAGE, "max_relations_per_page": LANGEXTRACT_MAX_RELATIONS_PER_PAGE}}
    (output_dir / "run-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
