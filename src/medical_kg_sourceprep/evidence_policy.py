"""Versioned, fail-closed contracts for evidence-backed medical claims.

This module only validates provenance contracts. It does not retrieve sources,
compute report values, or infer medical content.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

from .book_sources import SourceProvenanceError, validate_book_manifest, validate_text_anchor


class EvidenceKind(str, Enum):
    REPORT = "report"
    COMPUTATION = "computation"
    BOOK = "book"


class SourceType(str, Enum):
    REPORT = "report"
    BOOK = "book"
    INDEX = "index"
    NETWORK = "network"
    MODEL_TEXT = "model_text"
    TASK_DOCUMENT = "task_document"


REPORT_SNAPSHOT_SCHEMA_VERSION = "report-snapshot/v0.2"
_REPORT_SOURCE_FIELDS = frozenset({
    "citation_id", "source_type", "source_id", "version", "schema_version", "payload", "payload_hash",
})
_REPORT_PAYLOAD_FIELDS = frozenset({"schema_version", "items"})
_REPORT_ITEM_FIELDS = frozenset({"item_id", "value", "unit", "reference_interval", "report_flag"})
_REPORT_INTERVAL_FIELDS = frozenset({"lower", "upper", "lower_inclusive", "upper_inclusive"})
_REPORT_FLAGS = frozenset({"low", "normal", "high"})


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    claim_type: str
    strength: str = "assertive"


@dataclass(frozen=True)
class RuleVersionRef:
    rule_id: str
    version: str
    status: str


@dataclass(frozen=True)
class AtomicPredicateRef:
    predicate_id: str
    text: str
    citation_id: str


@dataclass(frozen=True)
class RuleMatchRef:
    match_id: str
    claim_id: str
    rule_id: str
    predicate_id: str
    citation_id: str
    rule_version: str | None = None


@dataclass(frozen=True)
class ComputationTraceRef:
    trace_id: str
    algorithm_version: str
    predicate_ids: tuple[str, ...]
    claim_id: str
    citation_ids: tuple[str, ...] = ()
    report_citation_id: str | None = None


@dataclass(frozen=True)
class EvidenceGap:
    subject_id: str
    reason: str
    required: bool


@dataclass(frozen=True)
class CitationBundle:
    claim: Claim
    rule: RuleVersionRef | None
    predicates: tuple[AtomicPredicateRef, ...]
    matches: tuple[RuleMatchRef, ...]
    computation: ComputationTraceRef | None
    sources: tuple[Mapping[str, Any], ...]
    gaps: tuple[EvidenceGap, ...] = ()
    conflicts: tuple[str, ...] = ()
    strength: str = "assertive"


_APPROVED_RULE_STATUS = "approved"
_MEDICAL_CLAIM_TYPES = frozenset({"medical_explanation", "medical_recommendation"})
_FACT_CLAIM_TYPES = frozenset({"reported_fact", "report_anomaly"})
_WEAKENED_STRENGTHS = frozenset({"qualified", "cautious", "uncertain"})


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value


def _source_type(source: Mapping[str, Any]) -> str:
    value = source.get("source_type")
    if isinstance(value, SourceType):
        return value.value
    if isinstance(value, EvidenceKind):
        return value.value
    return value if isinstance(value, str) else ""


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_report_source(
    *, citation_id: str, source_id: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a minimal immutable REPORT citation from a canonical snapshot."""
    if not isinstance(payload, Mapping):
        raise ValueError("REPORT snapshot payload must be a mapping")
    _validate_report_payload(payload)
    payload_hash = _canonical_hash(payload)
    source = {
        "citation_id": citation_id,
        "source_type": SourceType.REPORT.value,
        "source_id": source_id,
        "version": payload_hash,
        "schema_version": REPORT_SNAPSHOT_SCHEMA_VERSION,
        "payload": dict(payload),
        "payload_hash": payload_hash,
    }
    _validate_report_source_shape(source)
    return source


def _validate_report_source_shape(source: Mapping[str, Any]) -> None:
    if set(source) != _REPORT_SOURCE_FIELDS:
        raise ValueError("REPORT source has an unsupported field set")
    if source.get("source_type") != SourceType.REPORT.value:
        raise ValueError("REPORT source_type is unsupported")
    _require_text(source.get("citation_id"), "REPORT citation_id")
    _require_text(source.get("source_id"), "REPORT source_id")
    _require_text(source.get("version"), "REPORT version")
    if source.get("schema_version") != REPORT_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("REPORT snapshot schema version is unsupported")
    if not isinstance(source.get("payload"), Mapping):
        raise ValueError("REPORT citation requires a snapshot payload")
    if not isinstance(source.get("payload_hash"), str):
        raise ValueError("REPORT snapshot hash is invalid")


def _finite_decimal_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"REPORT {label} must be a finite decimal string")
    try:
        from decimal import Decimal, InvalidOperation
        parsed = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ValueError(f"REPORT {label} must be a finite decimal string") from None
    if not parsed.is_finite():
        raise ValueError(f"REPORT {label} must be a finite decimal string")


def _validate_report_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != _REPORT_PAYLOAD_FIELDS:
        raise ValueError("REPORT snapshot payload has an unsupported field set")
    if payload.get("schema_version") != REPORT_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("REPORT snapshot schema version is unsupported")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("REPORT snapshot items must be a list")
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping) or set(item) != _REPORT_ITEM_FIELDS:
            raise ValueError("REPORT snapshot item has an unsupported field set")
        item_id = item.get("item_id")
        _require_text(item_id, "REPORT item_id")
        if item_id in seen:
            raise ValueError("REPORT snapshot item IDs must be unique")
        seen.add(item_id)
        _finite_decimal_text(item.get("value"), "value")
        if not isinstance(item.get("unit"), str) or not item["unit"].strip():
            raise ValueError("REPORT unit must be a non-empty string")
        interval = item.get("reference_interval")
        if not isinstance(interval, Mapping) or set(interval) != _REPORT_INTERVAL_FIELDS:
            raise ValueError("REPORT reference_interval has an unsupported field set")
        for bound in ("lower", "upper"):
            _finite_decimal_text(interval.get(bound), f"reference_interval.{bound}")
        if not isinstance(interval["lower_inclusive"], bool) or not isinstance(interval["upper_inclusive"], bool):
            raise ValueError("REPORT interval inclusivity must be boolean")
        flag = item.get("report_flag")
        if flag is not None and flag not in _REPORT_FLAGS:
            raise ValueError("REPORT report_flag is unsupported")


def _validate_report_provenance(source: Mapping[str, Any]) -> None:
    _validate_report_source_shape(source)
    payload = source.get("payload")
    _validate_report_payload(payload)
    payload_hash = source.get("payload_hash")
    if not isinstance(payload_hash, str) or payload_hash != _canonical_hash(payload):
        raise ValueError("REPORT snapshot hash mismatch")
    if source.get("version") != payload_hash:
        raise ValueError("REPORT snapshot version drift detected")


def _validate_book_provenance(
    source: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Resolve one citation through caller-approved manifest and TextAnchor data."""
    if registry is None:
        raise ValueError("medical BOOK evidence requires an approved provenance registry")
    source_id = _require_text(source.get("source_id"), "source_id")
    approved = registry.get(source_id)
    if not isinstance(approved, Mapping):
        raise ValueError("BOOK source is not present in the approved registry")
    if approved.get("status") != "approved":
        raise ValueError("BOOK manifest is not approved")
    manifest = source.get("manifest")
    anchor = source.get("anchor")
    raw_text = source.get("raw_text")
    cleaned_text = source.get("cleaned_text")
    if not isinstance(manifest, Mapping) or not isinstance(anchor, Mapping):
        raise ValueError("BOOK citation requires a manifest and TextAnchor")
    if not isinstance(raw_text, str) or not isinstance(cleaned_text, str):
        raise ValueError("BOOK citation requires raw and cleaned anchor text")
    manifest_hash = source.get("manifest_hash")
    if manifest_hash != manifest.get("content_sha256") or manifest_hash != approved.get("manifest_hash"):
        raise ValueError("BOOK manifest hash mismatch")
    if source.get("version") != manifest_hash or approved.get("version") != manifest_hash:
        raise ValueError("BOOK manifest version drift detected")
    if source_id != manifest.get("book", {}).get("book_id") or approved.get("book_id") != source_id:
        raise ValueError("BOOK manifest identity is not bound to source_id")
    if source.get("source_hash") not in {manifest.get("pdf", {}).get("sha256"), manifest.get("markdown", {}).get("sha256")}:
        raise ValueError("BOOK source hash mismatch")
    if source.get("anchor_hash") != _canonical_hash(anchor):
        raise ValueError("TextAnchor hash mismatch")
    try:
        validate_book_manifest(manifest)
        validate_text_anchor(anchor, raw_text, cleaned_text)
    except SourceProvenanceError as error:
        raise ValueError(f"invalid approved BOOK provenance: {error}") from error
    page = next((item for item in manifest["pages"] if item["page_id"] == anchor.get("page_id")), None)
    if page is None:
        raise ValueError("TextAnchor references an unknown manifest page")
    if anchor.get("raw_sha256") != page["raw_sha256"] or anchor.get("cleaned_sha256") != page["cleaned_sha256"]:
        raise ValueError("TextAnchor source hash drift detected")


def _validate_sources(bundle: CitationBundle) -> dict[str, Mapping[str, Any]]:
    sources: dict[str, Mapping[str, Any]] = {}
    for source in bundle.sources:
        if not isinstance(source, Mapping):
            raise ValueError("source citation must be a mapping")
        citation_id = _require_text(source.get("citation_id"), "citation_id")
        if citation_id in sources:
            raise ValueError("citation IDs must be unique")
        source_id = _require_text(source.get("source_id"), "source_id")
        _require_text(source.get("version"), "source version")
        source_type = _source_type(source)
        if source_type not in {SourceType.BOOK.value, SourceType.REPORT.value}:
            raise ValueError("index, network, model, and task-document sources are not evidence")
        if source_type == SourceType.BOOK and source_id.startswith(("index:", "network:", "model:")):
            raise ValueError("BOOK citation must identify an allowed book source")
        if source_type == SourceType.REPORT:
            _validate_report_provenance(source)
        sources[citation_id] = source
    return sources


def _validate_identifiers(bundle: CitationBundle) -> None:
    claim = bundle.claim
    _require_text(claim.claim_id, "claim_id")
    _require_text(claim.text, "claim text")
    _require_text(claim.claim_type, "claim_type")
    if claim.strength not in {"assertive", *_WEAKENED_STRENGTHS}:
        raise ValueError("claim strength is unsupported")
    if bundle.strength not in {"assertive", *_WEAKENED_STRENGTHS}:
        raise ValueError("bundle strength is unsupported")
    predicate_ids = set()
    for predicate in bundle.predicates:
        _require_text(predicate.predicate_id, "predicate_id")
        _require_text(predicate.text, "predicate text")
        if predicate.predicate_id in predicate_ids:
            raise ValueError("predicate IDs must be unique")
        predicate_ids.add(predicate.predicate_id)
    match_ids = set()
    for match in bundle.matches:
        _require_text(match.match_id, "match_id")
        if match.match_id in match_ids:
            raise ValueError("match IDs must be unique")
        match_ids.add(match.match_id)


def validate_citation_bundle(
    bundle: CitationBundle,
    approved_book_registry: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[()]:
    """Validate one complete evidence contract and return an empty issue tuple.

    Validation is intentionally strict: callers receive no usable result when
    any required provenance edge, source, rule, or condition is missing.
    """
    if not isinstance(bundle, CitationBundle):
        raise ValueError("bundle must be a CitationBundle")
    _validate_identifiers(bundle)
    sources = _validate_sources(bundle)
    claim = bundle.claim
    medical = claim.claim_type in _MEDICAL_CLAIM_TYPES
    fact = claim.claim_type in _FACT_CLAIM_TYPES
    if not medical and not fact:
        raise ValueError("claim_type must be a supported fact or medical explanation")
    if any(gap.required and gap.reason.lower() == "unknown" for gap in bundle.gaps):
        raise ValueError("unknown required condition creates an evidence gap")
    if bundle.conflicts and bundle.strength not in _WEAKENED_STRENGTHS and claim.strength not in _WEAKENED_STRENGTHS:
        raise ValueError("conflict requires qualified or weaker claim wording")

    if fact:
        if not sources or any(_source_type(source) != SourceType.REPORT.value for source in sources.values()):
            raise ValueError("reported facts require REPORT evidence")
        if bundle.rule is not None or bundle.predicates or bundle.matches or bundle.computation:
            raise ValueError("reported facts cannot be upgraded with an incomplete medical chain")
        return ()

    if not sources or not any(_source_type(source) == SourceType.REPORT.value for source in sources.values()):
        raise ValueError("medical explanation requires REPORT evidence")
    if not any(_source_type(source) == SourceType.BOOK.value for source in sources.values()):
        raise ValueError("medical explanation requires BOOK evidence")
    for source in sources.values():
        if _source_type(source) == SourceType.BOOK.value:
            _validate_book_provenance(source, approved_book_registry)
    rule = bundle.rule
    if rule is None:
        raise ValueError("medical explanation requires a rule")
    _require_text(rule.rule_id, "rule_id")
    _require_text(rule.version, "rule version")
    if rule.status != _APPROVED_RULE_STATUS:
        raise ValueError("only approved rules may support medical claims")
    if bundle.computation is None:
        raise ValueError("medical explanation requires COMPUTATION evidence")
    computation = bundle.computation
    _require_text(computation.trace_id, "trace_id")
    _require_text(computation.algorithm_version, "algorithm version")
    if computation.claim_id != claim.claim_id or not computation.predicate_ids:
        raise ValueError("computation trace must bind claim and predicates")
    report_ids = {citation_id for citation_id, source in sources.items() if _source_type(source) == SourceType.REPORT.value}
    if computation.report_citation_id not in report_ids:
        raise ValueError("COMPUTATION must explicitly reference REPORT evidence")
    predicates = {predicate.predicate_id: predicate for predicate in bundle.predicates}
    matches = {match.match_id: match for match in bundle.matches}
    if not predicates or not matches:
        raise ValueError("every rule condition and conclusion needs a citation anchor")
    matched_predicates: set[str] = set()
    claim_citations: set[str] = set()
    for match in matches.values():
        if match.claim_id != claim.claim_id or match.rule_id != rule.rule_id:
            raise ValueError("rule match is not bound to the claim and rule")
        if match.rule_version is not None and match.rule_version != rule.version:
            raise ValueError("rule version drift detected")
        predicate = predicates.get(match.predicate_id)
        if predicate is None or predicate.citation_id != match.citation_id:
            raise ValueError("atomic predicate is missing its citation anchor")
        source = sources.get(match.citation_id)
        if source is None or _source_type(source) != SourceType.BOOK.value:
            raise ValueError("medical condition anchors must cite BOOK evidence")
        matched_predicates.add(match.predicate_id)
        claim_citations.add(match.citation_id)
    if matched_predicates != set(predicates) or set(computation.predicate_ids) != set(predicates):
        raise ValueError("every atomic condition must map one-to-one through the computation trace")
    if not claim_citations:
        raise ValueError("medical conclusion has no citation anchor")
    if any(citation_id not in sources for citation_id in computation.citation_ids):
        raise ValueError("computation citation is not present in the bundle")
    if computation.report_citation_id not in computation.citation_ids:
        raise ValueError("COMPUTATION citation list must include REPORT evidence")
    return ()


__all__ = [
    "AtomicPredicateRef", "CitationBundle", "Claim", "ComputationTraceRef",
    "EvidenceGap", "EvidenceKind", "RuleMatchRef", "RuleVersionRef", "SourceType",
    "REPORT_SNAPSHOT_SCHEMA_VERSION", "build_report_source", "validate_citation_bundle",
]
