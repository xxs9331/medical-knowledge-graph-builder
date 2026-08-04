"""Offline, deterministic retrieval over caller-supplied rule-like records."""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


_COMPONENTS = (
    "exact_phrase",
    "alias",
    "title",
    "rule_type",
    "condition_match",
    "bm25",
    "context",
)
_TEXT_FIELDS = ("standard_name", "title", "text")


@dataclass(frozen=True)
class RetrievalRecord:
    """Minimal read-only adapter for a KnowledgeRule or evidence record."""

    record_id: str
    standard_name: str = ""
    title: str = ""
    text: str = ""
    rule_type: str = ""
    conditions: tuple[str, ...] = ()
    anchor_ids: tuple[str, ...] = ()
    parent_id: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """A ranked record with stable, additive scoring evidence."""

    record_id: str
    score: float
    score_components: Mapping[str, float]
    reasons: tuple[str, ...]
    anchor_ids: tuple[str, ...]
    record: Any


def _normalized(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value)).casefold().strip()


def _tokens(value: str) -> tuple[str, ...]:
    latin = re.findall(r"[a-z0-9]+", value)
    han = re.findall(r"[\u3400-\u9fff]+", value)
    return tuple(sorted(set(latin + han)))


def _value(record: object, name: str, default: object = "") -> object:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _text(record: object, name: str) -> str:
    value = _value(record, name)
    return value if isinstance(value, str) else ""


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Iterable):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _record_id(record: object) -> str:
    value = _value(record, "record_id", _value(record, "id", ""))
    if not isinstance(value, str) or not value:
        raise ValueError("each retrieval record requires a non-empty record_id or id")
    return value


def _bm25(query_terms: tuple[str, ...], text: str, corpus: Sequence[str]) -> float:
    if not query_terms or not text:
        return 0.0
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    total_length = sum(max(1, len(_tokens(item))) for item in corpus)
    average_length = total_length / max(1, len(corpus))
    score = 0.0
    for term in query_terms:
        frequency = tokens.count(term)
        if not frequency:
            continue
        document_frequency = sum(term in _tokens(item) for item in corpus)
        inverse_frequency = math.log(1 + (len(corpus) - document_frequency + 0.5) / (document_frequency + 0.5))
        score += inverse_frequency * (frequency * 2.0) / (frequency + 1.2 * (1 - 0.75 + 0.75 * len(tokens) / average_length))
    return score


def _fts_scores(records: Sequence[object], query_terms: tuple[str, ...]) -> dict[str, float]:
    """Use FTS5 when available; callers retain deterministic BM25 fallback."""
    if not query_terms:
        return {}
    try:
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE VIRTUAL TABLE candidates USING fts5(record_id UNINDEXED, body)")
        connection.executemany(
            "INSERT INTO candidates(record_id, body) VALUES (?, ?)",
            [(_record_id(record), " ".join(_text(record, field) for field in _TEXT_FIELDS)) for record in records],
        )
        query = " OR ".join('"' + term.replace('"', '""') + '"' for term in query_terms)
        rows = connection.execute(
            "SELECT record_id, bm25(candidates) AS score FROM candidates WHERE candidates MATCH ?",
            (query,),
        ).fetchall()
        connection.close()
        # SQLite FTS5 ranks stronger hits with lower (usually negative) values.
        return {record_id: max(0.0, -float(score)) for record_id, score in rows}
    except sqlite3.Error:
        return {}


def retrieve(
    query: str,
    records: Sequence[object],
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
    rule_types: Sequence[str] = (),
    required_conditions: Sequence[str] = (),
    top_k: int = 5,
    include_context: bool = True,
    use_fts: bool = True,
) -> tuple[RetrievalResult, ...]:
    """Return explainable evidence-first results, or an empty tuple on no reliable hit."""
    if not isinstance(query, str) or not (normalized_query := _normalized(query)):
        return ()
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    query_terms = _tokens(normalized_query)
    if not query_terms:
        return ()
    allowed_types = {_normalized(item) for item in rule_types if _normalized(item)}
    required = {_normalized(item) for item in required_conditions if _normalized(item)}
    canonical_aliases = {
        record_id: tuple(_normalized(alias) for alias in values if _normalized(alias))
        for record_id, values in (aliases or {}).items()
    }
    corpus = [" ".join(_text(record, field) for field in _TEXT_FIELDS) for record in records]
    fts_scores = _fts_scores(records, query_terms) if use_fts else {}
    candidates: list[RetrievalResult] = []
    for record in records:
        record_id = _record_id(record)
        # Parent/adjacent material is context only and must never outrank evidence.
        if isinstance(_value(record, "parent_id", None), str):
            continue
        rule_type = _normalized(_text(record, "rule_type"))
        conditions = {_normalized(item) for item in _string_tuple(_value(record, "conditions", ())) if _normalized(item)}
        if allowed_types and rule_type not in allowed_types:
            continue
        if required and not required <= conditions:
            continue
        standard_name, title, body = (_normalized(_text(record, field)) for field in _TEXT_FIELDS)
        record_aliases = canonical_aliases.get(record_id, ())
        exact_phrase = 100.0 if normalized_query in {standard_name, title} else 0.0
        alias_score = 80.0 if normalized_query in record_aliases else 0.0
        title_score = 30.0 if normalized_query in title and normalized_query != title else 0.0
        type_score = 8.0 if allowed_types and rule_type in allowed_types else 0.0
        condition_score = 8.0 if required and required <= conditions else 0.0
        fallback_bm25 = _bm25(query_terms, " ".join((standard_name, title, body)), corpus)
        bm25_score = fts_scores.get(record_id, fallback_bm25)
        # A complete query phrase, explicit alias, title match, or tokenized FTS/BM25 match is required.
        if not any((exact_phrase, alias_score, title_score, bm25_score)):
            continue
        reasons = []
        if exact_phrase:
            reasons.append("standard_name_exact" if normalized_query == standard_name else "title_exact")
        if alias_score:
            reasons.append("alias_exact")
        if title_score:
            reasons.append("title_phrase")
        if bm25_score:
            reasons.append("fts5_bm25" if record_id in fts_scores else "bm25_fallback")
        components = {
            "exact_phrase": exact_phrase,
            "alias": alias_score,
            "title": title_score,
            "rule_type": type_score,
            "condition_match": condition_score,
            "bm25": round(bm25_score, 6),
            "context": 0.0,
        }
        anchors = _string_tuple(_value(record, "anchor_ids", ()))
        candidates.append(RetrievalResult(record_id, round(sum(components.values()), 6), components, tuple(reasons), anchors, record))
    ranked = sorted(candidates, key=lambda result: (-result.score, result.record_id))[:top_k]
    if not include_context or len(ranked) == top_k:
        return tuple(ranked)
    selected = {result.record_id for result in ranked}
    primary_ids = set(selected)
    contexts: list[RetrievalResult] = []
    for record in records:
        parent_id = _value(record, "parent_id", None)
        if not isinstance(parent_id, str) or parent_id not in primary_ids:
            continue
        record_id = _record_id(record)
        if record_id in selected:
            continue
        components = {component: 0.0 for component in _COMPONENTS}
        components["context"] = 1.0
        contexts.append(
            RetrievalResult(
                record_id,
                1.0,
                components,
                ("parent_context",),
                _string_tuple(_value(record, "anchor_ids", ())),
                record,
            )
        )
        selected.add(record_id)
        if len(ranked) + len(contexts) == top_k:
            break
    return tuple(ranked + contexts)


def retrieve_hybrid(
    query: str,
    records: Sequence[object],
    *,
    vector_hits: Sequence[object] = (),
    graph_hits: Sequence[object] = (),
    top_k: int = 5,
) -> tuple[RetrievalResult, ...]:
    """Fuse caller-provided read-only channel hits without weakening exact matches."""
    lexical = retrieve(query, records, top_k=max(top_k, len(records)), include_context=False)
    by_id = {result.record_id: result for result in lexical}
    vector = {_value(hit, "chunk_id", ""): float(_value(hit, "similarity", 0.0)) for hit in vector_hits}
    graph = {_value(hit, "chunk_id", ""): float(_value(hit, "graph_score", 0.0)) for hit in graph_hits}
    candidates = set(by_id) | {item for item, score in vector.items() if item and score >= 0.12} | set(graph)
    raw_records = {_record_id(record): record for record in records}
    result: list[RetrievalResult] = []
    for record_id in sorted(candidates):
        lexical_result = by_id.get(record_id)
        record = raw_records.get(record_id, lexical_result.record if lexical_result else None)
        if record is None:
            continue
        components = dict(lexical_result.score_components) if lexical_result else {name: 0.0 for name in _COMPONENTS}
        components.update({"vector": round(vector.get(record_id, 0.0), 6), "graph": round(graph.get(record_id, 0.0), 6)})
        # Exact and alias identities dominate; auxiliary channels are bounded recall/ranking evidence.
        score = components["exact_phrase"] + components["alias"] + components["title"] + components["bm25"] + components["vector"] + components["graph"]
        reasons = tuple((lexical_result.reasons if lexical_result else ()) + (("vector_tfidf",) if components["vector"] else ()) + (("graph_path",) if components["graph"] else ()))
        anchors = lexical_result.anchor_ids if lexical_result else _string_tuple(_value(record, "anchor_ids", ()))
        result.append(RetrievalResult(record_id, round(score, 6), components, reasons, anchors, record))
    return tuple(sorted(result, key=lambda item: (-item.score, item.record_id))[:top_k])


__all__ = ["RetrievalRecord", "RetrievalResult", "retrieve", "retrieve_hybrid"]
