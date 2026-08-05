"""Deterministic, standard-library sparse TF-IDF indexes for evidence chunks."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "evidence-vector-index/v0.3"
VECTORIZER_VERSION = "char-ngram-tfidf/v1"


class VectorIndexError(ValueError):
    pass


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    similarity: float
    vector_kind: str = "sparse_tfidf"


def normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _grams(value: str, minimum: int, maximum: int) -> tuple[str, ...]:
    text = re.sub(r"\s+", " ", normalize(value))
    return tuple(text[index:index + size] for size in range(minimum, maximum + 1) for index in range(len(text) - size + 1) if text[index:index + size].strip())


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evidence_rows(path: Path) -> tuple[str, tuple[tuple[str, str, str], ...]]:
    try:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise VectorIndexError("evidence SQLite integrity check failed")
            meta = dict(connection.execute("SELECT key, value FROM metadata"))
            if meta.get("schema_version") != "evidence-index/v0.1" or not re.fullmatch(r"[0-9a-f]{64}", meta.get("chunk_manifest_sha256", "")):
                raise VectorIndexError("unsupported evidence index metadata")
            rows = tuple(connection.execute("SELECT chunk_id, text, chunk_sha256 FROM chunks ORDER BY chunk_id"))
    except sqlite3.Error as error:
        raise VectorIndexError("evidence index is unreadable") from error
    if not rows or any(not isinstance(i, str) or not i or _sha(text) != digest for i, text, digest in rows):
        raise VectorIndexError("evidence chunk binding is invalid")
    return meta["chunk_manifest_sha256"], rows


def _parse_vector(payload: str, dimension: int) -> dict[str, float]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate vector dimension")
            result[key] = value
        return result

    try:
        vector = json.loads(payload, object_pairs_hook=pairs, parse_constant=reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise VectorIndexError("vector payload is invalid") from error
    if not isinstance(vector, dict) or not vector:
        raise VectorIndexError("vector payload is invalid")
    parsed: dict[str, float] = {}
    for key, value in vector.items():
        if not isinstance(key, str) or not re.fullmatch(r"(?:0|[1-9][0-9]*)", key):
            raise VectorIndexError("vector payload is invalid")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise VectorIndexError("vector payload is invalid")
        if int(key) >= dimension or value < 0:
            raise VectorIndexError("vector payload is invalid")
        parsed[key] = float(value)
    norm = math.sqrt(sum(value * value for value in parsed.values()))
    if norm == 0 or not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=2e-11):
        raise VectorIndexError("vector payload is invalid")
    return parsed


def build_vector_index(evidence_index: Path, output: Path, *, ngram_min: int = 2, ngram_max: int = 3) -> dict[str, object]:
    """Atomically build a reproducible sparse index, refusing output overwrite."""
    evidence_index, output = Path(evidence_index), Path(output)
    if output.exists():
        raise VectorIndexError("output already exists")
    if not 1 <= ngram_min <= ngram_max <= 8:
        raise VectorIndexError("invalid character ngram range")
    manifest, rows = _evidence_rows(evidence_index)
    terms = {chunk_id: Counter(_grams(text, ngram_min, ngram_max)) for chunk_id, text, _ in rows}
    document_frequency = Counter(term for counts in terms.values() for term in counts)
    vocabulary = {term: position for position, term in enumerate(sorted(document_frequency))}
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".sqlite", dir=output.parent)
    os.close(fd); staging = Path(raw)
    try:
        with sqlite3.connect(staging) as connection:
            connection.executescript("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL); CREATE TABLE vectors (chunk_id TEXT PRIMARY KEY, chunk_sha256 TEXT NOT NULL, vector_json TEXT NOT NULL);")
            metadata = {"schema_version": SCHEMA_VERSION, "vector_kind": "sparse_tfidf", "vectorizer_version": VECTORIZER_VERSION, "ngram_min": str(ngram_min), "ngram_max": str(ngram_max), "dimension": str(len(vocabulary)), "evidence_manifest_sha256": manifest}
            connection.executemany("INSERT INTO metadata VALUES (?, ?)", sorted(metadata.items()))
            total = len(rows)
            for chunk_id, _, digest in rows:
                weights = {str(vocabulary[term]): count * (math.log((total + 1) / (document_frequency[term] + 1)) + 1) for term, count in terms[chunk_id].items()}
                norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
                vector = {key: round(value / norm, 12) for key, value in sorted(weights.items(), key=lambda item: int(item[0]))}
                connection.execute("INSERT INTO vectors VALUES (?, ?, ?)", (chunk_id, digest, json.dumps(vector, separators=(",", ":"), sort_keys=True)))
            connection.execute("CREATE TABLE vocabulary (term TEXT PRIMARY KEY, dimension INTEGER NOT NULL, document_frequency INTEGER NOT NULL)")
            connection.executemany("INSERT INTO vocabulary VALUES (?, ?, ?)", ((term, dim, document_frequency[term]) for term, dim in vocabulary.items()))
        os.replace(staging, output)
    except Exception:
        staging.unlink(missing_ok=True); raise
    return {"schema_version": SCHEMA_VERSION, "chunk_count": len(rows), "dimension": len(vocabulary), "evidence_manifest_sha256": manifest}


def query_vector_index(index: Path, evidence_index: Path, query: str, *, top_k: int = 5, threshold: float = 0.12) -> tuple[VectorHit, ...]:
    if not isinstance(query, str) or not normalize(query): return ()
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1: raise VectorIndexError("top_k must be positive")
    manifest, evidence = _evidence_rows(Path(evidence_index)); hashes = {row[0]: row[2] for row in evidence}
    try:
        with sqlite3.connect(f"file:{Path(index).resolve()}?mode=ro", uri=True) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok": raise VectorIndexError("vector SQLite integrity check failed")
            meta = dict(connection.execute("SELECT key, value FROM metadata"))
            if meta.get("schema_version") != SCHEMA_VERSION or meta.get("vector_kind") != "sparse_tfidf" or meta.get("evidence_manifest_sha256") != manifest: raise VectorIndexError("vector index metadata drift")
            nmin, nmax, dimension = int(meta["ngram_min"]), int(meta["ngram_max"]), int(meta["dimension"])
            vocabulary = {
                term: (dimension, frequency)
                for term, dimension, frequency in connection.execute(
                    "SELECT term, dimension, document_frequency FROM vocabulary"
                )
            }
            rows = tuple(connection.execute("SELECT chunk_id, chunk_sha256, vector_json FROM vectors ORDER BY chunk_id"))
    except (sqlite3.Error, KeyError, ValueError) as error:
        raise VectorIndexError("vector index is unreadable") from error
    if dimension != len(vocabulary) or {row[0]: row[1] for row in rows} != hashes:
        raise VectorIndexError("vector chunk binding drift")
    if any(
        not isinstance(vector_dimension, int)
        or isinstance(vector_dimension, bool)
        or vector_dimension < 0
        or vector_dimension >= dimension
        or not isinstance(frequency, int)
        or isinstance(frequency, bool)
        or frequency < 1
        or frequency > len(rows)
        for vector_dimension, frequency in vocabulary.values()
    ):
        raise VectorIndexError("vector vocabulary is invalid")
    parsed_rows = tuple(
        (chunk_id, _parse_vector(payload, dimension)) for chunk_id, _, payload in rows
    )
    counts = Counter(_grams(query, nmin, nmax))
    weights = {
        str(vocabulary[term][0]): count * (math.log((len(rows) + 1) / (vocabulary[term][1] + 1)) + 1)
        for term, count in counts.items()
        if term in vocabulary
    }
    norm = math.sqrt(sum(value * value for value in weights.values()))
    if not norm: return ()
    query_vector = {key: value / norm for key, value in weights.items()}
    hits = [
        VectorHit(chunk_id, round(sum(query_vector.get(key, 0.0) * value for key, value in vector.items()), 12))
        for chunk_id, vector in parsed_rows
    ]
    return tuple(hit for hit in sorted(hits, key=lambda hit: (-hit.similarity, hit.chunk_id)) if hit.similarity >= threshold)[:top_k]
