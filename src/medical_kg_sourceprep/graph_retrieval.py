"""Read-only graph-to-evidence candidate projection."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path


class GraphRetrievalError(ValueError):
    pass


@dataclass(frozen=True)
class GraphHit:
    chunk_id: str
    graph_score: float
    anchor_ids: tuple[str, ...]
    path_relations: tuple[str, ...]


def _norm(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _evidence_by_chunk_id(path: Path) -> dict[str, str]:
    try:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
            rows = tuple(connection.execute("SELECT chunk_id, text, chunk_sha256 FROM chunks"))
    except sqlite3.Error as error:
        raise GraphRetrievalError("evidence index is unreadable") from error
    result = {}
    for chunk_id, text, digest in rows:
        if hashlib.sha256(text.encode()).hexdigest() != digest:
            raise GraphRetrievalError("evidence chunk hash is invalid")
        result[chunk_id] = digest
    return result


def graph_retrieve(knowledge_db: Path, evidence_index: Path, query: str, *, top_k: int = 5, max_hops: int = 3) -> tuple[GraphHit, ...]:
    """Find graph paths and project only exact hash-bound graph chunks to evidence."""
    if not isinstance(query, str) or not _norm(query): return ()
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1: raise GraphRetrievalError("top_k must be positive")
    evidence = _evidence_by_chunk_id(Path(evidence_index)); needle = _norm(query)
    try:
        with sqlite3.connect(f"file:{Path(knowledge_db).resolve()}?mode=ro", uri=True) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok": raise GraphRetrievalError("knowledge SQLite integrity check failed")
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if metadata.get("schema_version") != "knowledge-graph/v0.2": raise GraphRetrievalError("unsupported knowledge graph schema")
            nodes = {node_id: (kind, payload) for node_id, kind, payload in connection.execute("SELECT node_id, node_type, payload FROM nodes")}
            edges = tuple(connection.execute("SELECT source_id, target_id, relation FROM edges"))
            anchors = tuple(connection.execute("SELECT anchor_id, node_id FROM anchors"))
            chunks = {chunk_id: content for chunk_id, content in connection.execute("SELECT chunk_id, content FROM chunk_text")}
    except sqlite3.Error as error:
        raise GraphRetrievalError("knowledge graph is unreadable") from error
    if any(chunk not in nodes or nodes[chunk][0] != "EvidenceChunk" for chunk in chunks): raise GraphRetrievalError("graph chunk references are invalid")
    graph: dict[str, list[tuple[str, str]]] = {}
    for source, target, relation in edges:
        if source not in nodes or target not in nodes: raise GraphRetrievalError("graph has dangling edge")
        graph.setdefault(source, []).append((target, relation)); graph.setdefault(target, []).append((source, relation))
    starts = set()
    for node_id, (_, payload) in nodes.items():
        try: searchable = json.dumps(json.loads(payload), ensure_ascii=False, sort_keys=True)
        except (TypeError, json.JSONDecodeError): raise GraphRetrievalError("graph node payload is invalid")
        if needle in _norm(searchable): starts.add(node_id)
    starts.update(chunk_id for chunk_id, content in chunks.items() if needle in _norm(content))
    anchors_by_node: dict[str, list[str]] = {}
    for anchor_id, node_id in anchors:
        if node_id not in nodes: raise GraphRetrievalError("graph has dangling anchor")
        anchors_by_node.setdefault(node_id, []).append(anchor_id)
    candidates: list[GraphHit] = []
    for start in sorted(starts):
        queue = deque([(start, (), (start,))]); visited = {start}
        while queue:
            node, relations, path = queue.popleft()
            if node in chunks:
                digest = hashlib.sha256(chunks[node].encode()).hexdigest()
                evidence_digest = evidence.get(node)
                if evidence_digest is not None and evidence_digest != digest:
                    raise GraphRetrievalError("graph evidence chunk binding drift")
                if evidence_digest == digest:
                    anchor_ids = tuple(sorted({anchor for path_node in path for anchor in anchors_by_node.get(path_node, ())}))
                    candidates.append(GraphHit(node, round(1.0 / (1 + len(relations)), 6), anchor_ids, relations))
            if len(relations) < max_hops:
                for adjacent, relation in sorted(graph.get(node, ()), key=lambda item: (item[0], item[1])):
                    if adjacent not in visited:
                        visited.add(adjacent); queue.append((adjacent, relations + (relation,), path + (adjacent,)))
    best: dict[str, GraphHit] = {}
    for hit in candidates:
        if hit.chunk_id not in best or (-hit.graph_score, hit.path_relations) < (-best[hit.chunk_id].graph_score, best[hit.chunk_id].path_relations): best[hit.chunk_id] = hit
    return tuple(sorted(best.values(), key=lambda hit: (-hit.graph_score, hit.chunk_id))[:top_k])
