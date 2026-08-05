"""Read-only graph-to-evidence candidate projection."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import deque
from collections.abc import Mapping, Sequence
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
    matched_node_ids: tuple[str, ...] = ()
    matched_node_names: tuple[str, ...] = ()
    graph_status: str = "candidate"
    path_node_ids: tuple[str, ...] = ()
    path_node_names: tuple[str, ...] = ()
    path_node_types: tuple[str, ...] = ()
    path_triples: tuple[dict[str, object], ...] = ()
    match_mode: str = "exact"


@dataclass(frozen=True)
class GraphReasoningResult:
    paths: tuple[dict[str, object], ...] = ()
    rejections: tuple[dict[str, object], ...] = ()


def _norm(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _variant_norm(value: str) -> str:
    normalized = _norm(value)
    for source, target in (
        ("体积", "容积"),
        ("绝对数", "绝对值"),
        ("百分率", "百分数"),
        ("血红蛋白量", "血红蛋白含量"),
    ):
        normalized = normalized.replace(source, target)
    return normalized


def _node_terms(node: tuple[str, str, dict[str, object], tuple[dict[str, object], ...]]) -> tuple[tuple[str, str], ...]:
    node_type, name, properties, _evidence = node
    terms: list[tuple[str, str]] = [(name, "name")]
    for field in ("aliases", "synonyms", "candidate_keys"):
        values = properties.get(field, [])
        if isinstance(values, list):
            terms.extend((value, "alias") for value in values if isinstance(value, str) and value.strip())
    selector = properties.get("selector")
    if node_type == "ReferenceRange" and isinstance(selector, dict):
        cell_type, value_type = selector.get("细胞类型"), selector.get("数值类型")
        if isinstance(cell_type, str) and isinstance(value_type, str):
            terms.append((cell_type + value_type, "structural"))
    return tuple(terms)


def _match_nodes(
    nodes: Mapping[str, tuple[str, str, dict[str, object], tuple[dict[str, object], ...]]],
    query: str,
    *,
    node_types: set[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    needle = _norm(query)
    variant = _variant_norm(query)
    matches: dict[str, tuple[int, str]] = {}
    for node_id, node in nodes.items():
        node_type = node[0]
        if node_type in {"SourceLocator", "EntityAlias"} or (node_types and node_type not in node_types):
            continue
        for term, term_kind in _node_terms(node):
            candidate = _norm(term)
            mode: str | None = None
            rank = 99
            if needle == candidate:
                mode, rank = ("structural_match", 1) if term_kind == "structural" else ("exact", 0)
            elif variant == _variant_norm(term):
                mode, rank = "normalized_variant", 2
            elif len(needle) >= 4 and (needle.endswith(candidate) or candidate.endswith(needle)):
                mode, rank = "normalized_variant", 3
            if mode is not None and (node_id not in matches or rank < matches[node_id][0]):
                matches[node_id] = (rank, mode)
    if not matches:
        return ()
    best_rank = min(rank for rank, _mode in matches.values())
    return tuple(sorted(
        ((node_id, mode) for node_id, (rank, mode) in matches.items() if rank == best_rank),
        key=lambda item: (nodes[item[0]][1], item[0]),
    ))


def _candidate_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    if metadata.get("schema_version") not in {
        "chapter-knowledge-graph/v0.1", "chapter-knowledge-graph/v0.2",
    }:
        raise GraphRetrievalError("unsupported candidate graph schema")
    if metadata.get("approved") != "0" or metadata.get("status") != "candidate-only":
        raise GraphRetrievalError("candidate graph review boundary is invalid")
    return metadata


def _candidate_nodes(connection: sqlite3.Connection) -> dict[str, tuple[str, str, dict[str, object], tuple[dict[str, object], ...]]]:
    rows = tuple(connection.execute(
        "SELECT node_id, node_type, name, status, properties_json, evidence_json FROM nodes"
    ))
    nodes: dict[str, tuple[str, str, dict[str, object], tuple[dict[str, object], ...]]] = {}
    for node_id, node_type, name, status, properties_json, evidence_json in rows:
        properties = _json(properties_json, "node properties")
        if status != "candidate" or not isinstance(properties, dict):
            raise GraphRetrievalError("candidate graph node is invalid")
        nodes[node_id] = (node_type, name, properties, _evidence_items(_json(evidence_json, "node evidence")))
    return nodes


def _evidence_by_chunk_id(path: Path) -> dict[str, tuple[str, str]]:
    try:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
            rows = tuple(connection.execute("SELECT chunk_id, text, chunk_sha256 FROM chunks"))
    except sqlite3.Error as error:
        raise GraphRetrievalError("evidence index is unreadable") from error
    result = {}
    for chunk_id, text, digest in rows:
        if hashlib.sha256(text.encode()).hexdigest() != digest:
            raise GraphRetrievalError("evidence chunk hash is invalid")
        result[chunk_id] = (digest, text)
    return result


def _json(value: str, label: str) -> object:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise GraphRetrievalError(f"{label} JSON is invalid") from error


def _evidence_items(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise GraphRetrievalError("graph evidence must be a JSON array")
    if any(not isinstance(item, dict) for item in value):
        raise GraphRetrievalError("graph evidence item is invalid")
    return tuple(value)


def _project_candidate_evidence(
    items: tuple[dict[str, object], ...],
    evidence: dict[str, tuple[str, str]],
) -> tuple[str, ...]:
    projected: set[str] = set()
    for item in items:
        chunk_id, digest = item.get("chunk_id"), item.get("chunk_sha256")
        if isinstance(chunk_id, str) and chunk_id in evidence:
            actual_digest, _text = evidence[chunk_id]
            if isinstance(digest, str) and digest != actual_digest:
                raise GraphRetrievalError("graph evidence chunk binding drift")
            projected.add(chunk_id)
            continue
        quote = item.get("exact_quote") or item.get("source_quote") or item.get("quote")
        if not isinstance(quote, str) or not quote:
            continue
        matches = [candidate_id for candidate_id, (_digest, text) in evidence.items() if quote in text]
        if len(matches) == 1:
            projected.add(matches[0])
    return tuple(sorted(projected))


def _candidate_graph_retrieve(
    connection: sqlite3.Connection,
    evidence: dict[str, tuple[str, str]],
    query: str,
    top_k: int,
    max_hops: int,
) -> tuple[GraphHit, ...]:
    _candidate_metadata(connection)
    nodes = _candidate_nodes(connection)
    edge_rows = tuple(connection.execute(
        "SELECT subject_id, predicate, object_id, status, evidence_json FROM edges"
    ))
    graph: dict[str, list[tuple[str, str, tuple[dict[str, object], ...], str, str]]] = {}
    for source, predicate, target, status, evidence_json in edge_rows:
        if source not in nodes or target not in nodes or status != "candidate":
            raise GraphRetrievalError("candidate graph has a dangling or invalid edge")
        edge_evidence = _evidence_items(_json(evidence_json, "edge evidence"))
        graph.setdefault(source, []).append((target, predicate, edge_evidence, source, target))
        graph.setdefault(target, []).append((source, predicate, edge_evidence, source, target))
    starts = _match_nodes(nodes, query)
    candidates: list[GraphHit] = []
    for start, match_mode in starts:
        start_name = nodes[start][1]
        queue = deque([(start, (), (start,), (), nodes[start][3])])
        visited = {start}
        while queue:
            node_id, relations, path, path_triples, accumulated_evidence = queue.popleft()
            projected = _project_candidate_evidence(accumulated_evidence + nodes[node_id][3], evidence)
            score = round(1.0 / (1 + len(relations)), 6)
            for chunk_id in projected:
                candidates.append(GraphHit(
                    chunk_id, score, (), relations, (start,), (start_name,), "candidate-only",
                    path, tuple(nodes[value][1] for value in path),
                    tuple(nodes[value][0] for value in path), path_triples, match_mode,
                ))
            if len(relations) >= max_hops:
                continue
            for adjacent, relation, edge_evidence, subject, object_id in sorted(
                graph.get(node_id, ()), key=lambda item: (item[0], item[1])
            ):
                if adjacent in visited:
                    continue
                visited.add(adjacent)
                triple = {
                    "subject_id": subject,
                    "subject_name": nodes[subject][1],
                    "subject_type": nodes[subject][0],
                    "predicate": relation,
                    "object_id": object_id,
                    "object_name": nodes[object_id][1],
                    "object_type": nodes[object_id][0],
                    "traversal_direction": "forward" if node_id == subject else "reverse",
                }
                queue.append((
                    adjacent,
                    relations + (relation,),
                    path + (adjacent,),
                    path_triples + (triple,),
                    accumulated_evidence + edge_evidence,
                ))
    best: dict[str, GraphHit] = {}
    for hit in candidates:
        current = best.get(hit.chunk_id)
        if current is None or (-hit.graph_score, hit.path_relations) < (
            -current.graph_score, current.path_relations
        ):
            best[hit.chunk_id] = hit
    return tuple(sorted(best.values(), key=lambda hit: (-hit.graph_score, hit.chunk_id))[:top_k])


def graph_query_diagnostic(knowledge_db: Path, query: str) -> dict[str, object]:
    """Explain whether a graph lookup miss is caused by entity coverage or term linking."""
    if not isinstance(query, str) or not _norm(query):
        return {"query": query, "status": "entity_missing", "matches": []}
    try:
        with sqlite3.connect(f"file:{Path(knowledge_db).resolve()}?mode=ro", uri=True) as connection:
            _candidate_metadata(connection)
            nodes = _candidate_nodes(connection)
    except sqlite3.Error as error:
        raise GraphRetrievalError("knowledge graph is unreadable") from error
    matches = _match_nodes(nodes, query)
    if matches:
        modes = {mode for _node_id, mode in matches}
        status = "structural_match" if modes == {"structural_match"} else "matched"
        return {
            "query": query,
            "status": status,
            "match_mode": sorted(modes)[0],
            "matches": [
                {"node_id": node_id, "node_type": nodes[node_id][0], "name": nodes[node_id][1], "match_mode": mode}
                for node_id, mode in matches
            ],
        }
    abbreviation_like = bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9._+-]*[#%]?", query.strip()))
    return {
        "query": query,
        "status": "alias_missing" if abbreviation_like else "entity_missing",
        "matches": [],
    }


def graph_reasoning_paths(
    knowledge_db: Path,
    evidence_index: Path,
    observations: Sequence[Mapping[str, object]],
) -> GraphReasoningResult:
    """Aggregate multi-metric candidate rule paths without executing graph rules."""
    evidence = _evidence_by_chunk_id(Path(evidence_index))
    try:
        with sqlite3.connect(f"file:{Path(knowledge_db).resolve()}?mode=ro", uri=True) as connection:
            _candidate_metadata(connection)
            nodes = _candidate_nodes(connection)
            edge_rows = tuple(connection.execute(
                "SELECT subject_id, predicate, object_id, status, properties_json, evidence_json FROM edges"
            ))
    except sqlite3.Error as error:
        raise GraphRetrievalError("knowledge graph is unreadable") from error

    edges_by_rule: dict[str, list[tuple[str, str, dict[str, object], tuple[dict[str, object], ...]]]] = {}
    for subject, predicate, object_id, status, properties_json, evidence_json in edge_rows:
        properties = _json(properties_json, "edge properties")
        if status != "candidate" or subject not in nodes or object_id not in nodes or not isinstance(properties, dict):
            raise GraphRetrievalError("candidate graph has a dangling or invalid edge")
        if nodes[subject][0] == "InterpretationRule":
            edges_by_rule.setdefault(subject, []).append((predicate, object_id, properties, _evidence_items(_json(evidence_json, "edge evidence"))))

    observation_nodes: dict[str, set[str]] = {}
    for observation in observations:
        metric_id = str(observation.get("metric_id", "")).strip()
        terms = observation.get("terms", [])
        if not metric_id or not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
            continue
        for term in terms:
            if not isinstance(term, str):
                continue
            for node_id, _mode in _match_nodes(nodes, term, node_types={"TestItem"}):
                observation_nodes.setdefault(node_id, set()).add(metric_id)

    paths: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    for rule_id, relations in sorted(edges_by_rule.items()):
        rule_name, rule_properties = nodes[rule_id][1], nodes[rule_id][2]
        subject_edges = [item for item in relations if item[0] == "RULE_HAS_SUBJECT"]
        conclusions = [item for item in relations if item[0] == "RULE_HAS_CONCLUSION"]
        if not subject_edges or not conclusions:
            continue
        mismatch = False
        for _predicate, target_id, properties, _edge_evidence in subject_edges:
            input_terms = properties.get("input_terms", [])
            if not isinstance(input_terms, list):
                continue
            for term in input_terms:
                if term is None or _norm(str(term)) in {"", "none", "null"}:
                    continue
                matches = _match_nodes(nodes, str(term), node_types={"TestItem"})
                if matches and target_id not in {node_id for node_id, _mode in matches}:
                    mismatch = True
        applicability = rule_properties.get("applicability", {})
        required_inputs = applicability.get("required_inputs", []) if isinstance(applicability, dict) else []
        required_test_items: set[str] = set()
        if isinstance(required_inputs, list):
            for term in required_inputs:
                matches = _match_nodes(nodes, str(term), node_types={"TestItem"})
                if matches:
                    matched_required = {node_id for node_id, _mode in matches}
                    required_test_items.update(matched_required)
                    if not any(target_id in matched_required for _p, target_id, _props, _ev in subject_edges):
                        mismatch = True
        if mismatch:
            rejections.append({"rule_id": rule_id, "rule_name": rule_name, "reason": "subject_endpoint_mismatch"})
            continue
        subject_ids = {target_id for _predicate, target_id, _properties, _edge_evidence in subject_edges if nodes[target_id][0] == "TestItem"}
        matched_subjects = subject_ids & set(observation_nodes)
        matched_metric_ids = sorted({metric for node_id in matched_subjects for metric in observation_nodes[node_id]})
        if len(matched_metric_ids) < 2:
            continue
        required = required_test_items or subject_ids
        missing_inputs = sorted(nodes[node_id][1] for node_id in required - matched_subjects)
        preconditions = applicability.get("preconditions", []) if isinstance(applicability, dict) else []
        if missing_inputs:
            status = "candidate-partial"
        elif preconditions:
            status = "candidate-precondition-unverified"
        else:
            status = "candidate-complete"
        triples = [
            {
                "subject_id": rule_id, "subject_name": rule_name, "subject_type": "InterpretationRule",
                "predicate": predicate, "object_id": target_id,
                "object_name": nodes[target_id][1], "object_type": nodes[target_id][0],
            }
            for predicate, target_id, _properties, _edge_evidence in subject_edges + conclusions
        ]
        evidence_items = list(nodes[rule_id][3])
        for _predicate, _target_id, _properties, edge_evidence in subject_edges + conclusions:
            evidence_items.extend(edge_evidence)
        chunk_ids = list(_project_candidate_evidence(tuple(evidence_items), evidence))
        if not chunk_ids:
            rejections.append({"rule_id": rule_id, "rule_name": rule_name, "reason": "evidence_unresolved"})
            continue
        path_key = [rule_id, matched_metric_ids, [item["object_id"] for item in triples]]
        paths.append({
            "path_id": "candidate-path:" + hashlib.sha256(json.dumps(path_key, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16],
            "rule_id": rule_id,
            "rule_name": rule_name,
            "status": status,
            "graph_status": "candidate-only",
            "matched_metric_ids": matched_metric_ids,
            "missing_inputs": missing_inputs,
            "preconditions": preconditions if isinstance(preconditions, list) else [],
            "triples": triples,
            "chunk_ids": chunk_ids,
        })
    return GraphReasoningResult(tuple(paths), tuple(rejections))


def graph_retrieve(knowledge_db: Path, evidence_index: Path, query: str, *, top_k: int = 5, max_hops: int = 3) -> tuple[GraphHit, ...]:
    """Find graph paths and project only exact hash-bound graph chunks to evidence."""
    if not isinstance(query, str) or not _norm(query): return ()
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1: raise GraphRetrievalError("top_k must be positive")
    evidence = _evidence_by_chunk_id(Path(evidence_index)); needle = _norm(query)
    try:
        with sqlite3.connect(f"file:{Path(knowledge_db).resolve()}?mode=ro", uri=True) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok": raise GraphRetrievalError("knowledge SQLite integrity check failed")
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if metadata.get("schema_version") in {
                "chapter-knowledge-graph/v0.1", "chapter-knowledge-graph/v0.2",
            }:
                return _candidate_graph_retrieve(connection, evidence, query, top_k, max_hops)
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
                evidence_value = evidence.get(node)
                evidence_digest = evidence_value[0] if evidence_value is not None else None
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
