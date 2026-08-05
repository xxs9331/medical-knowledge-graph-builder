"""Read-only graph-to-evidence candidate projection."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class GraphRetrievalError(ValueError):
    pass


_CANDIDATE_CHAPTER_SCHEMAS = {
    "chapter-knowledge-graph/v0.1",
    "chapter-knowledge-graph/v0.2",
}
_FINAL_CHAPTER_SCHEMA = "chapter-final-knowledge-graph/v0.1"


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
    if metadata.get("schema_version") not in _CANDIDATE_CHAPTER_SCHEMAS:
        raise GraphRetrievalError("unsupported candidate graph schema")
    if metadata.get("approved") != "0" or metadata.get("status") != "candidate-only":
        raise GraphRetrievalError("candidate graph review boundary is invalid")
    return metadata


def _chapter_metadata(connection: sqlite3.Connection) -> tuple[dict[str, str], str]:
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    schema = metadata.get("schema_version")
    if schema in _CANDIDATE_CHAPTER_SCHEMAS:
        if metadata.get("approved") != "0" or metadata.get("status") != "candidate-only":
            raise GraphRetrievalError("candidate graph review boundary is invalid")
        return metadata, "candidate-only"
    if schema == _FINAL_CHAPTER_SCHEMA:
        if metadata.get("status") != "final" or "approved" in metadata:
            raise GraphRetrievalError("final graph publication boundary is invalid")
        return metadata, "final"
    raise GraphRetrievalError("unsupported chapter graph schema")


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


def _chapter_nodes(
    connection: sqlite3.Connection,
    graph_status: str,
) -> dict[str, tuple[str, str, dict[str, object], tuple[dict[str, object], ...]]]:
    if graph_status == "candidate-only":
        return _candidate_nodes(connection)
    rows = tuple(connection.execute(
        "SELECT node_id, node_type, name, properties_json, evidence_json FROM nodes"
    ))
    nodes: dict[str, tuple[str, str, dict[str, object], tuple[dict[str, object], ...]]] = {}
    for node_id, node_type, name, properties_json, evidence_json in rows:
        properties = _json(properties_json, "node properties")
        if not isinstance(properties, dict):
            raise GraphRetrievalError("final graph node is invalid")
        nodes[node_id] = (node_type, name, properties, _evidence_items(_json(evidence_json, "node evidence")))
    return nodes


def _chapter_edge_rows(
    connection: sqlite3.Connection,
    graph_status: str,
) -> tuple[tuple[str, str, str, str, str, str], ...]:
    if graph_status == "candidate-only":
        return tuple(connection.execute(
            "SELECT subject_id, predicate, object_id, status, properties_json, evidence_json FROM edges"
        ))
    return tuple(
        (subject, predicate, object_id, "final", properties_json, evidence_json)
        for subject, predicate, object_id, properties_json, evidence_json in connection.execute(
            "SELECT subject_id, predicate, object_id, properties_json, evidence_json FROM edges"
        )
    )


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


def _bounded_graph_paths(
    starts: Sequence[str],
    graph: Mapping[str, Sequence[tuple[str, str, Any]]],
    max_hops: int,
) -> Iterator[tuple[str, tuple[str, ...], tuple[str, ...], tuple[Any, ...]]]:
    """Yield deterministic simple paths for every graph representation."""
    for start in sorted(starts):
        queue = deque([(start, (), (start,), ())])
        visited = {start}
        while queue:
            node, relations, path, path_edges = queue.popleft()
            yield node, relations, path, path_edges
            if len(relations) >= max_hops:
                continue
            for adjacent, relation, payload in sorted(
                graph.get(node, ()), key=lambda item: (item[0], item[1])
            ):
                if adjacent in visited:
                    continue
                visited.add(adjacent)
                queue.append((
                    adjacent,
                    relations + (relation,),
                    path + (adjacent,),
                    path_edges + (payload,),
                ))


def _rank_graph_hits(candidates: Sequence[GraphHit], top_k: int) -> tuple[GraphHit, ...]:
    """Keep the strongest path per evidence chunk and apply the shared top-k order."""
    best: dict[str, GraphHit] = {}
    for hit in candidates:
        current = best.get(hit.chunk_id)
        if current is None or (-hit.graph_score, hit.path_relations) < (
            -current.graph_score, current.path_relations
        ):
            best[hit.chunk_id] = hit
    return tuple(sorted(best.values(), key=lambda hit: (-hit.graph_score, hit.chunk_id))[:top_k])


def _candidate_graph_retrieve(
    connection: sqlite3.Connection,
    evidence: dict[str, tuple[str, str]],
    query: str,
    top_k: int,
    max_hops: int,
) -> tuple[GraphHit, ...]:
    _metadata, graph_status = _chapter_metadata(connection)
    nodes = _chapter_nodes(connection, graph_status)
    edge_rows = _chapter_edge_rows(connection, graph_status)
    graph: dict[str, list[tuple[str, str, dict[str, object]]]] = {}
    for source, predicate, target, status, _properties_json, evidence_json in edge_rows:
        if source not in nodes or target not in nodes or status not in {"candidate", "final"}:
            raise GraphRetrievalError("candidate graph has a dangling or invalid edge")
        edge_evidence = _evidence_items(_json(evidence_json, "edge evidence"))
        forward = {
            "evidence": edge_evidence,
            "triple": {
                "subject_id": source,
                "subject_name": nodes[source][1],
                "subject_type": nodes[source][0],
                "predicate": predicate,
                "object_id": target,
                "object_name": nodes[target][1],
                "object_type": nodes[target][0],
                "traversal_direction": "forward",
            },
        }
        reverse = {
            "evidence": edge_evidence,
            "triple": {
                "subject_id": source,
                "subject_name": nodes[source][1],
                "subject_type": nodes[source][0],
                "predicate": predicate,
                "object_id": target,
                "object_name": nodes[target][1],
                "object_type": nodes[target][0],
                "traversal_direction": "reverse",
            },
        }
        graph.setdefault(source, []).append((target, predicate, forward))
        graph.setdefault(target, []).append((source, predicate, reverse))
    starts = _match_nodes(nodes, query)
    candidates: list[GraphHit] = []
    for start, match_mode in starts:
        start_name = nodes[start][1]
        for node_id, relations, path, path_edges in _bounded_graph_paths(
            (start,), graph, max_hops
        ):
            accumulated_evidence = nodes[path[0]][3]
            for edge in path_edges:
                accumulated_evidence += tuple(edge["evidence"])
            projected = _project_candidate_evidence(accumulated_evidence + nodes[node_id][3], evidence)
            score = round(1.0 / (1 + len(relations)), 6)
            for chunk_id in projected:
                path_triples = tuple(edge["triple"] for edge in path_edges)
                candidates.append(GraphHit(
                    chunk_id, score, (), relations, (start,), (start_name,), graph_status,
                    path, tuple(nodes[value][1] for value in path),
                    tuple(nodes[value][0] for value in path), path_triples, match_mode,
                ))
    return _rank_graph_hits(candidates, top_k)


def graph_query_diagnostic(knowledge_db: Path, query: str) -> dict[str, object]:
    """Explain whether a graph lookup miss is caused by entity coverage or term linking."""
    if not isinstance(query, str) or not _norm(query):
        return {"query": query, "status": "entity_missing", "matches": []}
    try:
        with sqlite3.connect(f"file:{Path(knowledge_db).resolve()}?mode=ro", uri=True) as connection:
            _metadata, graph_status = _chapter_metadata(connection)
            nodes = _chapter_nodes(connection, graph_status)
    except sqlite3.Error as error:
        raise GraphRetrievalError("knowledge graph is unreadable") from error
    matches = _match_nodes(nodes, query)
    if matches:
        modes = {mode for _node_id, mode in matches}
        status = "structural_match" if modes == {"structural_match"} else "matched"
        return {
            "query": query,
            "status": status,
            "graph_status": graph_status,
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
        "graph_status": graph_status,
        "matches": [],
    }


_FLAG_STATE = {"low": "低", "normal": "正常", "high": "高"}


def _unit_key(value: object) -> str:
    unit = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", "", unit).replace("μ", "u").replace("µ", "u")


def _decimal_value(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _condition_fact(
    input_name: str,
    nodes: Mapping[str, tuple[str, str, dict[str, object], tuple[dict[str, object], ...]]],
    observations_by_node: Mapping[str, tuple[Mapping[str, object], ...]],
    input_endpoints: Mapping[str, str],
    context_facts: Mapping[str, Mapping[str, object]],
) -> tuple[str, Mapping[str, object] | None, object | None]:
    context = context_facts.get(input_name)
    if context is not None:
        return "ok", context, context.get("value")
    state_input = input_name.endswith("状态")
    endpoint = input_endpoints.get(input_name)
    if endpoint is not None:
        matches = ((endpoint, "rule_role"),)
    else:
        lookup_name = input_name.removesuffix("状态") if state_input else input_name
        matches = _match_nodes(nodes, lookup_name, node_types={"TestItem"})
    records: dict[str, Mapping[str, object]] = {}
    for node_id, _mode in matches:
        for observation in observations_by_node.get(node_id, ()):
            records[str(observation.get("metric_id"))] = observation
    if not records:
        return "missing", None, None
    if len(records) != 1:
        return "ambiguous", None, None
    observation = next(iter(records.values()))
    if state_input:
        state = _FLAG_STATE.get(str(observation.get("computed_flag", "")))
        return ("ok", observation, state) if state is not None else ("missing", observation, None)
    return "ok", observation, observation.get("value")


def _evaluate_condition(
    condition: object,
    nodes: Mapping[str, tuple[str, str, dict[str, object], tuple[dict[str, object], ...]]],
    observations_by_node: Mapping[str, tuple[Mapping[str, object], ...]],
    input_endpoints: Mapping[str, str],
    context_facts: Mapping[str, Mapping[str, object]],
) -> tuple[str, list[dict[str, object]]]:
    if not isinstance(condition, Mapping):
        return "unsupported", [{"status": "unsupported", "reason": "condition_not_object"}]
    conjunction = condition.get("all")
    if isinstance(conjunction, list):
        traces: list[dict[str, object]] = []
        statuses = []
        for child in conjunction:
            status, child_trace = _evaluate_condition(
                child, nodes, observations_by_node, input_endpoints, context_facts
            )
            statuses.append(status)
            traces.extend(child_trace)
        for failure in ("unsupported", "unit_mismatch", "ambiguous", "missing", "boundary_ambiguous", "fail"):
            if failure in statuses:
                return failure, traces
        return "pass", traces

    input_name, op = condition.get("input"), condition.get("op")
    if not isinstance(input_name, str) or op not in {"EQ", "LT", "LE", "GT", "GE", "BETWEEN"}:
        return "unsupported", [{
            "input": input_name, "op": op, "status": "unsupported",
            "reason": "unsupported_condition_shape",
        }]
    fact_status, observation, actual = _condition_fact(
        input_name, nodes, observations_by_node, input_endpoints, context_facts
    )
    trace: dict[str, object] = {
        "input": input_name,
        "op": op,
        "metric_id": observation.get("metric_id") if observation else None,
        "actual_value": str(actual) if actual is not None else None,
        "actual_unit": None if input_name.endswith("状态") else observation.get("unit") if observation else None,
        "expected_unit": condition.get("unit"),
    }
    if fact_status != "ok":
        trace["status"] = fact_status
        return fact_status, [trace]
    expected_unit = condition.get("unit")
    actual_unit = trace["actual_unit"]
    if expected_unit and _unit_key(expected_unit) != _unit_key(actual_unit):
        trace["status"] = "unit_mismatch"
        return "unit_mismatch", [trace]

    if op == "EQ":
        expected = condition.get("value")
        trace["expected_value"] = str(expected) if expected is not None else None
        status = "pass" if _norm(str(actual)) == _norm(str(expected)) else "fail"
        trace["status"] = status
        return status, [trace]

    numeric = _decimal_value(actual)
    if numeric is None:
        trace["status"] = "missing"
        trace["reason"] = "numeric_value_missing"
        return "missing", [trace]
    if op == "BETWEEN":
        lower, upper = _decimal_value(condition.get("lower")), _decimal_value(condition.get("upper"))
        trace["expected_lower"] = str(lower) if lower is not None else None
        trace["expected_upper"] = str(upper) if upper is not None else None
        if lower is None or upper is None:
            trace["status"] = "unsupported"
            return "unsupported", [trace]
        if condition.get("bounds") == "source_tilde" and numeric in {lower, upper}:
            trace["status"] = "boundary_ambiguous"
            return "boundary_ambiguous", [trace]
        status = "pass" if lower <= numeric <= upper else "fail"
    else:
        expected = _decimal_value(condition.get("value"))
        trace["expected_value"] = str(expected) if expected is not None else None
        if expected is None:
            trace["status"] = "unsupported"
            return "unsupported", [trace]
        comparisons = {
            "LT": numeric < expected,
            "LE": numeric <= expected,
            "GT": numeric > expected,
            "GE": numeric >= expected,
        }
        status = "pass" if comparisons[op] else "fail"
    trace["status"] = status
    return status, [trace]


def _evaluate_rule_cases(
    rule_properties: Mapping[str, object],
    nodes: Mapping[str, tuple[str, str, dict[str, object], tuple[dict[str, object], ...]]],
    observations_by_node: Mapping[str, tuple[Mapping[str, object], ...]],
    input_endpoints: Mapping[str, str],
    context_facts: Mapping[str, Mapping[str, object]],
) -> dict[str, object] | None:
    cases = rule_properties.get("cases", [])
    if not isinstance(cases, list) or not cases:
        return None
    evaluations: list[dict[str, object]] = []
    matches: list[int] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            evaluations.append({"case_index": index, "status": "unsupported", "condition_trace": []})
            continue
        status, trace = _evaluate_condition(
            case.get("condition_ast"), nodes, observations_by_node, input_endpoints,
            context_facts,
        )
        evaluations.append({
            "case_index": index,
            "status": status,
            "candidate_result": case.get("result"),
            "condition_trace": trace,
        })
        if status == "pass":
            matches.append(index)
    if len(matches) == 1:
        selected = evaluations[matches[0]]
        evaluation_status = "case_match"
    elif len(matches) > 1:
        selected = None
        evaluation_status = "ambiguous_case_match"
    else:
        selected = None
        statuses = {str(item["status"]) for item in evaluations}
        evaluation_status = next((status for status in (
            "unit_mismatch", "unsupported", "ambiguous", "missing", "boundary_ambiguous"
        ) if status in statuses), "no_case_match")
    return {
        "status": evaluation_status,
        "matched_case_index": selected["case_index"] if selected else None,
        "candidate_result": selected["candidate_result"] if selected else None,
        "condition_trace": selected["condition_trace"] if selected else [],
        "case_evaluations": evaluations,
    }


def graph_reasoning_paths(
    knowledge_db: Path,
    evidence_index: Path,
    observations: Sequence[Mapping[str, object]],
    context_facts: Mapping[str, Mapping[str, object]] | None = None,
) -> GraphReasoningResult:
    """Evaluate candidate cases for context without promoting them to approved rules."""
    evidence = _evidence_by_chunk_id(Path(evidence_index))
    try:
        with sqlite3.connect(f"file:{Path(knowledge_db).resolve()}?mode=ro", uri=True) as connection:
            _metadata, graph_status = _chapter_metadata(connection)
            nodes = _chapter_nodes(connection, graph_status)
            edge_rows = _chapter_edge_rows(connection, graph_status)
    except sqlite3.Error as error:
        raise GraphRetrievalError("knowledge graph is unreadable") from error

    edges_by_rule: dict[str, list[tuple[str, str, dict[str, object], tuple[dict[str, object], ...]]]] = {}
    for subject, predicate, object_id, status, properties_json, evidence_json in edge_rows:
        properties = _json(properties_json, "edge properties")
        if status not in {"candidate", "final"} or subject not in nodes or object_id not in nodes or not isinstance(properties, dict):
            raise GraphRetrievalError("candidate graph has a dangling or invalid edge")
        if nodes[subject][0] in {"InterpretationRule", "Rule"}:
            edges_by_rule.setdefault(subject, []).append((predicate, object_id, properties, _evidence_items(_json(evidence_json, "edge evidence"))))

    observation_nodes: dict[str, list[Mapping[str, object]]] = {}
    for observation in observations:
        metric_id = str(observation.get("metric_id", "")).strip()
        terms = observation.get("terms", [])
        if not metric_id or not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
            continue
        for term in terms:
            if not isinstance(term, str):
                continue
            for node_id, _mode in _match_nodes(nodes, term, node_types={"TestItem"}):
                values = observation_nodes.setdefault(node_id, [])
                if not any(str(item.get("metric_id")) == metric_id for item in values):
                    values.append(observation)
    frozen_observations = {node_id: tuple(values) for node_id, values in observation_nodes.items()}

    context_facts = context_facts or {}
    paths: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    status_prefix = "final" if graph_status == "final" else "candidate"
    for rule_id, relations in sorted(edges_by_rule.items()):
        rule_name, rule_properties = nodes[rule_id][1], nodes[rule_id][2]
        subject_edges = [item for item in relations if item[0] in {"RULE_HAS_SUBJECT", "CONSUMES"}]
        population_edges = [
            item for item in relations
            if item[0] in {"RULE_APPLIES_TO_POPULATION", "APPLIES_TO"} and item[2].get("input_roles")
        ]
        conclusions = [item for item in relations if item[0] in {"RULE_HAS_CONCLUSION", "PRODUCES"}]
        if not subject_edges or not conclusions:
            continue
        mismatch = False
        for _predicate, target_id, properties, _edge_evidence in subject_edges:
            input_terms = properties.get("input_terms", [])
            if not isinstance(input_terms, list):
                continue
            resolved_input_nodes: set[str] = set()
            for term in input_terms:
                if term is None or _norm(str(term)) in {"", "none", "null"}:
                    continue
                matches = _match_nodes(nodes, str(term), node_types={"TestItem"})
                resolved_input_nodes.update(node_id for node_id, _mode in matches)
            if resolved_input_nodes and target_id not in resolved_input_nodes:
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
        matched_metric_ids = sorted({
            str(observation.get("metric_id"))
            for node_id in matched_subjects
            for observation in observation_nodes[node_id]
        })
        matched_context_roles = sorted({
            str(role)
            for _predicate, _target_id, properties, _edge_evidence in population_edges
            for role in properties.get("input_roles", [])
            if str(role) in context_facts
        })
        if len(matched_metric_ids) < 2 and not (matched_metric_ids and matched_context_roles):
            continue
        required = required_test_items or subject_ids
        missing_inputs = sorted(nodes[node_id][1] for node_id in required - matched_subjects)
        preconditions = applicability.get("preconditions", []) if isinstance(applicability, dict) else []
        input_endpoints = {
            str(term): target_id
            for _predicate, target_id, properties, _edge_evidence in subject_edges + population_edges
            for term in [*properties.get("input_roles", []), *properties.get("input_terms", [])]
            if isinstance(term, str) and not term.startswith("precondition:")
        }
        candidate_evaluation = None if missing_inputs else _evaluate_rule_cases(
            rule_properties, nodes, frozen_observations, input_endpoints, context_facts
        )
        if missing_inputs:
            status = f"{status_prefix}-partial"
        elif candidate_evaluation is not None:
            evaluation_status = candidate_evaluation["status"]
            if evaluation_status == "case_match" and preconditions:
                status = f"{status_prefix}-case-match-precondition-unverified"
            elif evaluation_status == "case_match":
                status = f"{status_prefix}-case-match"
            else:
                status = f"{status_prefix}-{str(evaluation_status).replace('_', '-')}"
        elif preconditions:
            status = f"{status_prefix}-precondition-unverified"
        else:
            status = f"{status_prefix}-complete"
        triples = [
            {
                "subject_id": rule_id, "subject_name": rule_name, "subject_type": nodes[rule_id][0],
                "predicate": predicate, "object_id": target_id,
                "object_name": nodes[target_id][1], "object_type": nodes[target_id][0],
            }
            for predicate, target_id, _properties, _edge_evidence in subject_edges + population_edges + conclusions
        ]
        evidence_items = list(nodes[rule_id][3])
        for _predicate, _target_id, _properties, edge_evidence in subject_edges + population_edges + conclusions:
            evidence_items.extend(edge_evidence)
        chunk_ids = list(_project_candidate_evidence(tuple(evidence_items), evidence))
        if not chunk_ids:
            rejections.append({"rule_id": rule_id, "rule_name": rule_name, "reason": "evidence_unresolved"})
            continue
        path_key = [rule_id, matched_metric_ids, [item["object_id"] for item in triples]]
        paths.append({
            "path_id": "candidate-path:" + hashlib.sha256(json.dumps(path_key, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16],
            "rule_id": str(rule_properties.get("rule_id") or rule_id),
            "rule_node_id": rule_id,
            "rule_name": rule_name,
            "status": status,
            "graph_status": graph_status,
            "approved_execution": graph_status == "final",
            "diagnostic_use": "allowed" if graph_status == "final" else "forbidden",
            "preconditions_verified": not bool(preconditions),
            "matched_metric_ids": matched_metric_ids,
            "matched_context_inputs": matched_context_roles,
            "missing_inputs": missing_inputs,
            "preconditions": preconditions if isinstance(preconditions, list) else [],
            "candidate_evaluation": candidate_evaluation,
            "triples": triples,
            "chunk_ids": chunk_ids,
        })
    paths_by_rule = {str(path["rule_node_id"]): path for path in paths}
    for producer_id, predicate, consumer_id, status, properties_json, _evidence_json in edge_rows:
        if predicate not in {"RULE_SATISFIES_PRECONDITION", "SATISFIES_PRECONDITION"} or status not in {"candidate", "final"}:
            continue
        producer, consumer = paths_by_rule.get(producer_id), paths_by_rule.get(consumer_id)
        properties = _json(properties_json, "dependency properties")
        if producer is None or consumer is None or not isinstance(properties, dict):
            continue
        satisfies = properties.get("satisfies")
        producer_results = properties.get("producer_results", [])
        producer_evaluation = producer.get("candidate_evaluation")
        if not isinstance(satisfies, Mapping) or not isinstance(producer_evaluation, Mapping):
            continue
        candidate_result = producer_evaluation.get("candidate_result")
        passed = (
            producer_evaluation.get("status") == "case_match"
            and candidate_result in producer_results
        )
        producer_status = str(producer_evaluation.get("status"))
        evaluation = {
            "context": satisfies.get("context"),
            "op": satisfies.get("op"),
            "expected_value": satisfies.get("value"),
            "actual_value": (
                satisfies.get("value") if passed
                else "未满足" if producer_status == "no_case_match"
                else "未确认"
            ),
            "status": "pass" if passed else "fail",
            "source_rule_id": producer.get("rule_id"),
            "source_rule_name": producer.get("rule_name"),
            "source_candidate_result": candidate_result,
            "source_path_id": producer.get("path_id"),
        }
        values = consumer.setdefault("precondition_evaluations", [])
        if isinstance(values, list):
            values.append(evaluation)
        expected_contexts = {
            str(item.get("context") or item.get("input"))
            for item in consumer.get("preconditions", [])
            if isinstance(item, Mapping)
        }
        evaluated = {
            str(item.get("context")): item
            for item in consumer.get("precondition_evaluations", [])
            if isinstance(item, Mapping)
        }
        if expected_contexts and expected_contexts <= set(evaluated):
            verified = all(evaluated[context].get("status") == "pass" for context in expected_contexts)
            consumer["preconditions_verified"] = verified
            if verified and isinstance(consumer.get("candidate_evaluation"), Mapping):
                if consumer["candidate_evaluation"].get("status") == "case_match":
                    consumer["status"] = f"{status_prefix}-case-match-precondition-derived"
            elif not verified:
                consumer["status"] = f"{status_prefix}-precondition-failed"
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
            if metadata.get("schema_version") in {*_CANDIDATE_CHAPTER_SCHEMAS, _FINAL_CHAPTER_SCHEMA}:
                return _candidate_graph_retrieve(connection, evidence, query, top_k, max_hops)
            if metadata.get("schema_version") != "knowledge-graph/v0.2": raise GraphRetrievalError("unsupported knowledge graph schema")
            nodes = {node_id: (kind, payload) for node_id, kind, payload in connection.execute("SELECT node_id, node_type, payload FROM nodes")}
            edges = tuple(connection.execute("SELECT source_id, target_id, relation FROM edges"))
            anchors = tuple(connection.execute("SELECT anchor_id, node_id FROM anchors"))
            chunks = {chunk_id: content for chunk_id, content in connection.execute("SELECT chunk_id, content FROM chunk_text")}
    except sqlite3.Error as error:
        raise GraphRetrievalError("knowledge graph is unreadable") from error
    if any(chunk not in nodes or nodes[chunk][0] != "EvidenceChunk" for chunk in chunks): raise GraphRetrievalError("graph chunk references are invalid")
    graph: dict[str, list[tuple[str, str, None]]] = {}
    for source, target, relation in edges:
        if source not in nodes or target not in nodes: raise GraphRetrievalError("graph has dangling edge")
        graph.setdefault(source, []).append((target, relation, None)); graph.setdefault(target, []).append((source, relation, None))
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
    for node, relations, path, _path_edges in _bounded_graph_paths(starts, graph, max_hops):
        if node not in chunks:
            continue
        digest = hashlib.sha256(chunks[node].encode()).hexdigest()
        evidence_value = evidence.get(node)
        evidence_digest = evidence_value[0] if evidence_value is not None else None
        if evidence_digest is not None and evidence_digest != digest:
            raise GraphRetrievalError("graph evidence chunk binding drift")
        if evidence_digest == digest:
            anchor_ids = tuple(sorted({anchor for path_node in path for anchor in anchors_by_node.get(path_node, ())}))
            candidates.append(GraphHit(node, round(1.0 / (1 + len(relations)), 6), anchor_ids, relations))
    return _rank_graph_hits(candidates, top_k)
