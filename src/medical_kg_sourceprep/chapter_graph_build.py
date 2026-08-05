"""Build a candidate-only Chapter 01 graph from reviewed extraction artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any

from .entity_extraction import _match_key
from .semantic_graph import SEMANTIC_RELATIONS


SCHEMA_VERSION = "chapter-knowledge-graph/v0.2"
ENTITY_NODE_TYPES = frozenset({"TestItem", "MedicalConcept", "Population", "TestMethod"})
BOOK_RULE_ORIGIN = "book-rule-library-v0.1"
_CATEGORY_TYPE = {
    "LabTest": "TestItem",
    "Disease": "MedicalConcept",
    "Population": "Population",
    "Etiology": "MedicalConcept",
    "MethodOrDrug": "MedicalConcept",
}
_OUTPUT_OVERRIDES = {
    "白细胞分类": "白细胞分类计数",
    "血清转铁蛋白": "转铁蛋白",
}
_SEX_VALUES = {"男": "男性", "男性": "男性", "女": "女性", "女性": "女性"}
_CONDITION = re.compile(r"^(.+?)\s+(LT|LE|GT|GE|EQ|IN|BETWEEN)\s+(.+)$")
_INPUT_CANONICAL_NAMES = {
    "HGB": "血红蛋白",
    "HCT": "血细胞比容",
    "MCV": "平均红细胞容积",
    "MCH": "平均红细胞血红蛋白含量",
    "MCHC": "平均红细胞血红蛋白浓度",
    "RDW": "红细胞容积分布宽度",
    "RDW-CV": "变异系数",
    "PLT": "血小板计数",
    "MPV": "平均血小板体积",
    "NEUT": "中性粒细胞绝对值",
    "TIBC": "总铁结合力",
    "PTR": "凝血酶原时间比值",
    "INR": "国际标准化比值",
    "受检血浆PT": "血浆凝血酶原时间",
    "正常对照PT": "血浆凝血酶原时间",
    "正常人血浆PT": "血浆凝血酶原时间",
}
_LEGACY_RULE_SUPERSESSIONS = {
    "indicator-rule:28fb53487c62c3932d5d7642": "chapter01:threshold:thrombocytopenia",
    "indicator-rule:57e7e4618a6d9d64511e0bc6": "chapter01:threshold:thrombocytosis",
    "indicator-rule:932edeb443539b3b472d06e3": "chapter01:threshold:neutropenia",
    "indicator-rule:b70c6f519d82198aa1eeeeae": "chapter01:temporal:mpv-plt-sustained-decline",
}


class ChapterGraphBuildError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(kind: str, *values: object) -> str:
    digest = hashlib.sha256(_canonical([SCHEMA_VERSION, kind, *values]).encode()).hexdigest()[:24]
    return f"chapter01:{kind}:{digest}"


def _dedupe_strings(values: Sequence[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        key = _match_key(value)
        if value and key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _evidence_list(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, Mapping):
        return [dict(value)]
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _book_rule_evidence(rule: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = rule.get("evidence", {})
    anchors = raw.get("anchors") if isinstance(raw, Mapping) else None
    values = anchors if isinstance(anchors, list) else _evidence_list(raw)
    result: list[dict[str, Any]] = []
    for value in values:
        evidence = dict(value)
        evidence["exact_quote"] = evidence.get("source_quote") or evidence.get("exact_quote")
        result.append(evidence)
    return result


def validate_rule_evidence(packages: Sequence[Mapping[str, Any]], chunk_manifest: Mapping[str, Any],
                           chunk_root: Path) -> int:
    """Replay every book-rule evidence anchor against its hash-bound source chunk."""
    chunks = {item["chunk_id"]: item for item in chunk_manifest.get("chunks", [])}
    replayed = 0
    seen_rule_ids: set[str] = set()
    for package in packages:
        for rule in package.get("rules", []):
            rule_id = str(rule.get("rule_id", ""))
            if not rule_id or rule_id in seen_rule_ids:
                raise ChapterGraphBuildError(f"duplicate or missing rule_id: {rule_id}")
            seen_rule_ids.add(rule_id)
            evidence_items = _book_rule_evidence(rule)
            if not evidence_items:
                raise ChapterGraphBuildError(f"rule evidence is missing: {rule_id}")
            for evidence in evidence_items:
                chunk_id = evidence.get("chunk_id")
                metadata = chunks.get(chunk_id)
                if metadata is None:
                    raise ChapterGraphBuildError(f"rule evidence chunk is missing: {rule_id}/{chunk_id}")
                path = Path(chunk_root) / metadata["chunk_path"]
                raw = path.read_bytes()
                actual_hash = hashlib.sha256(raw).hexdigest()
                declared_hash = evidence.get("chunk_sha256")
                if actual_hash != metadata.get("chunk_sha256") or actual_hash != declared_hash:
                    raise ChapterGraphBuildError(f"rule evidence chunk hash drift: {rule_id}/{chunk_id}")
                quote = evidence.get("exact_quote")
                if not isinstance(quote, str) or not quote or quote not in raw.decode("utf-8"):
                    raise ChapterGraphBuildError(f"rule evidence quote cannot be replayed: {rule_id}/{chunk_id}")
                replayed += 1
    return replayed


def _reference_item_name(output: str) -> str:
    name = re.sub(r"参考(?:区间|结果)$", "", output.strip())
    return _OUTPUT_OVERRIDES.get(name, name)


class ChapterGraphBuilder:
    """Merge entities, emit triples, and enforce a fail-closed candidate graph."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.typed_lookup: dict[tuple[str, str], str] = {}
        self.general_lookup: dict[str, list[str]] = {}
        self.catalog_lookup: dict[tuple[str, str], str] = {}
        self.review: list[dict[str, Any]] = []
        self.superseded_rules: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()

    def build(
        self,
        entities: Sequence[Mapping[str, Any]],
        ontology: Mapping[str, Any],
        catalog: Mapping[str, Any],
        semantic_relations: Mapping[str, Any],
        semantic_rules: Mapping[str, Any],
        reference_rules: Mapping[str, Any],
        core_rules: Mapping[str, Any],
        temporal_rules: Mapping[str, Any],
        quality_report: Mapping[str, Any],
        manual_review: Mapping[str, Any],
        entity_evidence: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        self._validate_inputs(ontology, catalog, semantic_relations, semantic_rules,
                              reference_rules, core_rules, temporal_rules, quality_report, manual_review)
        self._validate_rule_ids(reference_rules, core_rules, temporal_rules)
        method_terms = self._method_terms(catalog, reference_rules)
        self._base_entities(entities, method_terms, entity_evidence or {})
        self._catalog_endpoints(catalog, semantic_relations, semantic_rules)
        self._ontology_relations(ontology)
        self._reference_ranges(reference_rules)
        self._semantic_method_relations(semantic_relations)
        self._semantic_rule_nodes(semantic_rules)
        self._book_rules(core_rules, rule_kind="core")
        self._book_rules(temporal_rules, rule_kind="temporal")
        self._quality_review_items(quality_report)
        self._record_superseded_rules(manual_review)
        self._finalize_review_items()
        self._validate_graph()
        nodes = sorted(self.nodes.values(), key=lambda item: item["node_id"])
        edges = sorted(self.edges.values(), key=lambda item: item["triple_id"])
        merged = [node for node in nodes if node["node_type"] in ENTITY_NODE_TYPES]
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "candidate-only",
            "hold": True,
            "approved": 0,
            "nodes": nodes,
            "edges": edges,
            "merged_entities": merged,
            "review_items": self.review,
            "superseded_rules": sorted(self.superseded_rules, key=lambda item: item["legacy_rule_id"]),
            "counts": {
                "nodes": len(nodes),
                "edges": len(edges),
                "merged_entities": len(merged),
                "node_types": dict(sorted(Counter(node["node_type"] for node in nodes).items())),
                "predicates": dict(sorted(Counter(edge["predicate"] for edge in edges).items())),
                "layers": dict(sorted(Counter(edge["layer"] for edge in edges).items())),
                "origins": dict(sorted(self.counts.items())),
                "review_required": len(self.review),
                "superseded_rules": len(self.superseded_rules),
                "approved": 0,
            },
        }

    @staticmethod
    def _validate_inputs(*packages: Mapping[str, Any]) -> None:
        for package in packages:
            if package.get("approved") not in (0, None):
                raise ChapterGraphBuildError("input package contains approved candidates")

    @staticmethod
    def _validate_rule_ids(*packages: Mapping[str, Any]) -> None:
        rule_ids = [str(rule.get("rule_id", "")) for package in packages
                    for rule in package.get("rules", [])]
        if any(not rule_id for rule_id in rule_ids) or len(rule_ids) != len(set(rule_ids)):
            raise ChapterGraphBuildError("book rule IDs must be present and unique")

    @staticmethod
    def _method_terms(catalog: Mapping[str, Any], reference_rules: Mapping[str, Any]) -> set[str]:
        terms = {
            _match_key(entry["text"])
            for entry in catalog.get("entries", [])
            if entry.get("entity_type") == "TestMethod"
        }
        for rule in reference_rules.get("rules", []):
            method = rule.get("applicability", {}).get("method")
            values = method if isinstance(method, list) else [method]
            terms.update(_match_key(value) for value in values if value)
            for case in rule.get("cases", []):
                selected = case.get("selector", {}).get("检测方法")
                if selected:
                    terms.add(_match_key(selected))
        return terms

    def _add_node(self, node_type: str, name: str, *, properties: Mapping[str, Any] | None = None,
                  evidence: Sequence[Mapping[str, Any]] = (), origin: str,
                  identity: object | None = None) -> str:
        name = name.strip()
        if not name:
            raise ChapterGraphBuildError("node name is empty")
        node_id = _stable_id("node", node_type, _match_key(name) if identity is None else identity)
        props = dict(properties or {})
        if node_id in self.nodes:
            node = self.nodes[node_id]
            aliases = _dedupe_strings([*node["properties"].get("aliases", []), *props.get("aliases", [])])
            categories = _dedupe_strings([*node["properties"].get("categories", []), *props.get("categories", [])])
            node["properties"].update({key: value for key, value in props.items()
                                       if key not in {"aliases", "categories"}})
            if aliases:
                node["properties"]["aliases"] = aliases
            if categories:
                node["properties"]["categories"] = categories
            node["evidence"] = self._merge_evidence(node["evidence"], evidence)
            node["origins"] = _dedupe_strings([*node["origins"], origin])
            return node_id
        self.nodes[node_id] = {
            "node_id": node_id,
            "node_type": node_type,
            "name": name,
            "status": "candidate",
            "properties": props,
            "evidence": [dict(item) for item in evidence],
            "origins": [origin],
        }
        self.typed_lookup.setdefault((_match_key(name), node_type), node_id)
        self.general_lookup.setdefault(_match_key(name), []).append(node_id)
        self.counts[f"node:{origin}"] += 1
        return node_id

    @staticmethod
    def _merge_evidence(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for item in [*first, *second]:
            value = dict(item)
            result[_canonical(value)] = value
        return list(result.values())

    def _register_terms(self, node_id: str, terms: Sequence[object]) -> None:
        node_type = self.nodes[node_id]["node_type"]
        for term in terms:
            key = _match_key(str(term))
            if not key:
                continue
            self.typed_lookup.setdefault((key, node_type), node_id)
            values = self.general_lookup.setdefault(key, [])
            if node_id not in values:
                values.append(node_id)

    def _resolve(self, term: str, node_type: str | None = None) -> str | None:
        key = _match_key(term)
        if node_type:
            return self.typed_lookup.get((key, node_type))
        values = self.general_lookup.get(key, [])
        return values[0] if len(values) == 1 else None

    def _base_entities(self, entities: Sequence[Mapping[str, Any]], method_terms: set[str],
                       entity_evidence: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        seen: set[tuple[str, str]] = set()
        for entity in entities:
            category, name = str(entity["category"]), str(entity["name"])
            node_type = _CATEGORY_TYPE.get(category)
            if node_type is None:
                raise ChapterGraphBuildError(f"unknown entity category: {category}")
            if category == "MethodOrDrug" and _match_key(name) in method_terms:
                node_type = "TestMethod"
            key = (_match_key(name), node_type)
            if key in seen:
                raise ChapterGraphBuildError(f"duplicate canonical entity: {name}/{node_type}")
            seen.add(key)
            aliases = _dedupe_strings([*entity.get("aliases", []), *entity.get("synonyms", [])])
            evidence: list[Mapping[str, Any]] = []
            for term in [name, *aliases]:
                evidence.extend(entity_evidence.get(_match_key(term), []))
            node_id = self._add_node(node_type, name, properties={
                "aliases": aliases,
                "categories": [category],
                "parent": entity.get("parent"),
                "depends_on": list(entity.get("depends_on", [])),
            }, evidence=evidence, origin="entity-list-v0.4")
            self._register_terms(node_id, [name, *aliases])
            if not evidence:
                self.review.append({
                    "reason_code": "entity_without_verbatim_source",
                    "entity": {"category": category, "name": name},
                    "detail": "The merged entity has no replayable page-level mention; keep candidate-only.",
                })

    @staticmethod
    def _needed_catalog_keys(semantic_relations: Mapping[str, Any],
                             semantic_rules: Mapping[str, Any]) -> set[tuple[str, str]]:
        needed: set[tuple[str, str]] = set()
        for relation in semantic_relations.get("candidates", []):
            if relation.get("relation") != "ITEM_MEASURED_BY_METHOD":
                continue
            needed.add((relation["page_id"], relation["source_candidate_key"]))
            needed.add((relation["page_id"], relation["target_candidate_key"]))
        for rule in semantic_rules.get("candidates", []):
            page = rule["page_id"]
            for field in ("subject_candidate_keys", "population_candidate_keys", "method_candidate_keys"):
                needed.update((page, key) for key in rule.get(field, []))
            needed.add((page, rule["conclusion_candidate_key"]))
        return needed

    def _catalog_endpoints(self, catalog: Mapping[str, Any], semantic_relations: Mapping[str, Any],
                           semantic_rules: Mapping[str, Any]) -> None:
        needed = self._needed_catalog_keys(semantic_relations, semantic_rules)
        entries = {(entry["page_id"], entry["candidate_key"]): entry
                   for entry in catalog.get("entries", [])}
        missing = sorted(needed - set(entries))
        if missing:
            raise ChapterGraphBuildError(f"semantic catalog endpoints are missing: {missing[:3]}")
        for scoped in sorted(needed):
            entry = entries[scoped]
            node_type, text = entry["entity_type"], entry["text"]
            if node_type == "ReferenceRange":
                continue
            node_id = self._resolve(text, node_type)
            if node_id is None:
                node_id = self._add_node(node_type, text, properties={
                    "aliases": [], "categories": ["SemanticEndpoint"],
                    "candidate_keys": [entry["candidate_key"]],
                }, evidence=[entry["source"]], origin="semantic-catalog-v0.4")
                self._register_terms(node_id, [text, entry["candidate_key"]])
            else:
                node = self.nodes[node_id]
                keys = _dedupe_strings([*node["properties"].get("candidate_keys", []), entry["candidate_key"]])
                node["properties"]["candidate_keys"] = keys
                node["evidence"] = self._merge_evidence(node["evidence"], [entry["source"]])
            self.catalog_lookup[scoped] = node_id

    def _add_edge(self, source_id: str, predicate: str, target_id: str, *, layer: str,
                  origin: str, properties: Mapping[str, Any] | None = None,
                  evidence: Sequence[Mapping[str, Any]] = ()) -> str:
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ChapterGraphBuildError(f"dangling edge: {predicate}")
        key = (source_id, predicate, target_id)
        if key in self.edges:
            edge = self.edges[key]
            edge["evidence"] = self._merge_evidence(edge["evidence"], evidence)
            edge["origins"] = _dedupe_strings([*edge["origins"], origin])
            additions = dict(properties or {})
            for property_name, value in list(additions.items()):
                if isinstance(value, list):
                    merged_values: dict[str, Any] = {}
                    for item in [*edge["properties"].get(property_name, []), *value]:
                        merged_values[_canonical(item)] = item
                    additions[property_name] = list(merged_values.values())
            edge["properties"].update(additions)
            return edge["triple_id"]
        triple_id = _stable_id("triple", *key)
        self.edges[key] = {
            "triple_id": triple_id,
            "subject_id": source_id,
            "predicate": predicate,
            "object_id": target_id,
            "layer": layer,
            "status": "candidate",
            "properties": dict(properties or {}),
            "evidence": [dict(item) for item in evidence],
            "origins": [origin],
        }
        self.counts[f"edge:{origin}"] += 1
        return triple_id

    def _ensure_named(self, name: str, node_type: str, *, evidence: Sequence[Mapping[str, Any]],
                      origin: str, category: str = "Supplemental") -> str:
        node_id = self._resolve(name, node_type)
        if node_id is not None:
            return node_id
        node_id = self._add_node(node_type, name, properties={"aliases": [], "categories": [category]},
                                 evidence=evidence, origin=origin)
        self._register_terms(node_id, [name])
        return node_id

    def _ontology_relations(self, ontology: Mapping[str, Any]) -> None:
        for relation in ontology.get("relations", []):
            predicate = relation["relation"]
            evidence_map = relation.get("evidence", {})
            evidence = []
            for role, values in evidence_map.items():
                evidence.extend({**item, "evidence_role": role} for item in _evidence_list(values))
            if predicate == "SYNONYM_OF":
                target_id = self._resolve(relation["target"])
                if target_id is None:
                    self.review.append({"reason_code": "unresolved_synonym_target", "relation": relation})
                    continue
                alias_id = self._add_node("EntityAlias", relation["source"],
                                          properties={"category": relation.get("category")},
                                          evidence=evidence, origin="entity-ontology-v0.4")
                self._add_edge(alias_id, "ALIAS_OF", target_id, layer="ontology",
                               origin="entity-ontology-v0.4", evidence=evidence)
                self._register_terms(target_id, [relation["source"]])
                continue
            category_type = _CATEGORY_TYPE.get(str(relation.get("category")))
            source_id = self._resolve(relation["source"], category_type) if category_type else self._resolve(relation["source"])
            target_name = relation.get("target_entity") or relation["target"]
            target_id = self._resolve(target_name, category_type) if category_type else self._resolve(target_name)
            if target_id is None and predicate == "DEPENDS_ON":
                expected = "TestItem" if target_name != "红细胞变形性" else "MedicalConcept"
                target_id = self._ensure_named(target_name, expected, evidence=evidence,
                                               origin="entity-ontology-v0.4")
            if source_id is None or target_id is None:
                self.review.append({"reason_code": "unresolved_ontology_endpoint", "relation": relation})
                continue
            props = {key: relation[key] for key in ("calculation_kind", "target_status")
                     if relation.get(key) is not None}
            if predicate == "DEPENDS_ON":
                props["dependency_terms"] = [relation["target"]]
            if predicate == "DEPENDS_ON" and relation.get("target") != target_name:
                props["input_roles"] = [relation["target"]]
            self._add_edge(source_id, predicate, target_id, layer="ontology",
                           origin="entity-ontology-v0.4", properties=props, evidence=evidence)

    @staticmethod
    def _source_evidence(source: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [{
            "chunk_id": source.get("chunk_id"),
            "chunk_sha256": source.get("chunk_sha256"),
            "exact_quote": source.get("source_quote") or source.get("exact_quote"),
        }]

    def _source_locator(self, evidence: Mapping[str, Any], *, origin: str) -> str:
        name = str(evidence.get("chunk_id") or evidence.get("page_id") or "source")
        quote = str(evidence.get("exact_quote") or evidence.get("source_quote") or "")
        locator_name = f"{name}#{hashlib.sha256(quote.encode()).hexdigest()[:12]}"
        return self._add_node("SourceLocator", locator_name, properties=dict(evidence),
                              evidence=[evidence], origin=origin)

    def _reference_ranges(self, package: Mapping[str, Any]) -> None:
        for rule in package.get("rules", []):
            item_name = _reference_item_name(rule["function"]["output"])
            item_id = self._resolve(item_name, "TestItem")
            if item_id is None:
                item_id = self._ensure_named(item_name, "TestItem", evidence=self._source_evidence(rule["evidence"]),
                                             origin="reference-rules-v0.1", category="ReferenceRuleEndpoint")
            applicability = dict(rule.get("applicability", {}))
            methods = applicability.get("method")
            method_values = methods if isinstance(methods, list) else [methods]
            for method in [value for value in method_values if value]:
                method_id = self._ensure_named(str(method), "TestMethod", evidence=self._source_evidence(rule["evidence"]),
                                               origin="reference-rules-v0.1", category="DetectionMethod")
                self._add_edge(item_id, "ITEM_MEASURED_BY_METHOD", method_id, layer="reference",
                               origin="reference-rules-v0.1", evidence=self._source_evidence(rule["evidence"]))
            locator_id = self._source_locator(self._source_evidence(rule["evidence"])[0],
                                              origin="reference-rules-v0.1")
            self._add_edge(item_id, "ITEM_SUPPORTED_BY", locator_id, layer="provenance",
                           origin="reference-rules-v0.1", evidence=self._source_evidence(rule["evidence"]))
            for index, case in enumerate(rule.get("cases", [])):
                selector = dict(case.get("selector", {}))
                payload = dict(case.get("interval") or case.get("reference_result") or {})
                case_item_id = item_id
                cell_type, value_type = selector.get("细胞类型"), selector.get("数值类型")
                if isinstance(cell_type, str) and value_type in {"百分数", "绝对值"}:
                    derived_name = f"{cell_type}{value_type}"
                    case_item_id = self._add_node(
                        "TestItem",
                        derived_name,
                        properties={
                            "aliases": [],
                            "categories": ["TableDerivedTestItem"],
                            "derivation_kind": "table_row_column",
                            "selector": selector,
                            "parent": item_name,
                        },
                        evidence=self._source_evidence(rule["evidence"]),
                        origin="reference-rules-v0.1",
                    )
                    self._register_terms(case_item_id, [derived_name])
                    self._add_edge(case_item_id, "IS_A", item_id, layer="reference",
                                   origin="reference-rules-v0.1",
                                   properties={"derivation_kind": "table_row_column"},
                                   evidence=self._source_evidence(rule["evidence"]))
                    self._add_edge(case_item_id, "ITEM_SUPPORTED_BY", locator_id,
                                   layer="provenance", origin="reference-rules-v0.1",
                                   evidence=self._source_evidence(rule["evidence"]))
                range_name = f"{item_name}参考范围/{rule['rule_id']}/{index + 1}"
                range_id = self._add_node("ReferenceRange", range_name, properties={
                    "rule_id": rule["rule_id"], "selector": selector,
                    "applicability": applicability, "value": payload,
                }, evidence=self._source_evidence(rule["evidence"]), origin="reference-rules-v0.1")
                self._add_edge(item_id, "ITEM_HAS_REFERENCE_RANGE", range_id, layer="reference",
                               origin="reference-rules-v0.1", evidence=self._source_evidence(rule["evidence"]))
                if case_item_id != item_id:
                    self._add_edge(case_item_id, "ITEM_HAS_REFERENCE_RANGE", range_id,
                                   layer="reference", origin="reference-rules-v0.1",
                                   evidence=self._source_evidence(rule["evidence"]))
                self._add_edge(range_id, "RANGE_SUPPORTED_BY", locator_id, layer="provenance",
                               origin="reference-rules-v0.1", evidence=self._source_evidence(rule["evidence"]))
                populations = []
                if applicability.get("population"):
                    populations.append(str(applicability["population"]))
                sex = selector.get("性别")
                if sex in _SEX_VALUES:
                    populations.append(_SEX_VALUES[sex])
                for population in _dedupe_strings(populations):
                    population_id = self._ensure_named(population, "Population",
                                                       evidence=self._source_evidence(rule["evidence"]),
                                                       origin="reference-rules-v0.1", category="Population")
                    self._add_edge(range_id, "RANGE_APPLIES_TO_POPULATION", population_id,
                                   layer="reference", origin="reference-rules-v0.1",
                                   evidence=self._source_evidence(rule["evidence"]))
        self.review.append({
            "reason_code": "semantic_reference_candidates_superseded",
            "count": 12,
            "detail": "v0.4 free-text reference range relations were replaced by structured reference rules.",
        })

    def _semantic_method_relations(self, package: Mapping[str, Any]) -> None:
        for relation in package.get("candidates", []):
            if relation.get("relation") != "ITEM_MEASURED_BY_METHOD":
                continue
            scoped_source = (relation["page_id"], relation["source_candidate_key"])
            scoped_target = (relation["page_id"], relation["target_candidate_key"])
            source_id, target_id = self.catalog_lookup.get(scoped_source), self.catalog_lookup.get(scoped_target)
            if source_id is None or target_id is None:
                raise ChapterGraphBuildError("semantic method relation has unresolved endpoint")
            self._add_edge(source_id, relation["relation"], target_id, layer="semantic",
                           origin="semantic-relations-v0.4", properties={"relation_cue": relation.get("relation_cue")},
                           evidence=relation.get("evidence", []))

    def _semantic_rule_nodes(self, package: Mapping[str, Any]) -> None:
        for rule in package.get("candidates", []):
            source = dict(rule["source"])
            rule_id = self._add_node("InterpretationRule", rule["rule_key"], properties={
                "candidate_id": rule["candidate_id"], "semantic_type": rule["semantic_type"],
                "subject_logic": rule["subject_logic"], "components": rule["components"],
                "execution_scope": "retrieval_only",
            }, evidence=[source], origin="semantic-rules-v0.4")
            locator_id = self._source_locator(source, origin="semantic-rules-v0.4")
            edge_evidence = [source]
            for key in rule.get("subject_candidate_keys", []):
                endpoint = self.catalog_lookup.get((rule["page_id"], key))
                if endpoint is None:
                    raise ChapterGraphBuildError(f"unresolved semantic rule subject: {key}")
                self._add_edge(rule_id, "RULE_HAS_SUBJECT", endpoint, layer="semantic",
                               origin="semantic-rules-v0.4", evidence=edge_evidence)
            conclusion = self.catalog_lookup.get((rule["page_id"], rule["conclusion_candidate_key"]))
            if conclusion is None:
                raise ChapterGraphBuildError("unresolved semantic rule conclusion")
            self._add_edge(rule_id, "RULE_HAS_CONCLUSION", conclusion, layer="semantic",
                           origin="semantic-rules-v0.4", evidence=edge_evidence)
            for key in rule.get("population_candidate_keys", []):
                endpoint = self.catalog_lookup[(rule["page_id"], key)]
                self._add_edge(rule_id, "RULE_APPLIES_TO_POPULATION", endpoint, layer="semantic",
                               origin="semantic-rules-v0.4", evidence=edge_evidence)
            for key in rule.get("method_candidate_keys", []):
                endpoint = self.catalog_lookup[(rule["page_id"], key)]
                self._add_edge(rule_id, "RULE_REQUIRES_METHOD", endpoint, layer="semantic",
                               origin="semantic-rules-v0.4", evidence=edge_evidence)
            self._add_edge(rule_id, "RULE_SUPPORTED_BY", locator_id, layer="provenance",
                           origin="semantic-rules-v0.4", evidence=edge_evidence)

    @staticmethod
    def _input_terms(input_spec: Mapping[str, Any], required_input: object | None) -> list[str]:
        values = [input_spec.get("name"), input_spec.get("abbreviation"), input_spec.get("symbol"), required_input]
        expanded: list[object] = list(values)
        for value in values:
            if not value:
                continue
            text = str(value)
            for suffix in ("状态", "时序"):
                if text.endswith(suffix):
                    expanded.append(text.removesuffix(suffix))
            canonical = _INPUT_CANONICAL_NAMES.get(text)
            if canonical:
                expanded.append(canonical)
        return _dedupe_strings(expanded)

    def _rule_endpoint(self, input_spec: Mapping[str, Any], required_input: object | None,
                       evidence: Sequence[Mapping[str, Any]]) -> tuple[str, str, list[str]]:
        terms = self._input_terms(input_spec, required_input)
        name = str(input_spec.get("name") or required_input or "").strip()
        if not name:
            raise ChapterGraphBuildError("book rule input has no name")
        if name == "性别" or required_input == "性别":
            endpoint_type = "Population"
        elif input_spec.get("value_type") == "category":
            endpoint_type = "MedicalConcept"
        else:
            endpoint_type = "TestItem"
        endpoint = next((self._resolve(term, endpoint_type) for term in terms
                         if self._resolve(term, endpoint_type) is not None), None)
        if endpoint is None:
            preferred = next((_INPUT_CANONICAL_NAMES[term] for term in terms
                              if term in _INPUT_CANONICAL_NAMES), name)
            endpoint = self._add_node(endpoint_type, preferred, properties={
                "aliases": _dedupe_strings([term for term in terms if _match_key(term) != _match_key(preferred)]),
                "categories": ["RuleInputEndpoint"],
            }, evidence=evidence, origin=BOOK_RULE_ORIGIN)
        self._register_terms(endpoint, terms)
        return endpoint, endpoint_type, terms

    def _book_rules(self, package: Mapping[str, Any], *, rule_kind: str) -> None:
        execution_policy = dict(package.get("execution_policy", {}))
        for rule in package.get("rules", []):
            evidence = _book_rule_evidence(rule)
            function = dict(rule.get("function", {}))
            output = function.get("output", {})
            if not isinstance(output, Mapping) or not output.get("name"):
                raise ChapterGraphBuildError(f"book rule output is invalid: {rule.get('rule_id')}")
            rule_id_value = str(rule["rule_id"])
            rule_node = self._add_node("InterpretationRule", str(function.get("expression") or rule_id_value),
                                       properties={
                                           "rule_id": rule_id_value,
                                           "rule_type": rule.get("rule_type"),
                                           "rule_kind": rule_kind,
                                           "execution_readiness": rule.get("execution_readiness"),
                                           "execution_scope": (
                                               "longitudinal_only" if rule_kind == "temporal"
                                               else "method_only" if rule.get("rule_type") == "method_internal"
                                               else "candidate_rule_only"
                                           ),
                                           "standard_source": "book",
                                           "function": function,
                                           "applicability": dict(rule.get("applicability", {})),
                                           "cases": list(rule.get("cases", [])),
                                           "formula_ast": rule.get("formula_ast"),
                                           "source_ambiguity": rule.get("source_ambiguity"),
                                           "execution_policy": execution_policy,
                                       }, evidence=evidence, origin=BOOK_RULE_ORIGIN,
                                       identity=rule_id_value)

            applicability = dict(rule.get("applicability", {}))
            inputs = list(function.get("inputs", []))
            required_inputs = list(applicability.get("required_inputs", []))
            for index, input_spec in enumerate(inputs):
                required = required_inputs[index] if index < len(required_inputs) else None
                endpoint, endpoint_type, terms = self._rule_endpoint(input_spec, required, evidence)
                role = str(input_spec.get("name") or required)
                properties = {
                    "input_roles": [role],
                    "input_terms": terms,
                    "input_value_types": [str(input_spec.get("value_type") or "unspecified")],
                }
                if endpoint_type == "Population":
                    self._add_edge(rule_node, "RULE_APPLIES_TO_POPULATION", endpoint,
                                   layer="rule", origin=BOOK_RULE_ORIGIN,
                                   properties=properties, evidence=evidence)
                else:
                    self._add_edge(rule_node, "RULE_HAS_SUBJECT", endpoint,
                                   layer="rule", origin=BOOK_RULE_ORIGIN,
                                   properties=properties, evidence=evidence)

            for precondition in applicability.get("preconditions", []):
                if not isinstance(precondition, Mapping):
                    continue
                context = str(precondition.get("context") or precondition.get("input") or "").strip()
                if not context:
                    continue
                if context == "检测原理":
                    method_name = "电阻抗法" if precondition.get("value") == "电阻抗法" else str(precondition.get("value"))
                    endpoint = self._ensure_named(method_name, "TestMethod", evidence=evidence,
                                                  origin=BOOK_RULE_ORIGIN, category="RuleMethodEndpoint")
                    self._add_edge(rule_node, "RULE_REQUIRES_METHOD", endpoint, layer="rule",
                                   origin=BOOK_RULE_ORIGIN,
                                   properties={"precondition_roles": [context], "precondition": dict(precondition)},
                                   evidence=evidence)
                elif context == "仪器分类模式":
                    method_name = "三分类血细胞分析仪" if precondition.get("value") == "三分类" else str(precondition.get("value"))
                    endpoint = self._ensure_named(method_name, "TestMethod", evidence=evidence,
                                                  origin=BOOK_RULE_ORIGIN, category="RuleMethodEndpoint")
                    self._add_edge(rule_node, "RULE_REQUIRES_METHOD", endpoint, layer="rule",
                                   origin=BOOK_RULE_ORIGIN,
                                   properties={"precondition_roles": [context], "precondition": dict(precondition)},
                                   evidence=evidence)
                else:
                    endpoint = self._ensure_named(context, "MedicalConcept", evidence=evidence,
                                                  origin=BOOK_RULE_ORIGIN, category="RuleContextEndpoint")
                    self._add_edge(rule_node, "RULE_HAS_SUBJECT", endpoint, layer="rule",
                                   origin=BOOK_RULE_ORIGIN,
                                   properties={"input_roles": [f"precondition:{context}"],
                                               "precondition": dict(precondition)}, evidence=evidence)

            methods = applicability.get("method")
            method_values = methods if isinstance(methods, list) else [methods]
            for method in [str(value) for value in method_values if value]:
                endpoint = self._ensure_named(method, "TestMethod", evidence=evidence,
                                              origin=BOOK_RULE_ORIGIN, category="RuleMethodEndpoint")
                self._add_edge(rule_node, "RULE_REQUIRES_METHOD", endpoint, layer="rule",
                               origin=BOOK_RULE_ORIGIN, properties={"method_roles": ["applicability"]},
                               evidence=evidence)

            populations = applicability.get("population")
            population_values = populations if isinstance(populations, list) else [populations]
            for population in [str(value) for value in population_values if value]:
                endpoint = self._ensure_named(population, "Population", evidence=evidence,
                                              origin=BOOK_RULE_ORIGIN, category="Population")
                self._add_edge(rule_node, "RULE_APPLIES_TO_POPULATION", endpoint, layer="rule",
                               origin=BOOK_RULE_ORIGIN, properties={"population_roles": ["applicability"]},
                               evidence=evidence)

            output_name = str(output["name"])
            output_type = "TestItem" if rule.get("rule_type") == "formula" else "MedicalConcept"
            output_aliases = _dedupe_strings([output.get("abbreviation"), output.get("symbol")])
            conclusion = self._resolve(output_name, output_type)
            if conclusion is None:
                conclusion = self._add_node(output_type, output_name, properties={
                    "aliases": output_aliases,
                    "categories": ["RuleOutputEndpoint"],
                    "value_type": output.get("value_type"),
                    "semantic_level": output.get("semantic_level"),
                }, evidence=evidence, origin=BOOK_RULE_ORIGIN)
            self._register_terms(conclusion, [output_name, *output_aliases])
            self._add_edge(rule_node, "RULE_HAS_CONCLUSION", conclusion, layer="rule",
                           origin=BOOK_RULE_ORIGIN,
                           properties={"output_name": output_name, "output_value_type": output.get("value_type")},
                           evidence=evidence)
            for anchor in evidence:
                locator = self._source_locator(anchor, origin=BOOK_RULE_ORIGIN)
                self._add_edge(rule_node, "RULE_SUPPORTED_BY", locator, layer="provenance",
                               origin=BOOK_RULE_ORIGIN, evidence=[anchor])

    def _quality_review_items(self, package: Mapping[str, Any]) -> None:
        for item in package.get("review_items", []):
            self.review.append({
                "reason_code": "book_rule_review_required",
                "rule_id": item.get("rule_id"),
                "reason": item.get("reason"),
                "runtime_behavior": item.get("runtime_behavior"),
            })

    def _record_superseded_rules(self, package: Mapping[str, Any]) -> None:
        records = [*package.get("executable_rules", []), *package.get("temporal_rules", [])]
        for record in records:
            legacy_id = record.get("source_candidate_id")
            new_id = _LEGACY_RULE_SUPERSESSIONS.get(str(legacy_id))
            if new_id is None:
                continue
            self.superseded_rules.append({
                "legacy_rule_id": legacy_id,
                "new_rule_id": new_id,
                "reason": "Replaced by the final book-first rule with structured AST and provenance.",
                "legacy_origin": "manual-rule-review-v0.2",
                "new_origin": BOOK_RULE_ORIGIN,
            })

    def _finalize_review_items(self) -> None:
        for item in self.review:
            item.setdefault("review_id", _stable_id("review", item))

    def _manual_rules(self, package: Mapping[str, Any]) -> None:
        for reviewed in package.get("executable_rules", []):
            if reviewed.get("rule_type") != "threshold_decision":
                continue
            rule = reviewed["rule"]
            overall = rule.get("evidence_overall", {})
            evidence = [{"page_id": reviewed.get("page_id"), "exact_quote": overall.get("source_quote")}]
            rule_id = self._add_node("InterpretationRule", rule["rule_expression"], properties={
                "candidate_id": reviewed["source_candidate_id"],
                "semantic_type": "DIAGNOSTIC_CRITERION", "subject_logic": "SINGLE",
                "execution_scope": reviewed.get("execution_scope"), "cases": rule.get("cases", []),
                "review_action": reviewed.get("review_action"),
            }, evidence=evidence, origin="manual-rule-review-v0.2")
            locator = self._source_locator(evidence[0], origin="manual-rule-review-v0.2")
            for case in rule.get("cases", []):
                match = _CONDITION.match(case["condition"])
                if match is None:
                    raise ChapterGraphBuildError(f"unparseable threshold condition: {case['condition']}")
                subject_name, operator, threshold = match.groups()
                subject = self._resolve(subject_name, "TestItem")
                if subject is None:
                    raise ChapterGraphBuildError(f"threshold subject is missing: {subject_name}")
                conclusion = self._resolve(case["result"], "MedicalConcept")
                if conclusion is None:
                    conclusion = self._ensure_named(case["result"], "MedicalConcept", evidence=evidence,
                                                    origin="manual-rule-review-v0.2", category="RuleConclusion")
                self._add_edge(rule_id, "RULE_HAS_SUBJECT", subject, layer="semantic",
                               origin="manual-rule-review-v0.2",
                               properties={"operator": operator, "threshold": threshold}, evidence=evidence)
                self._add_edge(rule_id, "RULE_HAS_CONCLUSION", conclusion, layer="semantic",
                               origin="manual-rule-review-v0.2", evidence=evidence)
            self._add_edge(rule_id, "RULE_SUPPORTED_BY", locator, layer="provenance",
                           origin="manual-rule-review-v0.2", evidence=evidence)
        for temporal in package.get("temporal_rules", []):
            evidence = [{"page_id": temporal.get("page_id"), "exact_quote": temporal.get("source_quote")}]
            rule_id = self._add_node("InterpretationRule", temporal["rule_expression"], properties={
                "candidate_id": temporal["source_candidate_id"], "semantic_type": "PROGNOSTIC_INDICATOR",
                "subject_logic": "ALL", "execution_scope": "longitudinal_only",
                "condition": temporal["condition"], "required_data": temporal.get("required_data", []),
            }, evidence=evidence, origin="manual-rule-review-v0.2")
            for subject_name in ("平均血小板体积", "血小板计数"):
                subject = self._resolve(subject_name, "TestItem")
                if subject is None:
                    raise ChapterGraphBuildError(f"temporal subject is missing: {subject_name}")
                self._add_edge(rule_id, "RULE_HAS_SUBJECT", subject, layer="semantic",
                               origin="manual-rule-review-v0.2", evidence=evidence)
            conclusion_name = temporal["result"].removeprefix("提示")
            conclusion = self._resolve(conclusion_name, "MedicalConcept")
            if conclusion is None:
                conclusion = self._ensure_named(conclusion_name, "MedicalConcept", evidence=evidence,
                                                origin="manual-rule-review-v0.2", category="RuleConclusion")
            self._add_edge(rule_id, "RULE_HAS_CONCLUSION", conclusion, layer="semantic",
                           origin="manual-rule-review-v0.2", evidence=evidence)
            locator = self._source_locator(evidence[0], origin="manual-rule-review-v0.2")
            self._add_edge(rule_id, "RULE_SUPPORTED_BY", locator, layer="provenance",
                           origin="manual-rule-review-v0.2", evidence=evidence)
        self.review.extend([
            {"reason_code": "manual_reference_rules_superseded", "count": 6},
            {"reason_code": "manual_formula_rules_represented_by_dependencies", "count": 2},
            {"reason_code": "manual_semantic_facts_superseded", "count": len(package.get("semantic_facts", []))},
        ])

    def _validate_graph(self) -> None:
        for edge in self.edges.values():
            if edge["subject_id"] not in self.nodes or edge["object_id"] not in self.nodes:
                raise ChapterGraphBuildError("graph contains a dangling endpoint")
            relation = SEMANTIC_RELATIONS.get(edge["predicate"])
            if relation is None:
                continue
            expected_source, expected_target = relation
            actual_source = self.nodes[edge["subject_id"]]["node_type"]
            actual_target = self.nodes[edge["object_id"]]["node_type"]
            allowed_source = expected_source if isinstance(expected_source, tuple) else (expected_source,)
            allowed_target = expected_target if isinstance(expected_target, tuple) else (expected_target,)
            if actual_source not in allowed_source or actual_target not in allowed_target:
                raise ChapterGraphBuildError(
                    f"invalid endpoint types for {edge['predicate']}: {actual_source}->{actual_target}"
                )
        if any(node["status"] != "candidate" for node in self.nodes.values()):
            raise ChapterGraphBuildError("non-candidate node leaked into candidate graph")
        if any(edge["status"] != "candidate" for edge in self.edges.values()):
            raise ChapterGraphBuildError("non-candidate edge leaked into candidate graph")
        book_rules = [node for node in self.nodes.values()
                      if node["node_type"] == "InterpretationRule" and BOOK_RULE_ORIGIN in node["origins"]]
        rule_ids = [node["properties"].get("rule_id") for node in book_rules]
        if any(not value for value in rule_ids) or len(rule_ids) != len(set(rule_ids)):
            raise ChapterGraphBuildError("book rules do not have unique stable rule IDs")
        for rule in book_rules:
            conclusions = [edge for edge in self.edges.values()
                           if edge["subject_id"] == rule["node_id"]
                           and edge["predicate"] == "RULE_HAS_CONCLUSION"]
            support = [edge for edge in self.edges.values()
                       if edge["subject_id"] == rule["node_id"]
                       and edge["predicate"] == "RULE_SUPPORTED_BY"]
            if len(conclusions) != 1 or not support:
                raise ChapterGraphBuildError(
                    f"book rule requires one conclusion and source support: {rule['properties'].get('rule_id')}"
                )


def write_graph_package(output: Path, graph: Mapping[str, Any], input_hashes: Mapping[str, str],
                        validation: Mapping[str, Any] | None = None) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graph_doc = {key: graph[key] for key in ("schema_version", "status", "hold", "approved", "counts", "nodes", "edges")}
    documents = {
        "merged-entities.json": {
            "schema_version": "merged-entities/v0.2", "status": "candidate-only", "hold": True,
            "approved": 0, "count": len(graph["merged_entities"]), "entities": graph["merged_entities"],
        },
        "triples.json": {
            "schema_version": "chapter-triples/v0.2", "status": "candidate-only", "hold": True,
            "approved": 0, "count": len(graph["edges"]), "triples": graph["edges"],
        },
        "graph.json": graph_doc,
        "review-queue.json": {
            "schema_version": "chapter-graph-review-queue/v0.2", "status": "HOLD", "approved": 0,
            "count": len(graph["review_items"]), "items": graph["review_items"],
        },
        "superseded-rules.json": {
            "schema_version": "chapter-rule-supersession/v0.1", "status": "candidate-only",
            "hold": True, "approved": 0, "count": len(graph["superseded_rules"]),
            "items": graph["superseded_rules"],
        },
    }
    for name, document in documents.items():
        _atomic_json(output / name, document)
    database = output / "knowledge.sqlite"
    _write_sqlite(database, graph["nodes"], graph["edges"])
    manifest = {
        "schema_version": "chapter-graph-run/v0.2", "status": "candidate-only", "hold": True,
        "approved": 0, "input_sha256": dict(sorted(input_hashes.items())), "counts": graph["counts"],
        "validation": dict(validation or {}),
        "outputs": {name: _sha256(output / name) for name in [*documents, "knowledge.sqlite"]},
    }
    _atomic_json(output / "run-manifest.json", manifest)


def build_entity_evidence(checkpoint: Mapping[str, Any], source_manifest: Mapping[str, Any],
                          source_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Recover exact page-line evidence for page-level entity candidates."""
    if checkpoint.get("source_manifest_sha256") is None:
        raise ChapterGraphBuildError("entity checkpoint lacks source manifest binding")
    pages = source_manifest.get("pages", [])
    page_results = checkpoint.get("page_results", {})
    evidence: dict[str, list[dict[str, Any]]] = {}
    page_documents: list[tuple[Mapping[str, Any], str, list[str]]] = []
    for index, page in enumerate(pages):
        result = page_results.get(f"page:{index:04d}")
        if not isinstance(result, Mapping) or result.get("status") != "success":
            raise ChapterGraphBuildError(f"entity checkpoint page is incomplete: {index}")
        path = source_root / page["cleaned_path"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != page["cleaned_sha256"] or digest != result.get("source_sha256"):
            raise ChapterGraphBuildError(f"entity source page hash drift: {index}")
        page_documents.append((page, digest, raw.decode("utf-8").splitlines()))
    for index, page in enumerate(pages):
        result = page_results[f"page:{index:04d}"]
        for item in result.get("output", {}).get("entities", []):
            terms = _dedupe_strings([item.get("name", ""), *item.get("mentions", []), *item.get("aliases", [])])
            matched: dict[str, Any] | None = None
            for term in terms:
                key = _match_key(term)
                ordered_pages = [page_documents[index], *page_documents[:index], *page_documents[index + 1:]]
                for actual_page, actual_digest, lines in ordered_pages:
                    for line_number, line in enumerate(lines, 1):
                        if key and key in _match_key(line):
                            matched = {
                                "page_id": actual_page["page_id"],
                                "chapter_page_index": actual_page["chapter_page_index"],
                                "printed_page_number": actual_page.get("printed_page_number"),
                                "source_pdf_page_number": actual_page.get("source_pdf_page_number"),
                                "source_path": actual_page["cleaned_path"],
                                "source_sha256": actual_digest,
                                "line_number": line_number,
                                "exact_quote": line,
                                "matched_term": term,
                                "extracted_page_index": index,
                            }
                            break
                    if matched:
                        break
                if matched:
                    break
            if matched is None:
                raise ChapterGraphBuildError(f"entity mention cannot be replayed: {item.get('name')}")
            for term in terms:
                key = _match_key(term)
                values = evidence.setdefault(key, [])
                if _canonical(matched) not in {_canonical(value) for value in values}:
                    values.append(matched)
    return evidence


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        staging = Path(handle.name)
        handle.write(payload)
    staging.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_sqlite(path: Path, nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".sqlite", delete=False) as handle:
        staging = Path(handle.name)
    try:
        with sqlite3.connect(staging) as db:
            db.execute("PRAGMA foreign_keys = ON")
            db.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE nodes (
                    node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status = 'candidate'), properties_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL, origins_json TEXT NOT NULL
                );
                CREATE TABLE edges (
                    triple_id TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL REFERENCES nodes(node_id), predicate TEXT NOT NULL,
                    object_id TEXT NOT NULL REFERENCES nodes(node_id), layer TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status = 'candidate'), properties_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL, origins_json TEXT NOT NULL,
                    UNIQUE(subject_id, predicate, object_id)
                );
                CREATE INDEX nodes_name_type ON nodes(name, node_type);
                CREATE INDEX edges_subject ON edges(subject_id, predicate);
                CREATE INDEX edges_object ON edges(object_id, predicate);
            """)
            db.executemany("INSERT INTO metadata VALUES (?, ?)", (
                ("schema_version", SCHEMA_VERSION), ("status", "candidate-only"),
                ("approved", "0"), ("node_count", str(len(nodes))), ("edge_count", str(len(edges))),
            ))
            db.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)", (
                (node["node_id"], node["node_type"], node["name"], node["status"],
                 _canonical(node["properties"]), _canonical(node["evidence"]), _canonical(node["origins"]))
                for node in nodes
            ))
            db.executemany("INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                (edge["triple_id"], edge["subject_id"], edge["predicate"], edge["object_id"],
                 edge["layer"], edge["status"], _canonical(edge["properties"]),
                 _canonical(edge["evidence"]), _canonical(edge["origins"]))
                for edge in edges
            ))
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ChapterGraphBuildError("SQLite integrity check failed")
            if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise ChapterGraphBuildError("SQLite foreign key check failed")
        staging.replace(path)
    except Exception:
        staging.unlink(missing_ok=True)
        raise
