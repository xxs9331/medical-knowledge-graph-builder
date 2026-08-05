"""Build a source-bound ontology projection from chapter entity candidates.

The DeepSeek entity file is intentionally kept as a flat extraction result.
This module produces a separate candidate projection for synonym, hierarchy,
dependency, and rule-context mappings.  Every supplementary term remains
traceable to chapter text or to an existing rule artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .entity_extraction import _match_key


SYNONYM_CANONICAL = {
    "大细胞贫血": "大细胞性贫血",
    "珠蛋白合成障碍性贫血": "珠蛋白生成障碍性贫血",
    "巨幼红细胞贫血": "巨幼细胞贫血",
}

LABTEST_PARENTS = {
    "变异系数": "红细胞容积分布宽度",
    "标准差": "红细胞容积分布宽度",
    "白细胞分类计数": "白细胞计数",
}

POPULATION_PARENTS = {
    "男性": "性别",
    "女性": "性别",
    "成年男性": "男性",
    "成年女性": "女性",
    "孕妇": "女性",
    "哺乳期妇女": "女性",
    "准备怀孕的妇女": "女性",
    "妇女月经期": "女性",
    "妊娠期": "妊娠",
    "妊娠中、后期": "妊娠",
    "妊娠后期": "妊娠",
    "妊娠3个月以后": "妊娠",
    "60岁以上的老年人": "老年人",
    "婴幼儿": "儿童",
    "新生儿": "婴儿",
    "婴儿": "儿童",
}

DISEASE_PARENT_OVERRIDES = {
    "慢性再生障碍性贫血": "再生障碍性贫血",
    "免疫性溶血性贫血": "溶血性贫血",
    "血小板减少症": "血小板减少",
    "恶性淋巴瘤": "淋巴瘤",
    "恶性肿瘤": "肿瘤",
    "慢性粒细胞白血病": "粒细胞白血病",
    "嗜碱性粒细胞白血病": "粒细胞白血病",
    "嗜酸性粒细胞白血病": "粒细胞白血病",
    "肝硬化": "肝病",
    "慢性肝病": "肝病",
    "急性肝炎": "肝病",
    "严重肝病": "肝病",
    "肝癌": "肿瘤",
    "胰腺癌": "肿瘤",
    "继发性红细胞增多症": "红细胞增多症",
    "真性红细胞增多症": "红细胞增多症",
    "椭圆形红细胞增多症": "红细胞增多症",
    "遗传性椭圆形红细胞增多症": "椭圆形红细胞增多症",
    "球形红细胞增多症": "红细胞增多症",
    "遗传性球形红细胞增多症": "球形红细胞增多症",
    "缺血性脑卒中": "脑卒中",
    "急性心肌梗死": "心肌梗死",
    "急性脑梗死": "脑梗死",
    "周围血管深静脉血栓": "深静脉血栓",
    "高胆固醇血症": "高脂血症",
    "高血脂": "高脂血症",
}

CALCULATED_DEPENDENCIES = {
    "红细胞刚性指数": {
        "kind": "formula",
        "depends_on": ["全血黏度(高切)", "血浆黏度", "血细胞比容"],
    },
    "红细胞变形指数": {
        "kind": "formula",
        "depends_on": ["全血黏度(高切)", "血浆黏度", "血细胞比容"],
    },
    "红细胞滤过指数": {
        "kind": "measurement-derived",
        "depends_on": ["红细胞变形性"],
    },
    "凝血酶原时间比值": {
        "kind": "formula",
        "depends_on": ["受检血浆 PT", "正常人血浆 PT"],
    },
    "国际标准化比值": {
        "kind": "formula",
        "depends_on": ["凝血酶原时间比值"],
    },
}

CONTEXT_SLOT_SPECS = {
    "性别": {
        "mapped_population_entities": ["男性", "女性"],
        "reason": "规则条件使用性别槽位，实体层由性别父类统领男性和女性取值。",
    },
    "年龄": {
        "mapped_population_entities": [
            "儿童",
            "成人",
            "老年人",
            "60岁以上的老年人",
            "婴儿",
            "新生儿",
            "婴幼儿",
        ],
        "reason": "规则条件使用年龄槽位，实体层保留原文中的年龄人群，不新增年龄节点。",
    },
}

SUPPLEMENTAL_ENTITIES = (
    {
        "category": "Population",
        "name": "性别",
        "aliases": [],
        "parent": None,
        "synonyms": [],
        "depends_on": [],
    },
    {
        "category": "LabTest",
        "name": "中性粒细胞绝对值",
        "aliases": [],
        "parent": None,
        "synonyms": [],
        "depends_on": [],
    },
)

_CATEGORY_ORDER = {
    "LabTest": 0,
    "Disease": 1,
    "Population": 2,
    "Etiology": 3,
    "MethodOrDrug": 4,
}

_DISEASE_SUFFIX_PARENT = (
    ("白血病", "白血病"),
    ("淋巴瘤", "淋巴瘤"),
    ("贫血", "贫血"),
)
_RULE_STRING_LIMIT = 24


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _match_key(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _walk_strings(value: Any, path: tuple[str, ...] = ()) -> list[dict[str, str]]:
    """Flatten rule JSON strings while retaining a stable structural path."""
    if isinstance(value, str):
        return [{"path": ".".join(path), "text": value}]
    if isinstance(value, Mapping):
        result: list[dict[str, str]] = []
        for key, child in value.items():
            result.extend(_walk_strings(child, (*path, str(key))))
        return result
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value):
            result.extend(_walk_strings(child, (*path, str(index))))
        return result
    return []


def source_evidence(source_pages: Sequence[Mapping[str, Any]], term: str, *, limit: int = 3) -> list[dict[str, Any]]:
    """Return line-level evidence for a source term using normalized matching."""
    term_key = _match_key(term)
    if not term_key:
        return []
    evidence: list[dict[str, Any]] = []
    for page in source_pages:
        text = str(page.get("text", ""))
        for line_number, line in enumerate(text.splitlines(), 1):
            if term_key not in _match_key(line):
                continue
            evidence.append({
                "chapter_page_index": page.get("chapter_page_index"),
                "printed_page_number": page.get("printed_page_number"),
                "source_path": page.get("cleaned_path"),
                "source_sha256": page.get("cleaned_sha256"),
                "line_number": line_number,
                "quote": line.strip(),
            })
            break
        if len(evidence) >= limit:
            break
    return evidence


def _canonical_name(name: str, category: str, source_names: set[str]) -> str:
    candidate = SYNONYM_CANONICAL.get(name, name) if category == "Disease" else name
    return candidate if candidate == name or candidate in source_names else name


def _parent_for(category: str, name: str, names: set[str]) -> str | None:
    if category == "LabTest":
        candidate = LABTEST_PARENTS.get(name)
    elif category == "Population":
        candidate = POPULATION_PARENTS.get(name)
    elif category == "Disease":
        candidate = DISEASE_PARENT_OVERRIDES.get(name)
        if candidate is None:
            candidate = next(
                (parent for suffix, parent in _DISEASE_SUFFIX_PARENT if name != parent and name.endswith(suffix)),
                None,
            )
    else:
        candidate = None
    return candidate if candidate in names and candidate != name else None


def _entity_lookup(entities: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Resolve source dependency terms to canonical names where possible."""
    lookup: dict[str, str] = {}
    for entity in entities:
        name = str(entity["name"])
        lookup.setdefault(_match_key(name), name)
        for alias in entity.get("aliases", []):
            lookup.setdefault(_match_key(str(alias)), name)
    lookup[_match_key("受检血浆 PT")] = "血浆凝血酶原时间"
    lookup[_match_key("正常人血浆 PT")] = "血浆凝血酶原时间"
    return lookup


def _rule_mentions(rules: Any, term: str) -> list[dict[str, str]]:
    key = _match_key(term)
    matches = [item for item in _walk_strings(rules) if key in _match_key(item["text"])]
    return matches[:_RULE_STRING_LIMIT]


def _build_rule_alignment(
    rules: Any,
    entities: Sequence[Mapping[str, Any]],
    source_pages: Sequence[Mapping[str, Any]],
    supplemental: Mapping[str, Any],
    raw_entity_names: set[str],
) -> dict[str, Any]:
    entity_names = {str(entity["name"]) for entity in entities}
    population_names = {
        str(entity["name"])
        for entity in entities
        if entity.get("category") == "Population"
    }
    slots: list[dict[str, Any]] = []
    for slot, spec in CONTEXT_SLOT_SPECS.items():
        mapped = [name for name in spec["mapped_population_entities"] if name in population_names]
        is_entity = slot in entity_names
        raw_present = slot in raw_entity_names
        slots.append({
            "rule_slot": slot,
            "slot_type": "context",
            "is_entity": is_entity,
            "raw_entity_present": raw_present,
            "manually_supplemented": is_entity and not raw_present,
            "mapped_population_entities": mapped,
            "rule_mentions": _rule_mentions(rules, slot),
            "source_evidence": [evidence for name in mapped for evidence in source_evidence(source_pages, name, limit=1)],
            "reason": spec["reason"],
        })

    neut_name = "中性粒细胞绝对值"
    neut_present = neut_name in raw_entity_names
    slots.append({
        "rule_slot": neut_name,
        "slot_type": "LabTest",
        "is_entity": True,
        "raw_entity_present": neut_present,
        "supplemented_entity": supplemental if not neut_present else None,
        "requested_aliases_not_grounded": ["NEUT#"],
        "rule_mentions": _rule_mentions(rules, neut_name),
        "source_evidence": source_evidence(source_pages, neut_name),
        "reason": "原文明确出现指标名称，但未出现 NEUT#，因此只补充无别名候选并进入人工复核。",
    })
    return {
        "schema_version": "chapter-rule-entity-alignment/v0.1",
        "status": "candidate-only",
        "hold": True,
        "approved": 0,
        "manual_supplemented_entities": [
            slot["rule_slot"] for slot in slots if slot.get("manually_supplemented")
        ],
        "slots": slots,
    }


def _insert_by_category(entities: list[dict[str, Any]], candidate: dict[str, Any]) -> None:
    """Keep manually supplemented nodes in the same category ordering as raw output."""
    target_order = _CATEGORY_ORDER[candidate["category"]]
    for index, entity in enumerate(entities):
        if _CATEGORY_ORDER[entity["category"]] > target_order:
            entities.insert(index, candidate)
            return
    entities.append(candidate)


def build_ontology_candidate(
    raw_entities: Sequence[Mapping[str, Any]],
    source_pages: Sequence[Mapping[str, Any]],
    rules: Any,
) -> dict[str, Any]:
    """Create the repaired entity projection and its auditable relations."""
    raw_names_by_category = {
        category: {str(item["name"]) for item in raw_entities if item.get("category") == category}
        for category in ("LabTest", "Disease", "Population", "Etiology", "MethodOrDrug")
    }
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for item in raw_entities:
        category = str(item["category"])
        name = str(item["name"])
        canonical = _canonical_name(name, category, raw_names_by_category.get(category, set()))
        key = (category, canonical)
        if key not in grouped:
            grouped[key] = {
                "category": category,
                "name": canonical,
                "aliases": [],
                "synonyms": [],
            }
            order.append(key)
        group = grouped[key]
        group["aliases"] = _dedupe([*group["aliases"], *[str(value) for value in item.get("aliases", [])]])
        if name != canonical:
            group["synonyms"] = _dedupe([*group["synonyms"], name])

    entities: list[dict[str, Any]] = []
    for key in order:
        group = grouped[key]
        entities.append({
            "category": group["category"],
            "name": group["name"],
            "aliases": group["aliases"],
            "parent": None,
            "synonyms": group["synonyms"],
            "depends_on": list(CALCULATED_DEPENDENCIES.get(group["name"], {}).get("depends_on", [])),
        })

    supplemental_by_name = {item["name"]: dict(item) for item in SUPPLEMENTAL_ENTITIES}
    supplemented_names: list[str] = []
    for supplemental in supplemental_by_name.values():
        if not any(
            entity["category"] == supplemental["category"] and entity["name"] == supplemental["name"]
            for entity in entities
        ):
            _insert_by_category(entities, supplemental)
            supplemented_names.append(supplemental["name"])
    neut_supplemental = supplemental_by_name["中性粒细胞绝对值"]

    names_by_category = {
        category: {str(entity["name"]) for entity in entities if entity["category"] == category}
        for category in ("LabTest", "Disease", "Population", "Etiology", "MethodOrDrug")
    }
    for entity in entities:
        entity["parent"] = _parent_for(entity["category"], entity["name"], names_by_category[entity["category"]])

    lookup = _entity_lookup(entities)
    relations: list[dict[str, Any]] = []
    synonym_relations = 0
    parent_relations = 0
    dependency_relations = 0
    unresolved_dependencies: list[dict[str, Any]] = []

    for entity in entities:
        for synonym in entity["synonyms"]:
            relations.append({
                "relation": "SYNONYM_OF",
                "source": synonym,
                "target": entity["name"],
                "category": entity["category"],
                "origin": "deterministic exact-name merge",
                "evidence": {
                    "source": source_evidence(source_pages, synonym),
                    "target": source_evidence(source_pages, entity["name"]),
                },
            })
            synonym_relations += 1
        if entity["parent"]:
            parent_evidence = source_evidence(source_pages, entity["parent"])
            relations.append({
                "relation": "IS_A",
                "source": entity["name"],
                "target": entity["parent"],
                "category": entity["category"],
                "origin": "curated source-term hierarchy candidate" if parent_evidence else "manual ontology supplement",
                "target_status": "source-grounded" if parent_evidence else "manually_supplemented",
                "evidence": {
                    "source": source_evidence(source_pages, entity["name"]),
                    "target": parent_evidence,
                },
            })
            parent_relations += 1
        dependency_spec = CALCULATED_DEPENDENCIES.get(entity["name"])
        if not dependency_spec:
            continue
        for dependency in dependency_spec["depends_on"]:
            resolved = lookup.get(_match_key(dependency))
            relation = {
                "relation": "DEPENDS_ON",
                "source": entity["name"],
                "target": dependency,
                "target_entity": resolved,
                "target_status": "resolved" if resolved else "unresolved_source_term",
                "calculation_kind": dependency_spec["kind"],
                "origin": "source formula or definition",
                "evidence": {
                    "source": source_evidence(source_pages, entity["name"]),
                    "target": source_evidence(source_pages, dependency),
                },
            }
            relations.append(relation)
            dependency_relations += 1
            if not resolved:
                unresolved_dependencies.append({
                    "type": "dependency_target_missing_from_entity_output",
                    "source": entity["name"],
                    "target": dependency,
                })

    rule_alignment = _build_rule_alignment(
        rules,
        entities,
        source_pages,
        neut_supplemental,
        {str(item["name"]) for item in raw_entities},
    )
    review_items = [*unresolved_dependencies]
    if neut_supplemental["name"] in supplemented_names:
        review_items.append({
            "type": "supplemented_labtest_missing_abbreviation",
            "name": supplemental["name"],
            "requested_alias": "NEUT#",
            "reason": "NEUT# is not present in chapter text; do not approve until a source-grounded abbreviation is found.",
            "evidence": source_evidence(source_pages, supplemental["name"]),
        })
    review_items.extend({
        "type": "context_slot_is_not_entity",
        "name": slot["rule_slot"],
        "mapped_population_entities": slot.get("mapped_population_entities", []),
    } for slot in rule_alignment["slots"] if not slot["is_entity"])

    return {
        "entities": entities,
        "relations": relations,
        "rule_alignment": rule_alignment,
        "review_items": review_items,
        "audit": {
            "raw_entities": len(raw_entities),
            "repaired_entities": len(entities),
            "supplemented_entities": len(supplemented_names),
            "synonym_relations": synonym_relations,
            "parent_relations": parent_relations,
            "dependency_relations": dependency_relations,
            "unresolved_dependency_targets": len(unresolved_dependencies),
        },
    }
