"""Build the manually reviewed Chapter 01 rule inventory.

This inventory does not create canonical entities. Rules whose endpoints cannot
be bound to the frozen v0.8 catalog are retained as entity-completion requests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "source-packages/canonical/evidence/chapter-01/chunks"
ENTITY_PATH = ROOT / "evaluation/chapter-01/chapter-01-canonical-entities-v0.8.json"
EXTRACTED_PATH = ROOT / "runtime/candidates/chapter-01/rules-v0.12/rules.json"
OUTPUT_PATH = ROOT / "runtime/candidates/chapter-01/rules-manual-v0.1/manual-rule-audit.json"


def _read_chunk(short_id: str) -> tuple[str, str, str]:
    group, part = short_id.split(":")
    path = EVIDENCE_ROOT / group / f"{part}.md"
    text = path.read_text(encoding="utf-8")
    chunk_id = f"clinical-hematology:chapter-01:{group}:{part}"
    return chunk_id, text, hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(short_id: str, quote: str, role: str = "source_statement") -> dict[str, Any]:
    chunk_id, text, digest = _read_chunk(short_id)
    start = text.index(quote)
    return {
        "chunk_id": chunk_id,
        "chunk_sha256": digest,
        "char_start": start,
        "char_end": start + len(quote),
        "exact_quote": quote,
        "role": role,
    }


def _rule(
    rule_id: str,
    inputs: list[str],
    outputs: list[str],
    evidence: list[dict[str, Any]],
    *,
    excluded_outputs: list[str] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "inputs": inputs,
        "outputs": outputs,
        "excluded_outputs": excluded_outputs or [],
        "logic": "ALL",
        "evidence": evidence,
        "notes": notes or [],
    }


def _table_evidence(short_id: str, header: str, row: str) -> list[dict[str, Any]]:
    return [_evidence(short_id, header, "table_header"), _evidence(short_id, row, "table_row")]


def _remaining_rules() -> list[dict[str, Any]]:
    anemia_header = (
        "<tr><td>贫血形态与分类</td><td>MCV(80~100fL)</td>"
        "<td>MCH(26~32pg)</td><td>MCHC(310~350g/L)</td><td>病因</td></tr>"
    )
    anemia_rows = [
        ("大细胞性贫血", ["MCV增大", "MCH增大", "MCHC正常"],
         "<tr><td>大细胞性贫血</td><td>&gt;100</td><td>&gt;32</td><td>310~350</td><td>维生素\\( B_{12} \\)缺乏引起的巨幼细胞贫血</td></tr>"),
        ("正细胞性贫血", ["MCV正常", "MCH正常", "MCHC正常"],
         "<tr><td>正细胞性贫血</td><td>80~100</td><td>26~32</td><td>310~350</td><td>急性失血性贫血、急性溶血性贫血、再生障碍性贫血等</td></tr>"),
        ("单纯小细胞性贫血", ["MCV减小", "MCH减小", "MCHC正常"],
         "<tr><td>单纯小细胞性贫血</td><td>&lt;80</td><td>&lt;26</td><td>310~350</td><td>慢性感染、炎症、肝病等引起的贫血</td></tr>"),
        ("小细胞低色素性贫血", ["MCV减小", "MCH显著减小(<23pg)", "MCHC减小"],
         "<tr><td>小细胞低色素性贫血</td><td>&lt;80</td><td>&lt;23</td><td>&lt;300</td><td>缺铁性贫血、珠蛋白生成障碍性贫血等</td></tr>"),
    ]
    rules = [
        _rule(
            f"manual-anemia-{index}", inputs, [output],
            _table_evidence("0003:0000", anemia_header, row),
        )
        for index, (output, inputs, row) in enumerate(anemia_rows, 1)
    ]

    rules.append(_rule(
        "manual-mpv-1",
        ["MPV持续下降", "血小板数量持续下降"],
        ["骨髓造血功能衰竭"],
        [_evidence("0008:0001", "MPV 随血小板数量同时持续下降, 提示骨髓造血功能衰竭。")],
    ))

    iron_header = "<tr><td>血清铁</td><td>TIBC</td><td>原因</td></tr>"
    iron_rows = [
        (["血清铁降低", "总铁结合力增高"], ["缺铁性贫血", "铁吸收不良", "慢性失血", "需铁量增加"],
         "<tr><td>↓</td><td>↑</td><td>缺铁性贫血,铁吸收不良。痔、消化性溃疡出血、月经过多引起的慢性失血。妊娠、婴幼儿生长发育需铁量增加</td></tr>"),
        (["血清铁降低", "总铁结合力降低"], ["慢性感染", "肝硬变", "尿毒症", "肾病综合征", "恶性肿瘤"],
         "<tr><td>↓</td><td>↓</td><td>慢性感染、肝硬变、尿毒症、肾病综合征、恶性肿瘤</td></tr>"),
        (["血清铁升高", "总铁结合力降低"], ["铁剂治疗过量", "溶血性贫血", "再生障碍性贫血", "巨幼细胞贫血", "地中海贫血"],
         "<tr><td>↑</td><td>↓</td><td>铁剂治疗过量、溶血性贫血、再生障碍性贫血、巨幼细胞贫血、地中海贫血等</td></tr>"),
    ]
    rules.extend(
        _rule(f"manual-iron-{index}", inputs, outputs, _table_evidence("0012:0001", iron_header, row))
        for index, (inputs, outputs, row) in enumerate(iron_rows, 1)
    )

    rules.extend([
        _rule(
            "manual-viscosity-1", ["单次血黏度升高"], [],
            [_evidence("0020:0001", "决不能凭一次血黏度升高就认为患有心脑血管疾病")],
            excluded_outputs=["心脑血管疾病"],
        ),
        _rule(
            "manual-viscosity-2", ["仅使用血黏度参数"], [],
            [_evidence("0020:0001", "试图通过血黏度参数来做脑卒中预报是不科学的")],
            excluded_outputs=["脑卒中"],
        ),
    ])

    blood_header = (
        "<tr><td>父母血型</td><td>子女可能的血型</td>"
        "<td>子女不可能的血型</td></tr>"
    )
    blood_rows = [
        ("O", "O", ["O型血"], ["A型血", "B型血", "AB型血"], "O+O", "O", "A,B,AB"),
        ("A", "O", ["A型血", "O型血"], ["B型血", "AB型血"], "A+O(或O+A)", "A,O", "B,AB"),
        ("B", "O", ["B型血", "O型血"], ["A型血", "AB型血"], "B+O(或O+B)", "B,O", "A,AB"),
        ("AB", "O", ["A型血", "B型血"], ["O型血", "AB型血"], "AB+O(或O+AB)", "A,B", "O,AB"),
        ("A", "A", ["A型血", "O型血"], ["B型血", "AB型血"], "A+A", "A,O", "B,AB"),
        ("B", "B", ["B型血", "O型血"], ["A型血", "AB型血"], "B+B", "B,O", "A,AB"),
        ("AB", "AB", ["A型血", "B型血", "AB型血"], ["O型血"], "AB+AB", "A,B,AB", "O"),
        ("A", "B", ["A型血", "B型血", "AB型血", "O型血"], [], "A+B(或B+A)", "A,B,AB,O", "无"),
        ("A", "AB", ["A型血", "B型血", "AB型血"], ["O型血"], "A+AB(或AB+A)", "A,B,AB", "O"),
        ("B", "AB", ["A型血", "B型血", "AB型血"], ["O型血"], "B+AB(或AB+B)", "A,B,AB", "O"),
    ]
    blood_rule_index = 0
    for parent_a, parent_b, possible, excluded, raw_input, raw_possible, raw_excluded in blood_rows:
        row = f"<tr><td>{raw_input}</td><td>{raw_possible}</td><td>{raw_excluded}</td></tr>"
        orientations = [(parent_a, parent_b)]
        if parent_a != parent_b:
            orientations.append((parent_b, parent_a))
        for father, mother in orientations:
            blood_rule_index += 1
            rules.append(_rule(
                f"manual-blood-{blood_rule_index}",
                [f"父亲血型{father}", f"母亲血型{mother}"],
                possible,
                _table_evidence("0022:0000", blood_header, row),
                excluded_outputs=excluded,
                notes=[
                    "父亲/母亲角色由表头与无序组合结构派生；非原文连续逐字 mention。"
                ],
            ))

    rules.append(_rule(
        "manual-d-dimer-1",
        ["D-二聚体正常"],
        [],
        [_evidence("0023:0001", "D-二聚体正常，对排除深静脉血栓(DVT)有重要价值。")],
        excluded_outputs=["深静脉血栓"],
    ))
    return rules


def _supplemental_type(name: str) -> str:
    if name.startswith(("MCV", "MCH", "MCHC", "MPV", "TIBC", "血清铁", "血小板数量")):
        return "IndicatorState"
    if name.startswith(("父亲血型", "母亲血型")) or name.endswith("型血"):
        return "IndicatorState"
    return "ClinicalContext"


def _supplemental_entity(name: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256(f"{_supplemental_type(name)}\0{name}".encode()).hexdigest()[:20]
    return {
        "canonical_id": f"supplemental-entity:{digest}",
        "canonical_name": name,
        "entity_type": _supplemental_type(name),
        "derivation": "TABLE_STRUCTURE" if any(item["role"].startswith("table_") for item in evidence) else "SOURCE_SEMANTIC_NORMALIZATION",
        "evidence": evidence,
        "status": "ENTITY_COMPLETION_REQUIRED",
        "publication_status": "HOLD",
    }


def _entity_lookup(entity_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for entity in entity_doc["canonical_entities"]:
        lookup[entity["canonical_name"]] = entity
        for alias in entity.get("aliases", []):
            lookup.setdefault(alias, entity)
    return lookup


def _bind_endpoints(
    rule: dict[str, Any],
    entities: dict[str, dict[str, Any]],
    supplements: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    bound: dict[str, list[dict[str, str]]] = {"inputs": [], "outputs": [], "excluded_outputs": []}
    supplemental_names: list[str] = []
    for role in bound:
        for name in rule[role]:
            entity = entities.get(name)
            if entity is None:
                entity = supplements.setdefault(name, _supplemental_entity(name, rule["evidence"]))
                supplemental_names.append(name)
            bound[role].append({
                "canonical_id": entity["canonical_id"],
                "canonical_name": entity["canonical_name"],
                "entity_type": entity["entity_type"],
            })
    return {
        **rule,
        "status": "READY_WITH_SUPPLEMENTAL_ENTITIES" if supplemental_names else "VALIDATED_CANDIDATE",
        "bound_endpoints": bound,
        "supplemental_entity_names": sorted(set(supplemental_names)),
    }


def _requires_history(rule: dict[str, Any]) -> bool:
    return any("持续" in name for name in rule["inputs"])


def main() -> None:
    entity_doc = json.loads(ENTITY_PATH.read_text(encoding="utf-8"))
    entities = _entity_lookup(entity_doc)
    extracted_doc = json.loads(EXTRACTED_PATH.read_text(encoding="utf-8"))
    existing_rules = extracted_doc["rules"]
    supplements: dict[str, dict[str, Any]] = {}
    source_rules = _remaining_rules()
    temporal = [{
        **rule,
        "status": "DEFERRED_REQUIRES_HISTORY",
        "reason": "Input state requires longitudinal observations and cannot be computed from one report.",
    } for rule in source_rules if _requires_history(rule)]
    remaining = [
        _bind_endpoints(rule, entities, supplements)
        for rule in source_rules
        if not _requires_history(rule)
    ]
    blocked = [rule for rule in remaining if rule["status"] == "READY_WITH_SUPPLEMENTAL_ENTITIES"]
    ready = [rule for rule in remaining if rule["status"] == "VALIDATED_CANDIDATE"]
    output = {
        "schema_version": "chapter-manual-rule-audit/v0.1",
        "status": "HUMAN_REVIEW_REQUIRED",
        "publication_status": "HOLD",
        "boundary": (
            "Manual source review only. Missing endpoints are recorded for entity completion; "
            "this artifact never creates canonical entities."
        ),
        "source_entity_catalog": str(ENTITY_PATH.relative_to(ROOT)),
        "source_entity_catalog_status": entity_doc["status"],
        "counts": {
            "existing_validated_candidates": len(existing_rules),
            "remaining_source_rules_reviewed": len(remaining),
            "deferred_temporal_rules": len(temporal),
            "remaining_bound_to_frozen_catalog": len(ready),
            "remaining_requiring_entity_completion": len(blocked),
            "supplemental_entity_candidates": len(supplements),
            "total_source_rules_reviewed": len(existing_rules) + len(remaining) + len(temporal),
        },
        "existing_validated_candidates": existing_rules,
        "supplemental_entity_candidates": sorted(supplements.values(), key=lambda item: item["canonical_name"]),
        "remaining_bound_to_frozen_catalog": ready,
        "remaining_requiring_entity_completion": blocked,
        "deferred_temporal_rules": temporal,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
