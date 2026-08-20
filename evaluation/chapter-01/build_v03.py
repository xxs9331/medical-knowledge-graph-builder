"""从 Chapter 01 v0.2 构建补齐推理与排除规则的最终 v0.3。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import build_v02  # pyright: ignore[reportImplicitRelativeImport]


ROOT = Path(__file__).resolve().parent
GRAPH_PATH = ROOT / "chapter-01-graph-test-set-v0.3.json"
AUDIT_PATH = ROOT / "chapter-01-evidence-audit-v0.3.json"


def _append_unique(target: list[Any], values: list[Any]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _rule(inputs: list[str], outputs: list[str]) -> dict[str, Any]:
    return {
        "rule_stage": "GRAPH_COMPOSITE",
        "inputs": inputs,
        "outputs": outputs,
        "logic": "ALL",
    }


def _add_anemia_morphology_rules(case: dict[str, Any]) -> None:
    _append_unique(case["entities"], [
        ["IndicatorState", "MCH增大"],
        ["IndicatorState", "MCH正常"],
        ["IndicatorState", "MCH减小"],
        ["IndicatorState", "MCH显著减小(<23pg)"],
        ["IndicatorState", "MCHC正常"],
        ["IndicatorState", "MCHC减小"],
        ["ClinicalContext", "大细胞性贫血"],
        ["ClinicalContext", "正细胞性贫血"],
        ["ClinicalContext", "单纯小细胞性贫血"],
        ["ClinicalContext", "小细胞低色素性贫血"],
    ])
    _append_unique(case["rules"], [
        _rule(["MCV增大", "MCH增大", "MCHC正常"], ["大细胞性贫血"]),
        _rule(["MCV正常", "MCH正常", "MCHC正常"], ["正细胞性贫血"]),
        _rule(["MCV减小", "MCH减小", "MCHC正常"], ["单纯小细胞性贫血"]),
        _rule(
            ["MCV减小", "MCH显著减小(<23pg)", "MCHC减小"],
            ["小细胞低色素性贫血"],
        ),
    ])
    case["review_notes"].append(
        "补入表1-2的4条MCV/MCH/MCHC联合分类规则；病因列仍作为普通关联。"
    )


BLOOD_TYPE_ROWS = [
    ("O+O", ["O"], ["A", "B", "AB"]),
    ("A+O", ["A", "O"], ["B", "AB"]),
    ("B+O", ["B", "O"], ["A", "AB"]),
    ("AB+O", ["A", "B"], ["O", "AB"]),
    ("A+A", ["A", "O"], ["B", "AB"]),
    ("B+B", ["B", "O"], ["A", "AB"]),
    ("AB+AB", ["A", "B", "AB"], ["O"]),
    ("A+B", ["A", "B", "AB", "O"], []),
    ("A+AB", ["A", "B", "AB"], ["O"]),
    ("B+AB", ["A", "B", "AB"], ["O"]),
]


def _add_blood_type_rules(case: dict[str, Any]) -> None:
    case["entities"] = [
        ["IndicatorState", mention]
        if entity_type == "ClinicalContext" and mention in {
            "A型血", "B型血", "AB型血", "O型血",
        }
        else [entity_type, mention]
        for entity_type, mention in case["entities"]
    ]
    _append_unique(case["entities"], [["LabIndicator", "ABO血型"]])
    _append_unique(case["relationships"], [
        ["ABO血型鉴定", "HAS_METRIC", "ABO血型"],
        *[["ABO血型", "HAS_STATE", f"{value}型血"] for value in ("A", "B", "AB", "O")],
    ])

    _append_unique(case["entities"], [
        ["LabPanel", "ABO血型遗传规律"],
        ["LabIndicator", "父母血型"],
        ["LabIndicator", "子女可能的血型"],
        ["LabIndicator", "子女不可能的血型"],
    ])
    combination_entities = [
        ["IndicatorState", f"父母血型组合为{combination}"]
        for combination, _possible, _impossible in BLOOD_TYPE_ROWS
    ]
    result_entities = [
        ["IndicatorState", f"子女可能为{value}型血"]
        for value in ("A", "B", "AB", "O")
    ] + [
        ["IndicatorState", f"子女不可能为{value}型血"]
        for value in ("A", "B", "AB", "O")
    ]
    _append_unique(case["entities"], [*combination_entities, *result_entities])
    _append_unique(case["relationships"], [
        ["ABO血型遗传规律", "HAS_METRIC", "父母血型"],
        ["ABO血型遗传规律", "HAS_METRIC", "子女可能的血型"],
        ["ABO血型遗传规律", "HAS_METRIC", "子女不可能的血型"],
        *[
            ["父母血型", "HAS_STATE", f"父母血型组合为{combination}"]
            for combination, _possible, _impossible in BLOOD_TYPE_ROWS
        ],
        *[
            ["子女可能的血型", "HAS_STATE", f"子女可能为{value}型血"]
            for value in ("A", "B", "AB", "O")
        ],
        *[
            ["子女不可能的血型", "HAS_STATE", f"子女不可能为{value}型血"]
            for value in ("A", "B", "AB", "O")
        ],
    ])

    rules: list[dict[str, Any]] = []
    for combination, possible, impossible in BLOOD_TYPE_ROWS:
        input_text = f"父母血型组合为{combination}"
        rules.append(_rule(
            [input_text], [f"子女可能为{value}型血" for value in possible]
        ))
        if impossible:
            rules.append(_rule(
                [input_text], [f"子女不可能为{value}型血" for value in impossible]
            ))
    _append_unique(case["rules"], rules)
    case["review_notes"].append(
        "表1-6按无序父母血型组合拆为10条可能性推理和9条排除规则；"
        "可能/不可能直接保留在输出文本中。"
    )


def _add_viscosity_exclusions(case: dict[str, Any]) -> None:
    _append_unique(case["entities"], [
        ["ClinicalContext", "单次血黏度升高"],
        ["ClinicalContext", "仅使用血黏度参数"],
        ["ClinicalContext", "不能据此诊断心脑血管疾病"],
        ["ClinicalContext", "不能据此进行脑卒中预报"],
    ])
    _append_unique(case["rules"], [
        _rule(["单次血黏度升高"], ["不能据此诊断心脑血管疾病"]),
        _rule(["仅使用血黏度参数"], ["不能据此进行脑卒中预报"]),
    ])
    case["review_notes"].append(
        "补入原文明示的单次血黏度诊断限制和脑卒中预报排除规则。"
    )


def _add_d_dimer_exclusion(case: dict[str, Any]) -> None:
    _append_unique(case["entities"], [
        ["ClinicalContext", "排除深静脉血栓有重要价值"],
    ])
    _append_unique(case["rules"], [
        _rule(["D-二聚体正常"], ["排除深静脉血栓有重要价值"]),
    ])
    case["review_notes"].append(
        "补入D-二聚体正常对排除深静脉血栓具有重要价值的排除规则；"
        "不等同于深静脉血栓与D-二聚体阳性的正向关系。"
    )


def build_graph() -> dict[str, Any]:
    graph = build_v02.build_graph()
    graph["schema_version"] = "medical-kg-chapter-gold/v0.3"
    graph["provenance_audit"]["artifact"] = AUDIT_PATH.name
    graph["notes"].extend([
        "v0.3补齐表1-2贫血分类、表1-6血型可能性/排除规则及原文明示的诊断排除规则。",
        "完整组合条件和可能/不可能语义使用普通文本端点表达，不增加规则JSON字段。",
    ])

    for case in graph["cases"]:
        if case["case_id"] == "CH01-02":
            _add_anemia_morphology_rules(case)
        elif case["case_id"] == "CH01-06":
            _add_viscosity_exclusions(case)
        elif case["case_id"] == "CH01-07":
            _add_blood_type_rules(case)
        elif case["case_id"] == "CH01-08":
            _add_d_dimer_exclusion(case)

    graph["counts"] = {
        "entity_records": sum(len(case["entities"]) for case in graph["cases"]),
        "positive_relationships": sum(len(case["relationships"]) for case in graph["cases"]),
        "graph_rules": sum(len(case["rules"]) for case in graph["cases"]),
        "executor_rules": sum(len(case["executor_rules"]) for case in graph["cases"]),
        "held_rules": sum(len(case["held_rules"]) for case in graph["cases"]),
        "forbidden_relationships": sum(len(case["must_not_extract"]) for case in graph["cases"]),
    }
    return graph


def main() -> None:
    graph = build_graph()
    build_v02._write_json(GRAPH_PATH, graph)
    audit = build_v02.build_audit(graph)
    audit["schema_version"] = "medical-kg-evidence-audit/v0.3"
    audit["source_graph_dataset"] = GRAPH_PATH.name
    build_v02._write_json(AUDIT_PATH, audit)


if __name__ == "__main__":
    main()
