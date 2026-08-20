"""从 Chapter 01 v0.1 标注草稿构建带范围与证据审查队列的 v0.2。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
V01_PATH = ROOT / "chapter-01-graph-test-set-v0.1.json"
V02_PATH = ROOT / "chapter-01-graph-test-set-v0.2.json"
AUDIT_PATH = ROOT / "chapter-01-evidence-audit-v0.2.json"
MANIFEST_PATH = REPO_ROOT / "source-packages/canonical/evidence/chapter-01/manifest.json"

RUNTIME_PARAMETERS = frozenset({
    "年龄", "性别", "单位", "ISI", "试剂参考上限", "正常人血浆PT", "正常对照PT",
})


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _scope(chunk_id: str, char_counts: dict[str, int]) -> dict[str, Any]:
    return {"chunk_id": chunk_id, "start": 0, "end": char_counts[chunk_id]}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remove_dvt_positive_relationship(case: dict[str, Any]) -> None:
    target = ["深静脉血栓形成", "ASSOCIATED_WITH", "D-二聚体阳性"]
    case["relationships"] = [item for item in case["relationships"] if item != target]
    case["must_not_extract"] = [target]
    case["review_notes"] = [
        "原文只说明 D-二聚体正常对排除深静脉血栓有价值，"
        "不把医学常识推断为 D-二聚体阳性的直接关系。",
    ]


def build_graph() -> dict[str, Any]:
    graph = deepcopy(_load_json(V01_PATH))
    manifest = _load_json(MANIFEST_PATH)
    char_counts = {item["chunk_id"]: item["char_count"] for item in manifest["chunks"]}

    graph["schema_version"] = "medical-kg-chapter-gold/v0.2"
    graph["annotation_method"] = "MANUAL_READING_WITH_AUTO_PROVENANCE_AUDIT"
    graph["scope_contract"] = (
        "每个案例只评价 evaluation_scopes 内有完整证据的候选；"
        "范围内未匹配预测记 FP，未匹配金标记 FN。"
    )
    graph["provenance_audit"] = {
        "artifact": AUDIT_PATH.name,
        "contract": (
            "AUTO_SURFACE_MATCH 仅证明字面出现，不等于语义支持；"
            "所有 NEEDS_HUMAN_REVIEW 项必须在发布金标前人工裁决。"
        ),
    }
    graph["source_chunk_manifest"] = str(MANIFEST_PATH.relative_to(REPO_ROOT))
    graph["source_chunk_manifest_sha256"] = _sha256(MANIFEST_PATH)
    graph["scoring_status"] = "HOLD_UNTIL_HUMAN_VALIDATED"
    graph["notes"].extend([
        "v0.2 为每个主题单元冻结完整 evaluation_scopes，并单独保存逐项证据审查队列。",
        "rules 只保存当前图抽取合同支持的 GRAPH_COMPOSITE；"
        "PREPROCESS 放入 executor_rules，书中未给出具体输入配对的公式放入 held_rules。",
        "HUMAN_REVIEW_REQUIRED 数据只允许开发期诊断，不得报告为正式测试性能。",
    ])

    for case in graph["cases"]:
        case_id = case["case_id"]
        case["evaluation_scopes"] = [
            _scope(chunk_id, char_counts)
            for chunk_id in case["chunk_ids"]
        ]
        case.setdefault("must_not_extract", [])
        case.setdefault("review_notes", [])

        preprocess_rules = [
            item for item in case["rules"] if item["rule_stage"] == "PREPROCESS"
        ]
        case["rules"] = [
            item for item in case["rules"] if item["rule_stage"] == "GRAPH_COMPOSITE"
        ]
        held_rules: list[dict[str, Any]] = []
        if case_id == "CH01-02":
            case["rules"] = [
                {**rule, "inputs": [item for item in rule["inputs"] if item != "贫血"]}
                for rule in case["rules"]
            ]
            unsupported_formulas = {
                ("红细胞计数", "红细胞压积", "MCV"),
                ("红细胞计数", "血红蛋白", "MCH"),
                ("红细胞压积", "血红蛋白", "MCHC"),
            }
            retained: list[dict[str, Any]] = []
            for rule in preprocess_rules:
                signature = (*rule["inputs"], *rule["outputs"])
                if signature in unsupported_formulas:
                    held_rules.append({
                        **rule,
                        "hold_reason": (
                            "原文只说按公式计算三种平均指数，未给出具体公式或输入配对。"
                        ),
                    })
                else:
                    retained.append(rule)
            preprocess_rules = retained
        if case_id == "CH01-05":
            required_entities = [
                ["Disease", "慢性感染"],
                ["ClinicalContext", "铁剂治疗过量"],
                ["Disease", "地中海贫血"],
            ]
            for entity in required_entities:
                if entity not in case["entities"]:
                    case["entities"].append(entity)
            for rule in case["rules"]:
                if rule["inputs"] == ["血清铁增高", "TIBC降低"]:
                    rule["outputs"] = [
                        "铁剂治疗过量",
                        "溶血性贫血",
                        "再生障碍性贫血",
                        "巨幼细胞贫血",
                        "地中海贫血",
                    ]
            case["review_notes"].append(
                "补齐二维联合表规则输出，保证与 8 案例同源规则一致且端点闭合。"
            )
        case["executor_rules"] = preprocess_rules
        case["held_rules"] = held_rules

        if case_id == "CH01-06":
            case["entities"] = [
                ["IndicatorState", mention]
                if entity_type == "ClinicalContext" and mention == "红细胞压积增高"
                else [entity_type, mention]
                for entity_type, mention in case["entities"]
            ]
            case["review_notes"].append(
                "“红细胞压积增高”统一为 IndicatorState，与章节前段类型保持一致。"
            )
        if case_id == "CH01-08":
            _remove_dvt_positive_relationship(case)

    graph["counts"] = {
        "entity_records": sum(len(case["entities"]) for case in graph["cases"]),
        "positive_relationships": sum(
            len(case["relationships"]) for case in graph["cases"]
        ),
        "graph_rules": sum(len(case["rules"]) for case in graph["cases"]),
        "executor_rules": sum(len(case["executor_rules"]) for case in graph["cases"]),
        "held_rules": sum(len(case["held_rules"]) for case in graph["cases"]),
        "forbidden_relationships": sum(
            len(case["must_not_extract"]) for case in graph["cases"]
        ),
    }
    return graph


def _first_surface_match(text_by_chunk: dict[str, str], mention: str) -> dict[str, Any] | None:
    for chunk_id, text in text_by_chunk.items():
        start = text.find(mention)
        if start >= 0:
            line_start = text.rfind("\n", 0, start) + 1
            line_end = text.find("\n", start)
            if line_end < 0:
                line_end = len(text)
            return {
                "chunk_id": chunk_id,
                "start": start,
                "end": start + len(mention),
                "exact_quote": text[line_start:line_end],
            }
    return None


def _item_audit(
    *, kind: str, value: Any, text_by_chunk: dict[str, str]
) -> dict[str, Any]:
    if kind == "entity":
        mention = value[1]
        match = _first_surface_match(text_by_chunk, mention)
        return {
            "kind": kind,
            "target": value,
            "status": "AUTO_SURFACE_MATCH" if match else "NEEDS_HUMAN_REVIEW",
            **({"evidence": [match]} if match else {}),
        }

    if kind == "relationship":
        source, relation_type, target = value
        for chunk_id, text in text_by_chunk.items():
            source_start = text.find(source)
            target_start = text.find(target)
            if source_start >= 0 and target_start >= 0:
                line_start = text.rfind("\n", 0, min(source_start, target_start)) + 1
                line_end = text.find("\n", max(source_start, target_start))
                if line_end < 0:
                    line_end = len(text)
                quote = text[line_start:line_end]
                if source in quote and target in quote:
                    return {
                        "kind": kind,
                        "target": value,
                        "status": "AUTO_SURFACE_MATCH",
                        "evidence": [{"chunk_id": chunk_id, "exact_quote": quote}],
                    }
        return {"kind": kind, "target": value, "status": "NEEDS_HUMAN_REVIEW"}

    if kind == "held_rule":
        return {
            "kind": kind,
            "target": value,
            "status": "HELD_EXTERNAL_KNOWLEDGE",
            "reason": value["hold_reason"],
        }

    inputs = [item for item in value["inputs"] if item not in RUNTIME_PARAMETERS]
    outputs = value["outputs"]
    for chunk_id, text in text_by_chunk.items():
        if all(item in text for item in [*inputs, *outputs]):
            return {
                "kind": kind,
                "target": value,
                "status": "AUTO_SURFACE_MATCH",
                "evidence": [{"chunk_id": chunk_id, "exact_quote": text}],
            }
    return {"kind": kind, "target": value, "status": "NEEDS_HUMAN_REVIEW"}


def build_audit(graph: dict[str, Any]) -> dict[str, Any]:
    manifest = _load_json(MANIFEST_PATH)
    paths = {item["chunk_id"]: item["chunk_path"] for item in manifest["chunks"]}
    cases: list[dict[str, Any]] = []
    totals = {
        "AUTO_SURFACE_MATCH": 0,
        "NEEDS_HUMAN_REVIEW": 0,
        "HELD_EXTERNAL_KNOWLEDGE": 0,
        "MANUAL_NEGATIVE_RATIONALE_DRAFT": 0,
    }
    for case in graph["cases"]:
        text_by_chunk = {
            scope["chunk_id"]: (REPO_ROOT / "source-packages/canonical/evidence/chapter-01" / paths[scope["chunk_id"]]).read_text(encoding="utf-8")
            for scope in case["evaluation_scopes"]
        }
        items = [
            *[_item_audit(kind="entity", value=item, text_by_chunk=text_by_chunk) for item in case["entities"]],
            *[_item_audit(kind="relationship", value=item, text_by_chunk=text_by_chunk) for item in case["relationships"]],
            *[_item_audit(kind="graph_rule", value=item, text_by_chunk=text_by_chunk) for item in case["rules"]],
            *[_item_audit(kind="executor_rule", value=item, text_by_chunk=text_by_chunk) for item in case["executor_rules"]],
            *[_item_audit(kind="held_rule", value=item, text_by_chunk=text_by_chunk) for item in case["held_rules"]],
        ]
        for forbidden in case["must_not_extract"]:
            items.append({
                "kind": "forbidden_relationship",
                "target": forbidden,
                "status": "MANUAL_NEGATIVE_RATIONALE_DRAFT",
                "evidence": [{
                    "chunk_id": "clinical-hematology:chapter-01:0023:0001",
                    "exact_quote": (
                        "D-二聚体正常，对排除深静脉血栓(DVT)有重要价值。"
                    ),
                }],
                "reason": (
                    "原文表达排除诊断，不直接支持深静脉血栓形成与 D-二聚体阳性的正向关系。"
                ),
            })
        for item in items:
            totals[item["status"]] += 1
        cases.append({"case_id": case["case_id"], "items": items})
    return {
        "schema_version": "medical-kg-evidence-audit/v0.2",
        "status": "HUMAN_REVIEW_REQUIRED",
        "source_graph_dataset": V02_PATH.name,
        "contract": (
            "该清单用于定位人工审查；AUTO_SURFACE_MATCH 不是语义批准，"
            "NEEDS_HUMAN_REVIEW 包含派生、别名、跨句与规则语义；"
            "负例和 HELD 规则也必须单独裁决。"
        ),
        "summary": {"total_items": sum(totals.values()), **totals},
        "cases": cases,
    }


def main() -> None:
    graph = build_graph()
    _write_json(V02_PATH, graph)
    _write_json(AUDIT_PATH, build_audit(graph))


if __name__ == "__main__":
    main()
