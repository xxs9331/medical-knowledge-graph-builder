"""从典型案例图金标生成实体、关系和规则三个只读视图。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
GRAPH_PATH = ROOT / "typical-cases-v0.1.json"


def _base_view(graph: dict[str, Any], schema_version: str, identity: list[str]) -> dict[str, Any]:
    """复制所有派生数据集共用的身份和闭集范围合同。"""
    return {
        "schema_version": schema_version,
        "status": graph["status"],
        "source_graph_dataset": GRAPH_PATH.name,
        "annotation_method": graph["annotation_method"],
        "scope_contract": graph["scope_contract"],
        "identity": identity,
    }


def _case_base(case: dict[str, Any]) -> dict[str, Any]:
    """保留案例身份、真实 chunk 范围和冻结证据区间。"""
    return {
        "case_id": case["case_id"],
        "chunk_ids": case["chunk_ids"],
        "evaluation_scopes": case["evaluation_scopes"],
    }


def main() -> None:
    """机械生成三个视图；人工只编辑图金标这一份真源。"""
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    entities = {
        **_base_view(
            graph,
            "medical-kg-entity-test-set/v0.2",
            ["entity_type", "mention"],
        ),
        "cases": [
            {**_case_base(case), "expected": case["entities"]}
            for case in graph["cases"]
        ],
    }
    relationships = {
        **_base_view(
            graph,
            "medical-kg-relationship-test-set/v0.2",
            ["source_mention", "relation_type", "target_mention"],
        ),
        "notes": [
            "expected 是应抽取正例；forbidden 是范围内不支持或会破坏联合条件的负例。"
        ],
        "cases": [
            {
                **_case_base(case),
                "expected": case["relationships"],
                "forbidden": case["must_not_extract"],
                **(
                    {"held_semantics": case["held_semantics"]}
                    if "held_semantics" in case else {}
                ),
            }
            for case in graph["cases"]
        ],
    }
    rules = {
        **_base_view(
            graph,
            "medical-kg-rule-test-set/v0.2",
            ["rule_stage", "ordered_inputs", "ordered_outputs", "logic"],
        ),
        "notes": [
            "GRAPH_COMPOSITE 输入不得降级为输入到输出的普通直达边。",
            "公式、参考区间、阈值分级和单指标时间计算由后续执行器抽取负责。",
        ],
        "cases": [
            {
                **_case_base(case),
                "expected": case["rules"],
                "review_notes": case["review_notes"],
            }
            for case in graph["cases"]
        ],
    }
    for filename, document in (
        ("entity-test-set-v0.1.json", entities),
        ("relationship-test-set-v0.1.json", relationships),
        ("rule-test-set-v0.1.json", rules),
    ):
        (ROOT / filename).write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
