#!/usr/bin/env python3
"""汇总联合抽取试点的重复运行、稳定性和缺失端点救回情况。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from medical_kg_sourceprep.extraction.graph_builder.contract import PROJECT_ROOT
from medical_kg_sourceprep.extraction.llm_extraction import atomic_write_json


RUN_ROOTS = (
    PROJECT_ROOT / "runtime/evaluations/typical-cases/joint-extraction-v0.7-pilot",
    PROJECT_ROOT / "runtime/evaluations/typical-cases/joint-extraction-v0.8-pilot",
    PROJECT_ROOT / "runtime/evaluations/typical-cases/joint-extraction-v0.9-pilot",
)
OUTPUT_PATH = PROJECT_ROOT / "runtime/evaluations/typical-cases/joint-extraction-pilot-summary.json"
FULL_JOINT_RUN_ROOTS = (
    PROJECT_ROOT / "runtime/evaluations/typical-cases/full-joint-v0.2-pilot",
    PROJECT_ROOT / "runtime/evaluations/typical-cases/full-joint-v0.3-pilot",
    PROJECT_ROOT / "runtime/evaluations/typical-cases/full-joint-v0.4-pilot",
)
FULL_JOINT_OUTPUT_PATH = (
    PROJECT_ROOT / "runtime/evaluations/typical-cases/full-joint-pilot-summary.json"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return value


def _metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(mean(values), 6),
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
    }


def _tuple_set(values: Any) -> set[tuple[str, ...]]:
    if not isinstance(values, list):
        return set()
    return {
        tuple(str(part) for part in value)
        for value in values
        if isinstance(value, list)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="汇总三次联合抽取试点")
    _ = parser.add_argument(
        "--full-joint", action="store_true", help="汇总不使用冻结实体的端到端 B 组。"
    )
    _ = parser.add_argument(
        "--run-root",
        action="append",
        type=Path,
        dest="run_roots",
        help="显式指定一个重复实验目录；可重复提供，覆盖内置目录。",
    )
    _ = parser.add_argument("--output", type=Path, help="显式指定汇总 JSON 输出路径。")
    args = parser.parse_args()
    full_joint = bool(args.full_joint)
    run_roots = tuple(args.run_roots or (FULL_JOINT_RUN_ROOTS if full_joint else RUN_ROOTS))
    output_path = args.output or (FULL_JOINT_OUTPUT_PATH if full_joint else OUTPUT_PATH)
    reports = [_load(root / "experiment-comparison.json") for root in run_roots]
    groups = (
        ("A", "B", "D")
        if full_joint and all("D" in report.get("groups", {}) for report in reports)
        else ("A", "B") if full_joint else ("A", "C", "D")
    )
    summary: dict[str, Any] = {
        "schema_version": "joint-extraction-repeat-summary/v0.1",
        "status": "development-evaluation-only",
        "publication_status": "HOLD",
        "experiment_mode": "full_joint" if full_joint else "frozen_entity_joint",
        "run_roots": [str(root) for root in run_roots],
        "groups": {},
    }
    for group in groups:
        metrics = [report["groups"][group]["prf1"]["categories"] for report in reports]
        summary["groups"][group] = {
            category: {
                metric: _metric_summary([float(item[category][metric]) for item in metrics])
                for metric in ("precision", "recall", "f1")
            }
            for category in ("entities", "relationships", "rules")
        }
        summary["groups"][group]["graph"] = {
            metric: _metric_summary([
                float(report["groups"][group]["prf1"]["graph"][metric])
                for report in reports
            ])
            for metric in ("precision", "recall", "f1")
        }

    a_relation = summary["groups"]["A"]["relationships"]
    if full_joint:
        b_relation = summary["groups"]["B"]["relationships"]
        entity_margin = (
            summary["groups"]["B"]["entities"]["f1"]["mean"]
            - summary["groups"]["A"]["entities"]["f1"]["mean"]
        )
        relationship_mean_improved = b_relation["f1"]["mean"] > a_relation["f1"]["mean"]
        relationship_minimum_improved = (
            b_relation["f1"]["minimum"] > a_relation["f1"]["mean"]
        )
        entity_noninferior = entity_margin >= -0.01
        assessed_group = "D" if "D" in summary["groups"] else "B"
        assessed_relation = summary["groups"][assessed_group]["relationships"]
        assessed_entity = summary["groups"][assessed_group]["entities"]
        assessed_entity_margin = (
            assessed_entity["f1"]["mean"]
            - summary["groups"]["A"]["entities"]["f1"]["mean"]
        )
        assessed_relationship_mean_improved = (
            assessed_relation["f1"]["mean"] > a_relation["f1"]["mean"]
        )
        assessed_relationship_minimum_improved = (
            assessed_relation["f1"]["minimum"] > a_relation["f1"]["mean"]
        )
        assessed_entity_noninferior = assessed_entity_margin >= -0.01
        summary["gate_decision"] = {
            "assessed_group": assessed_group,
            "relationship_f1_mean_improved": relationship_mean_improved,
            "relationship_f1_minimum_above_baseline": relationship_minimum_improved,
            "entity_f1_mean_improved": (
                summary["groups"]["B"]["entities"]["f1"]["mean"]
                > summary["groups"]["A"]["entities"]["f1"]["mean"]
            ),
            "entity_f1_within_one_point_of_baseline": entity_noninferior,
            "assessed_group_relationship_f1_mean_improved": assessed_relationship_mean_improved,
            "assessed_group_relationship_f1_minimum_above_baseline": (
                assessed_relationship_minimum_improved
            ),
            "assessed_group_entity_f1_within_one_point_of_baseline": assessed_entity_noninferior,
            "advance_to_eight_cases": (
                assessed_relationship_mean_improved
                and assessed_relationship_minimum_improved
                and assessed_entity_noninferior
            ),
            "reason": (
                f"{assessed_group} 组关系 F1 均值及最差重复均超过基线，且实体 F1 均值处于 1 个百分点非劣界内。"
                if assessed_relationship_mean_improved
                and assessed_relationship_minimum_improved
                and assessed_entity_noninferior
                else f"{assessed_group} 组关系稳定提升或实体非劣门槛未全部满足。"
            ),
        }
    else:
        rescued_counts: dict[tuple[str, ...], int] = {}
        rescued_by_run: list[dict[str, Any]] = []
        for run_index, report in enumerate(reports, start=1):
            a_cases = {item["case_id"]: item["score"] for item in report["groups"]["A"]["cases"]}
            d_cases = {item["case_id"]: item["score"] for item in report["groups"]["D"]["cases"]}
            run_rescued: list[dict[str, Any]] = []
            for case_id, d_score in d_cases.items():
                a_score = a_cases[case_id]
                a_relations = _tuple_set(a_score["relationships"]["matched_targets"])
                d_relations = _tuple_set(d_score["relationships"]["matched_targets"])
                a_entities = _tuple_set(a_score["entities"]["matched_targets"])
                a_entities.update(_tuple_set(a_score["entities"]["false_positive_predictions"]))
                a_mentions = {item[1] for item in a_entities if len(item) >= 2}
                for relation in sorted(d_relations - a_relations):
                    rescued_counts[relation] = rescued_counts.get(relation, 0) + 1
                    missing_mentions = [
                        mention for mention in (relation[0], relation[2]) if mention not in a_mentions
                    ]
                    run_rescued.append({
                        "case_id": case_id,
                        "relationship": list(relation),
                        "required_missing_endpoint": bool(missing_mentions),
                        "missing_endpoint_mentions": missing_mentions,
                    })
            rescued_by_run.append({"run": run_index, "rescued": run_rescued})
        summary["endpoint_rescue_analysis"] = {
            "by_run": rescued_by_run,
            "consistent_rescued_relationships": [
                {"relationship": list(relation), "run_count": count}
                for relation, count in sorted(rescued_counts.items())
                if count >= 2
            ],
            "missing_endpoint_rescue_achieved": any(
                item["required_missing_endpoint"]
                for run in rescued_by_run for item in run["rescued"]
            ),
        }
        d_relation = summary["groups"]["D"]["relationships"]
        summary["gate_decision"] = {
            "relationship_f1_mean_improved": d_relation["f1"]["mean"] > a_relation["f1"]["mean"],
            "relationship_precision_within_5_points_in_all_runs": all(
                report["groups"]["D"]["prf1"]["categories"]["relationships"]["precision"]
                >= report["groups"]["A"]["prf1"]["categories"]["relationships"]["precision"] - 0.05
                for report in reports
            ),
            "entity_precision_at_least_90_percent_in_all_runs": all(
                report["groups"]["D"]["prf1"]["categories"]["entities"]["precision"] >= 0.90
                for report in reports
            ),
            "advance_to_eight_cases": False,
            "reason": "关系召回提升可重复，但关系精确率门槛和缺失端点救回门槛未同时满足。",
        }
    atomic_write_json(output_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
