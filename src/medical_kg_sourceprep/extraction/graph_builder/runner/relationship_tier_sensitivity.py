"""使用冻结候选图运行关系标注合同敏感性实验。"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _BootstrapPath

    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[4]))
    __package__ = "medical_kg_sourceprep.extraction.graph_builder.runner"

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..contract import DEFAULT_CHUNK_MANIFEST, PROJECT_ROOT, GraphBuilderConfigurationError
from ..evaluation.artifacts import load_json_object
from ..evaluation.scoring import merge_candidate_graphs, score_relationship_tier


DEFAULT_GOLD = PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json"
DEFAULT_POLICY = (
    PROJECT_ROOT / "evaluation/typical-cases/relationship-tier-policy-v0.1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "runtime/evaluations/relationship-tier-sensitivity-r1/evaluation-result.json"
)
BASELINE_ROOT = (
    PROJECT_ROOT / "runtime/evaluations/typical-cases/structured-rules-v0.10-full"
)
T1_ROOT = PROJECT_ROOT / "runtime/evaluations/table-context-prompt-t1-full"
T1_RUNS = (
    "20260817-table-context-full-r01",
    "20260817-table-context-full-r02",
    "20260817-table-context-full-r03",
)


def _sha256(path: Path) -> str:
    """计算输入工件哈希，保证离线重评分可以回放。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prf1(tp: int, fp: int, fn: int) -> dict[str, Any]:
    """根据跨案例累计计数计算标准监督 P/R/F1。"""
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_total": tp + fp,
        "gold_total": tp + fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "precision_percent": round(precision * 100, 2),
        "recall_percent": round(recall * 100, 2),
        "f1_percent": round(f1 * 100, 2),
    }


def _chunk_index(manifest_path: Path) -> dict[str, tuple[str, Path]]:
    """建立 chunk ID 到原文和文件路径的映射。"""
    manifest = load_json_object(manifest_path)
    index: dict[str, tuple[str, Path]] = {}
    for item in manifest.get("chunks", []):
        if not isinstance(item, Mapping):
            continue
        chunk_id = item.get("chunk_id")
        chunk_path = item.get("chunk_path")
        if not isinstance(chunk_id, str) or not isinstance(chunk_path, str):
            continue
        path = manifest_path.parent / chunk_path
        index[chunk_id] = (path.read_text(encoding="utf-8"), path)
    return index


def _graph_path(root: Path, chunk_id: str, *, t1: bool) -> Path:
    """按稳定 chunk 后缀解析冻结候选图路径。"""
    parts = chunk_id.split(":")
    if len(parts) < 2:
        raise GraphBuilderConfigurationError(f"invalid_chunk_id:{chunk_id}")
    chunk_dir = root / "chunks" / f"{parts[-2]}-{parts[-1]}"
    return chunk_dir / ("hybrid-replace-graph.json" if t1 else "candidate-graph/graph.json")


def _case_tiers(
    case: Mapping[str, Any], policy: Mapping[str, Any]
) -> tuple[list[list[Any]], list[list[Any]]]:
    """读取案例分层；没有覆盖项时把原金标关系全部归入显式层。"""
    def relationship_list(value: object) -> list[list[Any]]:
        if not isinstance(value, list):
            return []
        return [list(item) for item in value if isinstance(item, (list, tuple))]

    case_id = str(case.get("case_id"))
    raw_overrides = policy.get("case_overrides", {})
    overrides = raw_overrides if isinstance(raw_overrides, Mapping) else {}
    override = overrides.get(case_id)
    original = relationship_list(case.get("relationships", []))
    if not isinstance(override, Mapping):
        return original, []
    explicit = relationship_list(override.get("explicit_relationships", []))
    structural = relationship_list(override.get("structural_derived_relationships", []))
    # 分层只能重新划分现有金标，不能在实验中静默增删目标。
    if sorted(map(tuple, explicit + structural)) != sorted(map(tuple, original)):
        raise GraphBuilderConfigurationError(f"relationship_tier_partition_invalid:{case_id}")
    return explicit, structural


def _score_group(
    *,
    group_id: str,
    graph_root: Path,
    t1: bool,
    cases: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    chunks: Mapping[str, tuple[str, Path]],
) -> dict[str, Any]:
    """对一个冻结实验组逐案例评分并汇总两个关系层级。"""
    case_results: list[dict[str, Any]] = []
    graph_inputs: dict[str, dict[str, str]] = {}
    for case in cases:
        case_id = str(case.get("case_id"))
        chunk_ids = [str(item) for item in case.get("chunk_ids", [])]
        graph_paths = [_graph_path(graph_root, chunk_id, t1=t1) for chunk_id in chunk_ids]
        missing = [str(path) for path in graph_paths if not path.is_file()]
        if missing:
            raise GraphBuilderConfigurationError(f"frozen_graph_missing:{missing}")
        graph = merge_candidate_graphs(load_json_object(path) for path in graph_paths)
        source_text = "\n".join(chunks[chunk_id][0] for chunk_id in chunk_ids)
        explicit, structural = _case_tiers(case, policy)
        case_results.append({
            "case_id": case_id,
            "explicit": score_relationship_tier(
                graph,
                case,
                targets=explicit,
                ignored_targets=structural,
                source_text=source_text,
            ),
            "structural_derived": score_relationship_tier(
                graph,
                case,
                targets=structural,
                ignored_targets=explicit,
                source_text=source_text,
            ),
        })
        graph_inputs[case_id] = {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in graph_paths
        }

    tiers = {}
    for tier in ("explicit", "structural_derived"):
        # 结构层只在策略声明了结构目标的案例上有定义；无结构目标案例的普通关系
        # 不能被重复计为结构层 FP。显式主层则覆盖全部案例，包括零目标案例的错误边。
        tier_cases = case_results if tier == "explicit" else [
            item for item in case_results if int(item[tier]["target_total"]) > 0
        ]
        tiers[tier] = _prf1(
            sum(int(item[tier]["tp"]) for item in tier_cases),
            sum(int(item[tier]["fp"]) for item in tier_cases),
            sum(int(item[tier]["fn"]) for item in tier_cases),
        )
    return {
        "group_id": group_id,
        "tiers": tiers,
        "cases": case_results,
        "graph_inputs": graph_inputs,
    }


def run_relationship_tier_sensitivity(
    *,
    gold_path: Path = DEFAULT_GOLD,
    policy_path: Path = DEFAULT_POLICY,
    manifest_path: Path = DEFAULT_CHUNK_MANIFEST,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """对 A 和三次 T1-H 冻结预测运行关系分层敏感性实验。"""
    gold = load_json_object(gold_path)
    policy = load_json_object(policy_path)
    if policy.get("source_gold_sha256") != _sha256(gold_path):
        raise GraphBuilderConfigurationError("relationship_tier_gold_hash_mismatch")
    cases = [item for item in gold.get("cases", []) if isinstance(item, Mapping)]
    chunks = _chunk_index(manifest_path)
    groups = [
        _score_group(
            group_id="A",
            graph_root=BASELINE_ROOT,
            t1=False,
            cases=cases,
            policy=policy,
            chunks=chunks,
        )
    ]
    groups.extend(
        _score_group(
            group_id=f"T1-H-r{index:02d}",
            graph_root=T1_ROOT / run_id,
            t1=True,
            cases=cases,
            policy=policy,
            chunks=chunks,
        )
        for index, run_id in enumerate(T1_RUNS, start=1)
    )
    t1_groups = groups[1:]
    t1_mean = {
        tier: {
            metric: round(sum(group["tiers"][tier][metric] for group in t1_groups) / 3, 2)
            for metric in ("precision_percent", "recall_percent", "f1_percent")
        }
        for tier in ("explicit", "structural_derived")
    }
    result = {
        "schema_version": "relationship-tier-sensitivity-result/v0.1",
        "status": "HUMAN_REVIEW_REQUIRED",
        "publication_status": "HOLD",
        "gold_exposed_to_model": False,
        "model_calls": 0,
        "scoring_contract": "P=TP/(TP+FP), R=TP/(TP+FN); typed directed exact-tier micro scoring",
        "inputs": {
            "gold": {"path": str(gold_path), "sha256": _sha256(gold_path)},
            "policy": {"path": str(policy_path), "sha256": _sha256(policy_path)},
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        },
        "groups": groups,
        "t1_h_three_run_mean_percent": t1_mean,
        "interpretation_boundary": (
            "This is an annotation-contract sensitivity analysis over frozen candidate graphs, "
            "not a replacement gold dataset or evidence of publication readiness."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线运行关系标注合同敏感性实验。")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CHUNK_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    experiment = run_relationship_tier_sensitivity(
        gold_path=args.gold,
        policy_path=args.policy,
        manifest_path=args.manifest,
        output_path=args.output,
    )
    summary = {
        group["group_id"]: group["tiers"] for group in experiment["groups"]
    }
    summary["T1-H-mean-percent"] = experiment["t1_h_three_run_mean_percent"]
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
