"""使用第一章证据锚定关系参考集评测冻结实体候选关系。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = ROOT / "evaluation/chapter-01/chapter-01-relationship-gold-v1.1.json"
DEFAULT_GRAPH = ROOT / "runtime/candidates/chapter-01/knowledge-graph-v0.8/graph.json"
DEFAULT_OUTPUT = ROOT / "runtime/evaluations/chapter01-relationship-v1.1/score.json"


def _metrics(predicted: set[tuple[str, str, str]], expected: set[tuple[str, str, str]]) -> dict[str, Any]:
    tp = len(predicted & expected)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    precision = tp / (tp + fp) if tp + fp else (1.0 if not expected else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_total": len(predicted),
        "gold_total": len(expected),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "precision_percent": round(precision * 100, 2),
        "recall_percent": round(recall * 100, 2),
        "f1_percent": round(f1 * 100, 2),
    }


def _identity(item: dict[str, Any], *, gold: bool) -> tuple[str, str, str]:
    if gold:
        return (
            str(item["source_canonical_id"]),
            str(item["relation_type"]),
            str(item["target_canonical_id"]),
        )
    return (
        str(item["source_candidate_key"]),
        str(item["relation_type"]),
        str(item["target_candidate_key"]),
    )


def score(gold_path: Path, graph_path: Path) -> dict[str, Any]:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    predictions = [
        item for item in graph["relationships"]
        if isinstance(item, dict) and item.get("relation_type") not in {"RULE_INPUT", "RULE_OUTPUT"}
    ]
    cases: list[dict[str, Any]] = []
    all_predicted: set[tuple[str, str, str]] = set()
    all_expected: set[tuple[str, str, str]] = set()
    for case in gold["cases"]:
        chunk_ids = set(case["chunk_ids"])
        expected = {_identity(item, gold=True) for item in case["relationships"]}
        predicted = {
            _identity(item, gold=False)
            for item in predictions
            if item.get("source_ref", {}).get("chunk_id") in chunk_ids
        }
        all_predicted.update(predicted)
        all_expected.update(expected)
        cases.append({
            "case_id": case["case_id"],
            "title": case["title"],
            "chunk_count": len(chunk_ids),
            "metrics": _metrics(predicted, expected),
        })

    relation_types = sorted({item[1] for item in all_predicted | all_expected})
    by_type = {
        relation_type: _metrics(
            {item for item in all_predicted if item[1] == relation_type},
            {item for item in all_expected if item[1] == relation_type},
        )
        for relation_type in relation_types
    }
    confusion = Counter()
    for item in all_predicted - all_expected:
        confusion[f"FP:{item[1]}"] += 1
    for item in all_expected - all_predicted:
        confusion[f"FN:{item[1]}"] += 1
    expected_by_pair: dict[tuple[str, str], set[str]] = {}
    for source, relation_type, target in all_expected:
        expected_by_pair.setdefault((source, target), set()).add(relation_type)
    fp_diagnostics = Counter()
    type_confusions = Counter()
    for source, relation_type, target in all_predicted - all_expected:
        expected_types = expected_by_pair.get((source, target))
        if expected_types:
            fp_diagnostics["RIGHT_ENDPOINTS_WRONG_RELATION_TYPE"] += 1
            for expected_type in expected_types:
                type_confusions[f"{relation_type}->{expected_type}"] += 1
        elif (target, source) in expected_by_pair:
            fp_diagnostics["REVERSED_ENDPOINT_PAIR"] += 1
        else:
            fp_diagnostics["UNMATCHED_OVER_GENERATION"] += 1
    return {
        "schema_version": "chapter01-relationship-evaluation/v1.1",
        "status": "COMPLETED",
        "publication_status": "HOLD",
        "gold_path": str(gold_path.relative_to(ROOT)),
        "prediction_path": str(graph_path.relative_to(ROOT)),
        "gold_status": gold["status"],
        "gold_human_approved": gold["gold_provenance"]["human_approved"],
        "identity": ["source_canonical_id", "relation_type", "target_canonical_id"],
        "deduplication": "ONE_IDENTITY_COUNTS_ONCE_ACROSS_REPEATED_EVIDENCE",
        "prediction_rows": len(predictions),
        "prediction_unique_relationships": len(all_predicted),
        "prediction_duplicate_rows": len(predictions) - len(all_predicted),
        "micro": _metrics(all_predicted, all_expected),
        "by_relation_type": by_type,
        "cases": cases,
        "error_counts": dict(sorted(confusion.items())),
        "false_positive_diagnostics": dict(sorted(fp_diagnostics.items())),
        "relation_type_confusions": dict(sorted(type_confusions.items())),
        "boundary": (
            "All eight Chapter 01 sections. Gold is assistant-annotated and requires user "
            "validation; rules are excluded and cross-chunk predictions were not generated."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评测第一章全章证据锚定关系参考集")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = score(args.gold.resolve(), args.graph.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "micro": result["micro"],
        "by_relation_type": result["by_relation_type"],
        "cases": result["cases"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
