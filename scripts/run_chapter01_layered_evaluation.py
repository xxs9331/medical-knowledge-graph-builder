"""使用第一章现有候选图运行 v0.4 分层监督评测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medical_kg_sourceprep.extraction.graph_builder.evaluation.layered_scoring import (
    aggregate_layered_scores,
    score_layered_case,
)
from medical_kg_sourceprep.extraction.graph_builder.evaluation.scoring import (
    merge_candidate_graphs,
)


ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "evaluation/chapter-01/chapter-01-scoped-gold-v0.5.json"
DEFAULT_RUN_ROOT = (
    ROOT / "runtime/evaluations/chapter01-scale-l2-v03/"
    "20260817-152308-baseline-r01"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 v0.4 分层金标评测第一章候选图")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output_path = args.output or run_root / "layered-evaluation-result.json"

    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    case_results = []
    for case in gold["cases"]:
        graphs = []
        for chunk_id in case["chunk_ids"]:
            parts = str(chunk_id).split(":")
            chunk_slug = f"{parts[-2]}-{parts[-1]}"
            graph_path = run_root / "chunks" / chunk_slug / "candidate-graph/graph.json"
            graphs.append(json.loads(graph_path.read_text(encoding="utf-8")))
        cross_chunk_path = run_root / "cases" / case["case_id"] / "cross-chunk-graph.json"
        if cross_chunk_path.is_file():
            graphs.append(json.loads(cross_chunk_path.read_text(encoding="utf-8")))
        merged = merge_candidate_graphs(graphs)
        case_results.append({
            "case_id": case["case_id"],
            "scores": score_layered_case(merged, case),
        })

    payload = {
        "schema_version": "chapter01-layered-evaluation/v0.1",
        "gold": str(GOLD_PATH.relative_to(ROOT)),
        "predictions": str(run_root.relative_to(ROOT)),
        "status": "COMPLETED",
        "gold_status": gold["status"],
        "gold_provenance": gold["gold_provenance"],
        "cases": case_results,
        "micro": aggregate_layered_scores(item["scores"] for item in case_results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["micro"], ensure_ascii=False, indent=2, sort_keys=True))
