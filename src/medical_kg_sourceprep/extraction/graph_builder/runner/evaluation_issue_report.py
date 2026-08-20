"""从一次评分运行生成确定性问题报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation.issues import build_issues_from_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成候选图确定性评测问题报告")
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--review-queue", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_issues_from_paths(
        score_path=args.score,
        manifest_path=args.manifest,
        review_queue_path=args.review_queue,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "issue_count": report["issue_count"],
        "counts_by_severity": report["counts_by_severity"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
