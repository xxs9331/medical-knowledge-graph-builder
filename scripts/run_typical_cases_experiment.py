#!/usr/bin/env python3
"""运行 chunk 到候选知识图谱及两类质量评测的单轮主链路。"""

from __future__ import annotations

import asyncio
import argparse
import json
from pathlib import Path
from typing import cast

from medical_kg_sourceprep.extraction.graph_builder.client import create_deepseek_graph_builder
from medical_kg_sourceprep.extraction.graph_builder.contract import PROJECT_ROOT
from medical_kg_sourceprep.extraction.graph_builder.runner import (
    evaluation_summary,
    run_typical_cases_evaluation,
)


GOLD_PATH = PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json"
OUTPUT_ROOT = PROJECT_ROOT / "runtime/evaluations/typical-cases/v0.1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行真实典型案例的单轮抽取与评测")
    _ = parser.add_argument(
        "--gold",
        type=Path,
        default=GOLD_PATH,
        help="图金标数据集；默认使用 8 个典型案例。",
    )
    _ = parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="只运行指定案例；可重复提供。默认运行全部案例。",
    )
    _ = parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="不调用无监督 Judge，候选通过硬校验后直接进入本地金标评分。",
    )
    _ = parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="独立实验输出目录；已有完整工件会被复用。",
    )
    _ = parser.add_argument(
        "--relation-mode",
        choices=("generative", "two-stage-classification"),
        default="generative",
        help="关系抽取方式；二阶段模式先判有无关系，再分类类型和方向。",
    )
    _ = parser.add_argument(
        "--allow-review-required-gold",
        action="store_true",
        help="允许使用尚待人工终审的典型案例草稿做开发实验；报告仍保留其真实状态。",
    )
    args = parser.parse_args()
    selected_case_ids = cast(list[str] | None, args.case_ids)
    experiment_root = cast(Path, args.output_root)
    gold_path = cast(Path, args.gold)
    client = create_deepseek_graph_builder()
    with asyncio.Runner() as runner:
        try:
            report = runner.run(run_typical_cases_evaluation(
                client,
                gold_path=gold_path,
                output_root=experiment_root,
                case_ids=set(selected_case_ids) if selected_case_ids else None,
                relation_extraction_mode=args.relation_mode,
                allow_review_required_gold=args.allow_review_required_gold,
                run_judge=not args.skip_judge,
                progress=lambda message: print(message, flush=True),
            ))
        finally:
            runner.run(client.aclose())
    print(json.dumps(
        evaluation_summary(report), ensure_ascii=False, indent=2, sort_keys=True
    ))
