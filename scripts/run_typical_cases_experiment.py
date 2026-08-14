#!/usr/bin/env python3
"""运行 chunk 到候选知识图谱及两类质量评测的单轮主链路。"""

from __future__ import annotations

import asyncio
import json

from medical_kg_sourceprep.extraction.graph_builder.client import create_deepseek_graph_builder
from medical_kg_sourceprep.extraction.graph_builder.contract import PROJECT_ROOT
from medical_kg_sourceprep.extraction.graph_builder.runner import (
    evaluation_summary,
    run_typical_cases_evaluation,
)


GOLD_PATH = PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json"
OUTPUT_ROOT = PROJECT_ROOT / "runtime/evaluations/typical-cases/v0.1"


if __name__ == "__main__":
    client = create_deepseek_graph_builder()
    with asyncio.Runner() as runner:
        try:
            report = runner.run(run_typical_cases_evaluation(
                client,
                gold_path=GOLD_PATH,
                output_root=OUTPUT_ROOT,
                progress=lambda message: print(message, flush=True),
            ))
        finally:
            runner.run(client.aclose())
    print(json.dumps(
        evaluation_summary(report), ensure_ascii=False, indent=2, sort_keys=True
    ))
