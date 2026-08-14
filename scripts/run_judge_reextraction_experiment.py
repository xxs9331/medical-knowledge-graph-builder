#!/usr/bin/env python3
"""只对 TC-08 运行真实抽取、审查、二次抽取和并集评分。"""

from __future__ import annotations

import asyncio
import json

from medical_kg_sourceprep.extraction.graph_builder.client import create_deepseek_graph_builder
from medical_kg_sourceprep.extraction.graph_builder.contract import PROJECT_ROOT
from medical_kg_sourceprep.extraction.graph_builder.evaluation.runner import (
    comparison_summary,
    run_typical_cases_experiment,
)


CASE_ID = "TC-08"
CHUNK_ID = "clinical-hematology:chapter-01:0022:0000"
GOLD_PATH = PROJECT_ROOT / "evaluation/typical-cases/typical-cases-v0.1.json"
OUTPUT_ROOT = PROJECT_ROOT / "runtime/evaluations/judge-reextraction/TC-08-v0.1"


if __name__ == "__main__":
    client = create_deepseek_graph_builder()
    with asyncio.Runner() as runner:
        try:
            comparison = runner.run(run_typical_cases_experiment(
                client,
                gold_path=GOLD_PATH,
                output_root=OUTPUT_ROOT,
                case_ids={CASE_ID},
                chunk_output_overrides={CHUNK_ID: OUTPUT_ROOT},
                comparison_filename="comparison.json",
                progress=lambda message: print(message, flush=True),
            ))
        finally:
            runner.run(client.aclose())
    print(json.dumps(
        comparison_summary(comparison), ensure_ascii=False, indent=2, sort_keys=True
    ))
