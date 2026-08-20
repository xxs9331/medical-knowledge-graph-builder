"""使用 DeepSeek 对词典候选做上下文分类；不读取或发送金标答案。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from medical_kg_sourceprep.extraction.graph_builder.contract import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    PROJECT_ROOT,
)


LABELS = {
    "LabPanel", "LabIndicator", "IndicatorState", "ClinicalContext", "Disease", "REJECT"
}
SYSTEM_PROMPT = """你是医学知识图谱词典审核员。只能依据候选词和所给原文上下文分类。
标签：LabPanel=有独立业务含义的检验组合；LabIndicator=可观察、测量或计算的具体指标；
IndicatorState=边界完整的单指标状态；ClinicalContext=影响检验解释的可复用背景；
Disease=明确疾病、诊断或疾病分类；REJECT=普通词、动作、材料、器官、细胞成分、
数值单位、关系/句子片段、语义不完整修饰语或不属于前五类。
重叠不是拒绝理由：长实体和可独立指代的嵌套实体均可保留。并列实体分别保留；
完整组合自身有独立概念时也可保留。不得把相邻词拼成的并列句或共享谓词片段当实体。
缩写必须是完整 token。输出 JSON 对象 {"results":[{"term":原词,"label":标签,"reason":短理由}]}，
结果数量、顺序和 term 必须与输入完全一致。"""


def _classify_batch(client: OpenAI, items: list[dict[str, Any]]) -> list[dict[str, str]]:
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"items": items}, ensure_ascii=False)},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("DeepSeek 返回空内容")
    payload = json.loads(content)
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(items):
        raise ValueError("DeepSeek 结果数量与输入不一致")
    expected_terms = [str(item["term"]) for item in items]
    actual_terms = [str(item.get("term", "")) for item in results if isinstance(item, dict)]
    if actual_terms != expected_terms:
        raise ValueError("DeepSeek 结果词项或顺序与输入不一致")
    normalized: list[dict[str, str]] = []
    for result in results:
        label = str(result.get("label", ""))
        if label not in LABELS:
            raise ValueError(f"DeepSeek 返回未知标签：{label}")
        normalized.append({
            "term": str(result["term"]),
            "label": label,
            "reason": str(result.get("reason", "")),
        })
    return normalized


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="使用 DeepSeek 分类词典候选")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args(argv)

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    import os
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置")

    with args.input.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    pending = [row for row in rows if row["review_status"] == "PENDING"][: args.limit]
    items = [
        {"term": row["term"], "context": row["example"],
         "frequency": int(row["frequency"]), "document_frequency": int(row["document_frequency"])}
        for row in pending
    ]

    classified: list[dict[str, str]] = []
    with httpx.Client(trust_env=False, timeout=120.0) as http_client:
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, http_client=http_client)
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            classified.extend(_classify_batch(client, batch))
            print(json.dumps({"processed": len(classified), "total": len(items)}, ensure_ascii=False))

    by_term = {item["term"]: item for item in classified}
    for row in rows:
        result = by_term.get(row["term"])
        if result is None:
            continue
        row["entity_type"] = "" if result["label"] == "REJECT" else result["label"]
        row["review_status"] = "ASSISTANT_REJECTED" if result["label"] == "REJECT" else "ASSISTANT_LABELED"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    args.audit.write_text(
        json.dumps(
            {
                "schema_version": "terminology-deepseek-classification/v0.1",
                "model": DEEPSEEK_MODEL,
                "gold_answers_exposed": False,
                "items": classified,
                "counts": dict(sorted(Counter(item["label"] for item in classified).items())),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "audit": str(args.audit),
                      "classified": len(classified)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
