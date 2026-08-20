"""生成 8 个案例的模型辅助、逐字可回放实体 mention 标注稿。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "source-packages/canonical/evidence/chapter-01/manifest.json"
CASE_PATH = ROOT / "evaluation/chapter-01/chapter-01-scoped-gold-v0.5.json"
OUTPUT_PATH = ROOT / "evaluation/chapter-01/chapter-01-entity-mentions-v0.6.json"
RUN_ROOTS = {
    "baseline_v03": ROOT / "runtime/evaluations/chapter01-scale-l2-v03/20260817-152308-baseline-r01",
    "deepseek_r01": ROOT / "runtime/evaluations/chapter01-layered-v04/20260817-182149-deepseek-r01",
}

DISEASE_PATTERN = re.compile(
    r"(病|症|炎|癌|瘤|白血病|贫血|栓塞|梗死|中毒|感染|综合征|尿毒症|地中海贫血)$"
)
STATE_PATTERN = re.compile(
    r"(增高|增多|升高|加快|延长|降低|减少|减低|减慢|缩短|正常|异常|阳性|阴性|缺乏|过量)$"
)
CONTEXT_PATTERN = re.compile(
    r"(治疗|手术|妊娠|孕妇|婴儿|婴幼儿|老年人|高原|运动|呕吐|腹泻|出血|发热|月经|药|法|试验)$"
)

TYPE_OVERRIDES = {
    "红细胞平均指数": "LabPanel",
    "白细胞分类计数": "LabPanel",
    "血液流变学": "LabPanel",
    "维生素\\( B_{12} \\)缺乏": "ClinicalContext",
    "慢性感染": "ClinicalContext",
    "急性感染": "ClinicalContext",
    "某些病毒感染": "ClinicalContext",
    "革兰氏阴性杆菌感染": "ClinicalContext",
    "某些原虫感染": "ClinicalContext",
    "炎症": "ClinicalContext",
    "急性炎症": "ClinicalContext",
    "慢性炎症": "ClinicalContext",
    "急性溶血": "ClinicalContext",
    "严重烧伤": "ClinicalContext",
    "急性大出血": "ClinicalContext",
    "消化道大出血": "ClinicalContext",
    "血管内血栓形成": "ClinicalContext",
    "血栓形成": "ClinicalContext",
    "血栓": "ClinicalContext",
    "营养不良": "ClinicalContext",
    "胃切除术": "ClinicalContext",
    "流产": "ClinicalContext",
    "早产": "ClinicalContext",
    "出生低体重": "ClinicalContext",
    "缺氧": "ClinicalContext",
    "酸中毒": "ClinicalContext",
    "脱水": "ClinicalContext",
    "创伤": "ClinicalContext",
    "血液浓缩": "ClinicalContext",
    "贫血加重": "ClinicalContext",
    "氯霉素": "ClinicalContext",
    "纤维蛋白原": "LabIndicator",
    "球蛋白": "LabIndicator",
    "白蛋白": "LabIndicator",
    "胆固醇": "LabIndicator",
    "甘油三酯": "LabIndicator",
    "pH值": "LabIndicator",
    "渗透压": "LabIndicator",
}

EXCLUDED_MENTIONS = {
    "骨髓大量制造白细胞以后，制造红细胞的骨髓减少",
    "高原环境,由于氧气稀薄,当地居民",
    "补充水分后可恢复正常",
    "释放进入外周血液",
    "骨髓内白细胞大量增殖",
    "大量幼稚细胞",
    "由骨髓中的成熟巨核细胞产生",
    "直径2~4μm",
    "寿命10天左右",
    "结合血小板数量的变化",
    "常规体检中意义不大",
    "铁从血红蛋白中释放出来",
    "贮存铁的可溶性组织蛋白",
    "肝细胞和巨噬细胞合成",
    "与铁结合",
    "运送至需要铁的组织",
    "胎儿生长发育的需要",
    "大量蛋白质从尿液丢失",
    "每个红细胞内的血红蛋白量相对增加",
    "体内缺乏细胞分裂和细胞核形成必需的维生素",
    "红细胞膜表面的唾液酸带负电荷",
    "红细胞的数量、形状和大小等变化",
    "试图通过血黏度参数来做脑卒中预报是不科学的",
    "不能凭一次血黏度升高就认为患有心脑血管疾病",
    "喝水会稀释血液",
    "不同地区人们的生活习惯不同",
    "纤维蛋白降解产物(FDP)中的一个片段",
    "体内血栓形成和继发性纤溶亢进的特异性标志物",
    "形成血栓的危险性愈高",
    "流动性愈小",
    "随剪切率增高而降低",
    "随剪切率的降低而增高",
    "使血黏度降低",
    "乳胶凝集法: 阴性",
    "免疫比浊定量分析：<0.5mg/L",
    "MCV、RDW 均正常",
    "MCV、RDW 均增大",
    "蛔虫",
    "血吸虫",
    "肺吸虫",
    "丝虫",
    "包囊虫",
}


def _stable_id(chunk_id: str, start: int, end: int, entity_type: str) -> str:
    value = f"{chunk_id}\0{start}\0{end}\0{entity_type}".encode("utf-8")
    return "mention:" + hashlib.sha256(value).hexdigest()[:20]


def _choose_type(mention: str, candidates: set[str]) -> tuple[str, str]:
    """只裁决冲突类型；无冲突候选保持原抽取类型。"""
    if mention in TYPE_OVERRIDES:
        return TYPE_OVERRIDES[mention], "ASSISTANT_SCHEMA_ADJUDICATION"
    if len(candidates) == 1:
        return next(iter(candidates)), "MODEL_CONSENSUS"
    if "IndicatorState" in candidates and STATE_PATTERN.search(mention):
        return "IndicatorState", "SCHEMA_POLICY_CONFLICT_RESOLUTION"
    if "Disease" in candidates and DISEASE_PATTERN.search(mention):
        return "Disease", "SCHEMA_POLICY_CONFLICT_RESOLUTION"
    if "ClinicalContext" in candidates and CONTEXT_PATTERN.search(mention):
        return "ClinicalContext", "SCHEMA_POLICY_CONFLICT_RESOLUTION"
    if "LabIndicator" in candidates and not mention.endswith("法"):
        return "LabIndicator", "SCHEMA_POLICY_CONFLICT_RESOLUTION"
    priority = ("Disease", "ClinicalContext", "IndicatorState", "LabIndicator", "LabPanel")
    return next(item for item in priority if item in candidates), "SCHEMA_POLICY_CONFLICT_RESOLUTION"


if __name__ == "__main__":
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    package_root = MANIFEST_PATH.parent
    chunks = {item["chunk_id"]: item for item in manifest["chunks"]}
    texts = {
        chunk_id: (package_root / item["chunk_path"]).read_text(encoding="utf-8")
        for chunk_id, item in chunks.items()
    }
    dataset = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    case_by_chunk = {
        chunk_id: case["case_id"]
        for case in dataset["cases"] for chunk_id in case["chunk_ids"]
    }
    candidates: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for case in dataset["cases"]:
        evidence_by_id = {
            item["evidence_unit_id"]: item for item in case["evidence_units"]
        }
        canonical_by_id = {
            item["canonical_id"]: item for item in case["canonical_entities"]
        }
        for link in case["mention_to_canonical_links"]:
            evidence = evidence_by_id[link["evidence_unit_id"]]
            if not evidence["mention_eligible"]:
                continue
            canonical = canonical_by_id[link["canonical_id"]]
            identity = (
                str(evidence["chunk_id"]), int(evidence["start"]), int(evidence["end"]),
                str(evidence["exact_quote"]),
            )
            item = candidates.setdefault(identity, {"types": set(), "sources": set()})
            item["types"].add(str(canonical["entity_type"]))
            item["sources"].add("v05_reference")

    for source_name, run_root in RUN_ROOTS.items():
        for graph_path in sorted(run_root.glob("chunks/*/candidate-graph/graph.json")):
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            for node in graph.get("nodes", []):
                if node.get("entity_type") == "RuleDefinition":
                    continue
                source_ref = node.get("source_ref", {})
                chunk_id = source_ref.get("chunk_id")
                start = source_ref.get("mention_char_start", source_ref.get("char_start"))
                end = source_ref.get("mention_char_end", source_ref.get("char_end"))
                mention = node.get("mention")
                if not (
                    isinstance(chunk_id, str) and isinstance(start, int)
                    and isinstance(end, int) and isinstance(mention, str)
                ):
                    continue
                # 原文 mention 必须逐字出现；表格推导状态留在 canonical/派生层。
                if texts[chunk_id][start:end] != mention or len(mention.strip()) < 2:
                    continue
                identity = (chunk_id, start, end, mention)
                item = candidates.setdefault(identity, {"types": set(), "sources": set()})
                item["types"].add(str(node["entity_type"]))
                item["sources"].add(source_name)

    cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conflict_count = 0
    for (chunk_id, start, end, mention), item in sorted(candidates.items()):
        if mention in EXCLUDED_MENTIONS:
            continue
        candidate_types = set(item["types"])
        entity_type, adjudication = _choose_type(mention, candidate_types)
        if len(candidate_types) > 1:
            conflict_count += 1
        cases[case_by_chunk[chunk_id]].append({
            "mention_id": _stable_id(chunk_id, start, end, entity_type),
            "chunk_id": chunk_id,
            "start": start,
            "end": end,
            "exact_quote": mention,
            "entity_type": entity_type,
            "candidate_types": sorted(candidate_types),
            "candidate_sources": sorted(item["sources"]),
            "adjudication": adjudication,
            "review_status": "ASSISTANT_ANNOTATED",
        })

    ordered_cases = [
        {
            "case_id": case["case_id"],
            "title": case["title"],
            "chunk_ids": case["chunk_ids"],
            "mentions": cases[case["case_id"]],
        }
        for case in dataset["cases"]
    ]
    mention_count = sum(len(case["mentions"]) for case in ordered_cases)
    payload = {
        "schema_version": "medical-kg-exhaustive-entity-mentions/v0.6",
        "status": "ASSISTANT_ANNOTATED_REQUIRES_USER_VALIDATION",
        "annotation_method": "TWO_RUN_CANDIDATE_UNION_WITH_SCHEMA_POLICY_ADJUDICATION",
        "scope_contract": {
            "closed_world": True,
            "unit": "EXACT_CHARACTER_SPAN_WITH_ENTITY_TYPE",
            "nested_mentions": "ALLOWED",
            "derived_nonverbatim_entities": "EXCLUDED",
            "candidate_output_used_for_discovery": True,
            "same_run_evaluation_allowed": False,
        },
        "source_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "cases": ordered_cases,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "cases": len(payload["cases"]),
        "mentions": mention_count,
        "type_conflicts_adjudicated": conflict_count,
    }, ensure_ascii=False, sort_keys=True))
