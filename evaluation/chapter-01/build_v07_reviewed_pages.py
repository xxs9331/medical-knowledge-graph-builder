"""生成第一章前 8 页逐页审查后的原文实体标注。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT / "source-packages/canonical/source/chapter-01/manifest.json"
EVIDENCE_MANIFEST = ROOT / "source-packages/canonical/evidence/chapter-01/manifest.json"
OUTPUT_PATH = ROOT / "evaluation/chapter-01/chapter-01-entity-mentions-v0.7.json"

# 每个词项均由逐页阅读原文后确定。匹配时同类型优先保留更长词项，避免把
# “缺铁性贫血”同时拆成一个无上下文的“贫血”；不同类型仍允许合法嵌套。
PAGE_TERMS: dict[int, dict[str, tuple[str, ...]]] = {
    0: {
        "LabPanel": ("血液常规检验",),
        "LabIndicator": ("红细胞计数", "红细胞数量", "血红蛋白含量", "血红蛋白"),
        "IndicatorState": ("血红蛋白减少",),
        "Disease": ("血液系统疾病", "贫血"),
    },
    1: {
        "LabIndicator": ("促红细胞生成素", "血红蛋白"),
        "IndicatorState": (
            "促红细胞生成素增高", "红细胞相对增多", "血红蛋白增多",
            "血红蛋白减少", "红细胞增多", "红细胞减少",
        ),
        "ClinicalContext": (
            "3 个月的婴儿至 15 岁", "妊娠中后期的孕妇", "老年人",
            "造血原料相对不足", "血液稀释", "造血功能逐渐减退",
            "骨髓造血功能受损", "造血原料缺乏", "铁缺乏", "叶酸",
            "维生素 \\( B_{12} \\) 缺乏", "红细胞膜的结构异常",
            "血红蛋白结构异常", "急性或慢性失血", "大量血液流失",
            "血液浓缩", "连续剧烈呕吐", "严重腹泻", "大量出汗",
            "发热", "大量水分丢失", "机体缺氧",
        ),
        "Disease": (
            "极重度贫血", "轻度贫血", "中度贫血", "重度贫血",
            "再生障碍性贫血", "白血病", "缺铁性贫血",
            "巨幼细胞贫血", "溶血性贫血", "贫血",
        ),
    },
    2: {
        "LabPanel": ("红细胞平均指数",),
        "LabIndicator": (
            "平均红细胞血红蛋白浓度", "平均红细胞血红蛋白含量",
            "平均红细胞容积", "血红蛋白浓度", "血细胞比容",
            "红细胞压积", "红细胞计数", "MCV", "MCH", "HCT", "PCV",
        ),
        "IndicatorState": (
            "红细胞代偿性增多", "红细胞绝对值增多",
            "血细胞比容增高", "血细胞比容减少",
        ),
        "ClinicalContext": (
            "高原环境", "氧气稀薄", "血液中含氧量减少", "血液浓缩",
            "剧烈呕吐", "严重腹泻", "大量出汗",
        ),
        "Disease": (
            "发绀型先天性心脏病", "后天性肺源性心脏病",
            "真性红细胞增多症", "慢性肺源性心脏病", "肾胚胎瘤",
            "大细胞贫血", "小细胞贫血", "肾癌", "贫血",
        ),
    },
    3: {
        "LabIndicator": (
            "平均红细胞血红蛋白浓度", "红细胞容积分布宽度",
            "变异系数", "标准差", "MCHC", "MCV", "MCH", "RDW-CV",
            "RDW-SD", "RDW",
        ),
        "ClinicalContext": ("维生素\\( B_{12} \\)缺乏", "慢性感染", "炎症"),
        "Disease": (
            "小细胞低色素性贫血", "单纯小细胞性贫血", "大细胞性贫血",
            "正细胞性贫血", "珠蛋白生成障碍性贫血", "急性失血性贫血",
            "急性溶血性贫血", "再生障碍性贫血", "巨幼细胞贫血",
            "缺铁性贫血", "肝病", "贫血",
        ),
    },
    4: {
        "LabPanel": ("白细胞分类计数",),
        "LabIndicator": ("白细胞计数", "MCV", "RDW", "WBC"),
        "IndicatorState": (
            "白细胞计数增多", "白细胞明显增加", "MCV 减小", "MCV 正常",
            "MCV 增大", "RDW 正常", "RDW 增大",
        ),
        "ClinicalContext": (
            "急、慢性感染", "细菌性感染", "广泛的组织损伤", "大面积烧伤",
            "急性大出血",
            "消化道大出血", "急性溶血", "血型不合的输血", "急性中毒", "成人",
        ),
        "Disease": (
            "小细胞不均一性贫血", "正细胞不均一性贫血", "大细胞不均一性贫血",
            "小细胞均一性贫血", "正细胞均一性贫血", "大细胞均一性贫血",
            "慢性再生障碍性贫血", "珠蛋白生成障碍性贫血", "早期缺铁性贫血",
            "急性失血性贫血", "巨幼细胞贫血", "缺铁性贫血", "G6PD 缺乏症",
            "糖尿病酮症酸中毒", "有机磷中毒", "心肌梗死",
            "尿路感染", "食物中毒", "毒蛇咬伤", "慢性疾病", "脑膜炎",
            "扁桃体炎", "猩红热", "败血症", "肝破裂", "脾破裂", "宫外孕",
            "白血病", "肺炎", "痢疾", "丹毒", "尿毒症",
        ),
    },
    5: {
        "LabPanel": ("白细胞分类计数", "WBC-DC"),
        "LabIndicator": (
            "嗜酸性粒细胞", "嗜碱性粒细胞", "中性粒细胞", "淋巴细胞",
            "单核细胞",
        ),
        "IndicatorState": ("白细胞一过性增高", "中性粒细胞增多", "白细胞减少"),
        "ClinicalContext": (
            "长期接触放射线", "磺胺药", "氯霉素", "苯妥英钠", "抗肿瘤药",
            "环磷酰胺", "氨甲蝶呤", "阿糖胞苷", "苯", "铅", "汞", "饱食",
            "情绪激动", "剧烈运动", "高温或严寒", "新生儿",
            "妊娠 5 个月以上", "分娩阵痛",
        ),
        "Disease": ("再生障碍性贫血", "粒细胞减少症", "血液系统疾病"),
    },
    6: {
        "LabIndicator": ("中性粒细胞绝对值", "淋巴细胞百分率", "淋巴细胞比例"),
        "IndicatorState": (
            "淋巴细胞比例相对增高", "中性粒细胞相对偏低", "中性粒细胞减少",
            "淋巴细胞增多", "淋巴细胞减少", "单核细胞增多",
        ),
        "ClinicalContext": (
            "急性感染", "炎症", "广泛的组织损伤或坏死", "严重烧伤",
            "急性大出血", "急性溶血", "急性中毒", "革兰氏阴性杆菌感染",
            "某些病毒感染", "某些原虫感染", "长期接触放射线", "氯霉素",
            "磺胺药", "抗肿瘤药", "苯", "铅", "汞", "慢性炎症",
            "急性传染病的恢复期", "器官移植后的排斥反应",
            "应用肾上腺皮质激素", "抗淋巴细胞球蛋白", "儿童阶段",
        ),
        "Disease": (
            "淋巴细胞性白血病", "亚急性感染性心内膜炎",
            "糖尿病酮症酸中毒", "先天性免疫缺陷病", "系统性红斑狼疮",
            "真性红细胞增多症", "再生障碍性贫血", "巨幼细胞贫血",
            "粒细胞减少症", "粒细胞缺乏症", "粒细胞白血病",
            "骨髓增殖性疾病", "自身免疫性疾病", "脾功能亢进",
            "类脂质沉积病", "病毒感染性疾病", "恶性淋巴瘤", "恶性肿瘤",
            "安眠药中毒", "有机磷中毒", "代谢性中毒", "心肌梗死",
            "尿毒症", "脾破裂", "宫外孕", "伤寒", "副伤寒", "流感",
            "水痘", "疟疾", "黑热病", "艾滋病",
        ),
    },
    7: {
        "LabPanel": ("血细胞三分类",),
        "LabIndicator": (
            "中性分叶核粒细胞", "嗜酸性粒细胞", "嗜碱性粒细胞",
            "杆状核粒细胞", "淋巴细胞", "单核细胞", "血小板计数",
        ),
        "IndicatorState": (
            "嗜酸性粒细胞增多", "嗜酸性粒细胞减少", "嗜碱性粒细胞增多",
        ),
        "ClinicalContext": (
            "急性感染的恢复期", "长期使用肾上腺皮质激素", "恢复期",
        ),
        "Disease": (
            "亚急性感染性心内膜炎", "骨髓增生异常综合征", "嗜酸性粒细胞白血病",
            "嗜碱性粒细胞白血病", "慢性粒细胞白血病", "单核细胞白血病",
            "粒细胞缺乏症", "恶性组织细胞病", "多发性骨髓瘤",
            "血管神经性水肿", "支气管哮喘",
            "活动性肺结核", "剥脱性皮炎", "骨髓纤维化", "钩虫感染",
            "药物和食物过敏", "过敏性疾病", "寄生虫病", "淋巴瘤",
            "荨麻疹", "血清病", "湿疹", "天疱疮", "银屑病", "疟疾", "黑热病",
        ),
    },
}


def _mention_id(page: int, start: int, end: int, entity_type: str) -> str:
    raw = f"{page}\0{start}\0{end}\0{entity_type}".encode()
    return "mention:" + hashlib.sha256(raw).hexdigest()[:20]


def _find_mentions(text: str, terms: dict[str, tuple[str, ...]]) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    occupied: dict[str, list[tuple[int, int]]] = {}
    for entity_type, values in terms.items():
        occupied[entity_type] = []
        for value in sorted(values, key=lambda item: (-len(item), item)):
            start = 0
            while (start := text.find(value, start)) >= 0:
                end = start + len(value)
                if not any(start < old_end and old_start < end for old_start, old_end in occupied[entity_type]):
                    occupied[entity_type].append((start, end))
                    mentions.append({
                        "page_start": start,
                        "page_end": end,
                        "entity_type": entity_type,
                        "exact_quote": value,
                    })
                start = end
    return sorted(mentions, key=lambda item: (item["page_start"], item["page_end"], item["entity_type"]))


if __name__ == "__main__":
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE_MANIFEST.read_text(encoding="utf-8"))
    source_root = SOURCE_MANIFEST.parent
    chunks_by_page: dict[int, list[dict[str, Any]]] = {}
    for chunk in evidence["chunks"]:
        page = int(chunk["chapter_page_index"])
        if page < 8:
            chunks_by_page.setdefault(page, []).append(chunk)

    pages: list[dict[str, Any]] = []
    for page_index in range(8):
        page_meta = source["pages"][page_index]
        text = (source_root / page_meta["cleaned_path"]).read_text(encoding="utf-8")
        page_mentions = _find_mentions(text, PAGE_TERMS[page_index])
        output_mentions: list[dict[str, Any]] = []
        for mention in page_mentions:
            start = int(mention["page_start"])
            end = int(mention["page_end"])
            chunk = next(
                item for item in chunks_by_page[page_index]
                if int(item["cleaned_char_start"]) <= start
                and end <= int(item["cleaned_char_end"])
            )
            chunk_start = int(chunk["cleaned_char_start"])
            output_mentions.append({
                "mention_id": _mention_id(page_index, start, end, str(mention["entity_type"])),
                "chunk_id": chunk["chunk_id"],
                "start": start - chunk_start,
                "end": end - chunk_start,
                "exact_quote": mention["exact_quote"],
                "entity_type": mention["entity_type"],
                "review_status": "ASSISTANT_PAGE_REVIEWED",
            })
        pages.append({
            "page_index": page_index,
            "page_id": page_meta["page_id"],
            "source_path": page_meta["cleaned_path"],
            "closed_world": True,
            "mentions": output_mentions,
        })

    payload = {
        "schema_version": "medical-kg-reviewed-entity-mentions/v0.7",
        "status": "ASSISTANT_REVIEWED_REQUIRES_USER_VALIDATION",
        "annotation_unit": "EXACT_CHARACTER_SPAN_WITH_ENTITY_TYPE",
        "reviewed_page_range": [0, 7],
        "unreviewed_pages_excluded": True,
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "evidence_manifest": str(EVIDENCE_MANIFEST.relative_to(ROOT)),
        "pages": pages,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "pages": len(pages),
        "mentions": sum(len(page["mentions"]) for page in pages),
        "per_page": {str(page["page_index"] + 1): len(page["mentions"]) for page in pages},
    }, ensure_ascii=False, sort_keys=True))
