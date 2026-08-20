"""生成第一章全章关系语义标注稿，并用 v0.8 规范实体作为端点。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ENTITY_PATH = ROOT / "evaluation/chapter-01/chapter-01-canonical-entities-v0.8.json"
MENTION_PATH = ROOT / "evaluation/chapter-01/chapter-01-entity-mentions-v0.6.json"
MANUAL_GRAPH_PATH = ROOT / "evaluation/chapter-01/chapter-01-graph-test-set-v0.3.json"
MANUAL_AUDIT_PATH = ROOT / "evaluation/chapter-01/chapter-01-evidence-audit-v0.3.json"
OUTPUT_PATH = ROOT / "evaluation/chapter-01/chapter-01-relationship-gold-v1.0.json"


def _stable_id(*parts: str) -> str:
    return "gold-relation:" + hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]


def build_gold() -> dict[str, Any]:
    catalog = json.loads(ENTITY_PATH.read_text(encoding="utf-8"))
    mention_set = json.loads(MENTION_PATH.read_text(encoding="utf-8"))
    manual_graph = json.loads(MANUAL_GRAPH_PATH.read_text(encoding="utf-8"))
    manual_audit = json.loads(MANUAL_AUDIT_PATH.read_text(encoding="utf-8"))
    entity_by_name: dict[str, list[dict[str, Any]]] = {}
    for entity in catalog["canonical_entities"]:
        for name in (entity["canonical_name"], *entity.get("aliases", [])):
            entity_by_name.setdefault(name, []).append(entity)
    manual_types_by_name: dict[str, set[str]] = {}
    for case in manual_graph["cases"]:
        for entity_type, name in case["entities"]:
            manual_types_by_name.setdefault(name, set()).add(entity_type)

    sections = {
        case["case_id"]: {
            "case_id": case["case_id"],
            "title": case["title"],
            "chunk_ids": case["chunk_ids"],
            "relationships": [],
        }
        for case in mention_set["cases"]
    }
    seen: set[tuple[str, str, str]] = set()

    audit_evidence = {
        (case["case_id"], tuple(item["target"])): [
            evidence["chunk_id"] for evidence in item.get("evidence", [])
        ]
        for case in manual_audit["cases"]
        for item in case["items"]
        if item["kind"] == "relationship"
    }

    def resolve(name: str, entity_type: str | None = None) -> dict[str, Any]:
        all_matches = list({
            item["canonical_id"]: item for item in entity_by_name.get(name, [])
        }.values())
        matches = all_matches
        if entity_type is not None:
            matches = [item for item in matches if item["entity_type"] == entity_type]
        elif len(matches) > 1 and len(manual_types_by_name.get(name, set())) == 1:
            manual_type = next(iter(manual_types_by_name[name]))
            matches = [item for item in matches if item["entity_type"] == manual_type]
        active_matches = [item for item in matches if item.get("mention_ids")]
        if len(active_matches) == 1:
            return active_matches[0]
        # v0.3 手工图曾写入无 mention 的历史占位实体。关系金标应回写到当前
        # 目录中同名且有原文锚点的规范实体，即使旧图记录了过时类型。
        active_all_matches = [item for item in all_matches if item.get("mention_ids")]
        if len(matches) == 1 and not matches[0].get("mention_ids") and len(active_all_matches) == 1:
            return active_all_matches[0]
        if len(matches) != 1:
            raise ValueError(f"实体端点不能唯一解析: {entity_type}/{name}: {len(matches)}")
        return matches[0]

    def add(
        section_id: str,
        source: str,
        relation_type: str,
        target: str,
        chunk_id: str,
        *,
        source_type: str | None = None,
        target_type: str | None = None,
        rationale: str,
    ) -> None:
        source_entity = resolve(source, source_type)
        target_entity = resolve(target, target_type)
        identity = (
            source_entity["canonical_id"], relation_type, target_entity["canonical_id"]
        )
        if identity in seen:
            return
        if chunk_id not in sections[section_id]["chunk_ids"]:
            raise ValueError(f"证据 chunk 不属于 {section_id}: {chunk_id}")
        seen.add(identity)
        sections[section_id]["relationships"].append({
            "gold_relation_id": _stable_id(*identity),
            "source_canonical_id": source_entity["canonical_id"],
            "source_entity_type": source_entity["entity_type"],
            "source_canonical_name": source_entity["canonical_name"],
            "relation_type": relation_type,
            "target_canonical_id": target_entity["canonical_id"],
            "target_entity_type": target_entity["entity_type"],
            "target_canonical_name": target_entity["canonical_name"],
            "evidence_chunk_ids": [chunk_id],
            "annotation_rationale": rationale,
            "review_status": "AUTOMATED_AUGMENTATION_OF_MANUAL_GRAPH",
        })

    def add_targets(
        section_id: str,
        source: str,
        relation_type: str,
        targets: list[str],
        chunk_id: str,
        *,
        source_type: str | None = None,
        target_type: str | None = None,
        rationale: str,
    ) -> None:
        for target in targets:
            add(
                section_id, source, relation_type, target, chunk_id,
                source_type=source_type,
                target_type=target_type,
                rationale=rationale,
            )

    # The merged v0.3 graph is the semantic authority. Import every manually
    # annotated ordinary relation before adding later full-chapter annotations.
    manual_relationship_identities: set[tuple[str, str, str]] = set()
    for case in manual_graph["cases"]:
        entity_types: dict[str, set[str]] = {}
        for entity_type, name in case["entities"]:
            entity_types.setdefault(name, set()).add(entity_type)
        for source, relation_type, target in case["relationships"]:
            source_types = entity_types.get(source, set())
            target_types = entity_types.get(target, set())
            if len(source_types) != 1 or len(target_types) != 1:
                raise ValueError(
                    f"人工关系端点类型不能唯一解析: {case['case_id']} "
                    f"{source_types}/{source} -> {target_types}/{target}"
                )
            source_entity = resolve(source, next(iter(source_types)))
            target_entity = resolve(target, next(iter(target_types)))
            identity = (
                source_entity["canonical_id"], relation_type,
                target_entity["canonical_id"],
            )
            manual_relationship_identities.add(identity)
            if identity in seen:
                continue
            evidence_chunk_ids = audit_evidence.get(
                (case["case_id"], (source, relation_type, target)), []
            )
            if not set(evidence_chunk_ids) <= set(sections[case["case_id"]]["chunk_ids"]):
                raise ValueError(f"人工关系证据超出案例范围: {case['case_id']} {identity}")
            seen.add(identity)
            sections[case["case_id"]]["relationships"].append({
                "gold_relation_id": _stable_id(*identity),
                "source_canonical_id": source_entity["canonical_id"],
                "source_entity_type": source_entity["entity_type"],
                "source_canonical_name": source_entity["canonical_name"],
                "relation_type": relation_type,
                "target_canonical_id": target_entity["canonical_id"],
                "target_entity_type": target_entity["entity_type"],
                "target_canonical_name": target_entity["canonical_name"],
                "evidence_chunk_ids": evidence_chunk_ids,
                "annotation_rationale": "机械继承v0.3统一人工图关系",
                "review_status": "MANUAL_GRAPH_GOLD",
                "provenance": "chapter-01-graph-test-set-v0.3.json",
            })

    # CH01-01：红细胞、血红蛋白、血细胞比容和红细胞平均指数。
    c01 = "clinical-hematology:chapter-01:0000:0001"
    c10 = "clinical-hematology:chapter-01:0001:0000"
    c11 = "clinical-hematology:chapter-01:0001:0001"
    c20 = "clinical-hematology:chapter-01:0002:0000"
    c21 = "clinical-hematology:chapter-01:0002:0001"
    add_targets("CH01-01", "血液常规检验", "HAS_METRIC", [
        "红细胞计数", "白细胞", "血小板", "血红蛋白", "血小板参数",
    ], c01, rationale="原文明确列出血液常规检验包含的参数")
    add_targets("CH01-01", "红细胞平均指数", "HAS_METRIC", [
        "平均红细胞容积", "平均红细胞血红蛋白含量", "平均红细胞血红蛋白浓度",
    ], c21, rationale="原文明确列出三种红细胞平均指数")
    add_targets("CH01-01", "红细胞计数", "HAS_STATE", [
        "红细胞减少", "红细胞增多", "红细胞相对增多", "红细胞代偿性增多",
        "红细胞绝对值增多",
    ], c10, rationale="异常结果解读明确给出红细胞数量状态")
    add_targets("CH01-01", "血红蛋白", "HAS_STATE", [
        "血红蛋白减少", "血红蛋白增多",
    ], c10, rationale="异常结果解读明确给出血红蛋白状态")
    add("CH01-01", "血液含氧量", "HAS_STATE", "血液含氧量减少", c20,
        rationale="原文明示血液含氧量减少")
    add("CH01-01", "促红细胞生成素", "HAS_STATE", "促红细胞生成素增高", c11,
        target_type="IndicatorState", rationale="原文明示促红细胞生成素增高")
    add_targets("CH01-01", "血细胞比容", "HAS_STATE", [
        "血细胞比容增高", "血细胞比容减少",
    ], c20, rationale="血细胞比容异常结果解读的两个状态")
    for context in ["造血原料相对不足", "血液稀释", "造血功能逐渐减退"]:
        add_targets("CH01-01", context, "CAUSES", ["红细胞减少", "血红蛋白减少"], c10,
                    rationale="原文明示该因素使红细胞和血红蛋白减少")
    add_targets("CH01-01", "造血功能受损", "CAUSES", [
        "红细胞减少", "再生障碍性贫血",
    ], c10, rationale="选择原文中最小且完整的原因端点，表达直接因果")
    add("CH01-01", "各种白血病", "CAUSES", "骨髓减少", c10,
        rationale="原文明示白血病使制造红细胞的骨髓减少")
    add("CH01-01", "骨髓减少", "CAUSES", "红细胞减少", c10,
        rationale="原文明示制造红细胞的骨髓减少导致红细胞减少")
    add("CH01-01", "铁缺乏", "CAUSES", "缺铁性贫血", c10,
        rationale="原文使用“铁缺乏引起”")
    add("CH01-01", "红细胞膜结构异常", "CAUSES", "溶血性贫血", c10,
        rationale="原文明示膜结构异常导致红细胞破坏并引起贫血")
    add("CH01-01", "红细胞内血红蛋白结构异常", "CAUSES", "溶血性贫血", c10,
        rationale="原文明示血红蛋白结构异常导致红细胞破坏并引起贫血")
    add_targets("CH01-01", "急性失血", "CAUSES", ["贫血", "红细胞减少", "血红蛋白减少"], c10,
                rationale="原文明示急性失血导致贫血和红细胞/血红蛋白减少")
    add_targets("CH01-01", "慢性失血", "CAUSES", ["贫血", "红细胞减少", "血红蛋白减少"], c10,
                rationale="原文明示慢性失血导致贫血和红细胞/血红蛋白减少")
    for context in ["连续剧烈呕吐", "严重腹泻", "大量出汗", "发热"]:
        add("CH01-01", context, "CAUSES", "大量水分丢失", c11,
            rationale="原文将该因素列为大量水分丢失的原因")
    add("CH01-01", "大量水分丢失", "CAUSES", "血液浓缩", c11,
        rationale="原文明示水分丢失使血液浓缩")
    add("CH01-01", "血液浓缩", "CAUSES", "红细胞相对增多", c11,
        rationale="原文明示血液浓缩使红细胞相对增多")
    add("CH01-01", "血液浓缩", "CAUSES", "血红蛋白增多", c11,
        rationale="红细胞和血红蛋白增多条目把血液浓缩列为相对性增多原因")
    add("CH01-01", "机体缺氧", "CAUSES", "促红细胞生成素增高", c11,
        rationale="原文使用“由于机体缺氧导致”")
    # 该句具有因果语义，但当前 Schema 不允许 IndicatorState 作为 CAUSES 起点；
    # 不降级成 ASSOCIATED_WITH，留待 Schema 扩展后再纳入金标。
    add("CH01-01", "红细胞生成增多", "CAUSES", "红细胞增多", c11,
        rationale="制造红细胞增多直接产生红细胞绝对性增多")
    add("CH01-01", "氧气稀薄", "CAUSES", "红细胞代偿性增多", c20,
        rationale="原文明示高原氧气稀薄引起代偿性增多")
    for disease in ["发绀型先天性心脏病", "后天性肺源性心脏病"]:
        add("CH01-01", disease, "CAUSES", "血液含氧量减少", c20,
            rationale="原文明示这些疾病伴随的血液含氧量减少")
    add("CH01-01", "血液浓缩", "CAUSES", "血细胞比容增高", c20,
        rationale="血细胞比容增高条目明确归因于血液浓缩")
    # 两条状态到状态的因果（血液含氧量减少 -> 红细胞代偿性增多、
    # 红细胞绝对值增多 -> 血细胞比容增高）超出当前 CAUSES 端点合同，暂不纳入。
    for context in ["剧烈呕吐", "严重腹泻", "大量出汗"]:
        add("CH01-01", context, "CAUSES", "血液浓缩", c20,
            rationale="血细胞比容条目直接把该因素列为血液浓缩原因")
    add("CH01-01", "急性失血", "CAUSES", "急性失血性贫血", c10,
        rationale="规范疾病名直接表达急性失血导致该型贫血")
    add_targets("CH01-01", "红细胞减少", "INDICATES", [
        "贫血", "再生障碍性贫血", "白血病", "溶血性贫血",
    ], c10, rationale="异常状态及其疾病解释由原文直接对应")
    add_targets("CH01-01", "血红蛋白减少", "INDICATES", [
        "贫血", "再生障碍性贫血", "白血病", "溶血性贫血",
    ], c10, rationale="异常状态及其疾病解释由原文直接对应")
    add_targets("CH01-01", "红细胞代偿性增多", "INDICATES", [
        "发绀型先天性心脏病", "后天性肺源性心脏病",
    ], c20, rationale="原文在病理性代偿增多下列出相应疾病")
    add_targets("CH01-01", "红细胞增多", "INDICATES", [
        "真性红细胞增多症", "肾癌", "肾胚胎瘤",
    ], c20, rationale="原文使用“可见于”列出疾病")
    add_targets("CH01-01", "血细胞比容增高", "INDICATES", [
        "真性红细胞增多症", "慢性肺源性心脏病",
    ], c20, rationale="血细胞比容增高条目列出疾病")
    add_targets("CH01-01", "血细胞比容减少", "INDICATES", [
        "贫血", "大细胞贫血", "小细胞贫血",
    ], c20, rationale="血细胞比容减少条目明确见于不同类型贫血")
    for child, parent in [
        ("发绀型先天性心脏病", "先天性心脏病"),
        ("后天性肺源性心脏病", "肺源性心脏病"),
        ("慢性肺源性心脏病", "肺源性心脏病"),
        ("大细胞贫血", "贫血"), ("小细胞贫血", "贫血"),
        ("缺铁性贫血", "贫血"), ("巨幼细胞贫血", "贫血"),
        ("溶血性贫血", "贫血"), ("再生障碍性贫血", "贫血"),
    ]:
        add("CH01-01", child, "IS_A", parent, c20 if "肺源" in child or "细胞贫血" in child else c10,
            rationale="原文名称或分类结构明确表达疾病上下位")

    # CH01-02：红细胞平均指数、RDW 和贫血形态分类。
    c30 = "clinical-hematology:chapter-01:0003:0000"
    c40 = "clinical-hematology:chapter-01:0004:0000"
    add_targets("CH01-02", "红细胞平均指数", "HAS_METRIC", [
        "平均红细胞容积", "平均红细胞血红蛋白含量", "平均红细胞血红蛋白浓度",
    ], c30, rationale="原文明确三种平均指数")
    add_targets("CH01-02", "平均红细胞容积", "HAS_STATE", [
        "MCV减小", "MCV正常", "MCV增大",
    ], c40, rationale="形态分类条目明确给出 MCV 状态")
    add_targets("CH01-02", "红细胞容积分布宽度", "HAS_STATE", [
        "RDW正常", "RDW增大",
    ], c40, rationale="形态分类条目明确给出 RDW 状态")
    add("CH01-02", "维生素B12缺乏", "CAUSES", "巨幼细胞贫血", c30,
        rationale="表格病因栏明确使用“缺乏引起”")
    add_targets("CH01-02", "平均红细胞容积", "INDICATES", ["贫血形态学分类"], c30,
                rationale="原文明示该平均指数用于贫血形态学分类诊断")
    add_targets("CH01-02", "平均红细胞血红蛋白含量", "INDICATES", ["贫血形态学分类"], c30,
                rationale="原文明示该平均指数用于贫血形态学分类诊断")
    add_targets("CH01-02", "平均红细胞血红蛋白浓度", "INDICATES", ["贫血形态学分类"], c30,
                rationale="原文明示该平均指数用于贫血形态学分类诊断")
    for cause in ["慢性感染", "炎症", "肝病"]:
        add("CH01-02", cause, "CAUSES", "单纯小细胞性贫血", c30,
            rationale="表格病因栏将该因素列为单纯小细胞性贫血病因")
    for child, parent, chunk in [
        ("巨幼细胞贫血", "大细胞不均一性贫血", c40),
        ("缺铁性贫血", "小细胞低色素性贫血", c30),
        ("珠蛋白生成障碍性贫血", "小细胞低色素性贫血", c30),
        ("珠蛋白生成障碍性贫血", "小细胞均一性贫血", c40),
        ("缺铁性贫血", "小细胞不均一性贫血", c40),
        ("急性失血性贫血", "正细胞均一性贫血", c40),
        ("早期缺铁性贫血", "正细胞不均一性贫血", c40),
        ("G6PD缺乏症", "正细胞不均一性贫血", c40),
        ("慢性再生障碍性贫血", "大细胞均一性贫血", c40),
    ]:
        add("CH01-02", child, "IS_A", parent, chunk,
            rationale="原文以“如”把具体疾病列为相应形态分类实例")

    # CH01-03：白细胞计数和五分类异常结果解读。
    c41 = "clinical-hematology:chapter-01:0004:0001"
    c50 = "clinical-hematology:chapter-01:0005:0000"
    c51 = "clinical-hematology:chapter-01:0005:0001"
    c60 = "clinical-hematology:chapter-01:0006:0000"
    c61 = "clinical-hematology:chapter-01:0006:0001"
    c70 = "clinical-hematology:chapter-01:0007:0000"
    c71 = "clinical-hematology:chapter-01:0007:0001"
    # 五分类的报告端点由人工图中的绝对值指标及表格补充件中的百分数/绝对值
    # 指标承载；细胞类别本身不是可测指标，不能作为 HAS_METRIC 的替代端点。
    add_targets("CH01-03", "血细胞三分类", "HAS_METRIC", [
        "小细胞区", "中间细胞区", "大细胞区",
    ], c71, rationale="三分类定义明确列出三个细胞区域")
    add_targets("CH01-03", "白细胞计数", "HAS_STATE", [
        "白细胞计数增多", "白细胞计数减少",
    ], c41, rationale="白细胞计数异常结果的两个状态")
    add_targets("CH01-03", "中性粒细胞", "HAS_STATE", [
        "中性粒细胞增多", "中性粒细胞减少", "中性粒细胞异常增生性增多",
    ], c60, rationale="中性粒细胞异常结果的状态")
    add_targets("CH01-03", "淋巴细胞", "HAS_STATE", [
        "淋巴细胞增多", "淋巴细胞减少",
    ], c61, rationale="淋巴细胞异常结果的状态")
    add("CH01-03", "淋巴细胞比例", "HAS_STATE", "淋巴细胞比例相对增高", c61,
        rationale="原文明示淋巴细胞比例相对增高")
    add("CH01-03", "单核细胞", "HAS_STATE", "单核细胞增多", c61,
        rationale="原文明示单核细胞增多")
    add_targets("CH01-03", "嗜酸性粒细胞", "HAS_STATE", [
        "嗜酸性粒细胞增多", "嗜酸性粒细胞减少",
    ], c70, rationale="嗜酸性粒细胞异常结果的状态")
    add("CH01-03", "嗜碱性粒细胞", "HAS_STATE", "嗜碱性粒细胞增多", c70,
        rationale="原文明示嗜碱性粒细胞增多")
    add_targets("CH01-03", "白细胞计数增多", "INDICATES", [
        "急性感染", "慢性感染", "肺炎", "脑膜炎", "扁桃体炎", "痢疾", "猩红热",
        "败血症", "尿路感染", "丹毒", "广泛的组织损伤", "大面积烧伤", "心肌梗死",
        "急性大出血", "肝破裂", "脾破裂", "消化道大出血", "宫外孕", "急性溶血",
        "急性中毒", "有机磷中毒", "糖尿病酮症酸中毒", "尿毒症", "食物中毒",
        "毒蛇咬伤", "白血病",
    ], c41, rationale="白细胞计数增多的异常结果条目直接列出相应情况")
    add_targets("CH01-03", "白细胞计数减少", "INDICATES", [
        "长期接触放射线", "应用某些药物", "有毒有害化学物质", "再生障碍性贫血",
        "粒细胞减少症",
    ], c50, rationale="白细胞计数减少的异常结果条目直接列出相应情况")
    add("CH01-03", "长期接触放射线", "CAUSES", "白细胞计数减少", c50,
        rationale="原文明示放射线损伤骨髓并引起白细胞减少")
    for cause in [
        "磺胺药", "氯霉素", "苯妥英钠", "环磷酰胺", "氨甲蝶呤", "阿糖胞苷",
        "有毒有害化学物质",
    ]:
        add("CH01-03", cause, "CAUSES", "白细胞计数减少", c50,
            rationale="原文将该药物或化学暴露列为白细胞计数减少因素")
    for cause in ["应用某些药物", "抗肿瘤药"]:
        add("CH01-03", cause, "CAUSES", "白细胞计数减少", c50,
            rationale="原文把该药物类别直接列在白细胞计数减少条目下")
    add("CH01-03", "白血病", "CAUSES", "白细胞明显增加", c41,
        rationale="原文明示白血病使白细胞明显增加")
    add("CH01-03", "白细胞计数", "HAS_STATE", "白细胞明显增加", c41,
        rationale="白细胞明显增加是白细胞计数的明确异常状态")
    add_targets("CH01-03", "中性粒细胞增多", "INDICATES", [
        "急性感染", "炎症", "广泛的组织损伤", "广泛的组织坏死", "严重烧伤",
        "心肌梗死", "急性大出血", "脾破裂", "宫外孕", "急性溶血", "急性中毒",
        "安眠药中毒", "有机磷中毒", "尿毒症", "糖尿病酮症酸中毒", "恶性肿瘤",
    ], c60, rationale="中性粒细胞反应性增多条目直接列出相应情况")
    add_targets("CH01-03", "中性粒细胞异常增生性增多", "INDICATES", [
        "粒细胞白血病", "骨髓增殖性疾病", "真性红细胞增多症",
    ], c60, rationale="异常增生性增多条目直接列出疾病")
    add_targets("CH01-03", "中性粒细胞减少", "INDICATES", [
        "感染性疾病", "革兰氏阴性杆菌感染", "某些病毒感染", "某些原虫感染",
        "伤寒", "副伤寒", "流感", "水痘", "疟疾", "黑热病",
        "再生障碍性贫血", "巨幼细胞贫血", "粒细胞减少症", "慢性理化损伤",
        "长期接触放射线", "化学药物", "化学药物暴露", "氯霉素", "磺胺药", "抗肿瘤药",
        "某些化学物质", "自身免疫性疾病", "系统性红斑狼疮",
        "单核巨噬细胞系统功能亢进", "脾功能亢进", "类脂质沉积病",
    ], c60, rationale="中性粒细胞减少条目直接列出相应情况")
    add_targets("CH01-03", "淋巴细胞增多", "INDICATES", [
        "病毒感染性疾病", "急性淋巴细胞性白血病", "慢性淋巴细胞性白血病",
        "恶性淋巴瘤", "慢性炎症", "急性传染病恢复期", "器官移植排斥反应",
        "再生障碍性贫血", "粒细胞缺乏症",
    ], c61, rationale="淋巴细胞病理性增多条目直接列出相应情况")
    add_targets("CH01-03", "淋巴细胞比例相对增高", "INDICATES", [
        "再生障碍性贫血", "粒细胞缺乏症",
    ], c61, rationale="原文明示两种疾病时淋巴细胞比例相对增高")
    add_targets("CH01-03", "淋巴细胞减少", "INDICATES", [
        "长期接触放射线", "应用肾上腺皮质激素", "先天性免疫缺陷病", "艾滋病",
    ], c61, rationale="淋巴细胞减少条目直接列出相应情况")
    add_targets("CH01-03", "单核细胞增多", "INDICATES", [
        "儿童阶段", "疟疾", "黑热病", "亚急性感染性心内膜炎", "活动性肺结核",
        "急性感染恢复期", "单核细胞白血病", "粒细胞缺乏症恢复期",
        "恶性组织细胞病", "淋巴瘤", "骨髓增生异常综合征",
    ], c70, rationale="单核细胞增多条目直接列出相应情况")
    add_targets("CH01-03", "嗜酸性粒细胞增多", "INDICATES", [
        "过敏性疾病", "支气管哮喘", "荨麻疹", "药物过敏", "食物过敏",
        "血管神经性水肿", "血清病", "寄生虫病", "钩虫感染", "湿疹",
        "剥脱性皮炎", "天疱疮", "银屑病", "嗜酸性粒细胞白血病",
        "慢性粒细胞白血病", "多发性骨髓瘤",
    ], c70, rationale="嗜酸性粒细胞增多条目直接列出相应情况")
    add("CH01-03", "嗜酸性粒细胞减少", "INDICATES", "长期使用肾上腺皮质激素", c70,
        rationale="嗜酸性粒细胞减少条目明确见于长期激素使用")
    add_targets("CH01-03", "嗜碱性粒细胞增多", "INDICATES", [
        "慢性粒细胞白血病", "嗜碱性粒细胞白血病", "骨髓纤维化", "过敏性疾病",
    ], c70, rationale="嗜碱性粒细胞增多条目直接列出相应疾病")
    add_targets("CH01-03", "剧烈运动", "CAUSES", ["白细胞一过性增高"], c51,
                rationale="原文明示剧烈运动可使白细胞一过性增高")
    for cause in ["饱食", "情绪激动", "高温", "严寒", "新生儿", "妊娠5个月以上", "分娩阵痛"]:
        add("CH01-03", cause, "CAUSES", "白细胞一过性增高", c51,
            rationale="原文明示该生理因素可使白细胞一过性增高")
    early_infections = {
        "肺炎", "脑膜炎", "扁桃体炎", "痢疾", "猩红热", "败血症", "尿路感染", "丹毒",
    }
    later_infections = {"伤寒", "副伤寒", "流感", "水痘", "疟疾", "黑热病"}
    allergy_children = {
        "支气管哮喘", "荨麻疹", "药物过敏", "食物过敏", "血管神经性水肿", "血清病",
    }
    leukemia_children = {
        "急性淋巴细胞性白血病", "慢性淋巴细胞性白血病", "慢性粒细胞白血病",
        "单核细胞白血病", "嗜酸性粒细胞白血病", "嗜碱性粒细胞白血病",
    }
    for child, parent in [
        ("肺炎", "感染性疾病"), ("脑膜炎", "感染性疾病"),
        ("扁桃体炎", "感染性疾病"), ("痢疾", "感染性疾病"),
        ("猩红热", "感染性疾病"), ("败血症", "感染性疾病"),
        ("尿路感染", "感染性疾病"), ("丹毒", "感染性疾病"),
        ("伤寒", "感染性疾病"), ("副伤寒", "感染性疾病"),
        ("流感", "感染性疾病"), ("水痘", "感染性疾病"),
        ("疟疾", "感染性疾病"), ("黑热病", "感染性疾病"),
        ("支气管哮喘", "过敏性疾病"), ("荨麻疹", "过敏性疾病"),
        ("药物过敏", "过敏性疾病"), ("食物过敏", "过敏性疾病"),
        ("血管神经性水肿", "过敏性疾病"), ("血清病", "过敏性疾病"),
        ("急性淋巴细胞性白血病", "白血病"),
        ("慢性淋巴细胞性白血病", "白血病"),
        ("慢性粒细胞白血病", "白血病"),
        ("单核细胞白血病", "白血病"),
        ("嗜酸性粒细胞白血病", "白血病"),
        ("嗜碱性粒细胞白血病", "白血病"),
        ("恶性淋巴瘤", "淋巴瘤"),
    ]:
        evidence_chunk = (
            c41 if child in early_infections
            else c60 if child in later_infections
            else c70 if child in allergy_children | leukemia_children | {"恶性淋巴瘤"}
            else c70
        )
        add("CH01-03", child, "IS_A", parent, evidence_chunk,
            rationale="原文分类标题和“如”结构明确表达具体疾病属于父类")
    anemia_evidence = {
        "急性失血性贫血": c30, "急性溶血性贫血": c30,
        "再生障碍性贫血": c10, "巨幼细胞贫血": c30,
        "缺铁性贫血": c30, "珠蛋白生成障碍性贫血": c30,
        "慢性再生障碍性贫血": c40, "早期缺铁性贫血": c40,
    }
    for child, chunk in anemia_evidence.items():
        section = "CH01-02" if child != "再生障碍性贫血" else "CH01-01"
        add(section, child, "IS_A", "贫血", chunk,
            rationale="疾病名称及原文贫血分类明确表达其属于贫血")

    # CH01-04：血小板计数和平均血小板体积。
    c80 = "clinical-hematology:chapter-01:0008:0000"
    c81 = "clinical-hematology:chapter-01:0008:0001"
    c90 = "clinical-hematology:chapter-01:0009:0000"
    add_targets("CH01-04", "血小板参数", "HAS_METRIC", [
        "血小板数量", "平均血小板体积",
    ], c80, target_type="LabIndicator", rationale="原文依次定义血小板计数和平均血小板体积")
    add("CH01-04", "血小板参数", "HAS_METRIC", "血小板压积", c90,
        target_type="LabIndicator", rationale="原文在血小板参数序列中直接定义血小板压积")
    add_targets("CH01-04", "血小板数量", "HAS_STATE", ["血小板减少", "血小板增多"], c80,
                rationale="原文按血小板数量分别定义减少和增多")
    add_targets("CH01-04", "血小板数量", "ASSOCIATED_WITH", ["止血功能", "凝血功能"], c80,
                rationale="原文明示血小板数量和质量与止血、凝血功能密切相关")
    add("CH01-04", "PLT<100×10^9/L", "IS_A", "血小板减少", c80,
        rationale="原文明确把该阈值定义为血小板减少")
    add("CH01-04", "PLT>400×10^9/L", "IS_A", "血小板增多", c80,
        rationale="原文明确把该阈值定义为血小板增多")
    add_targets("CH01-04", "血小板减少", "CAUSES", [
        "鼻出血", "牙龈出血", "皮肤紫癜", "瘀斑", "呕血", "内脏出血",
    ], c80, rationale="原文明示血小板减少容易发生或可出现这些出血表现")
    add_targets("CH01-04", "血小板减少", "INDICATES", [
        "再生障碍性贫血", "放射性损伤", "巨幼细胞贫血", "急性白血病",
        "原发性血小板减少性紫癜", "弥散性血管内凝血", "脾肿大", "脾功能亢进",
    ], c80, rationale="血小板减少段直接列出相应疾病和常见病因")
    for cause in ["骨髓造血功能受损", "血小板破坏过多", "血小板消耗过多", "血小板分布异常", "体外循环手术"]:
        add("CH01-04", cause, "CAUSES", "血小板减少", c80,
            rationale="原文将该机制或操作列为血小板减少原因")
    for cause in ["再生障碍性贫血", "放射性损伤", "巨幼细胞贫血", "急性白血病", "原发性血小板减少性紫癜", "弥散性血管内凝血", "脾肿大", "脾功能亢进"]:
        add("CH01-04", cause, "CAUSES", "血小板减少", c80,
            rationale="原文在血小板减少的具体发生机制下列出该疾病")
    add_targets("CH01-04", "血小板增多", "INDICATES", [
        "原发性血小板增多症", "真性红细胞增多症", "慢性粒细胞白血病",
        "急性感染", "急性大出血", "急性溶血",
    ], c80, rationale="血小板增多段直接列出相关疾病和反应状态")
    add_targets("CH01-04", "血小板增多", "CAUSES", [
        "深静脉血栓", "脑血栓", "血栓性并发症",
    ], c80, rationale="原文明示血小板增多容易发生或可导致这些血栓结局")
    for cause in ["急性感染", "急性大出血", "急性溶血"]:
        add("CH01-04", cause, "CAUSES", "血小板增多", c80,
            rationale="原文将该情况列为反应性血小板增多原因")
    add("CH01-04", "血液容易凝固", "CAUSES", "血栓形成", c80,
        rationale="原文明示血液容易凝固可导致血管内血栓形成")
    add_targets("CH01-04", "平均血小板体积增高", "INDICATES", [
        "血小板破坏增多", "骨髓代偿功能良好",
    ], c81, rationale="原文直接描述平均血小板体积增高见于该联合背景")
    add_targets("CH01-04", "平均血小板体积减低", "INDICATES", [
        "骨髓造血功能不良", "血小板生成减少",
    ], c81, rationale="原文直接描述平均血小板体积减低见于该背景")
    add_targets("CH01-04", "平均血小板体积", "HAS_STATE", [
        "平均血小板体积增高", "平均血小板体积减低",
    ], c81, rationale="平均血小板体积异常结果的两个状态")
    add("CH01-04", "血小板破坏增多", "CAUSES", "平均血小板体积增高", c81,
        rationale="原文明确平均血小板体积增高见于血小板破坏增多")
    add("CH01-04", "骨髓造血功能不良", "CAUSES", "平均血小板体积减低", c81,
        rationale="原文明确骨髓造血功能不良时平均血小板体积减低")
    add("CH01-04", "血小板生成减少", "CAUSES", "平均血小板体积减低", c81,
        rationale="原文把血小板生成减少与平均血小板体积减低直接对应")

    # CH01-05：血小板压积/分布宽度、铁代谢和巨幼细胞贫血检验。
    c91 = "clinical-hematology:chapter-01:0009:0001"
    add("CH01-05", "血小板参数", "HAS_METRIC", "血小板体积分布宽度", c91,
        target_type="LabIndicator", rationale="原文在血小板参数序列中直接定义血小板体积分布宽度")
    c100 = "clinical-hematology:chapter-01:0010:0000"
    c101 = "clinical-hematology:chapter-01:0010:0001"
    c110 = "clinical-hematology:chapter-01:0011:0000"
    c111 = "clinical-hematology:chapter-01:0011:0001"
    c121 = "clinical-hematology:chapter-01:0012:0001"
    c130 = "clinical-hematology:chapter-01:0013:0000"
    add_targets("CH01-05", "血小板压积", "HAS_STATE", ["血小板压积增高", "血小板压积减低"], c91,
                rationale="原文分别列出血小板压积增高和减低")
    add_targets("CH01-05", "血小板压积增高", "INDICATES", [
        "骨髓纤维化", "慢性粒细胞白血病", "脾切除后",
    ], c91, rationale="血小板压积增高条目直接列出相应情况")
    add_targets("CH01-05", "血小板压积减低", "INDICATES", [
        "再生障碍性贫血", "血小板减少症", "化疗以后",
    ], c91, rationale="血小板压积减低条目直接列出相应情况")
    add("CH01-05", "血小板体积分布宽度", "HAS_STATE", "血小板体积分布宽度增高", c91,
        rationale="原文定义血小板体积分布宽度增高状态")
    add_targets("CH01-05", "血小板体积分布宽度增高", "INDICATES", [
        "巨幼细胞贫血", "急性白血病化疗后", "慢性粒细胞白血病", "血栓性疾病",
    ], c91, rationale="PDW 增高条目直接列出相应疾病或治疗后状态")
    add("CH01-05", "铁缺乏", "CAUSES", "血红蛋白合成减少", c91,
        rationale="原文明示铁缺乏时血红蛋白合成减少")
    add("CH01-05", "铁缺乏", "CAUSES", "缺铁性贫血", c91,
        rationale="章节定义和异常解读均明确铁缺乏导致缺铁性贫血")
    add("CH01-05", "血红蛋白合成减少", "CAUSES", "小红细胞低色素性贫血", c91,
        rationale="原文解释血红蛋白合成减少后呈现小红细胞低色素性贫血")
    add_targets("CH01-05", "缺铁性贫血检验", "HAS_METRIC", [
        "血清铁", "血清铁蛋白", "血清转铁蛋白", "总铁结合力",
    ], c100, rationale="本节依次把四项指标列为缺铁性贫血检验项目")
    add_targets("CH01-05", "血清铁", "HAS_STATE", ["血清铁降低", "血清铁升高"], c100,
                rationale="血清铁异常结果的两个方向")
    for cause in ["摄入不足", "缺铁性饮食", "肠道吸收不良", "慢性失血", "胃溃疡出血", "十二指肠溃疡出血", "钩虫病", "月经过多", "铁缺乏", "妊娠期", "哺乳期妇女", "婴幼儿时期", "严重感染", "恶性肿瘤", "肝硬化"]:
        add("CH01-05", cause, "CAUSES", "血清铁降低", c100,
            source_type="ClinicalContext" if cause == "月经过多" else None,
            rationale="原文把该因素列在血清铁降低的原因中")
    for cause in ["红细胞破坏过多", "溶血性贫血", "铁的利用障碍", "再生障碍性贫血", "巨幼红细胞贫血", "铅中毒", "铁吸收增加", "长期反复输血", "铁剂治疗"]:
        add("CH01-05", cause, "CAUSES", "血清铁升高", c100,
            rationale="原文把该因素列在血清铁升高的原因中")
    add_targets("CH01-05", "血清铁蛋白", "HAS_STATE", ["血清铁蛋白降低", "血清铁蛋白升高"], c101,
                rationale="血清铁蛋白异常结果的两个方向")
    for cause in ["体内贮存铁减少", "肝脏铁蛋白合成减少", "长期腹泻", "营养不良"]:
        add("CH01-05", cause, "CAUSES", "血清铁蛋白降低", c101,
            rationale="原文把该因素列为血清铁蛋白降低原因")
    add("CH01-05", "血清铁蛋白降低", "INDICATES", "缺铁性贫血", c101,
        rationale="原文称其为诊断缺铁性贫血的重要指标")
    for cause in ["体内贮存铁增加", "反复输血的患者", "铁蛋白合成增加", "急性感染", "急性炎症", "甲状腺功能亢进", "恶性肿瘤", "肝癌", "胰腺癌", "组织内铁蛋白释放增加", "慢性肝病", "肝坏死"]:
        add("CH01-05", cause, "CAUSES", "血清铁蛋白升高", c101,
            rationale="原文把该因素列为血清铁蛋白升高原因")
    add_targets("CH01-05", "血清转铁蛋白", "HAS_STATE", ["血清转铁蛋白升高", "血清转铁蛋白降低"], c110,
                rationale="转铁蛋白异常结果的两个方向")
    for cause in ["铁缺乏", "缺铁性贫血", "慢性失血", "妊娠中期", "妊娠后期", "长期口服避孕药"]:
        add("CH01-05", cause, "CAUSES", "血清转铁蛋白升高", c110,
            rationale="原文把该因素列为血清转铁蛋白升高原因")
    for cause in ["严重的肝病", "营养不良", "转铁蛋白合成减少", "肾病综合征", "转铁蛋白丢失增加"]:
        add("CH01-05", cause, "CAUSES", "血清转铁蛋白降低", c110,
            rationale="原文把该因素列为血清转铁蛋白降低原因")
    add_targets("CH01-05", "总铁结合力", "HAS_STATE", ["总铁结合力降低", "总铁结合力增高"], c111,
                rationale="总铁结合力异常结果的两个方向")
    for cause in ["转铁蛋白合成不足", "遗传性转铁蛋白缺乏症", "肝硬化", "转铁蛋白丢失增加", "肾病综合征", "尿毒症", "肿瘤", "慢性感染", "珠蛋白合成障碍性贫血"]:
        add("CH01-05", cause, "CAUSES", "总铁结合力降低", c111,
            source_type="Disease" if cause == "肿瘤" else None,
            rationale="原文把该因素列为总铁结合力降低原因")
    for cause in ["转铁蛋白合成增加", "缺铁性贫血", "妊娠后期", "转铁蛋白释放增加", "急性肝炎", "肝细胞坏死"]:
        add("CH01-05", cause, "CAUSES", "总铁结合力增高", c111,
            rationale="原文把该因素列为总铁结合力增高原因")
    add("CH01-05", "脱氧核糖核酸合成障碍", "CAUSES", "巨幼细胞贫血", c121,
        rationale="原文定义巨幼细胞贫血由 DNA 合成障碍引起")
    # 冻结实体目录没有“维生素 B12 缺乏”端点，因此不把其病因错误连接到
    # “维生素 B12”本身，也不跨过缺乏状态直接推导巨幼细胞贫血。
    for child in ["缺铁性贫血", "巨幼细胞贫血", "巨幼红细胞贫血", "小红细胞低色素性贫血", "大红细胞性贫血", "孕妇贫血", "轻度贫血"]:
        add("CH01-05", child, "IS_A", "贫血", c130 if "巨幼" in child or "大红" in child else c100,
            source_type="Disease",
            rationale="疾病名称和章节分类明确表达该疾病属于贫血")

    # CH01-06：叶酸缺乏的结局及血液流变学。
    c140 = "clinical-hematology:chapter-01:0014:0000"
    c141 = "clinical-hematology:chapter-01:0014:0001"
    c150 = "clinical-hematology:chapter-01:0015:0000"
    c151 = "clinical-hematology:chapter-01:0015:0001"
    c160 = "clinical-hematology:chapter-01:0016:0000"
    c170 = "clinical-hematology:chapter-01:0017:0000"
    c171 = "clinical-hematology:chapter-01:0017:0001"
    c180 = "clinical-hematology:chapter-01:0018:0000"
    c181 = "clinical-hematology:chapter-01:0018:0001"
    add("CH01-06", "叶酸缺乏", "CAUSES", "巨幼细胞贫血", c140,
        rationale="原文明示人体缺少叶酸可导致巨幼细胞贫血")
    for outcome in ["胎儿神经管畸形", "唇腭裂", "流产"]:
        add("CH01-06", "叶酸缺乏", "CAUSES", outcome, c140,
            rationale="原文明示怀孕早期叶酸缺乏可导致该结局")
    for outcome in ["胎儿宫内发育迟缓", "早产", "出生低体重"]:
        add("CH01-06", "叶酸缺乏", "CAUSES", outcome, c141,
            rationale="原文明示妊娠中晚期叶酸缺乏可导致该结局")
    add_targets("CH01-06", "血液流变学检查", "HAS_METRIC", [
        "全血黏度", "血浆黏度", "全血还原黏度", "红细胞变形性", "红细胞聚集指数", "红细胞电泳时间",
    ], c141, target_type="LabIndicator", rationale="原文在血液流变学检查章节依次定义这些检测指标")
    add_targets("CH01-06", "红细胞变形性", "HAS_METRIC", [
        "红细胞刚性指数", "红细胞变形指数", "红细胞滤过指数",
    ], c170, target_type="LabIndicator", rationale="原文明确列出红细胞变形性的三个测定指标")
    add_targets("CH01-06", "全血黏度", "HAS_STATE", ["全血黏度升高"], c151,
                rationale="异常结果解读定义全血黏度升高")
    add("CH01-06", "全血黏度(高切)", "HAS_STATE", "全血黏度(高切)升高", c150,
        rationale="原文直接定义高切全血黏度升高")
    add("CH01-06", "全血黏度(低切)", "HAS_STATE", "全血黏度(低切)升高", c150,
        rationale="原文直接定义低切全血黏度升高")
    # 高/中/低切属于指标分解，且其直接原因是状态到状态的因果；当前 Schema
    # 对这两类端点均无合法关系类型，因此只保留各自 HAS_STATE，不降级表示。
    for cause in ["脱水", "肺心病", "充血性心力衰竭", "高山病", "尘肺", "烧伤", "真性红细胞增多症", "脑血栓", "心肌梗死", "血栓闭塞性脉管炎", "糖尿病", "雷诺病", "缺氧", "酸中毒", "遗传性球形红细胞增多症", "遗传性椭圆形红细胞增多症", "高脂血症", "高血压", "多发性骨髓瘤", "原发性巨球蛋白血症", "免疫球蛋白增多症"]:
        add("CH01-06", cause, "CAUSES", "全血黏度升高", c151,
            rationale="全血黏度异常结果条目把该疾病或因素列为升高原因")
    add("CH01-06", "血浆黏度", "HAS_STATE", "血浆黏度升高", c160,
        source_type="LabIndicator",
        rationale="血浆黏度异常结果定义其升高状态")
    for cause in ["高脂血症", "脂肪肝", "动脉粥样硬化", "糖尿病", "肝硬化", "急性心肌梗死", "急性脑梗死", "多发性骨髓瘤", "原发性巨球蛋白血症", "自身免疫性疾病"]:
        add("CH01-06", cause, "CAUSES", "血浆黏度升高", c160,
            rationale="血浆黏度升高条目直接列出该疾病")
    # 全血还原黏度的高低切分解尚未补为实体层级关系。
    add_targets("CH01-06", "红细胞变形能力异常", "INDICATES", [
        "镰状红细胞贫血", "珠蛋白生成障碍性贫血", "遗传性球形红细胞增多症",
        "遗传性椭圆形红细胞增多症", "免疫性溶血性贫血", "高脂血症", "糖尿病",
        "脑血栓", "心肌梗死", "冠心病", "红细胞增多症", "继发性红细胞增多症",
        "法洛四联症", "肺心病",
    ], c171, rationale="红细胞变形能力异常条目直接列出相应疾病")
    add("CH01-06", "红细胞聚集指数", "HAS_STATE", "红细胞聚集性增高", c180,
        target_type="IndicatorState", rationale="原文把聚集指数异常解释为红细胞聚集性增高")
    add_targets("CH01-06", "红细胞聚集性增高", "INDICATES", [
        "急性心肌梗死", "脑血栓", "高脂血症", "周围血管深静脉血栓",
    ], c180, rationale="红细胞聚集性增高条目直接列出相应疾病")
    add("CH01-06", "红细胞电泳时间", "HAS_STATE", "红细胞电泳时间延长", c181,
        rationale="原文明确电泳时间延长状态")
    add_targets("CH01-06", "红细胞电泳时间延长", "INDICATES", [
        "缺血性脑卒中", "冠心病", "肺心病", "心肌梗死", "高血压", "系统性红斑狼疮",
    ], c181, rationale="电泳时间延长条目直接列出相应疾病")

    # CH01-07：红细胞沉降率、ABO 血型与凝血酶原时间定义。
    c190 = "clinical-hematology:chapter-01:0019:0000"
    c200 = "clinical-hematology:chapter-01:0020:0000"
    c201 = "clinical-hematology:chapter-01:0020:0001"
    add("CH01-07", "血液流变学检查", "HAS_METRIC", "红细胞沉降率", c190,
        target_type="LabIndicator", rationale="原文将红细胞沉降率列在血液流变学检查章节下")
    add_targets("CH01-07", "红细胞沉降率", "HAS_STATE", ["血沉加快", "血沉减慢"], c190,
                rationale="原文反复明确血沉加快和减慢两个状态")
    for cause in ["红细胞聚集", "贫血", "妇女月经期", "妊娠3个月以后", "60岁以上的老年人", "急性细菌性感染", "风湿热活动期", "肺结核活动期", "组织损伤", "组织坏死", "心肌梗死", "肺梗死", "手术创伤", "恶性肿瘤", "恶性淋巴瘤", "多发性骨髓瘤", "系统性红斑狼疮", "类风湿关节炎", "亚急性感染性心内膜炎", "慢性肾炎", "肝硬化", "高胆固醇血症", "动脉粥样硬化", "糖尿病", "肾病综合征"]:
        evidence_chunk = c200 if cause in {
            "高胆固醇血症", "动脉粥样硬化", "糖尿病", "肾病综合征",
        } else c190
        add("CH01-07", cause, "CAUSES", "血沉加快", evidence_chunk,
            rationale="原文将该因素或疾病列为血沉加快原因")
    add_targets("CH01-07", "血沉加快", "INDICATES", [
        "急性细菌性感染", "风湿热活动期", "肺结核活动期", "心肌梗死", "肺梗死",
        "恶性肿瘤", "恶性淋巴瘤", "多发性骨髓瘤", "系统性红斑狼疮",
        "类风湿关节炎", "亚急性感染性心内膜炎", "慢性肾炎", "肝硬化",
        "贫血",
    ], c190, rationale="病理性血沉增快条目直接列出这些疾病")
    add_targets("CH01-07", "血沉加快", "INDICATES", [
        "高胆固醇血症", "动脉粥样硬化", "糖尿病", "肾病综合征",
    ], c200, rationale="续页的血沉增快条目直接列出这些疾病")
    add("CH01-07", "喝水", "CAUSES", "血黏度降低", c200,
        target_type="IndicatorState",
        rationale="原文明示喝水稀释血液使血黏度降低")
    add("CH01-07", "冬天气温低", "CAUSES", "血黏度升高", c200,
        target_type="IndicatorState",
        rationale="原文明示冬天气温低使血黏度升高")
    add("CH01-07", "夏天气温高", "CAUSES", "血黏度降低", c200,
        target_type="IndicatorState",
        rationale="原文明示夏天气温高使血黏度降低")
    add("CH01-07", "阿司匹林", "CAUSES", "血黏度降低", c200,
        target_type="IndicatorState",
        rationale="原文明示阿司匹林可使血黏度降低")
    add_targets("CH01-07", "红细胞ABO血型", "HAS_STATE", ["A型血", "B型血", "O型血", "AB型血"], c201,
                rationale="原文明确 ABO 血型分为 A、B、O、AB 四型")
    add_targets("CH01-07", "红细胞ABO血型鉴定", "HAS_METRIC", [
        "标准血清+受检者红细胞", "标准红细胞+受检者血清", "被鉴定者血型",
    ], c201, rationale="鉴定表明确包含正定型、反定型和最终血型结果")
    # PT、PTR、INR 和 ISI 之间是计算公式，按合同只进入规则层。

    # CH01-08：PT 异常结果和 D-二聚体。
    c221 = "clinical-hematology:chapter-01:0022:0001"
    c230 = "clinical-hematology:chapter-01:0023:0000"
    c231 = "clinical-hematology:chapter-01:0023:0001"
    add_targets("CH01-08", "PT", "HAS_STATE", ["PT延长", "PT缩短"], c221,
                rationale="PT 异常结果明确分为延长和缩短")
    for cause in ["先天性凝血因子I缺乏", "先天性凝血因子II缺乏", "先天性凝血因子V缺乏", "先天性凝血因子VII缺乏", "先天性凝血因子X缺乏", "获得性凝血因子缺乏", "严重肝病", "维生素K缺乏", "纤溶亢进", "弥散性血管内凝血", "口服抗凝药物", "异常抗凝血物质"]:
        add("CH01-08", cause, "CAUSES", "PT延长", c221,
            rationale="原文将该因素列为 PT 延长原因")
    add_targets("CH01-08", "PT延长", "INDICATES", [
        "严重肝病", "弥散性血管内凝血",
    ], c221, rationale="PT 延长条目直接列出相应疾病")
    add("CH01-08", "血液高凝状态", "CAUSES", "PT缩短", c221,
        rationale="原文明示 PT 缩短见于血液高凝状态")
    add_targets("CH01-08", "PT缩短", "INDICATES", [
        "DIC早期", "心肌梗死", "脑血栓", "深静脉血栓",
    ], c221, rationale="PT 缩短条目直接列出相应高凝疾病")
    add_targets("CH01-08", "D-二聚体", "HAS_STATE", [
        "D-二聚体阳性", "D-二聚体为阴性", "D-二聚体正常",
    ], c230, rationale="D-二聚体异常解读明确阳性、阴性和正常状态")
    add_targets("CH01-08", "D-二聚体阳性", "INDICATES", [
        "心肌梗死", "脑梗死", "肺栓塞", "恶性肿瘤", "外科手术", "炎症", "感染", "妊娠", "继发性纤溶症", "弥散性血管内凝血",
    ], c231, rationale="D-二聚体阳性条目直接列出相应疾病和临床状态")
    add("CH01-08", "D-二聚体为阴性", "INDICATES", "原发性纤溶症", c231,
        rationale="原文明示原发性纤溶症时 D-二聚体阴性或不升高")
    # “D-二聚体正常可排除 DVT”是否定语义，当前普通关系不表达禁止关系。

    case_values = list(sections.values())
    if not manual_relationship_identities <= seen:
        missing = sorted(manual_relationship_identities - seen)
        raise ValueError(f"v0.3人工关系未被完整继承: {missing[:5]}")
    for section in case_values:
        section["relationships"].sort(key=lambda item: (
            item["evidence_chunk_ids"], item["relation_type"],
            item["source_canonical_name"], item["target_canonical_name"],
        ))
    return {
        "schema_version": "medical-kg-chapter-relationship-gold/v1.0",
        "status": "MANUAL_GRAPH_GOLD_WITH_AUTOMATED_AUGMENTATION",
        "gold_provenance": {
            "method": "V03_MANUAL_GRAPH_INHERITANCE_PLUS_CHAPTER_RELATION_AUDIT",
            "manual_graph_status": manual_graph["status"],
            "human_approved": False,
            "approval_note": "v0.3 is manually annotated; its existing HUMAN_REVIEW_REQUIRED status is preserved.",
            "prediction_blind_initial_annotation": True,
            "candidate_predictions_used_after_annotation_for_omission_audit": True,
        },
        "scope_contract": {
            "sections": [f"CH01-{index:02d}" for index in range(1, 9)],
            "closed_world_within_frozen_v08_entities": True,
            "rules_excluded": True,
            "cross_chunk_relationships_allowed": True,
            "identity": ["source_canonical_id", "relation_type", "target_canonical_id"],
        },
        "source_canonical_entities": str(ENTITY_PATH.relative_to(ROOT)),
        "source_manual_graph": str(MANUAL_GRAPH_PATH.relative_to(ROOT)),
        "cases": case_values,
        "statistics": {
            "case_count": len(case_values),
            "chunk_count": len({cid for case in case_values for cid in case["chunk_ids"]}),
            "relationship_count": sum(len(case["relationships"]) for case in case_values),
            "inherited_manual_relationship_count": len(manual_relationship_identities),
        },
    }


if __name__ == "__main__":
    payload = build_gold()
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["statistics"], ensure_ascii=False, sort_keys=True))
