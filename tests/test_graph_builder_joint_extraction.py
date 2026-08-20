"""联合抽取实验的临时端点映射测试。"""

from __future__ import annotations

import hashlib
import unittest

from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jRelationship

from medical_kg_sourceprep.extraction.graph_builder.joint_extraction import (
    TABLE_PROMPT_VERSION_CONTEXT,
    TABLE_PROMPT_VERSION_REFINED,
    _adapt_nodes,
    _adapt_relationships,
    _remap_relationships,
    build_evidence_units,
    build_routed_evidence_groups,
)
from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk


class JointExtractionTests(unittest.TestCase):
    def test结构路由分别保留表格列表和参考区间坐标(self) -> None:
        text = (
            "(1) 血清铁降低\n\n1) 摄入不足: 缺铁性饮食。\n"
            "<table><tr><td>血清铁</td><td>TIBC</td></tr>"
            "<tr><td>↓</td><td>↑</td></tr></table>\n"
            "（一）血浆凝血酶原时间\nPT 定义。\n"
            "(1) 仪器法: 11~13 秒, 超过正常对照值 3 秒为异常。\n"
            "【异常结果解读】"
        )

        groups = build_routed_evidence_groups(text)

        self.assertEqual({group.route for group in groups}, {"list", "table", "range"})
        for group in groups:
            self.assertEqual(group.units[0].text, text[group.units[0].start:group.units[0].end])
            self.assertTrue(all(group.start <= unit.start < unit.end <= group.end for unit in group.units))

    def test列表状态标题和编号原因形成连续上下文证据(self) -> None:
        text = "【异常结果解读】\n(1) 血清铁降低\n\n1) 摄入不足: 缺铁性饮食。"

        units = build_evidence_units(text)

        contexts = [unit.text for unit in units if unit.kind == "list_context"]
        self.assertEqual(contexts, ["(1) 血清铁降低\n\n1) 摄入不足: 缺铁性饮食。"])

    def test新版表格路由同时提供表题前文和全部表格行(self) -> None:
        text = (
            "血红蛋白: 男性 130~175g/L\n"
            "表 1-1 贫血程度的诊断标准\n"
            "<table><tr><td>贫血程度</td><td>轻度贫血</td></tr>"
            "<tr><td>血红蛋白</td><td>90~120g/L</td></tr></table>"
        )

        legacy = next(group for group in build_routed_evidence_groups(text) if group.route == "table")
        contextual = next(
            group
            for group in build_routed_evidence_groups(
                text,
                table_prompt_version=TABLE_PROMPT_VERSION_CONTEXT,
            )
            if group.route == "table"
        )

        self.assertTrue(all(unit.kind == "table_row" for unit in legacy.units))
        self.assertEqual(contextual.start, 0)
        self.assertEqual(
            [unit.text for unit in contextual.units if unit.kind == "line"],
            ["血红蛋白: 男性 130~175g/L", "表 1-1 贫血程度的诊断标准"],
        )
        self.assertEqual(
            len([unit for unit in contextual.units if unit.kind == "table_row"]),
            2,
        )
        self.assertIn("阈值分类表", contextual.instructions)
        self.assertIn("绝不能把例子", contextual.instructions)

    def test精修路由识别分类列表并禁止联合条件直连(self) -> None:
        text = (
            "(1) 小细胞均一性贫血: MCV 减小, RDW 正常, 如甲病。\n"
            "(2) 小细胞不均一性贫血: MCV 减小, RDW 增大, 如乙病。\n"
            "(3) 正细胞均一性贫血: MCV、RDW 均正常, 如丙病。"
        )

        groups = build_routed_evidence_groups(
            text,
            table_prompt_version=TABLE_PROMPT_VERSION_REFINED,
        )

        group = next(item for item in groups if item.route == "classification_list")
        self.assertEqual(len(group.units), 3)
        self.assertIn("不得把任一状态单独连接到分类名称", group.instructions)

    def test精修路由识别共享谓词条目并允许提出缺失端点(self) -> None:
        text = (
            "【异常结果解读】\n"
            "②造血原料缺乏所致的贫血：如铁缺乏引起的缺铁性贫血或"
            "叶酸、维生素 B12 缺乏引起的巨幼细胞贫血。"
        )

        groups = build_routed_evidence_groups(
            text,
            table_prompt_version=TABLE_PROMPT_VERSION_REFINED,
        )

        group = next(item for item in groups if item.route == "shared_predicate")
        self.assertEqual(len(group.units), 1)
        self.assertIn("主语必须分别输出", group.instructions)

    def test精修表格提示区分维度标签和语义父类(self) -> None:
        text = (
            "表 1-1 贫血程度的诊断标准\n"
            "<table><tr><td>贫血程度</td><td>轻度贫血</td></tr>"
            "<tr><td>血红蛋白</td><td>90~120g/L</td></tr></table>"
        )

        group = next(
            item
            for item in build_routed_evidence_groups(
                text,
                table_prompt_version=TABLE_PROMPT_VERSION_REFINED,
            )
            if item.route == "table"
        )

        self.assertIn("共同父类", group.instructions)
        self.assertIn("不是描述分类维度的‘贫血程度’", group.instructions)

    def test表格状态协议由代码生成逐字双锚点(self) -> None:
        text = (
            "<table><tr><td>血清铁</td><td>原因</td></tr>"
            "<tr><td>↓</td><td>缺铁性贫血</td></tr></table>"
        )
        chunk = EvidenceChunk("test:table", text, hashlib.sha256(text.encode()).hexdigest())
        units = build_evidence_units(text)
        unit_by_id = {unit.unit_id: unit for unit in units}
        header_id = next(unit.unit_id for unit in units if "血清铁" in unit.text)
        row_id = next(unit.unit_id for unit in units if "↓" in unit.text)
        payload = {
            "nodes": [
                {
                    "id": "indicator",
                    "entity_type": "LabIndicator",
                    "mention": "血清铁",
                    "evidence_unit_ids": [header_id],
                    "derivation": None,
                },
                {
                    "id": "state",
                    "entity_type": "IndicatorState",
                    "mention": "血清铁降低",
                    "evidence_unit_ids": [row_id],
                    "derivation": {
                        "kind": "TABLE_STATE",
                        "indicator_id": "indicator",
                        "state": "LOW",
                        "header_unit_id": header_id,
                        "row_unit_id": row_id,
                    },
                },
            ],
            "relationships": [
                {
                    "relation_type": "HAS_STATE",
                    "indicator_id": "indicator",
                    "state_id": "state",
                    "evidence_unit_ids": [header_id, row_id],
                }
            ],
        }

        nodes, mention_by_id, node_reviews = _adapt_nodes(payload, unit_by_id=unit_by_id)
        relationships, relation_reviews, _audit = _adapt_relationships(
            payload,
            chunk=chunk,
            unit_by_id=unit_by_id,
            mention_by_id=mention_by_id,
        )

        self.assertFalse(node_reviews)
        self.assertFalse(relation_reviews)
        self.assertIn("table_state_evidence_json", nodes[1].properties)
        self.assertEqual(relationships[0].start_node_id, "indicator")
        self.assertEqual(relationships[0].end_node_id, "state")
        self.assertEqual(relationships[0].properties, {})

    def test因果专用角色固定映射为原因指向结果(self) -> None:
        text = "缺铁性饮食导致摄入不足。"
        chunk = EvidenceChunk("test:causes", text, hashlib.sha256(text.encode()).hexdigest())
        units = build_evidence_units(text)
        unit_by_id = {unit.unit_id: unit for unit in units}
        unit_id = units[0].unit_id
        payload = {
            "nodes": [],
            "relationships": [
                {
                    "relation_type": "CAUSES",
                    "cause_id": "cause",
                    "effect_id": "effect",
                    "evidence_unit_ids": [unit_id],
                }
            ],
        }

        relationships, reviews, _audit = _adapt_relationships(
            payload,
            chunk=chunk,
            unit_by_id=unit_by_id,
            mention_by_id={"cause": "缺铁性饮食", "effect": "摄入不足"},
        )

        self.assertFalse(reviews)
        self.assertEqual(relationships[0].start_node_id, "cause")
        self.assertEqual(relationships[0].end_node_id, "effect")
        self.assertEqual(relationships[0].properties["exact_quote"], "缺铁性饮食导致摄入不足")

    def test过短子项证据回退到包含列表标题的共同单元(self) -> None:
        text = "(1) 血清铁降低\n\n1) 摄入不足: 缺铁性饮食。"
        chunk = EvidenceChunk("test:list", text, hashlib.sha256(text.encode()).hexdigest())
        units = build_evidence_units(text)
        unit_by_id = {unit.unit_id: unit for unit in units}
        child_id = next(unit.unit_id for unit in units if unit.text.startswith("1)"))
        payload = {
            "nodes": [],
            "relationships": [{
                "relation_type": "CAUSES",
                "cause_id": "cause",
                "effect_id": "effect",
                "evidence_unit_ids": [child_id],
            }],
        }

        relationships, reviews, _audit = _adapt_relationships(
            payload,
            chunk=chunk,
            unit_by_id=unit_by_id,
            mention_by_id={"cause": "摄入不足", "effect": "血清铁降低"},
        )

        self.assertFalse(reviews)
        self.assertEqual(relationships[0].properties["exact_quote"], "血清铁降低\n\n1) 摄入不足")

    def test将模型临时端点映射为稳定候选键(self) -> None:
        text = "甲导致乙。"
        chunk = EvidenceChunk("test:joint", text, hashlib.sha256(text.encode()).hexdigest())
        graph = Neo4jGraph(relationships=[Neo4jRelationship(
            start_node_id="temporary-a",
            end_node_id="temporary-b",
            type="CAUSES",
            properties={"exact_quote": text},
        )])

        remapped = _remap_relationships(
            graph,
            chunk=chunk,
            key_by_model_id={"temporary-a": "candidate:a", "temporary-b": "candidate:b"},
        )

        self.assertEqual(remapped.relationships[0].start_node_id, "test:joint:candidate:a")
        self.assertEqual(remapped.relationships[0].end_node_id, "test:joint:candidate:b")

    def test未知端点不被适配层猜测(self) -> None:
        text = "甲导致乙。"
        chunk = EvidenceChunk("test:joint", text, hashlib.sha256(text.encode()).hexdigest())
        graph = Neo4jGraph(relationships=[Neo4jRelationship(
            start_node_id="unknown",
            end_node_id="temporary-b",
            type="CAUSES",
            properties={"exact_quote": text},
        )])

        remapped = _remap_relationships(
            graph, chunk=chunk, key_by_model_id={"temporary-b": "candidate:b"}
        )

        self.assertEqual(remapped.relationships[0].start_node_id, "unknown")
        self.assertEqual(remapped.relationships[0].end_node_id, "test:joint:candidate:b")

    def test兼容节点和关系端点的不同命名空间层数(self) -> None:
        text = "甲导致乙。"
        chunk = EvidenceChunk("test:joint", text, hashlib.sha256(text.encode()).hexdigest())
        graph = Neo4jGraph(relationships=[Neo4jRelationship(
            start_node_id="test:joint:node_1",
            end_node_id="test:joint:node_2",
            type="CAUSES",
            properties={"exact_quote": text},
        )])

        remapped = _remap_relationships(
            graph,
            chunk=chunk,
            key_by_model_id={"node_1": "candidate:a", "node_2": "candidate:b"},
        )

        self.assertEqual(remapped.relationships[0].start_node_id, "test:joint:candidate:a")
        self.assertEqual(remapped.relationships[0].end_node_id, "test:joint:candidate:b")


if __name__ == "__main__":
    unittest.main()
