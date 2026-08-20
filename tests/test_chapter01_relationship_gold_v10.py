"""第一章全章 v1.0 关系标注稿的结构与 Schema 测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from medical_kg_sourceprep.extraction.graph_builder.schema import (
    _relation_endpoint_pairs,
    load_candidate_graph_schema,
)


ROOT = Path(__file__).resolve().parents[1]


class Chapter01RelationshipGoldV10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gold = json.loads(
            (ROOT / "evaluation/chapter-01/chapter-01-relationship-gold-v1.0.json")
            .read_text(encoding="utf-8")
        )
        catalog = json.loads(
            (ROOT / "evaluation/chapter-01/chapter-01-canonical-entities-v0.8.json")
            .read_text(encoding="utf-8")
        )
        cls.entity_by_id = {item["canonical_id"]: item for item in catalog["canonical_entities"]}
        cls.schema = load_candidate_graph_schema()

    def test_scope_and_counts_are_consistent(self) -> None:
        self.assertEqual(
            [case["case_id"] for case in self.gold["cases"]],
            [f"CH01-{index:02d}" for index in range(1, 9)],
        )
        relationships = [item for case in self.gold["cases"] for item in case["relationships"]]
        self.assertEqual(self.gold["statistics"]["relationship_count"], len(relationships))
        identities = {
            (item["source_canonical_id"], item["relation_type"], item["target_canonical_id"])
            for item in relationships
        }
        self.assertEqual(len(identities), len(relationships))

    def test_all_endpoints_and_schema_pairs_are_valid(self) -> None:
        for case in self.gold["cases"]:
            chunk_ids = set(case["chunk_ids"])
            for relationship in case["relationships"]:
                source = self.entity_by_id[relationship["source_canonical_id"]]
                target = self.entity_by_id[relationship["target_canonical_id"]]
                self.assertIn(
                    (source["entity_type"], target["entity_type"]),
                    _relation_endpoint_pairs(self.schema, relationship["relation_type"]),
                )
                self.assertTrue(set(relationship["evidence_chunk_ids"]) <= chunk_ids)

    def test_requested_nested_metric_relationships_are_complete(self) -> None:
        relationships = [item for case in self.gold["cases"] for item in case["relationships"]]
        actual = [
            (
                item["source_canonical_name"],
                item["relation_type"],
                item["target_canonical_name"],
            )
            for item in relationships
        ]
        expected = {
            ("血液常规检验", "HAS_METRIC", "血小板参数"),
            ("红细胞变形性", "HAS_METRIC", "红细胞刚性指数"),
            ("红细胞变形性", "HAS_METRIC", "红细胞变形指数"),
            ("红细胞变形性", "HAS_METRIC", "红细胞滤过指数"),
            ("血液流变学检查", "HAS_METRIC", "全血黏度"),
            ("血液流变学检查", "HAS_METRIC", "血浆黏度"),
            ("血小板参数", "HAS_METRIC", "血小板数量"),
            ("血小板参数", "HAS_METRIC", "平均血小板体积"),
            ("血小板参数", "HAS_METRIC", "血小板体积分布宽度"),
            ("血小板参数", "HAS_METRIC", "血小板压积"),
            ("血液流变学检查", "HAS_METRIC", "全血还原黏度"),
            ("血液流变学检查", "HAS_METRIC", "红细胞变形性"),
            ("血液流变学检查", "HAS_METRIC", "红细胞聚集指数"),
            ("血液流变学检查", "HAS_METRIC", "红细胞电泳时间"),
            ("血液流变学检查", "HAS_METRIC", "红细胞沉降率"),
        }
        for identity in expected:
            self.assertEqual(actual.count(identity), 1, identity)

    def test_every_manual_graph_relationship_is_preserved(self) -> None:
        manual = json.loads(
            (ROOT / "evaluation/chapter-01/chapter-01-graph-test-set-v0.3.json")
            .read_text(encoding="utf-8")
        )
        names: dict[tuple[str, str], str] = {}
        for entity in self.entity_by_id.values():
            for name in (entity["canonical_name"], *entity.get("aliases", [])):
                names[(entity["entity_type"], name)] = entity["canonical_name"]
        expected = set()
        for case in manual["cases"]:
            types = {name: entity_type for entity_type, name in case["entities"]}
            for source, relation_type, target in case["relationships"]:
                expected.add((
                    names[(types[source], source)], relation_type,
                    names[(types[target], target)],
                ))
        actual = {
            (
                item["source_canonical_name"], item["relation_type"],
                item["target_canonical_name"],
            )
            for case in self.gold["cases"]
            for item in case["relationships"]
        }
        self.assertTrue(expected <= actual, sorted(expected - actual)[:10])

    def test_wbc_panel_uses_measurement_indicators(self) -> None:
        actual = {
            item["target_canonical_name"]
            for case in self.gold["cases"]
            for item in case["relationships"]
            if item["source_canonical_name"] == "白细胞分类计数"
            and item["relation_type"] == "HAS_METRIC"
        }
        self.assertTrue({
            "中性粒细胞绝对值", "淋巴细胞绝对值", "单核细胞绝对值",
            "嗜酸性粒细胞绝对值", "嗜碱性粒细胞绝对值",
        } <= actual)
        self.assertTrue({
            "中性粒细胞", "淋巴细胞", "单核细胞", "嗜酸性粒细胞", "嗜碱性粒细胞",
        }.isdisjoint(actual))

    def test血小板出血和血栓结局使用因果关系及正确证据块(self) -> None:
        relationships = [item for case in self.gold["cases"] for item in case["relationships"]]
        expected_targets = {
            "血小板减少": {"鼻出血", "牙龈出血", "皮肤紫癜", "瘀斑", "呕血", "内脏出血"},
            "血小板增多": {"深静脉血栓", "脑血栓", "血栓性并发症"},
        }
        selected = [
            item
            for item in relationships
            if item["source_canonical_name"] in expected_targets
            and item["target_canonical_name"] in expected_targets[item["source_canonical_name"]]
        ]

        self.assertEqual(len(selected), 9)
        for relationship in selected:
            self.assertEqual(relationship["relation_type"], "CAUSES")
            self.assertEqual(
                relationship["evidence_chunk_ids"],
                ["clinical-hematology:chapter-01:0008:0000"],
            )


if __name__ == "__main__":
    unittest.main()
