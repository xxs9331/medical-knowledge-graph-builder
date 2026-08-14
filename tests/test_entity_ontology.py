import unittest
from typing import Any

from medical_kg_sourceprep.extraction.entity_ontology import build_ontology_candidate


class EntityOntologyTests(unittest.TestCase):
    pages: list[dict[str, Any]] = []

    def setUp(self):
        self.pages = [
            {
                "chapter_page_index": 0,
                "printed_page_number": 4,
                "cleaned_path": "pages/cleaned/0000.md",
                "cleaned_sha256": "a" * 64,
                "text": (
                    "男性 女性 成人 儿童 老年人 60 岁以上的老年人\n"
                    "红细胞容积分布宽度(RDW)以变异系数(RDW-CV)或标准差(RDW-SD)表示。\n"
                    "白细胞分类计数(WBC-DC)和白细胞计数(WBC)。\n"
                    "中性粒细胞绝对值 <1.5，称为粒细胞减少症。\n"
                    "红细胞刚性指数(IR)用全血黏度(高切)除以血浆黏度和血细胞压积来计算。\n"
                    "凝血酶原时间比值(PTR)=受检血浆 PT/正常人血浆 PT。\n"
                    "国际标准化比值(INR)=PTR。"
                ),
            }
        ]

    def test_synonyms_and_parents_are_projected(self):
        result = build_ontology_candidate([
            {"category": "Disease", "name": "大细胞贫血", "aliases": []},
            {"category": "Disease", "name": "大细胞性贫血", "aliases": []},
            {"category": "Disease", "name": "贫血", "aliases": []},
            {"category": "LabTest", "name": "红细胞容积分布宽度", "aliases": ["RDW"]},
            {"category": "LabTest", "name": "变异系数", "aliases": ["RDW-CV"]},
            {"category": "LabTest", "name": "标准差", "aliases": ["RDW-SD"]},
            {"category": "LabTest", "name": "白细胞计数", "aliases": ["WBC"]},
            {"category": "LabTest", "name": "白细胞分类计数", "aliases": ["WBC-DC"]},
            {"category": "Population", "name": "男性", "aliases": []},
            {"category": "Population", "name": "女性", "aliases": []},
            {"category": "Population", "name": "成人", "aliases": []},
            {"category": "Population", "name": "儿童", "aliases": []},
            {"category": "Population", "name": "老年人", "aliases": []},
            {"category": "Population", "name": "60岁以上的老年人", "aliases": []},
        ], self.pages, [])
        by_name = {item["name"]: item for item in result["entities"]}
        self.assertEqual(by_name["大细胞性贫血"]["synonyms"], ["大细胞贫血"])
        self.assertEqual(by_name["大细胞性贫血"]["parent"], "贫血")
        self.assertEqual(by_name["变异系数"]["parent"], "红细胞容积分布宽度")
        self.assertEqual(by_name["标准差"]["parent"], "红细胞容积分布宽度")
        self.assertEqual(by_name["白细胞分类计数"]["parent"], "白细胞计数")
        self.assertIsNone(by_name["性别"]["parent"])
        self.assertEqual(by_name["男性"]["parent"], "性别")
        self.assertEqual(by_name["女性"]["parent"], "性别")

    def test_context_slots_use_population_values_and_neut_is_supplemented_without_neut_hash(self):
        result = build_ontology_candidate([
            {"category": "Population", "name": "男性", "aliases": []},
            {"category": "Population", "name": "女性", "aliases": []},
        ], self.pages, [{"cases": [{"condition": "性别 EQ 男性"}, {"condition": "年龄 GE 60岁"}, {"condition": "中性粒细胞绝对值 LT 1.5"}]}])
        by_name = {item["name"]: item for item in result["entities"]}
        self.assertIn("中性粒细胞绝对值", by_name)
        self.assertEqual(by_name["中性粒细胞绝对值"]["aliases"], [])
        self.assertNotIn("NEUT#", by_name["中性粒细胞绝对值"]["aliases"])
        slots = {item["rule_slot"]: item for item in result["rule_alignment"]["slots"]}
        self.assertTrue(slots["性别"]["is_entity"])
        self.assertEqual(slots["性别"]["mapped_population_entities"], ["男性", "女性"])
        self.assertFalse(slots["性别"]["raw_entity_present"])
        self.assertTrue(slots["性别"]["manually_supplemented"])
        self.assertFalse(slots["年龄"]["is_entity"])
        self.assertFalse(slots["中性粒细胞绝对值"]["raw_entity_present"])
        self.assertIsNotNone(slots["中性粒细胞绝对值"]["supplemented_entity"])
        neut_review = next(
            item for item in result["review_items"]
            if item["type"] == "supplemented_labtest_missing_abbreviation"
        )
        self.assertEqual(neut_review["name"], "中性粒细胞绝对值")
        self.assertIn("中性粒细胞绝对值", neut_review["evidence"][0]["quote"])

    def test_calculation_dependencies_keep_unresolved_source_terms_visible(self):
        result = build_ontology_candidate([
            {"category": "LabTest", "name": "红细胞刚性指数", "aliases": ["IR"]},
            {"category": "LabTest", "name": "血浆黏度", "aliases": ["PV"]},
            {"category": "LabTest", "name": "血细胞比容", "aliases": ["HCT"]},
            {"category": "LabTest", "name": "凝血酶原时间比值", "aliases": ["PTR"]},
            {"category": "LabTest", "name": "国际标准化比值", "aliases": ["INR"]},
        ], self.pages, [])
        by_name = {item["name"]: item for item in result["entities"]}
        self.assertEqual(by_name["红细胞刚性指数"]["depends_on"], ["全血黏度(高切)", "血浆黏度", "血细胞比容"])
        self.assertEqual(by_name["国际标准化比值"]["depends_on"], ["凝血酶原时间比值"])
        unresolved = {(item["source"], item["target"]) for item in result["relations"] if item["target_status"] == "unresolved_source_term"}
        self.assertIn(("红细胞刚性指数", "全血黏度(高切)"), unresolved)


if __name__ == "__main__":
    unittest.main()
