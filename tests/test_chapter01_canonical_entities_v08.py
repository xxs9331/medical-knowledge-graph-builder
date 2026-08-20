from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation/chapter-01/chapter-01-canonical-entities-v0.8.json"
MENTION_PATH = ROOT / "evaluation/chapter-01/chapter-01-entity-mentions-v0.6.json"


class Chapter01CanonicalEntitiesV08Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        cls.mentions = json.loads(MENTION_PATH.read_text(encoding="utf-8"))

    def test_all_mentions_have_at_least_one_canonical_link(self) -> None:
        expected = {
            item["mention_id"]
            for case in self.mentions["cases"]
            for item in case["mentions"]
        }
        actual = {item["mention_id"] for item in self.dataset["mention_to_canonical_links"]}
        excluded = {item["mention_id"] for item in self.dataset["excluded_mentions"]}
        self.assertEqual(expected, actual | excluded)
        self.assertFalse(actual & excluded)

    def test_chapter_heading_is_excluded_but_manual_rule_endpoints_survive(self) -> None:
        names = {item["canonical_name"] for item in self.dataset["canonical_entities"]}
        self.assertNotIn("临床血液检验", names)
        self.assertTrue({
            "父母血型", "子女可能的血型", "子女不可能的血型",
        } <= names)
        reasons = {item["reason_code"] for item in self.dataset["excluded_mentions"]}
        self.assertEqual({
            "CHAPTER_HEADING_IS_SECTION_PATH", "RULE_TABLE_HEADER_NOT_INDICATOR",
        }, reasons)

    def test_every_manual_graph_entity_is_preserved(self) -> None:
        manual = json.loads(
            (ROOT / "evaluation/chapter-01/chapter-01-graph-test-set-v0.3.json")
            .read_text(encoding="utf-8")
        )
        expected = {
            (entity_type, name)
            for case in manual["cases"]
            for entity_type, name in case["entities"]
        }
        actual = {
            (item["entity_type"], name)
            for item in self.dataset["canonical_entities"]
            for name in (item["canonical_name"], *item.get("aliases", []))
        }
        self.assertTrue(expected <= actual, sorted(expected - actual)[:10])

    def test_blood_rheology_panel_synonyms_are_merged(self) -> None:
        by_name = {
            item["canonical_name"]: item for item in self.dataset["canonical_entities"]
        }
        self.assertNotIn("血液流变学", by_name)
        self.assertIn("血液流变学", by_name["血液流变学检查"]["aliases"])

    def test_entities_are_deduplicated_by_type_and_normalized_name(self) -> None:
        identities: set[tuple[str, str]] = set()
        ids: set[str] = set()
        for entity in self.dataset["canonical_entities"]:
            identity = (
                entity["entity_type"],
                "".join(entity["canonical_name"].split()).casefold(),
            )
            self.assertNotIn(identity, identities)
            self.assertNotIn(entity["canonical_id"], ids)
            identities.add(identity)
            ids.add(entity["canonical_id"])

    def test_links_resolve_and_statistics_match(self) -> None:
        entity_ids = {
            item["canonical_id"] for item in self.dataset["canonical_entities"]
        }
        self.assertTrue(all(
            item["canonical_id"] in entity_ids
            for item in self.dataset["mention_to_canonical_links"]
        ))
        statistics = self.dataset["statistics"]
        self.assertEqual(904, statistics["mention_count"])
        self.assertEqual(
            len(self.dataset["canonical_entities"]),
            statistics["canonical_entity_count"],
        )
        self.assertEqual(
            len(self.dataset["mention_to_canonical_links"]),
            statistics["mention_to_canonical_link_count"],
        )

    def test_all_coordination_mentions_are_expanded(self) -> None:
        expanded = [
            item for item in self.dataset["mention_to_canonical_links"]
            if item["derivation"] == "COORDINATION_EXPANSION"
        ]
        self.assertEqual(28, len({item["mention_id"] for item in expanded}))
        self.assertEqual(0, self.dataset["statistics"]["ambiguous_coordination_count"])

    def test_ambiguous_coordination_is_not_inventively_expanded(self) -> None:
        names = {
            item["canonical_name"] for item in self.dataset["canonical_entities"]
        }
        self.assertNotIn("红细胞膜的结构血红蛋白结构异常", names)
        self.assertNotIn("中缺乏", names)
        self.assertNotIn("V缺乏", names)

    def test_clear_shared_suffix_examples_are_expanded(self) -> None:
        names = {
            item["canonical_name"] for item in self.dataset["canonical_entities"]
        }
        self.assertTrue({"红细胞减少", "血红蛋白减少"} <= names)
        self.assertTrue({"A凝集原", "B凝集原"} <= names)
        self.assertTrue({"胃溃疡出血", "十二指肠溃疡出血"} <= names)

    def test_nested_entities_remain_distinct(self) -> None:
        identities = {
            (item["entity_type"], item["canonical_name"])
            for item in self.dataset["canonical_entities"]
        }
        self.assertIn(("ClinicalContext", "叶酸缺乏"), identities)
        self.assertIn(("ClinicalContext", "叶酸"), identities)
        self.assertIn(("ClinicalContext", "铁缺乏"), identities)
        self.assertIn(("ClinicalContext", "铁"), identities)

    def test_existing_nested_source_mentions_are_linked_from_outer_mentions(self) -> None:
        links = self.dataset["mention_to_canonical_links"]
        self.assertTrue(any(
            item["derivation"] == "NESTED_SOURCE_MENTION" for item in links
        ))
        self.assertGreater(self.dataset["statistics"]["nested_source_link_count"], 0)

    def test_lexical_substrings_are_not_treated_as_nested_entities(self) -> None:
        mention_ids = {
            item["mention_id"]
            for case in self.mentions["cases"]
            for item in case["mentions"]
            if item["exact_quote"] == "血清转铁蛋白升高"
        }
        entities = {
            item["canonical_id"]: item["canonical_name"]
            for item in self.dataset["canonical_entities"]
        }
        linked_names = {
            entities[item["canonical_id"]]
            for item in self.dataset["mention_to_canonical_links"]
            if item["mention_id"] in mention_ids
        }
        self.assertNotIn("铁蛋白", linked_names)

    def test_long_compound_mentions_are_structured(self) -> None:
        identities = {
            (item["entity_type"], item["canonical_name"])
            for item in self.dataset["canonical_entities"]
        }
        self.assertNotIn(
            ("ClinicalContext", "排除深静脉血栓(DVT)有重要价值"),
            identities,
        )
        self.assertNotIn(
            ("ClinicalContext", "血小板破坏增多但骨髓代偿功能良好"),
            identities,
        )
        self.assertIn(("ClinicalContext", "血小板破坏增多"), identities)
        self.assertIn(("ClinicalContext", "骨髓代偿功能良好"), identities)
        self.assertIn(("LabIndicator", "红细胞膜表面积/血细胞容积比值"), identities)

    def test_heading_and_latex_noise_is_removed(self) -> None:
        names = {item["canonical_name"] for item in self.dataset["canonical_entities"]}
        self.assertIn("维生素B12", names)
        self.assertFalse(any(name.startswith("(一)") for name in names))

    def test_plus_combinations_keep_outer_and_inner_entities(self) -> None:
        names = {item["canonical_name"] for item in self.dataset["canonical_entities"]}
        self.assertTrue({
            "抗A+抗B", "抗A", "抗B", "O型血清",
            "标准血清+受检者红细胞", "标准血清", "受检者红细胞",
        } <= names)

    def test_compound_diseases_link_to_existing_base_diseases(self) -> None:
        entities = {
            item["canonical_id"]: item for item in self.dataset["canonical_entities"]
        }
        links = self.dataset["mention_to_canonical_links"]
        source_mentions = {
            item["mention_id"]: item
            for case in self.mentions["cases"]
            for item in case["mentions"]
        }
        nested = {
            entities[item["canonical_id"]]["canonical_name"]
            for item in links
            if item["derivation"] == "NESTED_DISEASE_BASE"
            and source_mentions[item["mention_id"]]["exact_quote"]
            == "急、慢性淋巴细胞性白血病"
        }
        self.assertIn("白血病", nested)
        false_nested = {
            entities[item["canonical_id"]]["canonical_name"]
            for item in links
            if item["derivation"] == "NESTED_DISEASE_BASE"
            and source_mentions[item["mention_id"]]["exact_quote"] == "副伤寒"
        }
        self.assertNotIn("伤寒", false_nested)

    def test_longest_one_hundred_entities_have_no_rejected_boundaries(self) -> None:
        longest = sorted(
            [
                item for item in self.dataset["canonical_entities"]
                if "MANUAL_GRAPH_GOLD" not in item["derivations"]
            ],
            key=lambda item: (len(item["canonical_name"]), item["canonical_name"]),
            reverse=True,
        )[:100]
        rejected_fragments = (
            "一些", "某些", "各种", "是否", "有重要价值",
            "破坏增多但", "呈巨型改变", "似“缗钱状”",
        )
        for entity in longest:
            self.assertFalse(
                any(fragment in entity["canonical_name"] for fragment in rejected_fragments),
                entity["canonical_name"],
            )

    def test_long_entities_keep_required_nested_endpoints(self) -> None:
        identities = {
            (item["entity_type"], item["canonical_name"])
            for item in self.dataset["canonical_entities"]
        }
        self.assertTrue({
            ("LabIndicator", "红细胞膜表面积"),
            ("LabIndicator", "血细胞容积"),
            ("ClinicalContext", "凝血因子I"),
            ("ClinicalContext", "凝血因子VII"),
            ("LabIndicator", "促红细胞生成素"),
            ("LabIndicator", "转铁蛋白"),
        } <= identities)

    def test_parenthetical_indicator_aliases_survive_nested_expansion(self) -> None:
        by_name = {
            item["canonical_name"]: item
            for item in self.dataset["canonical_entities"]
        }
        self.assertIn("MCH", by_name["平均红细胞血红蛋白含量"]["aliases"])
        self.assertIn("MCHC", by_name["平均红细胞血红蛋白浓度"]["aliases"])

    def test_explicit_indicator_and_state_synonyms_are_merged(self) -> None:
        by_identity = {
            (item["entity_type"], item["canonical_name"]): item
            for item in self.dataset["canonical_entities"]
        }
        self.assertIn("ESR", by_identity[("LabIndicator", "红细胞沉降率")]["aliases"])
        self.assertIn("INR", by_identity[("LabIndicator", "国际标准化比值")]["aliases"])
        self.assertIn("D-二聚体为阳性", by_identity[
            ("IndicatorState", "D-二聚体阳性")
        ]["aliases"])
        self.assertIn("红细胞变形能力下降", by_identity[
            ("IndicatorState", "红细胞变形能力降低")
        ]["aliases"])
        self.assertNotIn(("LabIndicator", "ESR"), by_identity)
        self.assertNotIn(("IndicatorState", "D-二聚体为阳性"), by_identity)

    def test_single_report_computable_mch_states_are_deduplicated(self) -> None:
        expected = {
            "MCH增大", "MCH正常", "MCH减小", "MCH显著减小(<23pg)",
            "MCHC正常", "MCHC减小",
        }
        matching = [
            item for item in self.dataset["canonical_entities"]
            if item["entity_type"] == "IndicatorState"
            and item["canonical_name"] in expected
        ]
        self.assertEqual(expected, {item["canonical_name"] for item in matching})
        self.assertEqual(len(expected), len(matching))
        self.assertTrue(all(
            "TABLE_THRESHOLD_DERIVATION" in item["derivations"]
            for item in matching
        ))

    def test_temporal_states_are_not_added_as_computed_entities(self) -> None:
        automatic_names = {
            item["canonical_name"] for item in self.dataset["canonical_entities"]
            if "MANUAL_GRAPH_GOLD" not in item["derivations"]
        }
        self.assertNotIn("MPV持续下降", automatic_names)
        self.assertNotIn("血小板数量持续下降", automatic_names)

    def test_second_long_entity_review_keeps_atomic_endpoints(self) -> None:
        identities = {
            (item["entity_type"], item["canonical_name"])
            for item in self.dataset["canonical_entities"]
        }
        self.assertTrue({
            ("LabIndicator", "红细胞电泳时间"),
            ("LabIndicator", "血液含氧量"),
            ("IndicatorState", "血液含氧量减少"),
            ("Disease", "感染性心内膜炎"),
            ("Disease", "肺源性心脏病"),
            ("ClinicalContext", "粒细胞系"),
            ("ClinicalContext", "巨核细胞系"),
        } <= identities)

    def test_invented_marrow_compound_is_removed(self) -> None:
        names = {item["canonical_name"] for item in self.dataset["canonical_entities"]}
        self.assertNotIn("红细胞生成骨髓减少", names)
        self.assertTrue({"红细胞生成", "骨髓减少"} <= names)

    def test_anemia_class_names_do_not_invent_morphology_states(self) -> None:
        entities = {
            item["canonical_id"]: item["canonical_name"]
            for item in self.dataset["canonical_entities"]
        }
        mention_ids = {
            item["mention_id"]
            for case in self.mentions["cases"]
            for item in case["mentions"]
            if item["exact_quote"] == "正细胞均一性贫血"
        }
        linked_names = {
            entities[item["canonical_id"]]
            for item in self.dataset["mention_to_canonical_links"]
            if item["mention_id"] in mention_ids
        }
        self.assertNotIn("正细胞", linked_names)
        self.assertNotIn("红细胞大小均一", linked_names)
        self.assertIn("正细胞均一性贫血", linked_names)


if __name__ == "__main__":
    unittest.main()
