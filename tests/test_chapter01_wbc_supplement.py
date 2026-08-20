from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.import_chapter01_wbc_supplement_to_neo4j import load_supplement


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "knowledge/chapter-01/terminology/wbc-differential-supplement-v0.1.json"


class Chapter01WbcSupplementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = load_supplement(ARTIFACT)

    def test_only_currently_missing_entities_are_added(self) -> None:
        names = {item["canonical_name"] for item in self.payload["added_entities"]}
        self.assertEqual({
            "单核细胞百分数", "嗜酸性粒细胞百分数", "嗜碱性粒细胞百分数",
        }, names)

    def test_all_ten_book_table_metrics_have_rules_and_panel_edges(self) -> None:
        self.assertEqual(10, len(self.payload["rules"]))
        panel_edges = [item for item in self.payload["relationships"] if item["relation_type"] == "HAS_METRIC"]
        self.assertEqual(10, len(panel_edges))
        self.assertEqual(
            {item["indicator_canonical_id"] for item in self.payload["rules"]},
            {item["target_canonical_id"] for item in panel_edges},
        )

    def test_all_rule_indicators_resolve_to_base_or_added_entities(self) -> None:
        base = json.loads((
            ROOT / "evaluation/chapter-01/chapter-01-canonical-entities-v0.8.json"
        ).read_text(encoding="utf-8"))
        entity_ids = {
            item["canonical_id"] for item in base["canonical_entities"]
        } | {item["canonical_id"] for item in self.payload["added_entities"]}
        self.assertTrue({
            item["indicator_canonical_id"] for item in self.payload["rules"]
        } <= entity_ids)

    def test_rules_preserve_units_and_do_not_invent_unsupported_low_states(self) -> None:
        by_name = {item["indicator_name"]: item for item in self.payload["rules"]}
        self.assertEqual("x10^9/L", by_name["单核细胞绝对值"]["unit"])
        self.assertEqual("%", by_name["单核细胞百分数"]["unit"])
        self.assertIsNone(by_name["单核细胞绝对值"]["low_state_id"])
        self.assertEqual("BELOW_REFERENCE", by_name["单核细胞绝对值"]["low_result_without_state"])

    def test_no_disease_relationship_is_generated(self) -> None:
        entity_types = {
            item["entity_type"] for item in self.payload["added_entities"]
        }
        self.assertEqual({"LabIndicator"}, entity_types)
        self.assertEqual(
            {"HAS_METRIC", "HAS_STATE"},
            {item["relation_type"] for item in self.payload["relationships"]},
        )


if __name__ == "__main__":
    unittest.main()
