from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "knowledge/chapter-01/terminology/official-lab-alignment-v0.1.json"
SCRIPT = ROOT / "scripts/align_chapter01_official_lab_terms.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("official_alignment", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def module_identity(value: str) -> str:
    return _load_module()._identity(value)


class Chapter01OfficialLabAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_artifact_is_automated_and_conflict_free(self) -> None:
        self.assertEqual("AUTOMATED_ALIGNMENT_COMPLETE", self.payload["status"])
        self.assertFalse(self.payload["contract"]["user_validation_required"])
        self.assertEqual([], self.payload["conflicts"])

    def test_existing_entities_remain_primary(self) -> None:
        by_official_name = {
            item["official_name"]: item for item in self.payload["alignments"]
        }
        self.assertEqual("白细胞计数", by_official_name["白细胞计数"]["target_name"])
        self.assertEqual("平均红细胞容积", by_official_name["平均红细胞体积测定"]["target_name"])
        self.assertEqual("血浆凝血酶原时间", by_official_name["凝血酶原时间检测"]["target_name"])

    def test_absolute_and_percentage_terms_are_distinct(self) -> None:
        by_official_name = {
            item["official_name"]: item for item in self.payload["alignments"]
        }
        absolute = by_official_name["中性粒细胞绝对值"]
        percentage = by_official_name["中性粒细胞百分数"]
        self.assertNotEqual(absolute["target_canonical_id"], percentage["target_canonical_id"])
        terms = {item["name"]: item for item in self.payload["official_terms"]}
        self.assertEqual(["Neut#"], terms["中性粒细胞绝对值"]["abbreviations"])
        self.assertEqual(["Neut%"], terms["中性粒细胞百分数"]["abbreviations"])
        self.assertEqual("中性粒细胞比例", percentage["target_name"])

    def test_internal_merges_do_not_merge_cells_with_counts(self) -> None:
        redirects = {
            item["source_name"]: item["target_name"]
            for item in self.payload["internal_merge_redirects"]
        }
        self.assertEqual("铁蛋白", redirects["血清铁蛋白"])
        self.assertEqual("血细胞比容", redirects["红细胞压积"])
        self.assertNotIn("红细胞", redirects)
        self.assertNotIn("白细胞", redirects)
        self.assertNotIn("血小板", redirects)

    def test_counts_are_internally_consistent(self) -> None:
        statistics = self.payload["statistics"]
        self.assertEqual(50, statistics["official_term_count"])
        self.assertEqual(29, statistics["official_extension_count"])
        self.assertEqual(
            statistics["existing_lab_indicator_count"]
            - statistics["internal_merge_count"]
            + statistics["official_extension_count"],
            statistics["aligned_lab_indicator_count"],
        )
        self.assertEqual(
            statistics["aligned_lab_indicator_count"],
            len(self.payload["aligned_entities"]),
        )

    def test_aligned_snapshot_folds_redirected_evidence_and_official_aliases(self) -> None:
        by_name = {
            item["canonical_name"]: item for item in self.payload["aligned_entities"]
        }
        self.assertNotIn("血清铁蛋白", by_name)
        self.assertIn("血清铁蛋白", by_name["铁蛋白"]["aliases"])
        self.assertIn("Neut#", by_name["中性粒细胞绝对值"]["aliases"])
        identities = [module_identity(item["canonical_name"]) for item in by_name.values()]
        self.assertEqual(len(identities), len(set(identities)))

    def test_identity_keeps_neut_variants_separate(self) -> None:
        module = _load_module()
        self.assertNotEqual(module._identity("Neut#"), module._identity("Neut%"))


if __name__ == "__main__":
    unittest.main()
