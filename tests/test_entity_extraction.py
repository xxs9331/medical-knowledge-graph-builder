import unittest

from medical_kg_sourceprep.extraction.entity_extraction import merge_entities, validate_page_result


class EntityExtractionTest(unittest.TestCase):
    def test_validation_requires_grounded_alias_and_labtest_abbreviation(self):
        text = "血红蛋白(Hb)减少，见于贫血。"
        result = validate_page_result({"entities": [
            {"category": "LabTest", "name": "血红蛋白", "aliases": ["Hb"], "mentions": ["血红蛋白", "Hb"]},
            {"category": "Disease", "name": "贫血", "aliases": [], "mentions": ["贫血"]},
            {"category": "LabTest", "name": "血红蛋白", "aliases": ["HGB"], "mentions": ["血红蛋白"]},
        ]}, text, 0)
        self.assertEqual(len(result["entities"]), 2)
        self.assertEqual(result["rejections"][0]["reason"], "alias is not grounded in source")

    def test_merge_keeps_aliases_and_requested_shape(self):
        merged = merge_entities([
            {"entities": [{"category": "LabTest", "name": "血红蛋白", "aliases": ["Hb"], "mentions": ["血红蛋白(Hb)"]}], "rejections": []},
            {"entities": [{"category": "LabTest", "name": "血红蛋白", "aliases": ["血红蛋白浓度"], "mentions": ["血红蛋白浓度"]}], "rejections": []},
        ])
        self.assertEqual(merged["entities"], [{"category": "LabTest", "name": "血红蛋白", "aliases": ["Hb", "血红蛋白浓度"]}])

    def test_chapter_grounding_allows_cross_page_name(self):
        result = validate_page_result({"entities": [
            {"category": "LabTest", "name": "血小板计数", "aliases": ["PLT"], "mentions": ["血小板(PLT)"]},
        ]}, "PLT<100", 0, grounding_text="血小板计数\n血小板(PLT)")
        self.assertEqual(result["entities"][0]["name"], "血小板计数")

    def test_cross_category_exact_names_are_resolved_once(self):
        merged = merge_entities([
            {"entities": [{"category": "Disease", "name": "慢性感染", "aliases": [], "mentions": ["慢性感染"]}], "rejections": []},
            {"entities": [{"category": "Etiology", "name": "慢性感染", "aliases": [], "mentions": ["慢性感染"]}], "rejections": []},
        ])
        self.assertEqual(merged["entities"], [{"category": "Etiology", "name": "慢性感染", "aliases": []}])

    def test_formula_symbol_is_not_an_english_abbreviation(self):
        merged = merge_entities([
            {"entities": [{"category": "LabTest", "name": "全血黏度", "aliases": ["ηb"], "mentions": ["全血黏度"]}], "rejections": []},
        ])
        self.assertEqual(merged["entities"], [])
        self.assertEqual(merged["conflicts"][0]["type"], "labtest_missing_abbreviation_after_merge")


if __name__ == "__main__":
    unittest.main()
