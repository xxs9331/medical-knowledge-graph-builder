from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "evaluation" / "chapter-01"
GRAPH_PATH = DATASET_ROOT / "chapter-01-graph-test-set-v0.1.json"
MANIFEST_PATH = REPO_ROOT / "source-packages" / "canonical" / "evidence" / "chapter-01" / "manifest.json"


class Chapter01EvaluationSetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))

    def test_all_manifest_chunks_are_covered_exactly_once(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected = {chunk["chunk_id"] for chunk in manifest["chunks"]}
        actual = [chunk_id for case in self.graph["cases"] for chunk_id in case["chunk_ids"]]

        self.assertEqual(expected, set(actual))
        self.assertEqual([], [chunk_id for chunk_id, count in Counter(actual).items() if count != 1])

    def test_relationship_endpoints_exist_in_case_entities(self) -> None:
        for case in self.graph["cases"]:
            with self.subTest(case_id=case["case_id"]):
                entity_mentions = {entity[1] for entity in case["entities"]}
                for start, _relationship_type, end in case["relationships"]:
                    self.assertIn(start, entity_mentions)
                    self.assertIn(end, entity_mentions)

    def test_projected_datasets_match_graph_source(self) -> None:
        views = {
            "entities": "chapter-01-entity-test-set-v0.1.json",
            "relationships": "chapter-01-relationship-test-set-v0.1.json",
            "rules": "chapter-01-rule-test-set-v0.1.json",
        }
        graph_cases = {case["case_id"]: case for case in self.graph["cases"]}
        for field, filename in views.items():
            view = json.loads((DATASET_ROOT / filename).read_text(encoding="utf-8"))
            with self.subTest(filename=filename):
                self.assertEqual(self.graph["status"], view["status"])
                self.assertEqual(set(graph_cases), {case["case_id"] for case in view["cases"]})
                for case in view["cases"]:
                    source = graph_cases[case["case_id"]]
                    self.assertEqual(source["chunk_ids"], case["chunk_ids"])
                    self.assertEqual(source[field], case["expected"])


if __name__ == "__main__":
    unittest.main()
