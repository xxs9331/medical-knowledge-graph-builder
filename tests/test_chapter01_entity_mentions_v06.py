from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation/chapter-01/chapter-01-entity-mentions-v0.6.json"
MANIFEST_PATH = ROOT / "source-packages/canonical/evidence/chapter-01/manifest.json"


class Chapter01EntityMentionsV06Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.chunk_texts = {
            item["chunk_id"]: (MANIFEST_PATH.parent / item["chunk_path"]).read_text(
                encoding="utf-8"
            )
            for item in cls.manifest["chunks"]
        }

    def test_dataset_declares_closed_world_and_model_assisted_boundary(self) -> None:
        self.assertEqual(
            "ASSISTANT_ANNOTATED_REQUIRES_USER_VALIDATION", self.dataset["status"]
        )
        contract = self.dataset["scope_contract"]
        self.assertTrue(contract["closed_world"])
        self.assertTrue(contract["nested_mentions"] == "ALLOWED")
        self.assertFalse(contract["same_run_evaluation_allowed"])

    def test_all_chunks_are_assigned_to_one_of_eight_cases(self) -> None:
        self.assertEqual(8, len(self.dataset["cases"]))
        actual = [chunk_id for case in self.dataset["cases"] for chunk_id in case["chunk_ids"]]
        expected = {item["chunk_id"] for item in self.manifest["chunks"]}
        self.assertEqual(expected, set(actual))
        self.assertEqual([], [item for item, count in Counter(actual).items() if count != 1])

    def test_every_mention_replays_exact_source_characters(self) -> None:
        allowed = {"LabPanel", "LabIndicator", "IndicatorState", "ClinicalContext", "Disease"}
        identities: set[tuple[str, int, int, str]] = set()
        for case in self.dataset["cases"]:
            for mention in case["mentions"]:
                text = self.chunk_texts[mention["chunk_id"]]
                identity = (
                    mention["chunk_id"], mention["start"], mention["end"],
                    mention["entity_type"],
                )
                self.assertNotIn(identity, identities)
                identities.add(identity)
                self.assertIn(mention["entity_type"], allowed)
                self.assertEqual(
                    mention["exact_quote"], text[mention["start"]:mention["end"]]
                )
                self.assertGreater(mention["end"], mention["start"])


if __name__ == "__main__":
    unittest.main()
