"""第一章 v1.1 证据锚定关系参考集的回放测试。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from medical_kg_sourceprep.extraction.graph_builder.contract import DEFAULT_CHUNK_MANIFEST
from medical_kg_sourceprep.extraction.llm_extraction import load_chunk_manifest


ROOT = Path(__file__).resolve().parents[1]


class Chapter01RelationshipGoldV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gold = json.loads(
            (ROOT / "evaluation/chapter-01/chapter-01-relationship-gold-v1.1.json")
            .read_text(encoding="utf-8")
        )
        _manifest, chunks = load_chunk_manifest(DEFAULT_CHUNK_MANIFEST)
        cls.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def test_every_relationship_has_two_replayable_endpoint_spans(self) -> None:
        relationships = [
            relationship
            for case in self.gold["cases"]
            for relationship in case["relationships"]
        ]
        self.assertEqual(self.gold["statistics"]["relationship_count"], len(relationships))
        self.assertGreater(len(relationships), 0)
        for relationship in relationships:
            self.assertEqual(
                relationship["review_status"],
                "EVIDENCE_ANCHORED_REFERENCE_CANDIDATE",
            )
            self.assertEqual(len(relationship["evidence_spans"]), 2)
            for span in relationship["evidence_spans"]:
                chunk = self.chunk_by_id[span["chunk_id"]]
                self.assertEqual(
                    chunk.text[span["start"]:span["end"]], span["exact_quote"]
                )

    def test_reference_is_not_misrepresented_as_human_approved(self) -> None:
        self.assertFalse(self.gold["gold_provenance"]["human_approved"])
        self.assertEqual(self.gold["status"], "EVIDENCE_ANCHORED_REFERENCE_CANDIDATE")


if __name__ == "__main__":
    unittest.main()
