import json
import unittest
from pathlib import Path

from medical_kg_sourceprep.extraction.llm_extraction import load_chunk_manifest


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation/selected-chapters/negative-cases-v0.1.json"
MANIFEST = ROOT / "source-packages/canonical/evidence/full-book-v0.2/manifest.json"


class SelectedChapterNegativeCaseTests(unittest.TestCase):
    def test_cases_are_unique_and_replay_verbatim(self):
        dataset = json.loads(DATASET.read_text(encoding="utf-8"))
        _manifest, chunks = load_chunk_manifest(MANIFEST)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.assertEqual(dataset["status"], "HUMAN_REVIEW_REQUIRED")
        self.assertEqual({case["chapter_id"] for case in dataset["cases"]},
                         {"chapter-02", "chapter-06", "chapter-13", "chapter-18"})
        case_ids = [case["case_id"] for case in dataset["cases"]]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        for case in dataset["cases"]:
            self.assertIn(case["chunk_id"], by_id)
            self.assertIn(case["evidence_quote"], by_id[case["chunk_id"]].text)
            self.assertTrue(case["forbidden"])


if __name__ == "__main__":
    unittest.main()
