"""第一章前 8 页逐页审核实体集的合同测试。"""

from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation/chapter-01/chapter-01-entity-mentions-v0.7.json"
MANIFEST_PATH = ROOT / "source-packages/canonical/evidence/chapter-01/manifest.json"


class Chapter01EntityMentionsV07Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.chunk_texts = {
            item["chunk_id"]: (MANIFEST_PATH.parent / item["chunk_path"]).read_text(
                encoding="utf-8"
            )
            for item in manifest["chunks"]
        }

    def test_only_first_eight_reviewed_pages_are_closed_world(self) -> None:
        self.assertEqual([0, 7], self.dataset["reviewed_page_range"])
        self.assertTrue(self.dataset["unreviewed_pages_excluded"])
        self.assertEqual(list(range(8)), [page["page_index"] for page in self.dataset["pages"]])
        self.assertTrue(all(page["closed_world"] for page in self.dataset["pages"]))

    def test_every_mention_replays_exact_chunk_text(self) -> None:
        identities: set[tuple[str, int, int, str]] = set()
        for page in self.dataset["pages"]:
            for mention in page["mentions"]:
                text = self.chunk_texts[mention["chunk_id"]]
                self.assertEqual(
                    mention["exact_quote"], text[mention["start"]:mention["end"]]
                )
                identity = (
                    mention["chunk_id"], mention["start"], mention["end"],
                    mention["entity_type"],
                )
                self.assertNotIn(identity, identities)
                identities.add(identity)

    def test_ambiguous_single_letter_abbreviations_are_excluded(self) -> None:
        mentions = {
            mention["exact_quote"]
            for page in self.dataset["pages"]
            for mention in page["mentions"]
        }
        self.assertTrue({"N", "L", "M", "E", "B"}.isdisjoint(mentions))


if __name__ == "__main__":
    unittest.main()
