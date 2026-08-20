import json
import unittest
from pathlib import Path

from medical_kg_sourceprep.extraction.graph_builder.evaluation.layered_scoring import (
    aggregate_layered_scores,
    score_layered_case,
)


ROOT = Path(__file__).resolve().parents[1]
LAYERED_GOLD_PATH = ROOT / "evaluation/chapter-01/chapter-01-layered-test-set-v0.4.json"
MANIFEST_PATH = ROOT / "source-packages/canonical/evidence/chapter-01/manifest.json"


class LayeredScoringTests(unittest.TestCase):
    def test_scoped_gold_ignores_predictions_outside_annotated_domain(self) -> None:
        graph = {
            "nodes": [
                {
                    "candidate_key": "inside", "entity_type": "Disease",
                    "mention": "肾胚胎瘤", "canonical_name_candidate": "肾胚胎瘤",
                    "source_ref": {"chunk_id": "c1", "mention_char_start": 10,
                                   "mention_char_end": 15},
                },
                {
                    "candidate_key": "outside", "entity_type": "Disease",
                    "mention": "其他疾病", "canonical_name_candidate": "其他疾病",
                    "source_ref": {"chunk_id": "c1", "mention_char_start": 30,
                                   "mention_char_end": 34},
                },
            ],
            "relationships": [],
        }
        case = {
            "evidence_units": [{
                "evidence_unit_id": "m1", "chunk_id": "c1", "start": 10, "end": 15,
                "exact_quote": "肾胚胎瘤", "mention_eligible": True,
            }],
            "canonical_entities": [{
                "canonical_id": "e1", "entity_type": "Disease",
                "canonical_label": "肾母细胞瘤",
                "accepted_surface_forms": ["肾母细胞瘤", "肾胚胎瘤"],
            }],
            "mention_to_canonical_links": [{"evidence_unit_id": "m1", "canonical_id": "e1"}],
            "relationships": [],
        }

        score = score_layered_case(graph, case)

        self.assertEqual(1, score["canonical_entities"]["tp"])
        self.assertEqual(0, score["canonical_entities"]["fp"])
        self.assertEqual(1, score["links"]["tp"])

    def test_mention_and_canonical_metrics_are_separate(self) -> None:
        graph = {
        "nodes": [{
            "candidate_key": "n1",
            "entity_type": "ClinicalContext",
            "mention": "叶酸、维生素B12缺乏",
            "canonical_name_candidate": "叶酸缺乏",
            "source_ref": {
                "chunk_id": "c1", "mention_char_start": 10, "mention_char_end": 22,
            },
        }],
        "relationships": [],
    }
        case = {
        "evidence_units": [{
            "evidence_unit_id": "m1", "kind": "MENTION", "chunk_id": "c1",
            "start": 10, "end": 22, "mention_eligible": True,
        }],
        "canonical_entities": [
            {"canonical_id": "e1", "entity_type": "ClinicalContext",
             "canonical_label": "叶酸缺乏"},
            {"canonical_id": "e2", "entity_type": "ClinicalContext",
             "canonical_label": "维生素B12缺乏"},
        ],
        "mention_to_canonical_links": [
            {"evidence_unit_id": "m1", "canonical_id": "e1"},
            {"evidence_unit_id": "m1", "canonical_id": "e2"},
        ],
        "relationships": [],
    }

        score = score_layered_case(graph, case)

        self.assertEqual(1, score["mentions"]["tp"])
        self.assertEqual(1, score["canonical_entities"]["tp"])
        self.assertEqual(1, score["canonical_entities"]["fn"])
        self.assertEqual(1, score["links"]["tp"])
        self.assertEqual(1, score["links"]["fn"])

    def test_scores_use_micro_counts(self) -> None:
        aggregated = aggregate_layered_scores([
            {category: {"tp": 1, "fp": 1, "fn": 0} for category in
             ("mentions", "canonical_entities", "links", "relationships")},
            {category: {"tp": 1, "fp": 0, "fn": 2} for category in
             ("mentions", "canonical_entities", "links", "relationships")},
        ])

        self.assertEqual(2, aggregated["mentions"]["tp"])
        self.assertEqual(66.67, aggregated["mentions"]["precision_percent"])
        self.assertEqual(50.0, aggregated["mentions"]["recall_percent"])

    def test_all_strict_mentions_replay_original_chunks(self) -> None:
        gold = json.loads(LAYERED_GOLD_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        chunk_texts = {
            item["chunk_id"]: (MANIFEST_PATH.parent / item["chunk_path"]).read_text(
                encoding="utf-8"
            )
            for item in manifest["chunks"]
        }

        self.assertEqual(8, len(gold["cases"]))
        self.assertEqual("GENERATED_GOLD", gold["status"])
        self.assertTrue(gold["gold_provenance"]["scoring_eligible"])
        self.assertFalse(gold["gold_provenance"]["human_approved"])
        for case in gold["cases"]:
            for unit in case["evidence_units"]:
                text = chunk_texts[unit["chunk_id"]]
                self.assertEqual(unit["exact_quote"], text[unit["start"]:unit["end"]])


if __name__ == "__main__":
    unittest.main()
