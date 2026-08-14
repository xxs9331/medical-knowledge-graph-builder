import unittest

from medical_kg_sourceprep.extraction.graph_builder.evaluation.scoring import (
    merge_candidate_graphs,
    score_candidate_graph,
)


class GraphBuilderEvaluationTests(unittest.TestCase):
    def test_merges_nodes_and_relationships_from_multiple_chunks(self):
        merged = merge_candidate_graphs([
            {"nodes": [{"candidate_key": "a"}], "relationships": []},
            {"nodes": [{"candidate_key": "a"}, {"candidate_key": "b"}],
             "relationships": [{"candidate_key": "r", "relation_type": "IS_A"}]},
            {"nodes": [], "relationships": [{"candidate_key": "r", "relation_type": "IS_A"}]},
        ])

        self.assertEqual([item["candidate_key"] for item in merged["nodes"]], ["a", "b"])
        self.assertEqual(merged["relationships"][0]["relation_type"], "IS_A")

    def test_scores_selective_targets_without_treating_unannotated_candidates_as_false_positives(self):
        graph = {
            "nodes": [
                {"candidate_key": "a", "entity_type": "LabIndicator",
                 "mention": "凝血酶原时间比值"},
                {"candidate_key": "b", "entity_type": "LabIndicator", "mention": "EXTRA"},
                {"candidate_key": "r", "entity_type": "RuleDefinition",
                 "rule_stage_candidate": "PREPROCESS",
                 "rule_evidence_refs": [{
                     "role": "formula",
                     "exact_quote": "凝血酶原时间比值 (PTR) = 受检血浆 PT/正常人血浆 PT",
                 }]},
            ],
            "relationships": [
                {"source_candidate_key": "a", "relation_type": "RULE_OUTPUT",
                 "target_candidate_key": "unused"},
                {"source_candidate_key": "r", "relation_type": "RULE_OUTPUT",
                 "target_candidate_key": "a"},
            ],
        }
        gold = {
            "entities": [["LabIndicator", "PTR"], ["LabIndicator", "INR"]],
            "relationships": [],
            "rules": [{
                "rule_stage": "PREPROCESS",
                "inputs": ["受检血浆 PT", "正常人血浆 PT"],
                "outputs": ["PTR"],
                "logic": "FORMULA",
            }],
            "must_not_extract": [["PTR", "CAUSES", "INR"]],
        }
        score = score_candidate_graph(
            graph, gold,
            source_text="凝血酶原时间比值 (PTR) = 受检血浆 PT/正常人血浆 PT",
        )
        self.assertEqual(score["entities"]["matched"], 1)
        self.assertEqual(score["entities"]["missed"], 1)
        self.assertEqual(score["rules"]["matched"], 1)
        self.assertEqual(score["forbidden"]["violations"], 0)
        self.assertEqual(score["challenge"]["score"], 0.75)
        self.assertEqual(score["unscored_candidates"]["entities"], 2)

    def test_forbidden_alias_relation_reduces_challenge_score(self):
        graph = {
            "nodes": [
                {"candidate_key": "ptr", "entity_type": "LabIndicator",
                 "mention": "凝血酶原时间比值"},
                {"candidate_key": "inr", "entity_type": "LabIndicator",
                 "mention": "国际标准化比值"},
            ],
            "relationships": [{
                "source_candidate_key": "ptr", "relation_type": "CAUSES",
                "target_candidate_key": "inr",
            }],
        }
        gold = {
            "entities": [], "relationships": [], "rules": [],
            "must_not_extract": [["PTR", "CAUSES", "INR"]],
        }
        source = "凝血酶原时间比值 (PTR)。国际标准化比值 (INR)。"
        score = score_candidate_graph(graph, gold, source_text=source)
        self.assertEqual(score["forbidden"]["violations"], 1)
        self.assertEqual(score["challenge"]["score"], 0.0)

    def test_explicit_human_ocr_note_can_adjudicate_formula_parameter(self):
        graph = {
            "nodes": [
                {"candidate_key": "ptr", "entity_type": "LabIndicator", "mention": "PTR"},
                {"candidate_key": "inr", "entity_type": "LabIndicator", "mention": "INR"},
                {"candidate_key": "rule", "entity_type": "RuleDefinition",
                 "rule_stage_candidate": "PREPROCESS", "rule_evidence_refs": [{
                     "role": "formula", "exact_quote": r"INR = PTR \( ^{[S1]} \)",
                 }]},
            ],
            "relationships": [
                {"source_candidate_key": "ptr", "relation_type": "RULE_INPUT",
                 "target_candidate_key": "rule"},
                {"source_candidate_key": "rule", "relation_type": "RULE_OUTPUT",
                 "target_candidate_key": "inr"},
            ],
        }
        gold = {
            "entities": [], "relationships": [], "must_not_extract": [],
            "rules": [{"rule_stage": "PREPROCESS", "inputs": ["PTR", "ISI"],
                       "outputs": ["INR"], "logic": "FORMULA"}],
            "review_notes": ["规范源公式 OCR 为 S1，后文定义 ISI。"],
        }
        source = r"INR = PTR \( ^{[S1]} \)。ISI 为国际灵敏性指数。"

        score = score_candidate_graph(graph, gold, source_text=source)

        self.assertEqual(score["rules"]["matched"], 1)
        self.assertEqual(score["challenge"]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
