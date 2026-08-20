import unittest

from medical_kg_sourceprep.extraction.graph_builder.evaluation.scoring import (
    filter_candidate_graph_by_scopes,
    merge_candidate_graphs,
    score_candidate_graph,
    score_relationship_tier,
)


class GraphBuilderEvaluationTests(unittest.TestCase):
    def test_relationship_tier_ignores_correct_cross_tier_prediction(self):
        graph = {
            "nodes": [
                {"candidate_key": "iron", "entity_type": "Cause", "mention": "铁缺乏"},
                {"candidate_key": "material", "entity_type": "Cause",
                 "mention": "造血原料缺乏"},
            ],
            "relationships": [{
                "source_candidate_key": "iron", "relation_type": "IS_A",
                "target_candidate_key": "material",
            }],
        }

        score = score_relationship_tier(
            graph,
            {},
            targets=[],
            ignored_targets=[["铁缺乏", "IS_A", "造血原料缺乏"]],
        )

        self.assertEqual(score["fp"], 0)
        self.assertEqual(score["ignored_cross_tier_predictions"], 1)

    def test_relationship_tier_keeps_wrong_parent_as_false_positive(self):
        graph = {
            "nodes": [
                {"candidate_key": "mild", "entity_type": "Disease", "mention": "轻度贫血"},
                {"candidate_key": "dimension", "entity_type": "ClinicalContext",
                 "mention": "贫血程度"},
            ],
            "relationships": [{
                "source_candidate_key": "mild", "relation_type": "IS_A",
                "target_candidate_key": "dimension",
            }],
        }

        score = score_relationship_tier(
            graph,
            {},
            targets=[],
            ignored_targets=[["轻度贫血", "IS_A", "贫血"]],
        )

        self.assertEqual(score["fp"], 1)
        self.assertEqual(
            score["false_positive_predictions"], [("轻度贫血", "IS_A", "贫血程度")]
        )

    def test_merges_nodes_and_relationships_from_multiple_chunks(self):
        merged = merge_candidate_graphs([
            {"nodes": [{"candidate_key": "a"}], "relationships": []},
            {"nodes": [{"candidate_key": "a"}, {"candidate_key": "b"}],
             "relationships": [{"candidate_key": "r", "relation_type": "IS_A"}]},
            {"nodes": [], "relationships": [{"candidate_key": "r", "relation_type": "IS_A"}]},
        ])

        self.assertEqual([item["candidate_key"] for item in merged["nodes"]], ["a", "b"])
        self.assertEqual(merged["relationships"][0]["relation_type"], "IS_A")

    def test_scores_standard_tp_fp_fn_against_gold(self):
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
        self.assertEqual(
            {key: score["entities"][key] for key in ("tp", "fp", "fn")},
            {"tp": 1, "fp": 1, "fn": 1},
        )
        self.assertEqual(score["entities"]["precision"], 0.5)
        self.assertEqual(score["entities"]["recall"], 0.5)
        self.assertEqual(score["entities"]["f1"], 0.5)
        self.assertEqual(score["rules"]["tp"], 1)
        self.assertEqual(score["rules"]["fp"], 0)

    def test_one_prediction_cannot_match_two_gold_aliases(self):
        graph = {
            "nodes": [{
                "candidate_key": "ptr",
                "entity_type": "LabIndicator",
                "mention": "凝血酶原时间比值",
            }],
            "relationships": [],
        }
        gold = {
            "entities": [
                ["LabIndicator", "凝血酶原时间比值"],
                ["LabIndicator", "PTR"],
            ],
            "relationships": [],
            "rules": [],
            "must_not_extract": [],
        }

        score = score_candidate_graph(
            graph,
            gold,
            source_text="凝血酶原时间比值（PTR）。",
        )

        self.assertEqual(score["entities"]["tp"], 1)
        self.assertEqual(score["entities"]["fp"], 0)
        self.assertEqual(score["entities"]["fn"], 1)

    def test_explicit_indicator_alias_also_matches_same_state_suffix(self):
        graph = {
            "nodes": [
                {"candidate_key": "indicator", "entity_type": "LabIndicator",
                 "mention": "凝血酶原时间比值"},
                {"candidate_key": "state", "entity_type": "IndicatorState",
                 "mention": "凝血酶原时间比值正常"},
            ],
            "relationships": [{
                "candidate_key": "relation",
                "source_candidate_key": "indicator",
                "target_candidate_key": "state",
                "relation_type": "HAS_STATE",
            }],
        }
        gold = {
            "entities": [
                ["LabIndicator", "PTR"],
                ["IndicatorState", "PTR正常"],
            ],
            "relationships": [["PTR", "HAS_STATE", "PTR正常"]],
            "rules": [],
            "must_not_extract": [],
        }

        score = score_candidate_graph(
            graph,
            gold,
            source_text="凝血酶原时间比值 (PTR), 参考区间为 0.86~1.15。",
        )

        self.assertEqual(score["entities"]["tp"], 2)
        self.assertEqual(score["relationships"]["tp"], 1)

    def test_grammatical_state_variants_are_one_prediction_identity(self):
        graph = {
            "nodes": [
                {
                    "candidate_key": "state-1",
                    "entity_type": "IndicatorState",
                    "mention": "D-二聚体阳性",
                },
                {
                    "candidate_key": "state-2",
                    "entity_type": "IndicatorState",
                    "mention": "D-二聚体为阳性",
                },
            ],
            "relationships": [],
        }
        gold = {
            "entities": [["IndicatorState", "D-二聚体阳性"]],
            "relationships": [],
            "rules": [],
            "must_not_extract": [],
        }

        score = score_candidate_graph(graph, gold)

        self.assertEqual(score["entities"]["tp"], 1)
        self.assertEqual(score["entities"]["fp"], 0)

    def test_scope_filter_uses_candidate_evidence_offsets(self):
        graph = {
            "nodes": [
                {
                    "candidate_key": "inside",
                    "entity_type": "LabIndicator",
                    "mention": "范围内指标",
                    "source_ref": {
                        "chunk_id": "chunk-1", "char_start": 10, "char_end": 20,
                        "mention_char_start": 12, "mention_char_end": 17,
                    },
                },
                {
                    "candidate_key": "outside",
                    "entity_type": "Disease",
                    "mention": "范围外疾病",
                    "source_ref": {
                        "chunk_id": "chunk-1", "char_start": 40, "char_end": 50,
                        "mention_char_start": 42, "mention_char_end": 47,
                    },
                },
            ],
            "relationships": [{
                "candidate_key": "outside-relation",
                "source_candidate_key": "inside",
                "target_candidate_key": "outside",
                "relation_type": "ASSOCIATED_WITH",
                "source_ref": {"chunk_id": "chunk-1", "char_start": 10, "char_end": 50},
            }],
        }
        scopes = [{"chunk_id": "chunk-1", "start": 0, "end": 30}]

        filtered = filter_candidate_graph_by_scopes(graph, scopes)

        self.assertEqual([node["candidate_key"] for node in filtered["nodes"]], ["inside"])
        self.assertEqual(filtered["relationships"], [])

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

    def test_rule_requires_matching_logic_and_exact_outputs(self):
        base_nodes = [
            {"candidate_key": "a", "entity_type": "IndicatorState", "mention": "A降低"},
            {"candidate_key": "b", "entity_type": "IndicatorState", "mention": "B降低"},
            {"candidate_key": "out", "entity_type": "Disease", "mention": "目标疾病"},
            {"candidate_key": "extra", "entity_type": "Disease", "mention": "额外疾病"},
        ]
        relationships = [
            {"source_candidate_key": "a", "relation_type": "RULE_INPUT", "target_candidate_key": "rule"},
            {"source_candidate_key": "b", "relation_type": "RULE_INPUT", "target_candidate_key": "rule"},
            {"source_candidate_key": "rule", "relation_type": "RULE_OUTPUT", "target_candidate_key": "out"},
        ]
        gold = {
            "entities": [], "relationships": [], "must_not_extract": [],
            "rules": [{
                "rule_stage": "GRAPH_COMPOSITE", "logic": "ALL_SAME_WINDOW",
                "inputs": ["A降低", "B降低"], "outputs": ["目标疾病"],
            }],
        }
        wrong_logic = {
            "nodes": [*base_nodes, {
                "candidate_key": "rule", "entity_type": "RuleDefinition",
                "rule_stage_candidate": "GRAPH_COMPOSITE", "rule_logic_candidate": "ALL",
                "rule_evidence_refs": [],
            }],
            "relationships": relationships,
        }
        extra_output = {
            "nodes": [*base_nodes, {
                "candidate_key": "rule", "entity_type": "RuleDefinition",
                "rule_stage_candidate": "GRAPH_COMPOSITE",
                "rule_logic_candidate": "ALL_SAME_WINDOW", "rule_evidence_refs": [],
            }],
            "relationships": [
                *relationships,
                {"source_candidate_key": "rule", "relation_type": "RULE_OUTPUT",
                 "target_candidate_key": "extra"},
            ],
        }

        self.assertEqual(score_candidate_graph(wrong_logic, gold)["rules"]["matched"], 0)
        self.assertEqual(score_candidate_graph(extra_output, gold)["rules"]["matched"], 0)

    def test_composite_rule_evidence_cannot_replace_missing_input_edges(self):
        gold = {
            "entities": [],
            "relationships": [],
            "rules": [{
                "rule_stage": "GRAPH_COMPOSITE",
                "logic": "ALL",
                "inputs": ["MCV 正常", "RDW 增大"],
                "outputs": ["正细胞不均一性贫血"],
            }],
            "must_not_extract": [],
        }
        graph = {
            "nodes": [
                {
                    "candidate_key": "rule:partial",
                    "entity_type": "RuleDefinition",
                    "rule_stage_candidate": "GRAPH_COMPOSITE",
                    "rule_logic_candidate": "ALL",
                    "rule_evidence_refs": [{
                        "role": "definition",
                        "exact_quote": "正细胞不均一性贫血: MCV 正常, RDW 增大。",
                    }],
                },
                {
                    "candidate_key": "context:result",
                    "entity_type": "ClinicalContext",
                    "mention": "正细胞不均一性贫血",
                },
            ],
            "relationships": [{
                "relation_type": "RULE_OUTPUT",
                "source_candidate_key": "rule:partial",
                "target_candidate_key": "context:result",
            }],
        }

        score = score_candidate_graph(
            graph,
            gold,
            source_text="正细胞不均一性贫血: MCV 正常, RDW 增大。",
        )

        self.assertEqual(score["rules"]["matched"], 0)


if __name__ == "__main__":
    unittest.main()
