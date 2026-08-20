import unittest

from medical_kg_sourceprep.extraction.graph_builder.rule_gate import (
    partition_invalid_rules,
)


class RuleGateTests(unittest.TestCase):
    def test_rejects_rule_whose_output_is_also_an_input(self):
        rules = [{
            "candidate_key": "rule-self",
            "rule_inputs": ["指标增高", "原因甲", "原因乙"],
            "rule_outputs": ["指标增高"],
        }]

        accepted, rejected = partition_invalid_rules(rules)

        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["reason_code"], "rule_input_output_overlap")
        self.assertEqual(rejected[0]["overlap_mentions"], ["指标增高"])

    def test_preserves_non_self_referential_composite_rule(self):
        rule = {
            "candidate_key": "rule-valid",
            "rule_inputs": ["指标甲降低", "指标乙增高"],
            "rule_outputs": ["结果分类"],
            "rule_evidence_refs": [{
                "role": "source_text",
                "exact_quote": "结果分类：指标甲降低，指标乙增高。",
            }],
        }

        accepted, rejected = partition_invalid_rules([rule], strict_graph_shapes=True)

        self.assertEqual(accepted, [rule])
        self.assertEqual(rejected, [])

    def test_rejects_rule_whose_output_is_an_explicit_example(self):
        rules = [{
            "candidate_key": "rule-example",
            "rule_inputs": ["继发性纤溶症", "D-二聚体为阳性"],
            "rule_outputs": ["DIC"],
            "rule_evidence_refs": [{
                "exact_quote": "继发性纤溶症(如 DIC), D-二聚体为阳性",
            }],
        }]

        accepted, rejected = partition_invalid_rules(rules)

        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["reason_code"], "rule_output_is_explicit_example")
        self.assertEqual(rejected[0]["example_outputs"], ["DIC"])

    def test_does_not_reject_direct_output_followed_by_examples(self):
        rule = {
            "candidate_key": "rule-classification",
            "rule_inputs": ["MCV 减小", "RDW 增大"],
            "rule_outputs": ["小细胞不均一性贫血"],
            "rule_evidence_refs": [{
                "exact_quote": "小细胞不均一性贫血: MCV 减小, RDW 增大, 如缺铁性贫血。",
            }],
        }

        accepted, rejected = partition_invalid_rules([rule], strict_graph_shapes=True)

        self.assertEqual(accepted, [rule])
        self.assertEqual(rejected, [])

    def test_rejects_alternative_causes_listed_after_example_marker(self):
        rule = {
            "candidate_key": "rule-alternative-examples",
            "rule_inputs": ["剧烈呕吐", "严重腹泻", "大量出汗"],
            "rule_outputs": ["血液浓缩"],
            "rule_evidence_refs": [{
                "role": "source_text",
                "exact_quote": "血液浓缩，如剧烈呕吐、严重腹泻、大量出汗等。",
            }],
        }

        accepted, rejected = partition_invalid_rules([rule], strict_graph_shapes=True)

        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["reason_code"], "rule_inputs_are_explicit_examples")

    def test_rejects_disjunctive_inputs(self):
        rule = {
            "candidate_key": "rule-or",
            "rule_inputs": ["急性感染", "炎症"],
            "rule_outputs": ["反应性增多"],
            "rule_evidence_refs": [{
                "role": "source_text",
                "exact_quote": "急性感染或炎症可引起反应性增多。",
            }],
        }

        accepted, rejected = partition_invalid_rules([rule], strict_graph_shapes=True)

        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["reason_code"], "rule_inputs_are_alternatives")

    def test_rejects_positive_single_input_rule(self):
        rule = {
            "candidate_key": "rule-ordinary-cause",
            "rule_inputs": ["机体缺氧"],
            "rule_outputs": ["红细胞增多"],
            "rule_evidence_refs": [{
                "role": "source_text",
                "exact_quote": "机体缺氧导致红细胞增多。",
            }],
        }

        accepted, rejected = partition_invalid_rules([rule])

        self.assertEqual(accepted, [])
        self.assertEqual(
            rejected[0]["reason_code"], "rule_single_input_not_explicit_exclusion"
        )

    def test_preserves_single_input_explicit_exclusion(self):
        rule = {
            "candidate_key": "rule-exclusion",
            "rule_inputs": ["D-二聚体正常"],
            "rule_outputs": [],
            "rule_excluded_outputs": ["深静脉血栓"],
            "rule_evidence_refs": [{
                "role": "source_text",
                "exact_quote": "D-二聚体正常，对排除深静脉血栓有重要价值。",
            }],
        }

        accepted, rejected = partition_invalid_rules([rule])

        self.assertEqual(accepted, [rule])
        self.assertEqual(rejected, [])

    def test_rejects_sentence_shaped_negative_output(self):
        rule = {
            "candidate_key": "rule-legacy-exclusion",
            "rule_inputs": ["D-二聚体正常"],
            "rule_outputs": ["排除深静脉血栓有重要价值"],
            "rule_evidence_refs": [{
                "role": "source_text",
                "exact_quote": "D-二聚体正常，对排除深静脉血栓有重要价值。",
            }],
        }

        accepted, rejected = partition_invalid_rules([rule])

        self.assertEqual(accepted, [])
        self.assertEqual(
            rejected[0]["reason_code"], "rule_single_input_not_explicit_exclusion"
        )

    def test_rejects_threshold_rule(self):
        rule = {
            "candidate_key": "rule-threshold",
            "rule_inputs": ["PLT<100×10^9/L"],
            "rule_outputs": ["血小板减少"],
            "rule_evidence_refs": [{
                "role": "source_text",
                "exact_quote": "PLT<100×10^9/L为血小板减少。",
            }],
        }

        accepted, rejected = partition_invalid_rules([rule])

        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["reason_code"], "rule_threshold_belongs_to_preprocess")

    def test_rejects_cross_sentence_evidence_merge(self):
        rule = {
            "candidate_key": "rule-cross-sentence",
            "rule_inputs": ["原因甲", "原因乙"],
            "rule_outputs": ["结果甲"],
            "rule_evidence_refs": [
                {"role": "source_text", "exact_quote": "原因甲导致结果甲。"},
                {"role": "source_text", "exact_quote": "原因乙导致结果乙。"},
            ],
        }

        accepted, rejected = partition_invalid_rules([rule])

        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["reason_code"], "rule_cross_evidence_merge")

    def test_rejects_plain_causal_chain_disguised_as_composite(self):
        rule = {
            "candidate_key": "rule-causal-chain",
            "rule_inputs": ["机体缺氧", "促红细胞生成素增高"],
            "rule_outputs": ["红细胞生成增多"],
            "rule_logic_candidate": "ALL",
            "rule_evidence_refs": [{
                "role": "source_text",
                "exact_quote": "机体缺氧导致促红细胞生成素增高，使红细胞生成增多。",
            }],
        }

        accepted, rejected = partition_invalid_rules([rule], strict_graph_shapes=True)

        self.assertEqual(accepted, [])
        self.assertEqual(
            rejected[0]["reason_code"], "rule_source_shape_not_graph_composite"
        )

    def test_preserves_table_header_and_row_rule(self):
        rule = {
            "candidate_key": "rule-table",
            "rule_inputs": ["指标甲降低", "指标乙增高"],
            "rule_outputs": ["结论甲"],
            "rule_logic_candidate": "ALL",
            "rule_evidence_refs": [
                {"role": "table_header", "exact_quote": "指标甲|指标乙|结论"},
                {"role": "table_row", "exact_quote": "降低|增高|结论甲"},
            ],
        }

        accepted, rejected = partition_invalid_rules([rule], strict_graph_shapes=True)

        self.assertEqual(accepted, [rule])
        self.assertEqual(rejected, [])

    def test_preserves_coordinated_shared_state_classification(self):
        rule = {
            "candidate_key": "rule-shared-state",
            "rule_inputs": ["MCV正常", "RDW正常"],
            "rule_outputs": ["正细胞均一性贫血"],
            "rule_logic_candidate": "ALL",
            "rule_evidence_refs": [{
                "role": "source_text",
                "exact_quote": "正细胞均一性贫血: MCV、RDW 均正常。",
            }],
        }

        accepted, rejected = partition_invalid_rules([rule], strict_graph_shapes=True)

        self.assertEqual(accepted, [rule])
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()
