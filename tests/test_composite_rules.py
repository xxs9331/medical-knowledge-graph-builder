import unittest
from decimal import Decimal

from medical_kg_sourceprep.rules.composite_rules import (
    AtomicPredicate,
    CandidateStatus,
    DecisionRow,
    DecisionTable,
    EvaluationValue,
    TextAnchor,
    all_of,
    any_of,
    at_least,
    at_most,
    detect_composite_candidates,
    evaluate,
    validate_rule,
)


def anchor(name="a"):
    return TextAnchor(anchor_id=name, source_id="synthetic", page=1, raw_expression=name)


def predicate(pid="p", operator="ge", value="10", unit="U"):
    return AtomicPredicate(pid, "synthetic condition", operator, value, unit, anchor(pid))


class CompositeRuleTests(unittest.TestCase):
    def test_nested_logic_and_three_value_missing(self):
        rule = all_of(predicate("a"), any_of(predicate("b"), at_least(2, predicate("c"), predicate("d"))))
        self.assertEqual(evaluate(rule, {"a": Decimal("11"), "b": None, "c": 11, "d": 11}), EvaluationValue.TRUE)
        self.assertEqual(evaluate(rule, {"a": Decimal("11"), "b": None, "c": 1, "d": None}), EvaluationValue.UNKNOWN)
        self.assertEqual(evaluate(rule, {"a": Decimal("9"), "b": None, "c": 1, "d": 1}), EvaluationValue.FALSE)

    def test_comparisons_and_open_closed_between(self):
        self.assertEqual(evaluate(AtomicPredicate("p", "x", "between", (1, 5), "U", anchor()), {"p": 1}), EvaluationValue.TRUE)
        self.assertEqual(evaluate(AtomicPredicate("p", "x", "between", (1, 5), "U", anchor(), False, True), {"p": 1}), EvaluationValue.FALSE)
        self.assertEqual(evaluate(AtomicPredicate("p", "x", "positive", None, "U", anchor()), {"p": True}), EvaluationValue.TRUE)
        self.assertEqual(evaluate(at_most(1, predicate("a"), predicate("b")), {"a": 11, "b": None}), EvaluationValue.UNKNOWN)

    def test_at_most_uses_possible_true_count_for_three_valued_result(self):
        unknowns = at_most(1, predicate("a"), predicate("b"), predicate("c"))
        self.assertEqual(evaluate(unknowns, {"a": None, "b": None, "c": None}), EvaluationValue.UNKNOWN)
        self.assertEqual(evaluate(unknowns, {"a": 11, "b": 11, "c": 11}), EvaluationValue.FALSE)
        self.assertEqual(evaluate(unknowns, {"a": 11, "b": 11, "c": None}), EvaluationValue.FALSE)
        self.assertEqual(evaluate(unknowns, {"a": 11, "b": 0, "c": None}), EvaluationValue.UNKNOWN)
        self.assertEqual(evaluate(unknowns, {"a": 0, "b": 0, "c": None}), EvaluationValue.TRUE)
        self.assertEqual(evaluate(at_most(0, predicate("a"), predicate("b")), {"a": 0, "b": None}), EvaluationValue.UNKNOWN)

    def test_decision_table_detects_overlap_conflict_hole_and_missing_policy(self):
        rows = (
            DecisionRow("r1", {"p": True}, "out-a", (anchor("r1"),)),
            DecisionRow("r2", {"p": True}, "out-b", (anchor("r2"),)),
        )
        table = DecisionTable("t", ("p",), rows, "UNIQUE", (anchor("t"),))
        issues = validate_rule(table)
        self.assertIn("overlap", {issue.code for issue in issues})
        self.assertIn("conflict", {issue.code for issue in issues})
        self.assertIn("missing_policy", {issue.code for issue in validate_rule(DecisionTable("t", ("p",), rows, "UNIQUE", (anchor("t"),), missing_policy=None))})

    def test_missing_anchors_and_joint_statement_fail_closed(self):
        bad = AtomicPredicate("p", "x", "eq", 1, "U", None)
        self.assertIn("missing_anchor", {issue.code for issue in validate_rule(bad)})
        candidates = detect_composite_candidates([{"text": "joint testing is useful", "tests": ["x", "y"]}])
        self.assertEqual(candidates[0].classification, "JOINT_TESTING_STATEMENT")
        self.assertFalse(candidates[0].executable)

    def test_candidate_is_never_approved_automatically(self):
        candidates = detect_composite_candidates([{"tests": ["x", "y"], "operator": "and", "text": "synthetic"}])
        self.assertEqual(candidates[0].status, CandidateStatus.CANDIDATE)
        with self.assertRaises(ValueError):
            candidates[0].approve("reviewer", "1")

    def test_supported_structures_and_antipatch_fixture(self):
        candidates = detect_composite_candidates([
            {"tests": ["x", "y"], "operator": "and", "text": "synthetic"},
            {"rows": [{"x": True, "y": False}], "output": "synthetic", "text": "synthetic"},
        ])
        self.assertEqual({c.classification for c in candidates}, {"COMPOSITE_CLASSIFICATION", "COMPOSITE_INTERPRETATION"})
        source = repr(candidates)
        for forbidden in ("disease", "page-99", "123.45"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
