import unittest
from decimal import Decimal

from medical_kg_sourceprep.analysis import (
    AnalysisRule,
    analyze_report,
    result_to_dict,
)
from medical_kg_sourceprep.composite_rules import (
    AtomicPredicate,
    CandidateStatus,
    ReviewRecord,
    TextAnchor,
    all_of,
)
from medical_kg_sourceprep.report_model import AbnormalFlag, Observation, ReferenceInterval


def observation(name="synthetic_a", value="12", lower="1", upper="10", flag=AbnormalFlag.HIGH):
    return Observation(
        raw_name=name,
        standard_name=name,
        abbreviation=None,
        value=value,
        unit="U",
        reference_interval=ReferenceInterval(lower=lower, upper=upper),
        report_flag=flag,
    )


def rule(structure, *, status=CandidateStatus.APPROVED.value, review=True):
    return AnalysisRule(
        rule_id="rule-1",
        version="1.0.0",
        status=status,
        structure=structure,
        review=ReviewRecord("reviewer", "approved", "1.0.0", "synthetic review") if review else None,
        conclusion="synthetic conclusion",
    )


class AnalysisTests(unittest.TestCase):
    def test_invalid_report_with_approved_rule_returns_gap_and_zero_claims(self):
        from tests.test_evidence_policy import _provenance

        book_source, registry = _provenance()
        predicate = AtomicPredicate(
            "synthetic_a", "condition text", "ge", Decimal("10"), "U",
            TextAnchor("book-anchor", "book-fixed-1", 1, "condition text"),
        )
        configured = AnalysisRule(
            "rule-1", "1.0.0", CandidateStatus.APPROVED.value, predicate,
            ReviewRecord("reviewer", "approved", "1.0.0", "synthetic review"),
            "synthetic conclusion", {"synthetic_a": book_source},
        )
        result = analyze_report({"synthetic_a": observation(value="bad", flag=None)}, (configured,), approved_book_registry=registry)

        self.assertIsNotNone(result)
        self.assertFalse(result.claims)
        self.assertFalse(result.citation_bundles)
        self.assertTrue(any(g.required for g in result.gaps))
        self.assertTrue(any(g.subject_id == "synthetic_a" for g in result.gaps))
        self.assertEqual(result.rule_evaluations[0].value, "unknown")

    def test_successful_claim_contains_report_computation_and_book_chain(self):
        from tests.test_evidence_policy import _provenance

        book_source, registry = _provenance()
        predicate = AtomicPredicate(
            "synthetic_a", "condition text", "ge", Decimal("10"), "U",
            TextAnchor("book-anchor", "book-fixed-1", 1, "condition text"),
        )
        configured = AnalysisRule(
            "rule-1", "1.0.0", CandidateStatus.APPROVED.value, predicate,
            ReviewRecord("reviewer", "approved", "1.0.0", "synthetic review"),
            "synthetic conclusion", {"synthetic_a": book_source},
        )
        result = analyze_report({"synthetic_a": observation()}, (configured,), approved_book_registry=registry)

        self.assertEqual(len(result.claims), 1)
        bundle = result.citation_bundles[0]
        self.assertEqual({source["source_type"] for source in bundle.sources}, {"report", "book"})
        self.assertEqual(bundle.computation.report_citation_id, next(source["citation_id"] for source in bundle.sources if source["source_type"] == "report"))
        self.assertIn(bundle.computation.report_citation_id, bundle.computation.citation_ids)

    def test_report_recomputation_and_deterministic_group(self):
        report = {"synthetic_a": observation()}
        first = analyze_report(report, ())
        second = analyze_report(report, ())

        self.assertEqual(first, second)
        self.assertEqual(first.abnormalities[0].computed_flag, AbnormalFlag.HIGH)
        self.assertEqual(first.abnormalities[0].report_flag, AbnormalFlag.HIGH)
        self.assertEqual(result_to_dict(first), result_to_dict(second))

    def test_invalid_report_is_a_gap_and_never_a_claim(self):
        report = {"synthetic_a": observation(value="bad", flag=None)}
        result = analyze_report(report, ())
        self.assertEqual(result.abnormalities[0].computed_flag, None)
        self.assertTrue(result.gaps)
        self.assertEqual(result.claims, ())

    def test_approved_rule_with_missing_fact_is_unknown_and_not_claimed(self):
        predicate = AtomicPredicate("p1", "synthetic condition", "ge", Decimal("10"), "U", None)
        result = analyze_report({}, (rule(predicate),))
        self.assertEqual(result.rule_evaluations[0].value, "unknown")
        self.assertFalse(result.claims)
        self.assertTrue(any(g.subject_id == "p1" for g in result.gaps))

    def test_candidate_reviewed_rejected_and_missing_review_fail_closed(self):
        predicate = AtomicPredicate("p1", "synthetic condition", "ge", Decimal("10"), "U", None)
        for status, has_review in (("candidate", True), ("reviewed", True), ("rejected", True), ("approved", False)):
            with self.subTest(status=status, has_review=has_review):
                result = analyze_report({"synthetic_a": observation()}, (rule(predicate, status=status, review=has_review),))
                self.assertFalse(result.claims)
                self.assertTrue(result.rule_evaluations[0].gaps)

    def test_all_of_keeps_each_predicate_in_trace(self):
        left = AtomicPredicate("left", "left", "ge", 10, "U", None)
        right = AtomicPredicate("right", "right", "lt", 20, "U", None)
        result = analyze_report({"left": observation("left"), "right": observation("right", value="15", lower="1", upper="20", flag=AbnormalFlag.NORMAL)}, (rule(all_of(left, right)),))
        trace = result.rule_evaluations[0].trace
        self.assertEqual([item.predicate_id for item in trace.predicates], ["left", "right"])
        self.assertEqual(result.rule_evaluations[0].value, "true")


if __name__ == "__main__":
    unittest.main()
