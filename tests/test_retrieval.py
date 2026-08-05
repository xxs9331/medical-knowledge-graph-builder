import unittest
from unittest.mock import patch

from medical_kg_sourceprep.evidence.legacy.retrieval import RetrievalRecord, retrieve, retrieve_hybrid


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = (
            RetrievalRecord(
                record_id="alpha-rule",
                standard_name="Alpha Clearance Index",
                title="Alpha clearance interpretation",
                text="The alpha clearance index is calculated from the paired measurements.",
                rule_type="interpretation",
                conditions=("paired",),
                anchor_ids=("alpha-predicate", "alpha-conclusion"),
            ),
            RetrievalRecord(
                record_id="beta-rule",
                standard_name="Beta Clearance Index",
                title="Beta clearance interpretation",
                text="A clearance index can be reported after a review.",
                rule_type="interpretation",
                conditions=("paired",),
                anchor_ids=("beta-predicate", "beta-conclusion"),
            ),
            RetrievalRecord(
                record_id="context",
                title="Related context",
                text="Additional explanatory context for the alpha procedure.",
                parent_id="alpha-rule",
            ),
        )

    def test_full_standard_name_beats_shared_suffix_and_exposes_components(self) -> None:
        results = retrieve("Alpha Clearance Index", self.records)

        self.assertEqual([result.record_id for result in results[:1]], ["alpha-rule"])
        result = results[0]
        self.assertGreater(result.score_components["exact_phrase"], 0)
        self.assertIn("standard_name_exact", result.reasons)
        self.assertEqual(result.anchor_ids, ("alpha-predicate", "alpha-conclusion"))
        self.assertEqual(tuple(result.score_components), (
            "exact_phrase", "alias", "title", "rule_type", "condition_match", "bm25", "context"
        ))

    def test_abbreviation_and_caller_alias_are_precise_without_embedded_vocabulary(self) -> None:
        results = retrieve(
            "ACI",
            self.records,
            aliases={"alpha-rule": ("ACI", "alpha clearance")},
        )

        self.assertEqual(results[0].record_id, "alpha-rule")
        self.assertGreater(results[0].score_components["alias"], 0)
        self.assertIn("alias_exact", results[0].reasons)

    def test_structured_filters_and_conditions_fail_closed(self) -> None:
        matching = retrieve(
            "clearance",
            self.records,
            rule_types=("interpretation",),
            required_conditions=("paired",),
        )
        mismatched = retrieve("clearance", self.records, required_conditions=("unpaired",))

        self.assertTrue(matching)
        self.assertEqual(mismatched, ())

    def test_combination_returns_complete_rule_anchor_set_and_context_never_displaces_evidence(self) -> None:
        results = retrieve("Alpha Clearance Index", self.records, top_k=3, include_context=True)

        self.assertEqual(results[0].record_id, "alpha-rule")
        self.assertEqual(results[0].anchor_ids, ("alpha-predicate", "alpha-conclusion"))
        self.assertTrue(all(result.record_id != "context" for result in results[:1]))
        self.assertEqual(results[-1].record_id, "context")
        self.assertEqual(results[-1].reasons, ("parent_context",))

    def test_no_hit_top_k_stable_order_and_fts_fallback(self) -> None:
        first = retrieve("clearance", self.records, top_k=1, use_fts=False)
        second = retrieve("clearance", tuple(reversed(self.records)), top_k=1, use_fts=False)

        self.assertEqual(first, second)
        self.assertEqual(retrieve("unrelated vocabulary", self.records), ())
        self.assertEqual(len(first), 1)
        self.assertGreaterEqual(first[0].score_components["bm25"], 0)

    def test_fts_failure_uses_the_standard_library_bm25_fallback(self) -> None:
        with patch("medical_kg_sourceprep.evidence.legacy.retrieval._fts_scores", return_value={}):
            results = retrieve("Alpha Clearance Index", self.records)

        self.assertIn("bm25_fallback", results[0].reasons)

    def test_mapping_records_are_supported_by_the_read_only_adapter(self) -> None:
        results = retrieve(
            "signal",
            ({"record_id": "rule", "title": "Signal rule", "text": "signal evidence"},),
        )

        self.assertEqual(results[0].record_id, "rule")

    def test_hybrid_channels_change_order_but_exact_name_stays_first(self) -> None:
        ranked = retrieve_hybrid(
            "clearance",
            self.records[:2],
            vector_hits=(type("Hit", (), {"chunk_id": "beta-rule", "similarity": 0.9})(),),
            graph_hits=(type("Hit", (), {"chunk_id": "beta-rule", "graph_score": 0.8})(),),
        )
        exact = retrieve_hybrid(
            "Alpha Clearance Index", self.records[:2],
            vector_hits=(type("Hit", (), {"chunk_id": "beta-rule", "similarity": 1.0})(),),
            graph_hits=(type("Hit", (), {"chunk_id": "beta-rule", "graph_score": 1.0})(),),
        )
        self.assertEqual(ranked[0].record_id, "beta-rule")
        self.assertEqual(exact[0].record_id, "alpha-rule")
        self.assertIn("vector_tfidf", ranked[0].reasons)


if __name__ == "__main__":
    unittest.main()
