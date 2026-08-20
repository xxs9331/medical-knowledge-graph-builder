"""确定性评测问题工件测试。"""

from __future__ import annotations

import unittest

from medical_kg_sourceprep.extraction.graph_builder.evaluation.issues import (
    build_evaluation_issues,
)


class EvaluationIssuesTests(unittest.TestCase):
    def test_groups_relation_metrics_and_validation_items(self) -> None:
        result = build_evaluation_issues(
            score={
                "schema_version": "score/v1",
                "by_relation_type": {
                    "CAUSES": {"fn": 21, "fp": 3, "recall_percent": 10, "precision_percent": 80},
                },
                "false_positive_diagnostics": {"REVERSED_ENDPOINT_PAIR": 2},
            },
            run_manifest={"run_id": "run-1"},
            review_queue={"items": [{"reason_code": "endpoint_invalid", "candidate_key": "r1"}]},
        )
        self.assertEqual(result["run_id"], "run-1")
        self.assertEqual(result["issue_count"], 4)
        self.assertEqual(result["counts_by_severity"]["high"], 1)
        self.assertTrue(all(item["status"] == "OPEN" for item in result["issues"]))

    def test_empty_inputs_produce_auditable_no_issue_report(self) -> None:
        result = build_evaluation_issues(score={"schema_version": "score/v1"})
        self.assertEqual(result["status"], "NO_ISSUES")
        self.assertEqual(result["issue_count"], 0)


if __name__ == "__main__":
    unittest.main()
