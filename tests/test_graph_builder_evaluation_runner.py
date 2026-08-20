import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from medical_kg_sourceprep.extraction.graph_builder.runner import (
    aggregate_case_scores,
    aggregate_judge_results,
    build_revision_context,
    comparison_summary,
    evaluation_summary,
    run_evaluation_chunk,
)
from medical_kg_sourceprep.extraction.graph_builder.runner.single_pass_evaluation import (
    _cross_chunk_failure_graph,
)
from medical_kg_sourceprep.extraction.llm_extraction import EvidenceChunk
from neo4j_graphrag.exceptions import LLMGenerationError
from medical_kg_sourceprep.extraction.graph_builder.evaluation.aggregation import (
    aggregate_supervised_prf1,
)
from medical_kg_sourceprep.extraction.graph_builder.evaluation.artifacts import (
    first_extraction_is_usable,
    second_extraction_is_usable,
)


class GraphBuilderEvaluationRunnerTests(unittest.TestCase):
    def test_revision_context_contains_first_graph_and_only_actionable_advice(self):
        judge = {
            "input": {"graph_sha256": "abc123"},
            "counts": {"SUPPORTED": 1, "REPAIR": 1},
            "results": [
                {"judge_item_id": "node:a", "verdict": "SUPPORTED"},
                {
                    "judge_item_id": "node:b",
                    "verdict": "REPAIR",
                    "repair": {"action": "RETYPE_NODE"},
                },
            ],
        }
        coverage = {"missing_items": [{"kind": "node", "candidate": {"mention": "INR"}}]}
        first_graph = {
            "nodes": [{
                "candidate_key": "a",
                "entity_type": "LabIndicator",
                "mention": "PT",
                "evidence_refs": [{"exact_quote": "PT"}],
            }],
            "relationships": [],
        }

        context = json.loads(build_revision_context(judge, coverage, first_graph))

        self.assertEqual(context["previous_graph_sha256"], "abc123")
        self.assertEqual(context["first_candidate_graph"]["nodes"][0]["mention"], "PT")
        self.assertNotIn("evidence_refs", context["first_candidate_graph"]["nodes"][0])
        self.assertEqual(
            [item["judge_item_id"] for item in context["judge_actionable_results"]],
            ["node:b"],
        )
        self.assertNotIn("gold", json.dumps(context).lower())

    def test_extraction_usability_distinguishes_first_and_second_round(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            (output_dir / "graph.json").write_text("{}", encoding="utf-8")
            review_path = output_dir / "review-queue.json"
            review_path.write_text(json.dumps({
                "items": [{"reason_code": "relationship_phase_model_response_invalid"}],
            }), encoding="utf-8")

            self.assertFalse(first_extraction_is_usable(output_dir))
            self.assertTrue(second_extraction_is_usable(output_dir))

            review_path.write_text(json.dumps({
                "items": [{"reason_code": "entity_phase_model_response_invalid"}],
            }), encoding="utf-8")
            self.assertFalse(second_extraction_is_usable(output_dir))

    def test_aggregate_uses_constraint_weighted_micro_and_case_weighted_macro(self):
        case_results = [
            {"first": self._phase_score(satisfied=1, total=2, entity_matched=1)},
            {"first": self._phase_score(satisfied=3, total=3, entity_matched=2)},
        ]

        aggregate = aggregate_case_scores(case_results, "first")

        self.assertEqual(aggregate["micro"]["score_percent"], 80.0)
        self.assertEqual(aggregate["macro"]["score_percent"], 75.0)
        self.assertEqual(aggregate["categories"]["entities"]["matched"], 3)

    def test_supervised_prf1_aggregates_each_category_and_micro_counts(self):
        case_results = [{"score": {
            "entities": {"tp": 2, "fp": 1, "fn": 2},
            "relationships": {"tp": 1, "fp": 2, "fn": 1},
            "rules": {"tp": 0, "fp": 1, "fn": 1},
        }}]

        result = aggregate_supervised_prf1(case_results, "score")

        self.assertEqual(result["precision"], 0.428571)
        self.assertEqual(result["recall"], 0.428571)
        self.assertEqual(result["f1"], 0.428571)
        self.assertEqual(result["categories"]["entities"]["f1"], 0.571429)
        self.assertEqual(result["categories"]["relationships"]["precision"], 0.333333)
        self.assertTrue(result["standard_supervised_prf1"])

    def test_comparison_summary_keeps_only_display_fields(self):
        phase = {
            "micro": {"score": 0.5},
            "macro": {"score": 0.5},
            "categories": {},
        }
        comparison = {
            "case_count": 1,
            "unique_chunk_count": 1,
            "first": phase,
            "second": phase,
            "union": phase,
            "delta": {},
            "cases": [{
                "case_id": "TC-01",
                "first": {"challenge": {"score_percent": 50.0}},
                "second": {"challenge": {"score_percent": 60.0}},
                "union": {"challenge": {"score_percent": 70.0}},
                "delta": 0.1,
                "union_delta": 0.2,
            }],
            "chunk_artifacts": {"chunk": {"first_graph": "graph.json"}},
        }

        summary = comparison_summary(comparison)

        self.assertEqual(summary["case_scores"][0]["union"], 70.0)
        self.assertNotIn("chunk_artifacts", summary)

    def test_aggregate_judge_results_keeps_verdicts_separate_from_gold_score(self):
        aggregate = aggregate_judge_results([
            {"results": [
                {"verdict": "SUPPORTED"},
                {"verdict": "REPAIR"},
            ]},
            {"results": [{"verdict": "SUPPORTED"}]},
        ])

        self.assertEqual(aggregate["reviewed_candidates"], 3)
        self.assertEqual(aggregate["counts"]["SUPPORTED"], 2)
        self.assertEqual(aggregate["rates"]["REPAIR"], 0.333333)
        self.assertNotIn("score", aggregate)

    def test_evaluation_summary_contains_unsupervised_and_supervised_results(self):
        report = {
            "case_count": 1,
            "unique_chunk_count": 1,
            "failed_chunk_ids": ["chunk:failed"],
            "unsupervised_judge": {"reviewed_candidates": 2},
            "supervised_gold": {"micro": {"score_percent": 50.0}},
            "prf1": {"precision": 0.8, "recall": 0.5, "f1": 0.615385},
            "cases": [{
                "case_id": "TC-01",
                "score": {"challenge": {"score_percent": 50.0}},
            }],
            "chunk_artifacts": {"chunk": {"graph": "graph.json"}},
        }

        summary = evaluation_summary(report)

        self.assertEqual(summary["unsupervised_judge"]["reviewed_candidates"], 2)
        self.assertEqual(summary["supervised_gold"]["micro"]["score_percent"], 50.0)
        self.assertEqual(summary["prf1"]["f1"], 0.615385)
        self.assertEqual(summary["failed_chunk_ids"], ["chunk:failed"])
        self.assertEqual(summary["case_scores"], [{"case_id": "TC-01", "score_percent": 50.0}])
        self.assertNotIn("chunk_artifacts", summary)

    def test_skip_judge只生成候选图工件(self):
        async def exercise() -> None:
            with tempfile.TemporaryDirectory() as temporary_directory:
                output_root = Path(temporary_directory)
                graph_dir = output_root / "candidate-graph"
                graph_dir.mkdir(parents=True)
                (graph_dir / "graph.json").write_text(
                    json.dumps({"nodes": [], "relationships": []}), encoding="utf-8"
                )
                chunk = EvidenceChunk("test:chunk", "原文", "sha256")
                with (
                    patch(
                        "medical_kg_sourceprep.extraction.graph_builder.runner.single_pass_evaluation.first_extraction_is_usable",
                        return_value=True,
                    ),
                    patch(
                        "medical_kg_sourceprep.extraction.graph_builder.runner.single_pass_evaluation.judge_candidate_graph",
                        new=AsyncMock(),
                    ) as judge_mock,
                ):
                    result = await run_evaluation_chunk(
                        object(),
                        chunk=chunk,
                        schema={},
                        manifest_sha256="manifest",
                        output_root=output_root,
                        run_judge=False,
                    )

                judge_mock.assert_not_awaited()
                self.assertIsNone(result["judge"])
                self.assertEqual(result["artifacts"], {"graph": str(graph_dir / "graph.json")})
                self.assertFalse((output_root / "judge-result.json").exists())

        import asyncio

        asyncio.run(exercise())

    def test跨chunk格式失败保留审查记录而不是关系(self):
        graph = _cross_chunk_failure_graph(
            [{"candidate_key": "node:a", "mention": "A"}],
            LLMGenerationError("invalid response"),
        )

        self.assertEqual(graph["extraction_status"], "FAILED")
        self.assertEqual(graph["relationships"], [])
        self.assertEqual(
            graph["review_items"][0]["reason_code"],
            "cross_chunk_relation_phase_model_response_invalid",
        )
        self.assertEqual(graph["publication_status"], "HOLD")

    @staticmethod
    def _phase_score(
        *, satisfied: int, total: int, entity_matched: int
    ) -> dict[str, Any]:
        return {
            "challenge": {
                "satisfied_constraints": satisfied,
                "total_constraints": total,
                "score": satisfied / total,
            },
            "entities": {
                "matched": entity_matched,
                "target_total": 2,
                "tp": entity_matched,
                "fp": 0,
                "fn": 2 - entity_matched,
            },
            "relationships": {"matched": 0, "target_total": 0, "tp": 0, "fp": 0, "fn": 0},
            "rules": {"matched": 0, "target_total": 0, "tp": 0, "fp": 0, "fn": 0},
            "forbidden": {"violations": 0, "target_total": 0},
        }


if __name__ == "__main__":
    unittest.main()
